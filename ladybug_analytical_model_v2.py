# ladybug_analytical_model_v2.py
# Ladybug5+ analytical model with user-supplied RED overlays (unusable pixels) per lens.
# Azimuth convention: 0° = vehicle forward; +az to the vehicle’s left.
# Lens model: rectilinear, HFOV=87.6°, VFOV=77.5° (approx. Ladybug5+).
# Nadir occlusion: spherical cap of 36.87° around camera −Z (housing occlusion).
#
# You provide one RGB image per lens with a red-tinted overlay wherever the pixels are NOT usable.
# The code detects red pixels and excludes them when computing the spherical coverage.

from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np
import math, os
from PIL import Image
import matplotlib.pyplot as plt

# ---------------- Rotations ----------------
def Rz(yaw_deg: float) -> np.ndarray:
    a = math.radians(yaw_deg); c, s = math.cos(a), math.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=float)

def Ry(pitch_deg: float) -> np.ndarray:
    a = math.radians(pitch_deg); c, s = math.cos(a), math.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]], dtype=float)

def Rx(roll_deg: float) -> np.ndarray:
    a = math.radians(roll_deg); c, s = math.cos(a), math.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=float)

# ---------------- Ladybug model ----------------
HFOV = 87.6
VFOV = 77.5
HALF_H = HFOV/2.0
HALF_V = VFOV/2.0
NADIR_HALF = 36.87  # housing occlusion half-angle (deg)

SIDE_AZS = [0, 72, 144, -144, -72]  # deg around +Z_cam (boresights for the 5 side lenses)
TOP_VEC = np.array([0.0, 0.0, 1.0])

def side_forward(az_deg: float) -> np.ndarray:
    a = math.radians(az_deg)
    return np.array([math.cos(a), math.sin(a), 0.0])

@dataclass
class LadybugPose:
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    def R_world_from_cam(self) -> np.ndarray:
        # World-from-camera rotation
        return Rz(self.yaw_deg) @ Ry(self.pitch_deg) @ Rx(self.roll_deg)

@dataclass
class LadybugRig:
    name: str  # e.g., "Front" or "Rear"
    pose: LadybugPose
    def lenses(self) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        # (label, forward(cam), up(cam))
        out = [("Top", TOP_VEC, np.array([0,1.0,0]))]
        for az in SIDE_AZS:
            out.append((f"Side {az:+.0f}°", side_forward(az), np.array([0,0,1.0])))
        return out

# --------------- Spherical grid & optics ---------------
def build_grid(step_deg: int = 3):
    az = np.arange(-180, 181, step_deg)
    el = np.arange(-90,   91, step_deg)
    AZ, EL = np.meshgrid(az, el)
    U = np.stack([np.cos(np.radians(EL))*np.cos(np.radians(AZ)),
                  np.cos(np.radians(EL))*np.sin(np.radians(AZ)),
                  np.sin(np.radians(EL))], axis=-1)
    # area element for equirectangular plots (good enough for comparisons)
    d = math.radians(step_deg)
    area_w = np.cos(np.radians(EL))
    area_elem = area_w * d * d
    return az, el, AZ, EL, U, area_elem

def occluded_mask(U: np.ndarray, R_wc: np.ndarray) -> np.ndarray:
    # Occluded if angle to −Z_cam <= NADIR_HALF
    U_cam = U @ R_wc.T
    cosang = -U_cam[...,2]
    cosang = np.clip(cosang, -1, 1)
    ang = np.degrees(np.arccos(cosang))
    return ang <= NADIR_HALF

def angles_in_lens_frame(U: np.ndarray, R_wc: np.ndarray, f_cam: np.ndarray, up_cam: np.ndarray):
    # Build lens-local axes from world frame
    f_w = R_wc @ np.array(f_cam, float)
    up_w = R_wc @ up_cam
    r_w = np.cross(f_w, up_w)
    if np.linalg.norm(r_w) < 1e-6:
        r_w = np.cross(f_w, np.array([1.0,0,0]))
        if np.linalg.norm(r_w) < 1e-6:
            r_w = np.cross(f_w, np.array([0,1.0,0]))
    r_w = r_w / np.linalg.norm(r_w)
    u_w = np.cross(r_w, f_w); u_w /= np.linalg.norm(u_w)

    ux = np.tensordot(U, r_w, axes=([U.ndim-1],[0]))
    uy = np.tensordot(U, u_w, axes=([U.ndim-1],[0]))
    uz = np.tensordot(U, f_w, axes=([U.ndim-1],[0]))
    alpha = np.degrees(np.arctan2(ux, uz))  # horizontal angle
    beta  = np.degrees(np.arctan2(uy, uz))  # vertical angle
    return alpha, beta

def rectilinear_angles_to_uv(alpha_deg: np.ndarray, beta_deg: np.ndarray, img_w: int, img_h: int):
    # Rectilinear projection: x = tan(alpha), y = tan(beta); normalized by tan(HFOV/2), tan(VFOV/2)
    denom_x = math.tan(math.radians(HALF_H))
    denom_y = math.tan(math.radians(HALF_V))
    x = np.tan(np.radians(alpha_deg)) / denom_x  # [-1,1] at FOV edges
    y = np.tan(np.radians(beta_deg))  / denom_y
    u = (x + 1.0) * 0.5 * (img_w - 1)
    v = (y + 1.0) * 0.5 * (img_h - 1)
    return u, v

# --------------- RED overlay ingestion ---------------
def load_red_mask(image_path: str, min_R: int=120, delta: int=25) -> np.ndarray:
    """
    Return boolean mask (H,W): True where pixels are red-tinted (UNUSABLE).
    Threshold heuristic: R >= min_R and R >= G+delta and R >= B+delta.
    Adjust min_R/delta if your overlay is lighter/darker or semi-transparent.
    """
    im = Image.open(image_path).convert("RGB")
    arr = np.asarray(im).astype(np.int16)
    R, G, B = arr[...,0], arr[...,1], arr[...,2]
    return (R >= min_R) & (R >= G + delta) & (R >= B + delta)

def usable_mask_from_image(U, R_wc, f_cam, up_cam, img_path, min_R=120, delta=25):
    """
    Project each spherical direction to the lens image and mark usable if:
    - direction within the lens FOV, and
    - corresponding image pixel is NOT red.
    """
    im = Image.open(img_path).convert("RGB")
    w, h = im.size
    red = load_red_mask(img_path, min_R=min_R, delta=delta)  # True = unusable

    alpha, beta = angles_in_lens_frame(U, R_wc, f_cam, up_cam)
    within = (np.abs(alpha) <= HALF_H) & (np.abs(beta) <= HALF_V)
    if not np.any(within):
        return np.zeros(U.shape[:2], dtype=bool)

    u, v = rectilinear_angles_to_uv(alpha, beta, w, h)
    ui = np.clip(np.rint(u).astype(int), 0, w-1)
    vi = np.clip(np.rint(v).astype(int), 0, h-1)

    red_samp = red[vi, ui]
    usable = within & (~red_samp)
    return usable

# --------------- Coverage analytics ---------------
def coverage_for_rigs_with_red_masks(rigs: List[LadybugRig],
                                     lens_image_map: Dict[str,str],
                                     step_deg: int = 3):
    """
    lens_image_map keys must match: 'Front Top', 'Front Side +0°', 'Front Side +72°', 'Front Side +144°',
                                   'Front Side -144°', 'Front Side -72°', and the same for 'Rear ...'
    Missing images => assume fully usable FOV for that lens (minus nadir occlusion).
    """
    az, el, AZ, EL, U, area_elem = build_grid(step_deg)
    all_masks = []
    labels = []
    for rig in rigs:
        R_wc = rig.pose.R_world_from_cam()
        occ = occluded_mask(U, R_wc)  # housing occlusion
        for lname, f_cam, up_cam in rig.lenses():
            label = f"{rig.name} {lname}"
            if label in lens_image_map and os.path.exists(lens_image_map[label]):
                usable = usable_mask_from_image(U, R_wc, f_cam, up_cam, lens_image_map[label])
                m = usable & (~occ)
            else:
                # No red-mask image provided => use full FOV minus occlusion
                a,b = angles_in_lens_frame(U, R_wc, f_cam, up_cam)
                m = (np.abs(a) <= HALF_H) & (np.abs(b) <= HALF_V) & (~occ)
            all_masks.append(m)
            labels.append(label)

    masks = np.array(all_masks)        # (num_lenses, H, W)
    count = masks.sum(axis=0)          # lenses-per-direction
    return {"az":az,"el":el,"AZ":AZ,"EL":EL,"U":U,"area_elem":area_elem,"masks":masks,"labels":labels,"count":count}

def band_ok(count: np.ndarray, EL: np.ndarray, lo: float=-20, hi: float=75) -> bool:
    band = (EL >= lo) & (EL <= hi)
    return bool(np.all(count[band] >= 1))

def coverage_metrics(count: np.ndarray, EL: np.ndarray, area_elem: np.ndarray):
    mask_any = count > 0
    below = EL < 0
    total = float(np.sum(area_elem[mask_any])/(4*np.pi)*100.0)
    below_sph = float(np.sum(area_elem[mask_any & below])/(4*np.pi)*100.0)
    below_half = float(np.sum(area_elem[mask_any & below]) / np.sum(area_elem[below]) * 100.0)
    return {
        "Total union coverage (% of sphere)": round(total,2),
        "Below-horizon union (% of sphere)": round(below_sph,2),
        "Below-horizon union (% of below half)": round(below_half,2),
    }

def per_lens_contribution(masks: np.ndarray, area_elem: np.ndarray) -> np.ndarray:
    # Even-split overlap: divide each pixel’s area by coverage count
    count = masks.sum(axis=0)
    w = np.zeros_like(count, dtype=float)
    m = count > 0
    w[m] = area_elem[m] / count[m]
    eff = np.array([float(np.sum(w[mk])) for mk in masks])
    return eff/(4*np.pi)*100.0

# --------------- Convenience plotting ---------------
def save_coverage_map(res: dict, save_path: str, title: str = "Combined coverage (lenses per direction)"):
    count = res["count"]
    az, el = res["az"], res["el"]
    plt.figure(figsize=(10,5))
    plt.title(title)
    plt.imshow(count, origin="lower", extent=[az.min(), az.max(), el.min(), el.max()], aspect="auto")
    plt.xlabel("Azimuth (°)"); plt.ylabel("Elevation (°)")
    plt.colorbar(label="# of lenses seeing direction")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_band_check(res: dict, lo: float, hi: float, save_path: str, title: str = None):
    count, EL = res["count"], res["EL"]
    band = (EL >= lo) & (EL <= hi)
    uncovered = ((count < 1) & band).astype(int)
    az, el = res["az"], res["el"]
    plt.figure(figsize=(10,5))
    plt.title(title or f"Band coverage check ({lo:.0f}°..{hi:.0f}°)")
    plt.imshow(uncovered, origin="lower", extent=[az.min(), az.max(), el.min(), el.max()], aspect="auto")
    plt.xlabel("Azimuth (°)"); plt.ylabel("Elevation (°)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_per_lens_bar(res: dict, save_path: str, title: str = "Per-lens contribution (% of sphere)"):
    pct = per_lens_contribution(res["masks"], res["area_elem"])
    order = np.argsort(-pct)
    labels = [res["labels"][i] for i in order]
    vals = [pct[i] for i in order]
    plt.figure(figsize=(10,6))
    plt.title(title)
    plt.barh(labels, vals)
    plt.xlabel("% of full sphere")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# --------------- Minimal demo (replace with your data) ---------------
if __name__ == "__main__":
    # Example pose (recommended we converged on):
    front = LadybugRig("Front", LadybugPose(yaw_deg=0.0,   pitch_deg=18.0, roll_deg=0.0))
    rear  = LadybugRig("Rear",  LadybugPose(yaw_deg=216.0, pitch_deg=18.0, roll_deg=0.0))

    # Map each lens label to your red-masked image path
    # e.g. lens_image_map["Front Side +0°"] = "/path/to/front_side_0_red.png"
    lens_image_map = {}  # fill with your files; if a lens is missing, we assume fully-usable minus occlusion

    res = coverage_for_rigs_with_red_masks([front, rear], lens_image_map, step_deg=3)
    print("Band −20..+75° covered?:", band_ok(res["count"], res["EL"], -20, 75))
    print("Coverage metrics:", coverage_metrics(res["count"], res["EL"], res["area_elem"]))

    os.makedirs("./out", exist_ok=True)
    save_coverage_map(res, "./out/coverage_map.png")
    save_band_check(res, -20, 75, "./out/band_check.png")
    save_per_lens_bar(res, "./out/per_lens_contribution.png")
    print("Saved to ./out/")
