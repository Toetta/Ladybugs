# ladybug_analytical_model.py
# Red-mask–aware spherical coverage model for dual Ladybug5+ rigs
# Lens indexing (aligned to white "0" mark on housing):
# 0-front (+0°), 1-front-right (−72°), 2-back-right (−144°),
# 3-back-left (+144°), 4-front-left (+72°), 5-top (zenith)

from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np
import math, csv, os
from PIL import Image

# ---------------- Rotations ----------------
def Rz(yaw: float) -> np.ndarray:
    a = math.radians(yaw); c, s = math.cos(a), math.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=float)

def Ry(pitch: float) -> np.ndarray:
    a = math.radians(pitch); c, s = math.cos(a), math.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]], dtype=float)

def Rx(roll: float) -> np.ndarray:
    a = math.radians(roll); c, s = math.cos(a), math.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=float)

# ---------------- Ladybug optics ----------------
HFOV, VFOV = 87.6, 77.5       # degrees (rectilinear approximation)
HALF_H, HALF_V = HFOV/2, VFOV/2
NADIR_HALF = 36.87            # housing occlusion half-angle (deg)

# Five side lenses at 72° spacing + Top lens
# Ordered to match index scheme 0..4 on the ring:
SIDE_AZS = [0, -72, -144, +144, +72]   # deg around +Z_cam (counterclockwise is +)
TOP_VEC  = np.array([0.0, 0.0, 1.0])   # top lens (zenith)

def side_forward(az_deg: float) -> np.ndarray:
    a = math.radians(az_deg)
    return np.array([math.cos(a), math.sin(a), 0.0], dtype=float)

# ---------------- Index-based naming (0..5) ----------------
# Canonical set used internally (ring + top)
CANONICAL_LENSES = [
    "Side +0°",    # 0-front
    "Side -72°",   # 1-front-right
    "Side -144°",  # 2-back-right
    "Side +144°",  # 3-back-left
    "Side +72°",   # 4-front-left
    "Top"          # 5-top
]

# Your aliases -> canonical
ALIAS_TO_CANONICAL = {
    "0-front":        "Side +0°",
    "1-front-right":  "Side -72°",
    "2-back-right":   "Side -144°",
    "3-back-left":    "Side +144°",
    "4-front-left":   "Side +72°",
    "5-top":          "Top",
    # tolerate simple forms
    "0": "Side +0°", "1": "Side -72°", "2": "Side -144°",
    "3": "Side +144°", "4": "Side +72°", "5": "Top",
    "top": "Top",
}

# Canonical -> pretty label for charts
CANONICAL_TO_PRETTY = {
    "Side +0°":   "0-front",
    "Side -72°":  "1-front-right",
    "Side -144°": "2-back-right",
    "Side +144°": "3-back-left",
    "Side +72°":  "4-front-left",
    "Top":        "5-top",
}

def normalize_lens_label(raw: str) -> str:
    """Accept index aliases (0-front..5-top) or canonical 'Side ±…°'/'Top'; return canonical."""
    if raw is None:
        return raw
    s = raw.strip().lower().replace("degrees","°").replace("deg","°")
    # normalize variants
    s = (s.replace("side 0", "side +0°")
           .replace("side +0", "side +0°")
           .replace("side 72", "side +72°")
           .replace("side +72", "side +72°")
           .replace("side 144", "side +144°")
           .replace("side +144", "side +144°")
           .replace("side -72", "side -72°")
           .replace("side -144", "side -144°"))
    if s in ALIAS_TO_CANONICAL:
        return ALIAS_TO_CANONICAL[s]
    CANON_LO = {
        "top":"Top","side +0°":"Side +0°","side -72°":"Side -72°",
        "side -144°":"Side -144°","side +144°":"Side +144°","side +72°":"Side +72°"
    }
    if s in CANON_LO:
        return CANON_LO[s]
    return raw.strip()

def pretty_label(canonical: str) -> str:
    return CANONICAL_TO_PRETTY.get(canonical, canonical)

# ---------------- Poses & rigs ----------------
@dataclass
class LadybugPose:
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    def R_world_from_cam(self) -> np.ndarray:
        return Rz(self.yaw_deg) @ Ry(self.pitch_deg) @ Rx(self.roll_deg)

@dataclass
class LadybugRig:
    name: str
    pose: LadybugPose
    def lenses(self) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        """
        Returns (label, forward_cam, up_cam) for six lenses:
        five sides with up_cam = +Z_cam, and top with up_cam = +Y_cam.
        """
        return [
            ("Side +0°",   side_forward(+0),   np.array([0,0,1.0], float)),
            ("Side -72°",  side_forward(-72),  np.array([0,0,1.0], float)),
            ("Side -144°", side_forward(-144), np.array([0,0,1.0], float)),
            ("Side +144°", side_forward(+144), np.array([0,0,1.0], float)),
            ("Side +72°",  side_forward(+72),  np.array([0,0,1.0], float)),
            ("Top",        TOP_VEC,            np.array([0,1.0,0], float)),
        ]

# ---------------- Sphere grid ----------------
def build_grid(step: int = 3):
    az = np.arange(-180, 181, step)
    el = np.arange(-90,   91, step)
    AZ, EL = np.meshgrid(az, el)
    U = np.stack([np.cos(np.radians(EL))*np.cos(np.radians(AZ)),
                  np.cos(np.radians(EL))*np.sin(np.radians(AZ)),
                  np.sin(np.radians(EL))], axis=-1)
    weight = np.cos(np.radians(EL))
    d = math.radians(step)
    area_elem = weight * d * d
    return az, el, AZ, EL, U, area_elem

def occluded_mask(U: np.ndarray, R_wc: np.ndarray) -> np.ndarray:
    """Occluded if angle to -Z_cam <= NADIR_HALF."""
    U_cam = U @ R_wc.T
    cosang = -U_cam[...,2]
    cosang = np.clip(cosang, -1, 1)
    ang = np.degrees(np.arccos(cosang))
    return ang <= NADIR_HALF

def angles_in_lens_frame(U: np.ndarray, R_wc: np.ndarray, f_cam: np.ndarray, up_cam: np.ndarray):
    """Return (alpha,beta) in degrees for each world direction projected into a lens local frame."""
    f_w = R_wc @ np.array(f_cam, float)
    up_w = R_wc @ np.array(up_cam, float)
    r_w = np.cross(f_w, up_w)
    if np.linalg.norm(r_w) < 1e-6:
        r_w = np.cross(f_w, np.array([1.0,0,0], float))
        if np.linalg.norm(r_w) < 1e-6:
            r_w = np.cross(f_w, np.array([0,1.0,0], float))
    r_w = r_w / np.linalg.norm(r_w)
    u_w = np.cross(r_w, f_w); u_w = u_w / np.linalg.norm(u_w)

    ux = np.tensordot(U, r_w, axes=([U.ndim-1],[0]))
    uy = np.tensordot(U, u_w, axes=([U.ndim-1],[0]))
    uz = np.tensordot(U, f_w, axes=([U.ndim-1],[0]))
    alpha = np.degrees(np.arctan2(ux, uz))  # horizontal angle
    beta  = np.degrees(np.arctan2(uy, uz))  # vertical angle
    return alpha, beta

# ---------------- Red mask handling ----------------
def red_mask_from_image(path: str,
                        r_min: int = 180, g_max: int = 80, b_max: int = 80,
                        alpha_bins: int = 721, beta_bins: int = 641) -> np.ndarray:
    """
    Convert a pre-masked lens image into a boolean usable grid over (alpha,beta).
    Any sufficiently red pixel (R>=r_min, G<=g_max, B<=b_max) is excluded (False).
    Returns array shape (beta_bins, alpha_bins): True=usable, False=masked red.
    """
    A = Image.open(path).convert("RGB").resize((alpha_bins, beta_bins), Image.BILINEAR)
    A = np.asarray(A)
    R, G, B = A[...,0], A[...,1], A[...,2]
    is_red = (R >= r_min) & (G <= g_max) & (B <= b_max)
    usable = ~is_red
    return usable

def load_image_manifest_csv(path: str) -> Dict[Tuple[str,str], str]:
    """
    CSV columns: rig,lens,image_path
    lens: index aliases (e.g. '0-front', '5-top') or canonical ('Side +0°', 'Top').
    """
    out: Dict[Tuple[str,str], str] = {}
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            rig = row["rig"].strip()
            canon = normalize_lens_label(row["lens"])
            # ignore any accidental "bottom" names (no nadir lens exists)
            if canon.lower().startswith("bottom"):
                print(f"[warn] Ignoring non-existent lens ‘{row['lens']}’ on rig {rig}.")
                continue
            out[(rig, canon)] = row["image_path"]
    return out

# ---------------- Coverage with red masks ----------------
def coverage_for_rigs_with_red(rigs: List[LadybugRig],
                               image_manifest: Dict[Tuple[str,str], str],
                               r_min: int = 180, g_max: int = 80, b_max: int = 80,
                               alpha_bins: int = 721, beta_bins: int = 641,
                               step_deg: int = 3):
    """
    Build union coverage for multiple Ladybug rigs, where each lens’ FOV is gated by
    a pre-supplied red mask (True=usable, False=red).
    """
    # Precompute per-lens usable grids
    usable_grid: Dict[Tuple[str,str], np.ndarray] = {}
    for rig in rigs:
        for lname, _, _ in rig.lenses():
            key = (rig.name, lname)
            if key in image_manifest:
                usable_grid[key] = red_mask_from_image(
                    image_manifest[key], r_min=r_min, g_max=g_max, b_max=b_max,
                    alpha_bins=alpha_bins, beta_bins=beta_bins
                )
            else:
                usable_grid[key] = np.ones((beta_bins, alpha_bins), dtype=bool)

    az, el, AZ, EL, U, area_elem = build_grid(step_deg)
    masks = []
    labels = []

    for rig in rigs:
        Rwc = rig.pose.R_world_from_cam()
        occ = occluded_mask(U, Rwc)
        for lname, f_cam, up_cam in rig.lenses():
            alpha, beta = angles_in_lens_frame(U, Rwc, f_cam, up_cam)
            in_fov = (np.abs(alpha) <= HALF_H) & (np.abs(beta) <= HALF_V) & (~occ)
            if not np.any(in_fov):
                masks.append(in_fov)
                labels.append(lname)
                continue
            # map (alpha,beta) to indices in usable grid
            xi = np.clip(((alpha + HALF_H) / HFOV) * (alpha_bins - 1), 0, alpha_bins - 1).astype(int)
            yi = np.clip(((beta  + HALF_V) / VFOV) * (beta_bins  - 1), 0, beta_bins  - 1).astype(int)
            usable = usable_grid[(rig.name, lname)][yi, xi]
            masks.append(in_fov & usable)
            labels.append(lname)

    masks = np.array(masks)          # (num_lenses, H, W)
    count = masks.sum(axis=0)        # lenses per direction after red gating
    return {
        "az": az, "el": el, "AZ": AZ, "EL": EL, "area_elem": area_elem,
        "masks": masks, "labels": labels, "count": count
    }

# ---------------- Metrics ----------------
def band_ok(count: np.ndarray, EL: np.ndarray, lo: float = -20, hi: float = 75) -> bool:
    band = (EL >= lo) & (EL <= hi)
    return bool(np.all(count[band] >= 1))

def coverage_metrics(count: np.ndarray, EL: np.ndarray, area_elem: np.ndarray) -> Dict[str, float]:
    mask_any = count > 0
    below = EL < 0
    total = 100.0 * np.sum(area_elem[mask_any]) / (4*np.pi)
    below_sph = 100.0 * np.sum(area_elem[mask_any & below]) / (4*np.pi)
    below_half = 100.0 * np.sum(area_elem[mask_any & below]) / np.sum(area_elem[below]) * 100.0
    return {
        "Total union coverage (% of sphere)": round(float(total), 2),
        "Below-horizon union (% of sphere)": round(float(below_sph), 2),
        "Below-horizon union (% of below half)": round(float(below_half), 2),
    }

def per_lens_contribution(masks: np.ndarray, area_elem: np.ndarray) -> np.ndarray:
    """
    Effective % contribution per lens after red subtraction.
    Evenly split solid angle among all lenses that see a given direction.
    """
    count = masks.sum(axis=0)
    w = np.zeros_like(count, dtype=float)
    m = count > 0
    w[m] = area_elem[m] / count[m]
    eff = np.array([float(np.sum(w[mk])) for mk in masks])
    return eff / (4*np.pi) * 100.0  # % of full sphere
