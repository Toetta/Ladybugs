# run_from_images.py
# Use with ladybug_analytical_model.py (red-mask aware model with 0..5 lens naming)
# Example:
#   py run_from_images.py --poses_json poses.json --images_csv images.csv --out out_report --step 3 --band_lo -20 --band_hi 75

import argparse
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from ladybug_analytical_model import (
    LadybugPose, LadybugRig,
    load_image_manifest_csv,
    coverage_for_rigs_with_red, band_ok, coverage_metrics, per_lens_contribution, pretty_label
)

# --- Pose I/O (kept local so the library stays lean) ---
def load_poses_json(path):
    import json
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rigs = []
    for r in data["rigs"]:
        rigs.append(
            LadybugRig(
                r["name"],
                LadybugPose(float(r["yaw"]), float(r["pitch"]), float(r["roll"]))
            )
        )
    return rigs

def load_poses_csv(path):
    import csv
    rigs = []
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            rigs.append(
                LadybugRig(
                    r["name"],
                    LadybugPose(float(r["yaw"]), float(r["pitch"]), float(r["roll"]))
                )
            )
    return rigs

def write_dict_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)

def main():
    ap = argparse.ArgumentParser(
        description="Ladybug5+ spherical coverage using pre-masked (red) lens images"
    )
    ap.add_argument("--poses_json", default=None, help="Poses JSON: {'rigs':[{'name','yaw','pitch','roll'},...]}")
    ap.add_argument("--poses_csv",  default=None, help="Poses CSV: name,yaw,pitch,roll")
    ap.add_argument("--images_csv", required=True, help="CSV: rig,lens,image_path (lens may be 0-front..5-top or canonical)")
    ap.add_argument("--step", type=int, default=3, help="Spherical grid step (deg). Default 3")
    ap.add_argument("--band_lo", type=float, default=-20, help="Band low elevation (deg). Default -20")
    ap.add_argument("--band_hi", type=float, default=75,  help="Band high elevation (deg). Default +75")
    # Red detection thresholds (R>=, G<=, B<=)
    ap.add_argument("--red_r_min", type=int, default=180, help="Min R for red mask. Default 180")
    ap.add_argument("--red_g_max", type=int, default=80,  help="Max G for red mask. Default 80")
    ap.add_argument("--red_b_max", type=int, default=80,  help="Max B for red mask. Default 80")
    # Lens image sampling grid (alpha × beta)
    ap.add_argument("--alpha_bins", type=int, default=721, help="Horizontal bins across HFOV. Default 721")
    ap.add_argument("--beta_bins",  type=int, default=641, help="Vertical bins across VFOV. Default 641")
    ap.add_argument("--out", default="out_report", help="Output folder. Default out_report")
    ap.add_argument("--csv", action="store_true", help="Also write CSV summaries")

    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # --- Load poses ---
    if args.poses_json:
        rigs = load_poses_json(args.poses_json)
    elif args.poses_csv:
        rigs = load_poses_csv(args.poses_csv)
    else:
        raise SystemExit("Provide --poses_json or --poses_csv")

    # --- Load image manifest (each row points to a PRE-MASKED image with red over unusable regions) ---
    manifest = load_image_manifest_csv(args.images_csv)

    # --- Compute coverage with red-masked subtraction ---
    res = coverage_for_rigs_with_red(
        rigs, manifest,
        r_min=args.red_r_min, g_max=args.red_g_max, b_max=args.red_b_max,
        alpha_bins=args.alpha_bins, beta_bins=args.beta_bins,
        step_deg=args.step
    )

    # --- Band check + metrics ---
    ok = band_ok(res["count"], res["EL"], args.band_lo, args.band_hi)
    metrics = coverage_metrics(res["count"], res["EL"], res["area_elem"])
    print(f"Band {args.band_lo}..{args.band_hi} covered? -> {ok}")
    print("Coverage metrics:", metrics)

    # --- Figures ---
    # Coverage (lenses per direction)
    plt.figure(figsize=(10,5))
    plt.title("Combined coverage (lenses per direction) — after red masking")
    plt.imshow(
        res["count"], origin="lower",
        extent=[res["az"].min(), res["az"].max(), res["el"].min(), res["el"].max()],
        aspect="auto"
    )
    plt.xlabel("Azimuth (°)"); plt.ylabel("Elevation (°)")
    plt.colorbar(label="# of lenses seeing direction")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "coverage_map.png"))
    plt.close()

    # Band coverage check (0=covered, 1=uncovered)
    band = (res["EL"] >= args.band_lo) & (res["EL"] <= args.band_hi)
    uncovered = ((res["count"] < 1) & band).astype(int)
    plt.figure(figsize=(10,5))
    plt.title(f"Band coverage check ({args.band_lo}..{args.band_hi}°) — 0=covered, 1=uncovered")
    plt.imshow(
        uncovered, origin="lower",
        extent=[res["az"].min(), res["az"].max(), res["el"].min(), res["el"].max()],
        aspect="auto"
    )
    plt.xlabel("Azimuth (°)"); plt.ylabel("Elevation (°)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "band_check.png"))
    plt.close()

    # Per-lens contributions (post red subtraction)
    pct = per_lens_contribution(res["masks"], res["area_elem"])
    order = np.argsort(-pct)
    labels = [pretty_label(res["labels"][i]) for i in order]
    vals = [pct[i] for i in order]

    plt.figure(figsize=(10,6))
    plt.title("Effective per-lens contribution (after red masking)")
    plt.barh(labels, vals)
    plt.xlabel("% of full sphere")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "per_lens_contribution.png"))
    plt.close()

    # --- Optional CSV exports ---
    if args.csv:
        # summary
        summary_rows = [{
            "band_lo_deg": args.band_lo,
            "band_hi_deg": args.band_hi,
            "band_pass": "Yes" if ok else "No",
            **metrics
        }]
        write_dict_csv(os.path.join(args.out, "summary.csv"), summary_rows,
                       fieldnames=["band_lo_deg","band_hi_deg","band_pass",
                                   "Total union coverage (% of sphere)",
                                   "Below-horizon union (% of sphere)",
                                   "Below-horizon union (% of below half)"])
        # per-lens table
        lens_rows = [{"lens": labels[i], "percent_of_sphere": round(float(vals[i]), 5)} for i in range(len(labels))]
        write_dict_csv(os.path.join(args.out, "per_lens_contribution.csv"), lens_rows,
                       fieldnames=["lens","percent_of_sphere"])

if __name__ == "__main__":
    main()
