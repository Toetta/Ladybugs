# run_from_io.py
import argparse, os, numpy as np, matplotlib.pyplot as plt
import ladybug_analytical_model as lb

def main():
    ap = argparse.ArgumentParser(description="Ladybug5+ analytical coverage + red masking")
    ap.add_argument("--poses_json", default=None, help="Path to poses.json")
    ap.add_argument("--poses_csv",  default=None, help="Path to poses.csv")
    ap.add_argument("--masks_json", default=None, help="Path to masks.json")
    ap.add_argument("--masks_csv",  default=None, help="Path to masks.csv")
    ap.add_argument("--images_csv", default=None, help="Optional image manifest: rig,lens,image_path")
    ap.add_argument("--step", type=int, default=3, help="Spherical grid step in degrees (default: 3)")
    ap.add_argument("--band_lo", type=float, default=-20, help="Band low elevation (deg)")
    ap.add_argument("--band_hi", type=float, default=75, help="Band high elevation (deg)")
    ap.add_argument("--out", default="out_report", help="Output folder")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # --- load rigs (poses) ---
    rigs = []
    if args.poses_json:
        rigs = lb.load_poses_json(args.poses_json)
    elif args.poses_csv:
        rigs = lb.load_poses_csv(args.poses_csv)
    else:
        raise SystemExit("Provide --poses_json or --poses_csv")

    # --- coverage ---
    res = lb.coverage_for_rigs(rigs, step_deg=args.step)
    ok = lb.band_ok(res["count"], res["EL"], args.band_lo, args.band_hi)
    metrics = lb.coverage_metrics(res["count"], res["EL"], res["area_elem"])
    print(f"Band {args.band_lo}..{args.band_hi} covered? -> {ok}")
    print("Coverage metrics:", metrics)

    # Coverage map
    plt.figure(figsize=(10,5))
    plt.title("Combined coverage (lenses per direction)")
    plt.imshow(res["count"], origin="lower",
               extent=[res["az"].min(), res["az"].max(), res["el"].min(), res["el"].max()],
               aspect="auto")
    plt.xlabel("Azimuth (°)"); plt.ylabel("Elevation (°)")
    plt.colorbar(label="# of lenses seeing direction")
    plt.tight_layout(); plt.savefig(os.path.join(args.out, "coverage_map.png")); plt.close()

    # Band check
    band = (res["EL"]>=args.band_lo) & (res["EL"]<=args.band_hi)
    uncovered = ((res["count"]<1) & band).astype(int)
    plt.figure(figsize=(10,5))
    plt.title(f"Band coverage check ({args.band_lo}..{args.band_hi}°)")
    plt.imshow(uncovered, origin="lower",
               extent=[res["az"].min(), res["az"].max(), res["el"].min(), res["el"].max()],
               aspect="auto")
    plt.xlabel("Azimuth (°)"); plt.ylabel("Elevation (°)")
    plt.tight_layout(); plt.savefig(os.path.join(args.out, "band_check.png")); plt.close()

    # Per-lens contribution
    pct = lb.per_lens_contribution(res["masks"], res["area_elem"])
    order = np.argsort(-pct)
    labels = [res["labels"][i] for i in order]
    vals = [pct[i] for i in order]
    plt.figure(figsize=(10,6))
    plt.title("Effective per-lens contribution")
    plt.barh(labels, vals)
    plt.xlabel("% of full sphere")
    plt.gca().invert_yaxis()
    plt.tight_layout(); plt.savefig(os.path.join(args.out, "per_lens_contribution.png")); plt.close()

    # --- optional image masking (overlay polygons onto provided images) ---
    masks = {}
    if args.masks_json:
        masks = lb.load_masks_json(args.masks_json)
    elif args.masks_csv:
        masks = lb.load_masks_csv(args.masks_csv)

    if args.images_csv and masks:
        manifest = lb.load_image_manifest_csv(args.images_csv)
        masked_dir = os.path.join(args.out, "masked")
        for (rig,lens), img_path in manifest.items():
            key = (rig, lb.normalize_lens_label(lens))
            if key not in masks:
                continue
            base = os.path.basename(img_path)
            out_path = os.path.join(masked_dir, rig, base.replace(".jpg","_MASKED.jpg").replace(".png","_MASKED.png"))
            lb.overlay_red_mask(img_path, masks[key], out_path, alpha=120)
        print(f"Masked images written to: {masked_dir}")

if __name__ == "__main__":
    main()
