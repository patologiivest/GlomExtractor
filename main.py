import argparse
import os

from GlomExtractor import GlomExtractor
from Utils.saver import save_images


def main():
    parser = argparse.ArgumentParser(description="Glomerulus patch extraction & export (manual-draft compatible)")

    parser.add_argument("path", type=str, help="Dataset folder or single image")

    parser.add_argument("--out_folder", "-o", type=str, default="../OUTPUT/", help="Output root folder")

    # Match manual draft defaults
    parser.add_argument("--patch_size", "-w", type=int, default=1000, help="Patch width in pixels (default: 1000)")
    parser.add_argument("--out_size", type=int, default=224, help="Export size for PNGs (default: 224)")

    # Manual draft: most exports used these as True
    parser.add_argument("--save_masks", action="store_true", default=True, help="Also save masks (default: True)")
    parser.add_argument("--no_masks", action="store_true", help="Disable saving masks")
    parser.add_argument("--save_markups", action="store_true", default=True, help="Also save markups (default: True)")
    parser.add_argument("--no_markups", action="store_true", help="Disable saving markups")

    parser.add_argument("--csv", action="store_true", default=True, help="Save full dataframe CSV (default: True)")
    parser.add_argument("--no_csv", action="store_true", help="Disable saving CSV")

    # Filter thresholds to match manual draft
    parser.add_argument("--min_area", type=float, default=20000.0, help="FilteredSmall: keep area >= min_area (default: 20000)")
    parser.add_argument("--max_area", type=float, default=1200000.0, help="FilteredLarge: keep area <= max_area (default: 1200000)")
    parser.add_argument("--max_white", type=float, default=0.75, help="FilteredWhite: keep percWhite <= max_white (default: 0.75)")
    parser.add_argument("--min_lap", type=float, default=400.0, help="FilteredBlur: keep LaplacianVariance >= min_lap (default: 400)")

    # Manual draft uses 10th percentile thresholds for these
    parser.add_argument("--min_circ_q", type=float, default=0.10, help="FilteredCircularity: keep >= quantile (default: 0.10)")
    parser.add_argument("--min_ch_q", type=float, default=0.10, help="FilteredCHmetric: keep >= quantile (default: 0.10)")

    args = parser.parse_args()

    save_masks = False if args.no_masks else bool(args.save_masks)
    save_markups = False if args.no_markups else bool(args.save_markups)

    out_root = args.out_folder
    os.makedirs(out_root, exist_ok=True)

    extractor = GlomExtractor(args.path)

    extractor.calculate_shape_descriptors()

    in_folder = args.path if os.path.isdir(args.path) else os.path.dirname(os.path.dirname(args.path))
    extractor.add_white_and_lap(w=args.patch_size, in_folder=in_folder)

    df = extractor.return_dataframe()
    print(f"Found {len(df)} annotations.")

    save_csv = args.csv and not args.no_csv
    if save_csv:
        df.to_csv(os.path.join(out_root, "annotations_full.csv"), index=False)

    # -----------------------------
    # Filtered exports (match manual)
    # -----------------------------
    df_work = df.copy()

    # FilteredSmall: include area >= 20000
    filtered_small = df_work[df_work["area"] >= args.min_area].copy()
    _export_filtered(
        name="FilteredSmall",
        df=filtered_small,
        extractor=extractor,
        out_root=out_root,
        patch_size=args.patch_size,
        out_size=args.out_size,
        save_masks=save_masks,
        save_markups=save_markups,
    )

    # FilteredLarge: include area <= 1200000
    filtered_large = filtered_small[filtered_small["area"] <= args.max_area].copy()
    _export_filtered(
        name="FilteredLarge",
        df=filtered_large,
        extractor=extractor,
        out_root=out_root,
        patch_size=args.patch_size,
        out_size=args.out_size,
        save_masks=save_masks,
        save_markups=save_markups,
    )

    # FilteredWhite: include percWhite <= 0.75
    if "percWhite" in filtered_large.columns:
        filtered_white = filtered_large[filtered_large["percWhite"] <= args.max_white].copy()
    else:
        filtered_white = filtered_large.copy()
    _export_filtered(
        name="FilteredWhite",
        df=filtered_white,
        extractor=extractor,
        out_root=out_root,
        patch_size=args.patch_size,
        out_size=args.out_size,
        save_masks=save_masks,
        save_markups=save_markups,
    )

    # FilteredBlur: include LaplacianVariance >= 400
    if "LaplacianVariance" in filtered_white.columns:
        filtered_blur = filtered_white[filtered_white["LaplacianVariance"] >= args.min_lap].copy()
    else:
        filtered_blur = filtered_white.copy()
    _export_filtered(
        name="FilteredBlur",
        df=filtered_blur,
        extractor=extractor,
        out_root=out_root,
        patch_size=args.patch_size,
        out_size=args.out_size,
        save_masks=save_masks,
        save_markups=save_markups,
    )

    # FilteredCircularity: include >= quantile(0.1) computed from FULL anno distribution
    circ_thr = df["circularity"].quantile(args.min_circ_q) if "circularity" in df.columns else None
    if circ_thr is not None:
        filtered_circ = filtered_blur[filtered_blur["circularity"] >= circ_thr].copy()
    else:
        filtered_circ = filtered_blur.copy()
    _export_filtered(
        name="FilteredCircularity",
        df=filtered_circ,
        extractor=extractor,
        out_root=out_root,
        patch_size=args.patch_size,
        out_size=args.out_size,
        save_masks=save_masks,
        save_markups=save_markups,
    )

    # FilteredCHmetric: include >= quantile(0.1) computed from FULL anno distribution
    ch_thr = df["chMetric"].quantile(args.min_ch_q) if "chMetric" in df.columns else None
    if ch_thr is not None:
        filtered_ch = filtered_circ[filtered_circ["chMetric"] >= ch_thr].copy()
    else:
        filtered_ch = filtered_circ.copy()
    _export_filtered(
        name="FilteredCHmetric",
        df=filtered_ch,
        extractor=extractor,
        out_root=out_root,
        patch_size=args.patch_size,
        out_size=args.out_size,
        save_masks=save_masks,
        save_markups=save_markups,
    )

    # -----------------------------
    # Style/scaling exports (match manual)
    # -----------------------------
    PRESETS = [
        ("FixedSqueeze-Tissue", 3, False, (0, 0, 0)),
        ("Fixed-Tissue", 0, False, (0, 0, 0)),
        ("Fixed-Black", 0, True, (0, 0, 0)),
        ("Fixed-Green", 0, True, (0, 255, 0)),
        ("Fixed-White", 0, True, (255, 255, 255)),
        ("Independent-Black", 1, True, (0, 0, 0)),
        ("Dependent-Black", 2, True, (0, 0, 0)),
        ("Dependent-Tissue", 2, False, (0, 0, 0)),
        ("Independent-Tissue", 1, False, (0, 0, 0)),
    ]

    for name, scaling, crop, color in PRESETS:
        out_folder = os.path.join(out_root, name)
        print(f"\nExporting: {name} -> {out_folder}")

        save_images(
            df=df,
            extractor=extractor,
            out_folder=out_folder,
            scaling=scaling,
            save_mask=save_masks,
            save_markup=save_markups,
            crop=crop,
            color=color,
            patch_size=args.patch_size,
            out_size=args.out_size,
        )

    print("\nAll done!")


def _export_filtered(
    name: str,
    df,
    extractor,
    out_root: str,
    patch_size: int,
    out_size: int,
    save_masks: bool,
    save_markups: bool,
):
    out_folder = os.path.join(out_root, name)
    kept = len(df)
    print(f"\nExporting: {name} ({kept} kept) -> {out_folder}")

    if kept == 0:
        print("  Skipping (no rows after filtering).")
        return

    # Manual draft uses scaling=0, crop=False for all filtered exports
    save_images(
        df=df,
        extractor=extractor,
        out_folder=out_folder,
        scaling=0,
        save_mask=save_masks,
        save_markup=save_markups,
        crop=False,
        color=(0, 0, 0),
        patch_size=patch_size,
        out_size=out_size,
    )


if __name__ == "__main__":
    main()
