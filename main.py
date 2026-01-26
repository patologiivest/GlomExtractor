import argparse
import os

from GlomExtractor import GlomExtractor
from Utils.saver import save_images


def main():
    parser = argparse.ArgumentParser(description="Glomerulus patch extraction & export")

    parser.add_argument("path", type=str, help="Dataset folder or single image")
    parser.add_argument("--out_folder", "-o", type=str, default="../OUTPUT/", help="Output root folder")

    parser.add_argument("--patch_size", "-w", type=int, default=1000, help="Patch width in pixels (default: 1000)")
    parser.add_argument("--out_size", type=int, default=224, help="Export size for PNGs (default: 224)")

    # These default to True, and can be disabled with flags
    parser.add_argument("--save_masks", action="store_true", default=True, help="Also save masks (default: True)")
    parser.add_argument("--no_masks", action="store_true", help="Disable saving masks")
    parser.add_argument("--save_markups", action="store_true", default=True, help="Also save markups (default: True)")
    parser.add_argument("--no_markups", action="store_true", help="Disable saving markups")

    parser.add_argument("--csv", action="store_true", default=True, help="Save full dataframe CSV (default: True)")
    parser.add_argument("--no_csv", action="store_true", help="Disable saving CSV")

    # Filtering controls
    parser.add_argument("--no_filters", action="store_true", help="Disable all Filtered* exports")
    parser.add_argument(
        "--filters",
        nargs="+",
        default=["small", "large", "white", "blur", "circularity", "chmetric"],
        choices=["small", "large", "white", "blur", "circularity", "chmetric"],
        help="Which Filtered* exports to generate (default: all)",
    )

    # Filter thresholds
    parser.add_argument("--min_area", type=float, default=20000.0, help="FilteredSmall: keep area >= min_area (default: 20000)")
    parser.add_argument("--max_area", type=float, default=1200000.0, help="FilteredLarge: keep area <= max_area (default: 1200000)")
    parser.add_argument("--max_white", type=float, default=0.75, help="FilteredWhite: keep percWhite <= max_white (default: 0.75)")
    parser.add_argument("--min_lap", type=float, default=400.0, help="FilteredBlur: keep LaplacianVariance >= min_lap (default: 400)")

    # 10th percentile thresholds for these (computed from FULL dataset)
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

    if df is None or len(df) == 0:
        print("No annotations found. Exiting.")
        return

    save_csv = args.csv and not args.no_csv
    if save_csv:
        df.to_csv(os.path.join(out_root, "annotations_full.csv"), index=False)

    # -----------------------------
    # Filtered exports (optional)
    # -----------------------------
    enabled_filters = set(args.filters)

    if not args.no_filters:
        df_work = df.copy()

        # if shape columns don't exist, skip filters that need them
        have_area = "area" in df_work.columns
        have_white = "percWhite" in df_work.columns
        have_lap = "LaplacianVariance" in df_work.columns
        have_circ = "circularity" in df_work.columns
        have_ch = "chMetric" in df_work.columns

        # FilteredSmall
        if "small" in enabled_filters:
            if not have_area:
                print("Skipping FilteredSmall (missing column: area)")
            else:
                filtered_small = df_work[df_work["area"] >= args.min_area].copy()
                _export_filtered("FilteredSmall", filtered_small, extractor, out_root, args.patch_size, args.out_size, save_masks, save_markups)
        else:
            filtered_small = df_work

        current = df_work

        if "small" in enabled_filters and have_area:
            current = current[current["area"] >= args.min_area].copy()

        # FilteredLarge
        if "large" in enabled_filters:
            if not have_area:
                print("Skipping FilteredLarge (missing column: area)")
            else:
                filtered_large = current[current["area"] <= args.max_area].copy()
                _export_filtered("FilteredLarge", filtered_large, extractor, out_root, args.patch_size, args.out_size, save_masks, save_markups)
        if "large" in enabled_filters and have_area:
            current = current[current["area"] <= args.max_area].copy()

        # FilteredWhite
        if "white" in enabled_filters:
            if not have_white:
                print("Skipping FilteredWhite (missing column: percWhite)")
            else:
                filtered_white = current[current["percWhite"] <= args.max_white].copy()
                _export_filtered("FilteredWhite", filtered_white, extractor, out_root, args.patch_size, args.out_size, save_masks, save_markups)
        if "white" in enabled_filters and have_white:
            current = current[current["percWhite"] <= args.max_white].copy()

        # FilteredBlur
        if "blur" in enabled_filters:
            if not have_lap:
                print("Skipping FilteredBlur (missing column: LaplacianVariance)")
            else:
                filtered_blur = current[current["LaplacianVariance"] >= args.min_lap].copy()
                _export_filtered("FilteredBlur", filtered_blur, extractor, out_root, args.patch_size, args.out_size, save_masks, save_markups)
        if "blur" in enabled_filters and have_lap:
            current = current[current["LaplacianVariance"] >= args.min_lap].copy()

        # FilteredCircularity (threshold from full df)
        if "circularity" in enabled_filters:
            if not have_circ:
                print("Skipping FilteredCircularity (missing column: circularity)")
            else:
                circ_thr = df["circularity"].quantile(args.min_circ_q)
                filtered_circ = current[current["circularity"] >= circ_thr].copy()
                _export_filtered("FilteredCircularity", filtered_circ, extractor, out_root, args.patch_size, args.out_size, save_masks, save_markups)
        if "circularity" in enabled_filters and have_circ:
            circ_thr = df["circularity"].quantile(args.min_circ_q)
            current = current[current["circularity"] >= circ_thr].copy()

        # FilteredCHmetric (threshold from full df)
        if "chmetric" in enabled_filters:
            if not have_ch:
                print("Skipping FilteredCHmetric (missing column: chMetric)")
            else:
                ch_thr = df["chMetric"].quantile(args.min_ch_q)
                filtered_ch = current[current["chMetric"] >= ch_thr].copy()
                _export_filtered("FilteredCHmetric", filtered_ch, extractor, out_root, args.patch_size, args.out_size, save_masks, save_markups)

    # -----------------------------
    # Style/scaling exports (always created)
    # (folder_name, scaling_mode, crop_background, background_color)
    #Scaling modes 
    """0 = Fixed (get_img_fixed) 
        Extract a fixed-size square patch (patch_size × patch_size) centered on the glomerulus. 
        If the glomerulus is larger than the patch, it can get clipped. 
        
        1 = Independent (get_img_independent) 
        Extract a patch that is exactly the polygon bounding box (width and height can differ). 
        This does not preserve aspect ratio when later resized to 224×224 (it may stretch). 
        
        2 = Dependent (get_img_dependent) 
        Extract a square patch that tightly contains the polygon bounding box (uses max(width,height)). 
        Aspect ratio is preserved better (still resized, but starts square). 3
        = FixedSqueeze (get_img_fixedSqueeze) 
            Hybrid: if the glomerulus fits inside patch_size, it behaves like Fixed if it’s larger, it falls back to Dependent so you don’t clip large glomeruli
    """
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
