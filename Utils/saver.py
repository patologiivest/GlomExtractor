import os
import numpy as np
from PIL import Image
from openslide import open_slide


def save_images(
    df,
    extractor,
    out_folder: str,
    scaling: int = 0,
    save_mask: bool = False,
    save_markup: bool = False,
    crop: bool = False,
    color: tuple[int, int, int] = (0, 0, 0),
    patch_size: int = 2000,
    out_size: int = 224
):
    """
    Export extracted patches (and optional masks/markups) to disk.

    Assumptions:
      - df has columns: imgName, annId (optional), geometry (used inside extractor methods)
      - extractor has methods:
          get_img_fixed / get_img_independent / get_img_dependent / get_img_fixedSqueeze
          get_markup
          replace_background
          return_dict_of_images
      - dict_of_images keys are base names (stem), values contain "image_path"
      - df["imgName"] stores the original image filename (with extension)

    Notes:
      - All outputs are saved as PNG resized to (out_size, out_size).
    """
    _ensure_dirs(out_folder, save_mask, save_markup)

    dict_of_images = extractor.return_dict_of_images()
    images = np.unique(df["imgName"])

    k = 0
    for img_name in images:
        image_path = _resolve_image_path(dict_of_images, img_name)
        if image_path is None:
            print(f"Warning: could not resolve image path for '{img_name}', skipping.")
            continue
        if not os.path.exists(image_path):
            print(f"Warning: '{image_path}' does not exist, skipping.")
            continue

        slide = open_slide(image_path)
        sub = df[df["imgName"] == img_name]

        for j in sub.index.values:
            print(f"[{k}, {j}]", end="\r")

            img, mask, XY = _extract_patch(extractor, j, sub, slide, scaling, patch_size)

            ann_id = sub.at[j, "annId"] if "annId" in sub.columns else str(j)

            if save_markup:
                markup = extractor.get_markup(img, mask, XY)
                _save_rgb_png(markup, os.path.join(out_folder, "MarkUps", f"{ann_id}.png"), out_size)

            save_img = img
            if crop:
                save_img = extractor.replace_background(save_img, mask, color)

            _save_rgb_png(save_img, os.path.join(out_folder, "Images", f"{ann_id}.png"), out_size)

            if save_mask:
                _save_rgb_png(mask, os.path.join(out_folder, "Masks", f"{ann_id}.png"), out_size)

            k += 1


# -----------------------------
# Internal helpers
# -----------------------------
def _ensure_dirs(out_folder: str, save_mask: bool, save_markup: bool):
    os.makedirs(os.path.join(out_folder, "Images"), exist_ok=True)
    if save_mask:
        os.makedirs(os.path.join(out_folder, "Masks"), exist_ok=True)
    if save_markup:
        os.makedirs(os.path.join(out_folder, "MarkUps"), exist_ok=True)


def _resolve_image_path(dict_of_images: dict, img_name: str) -> str | None:
    """
    Resolve slide path from dict_of_images, using the stem of img_name.

    dict_of_images structure expected:
      dict_of_images[stem] = {"image_path": "...", "annotation_path": "..."}
    """
    stem = os.path.splitext(img_name)[0]
    entry = dict_of_images.get(stem)

    if entry and entry.get("image_path"):
        return entry["image_path"]

    # fallback search (useful if keys differ between -f and -d modes)
    for v in dict_of_images.values():
        p = v.get("image_path")
        if p and os.path.basename(p) == img_name:
            return p

    return None


def _extract_patch(extractor, row_idx: int, sub_df, slide, scaling: int, patch_size: int):
    if scaling == 0:
        return extractor.get_img_fixed(row_idx, sub_df, slide, patch_size)
    if scaling == 1:
        return extractor.get_img_independent(row_idx, sub_df, slide)
    if scaling == 2:
        return extractor.get_img_dependent(row_idx, sub_df, slide)
    if scaling == 3:
        return extractor.get_img_fixedSqueeze(row_idx, sub_df, slide, patch_size)
    raise ValueError(f"Invalid scaling mode: {scaling} (expected 0,1,2,3)")


def _save_rgb_png(arr: np.ndarray, path: str, out_size: int):
    img = Image.fromarray(arr.astype("uint8"), "RGB")
    if out_size is not None:
        img = img.resize((out_size, out_size))
    img.save(path, format="PNG")
