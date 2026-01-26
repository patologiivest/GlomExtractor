import os
import glob
from typing import Optional, List, Dict, Tuple, Any

import numpy as np
import pandas as pd
import tqdm
import cv2
from shapely.geometry import Polygon
from openslide import open_slide, OpenSlide

# Local imports
from Utils.loader import load_annotations
from Utils.utils import (
    replace_background,
    get_img_from_coordinates,
    calculate_polygon_coordinates_statistics,
)

from histomicstk.saliency.tissue_detection import threshold_multichannel
from histomicstk.preprocessing.color_conversion import rgb_to_hsi


class GlomExtractor:
    """
    Extract glomeruli information from medical images with annotations.

    Conventions:
      - All polygon points are stored as [y, x]
      - geometry in dataframe is a list/array of points: [[y,x], [y,x], ...]
      - get_img_from_coordinates expects points in [y,x] local patch coordinates
    """

    def __init__(self, path: str = None):
        assert path is not None, "Path cannot be None"

        self.mode = self._prompt_mode()
        self.dict_of_images: Dict[str, Dict[str, Optional[str]]] = {}
        self.dataframe: Optional[pd.DataFrame] = None

        self.image_path: Optional[str] = None
        self.structures: Optional[List[dict]] = None

        if self.mode == "-d":
            self._init_dataset(path)
        elif self.mode == "-f":
            self._init_single_image(path)
        else:
            raise ValueError("Undefined mode. Use '-f' for image file or '-d' for dataset.")

    # -----------------------------
    # Init helpers
    # -----------------------------
    @staticmethod
    def _prompt_mode() -> str:
        while True:
            mode = input("Is the path an image file (-f) or a dataset (-d)? ").lower()
            if mode in ["-f", "-d"]:
                return mode
            print("Invalid input. Please enter '-f' for image file or '-d' for dataset.")

    def _init_dataset(self, path: str):
        self._validate_dir(path)
        images_dir, annotations_dir = self._get_dataset_dirs(path)

        annotation_files = glob.glob(os.path.join(annotations_dir, "*"))
        for ann_path in annotation_files:
            base_name = os.path.splitext(os.path.basename(ann_path))[0]
            image_path = self._find_image_for_annotation(images_dir, base_name)

            self.dict_of_images[base_name] = {
                "image_path": self._safe_relpath(image_path),
                "annotation_path": self._safe_relpath(ann_path),
            }

        self.build_dataframe(self.mode, self.dict_of_images)

    def _init_single_image(self, path: str):
        if os.path.isdir(path):
            raise ValueError("The path provided is a directory and not a file.")

        images_dir, annotations_dir = self._get_single_image_dirs(path)
        base_name = os.path.splitext(os.path.basename(path))[0]

        image_path = path
        ann_path = self._find_annotation_for_image(annotations_dir, base_name)

        self.dict_of_images[base_name] = {
            "image_path": self._safe_relpath(image_path),
            "annotation_path": self._safe_relpath(ann_path),
        }

        self.image_path = image_path
        self.structures = self.load_single_image_annotations(self.dict_of_images, base_name)
        self.build_dataframe(self.mode, self.dict_of_images, self.structures)

    @staticmethod
    def _validate_dir(path: str):
        if not os.path.isdir(path):
            raise ValueError("The path provided is not a directory.")

    @staticmethod
    def _get_dataset_dirs(path: str) -> Tuple[str, str]:
        return (
            os.path.join(path, "Images"),
            os.path.join(path, "Annotations"),
        )
    

    @staticmethod
    def _get_single_image_dirs(path: str) -> Tuple[str, str]:
        root = os.path.dirname(os.path.dirname(path))
        return (
            os.path.join(root, "Images"),
            os.path.join(root, "Annotations"),
        )

    @staticmethod
    def _find_image_for_annotation(images_dir: str, base_name: str) -> Optional[str]:
        candidates = glob.glob(os.path.join(images_dir, base_name + ".*"))
        return candidates[0] if candidates else None

    @staticmethod
    def _find_annotation_for_image(annotations_dir: str, base_name: str) -> Optional[str]:
        candidates = glob.glob(os.path.join(annotations_dir, f"{base_name}.*"))
        return candidates[0] if candidates else None

    @staticmethod
    def _safe_relpath(path: Optional[str]) -> Optional[str]:
        return os.path.relpath(path) if path else None

    # -----------------------------
    # Loading + DataFrame
    # -----------------------------
    def load_single_image_annotations(self, dict_of_images: Dict[str, Dict[str, str]], key: str) -> List[dict]:
        if key not in dict_of_images:
            raise KeyError(f"Key '{key}' not found in dict_of_images.")
        annotation_path = dict_of_images[key]["annotation_path"]
        return load_annotations(annotation_path)

    def build_dataframe(
        self,
        mode: str,
        dict_of_images: Dict[str, Dict[str, str]],
        structures: Optional[List[dict]] = None,
    ) -> bool:
        def core_procedure(image_path: str, structs: List[dict]) -> pd.DataFrame:
            geometry = [self._get_geometry(s) for s in structs]
            geometry = [g for g in geometry if g is not None and len(g) >= 3]

            idx_list = list(range(len(geometry)))
            img_name = [os.path.basename(image_path)] * len(idx_list)
            ann_id = [f"{img_name[0]}_{idx+1}" for idx in idx_list]

            return pd.DataFrame(
                {
                    "IDX": idx_list,
                    "imgName": img_name,
                    "annId": ann_id,
                    "geometry": geometry,
                }
            )

        if mode == "-d":
            all_rows = []
            for key, img_info in dict_of_images.items():
                image_path = img_info.get("image_path")
                annotation_path = img_info.get("annotation_path")
                if image_path is None or annotation_path is None:
                    continue

                structs = self.load_single_image_annotations(dict_of_images, key)
                df = core_procedure(image_path, structs)
                all_rows.append(df)

            self.dataframe = (
                pd.concat(all_rows, ignore_index=True)
                if all_rows
                else pd.DataFrame(columns=["IDX", "imgName", "annId", "geometry"])
            )

        elif mode == "-f":
            if self.image_path is None:
                raise ValueError("image_path is not set for -f mode")
            self.dataframe = core_procedure(self.image_path, structures or [])

        else:
            raise ValueError("Unknown mode...")

        return True

    @staticmethod
    def _get_geometry(structure: dict) -> Any:
        return structure.get("geometry") if structure.get("geometry") is not None else None

    # -----------------------------
    # Shape descriptors
    # -----------------------------
    def calculate_shape_descriptors(self) -> bool:
        if self.dataframe is None or len(self.dataframe) == 0:
            return True

        for idx in tqdm.tqdm(self.dataframe.index.values, desc="Calculating shape features..."):
            self._calculate_single_shape_descriptor(idx)

        return True

    def _calculate_single_shape_descriptor(self, idx: int):
        try:
            geometry = self._get_geometry_from_row(idx)  # Nx2 [y,x]
            polygon = Polygon([(float(x), float(y)) for y, x in geometry])  # shapely expects (x,y)

            convex = polygon.convex_hull

            self.dataframe.at[idx, "area"] = float(polygon.area)
            self.dataframe.at[idx, "circularity"] = float(self._calculate_circularity(polygon))
            self.dataframe.at[idx, "chMetric"] = float(self._calculate_ch_metric(polygon, convex))

        except Exception as e:
            print(f"Error on index {idx}: {e}")

    def _get_geometry_from_row(self, idx: int) -> np.ndarray:
        geom = np.asarray(self.dataframe.at[idx, "geometry"], dtype=np.float32)
        return geom[:, 0:2]

    @staticmethod
    def _calculate_circularity(polygon: Polygon) -> float:
        return 4.0 * np.pi * polygon.area / (polygon.length ** 2) if polygon.length != 0 else 0.0

    @staticmethod
    def _calculate_ch_metric(polygon: Polygon, convex: Polygon) -> float:
        return polygon.area / convex.area if convex.area != 0 else float("nan")

    # -----------------------------
    # White % + Laplacian variance
    # -----------------------------
    def add_white_and_lap(self, w: int, in_folder: str) -> pd.DataFrame:
        """
        Compute:
          - percWhite: fraction of background pixels that are "white-ish" in HSI
          - LaplacianVariance: blur proxy on grayscale patch
        """
        if self.dataframe is None or len(self.dataframe) == 0:
            return self.dataframe

        images = np.unique(self.dataframe["imgName"])

        for img_name in images:
            sub = self.dataframe[self.dataframe["imgName"] == img_name]
            slide = open_slide(self._build_slide_path(in_folder, img_name))

            for j in sub.index.values:
                img, mask, _ = self.get_img_fixed(j, sub, slide, w, 1)

                mask_gray = self._to_gray(mask)
                gray_img = self._to_gray(img)

                pixels = img[mask_gray == 0]  # background pixels
                white_mask = self._calculate_white_mask(pixels)

                if len(white_mask) == 0:
                    self.dataframe.loc[j, "percWhite"] = float("nan")
                else:
                    self.dataframe.loc[j, "percWhite"] = float(np.sum(white_mask[:, 0]) / len(white_mask[:, 0]))

                self.dataframe.loc[j, "LaplacianVariance"] = float(cv2.Laplacian(gray_img, cv2.CV_64F).var())

        return self.dataframe

    @staticmethod
    def _build_slide_path(in_folder: str, img_name: str) -> str:
        """
        Build a slide path without forcing an extension.

        Strategy:
        1) If img_name already has an extension, try that exact file in in_folder/Images
        2) Otherwise (or if missing), search for any file with the same stem in in_folder/Images
        """
        images_dir = os.path.join(in_folder, "Images")

        # 1) Try exact filename first
        direct = os.path.join(images_dir, img_name)
        if os.path.exists(direct):
            return direct

        # 2) Try by stem (any extension)
        stem = os.path.splitext(img_name)[0]
        candidates = glob.glob(os.path.join(images_dir, stem + ".*"))
        if candidates:
            return candidates[0]  # pick first match

        raise FileNotFoundError(f"No slide found for '{img_name}' in '{images_dir}'")

    @staticmethod
    def _to_gray(img: np.ndarray) -> np.ndarray:
        if img.ndim == 3 and img.shape[-1] == 3:
            return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if img.ndim == 3 and img.shape[-1] == 1:
            return img.squeeze(-1)
        if img.ndim == 2:
            return img
        raise ValueError("Unsupported image format for grayscaling.")

    @staticmethod
    def _calculate_white_mask(pixels: np.ndarray) -> np.ndarray:
        if pixels is None or len(pixels) == 0:
            return np.zeros((0, 1), dtype=np.uint8)

        hsi = rgb_to_hsi(pixels)
        white_mask, _ = threshold_multichannel(
            hsi,
            {
                "hue": {"min": 0, "max": 1.0},
                "saturation": {"min": 0, "max": 0.2},
                "intensity": {"min": 220, "max": 255},
            },
            just_threshold=True,
        )
        return white_mask

    # -----------------------------
    # Markup + polygon helpers
    # -----------------------------
    @staticmethod
    def get_markup(img: np.ndarray, mask: np.ndarray, XY: np.ndarray) -> np.ndarray:
        """
        Draw polygon outline and fill overlay on img.
        XY expected in local coords as [y,x].
        """
        pts = np.asarray(XY, dtype=np.int32)
        pts_xy = np.stack([pts[:, 1], pts[:, 0]], axis=1)  # to [x,y] for OpenCV

        anno = cv2.fillPoly(img.copy(), pts_xy.reshape((1, pts_xy.shape[0], 2)), [0, 255, 255])
        markup = cv2.addWeighted(anno, 0.5, img, 0.5, 0)

        anno2 = cv2.polylines(markup.copy(), pts_xy.reshape((1, pts_xy.shape[0], 2)), True, [0, 255, 255], 5)
        markup = cv2.addWeighted(anno2, 0.8, markup, 0.2, 0)

        return markup

    @staticmethod
    def get_poly_points(
        r: int,
        df: pd.DataFrame,
        scale: float = 1.0
    ) -> Tuple[np.ndarray, int, int, int, int, int, int, int, int]:
        """
        Extract polygon points and bounding stats.

        Returns:
          XY: Nx2 int32 [y,x]
          xmin/xmax based on x
          ymin/ymax based on y
        """
        XY = np.asarray(df.at[r, "geometry"], dtype=np.float32)[:, 0:2].copy()

        if scale != 1.0:
            XY = XY / float(scale)

        XY = XY.astype(np.int32)

        xmin, xmax = int(np.min(XY[:, 1])), int(np.max(XY[:, 1]))
        ymin, ymax = int(np.min(XY[:, 0])), int(np.max(XY[:, 0]))

        xmean = round((xmin + xmax) / 2)
        ymean = round((ymin + ymax) / 2)

        xrange = xmax - xmin
        yrange = ymax - ymin

        return XY, xmin, xmax, ymin, ymax, xmean, ymean, xrange, yrange

    # -----------------------------
    # Cropping utilities
    # -----------------------------
    def get_img_fixed(self, r: int, df: pd.DataFrame, slide: OpenSlide, w: int, f: int = 0):
        scale = slide.level_downsamples[f]
        new_w = int(np.round(w / scale))

        _, _, _, _, _, xmean0, ymean0, _, _ = self.get_poly_points(r, df, scale=1.0)
        XY, _, _, _, _, xmean, ymean, _, _ = self.get_poly_points(r, df, scale=scale)

        # shift polygon into local patch coords
        XY[:, 0] = XY[:, 0] - ymean + int(new_w / 2)
        XY[:, 1] = XY[:, 1] - xmean + int(new_w / 2)

        img, mask = get_img_from_coordinates(
            XY,
            slide,
            xmin=xmean0 - int(w / 2),
            ymin=ymean0 - int(w / 2),
            wx=new_w,
            wy=new_w,
            f=f,
        )
        return img, mask, XY

    def get_img_independent(self, r: int, df: pd.DataFrame, slide: OpenSlide):
        XY, xmin, xmax, ymin, ymax, _, _, xrange, yrange = self.get_poly_points(r, df)

        XY[:, 0] = XY[:, 0] - ymin
        XY[:, 1] = XY[:, 1] - xmin

        img, mask = get_img_from_coordinates(XY, slide, xmin, ymin, xrange, yrange)
        return img, mask, XY

    def get_img_dependent(self, r: int, df: pd.DataFrame, slide: OpenSlide):
        XY, xmin, xmax, ymin, ymax, _, _, xrange, yrange = self.get_poly_points(r, df)

        w = max(xrange, yrange)
        if yrange > xrange:
            xmax += int(np.ceil((yrange - xrange) / 2))
            xmin -= int(np.floor((yrange - xrange) / 2))
        else:
            ymax += int(np.ceil((xrange - yrange) / 2))
            ymin -= int(np.floor((xrange - yrange) / 2))

        XY[:, 0] = XY[:, 0] - ymin
        XY[:, 1] = XY[:, 1] - xmin

        img, mask = get_img_from_coordinates(XY, slide, xmin, ymin, w, w)
        return img, mask, XY

    def get_img_fixedSqueeze(self, r: int, df: pd.DataFrame, slide: OpenSlide, w: int):
        XY, _, _, _, _, xmean, ymean, xrange, yrange = self.get_poly_points(r, df)

        if xrange > w or yrange > w:
            img, mask, XY = self.get_img_dependent(r, df, slide)
        else:
            XY[:, 0] = XY[:, 0] - ymean + int(w / 2)
            XY[:, 1] = XY[:, 1] - xmean + int(w / 2)
            img, mask = get_img_from_coordinates(XY, slide, xmean - int(w / 2), ymean - int(w / 2), w, w)

        return img, mask, XY

    # -----------------------------
    # Public utilities
    # -----------------------------
    def replace_background(self, img: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
        return replace_background(img, mask, color)

    def return_dataframe(self) -> pd.DataFrame:
        return self.dataframe

    def return_dict_of_images(self) -> Dict[str, Dict[str, Optional[str]]]:
        return self.dict_of_images
