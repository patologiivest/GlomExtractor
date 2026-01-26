import numpy as np
import cv2
from typing import Tuple
from openslide import OpenSlide


def replace_background(img: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
    """
    Replace background pixels (mask==0) with a solid RGB color.
    Assumptions:
      - img is RGB uint8 or compatible
      - mask is HxWx3 uint8 with 0 background and 255 foreground (as produced by fillPoly)
    """
    img_out = img.copy()
    img_out[mask[:, :, 0] == 0] = color
    return img_out


def get_img_from_coordinates(
    points: np.ndarray,
    slide: OpenSlide,
    xmin: int,
    ymin: int,
    wx: int,
    wy: int,
    f: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read a patch from an OpenSlide slide and rasterize a polygon mask.
    """
    img = np.array(slide.read_region((int(xmin), int(ymin)), f, (int(wx), int(wy))))[..., :3]

    mask = np.zeros(img.shape, dtype=np.uint8)
    pts = np.asarray(points, dtype=np.int32)

    pts_xy = np.stack([pts[:, 1], pts[:, 0]], axis=1)
    mask = cv2.fillPoly(mask, pts_xy.reshape((1, pts_xy.shape[0], 2)), [255, 255, 255])

    return img, mask


def calculate_polygon_coordinates_statistics(geometry: np.ndarray) -> Tuple[int, int, int, int, int, int, int, int]:
    """
    Compute min/max/mean/range in both axes for a polygon.
    geometry is expected as Nx2 with columns:
      geometry[:,0] = y
      geometry[:,1] = x
    """
    min_x, max_x = int(np.min(geometry[:, 1])), int(np.max(geometry[:, 1]))
    min_y, max_y = int(np.min(geometry[:, 0])), int(np.max(geometry[:, 0]))

    mean_x = round((min_x + max_x) / 2)
    mean_y = round((min_y + max_y) / 2)

    range_x = max_x - min_x
    range_y = max_y - min_y

    return min_x, max_x, min_y, max_y, mean_x, mean_y, range_x, range_y


def ensure_int_bounds(xmin: int, ymin: int, wx: int, wy: int) -> Tuple[int, int, int, int]:
    """
    Safe-Guard against negative window positions and non-positive sizes.
    Returns (xmin, ymin, wx, wy).
    """
    xmin = int(xmin)
    ymin = int(ymin)
    wx = int(wx)
    wy = int(wy)

    if wx <= 0 or wy <= 0:
        raise ValueError(f"Invalid patch size wx={wx}, wy={wy}")

    xmin = max(0, xmin)
    ymin = max(0, ymin)
    return xmin, ymin, wx, wy


def clip_points_to_patch(points: np.ndarray, wx: int, wy: int) -> np.ndarray:
    """
    Clip polygon points to patch bounds [0..wy-1] for y and [0..wx-1] for x.
    points: Nx2 in [y, x]
    """
    pts = np.asarray(points, dtype=np.float32).copy()
    pts[:, 0] = np.clip(pts[:, 0], 0, max(0, wy - 1))
    pts[:, 1] = np.clip(pts[:, 1], 0, max(0, wx - 1))
    return pts
