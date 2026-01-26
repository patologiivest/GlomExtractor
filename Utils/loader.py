import os
import json
import xmltodict


def load_annotations(annotations_path: str) -> list[dict]:
    """
    Load annotations and return a standardized list of structures.

    Standard output:
      [
        {
          "id": <str|int|None>,
          "geometry": [[y, x], [y, x], ...],   # IMPORTANT: always [y,x] for GlomExtractor
          "properties": {...}                 # optional metadata
        },
        ...
      ]

    Supported inputs:
      - .geojson / .json GeoJSON FeatureCollection (Polygon, MultiPolygon)
      - .json COCO-style (polygon segmentations only; RLE skipped) + bbox fallback
      - .json custom {"structures":[...]} or list of {geometry/points}
      - .xml Pascal VOC (bbox)
      - .xml QuPath/Aperio (Annotations/Annotation/Regions/Region/Vertices/Vertex[@X,@Y])
    """
    ext = os.path.splitext(annotations_path)[-1].lower()

    if ext in [".json", ".geojson"]:
        with open(annotations_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return _from_json_like(data)

    if ext == ".xml":
        with open(annotations_path, "r", encoding="utf-8") as f:
            xml_data = xmltodict.parse(f.read())
        return _from_xml_like(xml_data)

    raise ValueError(f"Unsupported annotation file extension: {ext}")


# -------------------------
# JSON / GEOJSON
# -------------------------
def _from_json_like(data) -> list[dict]:
    out: list[dict] = []

    # 1) GeoJSON FeatureCollection
    if isinstance(data, dict) and data.get("type") == "FeatureCollection":
        for feature in data.get("features", []):
            props = feature.get("properties") or {}
            fid = feature.get("id") or props.get("id") or props.get("AnnotationId")

            geom = feature.get("geometry") or {}
            for ring in _geojson_geom_to_rings_yx(geom):
                out.append({
                    "id": fid,
                    "geometry": ring,
                    "properties": props
                })
        return out

    # 2) COCO-style JSON
    if isinstance(data, dict) and "annotations" in data:
        for ann in data.get("annotations", []):
            aid = ann.get("id")
            props = _drop_keys(ann, keys={"segmentation", "bbox"})

            seg = ann.get("segmentation")
            bbox = ann.get("bbox")

            if isinstance(seg, list):
                # segmentation: list of flat arrays [x1,y1,x2,y2,...]
                k = 0
                for flat in seg:
                    ring = _coco_flat_to_ring_yx(flat)
                    if ring is None:
                        continue
                    out.append({
                        "id": f"{aid}_{k}" if aid is not None else None,
                        "geometry": ring,
                        "properties": props
                    })
                    k += 1
                continue

            # fallback: bbox
            ring = _bbox_to_ring_yx(bbox)
            if ring is not None:
                out.append({
                    "id": aid,
                    "geometry": ring,
                    "properties": props
                })
        return out

    # 3) Custom: {"structures":[...]}
    if isinstance(data, dict) and "structures" in data:
        for s in data.get("structures", []):
            ring = _points_to_ring_yx(s.get("geometry") or s.get("points"))
            if ring is None:
                continue
            out.append({
                "id": s.get("id"),
                "geometry": ring,
                "properties": s.get("properties") or {}
            })
        return out

    # 4) Custom list: [{...}, {...}]
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            ring = _points_to_ring_yx(item.get("geometry") or item.get("points"))
            if ring is None:
                continue
            out.append({
                "id": item.get("id"),
                "geometry": ring,
                "properties": item.get("properties") or {}
            })
        return out

    # 5) Single dict fallback
    if isinstance(data, dict):
        ring = _points_to_ring_yx(data.get("geometry") or data.get("points"))
        if ring is not None:
            out.append({
                "id": data.get("id"),
                "geometry": ring,
                "properties": data.get("properties") or {}
            })
        return out

    return out


def _geojson_geom_to_rings_yx(geom: dict) -> list[list[list[float]]]:
    """
    Convert GeoJSON geometry to a list of exterior rings.
    Each ring returned as [[y,x], ...] (exterior only; holes ignored).
    """
    gtype = (geom or {}).get("type")
    coords = (geom or {}).get("coordinates")
    rings: list[list[list[float]]] = []

    if gtype == "Polygon" and coords:
        ring = _xy_ring_to_yx(coords[0])
        if ring is not None:
            rings.append(ring)

    elif gtype == "MultiPolygon" and coords:
        for poly in coords:
            if not poly:
                continue
            ring = _xy_ring_to_yx(poly[0])
            if ring is not None:
                rings.append(ring)

    return rings


def _xy_ring_to_yx(exterior_xy) -> list[list[float]] | None:
    
    if not exterior_xy or len(exterior_xy) < 3:
        return None

    if exterior_xy[0] == exterior_xy[-1]:
        exterior_xy = exterior_xy[:-1]

    if len(exterior_xy) < 3:
        return None

    return [[float(y), float(x)] for x, y in exterior_xy]


def _coco_flat_to_ring_yx(flat) -> list[list[float]] | None:
    if not flat or len(flat) < 6:
        return None
    ring = []
    for j in range(0, len(flat), 2):
        x = float(flat[j])
        y = float(flat[j + 1])
        ring.append([y, x])
    return ring if len(ring) >= 3 else None


def _bbox_to_ring_yx(bbox) -> list[list[float]] | None:
    """
    COCO bbox expected: [x, y, w, h]
    Returns ring [[y,x], ...]
    """
    if bbox is None or len(bbox) != 4:
        return None
    x, y, w, h = bbox
    xmin, ymin = float(x), float(y)
    xmax, ymax = float(x + w), float(y + h)
    return [[ymin, xmin], [ymin, xmax], [ymax, xmax], [ymax, xmin]]


def _points_to_ring_yx(points) -> list[list[float]] | None:
    """
    Convert [(x,y), ...] or [[x,y], ...] to [[y,x], ...].
    """
    if not points or len(points) < 3:
        return None

    ring = []
    for p in points:
        if p is None or len(p) < 2:
            continue
        x, y = p[0], p[1]
        ring.append([float(y), float(x)])

    return ring if len(ring) >= 3 else None


# -------------------------
# XML
# -------------------------
def _from_xml_like(xml_data: dict) -> list[dict]:
    out: list[dict] = []

    # 1) Pascal VOC
    if "annotation" in xml_data:
        annotation = xml_data.get("annotation") or {}
        objects = annotation.get("object", [])
        if isinstance(objects, dict):
            objects = [objects]

        for obj in objects:
            obj_id = obj.get("name")
            bbox = obj.get("bndbox") or {}
            if not bbox:
                continue

            xmin = float(bbox["xmin"])
            ymin = float(bbox["ymin"])
            xmax = float(bbox["xmax"])
            ymax = float(bbox["ymax"])

            ring = [[ymin, xmin], [ymin, xmax], [ymax, xmax], [ymax, xmin]]
            out.append({
                "id": obj_id,
                "geometry": ring,
                "properties": {"source": "pascal_voc"}
            })
        return out

    # 2) QuPath/Aperio XML
    if "Annotations" in xml_data:
        annotations = xml_data.get("Annotations") or {}
        annots = annotations.get("Annotation", [])
        if isinstance(annots, dict):
            annots = [annots]

        for annot in annots:
            annot_id = annot.get("@Id")
            regions = (annot.get("Regions") or {}).get("Region", [])
            if isinstance(regions, dict):
                regions = [regions]

            for region in regions:
                region_id = region.get("@Id") or annot_id
                vertices = (region.get("Vertices") or {}).get("Vertex", [])
                if isinstance(vertices, dict):
                    vertices = [vertices]

                ring = []
                for v in vertices:
                    x = float(v["@X"])
                    y = float(v["@Y"])
                    ring.append([y, x])  # IMPORTANT: store [y,x]

                if len(ring) >= 3:
                    out.append({
                        "id": region_id,
                        "geometry": ring,
                        "properties": {"source": "qupath_aperio"}
                    })
        return out

    return out


def _drop_keys(d: dict, keys: set) -> dict:
    return {k: v for k, v in d.items() if k not in keys}
