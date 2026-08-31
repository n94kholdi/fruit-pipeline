#!/usr/bin/env python3
"""Generate a slicing-aided fine-tuning dataset: full images + tile crops, one COCO JSON.

We tile at inference (``fruit_pipeline.detection.tiling``) but train/use pretrained
weights on full images -- SAHI's own paper and ASAHI (arXiv 2604.19233) both
show slicing-aided *fine-tuning* is a separate recall gain on top of
slicing-aided *inference*. This script takes a COCO-format annotated
dataset (e.g. the ground truth bootstrapped via annotate_helper.py + CVAT/
Label Studio) and emits a training set containing both the original
full-resolution images (annotations unchanged) and sliced tile crops
(annotations transformed into tile-local coordinates), so a model can be
fine-tuned on the same tiled view it will see at inference time.

Usage (run from fruit-pipeline/):
    python scripts/make_tile_dataset.py \\
        --gt gt.json --images-dir path/to/images \\
        --output-dir tile_dataset --output-json tile_dataset/annotations.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from sahi.slicing import slice_image

from fruit_pipeline.detection.tiling import compute_tile_size_from_diameter

logger = logging.getLogger(__name__)


def transform_box_to_tile(
    box_xyxy: list[float],
    tile_rect: tuple[float, float, float, float],
) -> list[float] | None:
    """Clip a full-image ``[x1,y1,x2,y2]`` box into tile-local coordinates.

    Returns ``None`` if the box doesn't intersect the tile at all. The
    inverse of this (tile-local -> full-image) is just adding back
    ``(tile_x1, tile_y1)`` -- covered by the round-trip test in
    ``tests/test_make_tile_dataset.py``, since a sign/axis-swap bug here
    would otherwise be silent (every box would land somewhere plausible-
    looking, just wrong).
    """
    tx1, ty1, tx2, ty2 = tile_rect
    x1, y1, x2, y2 = box_xyxy
    cx1, cy1 = max(x1, tx1), max(y1, ty1)
    cx2, cy2 = min(x2, tx2), min(y2, ty2)
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    return [cx1 - tx1, cy1 - ty1, cx2 - tx1, cy2 - ty1]


def visible_fraction(original_box_xyxy: list[float], clipped_box_xyxy: list[float]) -> float:
    ox1, oy1, ox2, oy2 = original_box_xyxy
    original_area = max(0.0, ox2 - ox1) * max(0.0, oy2 - oy1)
    if original_area <= 0:
        return 0.0
    cx1, cy1, cx2, cy2 = clipped_box_xyxy
    clipped_area = max(0.0, cx2 - cx1) * max(0.0, cy2 - cy1)
    return clipped_area / original_area


def median_gt_diameter(annotations: list[dict]) -> float | None:
    """Median GT box diagonal (px), for resolution-aware ("diameter") adaptive tile sizing.

    Unlike the inference tiler's ``estimate_fruit_diameter_px`` (which needs a live
    detector coarse-pass), we already have real ground-truth boxes here, so
    the diameter estimate is exact rather than a detector-noise-prone guess.
    """
    diagonals = []
    for ann in annotations:
        w, h = ann["bbox"][2], ann["bbox"][3]
        if w > 0 and h > 0:
            diagonals.append(float(np.hypot(w, h)))
    if not diagonals:
        return None
    return float(np.median(diagonals))


def choose_tile_size(
    image_width: int,
    image_height: int,
    annotations: list[dict],
    adaptive_mode: str,
    tile_size_k: float,
    min_tile_size: int,
    max_tile_size: int,
    fixed_slice_count: int,
) -> int:
    """Pick a tile size for one image, per ``--adaptive-mode``.

    - ``diameter`` (default): reuses ``detect.compute_tile_size_from_diameter``
      fed the image's own median GT box diagonal -- the same "same physical
      fruit size -> similar pixel tile size" idea ASAHI (arXiv 2604.19233)
      argues for, just driven by real annotations instead of a coarse
      detector pass.
    - ``resolution``: tile_size = long_edge / fixed_slice_count -- a
      simpler knob that only looks at image resolution, if you'd rather not
      depend on annotation density for the size estimate.
    """
    if adaptive_mode == "resolution":
        long_edge = max(image_width, image_height)
        return max(min_tile_size, min(max_tile_size, int(round(long_edge / fixed_slice_count))))

    diameter = median_gt_diameter(annotations)
    if diameter is None:
        long_edge = max(image_width, image_height)
        return max(min_tile_size, min(max_tile_size, int(round(long_edge / fixed_slice_count))))
    return compute_tile_size_from_diameter(diameter, k=tile_size_k, min_tile_size=min_tile_size, max_tile_size=max_tile_size)


def make_tile_dataset(
    coco_gt: dict,
    images_dir: str,
    output_dir: str,
    tile_size: int | None,
    overlap_ratio: float,
    min_visible_fraction: float,
    negative_tile_fraction: float,
    adaptive_mode: str,
    tile_size_k: float,
    min_tile_size: int,
    max_tile_size: int,
    fixed_slice_count: int,
    seed: int = 0,
) -> dict:
    """Build the combined (full-image + tile-crop) COCO dataset and write tile crops to ``output_dir``.

    Returns the new COCO dict; does not write the annotations JSON itself
    (caller decides the output path) but DOES write every tile crop image
    file to ``output_dir`` as a side effect, since those files are what the
    new COCO ``file_name`` entries point at.
    """
    rng = random.Random(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    anns_by_image: dict[int, list[dict]] = {}
    for ann in coco_gt.get("annotations", []):
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    new_images: list[dict] = []
    new_annotations: list[dict] = []
    next_image_id = max((img["id"] for img in coco_gt["images"]), default=0) + 1
    next_ann_id = max((ann["id"] for ann in coco_gt.get("annotations", [])), default=0) + 1
    category_id = coco_gt["categories"][0]["id"] if coco_gt.get("categories") else 1

    n_positive_tiles = n_negative_candidates = n_negative_kept = 0

    for image_record in coco_gt["images"]:
        # Full-resolution image passes through unchanged.
        new_images.append(dict(image_record))
        for ann in anns_by_image.get(image_record["id"], []):
            new_annotations.append(dict(ann))

        src_path = Path(images_dir) / image_record["file_name"]
        image_bgr = cv2.imread(str(src_path))
        if image_bgr is None:
            logger.warning("Skipping tiling for %s: cannot read image at %s", image_record["file_name"], src_path)
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]

        this_tile_size = tile_size or choose_tile_size(
            width,
            height,
            anns_by_image.get(image_record["id"], []),
            adaptive_mode,
            tile_size_k,
            min_tile_size,
            max_tile_size,
            fixed_slice_count,
        )

        slice_result = slice_image(
            image=image_rgb,
            slice_height=this_tile_size,
            slice_width=this_tile_size,
            overlap_height_ratio=overlap_ratio,
            overlap_width_ratio=overlap_ratio,
            auto_slice_resolution=False,
        )

        stem = Path(image_record["file_name"]).stem
        negative_candidates: list[tuple[np.ndarray, str, int, int]] = []

        for tile_idx, (tile_rgb, starting_pixel) in enumerate(zip(slice_result.images, slice_result.starting_pixels)):
            shift_x, shift_y = int(starting_pixel[0]), int(starting_pixel[1])
            tile_h, tile_w = tile_rgb.shape[:2]
            tile_rect = (float(shift_x), float(shift_y), float(shift_x + tile_w), float(shift_y + tile_h))

            tile_anns = []
            for ann in anns_by_image.get(image_record["id"], []):
                x, y, w, h = ann["bbox"]
                original_box = [x, y, x + w, y + h]
                clipped = transform_box_to_tile(original_box, tile_rect)
                if clipped is None:
                    continue
                if visible_fraction(original_box, clipped) < min_visible_fraction:
                    continue
                tile_anns.append(clipped)

            tile_file_name = f"{stem}_tile{tile_idx:03d}.jpg"
            if not tile_anns:
                n_negative_candidates += 1
                negative_candidates.append((tile_rgb, tile_file_name, tile_w, tile_h))
                continue

            n_positive_tiles += 1
            cv2.imwrite(str(output_path / tile_file_name), cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2BGR))
            new_images.append(
                {
                    "id": next_image_id,
                    "file_name": tile_file_name,
                    "width": tile_w,
                    "height": tile_h,
                    "source_image_id": image_record["id"],
                }
            )
            for box in tile_anns:
                bx1, by1, bx2, by2 = box
                new_annotations.append(
                    {
                        "id": next_ann_id,
                        "image_id": next_image_id,
                        "category_id": category_id,
                        "bbox": [bx1, by1, bx2 - bx1, by2 - by1],
                        "area": (bx2 - bx1) * (by2 - by1),
                        "iscrowd": 0,
                    }
                )
                next_ann_id += 1
            next_image_id += 1

        n_keep = int(round(len(negative_candidates) * negative_tile_fraction))
        for tile_rgb, tile_file_name, tile_w, tile_h in rng.sample(negative_candidates, min(n_keep, len(negative_candidates))):
            n_negative_kept += 1
            cv2.imwrite(str(output_path / tile_file_name), cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2BGR))
            new_images.append(
                {
                    "id": next_image_id,
                    "file_name": tile_file_name,
                    "width": tile_w,
                    "height": tile_h,
                    "source_image_id": image_record["id"],
                }
            )
            next_image_id += 1

    logger.info(
        "Tile dataset: %d positive tile(s), %d/%d negative tile(s) kept (fraction=%.2f)",
        n_positive_tiles,
        n_negative_kept,
        n_negative_candidates,
        negative_tile_fraction,
    )

    return {
        "images": new_images,
        "annotations": new_annotations,
        "categories": coco_gt.get("categories", [{"id": category_id, "name": "fruit"}]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gt", required=True, help="Path to a ground-truth COCO JSON.")
    parser.add_argument("--images-dir", required=True, help="Directory containing the GT's source images.")
    parser.add_argument("--output-dir", required=True, help="Directory to write tile crop images into.")
    parser.add_argument("--output-json", required=True, help="Path to write the combined (full + tile) COCO JSON.")
    parser.add_argument("--tile-size", type=int, default=None, help="Fixed tile size; default: adaptive per --adaptive-mode.")
    parser.add_argument("--overlap-ratio", type=float, default=0.15, help="Fractional overlap between tiles (default 0.15).")
    parser.add_argument(
        "--min-visible-fraction",
        type=float,
        default=0.3,
        help="Keep a box clipped by a tile edge only if at least this fraction of its original area survives (default 0.3).",
    )
    parser.add_argument(
        "--negative-tile-fraction",
        type=float,
        default=0.1,
        help="Fraction of zero-annotation tiles to keep as negative examples (default 0.1).",
    )
    parser.add_argument(
        "--adaptive-mode",
        choices=["diameter", "resolution"],
        default="diameter",
        help="Adaptive tile sizing (ignored if --tile-size is set). 'diameter' (default): tile_size = k * median "
        "GT box diagonal, per image. 'resolution': tile_size = long_edge / --fixed-slice-count.",
    )
    parser.add_argument("--tile-size-k", type=float, default=8.0, help="--adaptive-mode diameter only (default 8.0).")
    parser.add_argument("--min-tile-size", type=int, default=320)
    parser.add_argument("--max-tile-size", type=int, default=2048)
    parser.add_argument("--fixed-slice-count", type=int, default=4, help="--adaptive-mode resolution only, or the fallback when no GT boxes exist for 'diameter' mode.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for negative-tile sampling.")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = build_parser().parse_args()

    with open(args.gt) as f:
        coco_gt = json.load(f)

    new_coco = make_tile_dataset(
        coco_gt=coco_gt,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        tile_size=args.tile_size,
        overlap_ratio=args.overlap_ratio,
        min_visible_fraction=args.min_visible_fraction,
        negative_tile_fraction=args.negative_tile_fraction,
        adaptive_mode=args.adaptive_mode,
        tile_size_k=args.tile_size_k,
        min_tile_size=args.min_tile_size,
        max_tile_size=args.max_tile_size,
        fixed_slice_count=args.fixed_slice_count,
        seed=args.seed,
    )

    # Full-resolution images are referenced by the new COCO JSON but were
    # never copied into output_dir (only tile crops were) -- copy them
    # alongside so the whole dataset resolves from one directory.
    for image_record in new_coco["images"]:
        if "source_image_id" in image_record:
            continue  # a tile crop, already written by make_tile_dataset
        src = Path(args.images_dir) / image_record["file_name"]
        dst = Path(args.output_dir) / image_record["file_name"]
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)

    output_json_path = Path(args.output_json)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(new_coco, f, indent=2)

    logger.info(
        "Wrote %d image(s) (%d annotation(s)) to %s",
        len(new_coco["images"]),
        len(new_coco["annotations"]),
        output_json_path,
    )


if __name__ == "__main__":
    main()
