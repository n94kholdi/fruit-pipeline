"""Bridge fruit-pipeline's JSON formats to and from COCO.

``metrics.py`` deliberately knows nothing about COCO or pycocotools — it
only takes plain ``{image_id: [[x1,y1,x2,y2], ...]}``-shaped dicts, so it's
trivially unit-testable with synthetic data. This module is where the COCO
<-> plain-box-dict conversion happens, and where our own pipeline's
``<stem>_detections.json`` output gets turned into COCO detection results.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pycocotools.coco import COCO

logger = logging.getLogger(__name__)


def load_coco_gt(path: str) -> COCO:
    """Load a ground-truth COCO JSON (images + annotations + categories)."""
    return COCO(path)


def _category_id_for_name(coco_gt: COCO, name: str, default_id: int = 1) -> int:
    """Map a detection's free-form ``category_name`` (e.g. "fruit") to a GT category id.

    Falls back to ``default_id`` (COCO convention: first real category id is
    1) if no exact (case-insensitive) name match is found — our pipeline
    only ever produces one category ("fruit"), and GT sets built via
    ``scripts/annotate_helper.py`` use the same single category, so this is
    expected to always hit the exact-match path in practice.
    """
    for cat_id, cat in coco_gt.cats.items():
        if cat["name"].strip().lower() == name.strip().lower():
            return cat_id
    logger.warning("Category name %r not found in GT categories %s; defaulting to id=%d", name, [c["name"] for c in coco_gt.cats.values()], default_id)
    return default_id


def _stem_to_gt_image(coco_gt: COCO) -> dict[str, dict]:
    """Map each GT image's filename stem (no directory, no extension) to its COCO image record."""
    by_stem: dict[str, dict] = {}
    for img in coco_gt.imgs.values():
        stem = Path(img["file_name"]).stem
        by_stem[stem] = img
    return by_stem


def load_predictions_dir(coco_gt: COCO, pred_dir: str) -> tuple[list[dict], list[str]]:
    """Load every ``*_detections.json`` in ``pred_dir`` and convert to COCO detection results.

    Matches each prediction file to a GT image by filename stem (the same
    stem convention ``pipeline.py`` already uses for its output filenames:
    ``<stem>_detections.json`` from an input image ``<stem>.<ext>``).

    Returns ``(coco_results, unmatched_stems)`` — ``coco_results`` is a list
    of ``{"image_id", "category_id", "bbox": [x,y,w,h], "score"}`` dicts
    ready for ``coco_gt.loadRes(...)``; ``unmatched_stems`` lists prediction
    files that had no corresponding GT image (logged, not silently dropped).
    """
    stem_to_image = _stem_to_gt_image(coco_gt)
    results: list[dict] = []
    unmatched: list[str] = []

    for pred_path in sorted(Path(pred_dir).glob("*_detections.json")):
        stem = pred_path.name[: -len("_detections.json")]
        gt_image = stem_to_image.get(stem)
        if gt_image is None:
            unmatched.append(stem)
            continue

        with open(pred_path) as f:
            payload = json.load(f)

        for det in payload.get("detections", []):
            x1, y1, x2, y2 = det["box"]
            category_id = _category_id_for_name(coco_gt, det.get("category_name", "fruit"))
            results.append(
                {
                    "image_id": gt_image["id"],
                    "category_id": category_id,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(det.get("detector_score", 1.0)),
                }
            )

    if unmatched:
        logger.warning("%d prediction file(s) had no matching GT image (by filename stem): %s", len(unmatched), unmatched)
    return results, unmatched


def extract_boxes(
    coco_gt: COCO, coco_results: list[dict]
) -> tuple[dict[int, list[list[float]]], dict[int, list[tuple[list[float], float]]]]:
    """Convert COCO GT annotations + COCO-format results into plain xyxy box dicts.

    Returns ``(gt_by_image, pred_by_image)``:
    - ``gt_by_image[image_id] -> [[x1,y1,x2,y2], ...]``
    - ``pred_by_image[image_id] -> [([x1,y1,x2,y2], score), ...]``

    Every GT image id is present in ``gt_by_image`` (empty list if that
    image has no annotations), so images with zero fruit still count
    correctly toward precision/recall/counting metrics.
    """
    gt_by_image: dict[int, list[list[float]]] = {img_id: [] for img_id in coco_gt.imgs}
    for ann in coco_gt.anns.values():
        x, y, w, h = ann["bbox"]
        gt_by_image[ann["image_id"]].append([x, y, x + w, y + h])

    pred_by_image: dict[int, list[tuple[list[float], float]]] = {img_id: [] for img_id in coco_gt.imgs}
    for res in coco_results:
        x, y, w, h = res["bbox"]
        pred_by_image.setdefault(res["image_id"], []).append(([x, y, x + w, y + h], float(res["score"])))

    return gt_by_image, pred_by_image
