"""Merge overlapping tiled detections into one clean set of per-fruit boxes.

Deliberately avoids plain hard-NMS by default: touching/adjacent fruit tends
to have real IoU with each other, and hard-NMS would delete legitimate
neighbors. Instead this wraps SAHI's Greedy NMM (non-maximum *merging*)
postprocessor, with plain NMS and full NMM available as configurable
alternatives.

Merging (not just suppressing) is a double-edged sword: SAHI's NMM/GreedyNMM
implementation combines a matched pair's boxes into their bounding-box
*union*, not just picking the higher-scored one. That is correct and wanted
for the common case (two tiles' overlapping partial views of the *same*
fruit near a tile seam), but on a densely packed crate of touching round
fruit, a metric that is too permissive can also "match" two genuinely
*different* neighboring fruits and union their boxes into one implausible
blob — which then either gets caught by ``filter_oversized_boxes`` (net
effect: both real fruits silently vanish from the final output) or, if the
union is a similar size to other fruit, prompts SAM with a box spanning two
objects, producing a bad mask likely to get dropped by the edge/aspect
filters in ``segment.py``. This was empirically observed and is why the
default below is ``IOU``, not ``IOS`` — see ``merge_detections``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sahi.postprocess.combine import GreedyNMMPostprocess, NMMPostprocess, NMSPostprocess
from sahi.prediction import ObjectPrediction

logger = logging.getLogger(__name__)

_STRATEGIES = {
    "greedy_nmm": GreedyNMMPostprocess,
    "nmm": NMMPostprocess,
    "nms": NMSPostprocess,
}


@dataclass
class Detection:
    """A single merged, pre-segmentation fruit candidate."""

    instance_id: int
    box: list[float]  # [x1, y1, x2, y2] in original image coordinates
    score: float
    category_name: str


def merge_detections(
    object_predictions: list[ObjectPrediction],
    strategy: str = "greedy_nmm",
    match_metric: str = "IOU",
    match_threshold: float = 0.5,
    class_agnostic: bool = True,
) -> list[ObjectPrediction]:
    """Deduplicate/merge raw per-tile detections into one set per real fruit.

    Args:
        object_predictions: Raw detections from ``detect.detect_tiled``,
            already shifted to full-image coordinates but not yet merged.
        strategy: One of "greedy_nmm" (default), "nmm", or "nms". "nms" is
            plain hard suppression and is only meant as an opt-in fallback.
        match_metric: "IOU" (default) or "IOS" (intersection-over-smaller-
            area). IOS was the original default on the theory that it's more
            forgiving when a smaller sliced-tile detection sits inside a
            larger one — but on a dense crate of touching round fruit, that
            same leniency also fires for two *different* adjacent fruit
            whenever one's (often loose/imprecise) box is mostly contained
            in its neighbor's, unioning them into one bad box (see module
            docstring). Measured on a real 20-tile crate photo: switching
            IOS -> IOU at the same threshold recovered ~50% more distinct
            fruit post-merge (33 -> 49) with no change in how many genuinely
            oversized/malformed boxes ``filter_oversized_boxes`` still had to
            drop. IOS remains available for images where under-merging
            (duplicate detections of the same fruit surviving as separate
            boxes) turns out to be the bigger problem.
        match_threshold: Overlap above which two detections are considered
            the same fruit.
        class_agnostic: Merge across categories. True is the right default
            here since every box is (or was relabeled to) "fruit".
    """
    if strategy not in _STRATEGIES:
        raise ValueError(f"Unknown merge strategy '{strategy}', expected one of {list(_STRATEGIES)}")
    if not object_predictions:
        return []

    postprocess = _STRATEGIES[strategy](
        match_threshold=match_threshold,
        match_metric=match_metric,
        class_agnostic=class_agnostic,
    )
    merged = postprocess(object_predictions)
    logger.info(
        "Merged %d raw detections -> %d fruit candidates (strategy=%s, %s>=%.2f)",
        len(object_predictions),
        len(merged),
        strategy,
        match_metric,
        match_threshold,
    )
    return merged


def filter_oversized_boxes(
    object_predictions: list[ObjectPrediction],
    max_area_ratio: float = 3.0,
    enabled: bool = True,
) -> list[ObjectPrediction]:
    """Drop boxes implausibly larger than the median fruit box in this image.

    Heuristic for "box drawn around the whole crate instead of one fruit":
    with dozens of similarly-sized fruit per image, the median box area is a
    reasonable stand-in for "one fruit," so anything far larger is suspect.
    """
    if not enabled or len(object_predictions) < 2:
        return object_predictions

    areas = np.array([_box_area(pred.bbox.to_xyxy()) for pred in object_predictions])
    median_area = float(np.median(areas))
    if median_area <= 0:
        return object_predictions

    keep = [pred for pred, area in zip(object_predictions, areas) if area <= max_area_ratio * median_area]
    dropped = len(object_predictions) - len(keep)
    if dropped:
        logger.info(
            "Oversized-box filter dropped %d/%d detections (> %.1fx median area %.0fpx^2)",
            dropped,
            len(object_predictions),
            max_area_ratio,
            median_area,
        )
    return keep


def to_detections(object_predictions: list[ObjectPrediction]) -> list[Detection]:
    """Assign stable instance ids and convert to the plain ``Detection`` type.

    Ids are assigned in reading order (top-to-bottom, then left-to-right) so
    they stay intuitive when cross-referenced against the visualization, and
    remain stable identifiers that later pipeline stages (classification,
    sizing, rotten/fine detection) can key their own per-fruit data on.
    """
    ordered = sorted(object_predictions, key=lambda pred: (pred.bbox.to_xyxy()[1], pred.bbox.to_xyxy()[0]))
    return [
        Detection(
            instance_id=idx,
            box=[float(v) for v in pred.bbox.to_xyxy()],
            score=float(pred.score.value),
            category_name=pred.category.name,
        )
        for idx, pred in enumerate(ordered)
    ]


def _box_area(box: list[float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)
