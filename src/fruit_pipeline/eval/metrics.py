"""Detection + counting metrics, broken out by object-area bucket.

Deliberately COCO-format-agnostic where possible: ``compute_precision_recall``
and ``compute_counting_metrics`` take plain ``{image_id: [box, ...]}``-shaped
dicts (see ``coco_io.extract_boxes``) so they're trivially unit-testable with
synthetic data, no COCO JSON or pycocotools involved. Only
``compute_map_metrics`` needs pycocotools, since mAP is inherently defined
via COCO's precision/recall-curve integration machinery.

Area buckets: COCO's own convention is small (<32^2), medium (32^2-96^2),
large (>96^2). A large fraction of the fruit in these images is smaller than
COCO's "small" bucket, so that bucket is split in two here: "tiny" (<16^2)
and "small" (16^2-32^2) — medium/large are unchanged COCO definitions. "all"
is the aggregate over every size.
"""

from __future__ import annotations

import contextlib
import io
import math

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from fruit_pipeline.utils.geometry import box_area, iou_xyxy

ALL_BUCKET = "all"

# Exhaustive, non-overlapping partition of [0, inf) area (pixels^2).
AREA_BUCKETS: dict[str, tuple[float, float]] = {
    "tiny": (0.0, 16.0**2),
    "small": (16.0**2, 32.0**2),
    "medium": (32.0**2, 96.0**2),
    "large": (96.0**2, float("inf")),
}

BUCKET_ORDER = ["tiny", "small", "medium", "large", ALL_BUCKET]


def _bucket_for_area(area: float, area_buckets: dict[str, tuple[float, float]]) -> str:
    for name, (lo, hi) in area_buckets.items():
        if lo <= area < hi:
            return name
    return next(reversed(area_buckets))  # area >= last bucket's (unbounded) hi shouldn't happen, but stay safe


def compute_map_metrics(
    coco_gt: COCO,
    coco_results: list[dict],
    area_buckets: dict[str, tuple[float, float]] = AREA_BUCKETS,
) -> dict[str, dict[str, float]]:
    """mAP50 / mAP50-95 per area bucket, via a separate ``COCOeval`` run per bucket.

    Runs ``COCOeval`` once per bucket with ``params.areaRng`` narrowed to
    that bucket's range, then extracts AP directly from
    ``coco_eval.eval["precision"]`` rather than calling ``summarize()`` —
    ``summarize()`` hardcodes lookups for COCO's own 'small'/'medium'/
    'large'/'all' labels, which doesn't generalize to our extra "tiny"
    bucket.
    """
    buckets = {**area_buckets, ALL_BUCKET: (0.0, float("inf"))}

    if not coco_results:
        return {name: {"mAP50": 0.0, "mAP50-95": 0.0} for name in buckets}

    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(coco_results)
    results: dict[str, dict[str, float]] = {}

    for name, (lo, hi) in buckets.items():
        with contextlib.redirect_stdout(io.StringIO()):
            coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
            coco_eval.params.imgIds = sorted(coco_gt.imgs.keys())
            coco_eval.params.areaRng = [[lo, hi]]
            coco_eval.params.areaRngLbl = [name]
            coco_eval.evaluate()
            coco_eval.accumulate()
        results[name] = {
            "mAP50": _extract_ap(coco_eval, iou_thr=0.5),
            "mAP50-95": _extract_ap(coco_eval, iou_thr=None),
        }
    return results


def _extract_ap(coco_eval: COCOeval, iou_thr: float | None) -> float:
    """Mean AP over recall thresholds, mirroring ``COCOeval.summarize()``'s ``_summarize`` internals.

    ``precision`` has shape ``(T, R, K, A, M)`` (IoU thresholds, recall
    thresholds, categories, area ranges, max-dets). We always set exactly
    one area range (index 0) and keep the default maxDets list, so we want
    the last (100) maxDets index.
    """
    params = coco_eval.params
    precision = coco_eval.eval["precision"]
    if precision.size == 0:
        return -1.0
    if iou_thr is not None:
        t_idx = np.where(np.isclose(params.iouThrs, iou_thr))[0]
        precision = precision[t_idx]
    precision = precision[:, :, :, 0, -1]
    valid = precision[precision > -1]
    return float(np.mean(valid)) if valid.size else -1.0


def compute_precision_recall(
    gt_by_image: dict[int, list[list[float]]],
    pred_by_image: dict[int, list[tuple[list[float], float]]],
    iou_thresh: float = 0.5,
    area_buckets: dict[str, tuple[float, float]] = AREA_BUCKETS,
) -> dict[str, dict[str, float | int | None]]:
    """Precision/recall per area bucket at a fixed IoU threshold.

    Greedy per-image matching: predictions are matched to unclaimed GT boxes
    in descending score order; a match requires IoU >= ``iou_thresh``. A
    matched pair is bucketed by the GT box's area (the canonical "object
    size"); an unmatched prediction (FP) is bucketed by its own box's area,
    since it has no GT to borrow a size from.

    ``precision``/``recall`` are ``None`` (not 0.0) when their denominator
    (tp+fp / tp+fn) is zero, so an empty bucket reads as "no data" rather
    than "0% correct".
    """
    buckets = list(area_buckets) + [ALL_BUCKET]
    counts = {b: {"tp": 0, "fp": 0, "fn": 0} for b in buckets}

    for image_id, gts in gt_by_image.items():
        preds = sorted(pred_by_image.get(image_id, []), key=lambda item: -item[1])
        matched = [False] * len(gts)

        for box, _score in preds:
            best_iou, best_idx = 0.0, -1
            for idx, gt_box in enumerate(gts):
                if matched[idx]:
                    continue
                iou = iou_xyxy(box, gt_box)
                if iou > best_iou:
                    best_iou, best_idx = iou, idx
            if best_iou >= iou_thresh and best_idx >= 0:
                matched[best_idx] = True
                bucket = _bucket_for_area(box_area(gts[best_idx]), area_buckets)
                counts[bucket]["tp"] += 1
                counts[ALL_BUCKET]["tp"] += 1
            else:
                bucket = _bucket_for_area(box_area(box), area_buckets)
                counts[bucket]["fp"] += 1
                counts[ALL_BUCKET]["fp"] += 1

        for idx, gt_box in enumerate(gts):
            if not matched[idx]:
                bucket = _bucket_for_area(box_area(gt_box), area_buckets)
                counts[bucket]["fn"] += 1
                counts[ALL_BUCKET]["fn"] += 1

    results: dict[str, dict[str, float | int | None]] = {}
    for bucket, c in counts.items():
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        results[bucket] = {"precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn}
    return results


def compute_counting_metrics(
    gt_by_image: dict[int, list[list[float]]],
    pred_by_image: dict[int, list[tuple[list[float], float]]],
    area_buckets: dict[str, tuple[float, float]] = AREA_BUCKETS,
) -> dict[str, dict[str, float | int | None]]:
    """Per-image instance-count MAE/RMSE per area bucket.

    Per image and bucket: ``gt_count`` = number of GT boxes whose own area
    falls in that bucket; ``pred_count`` = number of predicted boxes whose
    own area falls in that bucket (no GT/pred matching involved, just counts
    — this is deliberately a coarser, matching-free signal than
    precision/recall).
    """
    buckets = list(area_buckets) + [ALL_BUCKET]
    errors: dict[str, list[float]] = {b: [] for b in buckets}

    for image_id, gts in gt_by_image.items():
        preds = [box for box, _score in pred_by_image.get(image_id, [])]
        gt_counts = dict.fromkeys(buckets, 0)
        pred_counts = dict.fromkeys(buckets, 0)

        for gt_box in gts:
            bucket = _bucket_for_area(box_area(gt_box), area_buckets)
            gt_counts[bucket] += 1
            gt_counts[ALL_BUCKET] += 1
        for pred_box in preds:
            bucket = _bucket_for_area(box_area(pred_box), area_buckets)
            pred_counts[bucket] += 1
            pred_counts[ALL_BUCKET] += 1

        for b in buckets:
            errors[b].append(pred_counts[b] - gt_counts[b])

    results: dict[str, dict[str, float | int | None]] = {}
    for b in buckets:
        errs = errors[b]
        if not errs:
            results[b] = {"mae": None, "rmse": None, "n_images": 0}
            continue
        mae = sum(abs(e) for e in errs) / len(errs)
        rmse = math.sqrt(sum(e * e for e in errs) / len(errs))
        results[b] = {"mae": mae, "rmse": rmse, "n_images": len(errs)}
    return results


def evaluate_all(
    coco_gt: COCO,
    coco_results: list[dict],
    iou_thresh: float = 0.5,
    area_buckets: dict[str, tuple[float, float]] = AREA_BUCKETS,
) -> dict[str, dict]:
    """Run all three metric families and merge them into one per-bucket dict."""
    from fruit_pipeline.eval.coco_io import extract_boxes

    gt_by_image, pred_by_image = extract_boxes(coco_gt, coco_results)

    map_metrics = compute_map_metrics(coco_gt, coco_results, area_buckets)
    pr_metrics = compute_precision_recall(gt_by_image, pred_by_image, iou_thresh, area_buckets)
    counting_metrics = compute_counting_metrics(gt_by_image, pred_by_image, area_buckets)

    merged: dict[str, dict] = {}
    for bucket in list(area_buckets) + [ALL_BUCKET]:
        merged[bucket] = {**map_metrics[bucket], **pr_metrics[bucket], **counting_metrics[bucket]}
    return merged
