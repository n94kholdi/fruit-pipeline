"""Shared box-geometry helpers used by both ``merge.py`` and ``eval/metrics.py``.

Boxes are always ``[x1, y1, x2, y2]`` in the same coordinate space (pixels),
as plain sequences of 4 floats unless a function explicitly takes a numpy
array (documented per-function).
"""

from __future__ import annotations

import numpy as np


def box_area(box: list[float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou_xyxy(box_a: list[float], box_b: list[float]) -> float:
    """Plain IoU between two ``[x1, y1, x2, y2]`` boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    union = box_area(box_a) + box_area(box_b) - inter
    return inter / union if union > 0 else 0.0


def diou_xyxy(box_a: list[float], box_b: list[float]) -> float:
    """Distance-IoU: plain IoU minus a normalized center-distance penalty.

    ``DIoU = IoU - center_distance^2 / diagonal_of_smallest_enclosing_box^2``.
    Two boxes with high IoU but well-separated centers (e.g. two touching,
    same-sized adjacent fruit) score lower under DIoU than under plain IoU,
    while two boxes that are near-duplicates of the same object (nearly
    identical centers) are barely affected. See the Cluster-DIoU-NMS idea in
    ASAHI (arXiv 2604.19233).
    """
    iou = iou_xyxy(box_a, box_b)
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    a_cx, a_cy = (ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0
    b_cx, b_cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
    center_dist_sq = (a_cx - b_cx) ** 2 + (a_cy - b_cy) ** 2

    ex1, ey1 = min(ax1, bx1), min(ay1, by1)
    ex2, ey2 = max(ax2, bx2), max(ay2, by2)
    diag_sq = (ex2 - ex1) ** 2 + (ey2 - ey1) ** 2
    if diag_sq <= 0:
        return iou
    return iou - center_dist_sq / diag_sq


def containment_ratio(box_a: list[float], box_b: list[float]) -> float:
    """Intersection over the area of the SMALLER of the two boxes.

    Close to 1.0 when one box is almost entirely inside the other,
    regardless of how large the other box is (unlike IoU, which is diluted
    by the bigger box's area). This is the Mask-NMS rule from SDM-D
    (arXiv 2411.16196) applied to boxes instead of masks.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    smaller_area = min(box_area(box_a), box_area(box_b))
    return inter / smaller_area if smaller_area > 0 else 0.0


def rect_intersection(rect_a: tuple[float, float, float, float], rect_b: tuple[float, float, float, float]) -> tuple[float, float, float, float] | None:
    """Intersection of two ``(x1, y1, x2, y2)`` rects, or ``None`` if they don't overlap."""
    ax1, ay1, ax2, ay2 = rect_a
    bx1, by1, bx2, by2 = rect_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    return (ix1, iy1, ix2, iy2)


def expand_rect(rect: tuple[float, float, float, float], margin: float) -> tuple[float, float, float, float]:
    """Grow a ``(x1, y1, x2, y2)`` rect outward by ``margin`` px on every side."""
    x1, y1, x2, y2 = rect
    return (x1 - margin, y1 - margin, x2 + margin, y2 + margin)


def rects_overlap(rect_a: tuple[float, float, float, float], rect_b: tuple[float, float, float, float]) -> bool:
    """True if two ``(x1, y1, x2, y2)`` rects have any positive-area overlap."""
    return rect_intersection(rect_a, rect_b) is not None


def pairwise_iou(boxes: np.ndarray) -> np.ndarray:
    """Vectorized IoU matrix for an ``(N, 4)`` array of ``[x1, y1, x2, y2]`` boxes."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

    ix1 = np.maximum(x1[:, None], x1[None, :])
    iy1 = np.maximum(y1[:, None], y1[None, :])
    ix2 = np.minimum(x2[:, None], x2[None, :])
    iy2 = np.minimum(y2[:, None], y2[None, :])
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)

    union = areas[:, None] + areas[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    return iou


def pairwise_diou(boxes: np.ndarray) -> np.ndarray:
    """Vectorized DIoU matrix for an ``(N, 4)`` array of boxes. See ``diou_xyxy``."""
    iou = pairwise_iou(boxes)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

    center_dist_sq = (cx[:, None] - cx[None, :]) ** 2 + (cy[:, None] - cy[None, :]) ** 2

    ex1 = np.minimum(x1[:, None], x1[None, :])
    ey1 = np.minimum(y1[:, None], y1[None, :])
    ex2 = np.maximum(x2[:, None], x2[None, :])
    ey2 = np.maximum(y2[:, None], y2[None, :])
    diag_sq = (ex2 - ex1) ** 2 + (ey2 - ey1) ** 2

    with np.errstate(divide="ignore", invalid="ignore"):
        penalty = np.where(diag_sq > 0, center_dist_sq / diag_sq, 0.0)
    return iou - penalty


def pairwise_containment(boxes: np.ndarray) -> np.ndarray:
    """Vectorized containment-ratio matrix (intersection / smaller-box-area). See ``containment_ratio``."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

    ix1 = np.maximum(x1[:, None], x1[None, :])
    iy1 = np.maximum(y1[:, None], y1[None, :])
    ix2 = np.minimum(x2[:, None], x2[None, :])
    iy2 = np.minimum(y2[:, None], y2[None, :])
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)

    smaller = np.minimum(areas[:, None], areas[None, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        containment = np.where(smaller > 0, inter / smaller, 0.0)
    return containment
