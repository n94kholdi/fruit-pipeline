import numpy as np

from fruit_pipeline.geometry import (
    box_area,
    containment_ratio,
    diou_xyxy,
    expand_rect,
    iou_xyxy,
    pairwise_containment,
    pairwise_diou,
    pairwise_iou,
    rect_intersection,
    rects_overlap,
)


def test_box_area():
    assert box_area([0, 0, 10, 10]) == 100.0
    assert box_area([5, 5, 5, 5]) == 0.0  # degenerate
    assert box_area([10, 10, 5, 5]) == 0.0  # inverted, clamped to 0


def test_iou_identical_boxes_is_one():
    box = [10, 10, 30, 30]
    assert iou_xyxy(box, box) == 1.0


def test_iou_disjoint_boxes_is_zero():
    assert iou_xyxy([0, 0, 10, 10], [100, 100, 110, 110]) == 0.0


def test_iou_known_overlap():
    # Two 10x10 boxes overlapping in a 5x10 strip: intersection=50, union=150.
    a = [0, 0, 10, 10]
    b = [5, 0, 15, 10]
    assert iou_xyxy(a, b) == 50.0 / 150.0


def test_diou_equals_iou_for_concentric_boxes():
    # Same center -> center-distance penalty is 0, DIoU == IoU.
    a = [0, 0, 10, 10]
    b = [0, 0, 10, 10]
    assert diou_xyxy(a, b) == iou_xyxy(a, b)


def test_diou_penalizes_separated_centers_relative_to_iou():
    # Two adjacent (touching, non-concentric) boxes with real IoU > 0: DIoU
    # must be strictly less than IoU once centers are apart.
    a = [0, 0, 20, 20]
    b = [15, 0, 35, 20]  # shifted right, still overlapping
    iou = iou_xyxy(a, b)
    diou = diou_xyxy(a, b)
    assert iou > 0
    assert diou < iou


def test_containment_full_containment_is_one():
    outer = [0, 0, 100, 100]
    inner = [10, 10, 30, 30]
    assert containment_ratio(inner, outer) == 1.0
    assert containment_ratio(outer, inner) == 1.0  # symmetric by construction


def test_containment_partial_overlap_uses_smaller_box_denominator():
    small = [0, 0, 10, 10]  # area 100
    big = [5, 5, 25, 25]  # area 400, intersection with small = 5x5=25
    expected = 25.0 / 100.0  # intersection / smaller box's area
    assert containment_ratio(small, big) == expected


def test_pairwise_iou_matches_scalar_iou():
    boxes = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [100, 100, 110, 110]], dtype=float)
    matrix = pairwise_iou(boxes)
    assert matrix.shape == (3, 3)
    assert matrix[0, 0] == 1.0
    assert matrix[0, 1] == iou_xyxy(list(boxes[0]), list(boxes[1]))
    assert matrix[0, 2] == 0.0


def test_pairwise_diou_matches_scalar_diou():
    boxes = np.array([[0, 0, 20, 20], [15, 0, 35, 20]], dtype=float)
    matrix = pairwise_diou(boxes)
    assert matrix[0, 1] == diou_xyxy(list(boxes[0]), list(boxes[1]))


def test_pairwise_containment_matches_scalar():
    boxes = np.array([[0, 0, 100, 100], [10, 10, 30, 30]], dtype=float)
    matrix = pairwise_containment(boxes)
    assert matrix[0, 1] == 1.0
    assert matrix[1, 0] == 1.0


def test_rect_intersection_overlapping():
    assert rect_intersection((0, 0, 10, 10), (5, 5, 15, 15)) == (5, 5, 10, 10)


def test_rect_intersection_disjoint_is_none():
    assert rect_intersection((0, 0, 10, 10), (20, 20, 30, 30)) is None


def test_rect_intersection_touching_edges_is_none():
    # Zero-area shared edge doesn't count as an overlap.
    assert rect_intersection((0, 0, 10, 10), (10, 0, 20, 10)) is None


def test_expand_rect_grows_on_all_sides():
    assert expand_rect((10, 10, 20, 20), 5) == (5, 5, 25, 25)


def test_rects_overlap_true_and_false():
    assert rects_overlap((0, 0, 10, 10), (5, 5, 15, 15)) is True
    assert rects_overlap((0, 0, 10, 10), (100, 100, 110, 110)) is False
