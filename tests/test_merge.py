from sahi.prediction import ObjectPrediction

from fruit_pipeline.merge import filter_oversized_boxes, merge_detections, to_detections


def _pred(box, score=0.9, category_id=0, category_name="fruit"):
    return ObjectPrediction(bbox=box, category_id=category_id, category_name=category_name, score=score, shift_amount=[0, 0])


# --- seam-aware: union vs suppression gating -------------------------------


def test_seam_aware_unions_fragments_from_different_adjacent_tiles():
    # Tile 0 covers x[0,100], tile 1 covers x[90,200] -- they overlap in the
    # x[90,100] band (a realistic SAHI overlapping-tile pair). Both boxes
    # sit in/near that shared band and substantially overlap each other
    # (IoU ~0.75), the signature of "same fruit, split across the seam".
    box_a = [85, 40, 105, 60]  # from tile 0
    box_b = [87, 41, 107, 61]  # from tile 1
    predictions = [_pred(box_a, score=0.9), _pred(box_b, score=0.8)]
    tile_ids = [0, 1]
    tile_rects = {0: (0, 0, 100, 100), 1: (90, 0, 200, 100)}

    merged = merge_detections(
        predictions,
        strategy="seam-aware",
        match_threshold=0.5,
        tile_ids=tile_ids,
        tile_rects=tile_rects,
        seam_margin=5,
    )

    assert len(merged) == 1
    x1, y1, x2, y2 = merged[0].bbox.to_xyxy()
    # A proper union: strictly wider/taller than either source box alone.
    assert (x1, y1, x2, y2) == (85.0, 40.0, 107.0, 61.0)
    assert merged[0].score.value == 0.9


def test_seam_aware_suppresses_not_unions_overlap_within_same_tile():
    # Both detections came from the SAME tile -- even with real overlap,
    # this must never be treated as a seam split.
    box_a = [10, 10, 30, 30]
    box_b = [15, 12, 35, 32]
    predictions = [_pred(box_a, score=0.9), _pred(box_b, score=0.8)]
    tile_ids = [0, 0]
    tile_rects = {0: (0, 0, 100, 100)}

    merged = merge_detections(
        predictions,
        strategy="seam-aware",
        match_threshold=0.5,
        tile_ids=tile_ids,
        tile_rects=tile_rects,
        seam_margin=5,
    )

    assert len(merged) == 1
    # Suppression keeps the higher-scoring box UNCHANGED -- not a union.
    assert list(merged[0].bbox.to_xyxy()) == box_a


def test_seam_aware_suppresses_when_tiles_are_not_actually_adjacent():
    # Contrived: boxes overlap in raw coordinates, but their tiles' rects
    # don't overlap at all, so they can't be neighbors -- must never union.
    box_a = [10, 10, 30, 30]
    box_b = [15, 12, 35, 32]
    predictions = [_pred(box_a, score=0.9), _pred(box_b, score=0.8)]
    tile_ids = [0, 2]
    tile_rects = {0: (0, 0, 50, 50), 2: (200, 0, 250, 50)}

    merged = merge_detections(
        predictions,
        strategy="seam-aware",
        match_threshold=0.5,
        tile_ids=tile_ids,
        tile_rects=tile_rects,
        seam_margin=5,
    )

    assert len(merged) == 1
    assert list(merged[0].bbox.to_xyxy()) == box_a


# --- DIoU-NMS vs plain IoU-NMS ----------------------------------------------


def test_diou_nms_keeps_adjacent_distinct_center_boxes_that_iou_nms_drops():
    # Two boxes with real overlap (IoU ~0.524, above the 0.5 threshold) but
    # well-separated centers (offset by ~1/3 of box width). Hand-derived so
    # plain IoU is >= 0.5 (must-suppress) while DIoU's center-distance
    # penalty pulls it under 0.5 (must-not-suppress) -- exactly the "two
    # touching same-colored fruit" case DIoU-NMS is meant to fix.
    box_a = [0, 0, 32, 32]
    box_b = [10, 0, 42, 32]
    predictions = [_pred(box_a, score=0.9), _pred(box_b, score=0.85)]

    merged_iou = merge_detections(predictions, strategy="nms", match_threshold=0.5, nms_metric="iou", containment_threshold=0.0)
    merged_diou = merge_detections(predictions, strategy="nms", match_threshold=0.5, nms_metric="diou", containment_threshold=0.0)

    assert len(merged_iou) == 1  # plain IoU-NMS wrongly kills one
    assert len(merged_diou) == 2  # DIoU-NMS correctly keeps both


# --- containment suppression -------------------------------------------------


def test_containment_suppression_drops_lower_scoring_mostly_contained_box():
    # 'small' is 95% contained in 'big' (intersection / smaller-box-area =
    # 0.95); plain IoU between them is only ~0.19 (well under 0.5), so
    # containment suppression is the ONLY thing that can catch this pair.
    big = [0, 0, 100, 100]
    small = [5, 10, 105, 30]  # area 2000, 1900 of it inside `big`
    predictions = [_pred(big, score=0.9), _pred(small, score=0.6)]

    merged = merge_detections(predictions, strategy="nms", match_threshold=0.5, containment_threshold=0.9)

    assert len(merged) == 1
    assert list(merged[0].bbox.to_xyxy()) == big


def test_containment_suppression_disabled_above_actual_containment_ratio():
    # Same pair as above (containment=0.95), but the threshold is set
    # higher than the actual ratio -- containment suppression must not
    # fire, and since plain IoU between them is also low, both survive.
    big = [0, 0, 100, 100]
    small = [5, 10, 105, 30]
    predictions = [_pred(big, score=0.9), _pred(small, score=0.6)]

    merged = merge_detections(predictions, strategy="nms", match_threshold=0.5, containment_threshold=0.99)

    assert len(merged) == 2


def test_containment_suppression_is_pure_suppression_never_a_union():
    # Even under seam-aware with union allowed, a containment match must be
    # dropped outright, never unioned into a bigger box.
    big = [0, 0, 100, 100]
    small = [5, 10, 105, 30]
    predictions = [_pred(big, score=0.9), _pred(small, score=0.6)]

    merged = merge_detections(
        predictions,
        strategy="seam-aware",
        match_threshold=0.5,
        containment_threshold=0.9,
        tile_ids=[0, 1],
        tile_rects={0: (0, 0, 100, 100), 1: (0, 0, 100, 100)},
        seam_margin=0,
    )

    assert len(merged) == 1
    assert list(merged[0].bbox.to_xyxy()) == big  # unchanged, not widened


# --- legacy strategies still work (regression) ------------------------------


def test_legacy_nmm_strategy_still_unions_any_overlap_regardless_of_tiles():
    box_a = [0, 0, 20, 20]
    box_b = [5, 5, 25, 25]
    predictions = [_pred(box_a, score=0.9), _pred(box_b, score=0.8)]

    merged = merge_detections(predictions, strategy="nmm", match_metric="IOU", match_threshold=0.3)
    assert len(merged) == 1


def test_legacy_greedy_nmm_alias_still_works():
    box_a = [0, 0, 20, 20]
    box_b = [5, 5, 25, 25]
    predictions = [_pred(box_a, score=0.9), _pred(box_b, score=0.8)]

    merged = merge_detections(predictions, strategy="greedy_nmm", match_metric="IOU", match_threshold=0.3)
    assert len(merged) == 1


def test_unknown_strategy_raises():
    import pytest

    with pytest.raises(ValueError):
        merge_detections([_pred([0, 0, 10, 10])], strategy="bogus")


def test_merge_detections_empty_input_returns_empty():
    assert merge_detections([], strategy="seam-aware") == []


# --- filter_oversized_boxes / to_detections (unchanged behavior) -----------


def test_filter_oversized_boxes_drops_implausibly_large_box():
    normal1 = _pred([0, 0, 10, 10], score=0.9)
    normal2 = _pred([20, 20, 30, 30], score=0.9)
    huge = _pred([0, 0, 1000, 1000], score=0.9)

    kept = filter_oversized_boxes([normal1, normal2, huge], max_area_ratio=3.0)
    assert huge not in kept
    assert len(kept) == 2


def test_to_detections_orders_top_to_bottom_left_to_right():
    bottom = _pred([0, 100, 10, 110], score=0.9)
    top_right = _pred([50, 0, 60, 10], score=0.9)
    top_left = _pred([0, 0, 10, 10], score=0.9)

    detections = to_detections([bottom, top_right, top_left])
    assert [d.box[:2] for d in detections] == [[0.0, 0.0], [50.0, 0.0], [0.0, 100.0]]
    assert [d.instance_id for d in detections] == [0, 1, 2]
