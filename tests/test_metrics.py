import pytest
from pycocotools.coco import COCO

from fruit_pipeline.eval.metrics import (
    AREA_BUCKETS,
    compute_counting_metrics,
    compute_map_metrics,
    compute_precision_recall,
)


def _make_coco(images: list[dict], annotations: list[dict], categories: list[dict] | None = None) -> COCO:
    """Build an in-memory pycocotools COCO object without touching disk."""
    coco = COCO()
    coco.dataset = {
        "images": images,
        "annotations": annotations,
        "categories": categories or [{"id": 1, "name": "fruit"}],
    }
    coco.createIndex()
    return coco


def _ann(ann_id, image_id, box_xyxy, category_id=1):
    x1, y1, x2, y2 = box_xyxy
    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": [x1, y1, x2 - x1, y2 - y1],
        "area": (x2 - x1) * (y2 - y1),
        "iscrowd": 0,
    }


def _result(image_id, box_xyxy, score=1.0, category_id=1):
    x1, y1, x2, y2 = box_xyxy
    return {"image_id": image_id, "category_id": category_id, "bbox": [x1, y1, x2 - x1, y2 - y1], "score": score}


# --- mAP -----------------------------------------------------------------


def test_map_perfect_match_is_one():
    # A 10x10 box (area=100) falls in the "tiny" bucket (< 16^2=256).
    images = [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}]
    gt = _make_coco(images, [_ann(1, 1, [10, 10, 20, 20])])
    results = [_result(1, [10, 10, 20, 20])]

    metrics = compute_map_metrics(gt, results)
    assert metrics["tiny"]["mAP50"] == pytest.approx(1.0)
    assert metrics["tiny"]["mAP50-95"] == pytest.approx(1.0)
    assert metrics["all"]["mAP50"] == pytest.approx(1.0)


def test_map_missed_detection_is_zero_not_negative_one():
    # GT exists in this bucket but nothing was predicted -> AP=0 (not "no data").
    images = [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}]
    gt = _make_coco(images, [_ann(1, 1, [10, 10, 20, 20])])

    metrics = compute_map_metrics(gt, [])
    assert metrics["tiny"]["mAP50"] == 0.0


def test_map_bucket_with_no_gt_is_negative_one_sentinel():
    # GT only has a tiny box; the "large" bucket has zero GT in it, so it's
    # undefined ("no data"), which pycocotools represents as -1, regardless
    # of any stray false positives predicted in that bucket.
    images = [{"id": 1, "file_name": "a.jpg", "width": 500, "height": 500}]
    gt = _make_coco(images, [_ann(1, 1, [10, 10, 20, 20])])
    stray_fp_in_large_bucket = _result(1, [100, 100, 300, 300], score=0.9)

    metrics = compute_map_metrics(gt, [stray_fp_in_large_bucket])
    assert metrics["large"]["mAP50"] == -1.0
    assert metrics["large"]["mAP50-95"] == -1.0


def test_map_empty_predictions_list_returns_zeros_without_crashing():
    images = [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}]
    gt = _make_coco(images, [_ann(1, 1, [10, 10, 20, 20])])
    metrics = compute_map_metrics(gt, [])
    assert set(metrics) == set(AREA_BUCKETS) | {"all"}


# --- precision / recall ---------------------------------------------------


def test_precision_recall_perfect_match():
    gt_by_image = {1: [[10, 10, 20, 20]]}
    pred_by_image = {1: [([10, 10, 20, 20], 0.9)]}

    result = compute_precision_recall(gt_by_image, pred_by_image)
    assert result["tiny"]["precision"] == 1.0
    assert result["tiny"]["recall"] == 1.0
    assert result["tiny"]["tp"] == 1
    assert result["all"]["precision"] == 1.0


def test_precision_recall_total_miss():
    gt_by_image = {1: [[10, 10, 20, 20]]}
    pred_by_image = {1: []}

    result = compute_precision_recall(gt_by_image, pred_by_image)
    assert result["tiny"]["recall"] == 0.0
    assert result["tiny"]["precision"] is None  # no predictions -> undefined, not 0
    assert result["tiny"]["fn"] == 1


def test_precision_recall_extra_false_positive():
    gt_by_image = {1: []}
    pred_by_image = {1: [([10, 10, 20, 20], 0.9)]}

    result = compute_precision_recall(gt_by_image, pred_by_image)
    assert result["tiny"]["precision"] == 0.0
    assert result["tiny"]["recall"] is None  # no GT -> undefined, not 0
    assert result["tiny"]["fp"] == 1


def test_precision_recall_low_iou_counts_as_both_fp_and_fn():
    # Predicted box barely overlaps the GT box (IoU well under 0.5).
    gt_by_image = {1: [[0, 0, 10, 10]]}
    pred_by_image = {1: [([8, 8, 18, 18], 0.9)]}

    result = compute_precision_recall(gt_by_image, pred_by_image, iou_thresh=0.5)
    assert result["tiny"]["tp"] == 0
    assert result["tiny"]["fp"] == 1
    assert result["tiny"]["fn"] == 1


# --- counting MAE/RMSE ----------------------------------------------------


def test_counting_metrics_perfect_count_is_zero_error():
    gt_by_image = {1: [[10, 10, 20, 20]], 2: [[10, 10, 20, 20], [30, 30, 40, 40]]}
    pred_by_image = {
        1: [([10, 10, 20, 20], 0.9)],
        2: [([10, 10, 20, 20], 0.9), ([30, 30, 40, 40], 0.9)],
    }

    result = compute_counting_metrics(gt_by_image, pred_by_image)
    assert result["tiny"]["mae"] == 0.0
    assert result["tiny"]["rmse"] == 0.0
    assert result["tiny"]["n_images"] == 2


def test_counting_metrics_known_mae_rmse():
    # Image 1: gt=1, pred=0 (error=-1). Image 2: gt=1, pred=3 (error=+2).
    gt_by_image = {1: [[10, 10, 20, 20]], 2: [[10, 10, 20, 20]]}
    pred_by_image = {
        1: [],
        2: [([10, 10, 20, 20], 0.9), ([11, 10, 21, 20], 0.9), ([12, 10, 22, 20], 0.9)],
    }

    result = compute_counting_metrics(gt_by_image, pred_by_image)
    # MAE = (|-1| + |2|) / 2 = 1.5 ; RMSE = sqrt((1 + 4) / 2) = sqrt(2.5)
    assert result["tiny"]["mae"] == 1.5
    assert result["tiny"]["rmse"] == pytest.approx(2.5**0.5)


def test_counting_metrics_bucket_with_no_boxes_either_side_is_zero_error():
    # Neither GT nor predictions have any "large" box in this image, so the
    # per-image count error for that bucket is a clean 0 (0 pred - 0 gt),
    # not "no data" -- counting metrics always have a defined count per
    # bucket per image, unlike precision/recall's tp/fp/fn ratios.
    gt_by_image = {1: [[10, 10, 20, 20]]}
    pred_by_image = {1: [([10, 10, 20, 20], 0.9)]}

    result = compute_counting_metrics(gt_by_image, pred_by_image)
    assert result["large"]["mae"] == 0.0
    assert result["large"]["n_images"] == 1
