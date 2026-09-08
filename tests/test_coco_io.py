import json

from pycocotools.coco import COCO

from fruit_pipeline.eval.coco_io import extract_boxes, load_predictions_dir


def _make_coco(images, annotations, categories=None):
    coco = COCO()
    coco.dataset = {
        "images": images,
        "annotations": annotations,
        "categories": categories or [{"id": 1, "name": "fruit"}],
    }
    coco.createIndex()
    return coco


def test_extract_boxes_converts_xywh_gt_to_xyxy():
    images = [{"id": 5, "file_name": "img.jpg", "width": 100, "height": 100}]
    annotations = [{"id": 1, "image_id": 5, "category_id": 1, "bbox": [10, 20, 30, 40], "area": 1200, "iscrowd": 0}]
    coco_gt = _make_coco(images, annotations)

    gt_by_image, pred_by_image = extract_boxes(coco_gt, coco_results=[])
    assert gt_by_image[5] == [[10, 20, 40, 60]]
    assert pred_by_image[5] == []


def test_extract_boxes_includes_images_with_no_annotations():
    images = [{"id": 1, "file_name": "a.jpg", "width": 10, "height": 10}, {"id": 2, "file_name": "b.jpg", "width": 10, "height": 10}]
    coco_gt = _make_coco(images, annotations=[])

    gt_by_image, _ = extract_boxes(coco_gt, coco_results=[])
    assert gt_by_image == {1: [], 2: []}


def test_extract_boxes_converts_prediction_results_to_xyxy_with_score():
    images = [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}]
    coco_gt = _make_coco(images, annotations=[])
    coco_results = [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.75}]

    _, pred_by_image = extract_boxes(coco_gt, coco_results)
    assert pred_by_image[1] == [([0, 0, 10, 10], 0.75)]


def test_load_predictions_dir_matches_by_filename_stem(tmp_path):
    images = [{"id": 1, "file_name": "sample.jpg", "width": 200, "height": 200}]
    coco_gt = _make_coco(images, annotations=[])

    detections_payload = {
        "image_path": "sample.jpg",
        "detections": [
            {"instance_id": 0, "box": [1.0, 2.0, 11.0, 12.0], "detector_score": 0.8, "category_name": "fruit"},
        ],
    }
    (tmp_path / "sample_detections.json").write_text(json.dumps(detections_payload))

    coco_results, unmatched = load_predictions_dir(coco_gt, str(tmp_path))
    assert unmatched == []
    assert len(coco_results) == 1
    assert coco_results[0]["image_id"] == 1
    assert coco_results[0]["category_id"] == 1
    assert coco_results[0]["bbox"] == [1.0, 2.0, 10.0, 10.0]
    assert coco_results[0]["score"] == 0.8


def test_load_predictions_dir_reports_unmatched_stems(tmp_path):
    images = [{"id": 1, "file_name": "sample.jpg", "width": 200, "height": 200}]
    coco_gt = _make_coco(images, annotations=[])

    other_payload = {"image_path": "other.jpg", "detections": []}
    (tmp_path / "other_detections.json").write_text(json.dumps(other_payload))

    coco_results, unmatched = load_predictions_dir(coco_gt, str(tmp_path))
    assert coco_results == []
    assert unmatched == ["other"]
