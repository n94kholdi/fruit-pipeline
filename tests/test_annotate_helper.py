import importlib.util
import json
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "annotate_helper.py"
_spec = importlib.util.spec_from_file_location("annotate_helper", SCRIPT_PATH)
annotate_helper = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(annotate_helper)


def _write_detections(tmp_path, stem, image_path, width, height, detections):
    payload = {
        "image_path": image_path,
        "image_width": width,
        "image_height": height,
        "detections": detections,
    }
    (tmp_path / f"{stem}_detections.json").write_text(json.dumps(payload))


def test_build_coco_pre_annotations_shapes_boxes_and_categories(tmp_path):
    _write_detections(
        tmp_path,
        "img1",
        "data/img1.jpg",
        200,
        100,
        [{"instance_id": 0, "box": [10.0, 10.0, 30.0, 40.0], "detector_score": 0.9, "category_name": "fruit"}],
    )

    coco = annotate_helper.build_coco_pre_annotations(str(tmp_path))
    assert len(coco["images"]) == 1
    assert coco["images"][0] == {"id": 1, "file_name": "img1.jpg", "width": 200, "height": 100}
    assert coco["categories"] == [{"id": 1, "name": "fruit"}]
    assert len(coco["annotations"]) == 1
    ann = coco["annotations"][0]
    assert ann["image_id"] == 1
    assert ann["bbox"] == [10.0, 10.0, 20.0, 30.0]
    assert ann["area"] == 20.0 * 30.0
    assert ann["score"] == 0.9


def test_build_coco_pre_annotations_multiple_images_get_distinct_ids(tmp_path):
    _write_detections(tmp_path, "a", "a.jpg", 100, 100, [])
    _write_detections(tmp_path, "b", "b.jpg", 100, 100, [])

    coco = annotate_helper.build_coco_pre_annotations(str(tmp_path))
    ids = sorted(img["id"] for img in coco["images"])
    assert ids == [1, 2]


def test_build_label_studio_tasks_uses_percent_coordinates(tmp_path):
    _write_detections(
        tmp_path,
        "img1",
        "img1.jpg",
        200,
        100,
        [{"instance_id": 0, "box": [20.0, 10.0, 60.0, 60.0], "detector_score": 0.8, "category_name": "fruit"}],
    )

    tasks = annotate_helper.build_label_studio_tasks(str(tmp_path), image_url_prefix="prefix/")
    assert len(tasks) == 1
    task = tasks[0]
    assert task["data"]["image"] == "prefix/img1.jpg"
    result = task["predictions"][0]["result"][0]
    assert result["value"]["x"] == 100.0 * 20.0 / 200.0
    assert result["value"]["y"] == 100.0 * 10.0 / 100.0
    assert result["value"]["width"] == 100.0 * 40.0 / 200.0
    assert result["value"]["height"] == 100.0 * 50.0 / 100.0
    assert result["value"]["rectanglelabels"] == ["fruit"]


def test_build_coco_pre_annotations_empty_dir_returns_empty_lists(tmp_path):
    coco = annotate_helper.build_coco_pre_annotations(str(tmp_path))
    assert coco["images"] == []
    assert coco["annotations"] == []
