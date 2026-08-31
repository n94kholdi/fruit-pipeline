import argparse
from pathlib import Path

import pytest

from fruit_pipeline.inference import confidence_threshold, find_images, rfdetr_result_to_records, select_backend


def test_confidence_threshold_accepts_bounds():
    assert confidence_threshold("0") == 0.0
    assert confidence_threshold("0.35") == 0.35
    assert confidence_threshold("1") == 1.0


@pytest.mark.parametrize("value", ["-0.1", "1.1"])
def test_confidence_threshold_rejects_out_of_range_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        confidence_threshold(value)


def test_find_images_is_non_recursive_and_sorted(tmp_path: Path):
    (tmp_path / "b.JPG").touch()
    (tmp_path / "a.png").touch()
    (tmp_path / "notes.txt").touch()
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "hidden.jpg").touch()

    assert [path.name for path in find_images(tmp_path)] == ["a.png", "b.JPG"]


@pytest.mark.parametrize(
    ("weights", "expected"),
    [
        ("models/yolo11x.pt", "ultralytics"),
        ("models/rf-detr-base.pth", "rfdetr"),
        ("runs/checkpoint.ckpt", "rfdetr"),
    ],
)
def test_select_backend_from_weights(weights: str, expected: str):
    assert select_backend("auto", weights) == expected


def test_select_backend_honors_explicit_backend():
    assert select_backend("ultralytics", "model.pth") == "ultralytics"


def test_rfdetr_result_to_records():
    import numpy as np

    class Detections:
        xyxy = np.array([[1.2345, 2, 30, 40]])
        confidence = np.array([0.87654321])
        class_id = np.array([1])

        def __len__(self):
            return 1

    assert rfdetr_result_to_records(Detections(), ["apple", "orange"]) == [
        {
            "box_xyxy": [1.234, 2.0, 30.0, 40.0],
            "confidence": 0.876543,
            "class_id": 1,
            "class_name": "orange",
        }
    ]
