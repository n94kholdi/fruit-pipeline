"""Unit tests for detectors.py's pure logic (no real model checkpoint / network needed).

A real YOLOE checkpoint smoke test is documented in the README as a manual
follow-up (network download of a yoloe*.pt checkpoint) rather than run here.
"""

import numpy as np
import pytest

from fruit_pipeline.detectors import RfdetrBackend, YoloeBackend, load_detector_backend


class _FakeModel:
    def __init__(self):
        self.classes_set = None

    def set_classes(self, classes):
        self.classes_set = list(classes)


class _FakeBoxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = _FakeTensor(xyxy)
        self.conf = _FakeTensor(conf)
        self.cls = _FakeTensor(cls)

    def __len__(self):
        return len(self.xyxy.data)


class _FakeTensor:
    """Minimal stand-in for a torch tensor exposing .detach().cpu().numpy()."""

    def __init__(self, data):
        self.data = np.array(data, dtype=float)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.data


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


def test_yoloe_text_mode_requires_prompts():
    with pytest.raises(ValueError):
        YoloeBackend(model=_FakeModel(), mode="text", text_prompts=None)


def test_yoloe_visual_mode_requires_exemplar_paths():
    with pytest.raises(ValueError):
        YoloeBackend(model=_FakeModel(), mode="visual", visual_prompt_paths=None)


def test_yoloe_text_mode_calls_set_classes_with_combined_prompts():
    model = _FakeModel()
    backend = YoloeBackend(model=model, mode="text", text_prompts=["a round orange", "background"], num_fruit_classes=1)
    assert model.classes_set == ["a round orange", "background"]
    assert backend.num_fruit_classes == 1


def test_yoloe_prompt_free_mode_never_calls_set_classes():
    model = _FakeModel()
    YoloeBackend(model=model, mode="prompt_free")
    assert model.classes_set is None


def test_yoloe_unknown_mode_raises():
    with pytest.raises(ValueError):
        YoloeBackend(model=_FakeModel(), mode="bogus")


def test_results_to_predictions_filters_background_classes_in_text_mode():
    model = _FakeModel()
    backend = YoloeBackend(model=model, mode="text", text_prompts=["fruit prompt", "background prompt"], num_fruit_classes=1)

    results = [
        _FakeResult(
            _FakeBoxes(
                xyxy=[[0, 0, 10, 10], [20, 20, 30, 30]],
                conf=[0.9, 0.8],
                cls=[0, 1],  # class 0 = fruit, class 1 = background
            )
        )
    ]

    predictions = backend._results_to_predictions(results, shift_x=100, shift_y=200, filter_background=True)
    assert len(predictions) == 1
    assert predictions[0].category.name == "fruit"
    x1, y1, x2, y2 = predictions[0].bbox.to_xyxy()
    assert (x1, y1, x2, y2) == (100.0, 200.0, 110.0, 210.0)  # shifted into full-image coords


def test_results_to_predictions_visual_mode_keeps_everything():
    model = _FakeModel()
    backend = YoloeBackend(model=model, mode="prompt_free")
    results = [_FakeResult(_FakeBoxes(xyxy=[[0, 0, 5, 5]], conf=[0.5], cls=[0]))]

    predictions = backend._results_to_predictions(results, shift_x=0, shift_y=0, filter_background=False)
    assert len(predictions) == 1


def test_load_detector_backend_unknown_name_raises():
    with pytest.raises(ValueError):
        load_detector_backend("bogus", "weights.pt")


class _FakeRfdetrDetections:
    xyxy = np.array([[1, 2, 11, 22], [-2, -3, 500, 600], [4, 4, 4, 8]], dtype=float)
    confidence = np.array([0.9, 0.8, 0.7], dtype=float)
    class_id = np.array([1, 2, 3], dtype=int)

    def __len__(self):
        return len(self.xyxy)


class _FakeRfdetrModel:
    class_names = {1: "apple", 2: "orange", 3: "banana"}

    def __init__(self):
        self.calls = []

    def predict(self, image, threshold):
        self.calls.append((image, threshold))
        return _FakeRfdetrDetections()


def test_rfdetr_backend_converts_clamps_and_shifts_predictions():
    model = _FakeRfdetrModel()
    backend = RfdetrBackend(model)
    tile = np.zeros((100, 200, 3), dtype=np.uint8)

    predictions = backend.detect(tile, shift=(10, 20), full_shape=(150, 180), conf_threshold=0.35)

    assert model.calls == [(tile, 0.35)]
    assert len(predictions) == 2  # Degenerate third box is dropped.
    assert predictions[0].bbox.to_xyxy() == [11.0, 22.0, 21.0, 42.0]
    assert predictions[0].category.id == 1
    assert predictions[0].category.name == "apple"
    assert predictions[1].bbox.to_xyxy() == [10.0, 20.0, 180.0, 120.0]
