import numpy as np

from fruit_pipeline.segmentation.sam import FruitInstance
from fruit_pipeline.segmentation.tracking import OpticalFlowMaskTracker, build_tracker


def _textured_frame(height: int = 120, width: int = 160) -> np.ndarray:
    """A frame with enough texture for goodFeaturesToTrack to find corners."""
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)
    return frame


def _square_instance(instance_id: int, x1: int, y1: int, size: int, shape: tuple[int, int]) -> FruitInstance:
    mask = np.zeros(shape, dtype=bool)
    mask[y1 : y1 + size, x1 : x1 + size] = True
    return FruitInstance(instance_id, [x1, y1, x1 + size, y1 + size], 0.9, "fruit", 0.8, mask)


def test_optical_flow_tracker_translates_mask_with_shifted_frame():
    shape = (120, 160)
    base = _textured_frame(*shape)
    instance = _square_instance(1, 40, 30, 30, shape)

    tracker = OpticalFlowMaskTracker()
    tracker.initialize(base, [instance])

    shift = 5
    shifted = np.zeros_like(base)
    shifted[:, shift:] = base[:, : shape[1] - shift]

    updated = tracker.update(shifted)

    assert len(updated) == 1
    tracked = updated[0]
    assert tracked.instance_id == 1
    # The mask's centroid should have moved roughly by the applied shift.
    ys, xs = np.where(tracked.mask)
    orig_ys, orig_xs = np.where(instance.mask)
    assert abs(float(xs.mean() - orig_xs.mean()) - shift) < 2.0
    assert abs(float(ys.mean() - orig_ys.mean())) < 2.0


def test_optical_flow_tracker_drops_instance_without_trackable_points():
    shape = (60, 60)
    base = np.zeros((*shape, 3), dtype=np.uint8)  # featureless: no corners to track
    instance = _square_instance(1, 10, 10, 10, shape)

    tracker = OpticalFlowMaskTracker()
    tracker.initialize(base, [instance])

    updated = tracker.update(base)

    assert updated == []


def test_build_tracker_rejects_unknown_type():
    import pytest

    with pytest.raises(ValueError, match="Unknown tracker_type"):
        build_tracker("nonexistent")


def test_build_tracker_returns_optical_flow_by_default():
    tracker = build_tracker("optical_flow")
    assert isinstance(tracker, OpticalFlowMaskTracker)
