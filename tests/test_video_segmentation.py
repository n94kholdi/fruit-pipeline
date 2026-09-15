import numpy as np
import pytest

from fruit_pipeline.segmentation.sam import FruitInstance
from fruit_pipeline.segmentation.video_segmentation import (
    VideoSegmentationConfig,
    VideoSegmentationManager,
)


def _instance(instance_id: int, shape: tuple[int, int] = (40, 40)) -> FruitInstance:
    mask = np.zeros(shape, dtype=bool)
    mask[5:20, 5:20] = True
    return FruitInstance(instance_id, [5, 5, 20, 20], 0.9, "fruit", 0.8, mask)


def test_tracking_disabled_calls_sam_every_frame():
    config = VideoSegmentationConfig(tracking_enabled=False, sam_refresh_interval=10)
    manager = VideoSegmentationManager(config)
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    calls = []

    def run_sam():
        calls.append(1)
        return [_instance(1)]

    for _ in range(5):
        instances, timings = manager.process(frame, run_sam=run_sam)
        assert timings.used_sam is True
        assert len(instances) == 1

    assert len(calls) == 5


def test_tracking_enabled_refreshes_only_at_interval(monkeypatch):
    config = VideoSegmentationConfig(tracking_enabled=True, sam_refresh_interval=3)
    manager = VideoSegmentationManager(config)
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    sam_calls = []

    def run_sam():
        sam_calls.append(1)
        return [_instance(1)]

    # Stub the tracker to avoid depending on optical-flow specifics here;
    # this test is about the refresh-interval decision, not flow quality.
    used_sam_flags = []
    for i in range(7):
        instances, timings = manager.process(frame, run_sam=run_sam)
        used_sam_flags.append(timings.used_sam)

    # Frame 0 (first frame) and frame 3, 6 (every 3rd) should call SAM.
    assert used_sam_flags == [True, False, False, True, False, False, True]
    assert len(sam_calls) == 3


def test_refresh_interval_must_be_positive():
    with pytest.raises(ValueError, match="sam_refresh_interval"):
        VideoSegmentationConfig(sam_refresh_interval=0)


def test_unknown_tracker_type_rejected():
    with pytest.raises(ValueError, match="tracker_type"):
        VideoSegmentationConfig(tracker_type="does-not-exist")
