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


def test_static_masks_are_reused_until_elapsed_time_reaches_refresh_interval():
    config = VideoSegmentationConfig(
        tracking_enabled=False,
        static_mask_refresh_seconds=600,
    )
    manager = VideoSegmentationManager(config)
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    sam_calls = []

    def run_sam():
        instance = _instance(len(sam_calls) + 1)
        sam_calls.append(instance)
        return [instance]

    outputs = [
        manager.process(frame, run_sam=run_sam, timestamp_seconds=timestamp)
        for timestamp in (0, 120, 599, 600, 1199, 1200)
    ]

    assert [timings.used_sam for _, timings in outputs] == [
        True, False, False, True, False, True,
    ]
    assert [instances[0].instance_id for instances, _ in outputs] == [1, 1, 1, 2, 2, 3]
    assert len(sam_calls) == 3


def test_static_mask_mode_rejects_tracking():
    with pytest.raises(ValueError, match="cannot be enabled together"):
        VideoSegmentationConfig(
            tracking_enabled=True,
            static_mask_refresh_seconds=60,
        )


def test_refresh_interval_must_be_positive():
    with pytest.raises(ValueError, match="sam_refresh_interval"):
        VideoSegmentationConfig(sam_refresh_interval=0)

    with pytest.raises(ValueError, match="static_mask_refresh_seconds"):
        VideoSegmentationConfig(static_mask_refresh_seconds=0)


def test_unknown_tracker_type_rejected():
    with pytest.raises(ValueError, match="tracker_type"):
        VideoSegmentationConfig(tracker_type="does-not-exist")
