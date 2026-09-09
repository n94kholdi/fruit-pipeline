from __future__ import annotations

import numpy as np
import pytest

from fruit_pipeline.segmentation.sam import FruitInstance
from fruit_pipeline.segmentation.sam2_config import SAM2Config, SAM2_MODEL_NAMES
from fruit_pipeline.segmentation.sam2_manager import SAM2ModelManager
from fruit_pipeline.segmentation.sam2_tracking import (
    BoundedRefreshQueue,
    InstanceReconciler,
    stable_refresh_phase,
)


def _instance(instance_id, x1, y1, x2, y2, shape=(50, 60)):
    mask = np.zeros(shape, dtype=bool)
    mask[y1:y2, x1:x2] = True
    return FruitInstance(instance_id, [x1, y1, x2, y2], 1.0, "fruit", 0.9, mask)


def test_all_sam21_variants_are_available_and_base_plus_is_default():
    assert SAM2_MODEL_NAMES == (
        "sam2.1_hiera_tiny",
        "sam2.1_hiera_small",
        "sam2.1_hiera_base_plus",
        "sam2.1_hiera_large",
    )
    config = SAM2Config(device="cpu")
    assert config.model_name == "sam2.1_hiera_base_plus"
    assert config.resolved_checkpoint.endswith("sam2.1_hiera_base_plus.pt")
    assert config.resolved_config.endswith("sam2.1_hiera_b+.yaml")


def test_unknown_variant_is_rejected():
    with pytest.raises(ValueError, match="Unknown SAM2 model"):
        SAM2Config(model_name="everything", device="cpu")


def test_reconciliation_preserves_ids_and_graces_missing_objects():
    reconciler = InstanceReconciler(missing_grace_refreshes=1)
    first = reconciler.reconcile([], [_instance(0, 5, 5, 15, 15)], 0)
    fruit_id = first.instances[0].instance_id
    moved = reconciler.reconcile(first.instances, [_instance(0, 6, 5, 16, 15)], 10)
    assert moved.instances[0].instance_id == fruit_id
    assert [event.event for event in moved.events] == ["matched"]

    missing = reconciler.reconcile(moved.instances, [], 20)
    assert missing.instances[0].tracking_state == "uncertain"
    removed = reconciler.reconcile(missing.instances, [], 30)
    assert removed.instances == []
    assert removed.events[0].event == "removed"


def test_refresh_phase_is_deterministic_and_spans_interval():
    values = [stable_refresh_phase(f"camera-{index}", 10) for index in range(20)]
    assert values == [stable_refresh_phase(f"camera-{index}", 10) for index in range(20)]
    assert all(0 <= value < 10 for value in values)
    assert max(values) - min(values) > 5


def test_refresh_queue_is_bounded_prioritized_and_coalesced():
    queue = BoundedRefreshQueue(2)
    assert queue.submit("routine", "scheduled", now=10)
    assert not queue.submit("routine", "scene_change", now=11)
    assert queue.submit("urgent", "tracking_failure", now=12)
    assert not queue.submit("overflow", "initial", now=13)
    assert queue.pop().camera_id == "urgent"
    assert queue.metrics(now=20) == {
        "depth": 1,
        "oldest_request_age_seconds": 10,
        "accepted": 2,
        "delayed": 1,
        "dropped": 1,
    }


class _FakeGenerator:
    def __init__(self):
        self.calls = 0

    def generate(self, image):
        self.calls += 1
        mask = np.zeros(image.shape[:2], dtype=bool)
        mask[5:15, 6:16] = True
        return [{"segmentation": mask, "predicted_iou": 0.9}]


class _FakePredictor:
    def __init__(self):
        self.prompts = []
        self.propagations = 0
        self.reset = False

    def init_state(self, video_path):
        return {"source": video_path}

    def add_new_mask(self, state, frame_idx, obj_id, mask):
        self.prompts.append((frame_idx, obj_id, mask.copy()))

    def propagate_frame(self, state, frame_index):
        self.propagations += 1
        mask = np.full((1, 1, 50, 60), -1.0, dtype=np.float32)
        mask[:, :, 16:26, 26:36] = 1.0
        return [1], mask

    def reset_state(self, state):
        self.reset = True


def test_first_frame_discovers_then_intermediate_frame_propagates_with_roi_coordinates():
    predictor, generator = _FakePredictor(), _FakeGenerator()
    config = SAM2Config(
        device="cpu", refresh_seconds=999, refresh_processed_frames=30,
        min_mask_region_area=1,
    )
    manager = SAM2ModelManager(config, predictor=predictor, generator=generator)
    manager.start_camera("cam", "video.mp4", (50, 60), (20, 10, 50, 40))
    image = np.zeros((50, 60, 3), dtype=np.uint8)

    discovered, first_timing = manager.process_frame("cam", image, 0)
    tracked, second_timing = manager.process_frame("cam", image, 10)

    assert generator.calls == 1
    assert predictor.propagations == 1
    assert discovered[0].box == [26.0, 15.0, 36.0, 25.0]
    assert tracked[0].instance_id == discovered[0].instance_id == 1
    assert first_timing.discovery_ms > 0
    assert second_timing.propagation_ms > 0
    manager.stop_camera("cam")
    assert predictor.reset is True


def test_resolution_change_resets_camera_state():
    predictor, generator = _FakePredictor(), _FakeGenerator()
    manager = SAM2ModelManager(SAM2Config(device="cpu"), predictor=predictor, generator=generator)
    manager.start_camera("cam", "video.mp4", (50, 60))
    with pytest.raises(RuntimeError, match="resolution/crop/model changed"):
        manager.process_frame("cam", np.zeros((51, 60, 3), np.uint8), 0)
    assert predictor.reset is True
