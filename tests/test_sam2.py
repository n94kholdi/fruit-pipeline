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


class _SequencedGenerator:
    """Returns a different discovery batch on each successive call."""

    def __init__(self, batches):
        self._batches = batches
        self.calls = 0

    def generate(self, image):
        batch = self._batches[min(self.calls, len(self._batches) - 1)]
        self.calls += 1
        return batch


class _TrackingStartsFakePredictor:
    """Mirrors the real SAM2VideoPredictor's new-object-after-tracking guard."""

    def __init__(self, shape=(50, 60)):
        self.shape = shape
        self.known_ids: set[int] = set()
        self.reset_calls = 0

    def init_state(self, video_path):
        return {"tracking_has_started": False}

    def add_new_mask(self, state, frame_idx, obj_id, mask):
        if obj_id not in self.known_ids and state["tracking_has_started"]:
            raise RuntimeError(
                f"Cannot add new object id {obj_id} after tracking starts. "
                "Please call 'reset_state' to restart from scratch."
            )
        self.known_ids.add(obj_id)

    def propagate_frame(self, state, frame_index):
        state["tracking_has_started"] = True
        height, width = self.shape
        logits = np.full((1, 1, height, width), -1.0, dtype=np.float32)
        logits[:, :, 5:15, 6:16] = 1.0
        return [1], logits

    def reset_state(self, state):
        state["tracking_has_started"] = False
        self.known_ids.clear()


def _mask_region(y1, x1, y2, x2, shape=(50, 60)):
    mask = np.zeros(shape, dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


class _FakeImagePredictor:
    """Reproduces SAM2ImagePredictor.predict()'s own squeeze(0) quirk.

    Its wrapper only squeezes away the leading per-call box axis, so with
    multimask_output=False a single-box call keeps the mask-count axis
    instead -- (1, H, W) / (1,) -- while a multi-box call keeps the box axis
    -- (len(box), 1, H, W) / (len(box), 1).
    """

    def __init__(self, shape=(50, 60)):
        self.shape = shape
        self.box_counts_seen: list[int] = []

    def set_image(self, image):
        pass

    def predict(self, box, multimask_output):
        assert multimask_output is False
        count = np.asarray(box).shape[0]
        self.box_counts_seen.append(count)
        height, width = self.shape
        masks = np.zeros((count, 1, height, width), dtype=bool)
        scores = np.full((count, 1), 0.9, dtype=np.float32)
        # torch.Tensor.squeeze(0) (what predict() really calls) is a no-op
        # when dim 0 isn't size 1, unlike numpy's axis-checked squeeze.
        def squeeze0(arr):
            return arr[0] if arr.shape[0] == 1 else arr

        return squeeze0(masks), squeeze0(scores), None


def test_segment_boxes_handles_single_box_and_multi_box_chunks():
    manager = SAM2ModelManager(SAM2Config(device="cpu"))
    image = np.zeros((50, 60, 3), dtype=np.uint8)
    boxes = np.array([[0, 0, 5, 5], [1, 1, 6, 6], [2, 2, 7, 7]], dtype=np.float32)

    fake = _FakeImagePredictor()
    manager._image_predictor = fake
    masks, scores = manager.segment_boxes(image, boxes, batch_size=1)
    assert fake.box_counts_seen == [1, 1, 1]
    assert masks.shape == (3, 50, 60)
    assert scores.shape == (3,)

    fake = _FakeImagePredictor()
    manager._image_predictor = fake
    masks, scores = manager.segment_boxes(image, boxes, batch_size=16)
    assert fake.box_counts_seen == [3]
    assert masks.shape == (3, 50, 60)
    assert scores.shape == (3,)


def test_discover_resets_state_before_adding_an_object_found_after_tracking_started():
    predictor = _TrackingStartsFakePredictor()
    generator = _SequencedGenerator([
        [{"segmentation": _mask_region(5, 6, 15, 16), "predicted_iou": 0.9}],
        [
            {"segmentation": _mask_region(5, 6, 15, 16), "predicted_iou": 0.9},
            {"segmentation": _mask_region(30, 40, 40, 50), "predicted_iou": 0.9},
        ],
    ])
    config = SAM2Config(device="cpu", refresh_seconds=999, min_mask_region_area=1)
    manager = SAM2ModelManager(config, predictor=predictor, generator=generator)
    manager.start_camera("cam", "video.mp4", (50, 60))
    image = np.zeros((50, 60, 3), dtype=np.uint8)

    manager.process_frame("cam", image, 0)  # initial discovery: object 1
    manager.process_frame("cam", image, 1)  # propagates -> tracking_has_started = True

    instances, _ = manager.process_frame("cam", image, 2, force_refresh=True)

    assert {item.instance_id for item in instances} == {1, 2}
