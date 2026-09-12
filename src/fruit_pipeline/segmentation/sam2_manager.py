"""Persistent SAM2 discovery/video model and per-camera inference state."""

from __future__ import annotations

import gc
import hashlib
import importlib
import json
import logging
import os
import tempfile
import threading
import time
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from fruit_pipeline.segmentation.sam import FruitInstance, filter_masks
from fruit_pipeline.segmentation.sam2_config import SAM2Config
from fruit_pipeline.segmentation.sam2_tracking import (
    BoundedRefreshQueue,
    InstanceReconciler,
    mask_box,
    stable_refresh_phase,
)

logger = logging.getLogger(__name__)


@dataclass
class SAM2Timing:
    discovery_ms: float = 0.0
    propagation_ms: float = 0.0
    queue_wait_ms: float = 0.0


@dataclass
class CameraVideoState:
    camera_id: str
    inference_state: Any
    source: str
    frame_shape: tuple[int, int]
    crop_box: tuple[int, int, int, int] | None
    model_name: str
    instances: list[FruitInstance] = field(default_factory=list)
    reconciler: InstanceReconciler | None = None
    last_activity: float = field(default_factory=time.monotonic)
    last_discovery: float = 0.0
    last_discovery_frame: int = -1
    processed_frames: int = 0
    next_refresh_due: float = 0.0
    dynamic_frames: bool = False
    predictor_frame_index: int = 0
    lifecycle_events: list[dict[str, object]] = field(default_factory=list)
    lock: threading.RLock = field(default_factory=threading.RLock)


class SAM2ModelManager:
    """Own exactly one selected checkpoint and share it for both workloads."""

    def __init__(self, config: SAM2Config, *, predictor=None, generator=None):
        self.config = config
        self.device = torch.device(config.device)
        self._predictor = predictor
        self._generator = generator
        self._image_predictor = None
        self._load_lock = threading.Lock()
        self._discovery_slots = threading.BoundedSemaphore(config.max_concurrent_discoveries)
        self._states: dict[str, CameraVideoState] = {}
        self._states_lock = threading.RLock()
        self._latencies: dict[str, deque[float]] = {
            "discovery": deque(maxlen=2048),
            "propagation": deque(maxlen=8192),
        }
        self.refresh_queue = BoundedRefreshQueue(config.max_refresh_queue)
        self.load_time_ms = 0.0
        self.actual_runtime = "pytorch"
        self.engine_identity: str | None = None
        self.runtime_fallback_reason: str | None = None
        self._engine_metadata: dict[str, object] | None = None

    def _autocast(self):
        if self.device.type != "cuda" or self.config.precision == "fp32":
            return nullcontext()
        dtype = torch.bfloat16 if self.config.precision == "bf16" else torch.float16
        if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError("SAM2 BF16 was selected but this CUDA device does not support BF16; use SAM2_PRECISION=fp16 or fp32")
        return torch.autocast("cuda", dtype=dtype)

    def validate_readiness(self) -> None:
        checkpoint = Path(self.config.resolved_checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"SAM2 checkpoint not found: {checkpoint} (selected model: {self.config.model_name}). "
                "Mount /models or run the deployment model-downloader."
            )
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("sam2_video requires an available CUDA runtime and GPU")
        try:
            import sam2  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("sam2_video requires the pinned facebookresearch/sam2 package") from exc
        if self.config.require_cuda_extension:
            try:
                from sam2 import _C  # noqa: F401
            except ImportError as exc:
                raise RuntimeError("SAM2 CUDA extension is required but unavailable") from exc

    def _configure_runtime(self) -> None:
        if self.config.runtime == "pytorch":
            return
        manifest = Path(self.config.tensorrt_engine_dir) / f"{self.config.model_name}.json"
        reason = None
        if not manifest.is_file():
            reason = f"TensorRT engine manifest is missing: {manifest}"
        else:
            try:
                metadata = json.loads(manifest.read_text(encoding="utf-8"))
                checkpoint_sha = _sha256(Path(self.config.resolved_checkpoint))
                expected = {
                    "checkpoint_sha256": checkpoint_sha,
                    "model_name": self.config.model_name,
                    "precision": self.config.tensorrt_precision,
                    "torch_version": torch.__version__,
                    "cuda_version": torch.version.cuda,
                }
                mismatch = [key for key, value in expected.items() if metadata.get(key) != value]
                if mismatch:
                    reason = "TensorRT engine is incompatible: " + ", ".join(mismatch)
                else:
                    # Engine execution is deliberately feature flagged. The manifest
                    # identifies an offline-built hybrid adapter importable by name.
                    adapter = metadata.get("adapter")
                    if not adapter:
                        reason = "TensorRT engine manifest has no adapter"
                    else:
                        self._engine_metadata = metadata
            except (OSError, ValueError, TypeError) as exc:
                reason = f"Invalid TensorRT engine manifest: {exc}"
        if reason:
            if not self.config.tensorrt_allow_fallback:
                raise RuntimeError(reason)
            self.actual_runtime = "pytorch"
            self.runtime_fallback_reason = reason
            logger.warning("%s; falling back to compiled PyTorch", reason)

    def load(self):
        if self._predictor is not None and self._generator is not None:
            return self._predictor
        with self._load_lock:
            if self._predictor is not None and self._generator is not None:
                return self._predictor
            self.validate_readiness()
            self._configure_runtime()
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            from sam2.build_sam import build_sam2_video_predictor

            started = time.perf_counter()
            predictor = build_sam2_video_predictor(
                self.config.resolved_config,
                self.config.resolved_checkpoint,
                device=str(self.device),
                vos_optimized=self.config.vos_optimized,
            )
            if self._engine_metadata is not None:
                try:
                    module_name, function_name = str(self._engine_metadata["adapter"]).rsplit(".", 1)
                    activate = getattr(importlib.import_module(module_name), function_name)
                    predictor = activate(predictor, self._engine_metadata)
                    self.actual_runtime = self.config.runtime
                    self.engine_identity = str(
                        self._engine_metadata.get("engine_identity", self.config.model_name)
                    )
                except Exception as exc:
                    reason = f"TensorRT adapter activation failed: {exc}"
                    if not self.config.tensorrt_allow_fallback:
                        raise RuntimeError(reason) from exc
                    self.actual_runtime = "pytorch"
                    self.runtime_fallback_reason = reason
                    logger.warning("%s; using PyTorch", reason)
            # The video predictor is a SAM2Base model, so the automatic generator
            # shares these exact weights instead of allocating a second model.
            generator = SAM2AutomaticMaskGenerator(
                predictor,
                points_per_side=self.config.points_per_side,
                points_per_batch=self.config.points_per_batch,
                pred_iou_thresh=self.config.pred_iou_thresh,
                stability_score_thresh=self.config.stability_score_thresh,
                box_nms_thresh=self.config.box_nms_thresh,
                min_mask_region_area=self.config.min_mask_region_area,
                crop_n_layers=self.config.crop_n_layers,
            )
            self._predictor, self._generator = predictor, generator
            self.load_time_ms = (time.perf_counter() - started) * 1000
            logger.info("Loaded %s once on %s in %.1f ms", self.config.model_name, self.device, self.load_time_ms)
            return predictor

    def _get_image_predictor(self):
        """A box-prompted image predictor sharing the resident SAM2 weights.

        ``SAM2ImagePredictor`` wraps the same ``SAM2Base`` instance the video
        predictor and automatic generator use, so detection-mode box-prompted
        segmentation does not allocate a second model checkpoint in VRAM.
        """
        if self._image_predictor is None:
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            model = self.load()
            self._image_predictor = SAM2ImagePredictor(model)
        return self._image_predictor

    def segment_boxes(
        self,
        image_rgb: np.ndarray,
        boxes: np.ndarray,
        *,
        batch_size: int = 16,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Box-prompted SAM2 segmentation of one RGB image.

        Returns ``(masks, scores)`` where ``masks[i]`` is the boolean mask for
        ``boxes[i]`` (``multimask_output=False``, single mask per box) and
        ``scores[i]`` is SAM2's predicted mask-quality IoU. Runs ``segment_boxes``
        under the manager's lock and precision context so it never races video
        inference on the shared model.
        """
        predictor = self._get_image_predictor()
        with self._load_lock, torch.inference_mode(), self._autocast():
            try:
                predictor.set_image(image_rgb)
                boxes_np = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
                mask_chunks: list[np.ndarray] = []
                score_chunks: list[np.ndarray] = []
                for start in range(0, len(boxes_np), batch_size):
                    chunk = boxes_np[start : start + batch_size]
                    masks, scores, _ = predictor.predict(
                        box=chunk,
                        multimask_output=False,
                    )
                    # SAM2ImagePredictor.predict() only squeezes away its leading
                    # per-call batch axis, so with multimask_output=False a
                    # single-box chunk keeps the mask-count axis instead --
                    # (1, H, W) / (1,) -- while a multi-box chunk keeps the box
                    # axis -- (len(chunk), 1, H, W) / (len(chunk), 1).
                    if len(chunk) == 1:
                        mask_chunks.append(np.asarray(masks)[None, 0])
                        score_chunks.append(np.atleast_1d(np.asarray(scores)))
                    else:
                        mask_chunks.append(np.asarray(masks).squeeze(1))
                        score_chunks.append(np.asarray(scores).squeeze(1))
                if not mask_chunks:
                    height, width = image_rgb.shape[:2]
                    return np.empty((0, height, width), dtype=bool), np.empty((0,), dtype=np.float32)
                return np.concatenate(mask_chunks), np.concatenate(score_chunks)
            finally:
                # SAM2ImagePredictor otherwise retains the most recent image
                # embedding on the GPU for the lifetime of the API process.
                if hasattr(predictor, "reset_predictor"):
                    predictor.reset_predictor()

    def start_camera(self, camera_id: str, source: str, frame_shape: tuple[int, int],
                     crop_box: tuple[int, int, int, int] | None = None,
                     initial_frame_rgb: np.ndarray | None = None) -> CameraVideoState:
        self.load()
        with self._states_lock:
            if camera_id in self._states:
                self.stop_camera(camera_id)
            if len(self._states) >= self.config.max_cameras_per_gpu:
                raise RuntimeError("SAM2_MAX_CAMERAS_PER_GPU capacity reached")
            dynamic_frames = initial_frame_rgb is not None and hasattr(self._predictor, "image_size")
            inference_state = (
                self._init_state_from_frame(initial_frame_rgb)
                if dynamic_frames else self._predictor.init_state(
                    video_path=source,
                    offload_video_to_cpu=self.config.offload_video_to_cpu,
                    offload_state_to_cpu=self.config.offload_state_to_cpu,
                )
            )
            state = CameraVideoState(
                camera_id, inference_state, source, frame_shape, crop_box,
                self.config.model_name,
                reconciler=InstanceReconciler(
                    mask_iou_threshold=self.config.match_mask_iou,
                    box_iou_threshold=self.config.match_box_iou,
                    centroid_threshold=self.config.match_centroid_distance,
                    missing_grace_refreshes=self.config.missing_grace_refreshes,
                ),
            )
            state.next_refresh_due = _next_refresh_due(camera_id, time.monotonic(), self.config.refresh_seconds)
            state.dynamic_frames = dynamic_frames
            self._states[camera_id] = state
            return state

    def process_frame(self, camera_id: str, image_rgb: np.ndarray, frame_index: int,
                      *, force_refresh: bool = False, refresh_reason: str = "scheduled") -> tuple[list[FruitInstance], SAM2Timing]:
        state = self._get_state(camera_id)
        with state.lock, torch.inference_mode(), self._autocast():
            self._validate_state_frame(state, image_rgb)
            predictor_frame_index = self._prepare_dynamic_frame(state, image_rgb, frame_index)
            state.processed_frames += 1
            now = time.monotonic()
            due_time = state.last_discovery == 0 or (
                self.config.refresh_seconds > 0 and now >= state.next_refresh_due
            )
            due_frames = self.config.refresh_processed_frames > 0 and (
                state.processed_frames == 1 or state.processed_frames % self.config.refresh_processed_frames == 0
            )
            refresh = force_refresh or not state.instances or due_time or due_frames
            timing = SAM2Timing()
            if refresh:
                reason = "initial" if not state.instances else refresh_reason
                self.refresh_queue.submit(camera_id, reason, now=now)
                request = self.refresh_queue.take(camera_id)
                if request is None:
                    if not state.instances:
                        raise RuntimeError("Initial SAM2 discovery was rejected by the bounded refresh queue")
                    instances, elapsed = self._propagate(state, frame_index)
                    timing.propagation_ms = elapsed
                    self._latencies["propagation"].append(elapsed)
                    return instances, timing
                timing.queue_wait_ms = max(0.0, (time.monotonic() - request.requested_at) * 1000)
                instances, elapsed = self._discover(
                    state, image_rgb, frame_index, predictor_frame_index
                )
                timing.discovery_ms = elapsed
                self._latencies["discovery"].append(elapsed)
                state.last_discovery = now
                state.last_discovery_frame = frame_index
                state.next_refresh_due = _next_refresh_due(
                    camera_id, now + 1e-6, self.config.refresh_seconds,
                    self.config.refresh_jitter_seconds,
                )
                return instances, timing
            instances, elapsed = self._propagate(state, frame_index, predictor_frame_index)
            timing.propagation_ms = elapsed
            self._latencies["propagation"].append(elapsed)
            state.last_activity = now
            return instances, timing

    def _discover(self, state: CameraVideoState, image_rgb: np.ndarray, frame_index: int,
                  predictor_frame_index: int):
        started = time.perf_counter()
        self._log_cuda_memory("before discovery")
        with self._discovery_slots:
            crop, offset = _crop_image(image_rgb, state.crop_box)
            annotations = self._generator.generate(crop)
        proposals: list[FruitInstance] = []
        for annotation in annotations:
            local = np.asarray(annotation["segmentation"], dtype=bool)
            mask = _restore_mask(local, image_rgb.shape[:2], offset)
            box = mask_box(mask)
            score = float(annotation.get("predicted_iou", annotation.get("stability_score", 1.0)))
            proposals.append(FruitInstance(0, box, 1.0, "fruit", score, mask, confidence=score))
        raw_count = len(proposals)
        # NOTE: proposals are never truncated here to fit a VRAM-derived cap.
        # GPU memory constrains how many objects are pushed through the SAM2
        # tracker at once (see tracking_object_batch_size / _register_instances
        # below), not how many fruits a discovery pass is allowed to return.
        proposals = filter_masks(proposals, min_area=self.config.min_mask_region_area)
        if self.config.debug_memory:
            logger.info(
                "SAM2 discovery diagnostics camera=%s raw_masks=%d after_filtering=%d",
                state.camera_id, raw_count, len(proposals),
            )
        if len(proposals) > self.config.max_active_objects_per_camera:
            logger.warning(
                "SAM2 discovery on camera %s returned %d proposals, above the "
                "%d sanity ceiling (SAM2_MAX_ACTIVE_OBJECTS_PER_CAMERA); keeping all of them.",
                state.camera_id, len(proposals), self.config.max_active_objects_per_camera,
            )
        reconciled = state.reconciler.reconcile(state.instances, proposals, frame_index)
        prospective_total = self.total_active_objects - len(state.instances) + len(reconciled.instances)
        if prospective_total > self.config.max_total_active_objects:
            raise RuntimeError("SAM2_MAX_TOTAL_ACTIVE_OBJECTS capacity reached")
        state.instances = reconciled.instances
        state.lifecycle_events.extend(event.__dict__ for event in reconciled.events)
        if self.config.debug_memory:
            logger.info(
                "SAM2 objects sent to tracking: %d tracking_batch_size=%d",
                len(state.instances), self.config.tracking_object_batch_size,
            )
        if state.dynamic_frames:
            # Always start tracking from a fresh SAM2 state at every discovery
            # boundary instead of carrying an ever-growing video state through
            # the whole stream: only application-level instances/IDs survive,
            # the previous inference_state (and any CUDA tensors it held) is
            # dropped and can be reclaimed by the allocator.
            predictor_frame_index = self._reset_tracking_state(state, image_rgb)
        else:
            introduces_new_object = any(event.event == "discovered" for event in reconciled.events)
            if introduces_new_object and state.inference_state.get("tracking_has_started"):
                # SAM2's video predictor only allows registering object ids
                # before the first propagate call ("Cannot add new object id
                # ... after tracking starts"). reset_state() clears only that
                # per-object bookkeeping -- the buffered video frames are
                # untouched -- so a later refresh that finds a genuinely new
                # fruit can safely re-register every still-known instance at
                # the current frame instead of crashing.
                self._predictor.reset_state(state.inference_state)
            else:
                for event in reconciled.events:
                    if event.event == "removed" and hasattr(self._predictor, "remove_object"):
                        self._predictor.remove_object(
                            state.inference_state, event.instance_id, strict=False, need_output=False
                        )
            self._register_instances(state, predictor_frame_index)
        state.last_activity = time.monotonic()
        self._log_cuda_memory("after discovery")
        return state.instances, (time.perf_counter() - started) * 1000

    def _register_instances(self, state: CameraVideoState, predictor_frame_index: int) -> None:
        """Register trackable instances with SAM2, in bounded-size chunks.

        Chunking bounds peak GPU memory for the registration step without
        ever discarding objects: every trackable instance is eventually
        registered, just across multiple smaller calls instead of one huge
        one, with an interim CUDA cache flush between chunks.
        """
        trackable = [item for item in state.instances if item.tracking_state != "uncertain"]
        batch_size = max(1, self.config.tracking_object_batch_size)
        for start in range(0, len(trackable), batch_size):
            chunk = trackable[start:start + batch_size]
            for instance in chunk:
                self._predictor.add_new_mask(
                    state.inference_state, frame_idx=predictor_frame_index,
                    obj_id=instance.instance_id, mask=instance.mask,
                )
            if start + batch_size < len(trackable):
                self.release_unused_cuda_memory()

    def _reset_tracking_state(self, state: CameraVideoState, image_rgb: np.ndarray) -> int:
        """Release the current SAM2 state and re-register instances on a fresh one.

        This is the discovery-boundary lifecycle: preserve only the
        application-level instances/IDs, drop the previous ``inference_state``
        (and any CUDA tensors it references) entirely, and start tracking
        again from a single-frame state. Combined with the frame-history
        reset in ``_prepare_dynamic_frame``, GPU memory used by SAM2's
        per-object memory bank cannot grow unbounded across a long stream.
        """
        self._log_cuda_memory("before state reset")
        old_state = state.inference_state
        self._predictor.reset_state(old_state)
        state.inference_state = self._init_state_from_frame(image_rgb)
        state.predictor_frame_index = 0
        del old_state
        self._register_instances(state, 0)
        self.release_unused_cuda_memory()
        self._log_cuda_memory("after state reset")
        return 0

    def _log_cuda_memory(self, label: str) -> None:
        if not self.config.debug_memory:
            return
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return
        allocated = torch.cuda.memory_allocated(self.device) / 1024**2
        reserved = torch.cuda.memory_reserved(self.device) / 1024**2
        max_allocated = torch.cuda.max_memory_allocated(self.device) / 1024**2
        logger.info(
            "SAM2 CUDA memory [%s]: allocated=%.1fMB reserved=%.1fMB max_allocated=%.1fMB",
            label, allocated, reserved, max_allocated,
        )

    def _propagate(self, state: CameraVideoState, frame_index: int,
                   predictor_frame_index: int):
        started = time.perf_counter()
        self._log_cuda_memory("before propagation")
        if hasattr(self._predictor, "propagate_frame"):
            obj_ids, logits = self._predictor.propagate_frame(state.inference_state, predictor_frame_index)
        else:
            outputs = self._predictor.propagate_in_video(
                state.inference_state, start_frame_idx=predictor_frame_index, max_frame_num_to_track=0,
            )
            _index, obj_ids, logits = next(iter(outputs))
        logits_np = _as_numpy(logits)
        by_id = {item.instance_id: item for item in state.instances}
        tracked: list[FruitInstance] = []
        for index, object_id in enumerate(obj_ids):
            object_id = int(object_id)
            if object_id not in by_id:
                continue
            mask = np.asarray(logits_np[index] > 0).squeeze()
            old = by_id[object_id]
            confidence = float(1 / (1 + np.exp(-np.mean(np.abs(logits_np[index])))))
            tracked.append(replace(old, box=mask_box(mask), mask=mask,
                                   confidence=confidence, sam_score=confidence,
                                   last_seen_frame=frame_index, tracking_state="tracked"))
        tracked_ids = {item.instance_id for item in tracked}
        missing = [replace(item, tracking_state="uncertain") for item in state.instances
                   if item.instance_id not in tracked_ids]
        if missing:
            tracked.extend(missing)
            self.request_refresh(state.camera_id, "tracking_failure")
        elif tracked and np.mean([item.confidence or 0.0 for item in tracked]) < 0.5:
            self.request_refresh(state.camera_id, "low_confidence")
        state.instances = sorted(tracked, key=lambda item: item.instance_id)
        self._log_cuda_memory("after propagation")
        return state.instances, (time.perf_counter() - started) * 1000

    def request_refresh(self, camera_id: str, reason: str) -> bool:
        state = self._get_state(camera_id)
        if time.monotonic() - state.last_discovery < self.config.min_refresh_interval_seconds:
            self.refresh_queue.delayed += 1
            return False
        return self.refresh_queue.submit(camera_id, reason)

    def stop_camera(self, camera_id: str) -> None:
        with self._states_lock:
            state = self._states.pop(camera_id, None)
        if state is not None and self._predictor is not None:
            with state.lock:
                self._predictor.reset_state(state.inference_state)
                state.instances.clear()
                if isinstance(state.inference_state, dict):
                    state.inference_state.clear()
        self.release_unused_cuda_memory()

    def release_unused_cuda_memory(self) -> None:
        """Return unreferenced per-job CUDA blocks to the driver.

        This retains the model weights for fast reuse.  ``torch.cuda.empty_cache``
        only releases allocator cache; live tensors belonging to another active
        camera remain untouched.
        """
        gc.collect()
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def unload(self) -> None:
        """Drop all SAM2 weights and transient state from GPU memory.

        Intended for single-worker deployments that prefer releasing VRAM
        between jobs over keeping the model warm.
        """
        for camera_id in list(self._states):
            self.stop_camera(camera_id)
        with self._load_lock:
            if self._image_predictor is not None and hasattr(
                self._image_predictor, "reset_predictor"
            ):
                self._image_predictor.reset_predictor()
            self._image_predictor = None
            self._generator = None
            self._predictor = None
            self.actual_runtime = "pytorch"
            self.engine_identity = None
            self._engine_metadata = None
            self.load_time_ms = 0.0
        self.release_unused_cuda_memory()

    def cleanup_idle(self, now: float | None = None) -> list[str]:
        current = now or time.monotonic()
        with self._states_lock:
            stale = [key for key, state in self._states.items()
                     if current - state.last_activity >= self.config.camera_idle_timeout_seconds]
        for camera_id in stale:
            self.stop_camera(camera_id)
        return stale

    def _get_state(self, camera_id: str) -> CameraVideoState:
        with self._states_lock:
            if camera_id not in self._states:
                raise KeyError(f"SAM2 camera state is not initialized: {camera_id}")
            return self._states[camera_id]

    def _validate_state_frame(self, state: CameraVideoState, image_rgb: np.ndarray) -> None:
        if image_rgb.shape[:2] != state.frame_shape or state.model_name != self.config.model_name:
            self.stop_camera(state.camera_id)
            raise RuntimeError("SAM2 state reset because frame resolution/crop/model changed")

    def _init_state_from_frame(self, image_rgb: np.ndarray):
        """Use the official loader once, then extend its CPU frame tensor live."""
        with tempfile.TemporaryDirectory(prefix="fruit-sam2-") as directory:
            path = Path(directory) / "00000.jpg"
            if not cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)):
                raise OSError("Could not stage the initial SAM2 frame")
            return self._predictor.init_state(
                video_path=directory,
                offload_video_to_cpu=self.config.offload_video_to_cpu,
                offload_state_to_cpu=self.config.offload_state_to_cpu,
            )

    def _prepare_dynamic_frame(self, state: CameraVideoState, image_rgb: np.ndarray,
                               source_frame_index: int) -> int:
        if not state.dynamic_frames:
            return source_frame_index
        if state.processed_frames == 0:
            return 0
        if state.predictor_frame_index + 1 >= self.config.max_frame_history:
            self._log_cuda_memory("before frame-history reset")
            old_state = state.inference_state
            self._predictor.reset_state(old_state)
            state.inference_state = self._init_state_from_frame(image_rgb)
            state.predictor_frame_index = 0
            del old_state
            self._register_instances(state, 0)
            self.release_unused_cuda_memory()
            self._log_cuda_memory("after frame-history reset")
            state.lifecycle_events.append({
                "event": "reset", "instance_id": 0,
                "frame_index": state.processed_frames,
                "reason": "frame_history_limit",
            })
            return 0
        frame = torch.from_numpy(np.ascontiguousarray(image_rgb)).permute(2, 0, 1).float() / 255.0
        frame = torch.nn.functional.interpolate(
            frame[None], size=(self._predictor.image_size, self._predictor.image_size),
            mode="bilinear", align_corners=False,
        )[0]
        pixel_mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        pixel_std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        frame = (frame - pixel_mean) / pixel_std
        images = state.inference_state["images"]
        state.inference_state["images"] = torch.cat((images, frame[None].to(images.device)), dim=0)
        state.inference_state["num_frames"] += 1
        state.predictor_frame_index += 1
        return state.predictor_frame_index

    @property
    def total_active_objects(self) -> int:
        with self._states_lock:
            return sum(len(state.instances) for state in self._states.values())

    def status(self) -> dict[str, object]:
        allocated = reserved = 0.0
        if self.device.type == "cuda" and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self.device) / 1024**2
            reserved = torch.cuda.memory_reserved(self.device) / 1024**2
        with self._states_lock:
            cameras = len(self._states)
        return {
            "model_name": self.config.model_name,
            "checkpoint": self.config.resolved_checkpoint,
            "config": self.config.resolved_config,
            "precision": self.config.precision,
            "compiled": self.config.vos_optimized,
            "device": str(self.device),
            "loaded": self._predictor is not None,
            "load_time_ms": self.load_time_ms,
            "vram_allocated_mb": allocated,
            "vram_reserved_mb": reserved,
            "configured_runtime": self.config.runtime,
            "actual_runtime": self.actual_runtime,
            "engine_identity": self.engine_identity,
            "runtime_fallback_reason": self.runtime_fallback_reason,
            "active_cameras": cameras,
            "active_objects": self.total_active_objects,
            "refresh_queue": self.refresh_queue.metrics(),
            "timings": {
                name: _latency_summary(values) for name, values in self._latencies.items()
            },
            "discovery_parameters": self.config.discovery_metadata(),
        }

    def refresh_phase(self, camera_id: str) -> float:
        return stable_refresh_phase(camera_id, self.config.refresh_seconds)


def _crop_image(image: np.ndarray, crop_box):
    if crop_box is None:
        return image, (0, 0)
    x1, y1, x2, y2 = crop_box
    if not (0 <= x1 < x2 <= image.shape[1] and 0 <= y1 < y2 <= image.shape[0]):
        raise ValueError("SAM2 crop box is outside the frame")
    return image[y1:y2, x1:x2], (x1, y1)


def _restore_mask(mask: np.ndarray, shape: tuple[int, int], offset: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape and offset == (0, 0):
        return mask
    restored = np.zeros(shape, dtype=bool)
    x, y = offset
    restored[y:y + mask.shape[0], x:x + mask.shape[1]] = mask
    return restored


def _as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _next_refresh_due(camera_id: str, now: float, period: float, jitter: float = 0.0) -> float:
    if period <= 0:
        return float("inf")
    phase = stable_refresh_phase(camera_id, period)
    boundary = (now // period) * period + phase
    if boundary <= now:
        boundary += period
    if jitter > 0:
        digest = hashlib.sha256(f"{camera_id}:{int(boundary // period)}".encode()).digest()
        signed = int.from_bytes(digest[:8], "big") / (2**64 - 1) * 2 - 1
        boundary += signed * min(jitter, period / 4)
    return max(now, boundary)


def _latency_summary(values: deque[float]) -> dict[str, float | int]:
    if not values:
        return {"samples": 0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "fps": 0.0}
    data = np.asarray(values, dtype=np.float64)
    mean = float(data.mean())
    return {
        "samples": len(data),
        "p50_ms": float(np.percentile(data, 50)),
        "p95_ms": float(np.percentile(data, 95)),
        "p99_ms": float(np.percentile(data, 99)),
        "fps": 1000.0 / mean if mean > 0 else 0.0,
    }


_MANAGER: SAM2ModelManager | None = None
_MANAGER_LOCK = threading.Lock()


def get_sam2_model_manager(config: SAM2Config | None = None, *, eager: bool = True) -> SAM2ModelManager:
    global _MANAGER
    selected = config or SAM2Config.from_env()
    with _MANAGER_LOCK:
        if _MANAGER is None:
            _MANAGER = SAM2ModelManager(selected)
        elif _MANAGER.config != selected:
            raise RuntimeError(
                "A different SAM2 model is already resident. Change FRUIT_PIPELINE_SAM2_MODEL and restart the container; hot-swapping is disabled to protect VRAM."
            )
    if eager:
        _MANAGER.load()
    return _MANAGER


def clear_sam2_model_cache() -> None:
    global _MANAGER
    with _MANAGER_LOCK:
        manager, _MANAGER = _MANAGER, None
    if manager is not None:
        for camera_id in list(manager._states):
            manager.stop_camera(camera_id)
