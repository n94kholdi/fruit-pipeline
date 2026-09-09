"""Stable-ID reconciliation and bounded refresh scheduling for SAM2 video."""

from __future__ import annotations

import hashlib
import heapq
import math
import threading
import time
from dataclasses import dataclass, field, replace

import numpy as np

from fruit_pipeline.segmentation.sam import FruitInstance


def mask_box(mask: np.ndarray) -> list[float]:
    ys, xs = np.where(mask)
    if not len(xs):
        return [0.0, 0.0, 0.0, 0.0]
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = int(np.count_nonzero(left & right))
    union = int(np.count_nonzero(left | right))
    return intersection / union if union else 0.0


def box_iou(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def normalized_centroid_distance(left: FruitInstance, right: FruitInstance) -> float:
    def center(box):
        return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    lc, rc = center(left.box), center(right.box)
    diameter = max(
        1.0,
        math.sqrt(max(1.0, float(left.mask.sum()))),
        math.sqrt(max(1.0, float(right.mask.sum()))),
    )
    return math.hypot(lc[0] - rc[0], lc[1] - rc[1]) / diameter


@dataclass(frozen=True)
class LifecycleEvent:
    event: str
    instance_id: int
    frame_index: int


@dataclass
class ReconcileResult:
    instances: list[FruitInstance]
    events: list[LifecycleEvent]


class InstanceReconciler:
    """Deterministic greedy one-to-one matching with a missing-object grace."""

    def __init__(self, *, mask_iou_threshold=0.35, box_iou_threshold=0.2,
                 centroid_threshold=2.0, missing_grace_refreshes=2):
        self.mask_iou_threshold = mask_iou_threshold
        self.box_iou_threshold = box_iou_threshold
        self.centroid_threshold = centroid_threshold
        self.missing_grace_refreshes = missing_grace_refreshes
        self._next_id = 1
        self._missing: dict[int, int] = {}

    def reconcile(self, existing: list[FruitInstance], discovered: list[FruitInstance],
                  frame_index: int) -> ReconcileResult:
        if existing:
            self._next_id = max(self._next_id, max(item.instance_id for item in existing) + 1)
        candidates: list[tuple[float, int, int]] = []
        for old_index, old in enumerate(existing):
            for new_index, new in enumerate(discovered):
                miou = mask_iou(old.mask, new.mask)
                biou = box_iou(old.box, new.box)
                distance = normalized_centroid_distance(old, new)
                if miou >= self.mask_iou_threshold or (
                    biou >= self.box_iou_threshold and distance <= self.centroid_threshold
                ):
                    score = 0.65 * miou + 0.25 * biou + 0.10 / (1.0 + distance)
                    candidates.append((-score, old.instance_id, new_index))
        matched_old: set[int] = set()
        matched_new: set[int] = set()
        result: list[FruitInstance] = []
        events: list[LifecycleEvent] = []
        old_by_id = {item.instance_id: item for item in existing}
        for _negative_score, old_id, new_index in sorted(candidates):
            if old_id in matched_old or new_index in matched_new:
                continue
            old, new = old_by_id[old_id], discovered[new_index]
            result.append(replace(
                new, instance_id=old_id,
                first_seen_frame=old.first_seen_frame if old.first_seen_frame is not None else frame_index,
                last_seen_frame=frame_index, last_discovery_frame=frame_index,
                tracking_state="discovered",
            ))
            self._missing.pop(old_id, None)
            matched_old.add(old_id)
            matched_new.add(new_index)
            events.append(LifecycleEvent("matched", old_id, frame_index))
        for index, new in enumerate(discovered):
            if index in matched_new:
                continue
            instance_id = self._next_id
            self._next_id += 1
            result.append(replace(
                new, instance_id=instance_id, first_seen_frame=frame_index,
                last_seen_frame=frame_index, last_discovery_frame=frame_index,
                tracking_state="discovered",
            ))
            events.append(LifecycleEvent("discovered", instance_id, frame_index))
        for old in existing:
            if old.instance_id in matched_old:
                continue
            misses = self._missing.get(old.instance_id, 0) + 1
            self._missing[old.instance_id] = misses
            if misses <= self.missing_grace_refreshes:
                result.append(replace(old, tracking_state="uncertain"))
                events.append(LifecycleEvent("lost", old.instance_id, frame_index))
            else:
                self._missing.pop(old.instance_id, None)
                events.append(LifecycleEvent("removed", old.instance_id, frame_index))
        result.sort(key=lambda item: item.instance_id)
        return ReconcileResult(result, events)


def stable_refresh_phase(camera_id: str, period_seconds: float) -> float:
    if period_seconds <= 0:
        return 0.0
    digest = hashlib.sha256(camera_id.encode("utf-8")).digest()
    fraction = int.from_bytes(digest[:8], "big") / float(2**64)
    return fraction * period_seconds


_PRIORITY = {"initial": 0, "scene_change": 1, "tracking_failure": 1,
             "low_confidence": 2, "new_pallet": 1, "scheduled": 10}


@dataclass(order=True)
class RefreshRequest:
    priority: int
    requested_at: float
    camera_id: str = field(compare=True)
    reason: str = field(compare=False, default="scheduled")


class BoundedRefreshQueue:
    """Thread-safe priority queue that coalesces requests per camera."""

    def __init__(self, max_size: int):
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        self.max_size = max_size
        self._heap: list[RefreshRequest] = []
        self._cameras: set[str] = set()
        self._lock = threading.Lock()
        self.accepted = self.delayed = self.dropped = 0

    def submit(self, camera_id: str, reason: str, now: float | None = None) -> bool:
        with self._lock:
            if camera_id in self._cameras:
                self.delayed += 1
                return False
            if len(self._heap) >= self.max_size:
                self.dropped += 1
                return False
            request = RefreshRequest(_PRIORITY.get(reason, 5), now or time.monotonic(), camera_id, reason)
            heapq.heappush(self._heap, request)
            self._cameras.add(camera_id)
            self.accepted += 1
            return True

    def pop(self) -> RefreshRequest | None:
        with self._lock:
            if not self._heap:
                return None
            request = heapq.heappop(self._heap)
            self._cameras.remove(request.camera_id)
            return request

    def take(self, camera_id: str) -> RefreshRequest | None:
        """Take one camera's coalesced request when its frame is available."""
        with self._lock:
            for index, request in enumerate(self._heap):
                if request.camera_id == camera_id:
                    self._heap[index] = self._heap[-1]
                    self._heap.pop()
                    if index < len(self._heap):
                        heapq.heapify(self._heap)
                    self._cameras.remove(camera_id)
                    return request
            return None

    def metrics(self, now: float | None = None) -> dict[str, float | int]:
        with self._lock:
            current = now or time.monotonic()
            oldest = min((item.requested_at for item in self._heap), default=current)
            return {"depth": len(self._heap), "oldest_request_age_seconds": max(0.0, current - oldest),
                    "accepted": self.accepted, "delayed": self.delayed, "dropped": self.dropped}
