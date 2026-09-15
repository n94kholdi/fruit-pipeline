"""Lightweight inter-frame tracking to propagate SAM masks without re-running SAM.

SAM remains solely responsible for discovering objects; a tracker here only
carries an already-discovered instance's mask/box forward between SAM
refresh cycles so a video/stream pipeline does not need a fresh SAM pass on
every frame. See ``prompt2_sam_optimzation.md`` (repo root) for the original
design this implements.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import replace

import cv2
import numpy as np

from fruit_pipeline.segmentation.sam import FruitInstance

logger = logging.getLogger(__name__)

TRACKER_TYPES = ("optical_flow",)


class BaseMaskTracker(ABC):
    """Pluggable interface for propagating masks between SAM refreshes.

    Implementations must preserve ``instance_id`` across ``update`` calls so
    object identity stays stable between SAM refreshes; dropping an instance
    (rather than emitting a wrong mask for it) is the expected failure mode
    once tracking confidence is lost for that object.
    """

    @abstractmethod
    def initialize(self, frame_bgr: np.ndarray, instances: list[FruitInstance]) -> None:
        """Reset tracker state from a fresh, SAM-produced set of instances."""

    @abstractmethod
    def update(self, frame_bgr: np.ndarray) -> list[FruitInstance]:
        """Propagate the current instances onto ``frame_bgr`` and return them."""

    def get_masks(self) -> list[FruitInstance]:
        """Return the most recently tracked instances without advancing state."""
        raise NotImplementedError


class OpticalFlowMaskTracker(BaseMaskTracker):
    """Sparse Lucas-Kanade optical flow applied as a per-instance rigid translation.

    Intended for fixed or near-fixed cameras (per the design doc): for each
    tracked mask, a handful of good-features-to-track points sampled inside
    the mask are followed frame-to-frame with pyramidal LK flow; the median
    point displacement translates that instance's mask and box. This
    intentionally does not model rotation, scale, or deformation -- it is a
    simple baseline meant to validate the SAM-refresh architecture, not a
    replacement for SAM2/XMem-grade video segmentation (see
    ``prompt2_sam_optimzation.md``).
    """

    def __init__(
        self,
        *,
        max_corners_per_instance: int = 24,
        min_valid_points: int = 4,
        lk_win_size: tuple[int, int] = (21, 21),
        lk_max_level: int = 3,
    ) -> None:
        self._max_corners = max_corners_per_instance
        self._min_valid_points = min_valid_points
        self._lk_params = dict(
            winSize=lk_win_size,
            maxLevel=lk_max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        self._prev_gray: np.ndarray | None = None
        self._instances: list[FruitInstance] = []
        self._points: dict[int, np.ndarray] = {}

    def initialize(self, frame_bgr: np.ndarray, instances: list[FruitInstance]) -> None:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        self._prev_gray = gray
        self._instances = list(instances)
        self._points = {
            inst.instance_id: self._sample_points(gray, inst.mask) for inst in self._instances
        }

    def _sample_points(self, gray: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
        mask_u8 = mask.astype(np.uint8) * 255
        return cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self._max_corners,
            qualityLevel=0.01,
            minDistance=3,
            mask=mask_u8,
        )

    def update(self, frame_bgr: np.ndarray) -> list[FruitInstance]:
        if self._prev_gray is None:
            raise RuntimeError("OpticalFlowMaskTracker.update called before initialize")
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape

        updated: list[FruitInstance] = []
        next_points: dict[int, np.ndarray] = {}
        for inst in self._instances:
            points = self._points.get(inst.instance_id)
            if points is None or len(points) < self._min_valid_points:
                logger.debug(
                    "Tracker dropping instance %d: not enough trackable points", inst.instance_id
                )
                continue

            new_points, status, _err = cv2.calcOpticalFlowPyrLK(
                self._prev_gray, gray, points, None, **self._lk_params
            )
            status = status.reshape(-1).astype(bool)
            if int(status.sum()) < self._min_valid_points:
                logger.debug("Tracker dropping instance %d: flow lost track", inst.instance_id)
                continue

            displacement = (new_points[status] - points[status]).reshape(-1, 2)
            dx, dy = np.median(displacement, axis=0)

            shifted_mask = _translate_mask(inst.mask, dx, dy)
            if not shifted_mask.any():
                logger.debug("Tracker dropping instance %d: mask left the frame", inst.instance_id)
                continue

            x1, y1, x2, y2 = inst.box
            shifted_box = [
                float(np.clip(x1 + dx, 0, width)),
                float(np.clip(y1 + dy, 0, height)),
                float(np.clip(x2 + dx, 0, width)),
                float(np.clip(y2 + dy, 0, height)),
            ]

            updated.append(replace(inst, box=shifted_box, mask=shifted_mask))
            next_points[inst.instance_id] = new_points[status].reshape(-1, 1, 2)

        self._instances = updated
        self._points = next_points
        self._prev_gray = gray
        return updated

    def get_masks(self) -> list[FruitInstance]:
        return list(self._instances)


def _translate_mask(mask: np.ndarray, dx: float, dy: float) -> np.ndarray:
    height, width = mask.shape
    matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    shifted = cv2.warpAffine(
        mask.astype(np.uint8),
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return shifted.astype(bool)


def build_tracker(tracker_type: str) -> BaseMaskTracker:
    if tracker_type == "optical_flow":
        return OpticalFlowMaskTracker()
    raise ValueError(f"Unknown tracker_type '{tracker_type}', expected one of {TRACKER_TYPES}")
