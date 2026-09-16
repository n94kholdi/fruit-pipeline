"""Decide when to re-run SAM and how masks are reused for video.

Implements the refresh-interval + tracking design from
``prompt2_sam_optimzation.md`` (repo root): SAM remains the sole source of
new object discovery; a :class:`~fruit_pipeline.segmentation.tracking.BaseMaskTracker`
only carries existing instances across the frames between refreshes, so a
video/stream pipeline does not have to run SAM on every processed frame.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from fruit_pipeline.segmentation.sam import FruitInstance
from fruit_pipeline.segmentation.sam_manager import env_flag
from fruit_pipeline.segmentation.tracking import TRACKER_TYPES, BaseMaskTracker, build_tracker

logger = logging.getLogger(__name__)


@dataclass
class VideoSegmentationConfig:
    """Env-var-configurable knobs for the SAM-refresh + tracking video mode.

    Defaults preserve today's behavior exactly: ``tracking_enabled`` is
    false, so :class:`VideoSegmentationManager` calls SAM on every processed
    frame, identical to the pipeline before tracking existed.
    """

    tracking_enabled: bool = field(
        default_factory=lambda: env_flag("FRUIT_PIPELINE_TRACKING_ENABLED", False)
    )
    sam_refresh_interval: int = field(
        default_factory=lambda: int(os.getenv("FRUIT_PIPELINE_SAM_REFRESH_INTERVAL", "30"))
    )
    tracker_type: str = field(
        default_factory=lambda: os.getenv("FRUIT_PIPELINE_TRACKER_TYPE", "optical_flow")
    )
    static_mask_refresh_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.sam_refresh_interval <= 0:
            raise ValueError("sam_refresh_interval must be positive")
        if self.tracker_type not in TRACKER_TYPES:
            raise ValueError(f"tracker_type must be one of {TRACKER_TYPES}")
        if (
            self.static_mask_refresh_seconds is not None
            and self.static_mask_refresh_seconds <= 0
        ):
            raise ValueError("static_mask_refresh_seconds must be positive")
        if self.static_mask_refresh_seconds is not None and self.tracking_enabled:
            raise ValueError("static mask reuse and tracking cannot be enabled together")


@dataclass
class VideoFrameTimings:
    used_sam: bool
    sam_ms: float = 0.0
    tracking_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.sam_ms + self.tracking_ms


class VideoSegmentationManager:
    """Per-video-stream state: current tracker plus SAM-refresh bookkeeping.

    One instance is scoped to a single video/stream run. ``process`` counts
    consecutive *processed* frames (i.e. after any upstream frame-step
    sampling), so ``sam_refresh_interval`` is in units of processed frames,
    not raw video frames.
    """

    def __init__(self, config: VideoSegmentationConfig) -> None:
        self.config = config
        self._tracker: BaseMaskTracker | None = (
            build_tracker(config.tracker_type) if config.tracking_enabled else None
        )
        self._sample_index = 0
        self._has_state = False
        self._static_instances: list[FruitInstance] = []
        self._last_sam_timestamp_seconds: float | None = None

    def process(
        self,
        frame_bgr: np.ndarray,
        run_sam: Callable[[], list[FruitInstance]],
        *,
        timestamp_seconds: float | None = None,
    ) -> tuple[list[FruitInstance], VideoFrameTimings]:
        """Return this frame's instances, calling ``run_sam`` only when needed.

        SAM runs on the first frame, whenever tracking is disabled (today's
        behavior: every processed frame), and on every
        ``sam_refresh_interval``-th processed frame thereafter. All other
        frames reuse the tracker's propagated masks.
        """
        index = self._sample_index
        self._sample_index += 1

        static_interval = self.config.static_mask_refresh_seconds
        if static_interval is not None:
            if timestamp_seconds is None:
                raise ValueError("timestamp_seconds is required for static mask reuse")
            needs_sam = (
                not self._has_state
                or self._last_sam_timestamp_seconds is None
                or timestamp_seconds - self._last_sam_timestamp_seconds >= static_interval
                # A timestamp reset means a new/restarted media timeline.
                or timestamp_seconds < self._last_sam_timestamp_seconds
            )
        else:
            needs_sam = (
                not self.config.tracking_enabled
                or not self._has_state
                or index % self.config.sam_refresh_interval == 0
            )

        if needs_sam:
            started = time.perf_counter()
            instances = run_sam()
            sam_ms = (time.perf_counter() - started) * 1000.0
            if static_interval is not None:
                self._static_instances = instances
                self._last_sam_timestamp_seconds = timestamp_seconds
                self._has_state = True
            elif self.config.tracking_enabled:
                assert self._tracker is not None
                self._tracker.initialize(frame_bgr, instances)
                self._has_state = True
            timings = VideoFrameTimings(used_sam=True, sam_ms=sam_ms)
        elif static_interval is not None:
            # Deliberately keep masks at their last SAM coordinates. This mode
            # is for fixed cameras and mostly stationary fruit; no optical flow
            # or other tracker is run between inference timestamps.
            instances = self._static_instances
            timings = VideoFrameTimings(used_sam=False)
        else:
            assert self._tracker is not None
            started = time.perf_counter()
            instances = self._tracker.update(frame_bgr)
            tracking_ms = (time.perf_counter() - started) * 1000.0
            timings = VideoFrameTimings(used_sam=False, tracking_ms=tracking_ms)

        logger.info(
            "Video frame %d: %s (sam=%.1fms, tracking=%.1fms, total=%.1fms, instances=%d)",
            index,
            (
                "SAM refresh"
                if timings.used_sam
                else "reused static masks"
                if static_interval is not None
                else "tracked"
            ),
            timings.sam_ms,
            timings.tracking_ms,
            timings.total_ms,
            len(instances),
        )
        return instances, timings
