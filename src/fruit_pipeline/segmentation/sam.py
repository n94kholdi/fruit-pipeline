"""SAM2 box-prompted fruit segmentation.

Every mask is produced by prompting the resident SAM2 model with one merged
detection box (``SAM2ImagePredictor.predict(box=..., multimask_output=False)``).
Segmentation stays strictly detection-driven so background, stems, and shadows
are never proposed as separate "objects".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np

from fruit_pipeline.detection.merging import Detection

logger = logging.getLogger(__name__)


@dataclass
class FruitInstance:
    """A final per-fruit record: detection box + its SAM2 mask."""

    instance_id: int
    box: list[float]
    detector_score: float
    category_name: str
    sam_score: float
    mask: np.ndarray  # bool array, shape (H, W)
    # Video lifecycle fields are optional so image callers and their positional
    # constructor contract remain unchanged.
    confidence: float | None = None
    first_seen_frame: int | None = None
    last_seen_frame: int | None = None
    last_discovery_frame: int | None = None
    tracking_state: Literal["discovered", "tracked", "uncertain", "lost"] | None = None

    @property
    def bbox(self) -> list[float]:
        """Alias for the existing ``box`` field."""
        return self.box


def segment_boxes(
    image_rgb: np.ndarray,
    predictor,
    detections: list[Detection],
    batch_size: int = 16,
) -> list[FruitInstance]:
    """Run box-prompted SAM2 segmentation for every detection, batched.

    The image embedding is computed once via ``set_image``; boxes are then fed
    through ``SAM2ImagePredictor.predict`` in chunks of ``batch_size`` so many
    boxes per image don't require a full image-encoder pass for each.
    """
    if not detections:
        return []

    boxes_np = np.array([det.box for det in detections], dtype=np.float32)
    masks, scores = predictor.segment_boxes(
        image_rgb,
        boxes_np,
        batch_size=batch_size,
    )

    instances: list[FruitInstance] = []
    for det, mask, sam_score in zip(detections, masks, scores):
        instances.append(
            FruitInstance(
                instance_id=det.instance_id,
                box=det.box,
                detector_score=det.score,
                category_name=det.category_name,
                sam_score=float(sam_score),
                mask=np.asarray(mask).astype(bool, copy=False),
            )
        )
    logger.info("SAM2 produced %d masks (batch_size=%d)", len(instances), batch_size)
    return instances


def filter_masks(
    instances: list[FruitInstance],
    min_area: int = 30,
    border_filter_enabled: bool = True,
    border_touch_ratio: float = 0.6,
    aspect_ratio_filter_enabled: bool = True,
    max_aspect_ratio: float = 3.0,
) -> list[FruitInstance]:
    """Sanity-filter SAM2 masks before they become final fruit instances.

    - Drops near-zero-area masks (degenerate output).
    - Drops masks that hug an entire image edge rather than just touching it,
      which is the signature of a background strip/crate wall getting
      segmented instead of a single (possibly edge-cropped) fruit.
    - Optionally drops masks with an extreme aspect ratio, since fruit is
      roughly round/oval. This is a heuristic, not a learned rule, so it can
      be disabled entirely via ``aspect_ratio_filter_enabled``.
    """
    kept: list[FruitInstance] = []
    dropped_area = dropped_border = dropped_aspect = 0

    for inst in instances:
        area = int(inst.mask.sum())
        if area < min_area:
            dropped_area += 1
            continue

        if border_filter_enabled and _touches_border_excessively(inst.mask, inst.box, border_touch_ratio):
            dropped_border += 1
            continue

        if aspect_ratio_filter_enabled and not _plausible_aspect_ratio(inst.mask, max_aspect_ratio):
            dropped_aspect += 1
            continue

        kept.append(inst)

    if dropped_area or dropped_border or dropped_aspect:
        logger.info(
            "Mask sanity filters dropped %d (near-zero area=%d, border=%d, aspect-ratio=%d), %d remain",
            dropped_area + dropped_border + dropped_aspect,
            dropped_area,
            dropped_border,
            dropped_aspect,
            len(kept),
        )
    return kept


def _touches_border_excessively(mask: np.ndarray, box: list[float], touch_ratio: float) -> bool:
    """True if the mask spans most of an image edge it touches, not just a sliver."""
    height, width = mask.shape
    x1, y1, x2, y2 = box
    edge_margin = 2.0

    checks = []
    if y1 <= edge_margin:
        checks.append((mask[0, :], max(x2 - x1, 1.0)))
    if y2 >= height - edge_margin:
        checks.append((mask[-1, :], max(x2 - x1, 1.0)))
    if x1 <= edge_margin:
        checks.append((mask[:, 0], max(y2 - y1, 1.0)))
    if x2 >= width - edge_margin:
        checks.append((mask[:, -1], max(y2 - y1, 1.0)))

    for edge_pixels, box_extent in checks:
        if edge_pixels.sum() / box_extent > touch_ratio:
            return True
    return False


def _plausible_aspect_ratio(mask: np.ndarray, max_aspect_ratio: float) -> bool:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return False
    height = ys.max() - ys.min() + 1
    width = xs.max() - xs.min() + 1
    ratio = max(height, width) / max(1, min(height, width))
    return ratio <= max_aspect_ratio
