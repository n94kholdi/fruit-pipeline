"""SAM box-prompted segmentation.

Every mask is produced by prompting SAM with one merged detection box
(`predictor.predict_torch(boxes=..., multimask_output=False)`), never with
``SamAutomaticMaskGenerator``. This keeps segmentation strictly
detection-driven so background, stems, and shadows are never proposed as
separate "objects" the way automatic mask generation would.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch

from fruit_pipeline.merge import Detection
from fruit_pipeline.paths import resolve_model_path

logger = logging.getLogger(__name__)

# ViT-B is ~3-4x faster and much lighter on GPU/CPU memory than ViT-L, at a
# noticeable drop in mask boundary quality on cluttered/touching objects;
# ViT-H is the highest quality but slowest and heaviest. ViT-L is a
# reasonable default middle ground for dense fruit crates.
SAM_MODEL_TYPES = ("vit_b", "vit_l", "vit_h")


@dataclass
class FruitInstance:
    """A final per-fruit record: detection box + its SAM mask."""

    instance_id: int
    box: list[float]
    detector_score: float
    category_name: str
    sam_score: float
    mask: np.ndarray  # bool array, shape (H, W)


def load_sam(checkpoint: str, model_type: str = "vit_l", device: str = "cpu"):
    """Load a pretrained SAM checkpoint and return a ``SamPredictor``.

    No automatic mask generator is created here on purpose (see module
    docstring) — only the predictor, which is driven by explicit box prompts.
    """
    import os

    from segment_anything import SamPredictor, sam_model_registry

    if model_type not in SAM_MODEL_TYPES:
        raise ValueError(f"Unknown SAM model_type '{model_type}', expected one of {SAM_MODEL_TYPES}")
    checkpoint = resolve_model_path(checkpoint)
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(
            f"SAM checkpoint not found: {checkpoint}\n"
            "Download the matching checkpoint from "
            "https://github.com/facebookresearch/segment-anything#model-checkpoints "
            "or point --sam-checkpoint at an existing one."
        )

    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    logger.info("Loaded SAM (%s) from %s on %s", model_type, checkpoint, device)
    return SamPredictor(sam)


def segment_boxes(
    image_rgb: np.ndarray,
    predictor,
    detections: list[Detection],
    batch_size: int = 16,
) -> list[FruitInstance]:
    """Run box-prompted SAM segmentation for every detection, batched.

    The image embedding is computed once via ``set_image``; boxes are then
    fed through ``predict_torch`` in chunks of ``batch_size`` so 100+ boxes
    per image don't require 100+ separate forward passes through the encoder.
    """
    if not detections:
        return []

    predictor.set_image(image_rgb)
    device = predictor.device
    original_size = image_rgb.shape[:2]

    boxes_np = np.array([det.box for det in detections], dtype=np.float32)
    instances: list[FruitInstance] = []

    for start in range(0, len(detections), batch_size):
        chunk_dets = detections[start : start + batch_size]
        chunk_boxes = torch.as_tensor(boxes_np[start : start + batch_size], device=device)
        transformed_boxes = predictor.transform.apply_boxes_torch(chunk_boxes, original_size)

        masks, iou_predictions, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        masks = masks.squeeze(1).cpu().numpy()  # (chunk, H, W) bool
        scores = iou_predictions.squeeze(1).cpu().numpy()

        for det, mask, sam_score in zip(chunk_dets, masks, scores):
            instances.append(
                FruitInstance(
                    instance_id=det.instance_id,
                    box=det.box,
                    detector_score=det.score,
                    category_name=det.category_name,
                    sam_score=float(sam_score),
                    mask=mask.astype(bool),
                )
            )

    logger.info("SAM produced %d masks (batch_size=%d)", len(instances), batch_size)
    return instances


def filter_masks(
    instances: list[FruitInstance],
    min_area: int = 30,
    border_filter_enabled: bool = True,
    border_touch_ratio: float = 0.6,
    aspect_ratio_filter_enabled: bool = True,
    max_aspect_ratio: float = 3.0,
) -> list[FruitInstance]:
    """Sanity-filter SAM masks before they become final fruit instances.

    - Drops near-zero-area masks (degenerate SAM output).
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
