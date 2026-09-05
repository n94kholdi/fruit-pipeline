"""SAM-only fruit segmentation: no detector at all.

``fruit_pipeline.segmentation.sam`` prompts SAM with boxes coming from an
external detector (YOLO/YOLO-World/YOLOE). This module instead uses
``SamAutomaticMaskGenerator`` so SAM itself proposes the instances (a dense
grid of point prompts, one mask per surviving point after SAM's own
IoU/stability/NMS filtering) -- i.e. SAM does detection *and* segmentation.

Output shape matches ``fruit_pipeline.pipeline``/``fruit_pipeline.cli``
exactly: a list of ``FruitInstance`` (box, detector_score, sam_score,
category_name, mask), fed through the same ``filter_masks``/``save_json``/
``draw_overlays`` so downstream stages (size estimation, etc.) work
unchanged.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from fruit_pipeline.segmentation.sam import SAM_MODEL_TYPES, FruitInstance
from fruit_pipeline.utils.paths import resolve_model_path

logger = logging.getLogger(__name__)


def load_sam_automatic_generator(
    checkpoint: str,
    model_type: str = "vit_l",
    device: str = "cpu",
    points_per_side: int | None = 32,
    points_per_batch: int = 64,
    pred_iou_thresh: float = 0.88,
    stability_score_thresh: float = 0.95,
    stability_score_offset: float = 1.0,
    box_nms_thresh: float = 0.7,
    crop_n_layers: int = 0,
    crop_nms_thresh: float = 0.7,
    crop_overlap_ratio: float = 512 / 1500,
    crop_n_points_downscale_factor: int = 1,
    min_mask_region_area: int = 0,
):
    """Load a pretrained SAM checkpoint and return a ``SamAutomaticMaskGenerator``.

    Every keyword mirrors ``segment_anything.SamAutomaticMaskGenerator``
    verbatim (same names/defaults) so CLI flags map straight through with no
    surprises. See its docstring for what each one does; the short version
    used below in ``--help``-style terms:

    - ``points_per_side``: side length of the dense point-prompt grid (None
      to supply your own ``point_grids`` -- not exposed here).
    - ``pred_iou_thresh`` / ``stability_score_thresh``: SAM's own per-mask
      quality gates (both in [0, 1]); raise them to keep only confident,
      stable masks.
    - ``box_nms_thresh``: dedup threshold across the resulting mask boxes.
    - ``crop_n_layers`` > 0 additionally reruns the grid on image crops, for
      small objects that a single full-image pass misses.
    - ``min_mask_region_area`` > 0 uses OpenCV to drop small disconnected
      mask fragments and fill small holes (requires opencv-python, already
      a dependency here).
    """
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

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
    generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=points_per_side,
        points_per_batch=points_per_batch,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        stability_score_offset=stability_score_offset,
        box_nms_thresh=box_nms_thresh,
        crop_n_layers=crop_n_layers,
        crop_nms_thresh=crop_nms_thresh,
        crop_overlap_ratio=crop_overlap_ratio,
        crop_n_points_downscale_factor=crop_n_points_downscale_factor,
        min_mask_region_area=min_mask_region_area,
        output_mode="binary_mask",
    )
    logger.info("Loaded SAM (%s) automatic mask generator from %s on %s", model_type, checkpoint, device)
    return generator


def generate_instances(
    image_rgb: np.ndarray,
    generator,
    category_name: str = "fruit",
) -> list[FruitInstance]:
    """Run ``SamAutomaticMaskGenerator`` and package each proposal as a ``FruitInstance``.

    Ids are assigned in the same reading order (top-to-bottom, then
    left-to-right) that ``fruit_pipeline.detection.merging.to_detections``
    uses for detector-driven boxes, so ids stay meaningful once cross
    referenced against the saved visualization/JSON.

    There is no detector, so ``detector_score`` is filled in with SAM's own
    ``predicted_iou`` for that mask (the closest analogue: how confident SAM
    itself is that the mask is a full, correct object) and ``sam_score`` is
    ``stability_score`` (robustness of the mask to the exact threshold used
    to binarize SAM's output).
    """
    raw_masks = generator.generate(image_rgb)

    boxes = []
    for m in raw_masks:
        x, y, w, h = m["bbox"]
        boxes.append([float(x), float(y), float(x + w), float(y + h)])

    order = sorted(range(len(raw_masks)), key=lambda i: (boxes[i][1], boxes[i][0]))

    instances = [
        FruitInstance(
            instance_id=idx,
            box=boxes[i],
            detector_score=float(raw_masks[i].get("predicted_iou", 0.0)),
            category_name=category_name,
            sam_score=float(raw_masks[i].get("stability_score", raw_masks[i].get("predicted_iou", 0.0))),
            mask=raw_masks[i]["segmentation"].astype(bool),
        )
        for idx, i in enumerate(order)
    ]
    logger.info("SAM automatic mask generator produced %d raw mask proposal(s)", len(instances))
    return instances
