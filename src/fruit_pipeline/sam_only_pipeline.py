"""Orchestrates SAM-only inference: SamAutomaticMaskGenerator -> filter -> save.

Same output contract as :mod:`fruit_pipeline.pipeline` (``run_pipeline``): a
``list[FruitInstance]``, a ``<stem>_detections.json`` in the exact same
schema, and (optionally) a ``<stem>_final.png`` overlay -- so anything
downstream that consumes ``fruit_pipeline.cli`` output (size estimation,
etc.) works unchanged. The only difference is where the instances come from:
here it's SAM's automatic mask generator instead of detector boxes + SAM box
prompting, so there is no detector, no tiling, and no per-tile merge stage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2

from fruit_pipeline.detection.tiling import TileStats
from fruit_pipeline.pipeline import resolve_device, save_json
from fruit_pipeline.segmentation.sam import FruitInstance, filter_masks
from fruit_pipeline.segmentation.sam_auto import generate_instances, load_sam_automatic_generator
from fruit_pipeline.visualization.rendering import draw_overlays

logger = logging.getLogger(__name__)


@dataclass
class SamOnlyConfig:
    image_path: str
    output_dir: str

    # Segmentation / "detection" (SAM automatic mask generator)
    sam_checkpoint: str = "models/sam_vit_l_0b3195.pth"
    sam_model_type: str = "vit_l"
    category_name: str = "fruit"
    points_per_side: int = 32
    points_per_batch: int = 64
    pred_iou_thresh: float = 0.88
    stability_score_thresh: float = 0.95
    stability_score_offset: float = 1.0
    box_nms_thresh: float = 0.7
    crop_n_layers: int = 0
    crop_nms_thresh: float = 0.7
    crop_overlap_ratio: float = 512 / 1500
    crop_n_points_downscale_factor: int = 1
    min_mask_region_area: int = 0

    # Mask sanity filters (same knobs/defaults as fruit_pipeline.cli)
    min_mask_area: int = 30
    border_filter_enabled: bool = True
    border_touch_ratio: float = 0.6
    aspect_ratio_filter_enabled: bool = True
    max_aspect_ratio: float = 3.0

    device: str = "auto"
    save_visualization: bool = True


def load_models(config: SamOnlyConfig):
    """Load the SAM automatic mask generator once, for reuse across many images."""
    device = resolve_device(config.device)
    return load_sam_automatic_generator(
        checkpoint=config.sam_checkpoint,
        model_type=config.sam_model_type,
        device=device,
        points_per_side=config.points_per_side,
        points_per_batch=config.points_per_batch,
        pred_iou_thresh=config.pred_iou_thresh,
        stability_score_thresh=config.stability_score_thresh,
        stability_score_offset=config.stability_score_offset,
        box_nms_thresh=config.box_nms_thresh,
        crop_n_layers=config.crop_n_layers,
        crop_nms_thresh=config.crop_nms_thresh,
        crop_overlap_ratio=config.crop_overlap_ratio,
        crop_n_points_downscale_factor=config.crop_n_points_downscale_factor,
        min_mask_region_area=config.min_mask_region_area,
    )


def run_sam_only_pipeline(config: SamOnlyConfig, generator=None) -> list[FruitInstance]:
    """Run SAM-only inference on one image.

    Pass a preloaded ``generator`` (from ``load_models``) to avoid reloading
    SAM for every image in a batch; otherwise it is loaded fresh here.
    """
    device = resolve_device(config.device)
    logger.info("Using device: %s", device)

    if generator is None:
        generator = load_models(config)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(config.image_path).stem

    image_bgr = cv2.imread(config.image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {config.image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]

    instances = generate_instances(image_rgb, generator, category_name=config.category_name)
    instances = filter_masks(
        instances,
        min_area=config.min_mask_area,
        border_filter_enabled=config.border_filter_enabled,
        border_touch_ratio=config.border_touch_ratio,
        aspect_ratio_filter_enabled=config.aspect_ratio_filter_enabled,
        max_aspect_ratio=config.max_aspect_ratio,
    )

    tile_stats = TileStats(
        num_tiles=1,
        raw_detection_count=len(instances),
        image_size=(width, height),
        tile_size=0,
        estimated_fruit_diameter_px=None,
    )

    save_json(
        instances,
        config.image_path,
        output_dir / f"{stem}_detections.json",
        tile_stats,
        full_image_size=(width, height),
    )

    if config.save_visualization:
        final_canvas = draw_overlays(image_bgr, instances)
        cv2.imwrite(str(output_dir / f"{stem}_final.png"), final_canvas)

    print(f"{stem}: total fruits detected: {len(instances)}")
    return instances
