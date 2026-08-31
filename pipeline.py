"""Orchestrates detect -> merge -> segment -> save for a single image."""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch

from fruit_pipeline.detect import TileStats, detect_tiled
from fruit_pipeline.detectors import DetectorBackend, load_detector_backend
from fruit_pipeline.merge import filter_oversized_boxes, merge_detections, to_detections
from fruit_pipeline.prompts import DEFAULT_PROMPT_CONFIG_PATH, load_prompt_config
from fruit_pipeline.segment import FruitInstance, filter_masks, load_sam, segment_boxes
from fruit_pipeline.visualize import draw_overlays, draw_tile_grid

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    image_path: str
    output_dir: str

    # Detection
    detector_weights: str = "models/yolo11x.pt"
    detector: str = "default"  # "default" | "yolo-world" | "yoloe" | "rfdetr"
    use_yolo_world: bool = False  # deprecated: equivalent to detector="yolo-world", kept for old callers
    prompt_classes: list[str] | None = None  # legacy override; if set, wins over prompt_config (no background split)
    prompt_config: str | None = None  # path to a prompts.PromptConfig YAML; defaults to configs/prompts/default.yaml
    yoloe_mode: str = "text"  # "text" | "visual" | "prompt_free" -- detector="yoloe" only
    visual_prompt_paths: list[str] = field(default_factory=list)  # exemplar crop paths -- yoloe_mode="visual" only
    tile_size: int | None = None  # None = adaptive (estimated from fruit diameter); set to force a fixed size
    max_tiles: int = 12  # advisory budget, logged if exceeded; not enforced when tile_size is adaptive
    tile_size_k: float = 8.0
    min_tile_size: int = 320
    max_tile_size: int = 2048
    coarse_pass_long_edge: int = 1400
    coarse_max_box_area_fraction: float = 0.08
    fallback_tile_size: int = 640
    overlap_ratio: float = 0.15
    conf_threshold: float = 0.25
    include_standard_pred: bool = True
    debug_save_tiles: bool = False
    two_resolution_mode: bool = False
    working_long_edge: int = 2800

    # Merge
    merge_strategy: str = "seam-aware"
    merge_metric: str = "IOU"
    merge_iou_threshold: float = 0.5
    class_agnostic_merge: bool = True
    oversized_filter_enabled: bool = True
    oversized_max_area_ratio: float = 3.0
    seam_margin: float | None = None  # None = default to overlap_ratio * tile_size (computed at run time)
    nms_metric: str = "iou"  # "iou" or "diou", used by seam-aware/nms strategies' suppression core
    containment_threshold: float = 0.9  # intersection / smaller-box-area above which the lower-scoring box is dropped

    # Segmentation
    sam_checkpoint: str = "models/sam_vit_l_0b3195.pth"
    sam_model_type: str = "vit_l"
    sam_batch_size: int = 16

    # Mask sanity filters
    min_mask_area: int = 30
    border_filter_enabled: bool = True
    border_touch_ratio: float = 0.6
    aspect_ratio_filter_enabled: bool = True
    max_aspect_ratio: float = 3.0

    device: str = "auto"
    save_visualization: bool = True


def resolve_device(explicit: str) -> str:
    if explicit != "auto":
        return explicit
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_detector_name(config: PipelineConfig) -> str:
    return "yolo-world" if config.use_yolo_world else config.detector


def _resolve_prompts(config: PipelineConfig) -> tuple[list[str], list[str]]:
    """Fruit/background prompt lists for the yolo-world / yoloe-text backends.

    ``prompt_classes`` (legacy, comma-separated CLI flag) wins if set, with
    no background split. Otherwise loads ``prompt_config`` (default:
    ``configs/prompts/default.yaml``), which carries both.
    """
    if config.prompt_classes:
        return list(config.prompt_classes), []
    prompt_config = load_prompt_config(config.prompt_config or DEFAULT_PROMPT_CONFIG_PATH)
    return prompt_config.fruit_prompts, prompt_config.background_prompts


def _load_backend(config: PipelineConfig, device: str) -> DetectorBackend:
    detector_name = _resolve_detector_name(config)
    fruit_prompts: list[str] = []
    background_prompts: list[str] = []
    needs_prompts = detector_name == "yolo-world" or (detector_name == "yoloe" and config.yoloe_mode == "text")
    if needs_prompts:
        fruit_prompts, background_prompts = _resolve_prompts(config)

    return load_detector_backend(
        detector=detector_name,
        weights_path=config.detector_weights,
        device=device,
        conf_threshold=config.conf_threshold,
        fruit_prompts=fruit_prompts,
        background_prompts=background_prompts,
        yoloe_mode=config.yoloe_mode,
        visual_prompt_paths=config.visual_prompt_paths,
    )


def load_models(config: PipelineConfig):
    """Load the detector + SAM predictor once, for reuse across many images."""
    device = resolve_device(config.device)
    detector = _load_backend(config, device)
    sam_predictor = load_sam(
        checkpoint=config.sam_checkpoint,
        model_type=config.sam_model_type,
        device=device,
    )
    return detector, sam_predictor


def run_pipeline(config: PipelineConfig, detector=None, sam_predictor=None) -> list[FruitInstance]:
    """Run the full pipeline on one image.

    Pass a preloaded ``detector``/``sam_predictor`` (from ``load_models``) to
    avoid reloading the models for every image in a batch; otherwise they are
    loaded fresh for this single call.
    """
    device = resolve_device(config.device)
    logger.info("Using device: %s", device)

    if detector is None:
        detector = _load_backend(config, device)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(config.image_path).stem

    image_bgr = cv2.imread(config.image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {config.image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    full_height, full_width = image_rgb.shape[:2]

    # Two-resolution mode: run detection/tiling on a downscaled "working"
    # copy of the image (cheap — fewer/smaller tiles), then rescale the
    # merged boxes back up to full-resolution coordinates below, before
    # segmentation and any full-res crop it feeds. Opt-in, off by default.
    detect_image_path = config.image_path
    working_scale = 1.0
    temp_working_path: str | None = None
    if config.two_resolution_mode:
        long_edge = max(full_width, full_height)
        working_scale = min(1.0, config.working_long_edge / long_edge) if long_edge else 1.0
        if working_scale < 1.0:
            working_bgr = cv2.resize(
                image_bgr,
                (int(round(full_width * working_scale)), int(round(full_height * working_scale))),
                interpolation=cv2.INTER_AREA,
            )
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp.close()
            cv2.imwrite(tmp.name, working_bgr)
            temp_working_path = tmp.name
            detect_image_path = temp_working_path
            logger.info(
                "Two-resolution mode: detecting at %dx%d (scale=%.3f of full-res %dx%d)",
                working_bgr.shape[1],
                working_bgr.shape[0],
                working_scale,
                full_width,
                full_height,
            )
        else:
            logger.info(
                "Two-resolution mode requested but %dx%d is already <= working_long_edge=%d; running at full res",
                full_width,
                full_height,
                config.working_long_edge,
            )

    debug_tiles_dir = str(output_dir / "tiles_debug" / stem) if config.debug_save_tiles else None
    try:
        detection_result = detect_tiled(
            image_path=detect_image_path,
            backend=detector,
            tile_size=config.tile_size,
            overlap_ratio=config.overlap_ratio,
            conf_threshold=config.conf_threshold,
            include_standard_pred=config.include_standard_pred,
            class_agnostic_relabel=_resolve_detector_name(config) == "default",
            max_tiles=config.max_tiles,
            tile_size_k=config.tile_size_k,
            min_tile_size=config.min_tile_size,
            max_tile_size=config.max_tile_size,
            coarse_pass_long_edge=config.coarse_pass_long_edge,
            coarse_max_box_area_fraction=config.coarse_max_box_area_fraction,
            fallback_tile_size=config.fallback_tile_size,
            debug_tiles_dir=debug_tiles_dir,
        )
    finally:
        if temp_working_path:
            Path(temp_working_path).unlink(missing_ok=True)

    raw_predictions = detection_result.raw_predictions
    tile_results = detection_result.tile_results
    tile_stats = detection_result.stats

    # Default seam margin = the tile overlap in pixels, so a fruit split at
    # a tile seam (guaranteed to land within the overlap band on both
    # tiles) is always seam-eligible for union, without needing a separate
    # per-run flag in the common case.
    seam_margin = (
        config.seam_margin if config.seam_margin is not None else config.overlap_ratio * tile_stats.tile_size
    )

    merged = merge_detections(
        raw_predictions,
        strategy=config.merge_strategy,
        match_metric=config.merge_metric,
        match_threshold=config.merge_iou_threshold,
        class_agnostic=config.class_agnostic_merge,
        tile_ids=detection_result.tile_ids,
        tile_rects=detection_result.tile_rects,
        seam_margin=seam_margin,
        nms_metric=config.nms_metric,
        containment_threshold=config.containment_threshold,
    )
    merged = filter_oversized_boxes(
        merged,
        max_area_ratio=config.oversized_max_area_ratio,
        enabled=config.oversized_filter_enabled,
    )
    detections = to_detections(merged)

    if working_scale < 1.0:
        inv_scale = 1.0 / working_scale
        for det in detections:
            det.box = [v * inv_scale for v in det.box]
        logger.info(
            "Two-resolution mode: rescaled %d detection box(es) from working res back to full res (x%.3f)",
            len(detections),
            inv_scale,
        )

    if sam_predictor is None:
        sam_predictor = load_sam(
            checkpoint=config.sam_checkpoint,
            model_type=config.sam_model_type,
            device=device,
        )
    instances = segment_boxes(
        image_rgb=image_rgb,
        predictor=sam_predictor,
        detections=detections,
        batch_size=config.sam_batch_size,
    )
    instances = filter_masks(
        instances,
        min_area=config.min_mask_area,
        border_filter_enabled=config.border_filter_enabled,
        border_touch_ratio=config.border_touch_ratio,
        aspect_ratio_filter_enabled=config.aspect_ratio_filter_enabled,
        max_aspect_ratio=config.max_aspect_ratio,
    )

    save_json(
        instances,
        config.image_path,
        output_dir / f"{stem}_detections.json",
        tile_stats,
        full_image_size=(full_width, full_height),
    )
    if config.save_visualization:
        final_canvas = draw_overlays(image_bgr, instances)
        cv2.imwrite(str(output_dir / f"{stem}_final.png"), final_canvas)

        tiles_canvas = draw_tile_grid(tile_results)
        cv2.imwrite(str(output_dir / f"{stem}_tiles.png"), tiles_canvas)

    if config.two_resolution_mode:
        crops_dir = output_dir / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)
        for inst in instances:
            x1, y1, x2, y2 = (int(round(v)) for v in inst.box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(full_width, x2), min(full_height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = image_bgr[y1:y2, x1:x2]
            cv2.imwrite(str(crops_dir / f"{stem}_instance_{inst.instance_id:03d}.png"), crop)
        logger.info("Two-resolution mode: saved %d full-res instance crop(s) to %s", len(instances), crops_dir)

    print(f"{stem}: total fruits detected: {len(instances)}")
    return instances


def mask_to_polygons(mask: np.ndarray, epsilon: float = 1.0) -> list[list[float]]:
    """Convert a boolean mask to COCO-style polygons (one list of x,y per contour)."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) < 1:
            continue
        approx = cv2.approxPolyDP(contour, epsilon, closed=True)
        if len(approx) < 3:
            continue
        polygons.append(approx.reshape(-1).astype(float).tolist())
    return polygons


def save_json(
    instances: list[FruitInstance],
    image_path: str,
    output_path: Path,
    tile_stats: TileStats,
    full_image_size: tuple[int, int] | None = None,
) -> None:
    records = []
    for inst in instances:
        records.append(
            {
                "instance_id": inst.instance_id,
                "box": [round(v, 2) for v in inst.box],
                "detector_score": round(inst.detector_score, 4),
                "sam_score": round(inst.sam_score, 4),
                "category_name": inst.category_name,
                "mask_area": int(inst.mask.sum()),
                "mask_polygon": mask_to_polygons(inst.mask),
            }
        )

    image_width, image_height = full_image_size or tile_stats.image_size

    payload = {
        "image_path": str(image_path),
        "image_width": image_width,
        "image_height": image_height,
        "num_tiles": tile_stats.num_tiles,
        "raw_detection_count": tile_stats.raw_detection_count,
        "detection_tile_size": tile_stats.tile_size,
        "estimated_fruit_diameter_px": tile_stats.estimated_fruit_diameter_px,
        "detection_image_size": list(tile_stats.image_size),
        "num_fruits": len(records),
        "detections": records,
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Saved %d detections to %s", len(records), output_path)
