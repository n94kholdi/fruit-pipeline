"""CLI entrypoint: python -m fruit_pipeline.cli --image ... --output_dir ..."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from fruit_pipeline.detect import DEFAULT_PROMPT_CLASSES
from fruit_pipeline.pipeline import PipelineConfig, load_models, run_pipeline
from fruit_pipeline.segment import SAM_MODEL_TYPES

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect and segment individual fruits in a pallet/box image (pretrained models only)."
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to one input image, OR a directory of images (non-recursive) to batch-process.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory. Every image's results are written flat into this one directory as "
        "'<image_stem>_final.png', '<image_stem>_tiles.png', and '<image_stem>_detections.json' "
        "(no per-image subfolders).",
    )

    detector = parser.add_argument_group("detection")
    detector.add_argument(
        "--detector-weights",
        default="yolo11x.pt",
        help="Ultralytics detector checkpoint. Use a *-world.pt checkpoint with --use-yolo-world.",
    )
    detector.add_argument(
        "--use-yolo-world",
        action="store_true",
        help="Treat --detector-weights as YOLO-World and prompt it with --prompt-classes for open-vocabulary "
        "detection instead of relying on COCO's 80 classes. Requires the ultralytics CLIP extra "
        "(auto-installed on first use, downloads an additional ~350MB text-encoder checkpoint).",
    )
    detector.add_argument(
        "--prompt-classes",
        default=",".join(DEFAULT_PROMPT_CLASSES),
        help="Comma-separated text prompt classes for YOLO-World (only used with --use-yolo-world).",
    )
    detector.add_argument(
        "--tile-size",
        type=int,
        default=None,
        help="Square SAHI tile size in pixels. Default: adaptively estimated per image via a fast coarse "
        "detection pre-pass that measures median fruit diameter, then tile_size = --tile-size-k * diameter "
        "(clamped to [--min-tile-size, --max-tile-size]). Set explicitly to DISABLE the adaptive pre-pass and "
        "force a fixed size instead (e.g. to reproduce old fixed-size behavior for debugging/comparison).",
    )
    detector.add_argument(
        "--tile-size-k",
        type=float,
        default=8.0,
        help="Adaptive tiling only: tile_size = k * estimated median fruit diameter (default 8; try 6-10).",
    )
    detector.add_argument(
        "--min-tile-size",
        type=int,
        default=320,
        help="Adaptive tiling only: minimum tile size in pixels, clamps against degenerate (too-small) estimates.",
    )
    detector.add_argument(
        "--max-tile-size",
        type=int,
        default=2048,
        help="Adaptive tiling only: maximum tile size in pixels, clamps against degenerate (too-large) estimates.",
    )
    detector.add_argument(
        "--coarse-pass-long-edge",
        type=int,
        default=1400,
        help="Adaptive tiling only: long-edge resolution the image is downscaled to for the fruit-diameter "
        "pre-pass (default 1400).",
    )
    detector.add_argument(
        "--coarse-max-box-area-fraction",
        type=float,
        default=0.08,
        help="Adaptive tiling only: coarse pre-pass boxes covering more than this fraction of the downscaled "
        "image are dropped before computing the median fruit diameter (default 0.08) — filters out 'boxed a "
        "whole cluster/pile' detections that would otherwise skew the estimate on a dense, busy image.",
    )
    detector.add_argument(
        "--fallback-tile-size",
        type=int,
        default=640,
        help="Tile size used when the fruit-diameter pre-pass is degenerate (too few coarse detections, e.g. "
        "a near-empty crate); a warning is logged when this triggers.",
    )
    detector.add_argument(
        "--max-tiles",
        type=int,
        default=12,
        help="Advisory tile-count budget: logged as a warning if the adaptively-chosen tile size still "
        "produces more tiles than this. Not enforced by itself — use --max-tile-size to actually cap tile size.",
    )
    detector.add_argument(
        "--overlap-ratio",
        type=float,
        default=0.15,
        help="Fractional overlap between tiles (default 0.15). With adaptive tile sizing, tiles rarely cut "
        "through a single fruit, so this can be smaller than a fixed-tile-size setup would need (~0.10-0.15 "
        "is usually enough); raise it if fruit near tile boundaries are getting missed or double-counted.",
    )
    detector.add_argument("--conf-threshold", type=float, default=0.25, help="Per-tile detector confidence threshold.")
    detector.add_argument(
        "--no-standard-pred",
        action="store_true",
        help="Skip the extra full-image (unsliced) detection pass used to help catch large fruit.",
    )
    detector.add_argument(
        "--debug-save-tiles",
        action="store_true",
        help="Save every individual tile crop to '<output_dir>/tiles_debug/<image_stem>/' before it's passed "
        "to the detector, so tiling can be visually sanity-checked (blank/duplicate/out-of-bounds crops would "
        "be immediately obvious).",
    )
    detector.add_argument(
        "--two-resolution",
        action="store_true",
        help="Opt-in: run detection/tiling on a downscaled 'working' copy of the image (--working-long-edge) "
        "instead of full resolution, then map merged boxes back to full-resolution coordinates before "
        "segmentation. Also saves a full-resolution crop per instance to '<output_dir>/crops/', for a later "
        "sizing stage that needs true pixel precision. Off by default; does not change existing behavior "
        "unless passed.",
    )
    detector.add_argument(
        "--working-long-edge",
        type=int,
        default=2800,
        help="--two-resolution only: long-edge resolution (pixels) used for the downscaled detection pass "
        "(default 2800). Ignored if the image is already smaller than this.",
    )

    merge = parser.add_argument_group("merge")
    merge.add_argument(
        "--merge-strategy",
        choices=["greedy_nmm", "nmm", "nms"],
        default="greedy_nmm",
        help="How overlapping tiled detections are combined. greedy_nmm (default) merges touching/adjacent "
        "fruit instead of deleting them like plain nms would.",
    )
    merge.add_argument(
        "--merge-metric",
        choices=["IOU", "IOS"],
        default="IOU",
        help="Overlap metric for merging (default IOU). IOS can over-merge distinct, touching fruit into one "
        "unioned box on a dense crate photo (see merge.py); switch to IOS only if under-merging (duplicate "
        "detections of the same fruit) turns out to be the bigger problem for your images.",
    )
    merge.add_argument("--merge-iou-threshold", type=float, default=0.5, help="Overlap threshold to merge two boxes.")
    merge.add_argument(
        "--no-class-agnostic-merge",
        action="store_true",
        help="Merge per-category instead of across all categories (only matters with --use-yolo-world multi-class prompts).",
    )
    merge.add_argument(
        "--no-oversized-filter",
        action="store_true",
        help="Disable the heuristic filter that drops boxes far larger than the median fruit box.",
    )
    merge.add_argument(
        "--oversized-ratio",
        type=float,
        default=3.0,
        help="Reject boxes larger than this multiple of the median detected box area.",
    )

    sam = parser.add_argument_group("segmentation")
    sam.add_argument(
        "--sam-checkpoint",
        default="models/sam_vit_l_0b3195.pth",
        help="Path to a SAM checkpoint matching --sam-model-type.",
    )
    sam.add_argument("--sam-model-type", choices=list(SAM_MODEL_TYPES), default="vit_l", help="SAM backbone size.")
    sam.add_argument("--sam-batch-size", type=int, default=16, help="Boxes per batched SAM predict_torch call.")

    filters = parser.add_argument_group("mask sanity filters")
    filters.add_argument("--min-mask-area", type=int, default=30, help="Drop masks smaller than this, in pixels.")
    filters.add_argument("--no-border-filter", action="store_true", help="Disable the background-strip-at-edge filter.")
    filters.add_argument(
        "--border-touch-ratio",
        type=float,
        default=0.6,
        help="Fraction of an edge-adjacent box side that must be covered by mask to flag it as a border artifact.",
    )
    filters.add_argument(
        "--no-aspect-ratio-filter",
        action="store_true",
        help="Disable the round/oval-fruit-shape heuristic filter (it's a heuristic, not a learned rule).",
    )
    filters.add_argument("--max-aspect-ratio", type=float, default=3.0, help="Max mask bounding-box aspect ratio kept.")

    parser.add_argument("--device", default="auto", help="'auto', 'cpu', 'cuda', or 'cuda:N'.")
    parser.add_argument("--no-visualization", action="store_true", help="Skip writing visualization.png.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")

    return parser


def _config_from_args(args, image_path: str, output_dir: str) -> PipelineConfig:
    return PipelineConfig(
        image_path=image_path,
        output_dir=output_dir,
        detector_weights=args.detector_weights,
        use_yolo_world=args.use_yolo_world,
        prompt_classes=[c.strip() for c in args.prompt_classes.split(",") if c.strip()],
        tile_size=args.tile_size,
        max_tiles=args.max_tiles,
        tile_size_k=args.tile_size_k,
        min_tile_size=args.min_tile_size,
        max_tile_size=args.max_tile_size,
        coarse_pass_long_edge=args.coarse_pass_long_edge,
        coarse_max_box_area_fraction=args.coarse_max_box_area_fraction,
        fallback_tile_size=args.fallback_tile_size,
        overlap_ratio=args.overlap_ratio,
        conf_threshold=args.conf_threshold,
        include_standard_pred=not args.no_standard_pred,
        debug_save_tiles=args.debug_save_tiles,
        two_resolution_mode=args.two_resolution,
        working_long_edge=args.working_long_edge,
        merge_strategy=args.merge_strategy,
        merge_metric=args.merge_metric,
        merge_iou_threshold=args.merge_iou_threshold,
        class_agnostic_merge=not args.no_class_agnostic_merge,
        oversized_filter_enabled=not args.no_oversized_filter,
        oversized_max_area_ratio=args.oversized_ratio,
        sam_checkpoint=args.sam_checkpoint,
        sam_model_type=args.sam_model_type,
        sam_batch_size=args.sam_batch_size,
        min_mask_area=args.min_mask_area,
        border_filter_enabled=not args.no_border_filter,
        border_touch_ratio=args.border_touch_ratio,
        aspect_ratio_filter_enabled=not args.no_aspect_ratio_filter,
        max_aspect_ratio=args.max_aspect_ratio,
        device=args.device,
        save_visualization=not args.no_visualization,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    image_arg = Path(args.image)

    if image_arg.is_dir():
        image_paths = sorted(
            p for p in image_arg.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not image_paths:
            raise SystemExit(f"No images found directly inside {image_arg} (extensions: {sorted(IMAGE_EXTENSIONS)})")

        logging.info("Batch mode: %d images found in %s", len(image_paths), image_arg)
        first_config = _config_from_args(args, str(image_paths[0]), args.output_dir)
        detector, sam_predictor = load_models(first_config)

        for idx, image_path in enumerate(image_paths, start=1):
            logging.info("[%d/%d] %s -> %s", idx, len(image_paths), image_path, args.output_dir)
            config = _config_from_args(args, str(image_path), args.output_dir)
            try:
                run_pipeline(config, detector=detector, sam_predictor=sam_predictor)
            except Exception:
                logging.exception("Failed on %s, continuing with remaining images", image_path)
    else:
        config = _config_from_args(args, str(image_arg), args.output_dir)
        run_pipeline(config)


if __name__ == "__main__":
    main()
