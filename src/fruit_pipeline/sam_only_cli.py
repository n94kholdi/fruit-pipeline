"""CLI entrypoint: python -m fruit_pipeline.sam_only_cli --image ... --output_dir ...

Same job as ``fruit_pipeline.cli`` but with no detector at all: SAM's own
``SamAutomaticMaskGenerator`` proposes every instance (a dense point-prompt
grid, filtered/deduped by SAM's own quality scores), which SAM then also
segments. Output files and JSON schema match ``fruit_pipeline.cli`` exactly
(``<stem>_final.png`` / ``<stem>_detections.json``), so anything downstream
(size estimation, etc.) that consumes that output works unchanged. There is
no ``<stem>_tiles.png`` here since there is no tiling stage.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from fruit_pipeline.sam_only_pipeline import SamOnlyConfig, load_models, run_sam_only_pipeline
from fruit_pipeline.segmentation.sam import SAM_MODEL_TYPES

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Segment individual fruits in a pallet/box image using SAM alone -- no detector. "
        "SAM's automatic mask generator proposes every instance from a dense point-prompt grid; SAM then "
        "segments each one, same as it would from a detector's boxes."
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
        "'<image_stem>_final.png' and '<image_stem>_detections.json' (no per-image subfolders).",
    )

    sam = parser.add_argument_group("SAM")
    sam.add_argument(
        "--sam-checkpoint",
        default="models/sam_vit_l_0b3195.pth",
        help="Path to a SAM checkpoint matching --sam-model-type.",
    )
    sam.add_argument("--sam-model-type", choices=list(SAM_MODEL_TYPES), default="vit_l", help="SAM backbone size.")
    sam.add_argument(
        "--category-name",
        default="fruit",
        help="Category label written into every saved instance (default: 'fruit') -- there is no "
        "detector to classify instances, so this is a fixed label applied to everything SAM finds.",
    )

    mask_gen = parser.add_argument_group(
        "SamAutomaticMaskGenerator",
        description="Maps 1:1 onto segment_anything.SamAutomaticMaskGenerator's own parameters -- see "
        "its docstring for full details.",
    )
    mask_gen.add_argument(
        "--points-per-side",
        type=int,
        default=32,
        help="Side length of the dense point-prompt grid SAM uses to propose masks (default: 32, i.e. "
        "32x32=1024 points). Higher finds more/smaller instances but is slower.",
    )
    mask_gen.add_argument("--points-per-batch", type=int, default=64, help="Points run through SAM per batch.")
    mask_gen.add_argument(
        "--pred-iou-thresh",
        type=float,
        default=0.88,
        help="SAM's own predicted-mask-quality cutoff in [0,1]; masks below this are dropped.",
    )
    mask_gen.add_argument(
        "--stability-score-thresh",
        type=float,
        default=0.95,
        help="Cutoff in [0,1] on how stable a mask is to the binarization threshold; masks below this are dropped.",
    )
    mask_gen.add_argument("--stability-score-offset", type=float, default=1.0, help="Offset used when computing stability score.")
    mask_gen.add_argument(
        "--box-nms-thresh",
        type=float,
        default=0.7,
        help="IoU threshold for deduping overlapping mask proposals (SAM's own NMS over its point grid).",
    )
    mask_gen.add_argument(
        "--crop-n-layers",
        type=int,
        default=0,
        help="If > 0, additionally reruns the point grid on image crops at that many extra zoom layers -- "
        "helps catch small instances a single full-image pass misses, at extra runtime cost.",
    )
    mask_gen.add_argument("--crop-nms-thresh", type=float, default=0.7, help="NMS threshold between crop layers.")
    mask_gen.add_argument("--crop-overlap-ratio", type=float, default=512 / 1500, help="Overlap between image crops.")
    mask_gen.add_argument(
        "--crop-n-points-downscale-factor",
        type=int,
        default=1,
        help="Shrinks the point grid by this factor for each crop layer beyond the first.",
    )
    mask_gen.add_argument(
        "--min-mask-region-area",
        type=int,
        default=0,
        help="If > 0, uses OpenCV to drop small disconnected mask fragments and fill small holes below this "
        "area (pixels) in every returned mask.",
    )

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
    parser.add_argument("--no-visualization", action="store_true", help="Skip writing the final overlay PNG.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")

    return parser


def _config_from_args(args, image_path: str, output_dir: str) -> SamOnlyConfig:
    return SamOnlyConfig(
        image_path=image_path,
        output_dir=output_dir,
        sam_checkpoint=args.sam_checkpoint,
        sam_model_type=args.sam_model_type,
        category_name=args.category_name,
        points_per_side=args.points_per_side,
        points_per_batch=args.points_per_batch,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        stability_score_offset=args.stability_score_offset,
        box_nms_thresh=args.box_nms_thresh,
        crop_n_layers=args.crop_n_layers,
        crop_nms_thresh=args.crop_nms_thresh,
        crop_overlap_ratio=args.crop_overlap_ratio,
        crop_n_points_downscale_factor=args.crop_n_points_downscale_factor,
        min_mask_region_area=args.min_mask_region_area,
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
        generator = load_models(first_config)

        for idx, image_path in enumerate(image_paths, start=1):
            logging.info("[%d/%d] %s -> %s", idx, len(image_paths), image_path, args.output_dir)
            config = _config_from_args(args, str(image_path), args.output_dir)
            try:
                run_sam_only_pipeline(config, generator=generator)
            except Exception:
                logging.exception("Failed on %s, continuing with remaining images", image_path)
    else:
        config = _config_from_args(args, str(image_arg), args.output_dir)
        run_sam_only_pipeline(config)


if __name__ == "__main__":
    main()
