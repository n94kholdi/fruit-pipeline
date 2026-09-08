"""Command line interface for end-to-end fruit counting and sizing."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

from fruit_pipeline.cli import _config_from_args, build_parser as build_detection_parser
from fruit_pipeline.integrated_pipeline import (
    INPUT_ROTATIONS,
    IntegratedFruitSizingPipeline,
    IntegratedPipelineConfig,
    media_source_stem,
)
from fruit_pipeline.live import FruitLiveReporter
from fruit_pipeline.sam_only_pipeline import SamOnlyConfig
from fruit_pipeline.size_estimation.pipeline import SizeEstimationConfig

INFERENCE_MODES = ("sam_only", "detector")


def build_parser():
    parser = build_detection_parser()
    parser.description = (
        "Select/load pallet corners, then detect, segment, count, and size fruit in an image or video. "
        "Two inference modes: 'sam_only' (default, see --inference-mode) needs no detector at all -- "
        "SAM's own automatic mask generator proposes and segments every fruit; 'detector' is the original "
        "detector + box-prompted-SAM pipeline (see the 'detection'/'merge' groups below)."
    )
    for action in parser._actions:
        if action.dest == "image":
            action.help = "Path to one input image or video."
            break

    mode = parser.add_argument_group("inference mode")
    mode.add_argument(
        "--inference-mode",
        choices=INFERENCE_MODES,
        default="sam_only",
        help="'sam_only' (default): no detector -- SAM's automatic mask generator proposes and segments "
        "every fruit instance itself (tune it via the 'SAM automatic mask generator' group below; the "
        "'detection'/'merge' groups are ignored in this mode). 'detector': the original detector + "
        "box-prompted-SAM pipeline (the 'SAM automatic mask generator' group is ignored in this mode).",
    )

    mask_gen = parser.add_argument_group(
        "SAM automatic mask generator",
        description="--inference-mode sam_only only. Maps 1:1 onto "
        "segment_anything.SamAutomaticMaskGenerator's own parameters.",
    )
    mask_gen.add_argument(
        "--category-name",
        default="fruit",
        help="Category label written into every saved instance (default: 'fruit') -- there is no "
        "detector to classify instances in this mode, so this is a fixed label applied to everything "
        "SAM finds.",
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

    sizing = parser.add_argument_group("calibrated fruit sizing")
    sizing.add_argument("--camera-id", required=True, help="Camera calibration identifier.")
    sizing.add_argument("--camera-group", help="Optional fallback camera calibration group.")
    sizing.add_argument(
        "--calibration-dir",
        required=True,
        help="Calibration store containing cameras/ and optionally groups/.",
    )
    sizing.add_argument(
        "--pallet-type",
        required=True,
        help="Pallet type key from --pallet-config.",
    )
    sizing.add_argument(
        "--pallet-config",
        default="config/pallet_types.yaml",
        help="Pallet dimensions YAML (default: config/pallet_types.yaml).",
    )
    sizing.add_argument(
        "--pallet-selection",
        help="Destination for this run's manual corner JSON. Default: "
        "<output_dir>/pallet_selection.json.",
    )
    sizing.add_argument(
        "--pallet-points-file",
        help="Headless/dashboard alternative: JSON with four TL,TR,BR,BL points. It replaces and saves "
        "--pallet-selection without opening the interactive selector.",
    )
    sizing.add_argument("--min-pallet-confidence", type=float, default=0.5)
    sizing.add_argument(
        "--reuse-pallet-selection",
        action="store_true",
        help="Reuse an existing --pallet-selection instead of selecting the pallet at the start of this run.",
    )
    sizing.add_argument(
        "--min-pallet-overlap",
        type=float,
        default=0.5,
        help="Minimum fraction of a fruit mask that must lie inside the selected pallet to be counted and "
        "measured (default: 0.5).",
    )
    sizing.add_argument("--max-calibration-error", type=float, default=2.0)
    sizing.add_argument("--rectified-pixels-per-mm", type=float, default=0.5)
    sizing.add_argument(
        "--no-size-debug",
        action="store_true",
        help="Skip the measurement overlay and rectified pallet image.",
    )
    sizing.add_argument(
        "--max-preview-size",
        type=int,
        default=900,
        help="Maximum interactive pallet-preview dimension in pixels (default: 900).",
    )
    sizing.add_argument(
        "--resize-to-calibration",
        action="store_true",
        help="Temporary compatibility mode: rotate if needed and resize each input frame to the stored "
        "calibration resolution before pallet setup and inference. Only safe for the same camera view, "
        "aspect ratio, and crop used during calibration.",
    )
    sizing.add_argument(
        "--input-rotation",
        choices=INPUT_ROTATIONS,
        default="auto",
        help="Rotation applied before temporary calibration resizing (default: auto). Auto uses a clockwise "
        "quarter-turn when that matches the calibration aspect ratio better.",
    )
    sizing.add_argument(
        "--allow-unsafe-resize",
        action="store_true",
        help="Testing only: stretch aspect-mismatched inputs to the calibration resolution. "
        "Resulting physical measurements are invalid.",
    )

    video = parser.add_argument_group("video sampling")
    video.add_argument(
        "--frame-step",
        type=int,
        default=10,
        help="Process every Nth video frame (default: 10).",
    )
    video.add_argument(
        "--max-frames",
        type=int,
        help="Optional maximum number of sampled video frames to process.",
    )
    live = parser.add_argument_group("dashboard live reporting")
    live.add_argument("--live-job-dir", help="Dashboard job directory for live events and preview.")
    live.add_argument("--live-job-id", help="Dashboard job identifier used in live events.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    source: str | Path = args.image if "://" in args.image else Path(args.image)
    if isinstance(source, Path) and source.is_dir():
        parser.error("The integrated sizing command accepts one image or video, not a directory")

    output_dir = Path(args.output_dir)
    selection_path = Path(args.pallet_selection) if args.pallet_selection else (
        output_dir / "pallet_selection.json"
    )
    detection_config = None
    sam_only_config = None
    if args.inference_mode == "sam_only":
        sam_only_config = SamOnlyConfig(
            image_path=str(source),
            output_dir=str(output_dir),
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
    else:
        detection_config = _config_from_args(args, str(source), str(output_dir))
    sizing_config = SizeEstimationConfig(
        camera_id=args.camera_id,
        camera_group=args.camera_group,
        calibration_dir=args.calibration_dir,
        pallet_config_path=args.pallet_config,
        min_pallet_confidence=args.min_pallet_confidence,
        max_calibration_error=args.max_calibration_error,
        debug=not args.no_size_debug,
        rectified_pixels_per_mm=args.rectified_pixels_per_mm,
    )
    config = IntegratedPipelineConfig(
        detection=detection_config,
        sam_only=sam_only_config,
        sizing=sizing_config,
        pallet_type=args.pallet_type,
        pallet_selection_path=selection_path,
        pallet_points_file=args.pallet_points_file,
        frame_step=args.frame_step,
        max_frames=args.max_frames,
        max_preview_size=args.max_preview_size,
        resize_to_calibration=args.resize_to_calibration,
        allow_unsafe_resize=args.allow_unsafe_resize,
        input_rotation=args.input_rotation,
        reuse_pallet_selection=args.reuse_pallet_selection,
        min_pallet_overlap=args.min_pallet_overlap,
    )
    reporter = None
    if args.live_job_dir or args.live_job_id:
        if not args.live_job_dir or not args.live_job_id:
            parser.error("--live-job-dir and --live-job-id must be provided together")
        reporter = FruitLiveReporter(args.live_job_dir, args.live_job_id)

    def publish_frame(frame_result, preview, processed_count, total_count):
        if reporter is None:
            return
        reporter.publish_frame(
            preview,
            frame_index=frame_result.frame_index,
            timestamp_ms=frame_result.timestamp_ms,
            processed_frame_count=processed_count,
            total_sampled_frames=total_count,
            num_fruits=frame_result.num_fruits,
            num_measured_fruits=len(frame_result.sizing.measurements),
        )

    result = IntegratedFruitSizingPipeline(
        config,
        frame_processed=publish_frame if reporter is not None else None,
    ).run(source)
    summary_path = output_dir / f"{media_source_stem(source)}_summary.json"
    print(
        f"Processed {len(result.frames)} frame(s); "
        f"fruit observations: {result.num_fruits}; summary: {summary_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
