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
)
from fruit_pipeline.size_estimation.pipeline import SizeEstimationConfig


def build_parser():
    parser = build_detection_parser()
    parser.description = (
        "Select/load pallet corners, then detect, segment, count, and size fruit in an image or video."
    )
    for action in parser._actions:
        if action.dest == "image":
            action.help = "Path to one input image or video."
            break

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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    source = Path(args.image)
    if source.is_dir():
        parser.error("The integrated sizing command accepts one image or video, not a directory")

    output_dir = Path(args.output_dir)
    selection_path = Path(args.pallet_selection) if args.pallet_selection else (
        output_dir / "pallet_selection.json"
    )
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
    result = IntegratedFruitSizingPipeline(config).run(source)
    summary_path = output_dir / f"{source.stem}_summary.json"
    print(
        f"Processed {len(result.frames)} frame(s); "
        f"fruit observations: {result.num_fruits}; summary: {summary_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
