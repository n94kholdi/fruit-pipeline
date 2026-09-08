from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from fruit_pipeline.camera_calibration.calibration_store import CalibrationStore
from fruit_pipeline.measurement.contour_transform import contour_to_pallet_mm
from fruit_pipeline.measurement.fruit_measurement import measure_contour
from fruit_pipeline.pallet_geometry.detector import ManualPalletDetector, PalletGeometryError
from fruit_pipeline.pallet_geometry.homography import compute_pallet_homography, rectify_pallet
from fruit_pipeline.pallet_geometry.pallet_config import PalletTypeConfig

from .manual_selection import load_points, select_bounding_box, select_points

CORNER_LABELS = ("TL", "TR", "BR", "BL")


def _read_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def _draw_pallet(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    canvas = image.copy()
    integer_corners = np.round(corners).astype(np.int32)
    cv2.polylines(canvas, [integer_corners], True, (0, 220, 255), 3, cv2.LINE_AA)
    for label, point in zip(CORNER_LABELS, integer_corners):
        center = tuple(point)
        cv2.circle(canvas, center, 7, (0, 220, 255), -1, cv2.LINE_AA)
        cv2.putText(canvas, label, (center[0] + 9, center[1] - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2, cv2.LINE_AA)
    return canvas


def select_pallet_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Select and save pallet corners in TL, TR, BR, BL order."
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--pallet-type", required=True)
    parser.add_argument("--output", required=True, help="Destination corners JSON")
    parser.add_argument("--visualization", help="Corner-overlay image (default: beside JSON)")
    parser.add_argument("--points-file", help="Headless/testing alternative to interactive clicking")
    parser.add_argument(
        "--max-preview-size",
        type=int,
        default=900,
        help="Maximum preview width or height in pixels (default: 900)",
    )
    args = parser.parse_args(argv)

    image = _read_image(args.image)
    corners = (
        load_points(args.points_file)
        if args.points_file
        else select_points(
            image,
            title="Pallet corners: TL, TR, BR, BL",
            labels=CORNER_LABELS,
            exact_count=4,
            max_display_size=args.max_preview_size,
        )
    )
    if corners.shape != (4, 2):
        raise PalletGeometryError("Exactly four pallet corners are required")
    detector = ManualPalletDetector(
        corners,
        args.pallet_type,
        image_resolution=(image.shape[1], image.shape[0]),
    )
    output_path = detector.save(args.output)
    visualization = Path(args.visualization) if args.visualization else output_path.with_name(
        f"{output_path.stem}_overlay.png"
    )
    visualization.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(visualization), _draw_pallet(image, corners)):
        raise OSError(f"Cannot write visualization: {visualization}")
    print(f"Saved pallet corners: {output_path}")
    print(f"Saved corner overlay: {visualization}")
    return 0


def _positive_or_none(value: float | None, name: str) -> float | None:
    if value is not None and value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _ground_truth(
    measurement_width: float,
    measurement_length: float,
    width_mm: float | None,
    length_mm: float | None,
) -> dict[str, float] | None:
    width = _positive_or_none(width_mm, "ground-truth width")
    length = _positive_or_none(length_mm, "ground-truth length")
    if width is None and length is None:
        return None
    result: dict[str, float] = {}
    if width is not None:
        result.update(
            width_mm=width,
            estimated_width_mm=measurement_width,
            width_error_percent=abs(measurement_width - width) / width * 100.0,
        )
    if length is not None:
        result.update(
            length_mm=length,
            estimated_length_mm=measurement_length,
            length_error_percent=abs(measurement_length - length) / length * 100.0,
        )
    return result


def _draw_object(
    pallet_overlay: np.ndarray,
    object_points: np.ndarray,
    width_mm: float,
    length_mm: float,
    area_mm2: float,
) -> np.ndarray:
    canvas = pallet_overlay.copy()
    contour = np.round(object_points).astype(np.int32)
    cv2.polylines(canvas, [contour], True, (40, 220, 40), 3, cv2.LINE_AA)
    center = tuple(np.round(object_points.mean(axis=0)).astype(int))
    lines = (f"W: {width_mm:.1f} mm  L: {length_mm:.1f} mm", f"Area: {area_mm2:.1f} mm2")
    for index, label in enumerate(lines):
        position = (center[0], center[1] + index * 24)
        cv2.putText(canvas, label, position, cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(canvas, label, position, cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (25, 25, 25), 1, cv2.LINE_AA)
    return canvas


def measure_object_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure a manually selected object on a calibrated pallet plane."
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--pallet-corners", required=True, help="JSON from select-pallet-corners")
    parser.add_argument("--calibration-dir", required=True)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--camera-group")
    parser.add_argument("--pallet-config", default="config/pallet_types.yaml")
    parser.add_argument("--selection-mode", choices=("polygon", "bbox"), default="polygon")
    parser.add_argument("--object-points", help="JSON point array; bypasses the interactive selector")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ground-truth-width-mm", type=float)
    parser.add_argument("--ground-truth-length-mm", type=float)
    parser.add_argument("--max-calibration-error", type=float, default=3.0)
    parser.add_argument("--rectified-pixels-per-mm", type=float, default=0.5)
    parser.add_argument(
        "--max-preview-size",
        type=int,
        default=900,
        help="Maximum polygon-selection preview width or height in pixels (default: 900)",
    )
    args = parser.parse_args(argv)

    image_path = Path(args.image)
    image = _read_image(image_path)
    calibration = CalibrationStore(args.calibration_dir).load(args.camera_id, args.camera_group)
    calibration.validate_image_resolution(image.shape)
    if calibration.reprojection_error > args.max_calibration_error:
        raise PalletGeometryError(
            f"Calibration error {calibration.reprojection_error:.3f}px exceeds allowed "
            f"{args.max_calibration_error:.3f}px"
        )

    detector = ManualPalletDetector.load(args.pallet_corners)
    detection = detector.detect(image)
    dimensions = PalletTypeConfig.load(args.pallet_config).get(detection.pallet_type)
    homography = compute_pallet_homography(detection.corners_px, dimensions, calibration)

    if args.object_points:
        object_points = load_points(args.object_points)
    elif args.selection_mode == "bbox":
        object_points = select_bounding_box(
            image,
            max_display_size=args.max_preview_size,
        )
    else:
        object_points = select_points(
            image,
            title="Select object polygon",
            minimum_count=3,
            max_display_size=args.max_preview_size,
        )
    if len(object_points) < 3:
        raise ValueError("At least three object points are required")

    measurement = measure_contour(1, object_points, 1.0, calibration, homography)
    object_points_mm = contour_to_pallet_mm(object_points, calibration, homography)
    truth = _ground_truth(
        measurement.width_mm,
        measurement.length_mm,
        args.ground_truth_width_mm,
        args.ground_truth_length_mm,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    pallet_overlay = _draw_pallet(image, detection.corners_px)
    object_overlay = _draw_object(
        pallet_overlay,
        object_points,
        measurement.width_mm,
        measurement.length_mm,
        measurement.area_mm2,
    )
    undistorted = cv2.undistort(
        image, calibration.camera_matrix, calibration.distortion_coefficients
    )
    rectified = rectify_pallet(undistorted, homography, args.rectified_pixels_per_mm)
    outputs = {
        "original": output_dir / f"{stem}_original.png",
        "pallet": output_dir / f"{stem}_pallet_corners.png",
        "measurement": output_dir / f"{stem}_object_measurement.png",
        "rectified": output_dir / f"{stem}_rectified_pallet.png",
    }
    for name, view in (
        ("original", image),
        ("pallet", pallet_overlay),
        ("measurement", object_overlay),
        ("rectified", rectified),
    ):
        if not cv2.imwrite(str(outputs[name]), view):
            raise OSError(f"Cannot write output: {outputs[name]}")

    payload = {
        "image": str(image_path),
        "camera_id": calibration.camera_id,
        "pallet_type": detection.pallet_type,
        "pallet_corners_px": detection.corners_px.tolist(),
        "object_points_px": object_points.tolist(),
        "object_points_mm": object_points_mm.tolist(),
        "image_to_pallet_homography": homography.image_to_pallet.tolist(),
        "measurement": measurement.to_dict(),
        "ground_truth": truth,
        "units": {"linear": "mm", "area": "mm2"},
    }
    result_path = output_dir / f"{stem}_measurement.json"
    result_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(
        f"Width: {measurement.width_mm:.2f} mm | Length: {measurement.length_mm:.2f} mm | "
        f"Area: {measurement.area_mm2:.2f} mm2"
    )
    if truth:
        if "width_error_percent" in truth:
            print(f"Width error: {truth['width_error_percent']:.2f}%")
        if "length_error_percent" in truth:
            print(f"Length error: {truth['length_error_percent']:.2f}%")
    print(f"Saved results: {result_path}")
    return 0
