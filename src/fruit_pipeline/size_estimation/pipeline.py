from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

import cv2
import numpy as np

from fruit_pipeline.camera_calibration.calibration_store import CalibrationStore
from fruit_pipeline.camera_calibration.models import CameraCalibration
from fruit_pipeline.measurement.contour_transform import extract_primary_contour
from fruit_pipeline.measurement.fruit_measurement import FruitMeasurement, measure_fruit_mask
from fruit_pipeline.pallet_geometry.detector import PalletDetection, PalletDetector, PalletGeometryError
from fruit_pipeline.pallet_geometry.homography import PalletHomography, compute_pallet_homography, rectify_pallet
from fruit_pipeline.pallet_geometry.pallet_config import PalletTypeConfig

logger = logging.getLogger(__name__)


class SegmentedFruit(Protocol):
    instance_id: int
    detector_score: float
    sam_score: float
    mask: np.ndarray


@dataclass(frozen=True)
class SizeEstimationConfig:
    camera_id: str
    calibration_dir: str | Path
    pallet_config_path: str | Path
    camera_group: str | None = None
    min_pallet_confidence: float = 0.5
    max_calibration_error: float = 2.0
    debug: bool = False
    rectified_pixels_per_mm: float = 0.5


@dataclass
class SizeEstimationResult:
    calibration: CameraCalibration
    pallet_detection: PalletDetection
    homography: PalletHomography
    measurements: list[FruitMeasurement]
    debug_overlay: np.ndarray | None = None
    rectified_pallet: np.ndarray | None = None

    def save(self, output_dir: str | Path, stem: str = "size_estimation") -> None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        payload = {
            "camera_id": self.calibration.camera_id,
            "camera_group": self.calibration.camera_group,
            "pallet_type": self.pallet_detection.pallet_type,
            "pallet_confidence": self.pallet_detection.confidence,
            "pallet_corners_px": self.pallet_detection.corners_px.tolist(),
            "image_to_pallet_homography": self.homography.image_to_pallet.tolist(),
            "projection_plane": "pallet",
            "linear_units": "mm",
            "area_units": "mm2",
            "measurements": [measurement.to_dict() for measurement in self.measurements],
        }
        (destination / f"{stem}_measurements.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        if self.debug_overlay is not None:
            cv2.imwrite(str(destination / f"{stem}_measurement_debug.png"), self.debug_overlay)
        if self.rectified_pallet is not None:
            cv2.imwrite(str(destination / f"{stem}_rectified_pallet.png"), self.rectified_pallet)


class SizeEstimationPipeline:
    """Calibrated 2D measurement downstream of any instance segmenter."""

    def __init__(self, config: SizeEstimationConfig, pallet_detector: PalletDetector):
        self.config = config
        self.pallet_detector = pallet_detector
        self.calibration_store = CalibrationStore(config.calibration_dir)
        self.pallet_types = PalletTypeConfig.load(config.pallet_config_path)

    def run(self, image_bgr: np.ndarray, fruits: Iterable[SegmentedFruit]) -> SizeEstimationResult:
        calibration = self.calibration_store.load(self.config.camera_id, self.config.camera_group)
        calibration.validate_image_resolution(image_bgr.shape)
        if calibration.reprojection_error > self.config.max_calibration_error:
            raise PalletGeometryError(
                f"Calibration error {calibration.reprojection_error:.3f}px exceeds "
                f"allowed {self.config.max_calibration_error:.3f}px"
            )
        detection = self.pallet_detector.detect(image_bgr)
        if detection is None:
            raise PalletGeometryError("No pallet detected")
        if detection.confidence < self.config.min_pallet_confidence:
            raise PalletGeometryError(
                f"Pallet confidence {detection.confidence:.3f} is below "
                f"threshold {self.config.min_pallet_confidence:.3f}"
            )
        dimensions = self.pallet_types.get(detection.pallet_type)
        homography = compute_pallet_homography(detection.corners_px, dimensions, calibration)

        fruit_list = list(fruits)
        measurements: list[FruitMeasurement] = []
        for fruit in fruit_list:
            if np.asarray(fruit.mask).shape != image_bgr.shape[:2]:
                raise ValueError(
                    f"Fruit {fruit.instance_id} mask shape {np.asarray(fruit.mask).shape} "
                    f"does not match image shape {image_bgr.shape[:2]}"
                )
            confidence = min(float(fruit.detector_score), float(fruit.sam_score))
            try:
                measurements.append(
                    measure_fruit_mask(fruit.instance_id, fruit.mask, confidence, calibration, homography)
                )
            except PalletGeometryError as exc:
                logger.warning("Skipping unmeasurable fruit %s: %s", fruit.instance_id, exc)

        overlay = _draw_debug(image_bgr, detection, fruit_list, measurements) if self.config.debug else None
        rectified = (
            rectify_pallet(
                cv2.undistort(
                    image_bgr, calibration.camera_matrix, calibration.distortion_coefficients
                ),
                homography,
                self.config.rectified_pixels_per_mm,
            )
            if self.config.debug else None
        )
        return SizeEstimationResult(calibration, detection, homography, measurements, overlay, rectified)


def _draw_debug(
    image_bgr: np.ndarray, detection: PalletDetection,
    fruits: list[SegmentedFruit], measurements: list[FruitMeasurement],
) -> np.ndarray:
    canvas = image_bgr.copy()
    tint = np.zeros_like(canvas)
    for fruit in fruits:
        tint[np.asarray(fruit.mask, dtype=bool)] = (40, 180, 40)
    canvas = cv2.addWeighted(canvas, 1.0, tint, 0.35, 0)
    corners = np.round(detection.corners_px).astype(np.int32)
    cv2.polylines(canvas, [corners], True, (0, 220, 255), 3, cv2.LINE_AA)
    labels = {measurement.fruit_id: measurement for measurement in measurements}
    for fruit in fruits:
        measurement = labels.get(fruit.instance_id)
        if measurement is None:
            continue
        contour = extract_primary_contour(fruit.mask).astype(np.int32)
        cv2.polylines(canvas, [contour], True, (255, 120, 0), 2, cv2.LINE_AA)
        center = tuple(np.round(contour.mean(axis=0)).astype(int))
        text = f"{measurement.length_mm:.1f} x {measurement.width_mm:.1f} mm"
        cv2.putText(canvas, text, center, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, text, center, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1, cv2.LINE_AA)
    return canvas
