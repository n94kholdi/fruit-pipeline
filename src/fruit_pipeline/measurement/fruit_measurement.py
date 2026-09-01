from __future__ import annotations

from dataclasses import asdict, dataclass
from math import pi, sqrt

import cv2
import numpy as np

from fruit_pipeline.camera_calibration.models import CameraCalibration
from fruit_pipeline.pallet_geometry.homography import PalletHomography

from .contour_transform import contour_to_pallet_mm, extract_primary_contour


@dataclass(frozen=True)
class FruitMeasurement:
    fruit_id: int
    width_mm: float
    length_mm: float
    area_mm2: float
    equivalent_diameter_mm: float
    confidence: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def measure_fruit_mask(
    fruit_id: int, mask: np.ndarray, confidence: float,
    calibration: CameraCalibration, homography: PalletHomography,
) -> FruitMeasurement:
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("Fruit confidence must be between 0 and 1")
    contour_px = extract_primary_contour(mask)
    return measure_contour(fruit_id, contour_px, confidence, calibration, homography)


def measure_contour(
    object_id: int,
    contour_px: np.ndarray,
    confidence: float,
    calibration: CameraCalibration,
    homography: PalletHomography,
) -> FruitMeasurement:
    """Measure any image-space contour on the calibrated pallet plane."""
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("Object confidence must be between 0 and 1")
    contour = np.asarray(contour_px, dtype=np.float32)
    if contour.ndim != 2 or contour.shape[1:] != (2,) or len(contour) < 3:
        raise ValueError("Object contour must contain at least three (x, y) points")
    if not np.isfinite(contour).all():
        raise ValueError("Object contour points must be finite")
    contour_mm = contour_to_pallet_mm(contour, calibration, homography).astype(np.float32)
    (_, _), (side_a, side_b), _ = cv2.minAreaRect(contour_mm.reshape(-1, 1, 2))
    width, length = sorted((float(side_a), float(side_b)))
    area = abs(float(cv2.contourArea(contour_mm.reshape(-1, 1, 2))))
    equivalent_diameter = sqrt(4.0 * area / pi)
    return FruitMeasurement(
        fruit_id=int(object_id), width_mm=width, length_mm=length, area_mm2=area,
        equivalent_diameter_mm=equivalent_diameter, confidence=float(confidence),
    )
