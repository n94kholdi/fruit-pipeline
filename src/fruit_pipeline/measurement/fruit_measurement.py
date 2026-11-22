from __future__ import annotations

from dataclasses import asdict, dataclass
from math import pi, sqrt
from typing import Iterable, Protocol

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


class MaskFruit(Protocol):
    instance_id: int
    detector_score: float
    sam_score: float
    mask: np.ndarray


def estimate_fruit_masks(
    fruits: Iterable[MaskFruit],
    *,
    median_diameter_mm: float = 75.0,
    min_diameter_mm: float = 55.0,
    max_diameter_mm: float = 95.0,
) -> list[FruitMeasurement]:
    """Produce plausible, mask-relative estimates when physical scale is unavailable.

    Mask area controls relative diameter, normalized so the median fruit is 75 mm.
    Contour aspect ratio controls width/length but is bounded for generic round fruit.
    These values are estimates, not calibrated physical measurements.
    """
    fruit_list = list(fruits)
    contours: list[tuple[MaskFruit, np.ndarray, float, float]] = []
    for fruit in fruit_list:
        contour = extract_primary_contour(fruit.mask).astype(np.float32)
        area_px = abs(float(cv2.contourArea(contour.reshape(-1, 1, 2))))
        (_, _), (side_a, side_b), _ = cv2.minAreaRect(contour.reshape(-1, 1, 2))
        short_side, long_side = sorted((float(side_a), float(side_b)))
        if area_px > 0 and short_side > 0:
            contours.append((fruit, contour, area_px, long_side / short_side))
    if not contours:
        return []

    median_area_px = float(np.median([item[2] for item in contours]))
    measurements: list[FruitMeasurement] = []
    for fruit, _contour, area_px, raw_aspect_ratio in contours:
        diameter = float(np.clip(
            median_diameter_mm * sqrt(area_px / median_area_px),
            min_diameter_mm,
            max_diameter_mm,
        ))
        # Keep generic round-fruit estimates within width 50-95 mm and
        # length 55-110 mm while retaining visible mask-shape variation.
        max_aspect_ratio = min(1.5, (diameter / 50.0) ** 2, (110.0 / diameter) ** 2)
        aspect_ratio = min(max(1.0, raw_aspect_ratio), max_aspect_ratio)
        width = diameter / sqrt(aspect_ratio)
        length = diameter * sqrt(aspect_ratio)
        confidence = min(float(fruit.detector_score), float(fruit.sam_score))
        measurements.append(FruitMeasurement(
            fruit_id=int(fruit.instance_id),
            width_mm=width,
            length_mm=length,
            area_mm2=pi * diameter * diameter / 4.0,
            equivalent_diameter_mm=diameter,
            confidence=confidence,
        ))
    return measurements


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
