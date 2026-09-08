from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from fruit_pipeline.camera_calibration.models import CameraCalibration

from .detector import PalletGeometryError, validate_corner_order
from .pallet_config import PalletDimensions


@dataclass(frozen=True)
class PalletHomography:
    image_to_pallet: np.ndarray
    undistorted_corners_px: np.ndarray
    dimensions: PalletDimensions

    def transform_points(self, points_px: np.ndarray) -> np.ndarray:
        points = np.asarray(points_px, dtype=np.float32).reshape(-1, 1, 2)
        result = cv2.perspectiveTransform(points, self.image_to_pallet).reshape(-1, 2)
        if not np.isfinite(result).all():
            raise PalletGeometryError("Homography produced non-finite pallet coordinates")
        return result


def undistort_points(points_px: np.ndarray, calibration: CameraCalibration) -> np.ndarray:
    points = np.asarray(points_px, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.undistortPoints(
        points, calibration.camera_matrix, calibration.distortion_coefficients,
        P=calibration.camera_matrix,
    ).reshape(-1, 2)


def compute_pallet_homography(
    corners_px: np.ndarray, dimensions: PalletDimensions, calibration: CameraCalibration,
) -> PalletHomography:
    corners = validate_corner_order(corners_px)
    undistorted = undistort_points(corners, calibration).astype(np.float32)
    destination = np.asarray(dimensions.corners_mm, dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(undistorted, destination)
    if not np.isfinite(matrix).all() or abs(float(np.linalg.det(matrix))) < 1e-12:
        raise PalletGeometryError("Degenerate pallet homography")
    condition = float(np.linalg.cond(matrix))
    if not np.isfinite(condition) or condition > 1e12:
        raise PalletGeometryError(f"Numerically unstable pallet homography (condition={condition:.3g})")
    projected = cv2.perspectiveTransform(undistorted.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    if float(np.max(np.linalg.norm(projected - destination, axis=1))) > 1e-2:
        raise PalletGeometryError("Pallet homography failed corner consistency validation")
    return PalletHomography(matrix, undistorted, dimensions)


def rectify_pallet(
    undistorted_image_bgr: np.ndarray, homography: PalletHomography, pixels_per_mm: float = 0.5,
) -> np.ndarray:
    if pixels_per_mm <= 0:
        raise ValueError("pixels_per_mm must be positive")
    scale = np.array([[pixels_per_mm, 0, 0], [0, pixels_per_mm, 0], [0, 0, 1]], dtype=np.float64)
    output_size = (
        max(1, int(round(homography.dimensions.width_mm * pixels_per_mm))),
        max(1, int(round(homography.dimensions.length_mm * pixels_per_mm))),
    )
    return cv2.warpPerspective(undistorted_image_bgr, scale @ homography.image_to_pallet, output_size)
