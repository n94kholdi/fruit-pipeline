from __future__ import annotations

import cv2
import numpy as np

from fruit_pipeline.camera_calibration.models import CameraCalibration
from fruit_pipeline.pallet_geometry.detector import PalletGeometryError
from fruit_pipeline.pallet_geometry.homography import PalletHomography, undistort_points


def extract_primary_contour(mask: np.ndarray) -> np.ndarray:
    """Extract the largest external fruit contour as an Nx2 pixel array."""
    if mask.ndim != 2:
        raise ValueError("Fruit mask must be a two-dimensional array")
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [contour for contour in contours if len(contour) >= 3 and cv2.contourArea(contour) > 0]
    if not contours:
        raise PalletGeometryError("Fruit mask has no measurable contour")
    return max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)


def contour_to_pallet_mm(
    contour_px: np.ndarray, calibration: CameraCalibration, homography: PalletHomography,
) -> np.ndarray:
    """Undistort raw image contour points, then map them onto the pallet plane."""
    undistorted = undistort_points(contour_px, calibration)
    return homography.transform_points(undistorted)
