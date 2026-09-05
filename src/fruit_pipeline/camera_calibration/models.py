from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class CalibrationError(ValueError):
    """Raised when calibration data is absent, inconsistent, or unusable."""


@dataclass(frozen=True)
class CameraCalibration:
    camera_id: str | None
    camera_group: str | None
    resolution: tuple[int, int]  # width, height
    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    reprojection_error: float

    def __post_init__(self) -> None:
        matrix = np.asarray(self.camera_matrix, dtype=np.float64)
        distortion = np.asarray(self.distortion_coefficients, dtype=np.float64).reshape(-1)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise CalibrationError("camera_matrix must be a finite 3x3 matrix")
        if distortion.size < 4 or not np.isfinite(distortion).all():
            raise CalibrationError("distortion_coefficients must contain at least four finite values")
        if len(self.resolution) != 2 or min(self.resolution) <= 0:
            raise CalibrationError("resolution must be positive (width, height)")
        if not np.isfinite(self.reprojection_error) or self.reprojection_error < 0:
            raise CalibrationError("reprojection_error must be a finite non-negative number")
        object.__setattr__(self, "camera_matrix", matrix)
        object.__setattr__(self, "distortion_coefficients", distortion)
        object.__setattr__(self, "resolution", tuple(int(value) for value in self.resolution))

    @property
    def fx(self) -> float:
        return float(self.camera_matrix[0, 0])

    @property
    def fy(self) -> float:
        return float(self.camera_matrix[1, 1])

    @property
    def cx(self) -> float:
        return float(self.camera_matrix[0, 2])

    @property
    def cy(self) -> float:
        return float(self.camera_matrix[1, 2])

    def validate_image_resolution(self, image_shape: tuple[int, ...]) -> None:
        actual = (int(image_shape[1]), int(image_shape[0]))
        if actual != self.resolution:
            raise CalibrationError(
                f"Calibration resolution is {self.resolution[0]}x{self.resolution[1]}, "
                f"but image is {actual[0]}x{actual[1]}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "camera_group": self.camera_group,
            "resolution": list(self.resolution),
            "camera_matrix": self.camera_matrix.tolist(),
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "distortion_coefficients": self.distortion_coefficients.tolist(),
            "reprojection_error": float(self.reprojection_error),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CameraCalibration":
        required = {"resolution", "camera_matrix", "distortion_coefficients", "reprojection_error"}
        missing = required - data.keys()
        if missing:
            raise CalibrationError(f"Missing calibration field(s): {', '.join(sorted(missing))}")
        return cls(
            camera_id=data.get("camera_id"),
            camera_group=data.get("camera_group"),
            resolution=tuple(data["resolution"]),
            camera_matrix=np.asarray(data["camera_matrix"], dtype=np.float64),
            distortion_coefficients=np.asarray(data["distortion_coefficients"], dtype=np.float64),
            reprojection_error=float(data["reprojection_error"]),
        )
