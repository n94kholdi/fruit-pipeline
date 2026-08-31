from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


class PalletGeometryError(ValueError):
    """Raised for invalid corners, dimensions, or homographies."""


def validate_corner_order(corners: np.ndarray) -> np.ndarray:
    """Validate TL, TR, BR, BL ordering without silently guessing keypoint labels."""
    points = np.asarray(corners, dtype=np.float32)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise PalletGeometryError("Pallet corners must be a finite 4x2 array ordered TL, TR, BR, BL")
    if min(np.linalg.norm(points[i] - points[j]) for i in range(1, 4) for j in range(i)) < 1.0:
        raise PalletGeometryError("Pallet corners contain duplicate or near-duplicate points")
    cross = []
    for index in range(4):
        a, b, c = points[index], points[(index + 1) % 4], points[(index + 2) % 4]
        first, second = b - a, c - b
        cross.append(float(first[0] * second[1] - first[1] * second[0]))
    if not (all(value > 0 for value in cross) or all(value < 0 for value in cross)):
        raise PalletGeometryError("Pallet corners are crossed, concave, or incorrectly ordered")
    # In image coordinates, TL->TR->BR->BL is clockwise and has positive shoelace area.
    signed_area = 0.5 * float(
        np.dot(points[:, 0], np.roll(points[:, 1], -1))
        - np.dot(points[:, 1], np.roll(points[:, 0], -1))
    )
    if signed_area <= 1.0:
        raise PalletGeometryError("Pallet corners must be ordered TL, TR, BR, BL with non-zero area")
    return points


@dataclass(frozen=True)
class PalletDetection:
    corners_px: np.ndarray
    confidence: float
    pallet_type: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "corners_px", validate_corner_order(self.corners_px))
        if not 0.0 <= self.confidence <= 1.0:
            raise PalletGeometryError("Pallet confidence must be between 0 and 1")
        if not self.pallet_type:
            raise PalletGeometryError("Pallet type cannot be empty")


@runtime_checkable
class PalletDetector(Protocol):
    """Replaceable interface for a four-keypoint pallet model."""

    def detect(self, image_bgr: np.ndarray) -> PalletDetection | None:
        ...
