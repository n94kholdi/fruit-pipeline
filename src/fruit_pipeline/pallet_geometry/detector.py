from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
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


class ManualPalletDetector:
    """Pallet provider backed by user-selected or previously saved corners.

    It intentionally implements the same ``PalletDetector`` protocol as a
    future keypoint model.  Consumers therefore do not need a manual-mode
    branch when the source of the corners changes.
    """

    def __init__(
        self,
        corners_px: np.ndarray,
        pallet_type: str,
        *,
        image_resolution: tuple[int, int] | None = None,
    ) -> None:
        self._detection = PalletDetection(corners_px, 1.0, pallet_type)
        self.image_resolution = image_resolution

    def detect(self, image_bgr: np.ndarray) -> PalletDetection:
        if image_bgr.ndim < 2:
            raise PalletGeometryError("Image must have at least two dimensions")
        actual = (int(image_bgr.shape[1]), int(image_bgr.shape[0]))
        if self.image_resolution is not None and actual != self.image_resolution:
            raise PalletGeometryError(
                f"Saved pallet corners are for {self.image_resolution[0]}x"
                f"{self.image_resolution[1]}, but image is {actual[0]}x{actual[1]}"
            )
        return self._detection

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, object] = {
            "source": "manual",
            "corner_order": ["top_left", "top_right", "bottom_right", "bottom_left"],
            "pallet_type": self._detection.pallet_type,
            "confidence": self._detection.confidence,
            "corners_px": self._detection.corners_px.tolist(),
        }
        if self.image_resolution is not None:
            payload["image_resolution"] = list(self.image_resolution)
        destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "ManualPalletDetector":
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            resolution = payload.get("image_resolution")
            return cls(
                np.asarray(payload["corners_px"], dtype=np.float32),
                str(payload["pallet_type"]),
                image_resolution=tuple(int(value) for value in resolution) if resolution else None,
            )
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise PalletGeometryError(f"Invalid manual pallet selection {path}: {exc}") from exc
