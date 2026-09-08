from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .detector import PalletGeometryError


@dataclass(frozen=True)
class PalletDimensions:
    width_mm: float
    length_mm: float

    def __post_init__(self) -> None:
        if self.width_mm <= 0 or self.length_mm <= 0:
            raise PalletGeometryError("Pallet width and length must be positive")

    @property
    def corners_mm(self) -> list[list[float]]:
        return [[0.0, 0.0], [self.width_mm, 0.0], [self.width_mm, self.length_mm], [0.0, self.length_mm]]


class PalletTypeConfig:
    def __init__(self, pallet_types: dict[str, PalletDimensions]):
        if not pallet_types:
            raise PalletGeometryError("At least one pallet type must be configured")
        self.pallet_types = dict(pallet_types)

    @classmethod
    def load(cls, path: str | Path) -> "PalletTypeConfig":
        try:
            data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
            raw_types = data["pallet_types"]
            types = {
                name: PalletDimensions(float(values["width_mm"]), float(values["length_mm"]))
                for name, values in raw_types.items()
            }
        except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
            raise PalletGeometryError(f"Invalid pallet configuration {path}: {exc}") from exc
        return cls(types)

    def get(self, pallet_type: str) -> PalletDimensions:
        try:
            return self.pallet_types[pallet_type]
        except KeyError as exc:
            raise PalletGeometryError(f"Unknown pallet type: {pallet_type!r}") from exc
