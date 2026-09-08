from __future__ import annotations

import json
from pathlib import Path

from .models import CalibrationError, CameraCalibration


class CalibrationStore:
    """JSON store with camera-specific calibration taking priority over group data."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _path(self, scope: str, name: str) -> Path:
        if not name or Path(name).name != name:
            raise CalibrationError(f"Invalid {scope} name: {name!r}")
        return self.root / ("cameras" if scope == "camera" else "groups") / f"{name}.json"

    def save(self, calibration: CameraCalibration, *, as_group: bool = False) -> Path:
        name = calibration.camera_group if as_group else calibration.camera_id
        scope = "group" if as_group else "camera"
        if not name:
            raise CalibrationError(f"Cannot save {scope} calibration without {scope}_id")
        path = self._path(scope, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(calibration.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path

    def load(self, camera_id: str, camera_group: str | None = None) -> CameraCalibration:
        camera_path = self._path("camera", camera_id)
        if camera_path.is_file():
            return self._read(camera_path)
        if camera_group:
            group_path = self._path("group", camera_group)
            if group_path.is_file():
                return self._read(group_path)
        fallback = f" then group {camera_group!r}" if camera_group else ""
        raise CalibrationError(f"No calibration found for camera {camera_id!r}{fallback}")

    @staticmethod
    def _read(path: Path) -> CameraCalibration:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CalibrationError(f"Cannot read calibration {path}: {exc}") from exc
        return CameraCalibration.from_dict(payload)
