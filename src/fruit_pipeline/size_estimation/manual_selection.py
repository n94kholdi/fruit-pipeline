from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


def load_points(path: str | Path) -> np.ndarray:
    """Load points from a JSON list or a mapping containing ``points_px``."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        values = payload.get("points_px") if isinstance(payload, dict) else payload
        points = np.asarray(values, dtype=np.float32)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(f"Cannot load points from {path}: {exc}") from exc
    if points.ndim != 2 or points.shape[1:] != (2,) or len(points) < 1:
        raise ValueError(f"Points in {path} must be an Nx2 array")
    if not np.isfinite(points).all():
        raise ValueError(f"Points in {path} must be finite")
    return points


def select_points(
    image_bgr: np.ndarray,
    *,
    title: str,
    labels: Sequence[str] | None = None,
    exact_count: int | None = None,
    minimum_count: int = 3,
    max_display_size: int = 900,
) -> np.ndarray:
    """Interactively collect image points while retaining original coordinates."""
    if max_display_size <= 0:
        raise ValueError("max_display_size must be positive")
    height, width = image_bgr.shape[:2]
    scale = min(1.0, max_display_size / max(width, height))
    display_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    base = cv2.resize(image_bgr, display_size, interpolation=cv2.INTER_AREA)
    selected: list[tuple[float, float]] = []

    def on_mouse(event: int, x: int, y: int, _flags: int, _data: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and (exact_count is None or len(selected) < exact_count):
            selected.append((x / scale, y / scale))

    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(title, on_mouse)
    try:
        while True:
            canvas = base.copy()
            shown = np.asarray(selected, dtype=np.float32) * scale if selected else np.empty((0, 2))
            for index, point in enumerate(shown):
                center = tuple(np.round(point).astype(int))
                cv2.circle(canvas, center, 6, (0, 220, 255), -1, cv2.LINE_AA)
                label = labels[index] if labels and index < len(labels) else str(index + 1)
                cv2.putText(canvas, label, (center[0] + 8, center[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2, cv2.LINE_AA)
            if len(shown) > 1:
                cv2.polylines(canvas, [shown.astype(np.int32)], exact_count == len(shown),
                              (0, 220, 255), 2, cv2.LINE_AA)
            instruction = "Left-click points | Backspace/U undo | R reset | Enter accept | Esc cancel"
            cv2.putText(canvas, instruction, (12, 26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, instruction, (12, 26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (30, 30, 30), 1, cv2.LINE_AA)
            cv2.imshow(title, canvas)
            key = cv2.waitKey(20) & 0xFF
            if key in (8, 127, ord("u")) and selected:
                selected.pop()
            elif key == ord("r"):
                selected.clear()
            elif key in (10, 13):
                required = exact_count if exact_count is not None else minimum_count
                enough_points = (
                    len(selected) == required
                    if exact_count is not None
                    else len(selected) >= required
                )
                if enough_points:
                    return np.asarray(selected, dtype=np.float32)
            elif key == 27:
                raise RuntimeError("Point selection cancelled")
    finally:
        cv2.destroyWindow(title)


def select_bounding_box(
    image_bgr: np.ndarray,
    title: str = "Select object",
    max_display_size: int = 900,
) -> np.ndarray:
    if max_display_size <= 0:
        raise ValueError("max_display_size must be positive")
    image_height, image_width = image_bgr.shape[:2]
    scale = min(1.0, max_display_size / max(image_width, image_height))
    preview = cv2.resize(
        image_bgr,
        (max(1, round(image_width * scale)), max(1, round(image_height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    x, y, width, height = cv2.selectROI(title, preview, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(title)
    if width <= 0 or height <= 0:
        raise RuntimeError("Bounding-box selection cancelled")
    return np.asarray(
        [
            [x / scale, y / scale],
            [(x + width) / scale, y / scale],
            [(x + width) / scale, (y + height) / scale],
            [x / scale, (y + height) / scale],
        ],
        dtype=np.float32,
    )
