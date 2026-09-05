"""Low-overhead live progress and preview reporting for fruit-analysis jobs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any, Mapping

import cv2


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class FruitLiveReporter:
    """Publish one event and annotated JPEG after each processed sample."""

    def __init__(
        self,
        directory: str | Path,
        job_id: str,
        *,
        preview_width: int = 1280,
        jpeg_quality: int = 78,
    ) -> None:
        self.directory = Path(directory).expanduser().resolve()
        self.job_id = job_id
        self.preview_width = max(320, preview_width)
        self.jpeg_quality = min(90, max(35, jpeg_quality))
        self.started = monotonic()
        self.total_fruit_observations = 0
        self.directory.mkdir(parents=True, exist_ok=True)

    def publish_frame(
        self,
        frame: Any,
        *,
        frame_index: int | None,
        timestamp_ms: float | None,
        processed_frame_count: int,
        total_sampled_frames: int | None,
        num_fruits: int,
        num_measured_fruits: int,
    ) -> dict[str, object]:
        self.total_fruit_observations += num_fruits
        preview_reference = self._write_preview(frame)
        progress = (
            min(100.0, processed_frame_count * 100.0 / total_sampled_frames)
            if total_sampled_frames
            else None
        )
        return self.emit(
            "preview_updated",
            status="running",
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            progress=progress,
            preview_reference=preview_reference,
            metrics={
                "processed_frame_count": processed_frame_count,
                "total_sampled_frames": total_sampled_frames,
                "num_fruits": num_fruits,
                "num_measured_fruits": num_measured_fruits,
                "total_fruit_observations": self.total_fruit_observations,
            },
        )

    def emit(
        self,
        event_type: str,
        *,
        status: str,
        frame_index: int | None = None,
        timestamp_ms: float | None = None,
        progress: float | None = None,
        preview_reference: str | None = None,
        metrics: Mapping[str, object] | None = None,
        message: str | None = None,
    ) -> dict[str, object]:
        previous = self._read_state()
        payload: dict[str, object] = {
            "type": event_type,
            "job_id": self.job_id,
            "timestamp": utc_now(),
            "status": status,
            "frame_index": frame_index if frame_index is not None else previous.get("frame_index"),
            "timestamp_ms": timestamp_ms if timestamp_ms is not None else previous.get("timestamp_ms"),
            "progress": progress if progress is not None else previous.get("progress"),
            "elapsed_seconds": monotonic() - self.started,
            "metrics": dict(metrics) if metrics is not None else previous.get("metrics", {}),
            "preview_reference": preview_reference or previous.get("preview_reference"),
            "message": message,
        }
        self._append_json("events.jsonl", payload)
        self._write_json("live_state.json", payload)
        return payload

    def _read_state(self) -> dict[str, object]:
        path = self.directory / "live_state.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    def _write_preview(self, frame: Any) -> str:
        height, width = frame.shape[:2]
        preview = frame
        if width > self.preview_width:
            scale = self.preview_width / width
            preview = cv2.resize(
                frame,
                (self.preview_width, max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        ok, encoded = cv2.imencode(
            ".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        )
        if not ok:
            raise RuntimeError("OpenCV could not encode the fruit preview frame")
        temporary = self.directory / "preview.jpg.tmp"
        final = self.directory / "preview.jpg"
        temporary.write_bytes(encoded.tobytes())
        temporary.replace(final)
        return f"/api/v1/jobs/{self.job_id}/preview"

    def _append_json(self, filename: str, payload: Mapping[str, object]) -> None:
        with (self.directory / filename).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, separators=(",", ":")) + "\n")
            stream.flush()

    def _write_json(self, filename: str, payload: Mapping[str, object]) -> None:
        destination = self.directory / filename
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(destination)
