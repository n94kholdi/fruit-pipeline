"""HTTP adapter for camera calibration and dashboard fruit-analysis jobs."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Iterator, Literal
from urllib.parse import urlsplit

import cv2
from fastapi import FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from fruit_pipeline.camera_calibration.calibrate import BoardSpec
from fruit_pipeline.camera_calibration.calibration_store import CalibrationStore
from fruit_pipeline.camera_calibration.models import CalibrationError
from fruit_pipeline.integrated_pipeline import media_source_stem, normalize_to_resolution
from fruit_pipeline.live import FruitLiveReporter
from fruit_pipeline.pallet_geometry.pallet_config import PalletTypeConfig


DATA_DIR = Path(os.getenv("FRUIT_PIPELINE_DATA_DIR", "outputs/dashboard")).resolve()
CALIBRATION_DIR = DATA_DIR / "calibrations"
INPUT_DIR = DATA_DIR / "inputs"
JOB_DIR = DATA_DIR / "jobs"
PALLET_CONFIG = Path(os.getenv("FRUIT_PIPELINE_PALLET_CONFIG", "config/pallet_types.yaml")).resolve()
DETECTOR_WEIGHTS = os.getenv("FRUIT_PIPELINE_DETECTOR_WEIGHTS", "models/yolo11x.pt")
SAM_CHECKPOINT = os.getenv("FRUIT_PIPELINE_SAM_CHECKPOINT", "models/sam_vit_l_0b3195.pth")
DEVICE = os.getenv("FRUIT_PIPELINE_DEVICE", "cpu")
MAX_UPLOAD_BYTES = int(os.getenv("FRUIT_PIPELINE_MAX_UPLOAD_BYTES", str(1024**3)))
WORKERS = max(1, int(os.getenv("FRUIT_PIPELINE_JOB_WORKERS", "1")))

for directory in (CALIBRATION_DIR, INPUT_DIR, JOB_DIR):
    directory.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Tarebar Fruit Pipeline API", version="1.0.0")
origins = [item.strip() for item in os.getenv("FRUIT_PIPELINE_CORS_ORIGINS", "http://localhost:3000").split(",") if item.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=WORKERS, thread_name_prefix="fruit-dashboard")
jobs: dict[str, dict[str, object]] = {}
jobs_lock = threading.Lock()


class Point(BaseModel):
    x: float = Field(ge=0)
    y: float = Field(ge=0)


class FruitJobRequest(BaseModel):
    input_id: str
    camera_id: str
    pallet_type: str = "standard_large"
    corners: list[Point] = Field(min_length=4, max_length=4)
    max_calibration_error: float = Field(default=3.0, gt=0)
    frame_step: int = Field(default=10, ge=1)
    min_pallet_overlap: float = Field(default=0.5, ge=0, le=1)
    resize_to_calibration: bool = True
    allow_unsafe_resize: bool = False
    max_frames: int | None = Field(default=None, ge=1)


class StreamInputRequest(BaseModel):
    camera_id: str
    stream_url: str
    allow_unsafe_resize: bool = False


def _safe_name(value: str, label: str) -> str:
    if not value or Path(value).name != value or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in value):
        raise HTTPException(422, f"Invalid {label}")
    return value


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_job(job_id: str, **changes: object) -> dict[str, object]:
    with jobs_lock:
        current = jobs.setdefault(job_id, {"id": job_id, "created_at": _now()})
        current.update(changes, updated_at=_now())
        snapshot = dict(current)
    destination = JOB_DIR / job_id
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "job.json").write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    return snapshot


def _job(job_id: str) -> dict[str, object]:
    with jobs_lock:
        record = jobs.get(job_id)
    if record is not None:
        return _with_live_state(job_id, dict(record))
    path = JOB_DIR / job_id / "job.json"
    if path.is_file():
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            record = None
    if record is None:
        raise HTTPException(404, "Job not found")
    return _with_live_state(job_id, record)


def _with_live_state(job_id: str, record: dict[str, object]) -> dict[str, object]:
    path = JOB_DIR / job_id / "live_state.json"
    try:
        live = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        live = None
    if isinstance(live, dict):
        record["live"] = live
    return record


def _save_upload(upload: UploadFile, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with destination.open("wb") as output:
        while chunk := upload.file.read(1024 * 1024):
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                output.close()
                destination.unlink(missing_ok=True)
                raise HTTPException(413, "Uploaded file is too large")
            output.write(chunk)


def _run_calibration(
    job_id: str,
    camera_id: str,
    camera_group: str | None,
    source: Path,
    board: BoardSpec,
    frame_step: int,
    min_frames: int,
    max_error: float,
) -> None:
    _write_job(job_id, status="running")
    detection_dir = JOB_DIR / job_id / "detections"
    command = [
        sys.executable, "-m", "fruit_pipeline.camera_calibration.calibrate",
        "--camera-id", camera_id,
        "--images", str(source),
        "--output-dir", str(CALIBRATION_DIR),
        "--detection-output-dir", str(detection_dir),
        "--board", board.kind,
        "--columns", str(board.columns),
        "--rows", str(board.rows),
        "--square-size-mm", str(board.square_size_mm),
        "--marker-size-mm", str(board.marker_size_mm),
        "--dictionary", board.dictionary,
        "--frame-step", str(frame_step),
        "--min-frames", str(min_frames),
        "--max-reprojection-error", str(max_error),
        "-v",
    ]
    if camera_group:
        command.extend(["--camera-group", camera_group])
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        log = (completed.stdout + "\n" + completed.stderr).strip()
        (JOB_DIR / job_id / "calibration.log").write_text(log + "\n", encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(log[-4000:] or f"Calibration exited with code {completed.returncode}")
        calibration = CalibrationStore(CALIBRATION_DIR).load(camera_id, camera_group)
        calibration_path = CALIBRATION_DIR / "cameras" / f"{camera_id}.json"
        _write_job(
            job_id,
            status="completed",
            calibration=calibration.to_dict(),
            calibration_path=str(calibration_path.relative_to(CALIBRATION_DIR)),
            error=None,
        )
    except Exception as exc:  # worker boundary: expose a useful failed state
        _write_job(job_id, status="failed", error=str(exc))


def _read_media_first_frame(source: str | Path):
    source_text = str(source)
    image = None if "://" in source_text else cv2.imread(source_text, cv2.IMREAD_COLOR)
    if image is not None:
        return image
    capture = cv2.VideoCapture(source_text)
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise HTTPException(422, "Could not read the image, video, or live stream")
    return frame


def _artifact_url(job_id: str, path: Path) -> str:
    relative = path.relative_to(JOB_DIR / job_id)
    return f"/api/v1/jobs/{job_id}/artifacts/{relative.as_posix()}"


def _result_payload(job_id: str, output_dir: Path, source: str | Path) -> dict[str, object]:
    summary_path = output_dir / f"{media_source_stem(source)}_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if "://" in str(source):
        # Do not expose camera credentials that may be embedded in the stream URL.
        summary["source"] = "live-stream"
    frames = summary.get("frames", [])
    for frame in frames:
        artifact_dir = Path(frame["artifact_dir"])
        stem = Path(frame["source_image"]).stem
        overlay = artifact_dir / f"{stem}_measurement_debug.png"
        if overlay.is_file():
            frame["measurement_overlay_url"] = _artifact_url(job_id, overlay)
    summary["summary_url"] = _artifact_url(job_id, summary_path)
    return summary


def _run_fruit_job(job_id: str, request: FruitJobRequest, source: str | Path) -> None:
    _write_job(job_id, status="running")
    job_dir = JOB_DIR / job_id
    reporter = FruitLiveReporter(job_dir, job_id)
    reporter.emit(
        "job_started",
        status="running",
        message="Loading detection and segmentation models.",
    )
    output_dir = JOB_DIR / job_id / "results"
    points_path = JOB_DIR / job_id / "pallet_points.json"
    points_path.write_text(
        json.dumps([[point.x, point.y] for point in request.corners], indent=2) + "\n",
        encoding="utf-8",
    )
    command = [
        sys.executable, "-m", "fruit_pipeline.integrated_cli",
        "--image", str(source),
        "--output_dir", str(output_dir),
        "--camera-id", request.camera_id,
        "--calibration-dir", str(CALIBRATION_DIR),
        "--pallet-type", request.pallet_type,
        "--pallet-config", str(PALLET_CONFIG),
        "--pallet-points-file", str(points_path),
        "--max-calibration-error", str(request.max_calibration_error),
        "--frame-step", str(request.frame_step),
        "--min-pallet-overlap", str(request.min_pallet_overlap),
        "--detector-weights", DETECTOR_WEIGHTS,
        "--sam-checkpoint", SAM_CHECKPOINT,
        "--device", DEVICE,
        "--sam-batch-size", "1",
        "--tile-size-k", "8",
        "--min-tile-size", "320",
        "--max-tile-size", "2048",
        "--max-tiles", "12",
        "--overlap-ratio", "0.15",
        "--nms-metric", "diou",
        "--merge-iou-threshold", "0.5",
        "--containment-threshold", "0",
        "--conf-threshold", "0.05",
        "--live-job-dir", str(job_dir),
        "--live-job-id", job_id,
        "-v",
    ]
    if request.resize_to_calibration:
        command.append("--resize-to-calibration")
    if request.allow_unsafe_resize:
        command.append("--allow-unsafe-resize")
    if request.max_frames is not None:
        command.extend(["--max-frames", str(request.max_frames)])
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        log = (completed.stdout + "\n" + completed.stderr).strip()
        (JOB_DIR / job_id / "pipeline.log").write_text(log + "\n", encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(log[-4000:] or f"Pipeline exited with code {completed.returncode}")
        result = _result_payload(job_id, output_dir, source)
        _write_job(job_id, status="completed", result=result, error=None)
        reporter.emit("job_completed", status="completed", progress=100.0)
    except Exception as exc:
        _write_job(job_id, status="failed", error=str(exc))
        reporter.emit("job_failed", status="failed", message=str(exc))


@app.get("/health")
def health() -> dict[str, object]:
    detector_exists = Path(DETECTOR_WEIGHTS).is_file()
    sam_exists = Path(SAM_CHECKPOINT).is_file()
    return {
        "status": "ok",
        "models_ready": detector_exists and sam_exists,
        "models": {
            "detector": detector_exists,
            "sam": sam_exists,
        },
    }


@app.get("/api/v1/pallet-types")
def pallet_types() -> dict[str, object]:
    config = PalletTypeConfig.load(PALLET_CONFIG)
    return {"data": sorted(config.pallet_types.keys())}


@app.post("/api/v1/calibrations", status_code=202)
def create_calibration(
    files: Annotated[list[UploadFile], File()],
    camera_id: Annotated[str, Form()],
    camera_group: Annotated[str | None, Form()] = None,
    board: Annotated[Literal["checkerboard", "charuco"], Form()] = "charuco",
    columns: Annotated[int, Form(ge=2)] = 11,
    rows: Annotated[int, Form(ge=2)] = 8,
    square_size_mm: Annotated[float, Form(gt=0)] = 20,
    marker_size_mm: Annotated[float, Form(gt=0)] = 15,
    dictionary: Annotated[str, Form()] = "DICT_5X5_50",
    frame_step: Annotated[int, Form(ge=1)] = 10,
    min_frames: Annotated[int, Form(ge=3)] = 8,
    max_reprojection_error: Annotated[float, Form(gt=0)] = 2.5,
) -> dict[str, object]:
    camera_id = _safe_name(camera_id, "camera id")
    camera_group = _safe_name(camera_group, "camera group") if camera_group else None
    if not files:
        raise HTTPException(422, "At least one image or video is required")
    job_id = uuid.uuid4().hex
    upload_dir = JOB_DIR / job_id / "uploads"
    saved: list[Path] = []
    for index, upload in enumerate(files):
        suffix = Path(upload.filename or "capture").suffix.lower()
        destination = upload_dir / f"{index:04d}{suffix}"
        _save_upload(upload, destination)
        saved.append(destination)
    source = saved[0] if len(saved) == 1 else upload_dir
    board_spec = BoardSpec(board, columns, rows, square_size_mm, marker_size_mm, dictionary)
    _write_job(job_id, kind="calibration", status="queued", camera_id=camera_id)
    executor.submit(
        _run_calibration, job_id, camera_id, camera_group, source, board_spec,
        frame_step, min_frames, max_reprojection_error,
    )
    return {"data": _job(job_id)}


@app.post("/api/v1/inputs", status_code=201)
def create_input(
    file: Annotated[UploadFile, File()],
    camera_id: Annotated[str, Form()],
    allow_unsafe_resize: Annotated[bool, Form()] = False,
) -> dict[str, object]:
    camera_id = _safe_name(camera_id, "camera id")
    try:
        calibration = CalibrationStore(CALIBRATION_DIR).load(camera_id)
    except CalibrationError as exc:
        raise HTTPException(404, str(exc)) from exc
    input_id = uuid.uuid4().hex
    suffix = Path(file.filename or "input").suffix.lower()
    source = INPUT_DIR / input_id / f"source{suffix}"
    _save_upload(file, source)
    frame = _read_media_first_frame(source)
    try:
        normalized, _ = normalize_to_resolution(
            frame,
            calibration.resolution,
            rotation="auto",
            allow_aspect_mismatch=allow_unsafe_resize,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    preview = source.with_name("preview.jpg")
    if not cv2.imwrite(str(preview), normalized):
        raise HTTPException(500, "Could not create the pallet selection preview")
    return {
        "data": {
            "id": input_id,
            "filename": file.filename,
            "camera_id": camera_id,
            "width": normalized.shape[1],
            "height": normalized.shape[0],
            "preview_url": f"/api/v1/inputs/{input_id}/preview",
            "allow_unsafe_resize": allow_unsafe_resize,
        }
    }


@app.post("/api/v1/stream-inputs", status_code=201)
def create_stream_input(request: StreamInputRequest) -> dict[str, object]:
    camera_id = _safe_name(request.camera_id, "camera id")
    parsed_url = urlsplit(request.stream_url)
    if parsed_url.scheme.lower() not in {"rtsp", "rtsps", "http", "https"} or not parsed_url.netloc:
        raise HTTPException(422, "A valid RTSP or HTTP camera stream URL is required")
    try:
        calibration = CalibrationStore(CALIBRATION_DIR).load(camera_id)
    except CalibrationError as exc:
        raise HTTPException(404, str(exc)) from exc

    frame = _read_media_first_frame(request.stream_url)
    try:
        normalized, _ = normalize_to_resolution(
            frame,
            calibration.resolution,
            rotation="auto",
            allow_aspect_mismatch=request.allow_unsafe_resize,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    input_id = uuid.uuid4().hex
    input_dir = INPUT_DIR / input_id
    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "stream.json").write_text(
        json.dumps({"camera_id": camera_id, "stream_url": request.stream_url}, indent=2) + "\n",
        encoding="utf-8",
    )
    preview = input_dir / "preview.jpg"
    if not cv2.imwrite(str(preview), normalized):
        raise HTTPException(500, "Could not create the pallet selection preview")
    return {
        "data": {
            "id": input_id,
            "filename": "live-stream",
            "camera_id": camera_id,
            "width": normalized.shape[1],
            "height": normalized.shape[0],
            "preview_url": f"/api/v1/inputs/{input_id}/preview",
            "allow_unsafe_resize": request.allow_unsafe_resize,
            "source_type": "stream",
        }
    }


@app.get("/api/v1/inputs/{input_id}/preview")
def input_preview(input_id: str):
    _safe_name(input_id, "input id")
    path = INPUT_DIR / input_id / "preview.jpg"
    if not path.is_file():
        raise HTTPException(404, "Input preview not found")
    return FileResponse(path, media_type="image/jpeg")


@app.post("/api/v1/jobs", status_code=202)
def create_fruit_job(request: FruitJobRequest) -> dict[str, object]:
    input_id = _safe_name(request.input_id, "input id")
    camera_id = _safe_name(request.camera_id, "camera id")
    input_folder = INPUT_DIR / input_id
    sources: list[str | Path] = [path for path in input_folder.glob("source.*") if path.is_file()]
    stream_metadata_path = input_folder / "stream.json"
    source_type = "file"
    if not sources and stream_metadata_path.is_file():
        try:
            stream_metadata = json.loads(stream_metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(500, "Live stream input metadata is invalid") from exc
        if stream_metadata.get("camera_id") != camera_id:
            raise HTTPException(409, "The selected calibration does not match the live camera")
        stream_url = stream_metadata.get("stream_url")
        if not isinstance(stream_url, str):
            raise HTTPException(500, "Live stream input metadata is invalid")
        sources = [stream_url]
        source_type = "stream"
    if len(sources) != 1:
        raise HTTPException(404, "Uploaded or live input not found")
    try:
        CalibrationStore(CALIBRATION_DIR).load(camera_id)
    except CalibrationError as exc:
        raise HTTPException(404, str(exc)) from exc
    missing_models = [
        path for path in (DETECTOR_WEIGHTS, SAM_CHECKPOINT) if not Path(path).is_file()
    ]
    if missing_models:
        raise HTTPException(
            503,
            "Required model files are not mounted: " + ", ".join(missing_models),
        )
    request.camera_id = camera_id
    if source_type == "stream" and request.max_frames is None:
        request.max_frames = 100
    job_id = uuid.uuid4().hex
    _write_job(job_id, kind="fruit_analysis", source_type=source_type, status="queued", camera_id=camera_id)
    executor.submit(_run_fruit_job, job_id, request, sources[0])
    return {"data": _job(job_id)}


@app.get("/api/v1/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, object]:
    return {"data": _job(_safe_name(job_id, "job id"))}


@app.get("/api/v1/jobs/{job_id}/events")
def stream_job_events(
    job_id: str,
    last_event_id: Annotated[str | None, Header(alias="Last-Event-ID")] = None,
    after: Annotated[int | None, Query(ge=0)] = None,
) -> StreamingResponse:
    job_id = _safe_name(job_id, "job id")
    _job(job_id)
    cursor = after or 0
    if last_event_id and last_event_id.isdigit():
        cursor = max(cursor, int(last_event_id))
    return StreamingResponse(
        _event_stream(job_id, cursor),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/v1/jobs/{job_id}/preview")
def job_preview(job_id: str) -> FileResponse:
    job_id = _safe_name(job_id, "job id")
    _job(job_id)
    path = JOB_DIR / job_id / "preview.jpg"
    if not path.is_file():
        raise HTTPException(404, "Fruit preview not available yet")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/v1/jobs/{job_id}/preview-stream")
def job_preview_stream(job_id: str) -> StreamingResponse:
    job_id = _safe_name(job_id, "job id")
    _job(job_id)
    return StreamingResponse(
        _preview_stream(job_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


def _event_stream(job_id: str, cursor: int) -> Iterator[str]:
    """Tail live job events with resumable SSE line-number identifiers."""

    idle_started = time.monotonic()
    stream = None
    line_number = 0
    terminal_seen_at: float | None = None
    try:
        while True:
            record = _job(job_id)
            path = JOB_DIR / job_id / "events.jsonl"
            if stream is None and path.is_file():
                try:
                    stream = path.open("r", encoding="utf-8")
                    while line_number < cursor and stream.readline():
                        line_number += 1
                except OSError:
                    stream = None
            line = stream.readline() if stream is not None else ""
            if line:
                line_number += 1
                try:
                    event_type = str(json.loads(line).get("type", "message"))
                except (json.JSONDecodeError, AttributeError):
                    event_type = "message"
                yield f"id: {line_number}\nevent: {event_type}\ndata: {line.rstrip()}\n\n"
                idle_started = time.monotonic()
                continue
            if record.get("status") in {"completed", "failed"}:
                terminal_seen_at = terminal_seen_at or time.monotonic()
                if time.monotonic() - terminal_seen_at >= 0.5:
                    return
            else:
                terminal_seen_at = None
            if time.monotonic() - idle_started >= 15.0:
                yield ": keep-alive\n\n"
                idle_started = time.monotonic()
            time.sleep(0.25)
    finally:
        if stream is not None:
            stream.close()


def _preview_stream(job_id: str) -> Iterator[bytes]:
    """Stream each completed annotated JPEG while retaining the last frame."""

    last_modified = -1
    terminal_seen_at: float | None = None
    while True:
        record = _job(job_id)
        preview = JOB_DIR / job_id / "preview.jpg"
        try:
            modified = preview.stat().st_mtime_ns
            if modified != last_modified:
                image = preview.read_bytes()
                last_modified = modified
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(image)).encode("ascii")
                    + b"\r\n\r\n"
                    + image
                    + b"\r\n"
                )
        except OSError:
            pass
        if record.get("status") in {"completed", "failed"}:
            terminal_seen_at = terminal_seen_at or time.monotonic()
            if time.monotonic() - terminal_seen_at >= 1.0:
                return
        else:
            terminal_seen_at = None
        time.sleep(0.05)


@app.get("/api/v1/jobs/{job_id}/artifacts/{artifact_path:path}")
def artifact(job_id: str, artifact_path: str):
    job_id = _safe_name(job_id, "job id")
    root = (JOB_DIR / job_id).resolve()
    path = (root / artifact_path).resolve()
    if root not in path.parents or not path.is_file():
        raise HTTPException(404, "Artifact not found")
    return FileResponse(path)
