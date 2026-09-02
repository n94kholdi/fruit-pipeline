"""HTTP adapter for camera calibration and dashboard fruit-analysis jobs."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from fruit_pipeline.camera_calibration.calibrate import BoardSpec
from fruit_pipeline.camera_calibration.calibration_store import CalibrationStore
from fruit_pipeline.camera_calibration.models import CalibrationError
from fruit_pipeline.integrated_pipeline import normalize_to_resolution
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
    max_frames: int | None = Field(default=None, ge=1)


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
        return dict(record)
    path = JOB_DIR / job_id / "job.json"
    if path.is_file():
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            record = None
    if record is None:
        raise HTTPException(404, "Job not found")
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


def _read_media_first_frame(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is not None:
        return image
    capture = cv2.VideoCapture(str(path))
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise HTTPException(422, "Could not read the uploaded image or first video frame")
    return frame


def _artifact_url(job_id: str, path: Path) -> str:
    relative = path.relative_to(JOB_DIR / job_id)
    return f"/api/v1/jobs/{job_id}/artifacts/{relative.as_posix()}"


def _result_payload(job_id: str, output_dir: Path, source: Path) -> dict[str, object]:
    summary_path = output_dir / f"{source.stem}_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    frames = summary.get("frames", [])
    for frame in frames:
        artifact_dir = Path(frame["artifact_dir"])
        stem = Path(frame["source_image"]).stem
        overlay = artifact_dir / f"{stem}_measurement_debug.png"
        if overlay.is_file():
            frame["measurement_overlay_url"] = _artifact_url(job_id, overlay)
    summary["summary_url"] = _artifact_url(job_id, summary_path)
    return summary


def _run_fruit_job(job_id: str, request: FruitJobRequest, source: Path) -> None:
    _write_job(job_id, status="running")
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
        "-v",
    ]
    if request.resize_to_calibration:
        command.append("--resize-to-calibration")
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
    except Exception as exc:
        _write_job(job_id, status="failed", error=str(exc))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


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
        normalized, _ = normalize_to_resolution(frame, calibration.resolution, rotation="auto")
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
    sources = [path for path in input_folder.glob("source.*") if path.is_file()]
    if len(sources) != 1:
        raise HTTPException(404, "Uploaded input not found")
    try:
        CalibrationStore(CALIBRATION_DIR).load(camera_id)
    except CalibrationError as exc:
        raise HTTPException(404, str(exc)) from exc
    request.camera_id = camera_id
    job_id = uuid.uuid4().hex
    _write_job(job_id, kind="fruit_analysis", status="queued", camera_id=camera_id)
    executor.submit(_run_fruit_job, job_id, request, sources[0])
    return {"data": _job(job_id)}


@app.get("/api/v1/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, object]:
    return {"data": _job(_safe_name(job_id, "job id"))}


@app.get("/api/v1/jobs/{job_id}/artifacts/{artifact_path:path}")
def artifact(job_id: str, artifact_path: str):
    job_id = _safe_name(job_id, "job id")
    root = (JOB_DIR / job_id).resolve()
    path = (root / artifact_path).resolve()
    if root not in path.parents or not path.is_file():
        raise HTTPException(404, "Artifact not found")
    return FileResponse(path)
