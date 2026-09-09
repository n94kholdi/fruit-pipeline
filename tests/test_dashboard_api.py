import json
import threading
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi import HTTPException

from fruit_pipeline import dashboard_api
from fruit_pipeline.camera_calibration.calibration_store import CalibrationStore
from fruit_pipeline.camera_calibration.models import CameraCalibration
from fruit_pipeline.live import FruitLiveReporter
from fruit_pipeline.segmentation.sam2_config import SAM2Config


def _fruit_job_request(**changes):
    values = {
        "input_id": "input-01",
        "camera_id": "camera-01",
        "corners": [
            {"x": 10, "y": 10},
            {"x": 110, "y": 10},
            {"x": 110, "y": 210},
            {"x": 10, "y": 210},
        ],
    }
    values.update(changes)
    return dashboard_api.FruitJobRequest(**values)


def test_startup_preloads_the_persistent_sam2_manager_for_detector(tmp_path, monkeypatch):
    checkpoint = tmp_path / "sam2.pt"
    checkpoint.touch()
    config = SAM2Config(device="cpu", checkpoint=str(checkpoint))
    calls = []
    monkeypatch.setattr(dashboard_api, "SERVICE_INFERENCE_MODE", "detector")
    monkeypatch.setattr(dashboard_api, "SAM2_CONFIG", config)
    monkeypatch.setattr(
        dashboard_api, "get_sam2_model_manager",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    dashboard_api.preload_sam_model()

    assert calls == [((config,), {"eager": True})]


def test_fruit_job_runs_in_process_to_reuse_startup_model(tmp_path, monkeypatch):
    from fruit_pipeline import integrated_cli

    calls = []
    source = tmp_path / "input.jpg"
    source.touch()
    monkeypatch.setattr(dashboard_api, "JOB_DIR", tmp_path / "jobs")
    monkeypatch.setattr(dashboard_api, "CALIBRATION_DIR", tmp_path / "calibrations")
    monkeypatch.setattr(dashboard_api, "PALLET_CONFIG", tmp_path / "pallet_types.yaml")
    monkeypatch.setattr(integrated_cli, "main", lambda argv: calls.append(argv) or 0)
    monkeypatch.setattr(
        dashboard_api,
        "_result_payload",
        lambda job_id, output_dir, source: {"job_id": job_id},
    )
    with dashboard_api.jobs_lock:
        dashboard_api.jobs.clear()
        dashboard_api.job_processes.clear()
    dashboard_api._write_job("job-in-process", kind="fruit_analysis", status="queued")

    dashboard_api._run_fruit_job("job-in-process", _fruit_job_request(), source)

    assert calls and calls[0][0:2] == ["--image", str(source)]
    assert dashboard_api._job("job-in-process")["status"] == "completed"
    assert "job-in-process" not in dashboard_api.job_processes


def test_custom_pallet_dimensions_create_a_job_scoped_config(tmp_path):
    request = _fruit_job_request(
        pallet_type="custom",
        pallet_width_mm=850.5,
        pallet_length_mm=1350,
    )

    dashboard_api._validate_requested_pallet(request)
    config_path = dashboard_api._pallet_config_for_job(tmp_path, request)

    assert config_path.parent == tmp_path
    assert json.loads(config_path.read_text(encoding="utf-8")) == {
        "pallet_types": {
            "custom": {"width_mm": 850.5, "length_mm": 1350.0},
        }
    }


def test_custom_pallet_requires_both_dimensions():
    import pytest
    from fastapi import HTTPException

    request = _fruit_job_request(pallet_type="custom", pallet_width_mm=850)

    with pytest.raises(HTTPException, match="width and length"):
        dashboard_api._validate_requested_pallet(request)


def test_result_payload_exposes_average_and_overlay(tmp_path, monkeypatch):
    job_id = "job1"
    job_dir = tmp_path / job_id
    output = job_dir / "results"
    artifact = output / "frame"
    artifact.mkdir(parents=True)
    source = tmp_path / "fruit.jpg"
    source.touch()
    overlay = artifact / "fruit_measurement_debug.png"
    cv2.imwrite(str(overlay), np.zeros((10, 10, 3), np.uint8))
    (output / "fruit_summary.json").write_text(
        '{"total_fruit_observations": 2, "average_fruit_size_mm": {"width": 30}, '
        f'"frames": [{{"artifact_dir": "{artifact}", "source_image": "{source}"}}]}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(dashboard_api, "JOB_DIR", tmp_path)

    result = dashboard_api._result_payload(job_id, output, source)

    assert result["total_fruit_observations"] == 2
    assert result["frames"][0]["measurement_overlay_url"].endswith("fruit_measurement_debug.png")


def test_safe_name_rejects_path_segments():
    import pytest
    from fastapi import HTTPException

    with pytest.raises(HTTPException):
        dashboard_api._safe_name("../camera", "camera id")


def test_job_polling_includes_latest_live_state(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard_api, "JOB_DIR", tmp_path)
    with dashboard_api.jobs_lock:
        dashboard_api.jobs.clear()
    dashboard_api._write_job("job-live", status="running")
    reporter = FruitLiveReporter(tmp_path / "job-live", "job-live")
    reporter.publish_frame(
        np.zeros((20, 20, 3), np.uint8),
        frame_index=10,
        timestamp_ms=400.0,
        processed_frame_count=2,
        total_sampled_frames=5,
        num_fruits=3,
        num_measured_fruits=2,
    )

    record = dashboard_api._job("job-live")

    assert record["live"]["frame_index"] == 10
    assert record["live"]["progress"] == 40.0


def test_preview_and_event_streams_expose_persisted_updates(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard_api, "JOB_DIR", tmp_path)
    with dashboard_api.jobs_lock:
        dashboard_api.jobs.clear()
    dashboard_api._write_job("job-stream", status="completed")
    reporter = FruitLiveReporter(tmp_path / "job-stream", "job-stream")
    reporter.publish_frame(
        np.zeros((20, 20, 3), np.uint8),
        frame_index=0,
        timestamp_ms=0.0,
        processed_frame_count=1,
        total_sampled_frames=1,
        num_fruits=1,
        num_measured_fruits=1,
    )

    preview_chunk = next(dashboard_api._preview_stream("job-stream"))
    event_chunk = next(dashboard_api._event_stream("job-stream", 0))

    assert preview_chunk.startswith(b"--frame\r\nContent-Type: image/jpeg")
    assert "event: preview_updated" in event_chunk
    assert '"frame_index":0' in event_chunk


def test_create_stream_input_captures_normalized_preview_and_keeps_source_private(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard_api, "INPUT_DIR", tmp_path / "inputs")
    monkeypatch.setattr(
        dashboard_api.CalibrationStore,
        "load",
        lambda _store, _camera_id: SimpleNamespace(resolution=(160, 240)),
    )
    monkeypatch.setattr(
        dashboard_api,
        "_read_media_first_frame",
        lambda source: np.zeros((240, 160, 3), np.uint8),
    )

    response = dashboard_api.create_stream_input(dashboard_api.StreamInputRequest(
        camera_id="camera-01",
        stream_url="rtsp://user:secret@mediamtx:8554/camera-01",
    ))

    input_id = response["data"]["id"]
    input_dir = tmp_path / "inputs" / input_id
    assert response["data"]["source_type"] == "stream"
    assert (input_dir / "preview.jpg").is_file()
    metadata = (input_dir / "stream.json").read_text(encoding="utf-8")
    assert "user:secret" in metadata
    assert "stream_url" not in response["data"]


def test_create_stream_calibration_queues_private_stream_capture(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard_api, "JOB_DIR", tmp_path / "jobs")
    submitted = []
    monkeypatch.setattr(
        dashboard_api,
        "executor",
        SimpleNamespace(submit=lambda *args: submitted.append(args)),
    )
    with dashboard_api.jobs_lock:
        dashboard_api.jobs.clear()

    response = dashboard_api.create_stream_calibration(
        dashboard_api.StreamCalibrationRequest(
            camera_id="camera-01",
            stream_url="rtsp://user:secret@mediamtx:8554/camera-01",
            capture_seconds=20,
        )
    )

    record = response["data"]
    assert record["status"] == "queued"
    assert record["source_type"] == "stream"
    assert record["capture_seconds"] == 20
    assert submitted[0][0] is dashboard_api._run_stream_calibration
    persisted = (tmp_path / "jobs" / record["id"] / "job.json").read_text(encoding="utf-8")
    assert "user:secret" not in persisted


def test_capture_calibration_frames_samples_a_bounded_stream(tmp_path, monkeypatch):
    frame = np.zeros((12, 16, 3), np.uint8)

    class FakeCapture:
        released = False

        def isOpened(self):
            return True

        def read(self):
            return True, frame

        def release(self):
            self.released = True

    capture = FakeCapture()
    times = iter([0.0, 0.0, 1.0, 2.0, 6.0])
    monkeypatch.setattr(dashboard_api.cv2, "VideoCapture", lambda _url: capture)
    monkeypatch.setattr(dashboard_api.time, "monotonic", lambda: next(times))

    count = dashboard_api._capture_calibration_frames(
        "rtsp://camera/live", tmp_path / "captures", capture_seconds=5, frame_step=2,
    )

    assert count == 2
    assert capture.released is True
    assert sorted(path.name for path in (tmp_path / "captures").iterdir()) == ["0000.jpg", "0001.jpg"]


def test_cancel_queued_fruit_job_marks_it_cancelled(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard_api, "JOB_DIR", tmp_path)
    with dashboard_api.jobs_lock:
        dashboard_api.jobs.clear()
        dashboard_api.job_futures.clear()
        dashboard_api.job_processes.clear()
    dashboard_api._write_job(
        "job-cancel-queued",
        kind="fruit_analysis",
        status="queued",
    )
    future = SimpleNamespace(cancel=lambda: True)
    with dashboard_api.jobs_lock:
        dashboard_api.job_futures["job-cancel-queued"] = future

    response = dashboard_api.cancel_job("job-cancel-queued")

    assert response["data"]["status"] == "cancelled"
    assert (tmp_path / "job-cancel-queued" / "cancel.requested").is_file()
    events = (tmp_path / "job-cancel-queued" / "events.jsonl").read_text(encoding="utf-8")
    assert '"type":"job_cancelled"' in events


def test_cancel_running_fruit_job_terminates_process(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard_api, "JOB_DIR", tmp_path)
    with dashboard_api.jobs_lock:
        dashboard_api.jobs.clear()
        dashboard_api.job_futures.clear()
        dashboard_api.job_processes.clear()
    dashboard_api._write_job(
        "job-cancel-running",
        kind="fruit_analysis",
        status="running",
    )
    process = SimpleNamespace(poll=lambda: None, terminate_called=False)

    def terminate():
        process.terminate_called = True

    process.terminate = terminate
    with dashboard_api.jobs_lock:
        dashboard_api.job_processes["job-cancel-running"] = process

    response = dashboard_api.cancel_job("job-cancel-running")

    assert response["data"]["status"] == "cancelling"
    assert process.terminate_called is True
    assert (tmp_path / "job-cancel-running" / "cancel.requested").is_file()


def test_startup_preloads_the_persistent_sam2_manager(monkeypatch):
    config = SAM2Config(device="cpu")
    calls = []
    monkeypatch.setattr(dashboard_api, "SERVICE_INFERENCE_MODE", "sam2_video")
    monkeypatch.setattr(dashboard_api, "SAM2_CONFIG", config)
    monkeypatch.setattr(
        dashboard_api, "get_sam2_model_manager",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    dashboard_api.preload_sam_model()

    assert calls == [((config,), {"eager": True})]


def test_health_reports_sam2_status_when_selected(tmp_path, monkeypatch):
    checkpoint = tmp_path / "sam2.pt"
    checkpoint.touch()
    config = SAM2Config(device="cpu", checkpoint=str(checkpoint))
    fake_manager = SimpleNamespace(status=lambda: {"model_name": config.model_name, "loaded": True})
    monkeypatch.setattr(dashboard_api, "SERVICE_INFERENCE_MODE", "sam2_video")
    monkeypatch.setattr(dashboard_api, "SAM2_CONFIG", config)
    monkeypatch.setattr(dashboard_api, "get_sam2_model_manager", lambda *a, **k: fake_manager)

    payload = dashboard_api.health()

    assert payload["inference_mode"] == "sam2_video"
    assert payload["models_ready"] is True
    assert payload["models"]["sam2_selected"] is True
    assert payload["sam2"] == {"model_name": config.model_name, "loaded": True}


def _save_calibration(calibration_dir: Path, camera_id: str) -> None:
    CalibrationStore(calibration_dir).save(
        CameraCalibration(
            camera_id=camera_id,
            camera_group=None,
            resolution=(160, 240),
            camera_matrix=np.array([[500.0, 0, 80.0], [0, 500.0, 120.0], [0, 0, 1.0]]),
            distortion_coefficients=np.zeros(5),
            reprojection_error=0.1,
        )
    )


def test_create_fruit_job_rejects_a_mode_the_worker_was_not_started_with(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard_api, "INPUT_DIR", tmp_path / "inputs")
    monkeypatch.setattr(dashboard_api, "CALIBRATION_DIR", tmp_path / "calibrations")
    monkeypatch.setattr(dashboard_api, "SERVICE_INFERENCE_MODE", "sam2_video")
    input_folder = dashboard_api.INPUT_DIR / "input-01"
    input_folder.mkdir(parents=True)
    (input_folder / "source.jpg").touch()
    _save_calibration(dashboard_api.CALIBRATION_DIR, "camera-01")

    with pytest.raises(HTTPException) as excinfo:
        dashboard_api.create_fruit_job(_fruit_job_request(inference_mode="detector"))

    assert excinfo.value.status_code == 422
    assert "sam2_video" in str(excinfo.value.detail)


def test_sam2_idle_cleanup_loop_sweeps_immediately_and_stops_cleanly(monkeypatch):
    calls = []
    manager = SimpleNamespace(
        config=SimpleNamespace(camera_idle_timeout_seconds=60.0),
        cleanup_idle=lambda: calls.append(1) or ["stale-camera"],
    )
    dashboard_api._sam2_cleanup_stop.clear()
    thread = threading.Thread(target=dashboard_api._sam2_idle_cleanup_loop, args=(manager,), daemon=True)
    thread.start()
    try:
        for _ in range(200):
            if calls:
                break
            threading.Event().wait(0.01)
        assert calls, "cleanup_idle should run once immediately, not only after the first interval"
    finally:
        dashboard_api._sam2_cleanup_stop.set()
        thread.join(timeout=5)
    assert not thread.is_alive()
