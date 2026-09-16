import json

import cv2
import numpy as np

from fruit_pipeline.live import FruitLiveReporter


def test_reporter_writes_annotated_preview_and_cumulative_progress(tmp_path):
    reporter = FruitLiveReporter(tmp_path, "fruit-job", preview_width=320)
    frame = np.full((240, 640, 3), 127, np.uint8)

    first = reporter.publish_frame(
        frame,
        frame_index=0,
        timestamp_ms=0.0,
        processed_frame_count=1,
        total_sampled_frames=4,
        num_fruits=3,
        num_measured_fruits=2,
    )
    second = reporter.publish_frame(
        frame,
        frame_index=10,
        timestamp_ms=400.0,
        processed_frame_count=2,
        total_sampled_frames=4,
        num_fruits=4,
        num_measured_fruits=4,
    )

    preview = cv2.imread(str(tmp_path / "preview.jpg"))
    assert preview.shape[:2] == (120, 320)
    assert first["progress"] == 25.0
    assert second["progress"] == 50.0
    assert second["metrics"]["total_fruit_observations"] == 7
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [event["frame_index"] for event in events] == [0, 10]


def test_terminal_event_preserves_latest_preview_and_metrics(tmp_path):
    reporter = FruitLiveReporter(tmp_path, "fruit-job")
    reporter.publish_frame(
        np.zeros((20, 20, 3), np.uint8),
        frame_index=20,
        timestamp_ms=800.0,
        processed_frame_count=3,
        total_sampled_frames=3,
        num_fruits=5,
        num_measured_fruits=4,
    )

    completed = reporter.emit("job_completed", status="completed", progress=100.0)

    assert completed["preview_reference"] == "/api/v1/jobs/fruit-job/preview"
    assert completed["frame_index"] == 20
    assert completed["metrics"]["num_fruits"] == 5


def test_cached_mask_preview_keeps_last_refresh_totals_and_reports_sizes(tmp_path):
    reporter = FruitLiveReporter(tmp_path, "fruit-job")
    frame = np.zeros((20, 20, 3), np.uint8)
    reporter.publish_frame(
        frame,
        frame_index=0,
        timestamp_ms=0.0,
        processed_frame_count=1,
        total_sampled_frames=None,
        num_fruits=5,
        num_measured_fruits=4,
        average_fruit_size_mm={"width": 31.5, "length": 35.0, "equivalent_diameter": 33.0},
    )

    cached = reporter.publish_frame(
        frame,
        frame_index=10,
        timestamp_ms=400.0,
        processed_frame_count=1,
        total_sampled_frames=None,
        num_fruits=5,
        num_measured_fruits=4,
        inference_refreshed=False,
        average_fruit_size_mm={"width": 31.5, "length": 35.0, "equivalent_diameter": 33.0},
    )

    assert cached["metrics"]["total_fruit_observations"] == 5
    assert cached["metrics"]["inference_refreshed"] is False
    assert cached["metrics"]["average_fruit_size_mm"]["width"] == 31.5
