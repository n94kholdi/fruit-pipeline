from pathlib import Path

import cv2
import numpy as np

from fruit_pipeline import dashboard_api


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
