import json

import cv2
import numpy as np
import pytest
import yaml

from fruit_pipeline.camera_calibration.calibration_store import CalibrationStore
from fruit_pipeline.camera_calibration.models import CameraCalibration
from fruit_pipeline.pallet_geometry.detector import (
    ManualPalletDetector,
    PalletDetector,
    PalletGeometryError,
)
from fruit_pipeline.size_estimation.manual_cli import measure_object_main, select_pallet_main


def _write_image(path):
    image = np.zeros((240, 160, 3), np.uint8)
    assert cv2.imwrite(str(path), image)


def _write_points(path, points):
    path.write_text(json.dumps(points), encoding="utf-8")


def _calibration():
    return CameraCalibration(
        camera_id="manual_cam",
        camera_group=None,
        resolution=(160, 240),
        camera_matrix=np.array([[500.0, 0, 80.0], [0, 500.0, 120.0], [0, 0, 1.0]]),
        distortion_coefficients=np.zeros(5),
        reprojection_error=0.2,
    )


def test_manual_detector_round_trip_implements_replaceable_protocol(tmp_path):
    corners = np.array([[10, 10], [110, 10], [110, 210], [10, 210]], np.float32)
    detector = ManualPalletDetector(corners, "test", image_resolution=(160, 240))
    assert isinstance(detector, PalletDetector)
    path = detector.save(tmp_path / "corners.json")
    loaded = ManualPalletDetector.load(path)
    np.testing.assert_array_equal(loaded.detect(np.zeros((240, 160, 3))).corners_px, corners)
    with pytest.raises(PalletGeometryError, match="but image is"):
        loaded.detect(np.zeros((120, 80, 3)))


def test_manual_cli_exposes_configurable_preview_size(capsys):
    with pytest.raises(SystemExit) as exc:
        select_pallet_main(["--help"])
    assert exc.value.code == 0
    assert "--max-preview-size" in capsys.readouterr().out


def test_headless_manual_cli_pipeline_saves_measurement_and_debug_views(tmp_path):
    image_path = tmp_path / "frame.png"
    _write_image(image_path)
    pallet_points = tmp_path / "pallet_points.json"
    _write_points(pallet_points, [[10, 10], [110, 10], [110, 210], [10, 210]])
    saved_corners = tmp_path / "pallet_selection.json"
    assert select_pallet_main([
        "--image", str(image_path),
        "--pallet-type", "test",
        "--points-file", str(pallet_points),
        "--output", str(saved_corners),
    ]) == 0

    calibration_dir = tmp_path / "calibrations"
    CalibrationStore(calibration_dir).save(_calibration())
    config_path = tmp_path / "pallets.yaml"
    config_path.write_text(
        yaml.safe_dump({"pallet_types": {"test": {"width_mm": 100, "length_mm": 200}}}),
        encoding="utf-8",
    )
    object_points = tmp_path / "object.json"
    _write_points(object_points, [[30, 40], [50, 40], [50, 80], [30, 80]])
    output_dir = tmp_path / "results"

    assert measure_object_main([
        "--image", str(image_path),
        "--pallet-corners", str(saved_corners),
        "--calibration-dir", str(calibration_dir),
        "--camera-id", "manual_cam",
        "--pallet-config", str(config_path),
        "--object-points", str(object_points),
        "--ground-truth-width-mm", "21",
        "--ground-truth-length-mm", "42",
        "--output-dir", str(output_dir),
    ]) == 0

    payload = json.loads((output_dir / "frame_measurement.json").read_text(encoding="utf-8"))
    assert payload["measurement"]["width_mm"] == pytest.approx(20.0, abs=0.01)
    assert payload["measurement"]["length_mm"] == pytest.approx(40.0, abs=0.01)
    assert payload["measurement"]["area_mm2"] == pytest.approx(800.0, abs=0.01)
    assert payload["ground_truth"]["width_error_percent"] == pytest.approx(100 / 21)
    assert payload["ground_truth"]["length_error_percent"] == pytest.approx(200 / 42)
    assert (output_dir / "frame_original.png").is_file()
    assert (output_dir / "frame_pallet_corners.png").is_file()
    assert (output_dir / "frame_object_measurement.png").is_file()
    assert (output_dir / "frame_rectified_pallet.png").is_file()
