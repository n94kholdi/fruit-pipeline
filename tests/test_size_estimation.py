from dataclasses import dataclass

import cv2
import numpy as np
import pytest
import yaml

from fruit_pipeline.camera_calibration.calibration_store import CalibrationStore
from fruit_pipeline.camera_calibration.models import CalibrationError, CameraCalibration
from fruit_pipeline.measurement.fruit_measurement import measure_fruit_mask
from fruit_pipeline.pallet_geometry.detector import PalletDetection, PalletGeometryError
from fruit_pipeline.pallet_geometry.homography import compute_pallet_homography
from fruit_pipeline.pallet_geometry.pallet_config import PalletDimensions
from fruit_pipeline.size_estimation.pipeline import SizeEstimationConfig, SizeEstimationPipeline


def calibration(camera_id="cam_001", camera_group="model_a", error=0.2):
    return CameraCalibration(
        camera_id=camera_id,
        camera_group=camera_group,
        resolution=(160, 240),
        camera_matrix=np.array([[500.0, 0, 80.0], [0, 500.0, 120.0], [0, 0, 1.0]]),
        distortion_coefficients=np.zeros(5),
        reprojection_error=error,
    )


def test_calibration_round_trip_and_camera_override(tmp_path):
    store = CalibrationStore(tmp_path)
    group = calibration(camera_id=None, error=0.4)
    camera = calibration(error=0.1)
    store.save(group, as_group=True)
    assert store.load("cam_001", "model_a").reprojection_error == pytest.approx(0.4)
    store.save(camera)
    loaded = store.load("cam_001", "model_a")
    assert loaded.reprojection_error == pytest.approx(0.1)
    assert (loaded.fx, loaded.fy, loaded.cx, loaded.cy) == (500.0, 500.0, 80.0, 120.0)


def test_missing_and_resolution_mismatch_are_explicit(tmp_path):
    with pytest.raises(CalibrationError, match="No calibration"):
        CalibrationStore(tmp_path).load("missing", "missing_group")
    with pytest.raises(CalibrationError, match="but image is"):
        calibration().validate_image_resolution((100, 100, 3))


def test_homography_and_measurement_return_real_units():
    cal = calibration()
    corners = np.array([[10, 10], [110, 10], [110, 210], [10, 210]], np.float32)
    homography = compute_pallet_homography(corners, PalletDimensions(100, 200), cal)
    np.testing.assert_allclose(
        homography.transform_points(corners),
        np.array([[0, 0], [100, 0], [100, 200], [0, 200]], np.float32),
        atol=1e-4,
    )
    mask = np.zeros((240, 160), np.uint8)
    cv2.rectangle(mask, (30, 40), (50, 80), 1, thickness=-1)
    measured = measure_fruit_mask(7, mask, 0.8, cal, homography)
    assert measured.width_mm == pytest.approx(20, abs=0.01)
    assert measured.length_mm == pytest.approx(40, abs=0.01)
    assert measured.area_mm2 == pytest.approx(800, abs=0.1)
    assert measured.equivalent_diameter_mm == pytest.approx(np.sqrt(3200 / np.pi), abs=0.01)


def test_crossed_or_degenerate_corners_are_rejected():
    crossed = np.array([[0, 0], [100, 100], [100, 0], [0, 100]], np.float32)
    with pytest.raises(PalletGeometryError):
        PalletDetection(crossed, 0.9, "standard")
    duplicate = np.array([[0, 0], [100, 0], [100, 0], [0, 100]], np.float32)
    with pytest.raises(PalletGeometryError):
        PalletDetection(duplicate, 0.9, "standard")


@dataclass
class Fruit:
    instance_id: int
    detector_score: float
    sam_score: float
    mask: np.ndarray


class Detector:
    def __init__(self, confidence=0.9):
        self.confidence = confidence

    def detect(self, image):
        return PalletDetection(
            np.array([[10, 10], [110, 10], [110, 210], [10, 210]], np.float32),
            self.confidence,
            "test_pallet",
        )


def pipeline(tmp_path, detector, debug=False):
    calibration_dir = tmp_path / "calibrations"
    CalibrationStore(calibration_dir).save(calibration())
    config_path = tmp_path / "pallets.yaml"
    config_path.write_text(
        yaml.safe_dump({"pallet_types": {"test_pallet": {"width_mm": 100, "length_mm": 200}}}),
        encoding="utf-8",
    )
    return SizeEstimationPipeline(
        SizeEstimationConfig(
            camera_id="cam_001", calibration_dir=calibration_dir,
            pallet_config_path=config_path, debug=debug,
        ),
        detector,
    )


def test_pipeline_rejects_low_confidence_pallet(tmp_path):
    with pytest.raises(PalletGeometryError, match="below threshold"):
        pipeline(tmp_path, Detector(0.1)).run(np.zeros((240, 160, 3), np.uint8), [])


def test_pipeline_produces_measurement_and_debug_views(tmp_path):
    mask = np.zeros((240, 160), bool)
    mask[40:81, 30:51] = True
    result = pipeline(tmp_path, Detector(), debug=True).run(
        np.zeros((240, 160, 3), np.uint8), [Fruit(3, 0.9, 0.7, mask)]
    )
    assert len(result.measurements) == 1
    assert result.measurements[0].confidence == pytest.approx(0.7)
    assert result.debug_overlay.shape == (240, 160, 3)
    assert result.rectified_pallet.shape[:2] == (100, 50)
    result.save(tmp_path / "outputs", "frame_1")
    assert (tmp_path / "outputs/frame_1_measurements.json").is_file()
    assert (tmp_path / "outputs/frame_1_measurement_debug.png").is_file()
    assert (tmp_path / "outputs/frame_1_rectified_pallet.png").is_file()
