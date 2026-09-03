import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from fruit_pipeline.camera_calibration.calibration_store import CalibrationStore
from fruit_pipeline.camera_calibration.models import CameraCalibration
from fruit_pipeline.integrated_pipeline import (
    IntegratedFruitSizingPipeline,
    IntegratedPipelineConfig,
    filter_instances_to_pallet,
    normalize_to_resolution,
)
from fruit_pipeline.pallet_geometry.detector import ManualPalletDetector
from fruit_pipeline.pipeline import PipelineConfig
from fruit_pipeline.segmentation.sam import FruitInstance
from fruit_pipeline.size_estimation.pipeline import SizeEstimationConfig


def _config(tmp_path: Path, source: Path, *, frame_step: int = 10) -> IntegratedPipelineConfig:
    calibration = CameraCalibration(
        camera_id="cam_001",
        camera_group=None,
        resolution=(160, 240),
        camera_matrix=np.array([[500.0, 0, 80.0], [0, 500.0, 120.0], [0, 0, 1.0]]),
        distortion_coefficients=np.zeros(5),
        reprojection_error=0.1,
    )
    calibration_dir = tmp_path / "calibrations"
    CalibrationStore(calibration_dir).save(calibration)
    pallet_config = tmp_path / "pallets.yaml"
    pallet_config.write_text(
        yaml.safe_dump({"pallet_types": {"test": {"width_mm": 100, "length_mm": 200}}}),
        encoding="utf-8",
    )
    points_file = tmp_path / "points.json"
    points_file.write_text(
        json.dumps([[10, 10], [110, 10], [110, 210], [10, 210]]),
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"
    return IntegratedPipelineConfig(
        detection=PipelineConfig(image_path=str(source), output_dir=str(output_dir)),
        sizing=SizeEstimationConfig(
            camera_id="cam_001",
            calibration_dir=calibration_dir,
            pallet_config_path=pallet_config,
        ),
        pallet_type="test",
        pallet_selection_path=output_dir / "pallet_selection.json",
        pallet_points_file=points_file,
        frame_step=frame_step,
    )


def _fake_detection_runner(config, detector, sam_predictor):
    image = cv2.imread(config.image_path)
    mask = np.zeros(image.shape[:2], dtype=bool)
    mask[40:81, 30:51] = True
    return [FruitInstance(1, [30, 40, 51, 81], 0.9, "fruit", 0.8, mask)]


def test_image_pipeline_selects_pallet_before_models_and_returns_sizes(tmp_path):
    image_path = tmp_path / "fruit.jpg"
    cv2.imwrite(str(image_path), np.zeros((240, 160, 3), np.uint8))
    config = _config(tmp_path, image_path)
    selection_path = Path(config.pallet_selection_path)

    def model_loader(_config):
        assert selection_path.is_file()
        assert selection_path.with_name("pallet_selection_preview.png").is_file()
        return object(), object()

    result = IntegratedFruitSizingPipeline(
        config,
        model_loader=model_loader,
        detection_runner=_fake_detection_runner,
    ).run(image_path)

    assert result.num_fruits == 1
    frame = result.frames[0].to_dict()
    assert frame["num_fruits"] == 1
    assert frame["full_image_num_fruits"] == 1
    assert frame["num_measured_fruits"] == 1
    assert frame["fruits"][0]["size"]["width_mm"] == 20.0
    assert result.to_dict()["average_fruit_size_mm"]["width"] == 20.0
    assert (tmp_path / "output/fruit_summary.json").is_file()
    assert (tmp_path / "output/fruit_result.json").is_file()


def test_video_pipeline_processes_every_tenth_frame(tmp_path, monkeypatch):
    video_path = tmp_path / "fruit.mp4"
    video_path.touch()
    config = _config(tmp_path, video_path, frame_step=10)
    frames = [np.zeros((240, 160, 3), np.uint8) for _ in range(25)]
    published = []

    class FakeCapture:
        def __init__(self, _path):
            self.index = 0
            self.last_read = -1
            self.released = False

        def isOpened(self):
            return True

        def read(self):
            if self.index >= len(frames):
                return False, None
            frame = frames[self.index]
            self.last_read = self.index
            self.index += 1
            return True, frame

        def get(self, property_id):
            if property_id == cv2.CAP_PROP_FRAME_COUNT:
                return len(frames)
            return self.last_read * 40.0

        def release(self):
            self.released = True

    monkeypatch.setattr("fruit_pipeline.integrated_pipeline.cv2.VideoCapture", FakeCapture)
    result = IntegratedFruitSizingPipeline(
        config,
        model_loader=lambda _config: (object(), object()),
        detection_runner=_fake_detection_runner,
        frame_processed=lambda result, preview, processed, total: published.append(
            (result.frame_index, preview.shape, processed, total)
        ),
    ).run(video_path)

    assert [frame.frame_index for frame in result.frames] == [0, 10, 20]
    assert [frame.num_fruits for frame in result.frames] == [1, 1, 1]
    assert published == [
        (0, (240, 160, 3), 1, 3),
        (10, (240, 160, 3), 2, 3),
        (20, (240, 160, 3), 3, 3),
    ]
    assert (tmp_path / "output/fruit_summary.json").is_file()
    assert (tmp_path / "output/frames/frame_000020/fruit_frame_000020_result.json").is_file()


def test_temporary_resize_rotates_landscape_input_to_portrait_calibration(tmp_path):
    image_path = tmp_path / "large.jpg"
    cv2.imwrite(str(image_path), np.zeros((320, 480, 3), np.uint8))
    config = replace(_config(tmp_path, image_path), resize_to_calibration=True)

    def model_loader(model_config):
        normalized = cv2.imread(model_config.image_path)
        assert normalized.shape[:2] == (240, 160)
        return object(), object()

    result = IntegratedFruitSizingPipeline(
        config,
        model_loader=model_loader,
        detection_runner=_fake_detection_runner,
    ).run(image_path)

    assert result.frames[0].num_fruits == 1
    assert (tmp_path / "output/normalized_inputs/large.png").is_file()


def test_temporary_resize_refuses_aspect_ratio_stretching():
    image = np.zeros((500, 500, 3), np.uint8)
    with pytest.raises(ValueError, match="Cropping or stretching"):
        normalize_to_resolution(image, (960, 1280))


def test_temporary_resize_can_force_aspect_ratio_for_testing():
    image = np.zeros((500, 500, 3), np.uint8)

    normalized, rotation = normalize_to_resolution(
        image,
        (960, 1280),
        allow_aspect_mismatch=True,
    )

    assert normalized.shape[:2] == (1280, 960)
    assert rotation == "none"


def test_fruit_outside_selected_pallet_is_not_counted_or_sized():
    inside_mask = np.zeros((240, 160), dtype=bool)
    inside_mask[40:81, 30:51] = True
    outside_mask = np.zeros((240, 160), dtype=bool)
    outside_mask[40:81, 130:151] = True
    instances = [
        FruitInstance(1, [30, 40, 51, 81], 0.9, "fruit", 0.8, inside_mask),
        FruitInstance(2, [130, 40, 151, 81], 0.9, "fruit", 0.8, outside_mask),
    ]
    corners = np.array([[10, 10], [110, 10], [110, 210], [10, 210]], np.float32)

    kept = filter_instances_to_pallet(instances, corners, min_overlap=0.5)

    assert [instance.instance_id for instance in kept] == [1]


def test_pallet_is_reselected_on_each_run_by_default(tmp_path, monkeypatch):
    image_path = tmp_path / "fruit.jpg"
    image = np.zeros((240, 160, 3), np.uint8)
    cv2.imwrite(str(image_path), image)
    config = replace(_config(tmp_path, image_path), pallet_points_file=None)
    selection_path = Path(config.pallet_selection_path)
    ManualPalletDetector(
        np.array([[20, 20], [100, 20], [100, 200], [20, 200]], np.float32),
        "test",
        image_resolution=(160, 240),
    ).save(selection_path)
    selected = np.array([[10, 10], [110, 10], [110, 210], [10, 210]], np.float32)
    calls = []

    def fake_select_points(*_args, **_kwargs):
        calls.append(True)
        return selected

    monkeypatch.setattr("fruit_pipeline.integrated_pipeline.select_points", fake_select_points)
    pipeline = IntegratedFruitSizingPipeline(config)
    pipeline.prepare_pallet(image)
    pipeline.prepare_pallet(image)

    assert len(calls) == 2
    np.testing.assert_array_equal(
        ManualPalletDetector.load(selection_path).detect(image).corners_px,
        selected,
    )
