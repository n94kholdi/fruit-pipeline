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
    media_source_stem,
    normalize_to_resolution,
)
from fruit_pipeline.pallet_geometry.detector import ManualPalletDetector
from fruit_pipeline.pipeline import PipelineConfig
from fruit_pipeline.segmentation.sam import FruitInstance
from fruit_pipeline.segmentation.sam2_config import SAM2Config
from fruit_pipeline.segmentation.sam2_manager import SAM2Timing
from fruit_pipeline.size_estimation.pipeline import SizeEstimationConfig, SizeEstimationPipeline


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


def _sam2_config(tmp_path: Path, source: Path, *, frame_step: int = 10) -> IntegratedPipelineConfig:
    return replace(
        _config(tmp_path, source, frame_step=frame_step),
        detection=None,
        sam2_video=SAM2Config(device="cpu"),
    )


class _FakeSAM2Manager:
    """Stands in for ``SAM2ModelManager`` so tests never touch a real model."""

    def __init__(self, instances_by_frame: dict[int, list[FruitInstance]],
                 image_instances: list[FruitInstance] | None = None):
        self.instances_by_frame = instances_by_frame
        self.image_instances = image_instances if image_instances is not None else []
        self.start_calls: list[tuple] = []
        self.process_calls: list[tuple[str, int]] = []
        self.stop_calls: list[str] = []
        self.discover_image_calls: list[tuple] = []

    def start_camera(self, camera_id, source, frame_shape, crop_box=None, initial_frame_rgb=None):
        self.start_calls.append((camera_id, source, frame_shape, crop_box))

    def process_frame(self, camera_id, image_rgb, frame_index, **kwargs):
        self.process_calls.append((camera_id, frame_index))
        return self.instances_by_frame[frame_index], SAM2Timing()

    def discover_image(self, image_rgb, crop_box=None):
        self.discover_image_calls.append((image_rgb.shape, crop_box))
        return self.image_instances, SAM2Timing()

    def stop_camera(self, camera_id):
        self.stop_calls.append(camera_id)


def _fruit_mask(y1, y2, x1, x2, shape=(240, 160)):
    mask = np.zeros(shape, dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


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


def test_live_stream_url_is_preserved_and_uses_safe_artifact_names(tmp_path, monkeypatch):
    config = replace(_config(tmp_path, tmp_path / "placeholder.mp4"), max_frames=1)
    stream_url = "rtsp://user:secret@mediamtx:8554/camera-01?token=private"
    captured_sources = []
    frame = np.zeros((240, 160, 3), np.uint8)

    class FakeCapture:
        def __init__(self, source):
            captured_sources.append(source)
            self.read_count = 0

        def isOpened(self):
            return True

        def read(self):
            self.read_count += 1
            return (True, frame) if self.read_count == 1 else (False, None)

        def get(self, _property_id):
            return 0

        def release(self):
            pass

    monkeypatch.setattr("fruit_pipeline.integrated_pipeline.cv2.VideoCapture", FakeCapture)
    result = IntegratedFruitSizingPipeline(
        config,
        model_loader=lambda _config: (object(), object()),
        detection_runner=_fake_detection_runner,
    ).run(stream_url)

    assert captured_sources == [stream_url]
    assert result.source == stream_url
    assert media_source_stem(stream_url) == "camera-01"
    assert (tmp_path / "output/camera-01_summary.json").is_file()


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


def _make_fake_capture(frames: list[np.ndarray]):
    class FakeCapture:
        def __init__(self, _path):
            self.index = 0
            self.last_read = -1

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
            pass

    return FakeCapture


def test_sam2_video_pipeline_discovers_then_propagates_and_records_lifecycle_fields(tmp_path, monkeypatch):
    video_path = tmp_path / "fruit.mp4"
    video_path.touch()
    config = _sam2_config(tmp_path, video_path, frame_step=10)
    frames = [np.zeros((240, 160, 3), np.uint8) for _ in range(25)]
    monkeypatch.setattr(
        "fruit_pipeline.integrated_pipeline.cv2.VideoCapture", _make_fake_capture(frames)
    )
    mask = _fruit_mask(40, 81, 30, 51)
    manager = _FakeSAM2Manager({
        0: [FruitInstance(1, [30, 40, 51, 81], 1.0, "fruit", 1.0, mask,
                           confidence=0.9, first_seen_frame=0, last_seen_frame=0,
                           last_discovery_frame=0, tracking_state="discovered")],
        10: [FruitInstance(1, [30, 40, 51, 81], 1.0, "fruit", 1.0, mask,
                            confidence=0.85, first_seen_frame=0, last_seen_frame=10,
                            last_discovery_frame=0, tracking_state="tracked")],
        20: [FruitInstance(1, [30, 40, 51, 81], 1.0, "fruit", 1.0, mask,
                            confidence=0.85, first_seen_frame=0, last_seen_frame=20,
                            last_discovery_frame=0, tracking_state="tracked")],
    })

    result = IntegratedFruitSizingPipeline(config, sam2_manager=manager).run(video_path)

    assert [frame.frame_index for frame in result.frames] == [0, 10, 20]
    assert manager.process_calls == [("cam_001", 0), ("cam_001", 10), ("cam_001", 20)]
    assert len(manager.start_calls) == 1
    camera_id, source, frame_shape, crop_box = manager.start_calls[0]
    assert (camera_id, source, frame_shape) == ("cam_001", str(video_path), (240, 160))
    # The pallet corners are (10,10)-(110,10)-(110,210)-(10,210); the crop is
    # their bounding box, in full-frame pixel coordinates (cv2.boundingRect is
    # inclusive of the corner pixel, hence the +1 on each far edge).
    assert crop_box == (10, 10, 111, 211)
    assert manager.stop_calls == ["cam_001"]

    first_fruit = result.frames[0].to_dict()["fruits"][0]
    assert first_fruit["tracking_state"] == "discovered"
    assert first_fruit["confidence"] == 0.9
    assert first_fruit["last_discovery_frame"] == 0
    tracked_fruit = result.frames[1].to_dict()["fruits"][0]
    assert tracked_fruit["tracking_state"] == "tracked"
    assert tracked_fruit["last_seen_frame"] == 10


def test_sam2_video_releases_camera_state_even_if_a_frame_raises(tmp_path, monkeypatch):
    video_path = tmp_path / "fruit.mp4"
    video_path.touch()
    config = _sam2_config(tmp_path, video_path, frame_step=10)
    frames = [np.zeros((240, 160, 3), np.uint8) for _ in range(25)]
    monkeypatch.setattr(
        "fruit_pipeline.integrated_pipeline.cv2.VideoCapture", _make_fake_capture(frames)
    )

    class _RaisingManager(_FakeSAM2Manager):
        def process_frame(self, camera_id, image_rgb, frame_index, **kwargs):
            if frame_index == 10:
                raise RuntimeError("SAM2 tracking failure")
            return super().process_frame(camera_id, image_rgb, frame_index, **kwargs)

    mask = _fruit_mask(40, 81, 30, 51)
    manager = _RaisingManager({0: [], 10: [], 20: []})
    manager.instances_by_frame[0] = [
        FruitInstance(1, [30, 40, 51, 81], 1.0, "fruit", 1.0, mask, tracking_state="discovered")
    ]

    with pytest.raises(RuntimeError, match="SAM2 tracking failure"):
        IntegratedFruitSizingPipeline(config, sam2_manager=manager).run(video_path)

    assert manager.stop_calls == ["cam_001"]


def test_run_image_uses_lightweight_discovery_and_never_starts_a_camera(tmp_path):
    """A still image takes the image-only discovery path: no video/camera
    session is ever started, registered with tracking objects, or torn down.
    """
    image_path = tmp_path / "fruit.jpg"
    cv2.imwrite(str(image_path), np.zeros((240, 160, 3), np.uint8))
    config = _sam2_config(tmp_path, image_path)
    mask = _fruit_mask(40, 81, 30, 51)
    manager = _FakeSAM2Manager({}, image_instances=[
        FruitInstance(1, [30, 40, 51, 81], 1.0, "fruit", 1.0, mask,
                      confidence=0.9, tracking_state="discovered")
    ])

    result = IntegratedFruitSizingPipeline(config, sam2_manager=manager).run(image_path)

    assert manager.start_calls == []
    assert manager.process_calls == []
    assert manager.stop_calls == []
    assert len(manager.discover_image_calls) == 1
    image_shape, crop_box = manager.discover_image_calls[0]
    assert image_shape == (240, 160, 3)
    assert crop_box == (10, 10, 111, 211)
    assert len(result.frames) == 1
    assert result.frames[0].num_fruits == 1
    assert result.frames[0].to_dict()["fruits"][0]["tracking_state"] == "discovered"


def test_run_image_never_starts_a_camera_even_if_discovery_raises(tmp_path):
    image_path = tmp_path / "fruit.jpg"
    cv2.imwrite(str(image_path), np.zeros((240, 160, 3), np.uint8))
    config = _sam2_config(tmp_path, image_path)

    class _RaisingManager(_FakeSAM2Manager):
        def discover_image(self, image_rgb, crop_box=None):
            raise RuntimeError("SAM2 discovery failure")

    manager = _RaisingManager({})

    with pytest.raises(RuntimeError, match="SAM2 discovery failure"):
        IntegratedFruitSizingPipeline(config, sam2_manager=manager).run(image_path)

    assert manager.start_calls == []
    assert manager.stop_calls == []


def test_sam2_propagated_frames_reuse_measurement_until_mask_changes(tmp_path, monkeypatch):
    video_path = tmp_path / "fruit.mp4"
    video_path.touch()
    config = _sam2_config(tmp_path, video_path, frame_step=10)
    frames = [np.zeros((240, 160, 3), np.uint8) for _ in range(25)]
    monkeypatch.setattr(
        "fruit_pipeline.integrated_pipeline.cv2.VideoCapture", _make_fake_capture(frames)
    )
    stable_mask = _fruit_mask(40, 81, 30, 51)  # 41 x 21 = 861px
    grown_mask = _fruit_mask(40, 100, 30, 51)  # 60 x 21 = 1260px, +46%
    manager = _FakeSAM2Manager({
        0: [FruitInstance(1, [30, 40, 51, 81], 1.0, "fruit", 1.0, stable_mask,
                           tracking_state="discovered")],
        10: [FruitInstance(1, [30, 40, 51, 81], 1.0, "fruit", 1.0, stable_mask,
                            tracking_state="tracked")],
        20: [FruitInstance(1, [30, 40, 51, 100], 1.0, "fruit", 1.0, grown_mask,
                            tracking_state="tracked")],
    })
    recorded_ids: list[list[int]] = []
    original_run = SizeEstimationPipeline.run

    def spy_run(self, image_bgr, fruits):
        fruits = list(fruits)
        recorded_ids.append([fruit.instance_id for fruit in fruits])
        return original_run(self, image_bgr, fruits)

    monkeypatch.setattr(SizeEstimationPipeline, "run", spy_run)

    result = IntegratedFruitSizingPipeline(config, sam2_manager=manager).run(video_path)

    # Frame 0 measures the newly discovered fruit; frame 10's mask is
    # unchanged so it is skipped and the frame-0 measurement is reused;
    # frame 20's mask grew well past the change threshold, so it is
    # remeasured.
    assert recorded_ids == [[1], [], [1]]
    first_measurement = result.frames[0].sizing.measurements[0]
    reused_measurement = result.frames[1].sizing.measurements[0]
    remeasured = result.frames[2].sizing.measurements[0]
    assert reused_measurement is first_measurement
    assert remeasured is not first_measurement
    assert remeasured.length_mm > first_measurement.length_mm
