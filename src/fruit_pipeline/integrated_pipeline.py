"""End-to-end pallet setup, fruit detection, segmentation, and sizing.

The manual pallet provider used here implements the same ``PalletDetector``
interface as the planned learned pallet model.  Replacing it therefore does
not change the fruit or measurement stages.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from fruit_pipeline.camera_calibration.calibration_store import CalibrationStore
from fruit_pipeline.measurement.fruit_measurement import FruitMeasurement
from fruit_pipeline.pallet_geometry.detector import (
    ManualPalletDetector,
    PalletDetector,
    PalletGeometryError,
)
from fruit_pipeline.pipeline import PipelineConfig, load_models, run_pipeline
from fruit_pipeline.segmentation.sam import FruitInstance
from fruit_pipeline.size_estimation.manual_selection import load_points, select_points
from fruit_pipeline.size_estimation.pipeline import (
    SizeEstimationConfig,
    SizeEstimationPipeline,
    SizeEstimationResult,
)

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
PALLET_CORNER_LABELS = ("TL", "TR", "BR", "BL")
INPUT_ROTATIONS = ("auto", "none", "clockwise", "counterclockwise", "180")


@dataclass(frozen=True)
class IntegratedPipelineConfig:
    detection: PipelineConfig
    sizing: SizeEstimationConfig
    pallet_type: str
    pallet_selection_path: str | Path
    pallet_points_file: str | Path | None = None
    frame_step: int = 10
    max_frames: int | None = None
    max_preview_size: int = 900
    resize_to_calibration: bool = False
    allow_unsafe_resize: bool = False
    input_rotation: str = "auto"
    reuse_pallet_selection: bool = False
    min_pallet_overlap: float = 0.5


@dataclass
class FrameResult:
    source_image: str
    frame_index: int | None
    timestamp_ms: float | None
    instances: list[FruitInstance]
    sizing: SizeEstimationResult
    artifact_dir: str
    full_image_num_fruits: int

    @property
    def num_fruits(self) -> int:
        return len(self.instances)

    def to_dict(self) -> dict[str, object]:
        measurements = {item.fruit_id: item for item in self.sizing.measurements}
        return {
            "source_image": self.source_image,
            "frame_index": self.frame_index,
            "timestamp_ms": self.timestamp_ms,
            "num_fruits": self.num_fruits,
            "full_image_num_fruits": self.full_image_num_fruits,
            "num_measured_fruits": len(measurements),
            "pallet_type": self.sizing.pallet_detection.pallet_type,
            "pallet_confidence": self.sizing.pallet_detection.confidence,
            "artifact_dir": self.artifact_dir,
            "fruits": [
                _fruit_record(instance, measurements.get(instance.instance_id))
                for instance in self.instances
            ],
        }


@dataclass
class MediaResult:
    source: str
    pallet_selection_path: str
    frames: list[FrameResult]

    @property
    def num_fruits(self) -> int:
        """Fruit count for an image, or the sum across sampled video frames."""
        return sum(frame.num_fruits for frame in self.frames)

    @property
    def average_size_mm(self) -> dict[str, float] | None:
        measurements = [
            measurement
            for frame in self.frames
            for measurement in frame.sizing.measurements
        ]
        if not measurements:
            return None
        return {
            "width": float(np.mean([item.width_mm for item in measurements])),
            "length": float(np.mean([item.length_mm for item in measurements])),
            "equivalent_diameter": float(
                np.mean([item.equivalent_diameter_mm for item in measurements])
            ),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "pallet_selection_path": self.pallet_selection_path,
            "processed_frame_count": len(self.frames),
            "total_fruit_observations": self.num_fruits,
            "average_fruit_size_mm": self.average_size_mm,
            "frames": [frame.to_dict() for frame in self.frames],
        }

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return destination


def _fruit_record(
    instance: FruitInstance, measurement: FruitMeasurement | None
) -> dict[str, object]:
    return {
        "fruit_id": instance.instance_id,
        "box": [round(float(value), 2) for value in instance.box],
        "category_name": instance.category_name,
        "detector_score": round(float(instance.detector_score), 4),
        "sam_score": round(float(instance.sam_score), 4),
        "size": measurement.to_dict() if measurement is not None else None,
    }


def draw_pallet_preview(image_bgr: np.ndarray, detector: PalletDetector) -> np.ndarray:
    """Draw the active pallet provider's corners for setup verification."""
    detection = detector.detect(image_bgr)
    if detection is None:
        raise ValueError("No pallet available for preview")
    canvas = image_bgr.copy()
    corners = np.round(detection.corners_px).astype(np.int32)
    cv2.polylines(canvas, [corners], True, (0, 220, 255), 3, cv2.LINE_AA)
    for label, point in zip(PALLET_CORNER_LABELS, corners):
        center = tuple(point)
        cv2.circle(canvas, center, 7, (0, 220, 255), -1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            label,
            (center[0] + 9, center[1] - 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )
    return canvas


class IntegratedFruitSizingPipeline:
    """Run manual pallet setup first, then detection and calibrated sizing."""

    def __init__(
        self,
        config: IntegratedPipelineConfig,
        *,
        pallet_detector: PalletDetector | None = None,
        detector=None,
        sam_predictor=None,
        model_loader: Callable[[PipelineConfig], tuple[object, object]] = load_models,
        detection_runner: Callable[..., list[FruitInstance]] = run_pipeline,
    ) -> None:
        if config.frame_step <= 0:
            raise ValueError("frame_step must be positive")
        if config.max_frames is not None and config.max_frames <= 0:
            raise ValueError("max_frames must be positive when provided")
        if config.max_preview_size <= 0:
            raise ValueError("max_preview_size must be positive")
        if not config.pallet_type:
            raise ValueError("pallet_type cannot be empty")
        if config.input_rotation not in INPUT_ROTATIONS:
            raise ValueError(f"input_rotation must be one of {INPUT_ROTATIONS}")
        if not 0.0 <= config.min_pallet_overlap <= 1.0:
            raise ValueError("min_pallet_overlap must be between 0 and 1")
        self.config = config
        self.pallet_detector = pallet_detector
        self._pallet_detector_injected = pallet_detector is not None
        self.detector = detector
        self.sam_predictor = sam_predictor
        self._model_loader = model_loader
        self._detection_runner = detection_runner
        self._sizing_pipeline: SizeEstimationPipeline | None = None
        self._calibration_resolution: tuple[int, int] | None = None

    def prepare_pallet(self, image_bgr: np.ndarray) -> PalletDetector:
        """Load or collect pallet corners, validate them, and save a preview.

        This method intentionally runs before model loading.  A dashboard can
        call it as its setup step, then call ``run`` once the user accepts the
        generated preview.
        """
        selection_path = Path(self.config.pallet_selection_path)
        if not self._pallet_detector_injected and not self.config.reuse_pallet_selection:
            self.pallet_detector = None
        if self.pallet_detector is None:
            if self.config.pallet_points_file is not None:
                corners = load_points(self.config.pallet_points_file)
                self.pallet_detector = ManualPalletDetector(
                    corners,
                    self.config.pallet_type,
                    image_resolution=(image_bgr.shape[1], image_bgr.shape[0]),
                )
                self.pallet_detector.save(selection_path)
            elif self.config.reuse_pallet_selection and selection_path.is_file():
                self.pallet_detector = ManualPalletDetector.load(selection_path)
            else:
                corners = select_points(
                    image_bgr,
                    title="Pallet corners: TL, TR, BR, BL",
                    labels=PALLET_CORNER_LABELS,
                    exact_count=4,
                    max_display_size=self.config.max_preview_size,
                )
                self.pallet_detector = ManualPalletDetector(
                    corners,
                    self.config.pallet_type,
                    image_resolution=(image_bgr.shape[1], image_bgr.shape[0]),
                )
                self.pallet_detector.save(selection_path)
        elif isinstance(self.pallet_detector, ManualPalletDetector):
            self.pallet_detector.save(selection_path)

        # detect() validates resolution and any provider-specific constraints.
        detection = self.pallet_detector.detect(image_bgr)
        if detection is None:
            raise PalletGeometryError("No pallet detected during setup")
        if (
            isinstance(self.pallet_detector, ManualPalletDetector)
            and detection.pallet_type != self.config.pallet_type
        ):
            raise PalletGeometryError(
                f"Saved pallet type '{detection.pallet_type}' does not match requested "
                f"'{self.config.pallet_type}'. Provide --pallet-points-file or a different "
                "--pallet-selection to create a new selection."
            )
        preview = draw_pallet_preview(image_bgr, self.pallet_detector)
        preview_path = Path(self.config.pallet_selection_path).with_name(
            f"{Path(self.config.pallet_selection_path).stem}_preview.png"
        )
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(preview_path), preview):
            raise OSError(f"Cannot write pallet preview: {preview_path}")
        self._sizing_pipeline = SizeEstimationPipeline(self.config.sizing, self.pallet_detector)
        logger.info(
            "Pallet setup ready: %s (preview: %s)",
            self.config.pallet_selection_path,
            preview_path,
        )
        return self.pallet_detector

    def run(self, source: str | Path) -> MediaResult:
        source_path = Path(source)
        if source_path.suffix.lower() in IMAGE_EXTENSIONS:
            return self.run_image(source_path)
        return self.run_video(source_path)

    def run_image(self, image_path: str | Path) -> MediaResult:
        path = Path(image_path)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        original_shape = image.shape[:2]
        image = self._normalize_frame(image)
        processing_path = path
        if image.shape[:2] != original_shape:
            input_dir = Path(self.config.detection.output_dir) / "normalized_inputs"
            input_dir.mkdir(parents=True, exist_ok=True)
            processing_path = input_dir / f"{path.stem}.png"
            if not cv2.imwrite(str(processing_path), image):
                raise OSError(f"Cannot write normalized input: {processing_path}")
        self.prepare_pallet(image)
        self._ensure_models(str(processing_path))
        frame = self._process_frame(
            image,
            processing_path,
            None,
            None,
            Path(self.config.detection.output_dir),
        )
        result = MediaResult(str(path), str(self.config.pallet_selection_path), [frame])
        result.save(Path(self.config.detection.output_dir) / f"{path.stem}_summary.json")
        return result

    def run_video(self, video_path: str | Path) -> MediaResult:
        path = Path(video_path)
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise FileNotFoundError(f"Cannot open video: {path}")

        frames: list[FrameResult] = []
        try:
            ok, first_frame = capture.read()
            if not ok or first_frame is None:
                raise ValueError(f"Video contains no readable frames: {path}")
            self.prepare_pallet(self._normalize_frame(first_frame))
            self._ensure_models(str(path))

            frame_index = 0
            frame = first_frame
            ok = True
            while ok:
                if frame_index % self.config.frame_step == 0:
                    processing_frame = self._normalize_frame(frame)
                    artifact_dir = (
                        Path(self.config.detection.output_dir)
                        / "frames"
                        / f"frame_{frame_index:06d}"
                    )
                    artifact_dir.mkdir(parents=True, exist_ok=True)
                    frame_path = artifact_dir / f"{path.stem}_frame_{frame_index:06d}.jpg"
                    if not cv2.imwrite(str(frame_path), processing_frame):
                        raise OSError(f"Cannot write sampled frame: {frame_path}")
                    frames.append(
                        self._process_frame(
                            processing_frame,
                            frame_path,
                            frame_index,
                            _finite_float_or_none(capture.get(cv2.CAP_PROP_POS_MSEC)),
                            artifact_dir,
                        )
                    )
                    if self.config.max_frames is not None and len(frames) >= self.config.max_frames:
                        break
                ok, frame = capture.read()
                frame_index += 1
        finally:
            capture.release()

        result = MediaResult(str(path), str(self.config.pallet_selection_path), frames)
        result.save(Path(self.config.detection.output_dir) / f"{path.stem}_summary.json")
        return result

    def _normalize_frame(self, image_bgr: np.ndarray) -> np.ndarray:
        if not self.config.resize_to_calibration:
            return image_bgr
        if self._calibration_resolution is None:
            calibration = CalibrationStore(self.config.sizing.calibration_dir).load(
                self.config.sizing.camera_id,
                self.config.sizing.camera_group,
            )
            self._calibration_resolution = calibration.resolution
        normalized, applied_rotation = normalize_to_resolution(
            image_bgr,
            self._calibration_resolution,
            rotation=self.config.input_rotation,
            allow_aspect_mismatch=self.config.allow_unsafe_resize,
        )
        if normalized is not image_bgr:
            logger.warning(
                "Temporary input normalization: %dx%d -> %dx%d (rotation=%s)",
                image_bgr.shape[1],
                image_bgr.shape[0],
                normalized.shape[1],
                normalized.shape[0],
                applied_rotation,
            )
        return normalized

    def _ensure_models(self, image_path: str) -> None:
        if self.detector is None or self.sam_predictor is None:
            model_config = replace(self.config.detection, image_path=image_path)
            self.detector, self.sam_predictor = self._model_loader(model_config)

    def _process_frame(
        self,
        image_bgr: np.ndarray,
        image_path: Path,
        frame_index: int | None,
        timestamp_ms: float | None,
        artifact_dir: Path,
    ) -> FrameResult:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        detection_config = replace(
            self.config.detection,
            image_path=str(image_path),
            output_dir=str(artifact_dir),
        )
        full_image_instances = self._detection_runner(
            detection_config,
            detector=self.detector,
            sam_predictor=self.sam_predictor,
        )
        pallet_detection = self.pallet_detector.detect(image_bgr)
        if pallet_detection is None:
            raise PalletGeometryError("No pallet detected while filtering fruit")
        instances = filter_instances_to_pallet(
            full_image_instances,
            pallet_detection.corners_px,
            min_overlap=self.config.min_pallet_overlap,
        )
        logger.info(
            "Pallet-region filter kept %d/%d fruit(s) (minimum mask overlap %.2f)",
            len(instances),
            len(full_image_instances),
            self.config.min_pallet_overlap,
        )
        if self._sizing_pipeline is None:
            raise RuntimeError("Pallet setup must complete before frame processing")
        sizing_result = self._sizing_pipeline.run(image_bgr, instances)
        sizing_result.save(artifact_dir, image_path.stem)
        frame_result = FrameResult(
            source_image=str(image_path),
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            instances=instances,
            sizing=sizing_result,
            artifact_dir=str(artifact_dir),
            full_image_num_fruits=len(full_image_instances),
        )
        result_path = artifact_dir / f"{image_path.stem}_result.json"
        result_path.write_text(json.dumps(frame_result.to_dict(), indent=2) + "\n", encoding="utf-8")
        logger.info(
            "%s: detected %d fruit(s), measured %d",
            image_path.name,
            frame_result.num_fruits,
            len(sizing_result.measurements),
        )
        return frame_result


def _finite_float_or_none(value: float) -> float | None:
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


def filter_instances_to_pallet(
    instances: list[FruitInstance],
    corners_px: np.ndarray,
    *,
    min_overlap: float = 0.5,
) -> list[FruitInstance]:
    """Keep fruit whose mask area lies sufficiently inside the pallet polygon."""
    if not 0.0 <= min_overlap <= 1.0:
        raise ValueError("min_overlap must be between 0 and 1")
    if not instances:
        return []
    image_shape = np.asarray(instances[0].mask).shape
    if len(image_shape) != 2:
        raise ValueError("Fruit masks must be two-dimensional")
    pallet_mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.fillConvexPoly(pallet_mask, np.round(corners_px).astype(np.int32), 1)
    pallet_region = pallet_mask.astype(bool)

    kept: list[FruitInstance] = []
    for instance in instances:
        mask = np.asarray(instance.mask, dtype=bool)
        if mask.shape != image_shape:
            raise ValueError("All fruit masks must have the same image shape")
        area = int(mask.sum())
        if area == 0:
            continue
        overlap = int(np.count_nonzero(mask & pallet_region)) / area
        if overlap >= min_overlap:
            kept.append(instance)
    return kept


def normalize_to_resolution(
    image_bgr: np.ndarray,
    target_resolution: tuple[int, int],
    *,
    rotation: str = "auto",
    aspect_tolerance: float = 0.01,
    allow_aspect_mismatch: bool = False,
) -> tuple[np.ndarray, str]:
    """Rotate and resize an image to a calibration resolution without stretching.

    ``target_resolution`` is ``(width, height)``. Auto rotation chooses a
    clockwise quarter-turn only when it makes the source aspect ratio closer
    to the target. Use an explicit direction when camera orientation metadata
    is unavailable or auto chooses the wrong physical orientation.
    """
    if rotation not in INPUT_ROTATIONS:
        raise ValueError(f"rotation must be one of {INPUT_ROTATIONS}")
    target_width, target_height = (int(value) for value in target_resolution)
    if target_width <= 0 or target_height <= 0:
        raise ValueError("target_resolution must contain positive dimensions")
    if image_bgr.ndim < 2:
        raise ValueError("image must have at least two dimensions")

    source_height, source_width = image_bgr.shape[:2]
    if (source_width, source_height) == (target_width, target_height) and rotation in {
        "auto",
        "none",
    }:
        return image_bgr, "none"

    applied_rotation = rotation
    if rotation == "auto":
        target_aspect = target_width / target_height
        direct_error = abs(source_width / source_height - target_aspect)
        rotated_error = abs(source_height / source_width - target_aspect)
        applied_rotation = "clockwise" if rotated_error < direct_error else "none"

    rotation_codes = {
        "clockwise": cv2.ROTATE_90_CLOCKWISE,
        "counterclockwise": cv2.ROTATE_90_COUNTERCLOCKWISE,
        "180": cv2.ROTATE_180,
    }
    oriented = (
        cv2.rotate(image_bgr, rotation_codes[applied_rotation])
        if applied_rotation in rotation_codes
        else image_bgr
    )
    oriented_height, oriented_width = oriented.shape[:2]
    source_aspect = oriented_width / oriented_height
    target_aspect = target_width / target_height
    relative_aspect_error = abs(source_aspect - target_aspect) / target_aspect
    if relative_aspect_error > aspect_tolerance and not allow_aspect_mismatch:
        raise ValueError(
            f"Cannot safely resize {source_width}x{source_height} to calibration resolution "
            f"{target_width}x{target_height}: aspect ratios differ after rotation "
            f"'{applied_rotation}'. Cropping or stretching would invalidate measurements."
        )
    if relative_aspect_error > aspect_tolerance:
        logger.warning(
            "Unsafe testing resize is stretching aspect ratio from %.4f to %.4f; "
            "physical measurements are invalid",
            source_aspect,
            target_aspect,
        )
    if (oriented_width, oriented_height) == (target_width, target_height):
        return oriented, applied_rotation
    interpolation = (
        cv2.INTER_AREA
        if oriented_width > target_width or oriented_height > target_height
        else cv2.INTER_LINEAR
    )
    return (
        cv2.resize(oriented, (target_width, target_height), interpolation=interpolation),
        applied_rotation,
    )
