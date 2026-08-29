"""Detector backend abstraction: one interface, three implementations.

Tiling (``detect.py``) and merging (``merge.py``) only ever talk to a
``DetectorBackend``'s ``detect()`` method, which always returns
already-full-image-shifted ``sahi.prediction.ObjectPrediction``s — so
neither of those stages needs to change no matter which backend runs.

- ``SahiUltralyticsBackend``: today's path (default plain detector, or
  YOLO-World), unchanged — wraps SAHI's ``AutoDetectionModel`` +
  ``get_prediction()``.
- ``YoloeBackend``: native Ultralytics YOLOE (ICCV 2025), supporting all
  three of its prompt modes. Bypasses SAHI's detection-model wrapper
  entirely (it has no notion of YOLOE's ``visual_prompts``/``refer_image``
  kwargs), calling ``ultralytics.YOLOE.predict()`` directly per tile and
  building ``ObjectPrediction``s by hand, shifted into full-image
  coordinates itself.

Visual-prompt mode is genuinely cross-image: a user-supplied exemplar crop
(``--visual-prompt path/to/crop.png``, one tight crop of a single instance)
is passed as YOLOE's ``refer_image``, with a single box
``[0, 0, crop_width, crop_height]`` as the "example" region on it — the
whole crop *is* the example. Multiple exemplars run as multiple
visual-prompt passes per tile (one per exemplar), unioned into the raw
prediction list like any other raw detection, same as running more tiles.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
from sahi.annotation import Category
from sahi.auto_model import AutoDetectionModel
from sahi.models.base import DetectionModel
from sahi.predict import get_prediction
from sahi.prediction import ObjectPrediction

from fruit_pipeline.paths import resolve_model_path

logger = logging.getLogger(__name__)

YOLOE_MODES = ("text", "visual", "prompt_free")


class DetectorBackend(ABC):
    """Common interface: run detection on one tile, return full-image-shifted predictions."""

    @abstractmethod
    def detect(
        self,
        tile_rgb: np.ndarray,
        shift: tuple[int, int],
        full_shape: tuple[int, int],
        conf_threshold: float,
    ) -> list[ObjectPrediction]:
        """Detect on one tile crop.

        Args:
            tile_rgb: The tile's own crop, RGB.
            shift: ``(shift_x, shift_y)`` — this tile's top-left corner in
                full-image coordinates. Returned boxes must already be
                shifted by this (i.e. in full-image coordinates), matching
                ``ObjectPrediction.get_shifted_object_prediction()``'s
                convention elsewhere in this pipeline.
            full_shape: ``(height, width)`` of the full image (SAHI's own
                axis order, used for box-clamping).
            conf_threshold: Per-call confidence threshold.
        """


class SahiUltralyticsBackend(DetectorBackend):
    """Wraps SAHI's ``AutoDetectionModel`` + ``get_prediction()`` — today's default/YOLO-World path."""

    def __init__(self, detection_model: DetectionModel, num_fruit_classes: int | None = None):
        """``num_fruit_classes``: if set (YOLO-World multi-class prompting with a background
        class appended), predictions whose ``category_id >= num_fruit_classes``
        are dropped here — they matched a background/other prompt, not fruit.
        """
        self.detection_model = detection_model
        self.num_fruit_classes = num_fruit_classes

    def detect(self, tile_rgb, shift, full_shape, conf_threshold):
        shift_x, shift_y = shift
        result = get_prediction(
            image=tile_rgb,
            detection_model=self.detection_model,
            shift_amount=[shift_x, shift_y],
            full_shape=list(full_shape),
            confidence_threshold=conf_threshold,
            verbose=0,
        )
        predictions = [pred.get_shifted_object_prediction() for pred in result.object_prediction_list]
        if self.num_fruit_classes is not None:
            predictions = [p for p in predictions if p.category.id < self.num_fruit_classes]
        return predictions


class YoloeBackend(DetectorBackend):
    """Native Ultralytics YOLOE: text, visual, or prompt-free prompting."""

    def __init__(
        self,
        model,
        mode: str,
        category_name: str = "fruit",
        text_prompts: list[str] | None = None,
        num_fruit_classes: int | None = None,
        visual_prompt_paths: list[str] | None = None,
    ):
        if mode not in YOLOE_MODES:
            raise ValueError(f"Unknown YOLOE mode {mode!r}, expected one of {YOLOE_MODES}")
        self.model = model
        self.mode = mode
        self.category_name = category_name
        self.num_fruit_classes = num_fruit_classes
        self._exemplars: list[np.ndarray] = []

        if mode == "text":
            if not text_prompts:
                raise ValueError("YOLOE text mode requires text_prompts")
            self.model.set_classes(text_prompts)
            if self.num_fruit_classes is None:
                self.num_fruit_classes = len(text_prompts)
            logger.info("YOLOE text mode: %d fruit prompt(s), %d background prompt(s)", self.num_fruit_classes, len(text_prompts) - self.num_fruit_classes)
        elif mode == "visual":
            if not visual_prompt_paths:
                raise ValueError("YOLOE visual mode requires at least one --visual-prompt exemplar crop")
            import cv2

            for path in visual_prompt_paths:
                image_bgr = cv2.imread(path)
                if image_bgr is None:
                    raise FileNotFoundError(f"Cannot read visual-prompt exemplar image: {path}")
                self._exemplars.append(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
            logger.info("YOLOE visual mode: %d exemplar crop(s)", len(self._exemplars))
        elif mode == "prompt_free":
            # No set_classes() call -- the checkpoint must already be a
            # prompt-free variant (Ultralytics' *-seg-pf.pt); calling
            # set_classes() on one raises inside ultralytics itself.
            logger.info("YOLOE prompt-free mode: using the checkpoint's built-in vocabulary")

    def detect(self, tile_rgb, shift, full_shape, conf_threshold):
        shift_x, shift_y = shift

        if self.mode in ("text", "prompt_free"):
            results = self.model.predict(tile_rgb, conf=conf_threshold, verbose=False)
            predictions = self._results_to_predictions(results, shift_x, shift_y, filter_background=self.mode == "text")
            return predictions

        # visual: one predict() call per exemplar, unioned like extra tiles.
        from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

        predictions: list[ObjectPrediction] = []
        for exemplar in self._exemplars:
            eh, ew = exemplar.shape[:2]
            visual_prompts = {"bboxes": np.array([[0, 0, ew, eh]], dtype=float), "cls": np.array([0])}
            results = self.model.predict(
                tile_rgb,
                refer_image=exemplar,
                visual_prompts=visual_prompts,
                predictor=YOLOEVPSegPredictor,
                conf=conf_threshold,
                verbose=False,
            )
            predictions.extend(self._results_to_predictions(results, shift_x, shift_y, filter_background=False))
        return predictions

    def _results_to_predictions(self, results, shift_x: int, shift_y: int, filter_background: bool) -> list[ObjectPrediction]:
        predictions: list[ObjectPrediction] = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.detach().cpu().numpy()
            conf = boxes.conf.detach().cpu().numpy()
            cls = boxes.cls.detach().cpu().numpy().astype(int)
            for (x1, y1, x2, y2), score, class_idx in zip(xyxy, conf, cls):
                if filter_background and self.num_fruit_classes is not None and class_idx >= self.num_fruit_classes:
                    continue
                predictions.append(
                    ObjectPrediction(
                        bbox=[float(x1 + shift_x), float(y1 + shift_y), float(x2 + shift_x), float(y2 + shift_y)],
                        category_id=0,
                        category_name=self.category_name,
                        score=float(score),
                        shift_amount=[0, 0],
                    )
                )
        return predictions


def _build_sahi_detection_model(
    weights_path: str,
    device: str,
    conf_threshold: float,
    prompt_classes: list[str] | None,
) -> DetectionModel:
    """The plain-detector / YOLO-World loading logic, factored out of the old ``load_detector``."""
    from ultralytics import YOLO

    model = YOLO(weights_path)
    if prompt_classes:
        logger.info("Loading YOLO-World with prompt classes: %s", prompt_classes)
        model.set_classes(prompt_classes)

    return AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model=model,
        device=device,
        confidence_threshold=conf_threshold,
    )


def load_detector_backend(
    detector: str,
    weights_path: str,
    device: str = "cpu",
    conf_threshold: float = 0.25,
    fruit_prompts: list[str] | None = None,
    background_prompts: list[str] | None = None,
    yoloe_mode: str = "text",
    visual_prompt_paths: list[str] | None = None,
) -> DetectorBackend:
    """Load a ``DetectorBackend`` by name.

    Args:
        detector: ``"default"`` (plain, class-agnostic), ``"yolo-world"``,
            or ``"yoloe"``.
        weights_path: Ultralytics checkpoint path.
        fruit_prompts / background_prompts: text prompt lists (see
            ``prompts.PromptConfig``) — used by ``yolo-world`` and
            ``yoloe`` text mode. ``background_prompts`` detections are
            dropped before they're ever returned from ``detect()``.
        yoloe_mode / visual_prompt_paths: ``yoloe`` only.
    """
    weights_path = resolve_model_path(weights_path)
    if detector in ("default", "yolo-world"):
        prompt_classes = None
        num_fruit_classes = None
        if detector == "yolo-world":
            prompt_classes = [*(fruit_prompts or []), *(background_prompts or [])]
            num_fruit_classes = len(fruit_prompts or [])
        detection_model = _build_sahi_detection_model(weights_path, device, conf_threshold, prompt_classes)
        return SahiUltralyticsBackend(detection_model, num_fruit_classes=num_fruit_classes if detector == "yolo-world" else None)

    if detector == "yoloe":
        from ultralytics import YOLOE

        model = YOLOE(weights_path)
        model.to(device)
        text_prompts = None
        num_fruit_classes = None
        if yoloe_mode == "text":
            text_prompts = [*(fruit_prompts or []), *(background_prompts or [])]
            num_fruit_classes = len(fruit_prompts or [])
        return YoloeBackend(
            model=model,
            mode=yoloe_mode,
            text_prompts=text_prompts,
            num_fruit_classes=num_fruit_classes,
            visual_prompt_paths=visual_prompt_paths,
        )

    raise ValueError(f"Unknown --detector '{detector}', expected one of ('default', 'yolo-world', 'yoloe')")


def _relabel_class_agnostic(predictions: list[ObjectPrediction], label: str = "fruit") -> None:
    """Relabel every prediction's category in-place so merging treats them as one class.

    Only meaningful for the plain default detector (COCO's 80 classes
    collapsed to one "fruit" candidate class) — a no-op in effect for
    YOLO-World/YOLOE, whose predictions are already prompted to be
    fruit-relevant and already carry a single synthetic category.
    """
    for pred in predictions:
        pred.category = Category(id=0, name=label)
