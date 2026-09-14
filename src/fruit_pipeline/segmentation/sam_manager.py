"""Persistent, reusable SAM model lifecycle and box-prompt inference.

The manager deliberately separates image encoding from prompt decoding.  That
keeps the expensive image embedding reusable and provides the seam needed by a
future video tracker or embedding cache without coupling either concern to the
CLI/API layer.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch

from fruit_pipeline.utils.paths import resolve_model_path

logger = logging.getLogger(__name__)

SAM_MODEL_TYPES = ("vit_b", "vit_l", "vit_h")


def env_flag(name: str, default: bool) -> bool:
    """Read a conventional boolean environment variable."""
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be one of true/false, 1/0, yes/no, or on/off")


@dataclass
class SAMInferenceTimings:
    """Wall-clock stage timings in milliseconds.

    CUDA is synchronized only when profiling is requested.  Routine inference
    therefore does not pay for otherwise unnecessary synchronization points.
    """

    model_loading_ms: float = 0.0
    preprocessing_ms: float = 0.0
    image_encoder_ms: float = 0.0
    prompt_encoder_ms: float = 0.0
    mask_decoder_ms: float = 0.0
    postprocessing_ms: float = 0.0
    total_inference_ms: float = 0.0
    gpu_memory_allocated_mb: float = 0.0
    gpu_memory_reserved_mb: float = 0.0

    @property
    def effective_fps(self) -> float:
        return 1000.0 / self.total_inference_ms if self.total_inference_ms > 0 else 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "model_loading_ms": self.model_loading_ms,
            "preprocessing_ms": self.preprocessing_ms,
            "image_encoder_ms": self.image_encoder_ms,
            "prompt_encoder_ms": self.prompt_encoder_ms,
            "mask_decoder_ms": self.mask_decoder_ms,
            "postprocessing_ms": self.postprocessing_ms,
            "total_inference_ms": self.total_inference_ms,
            "effective_fps": self.effective_fps,
            "gpu_memory_allocated_mb": self.gpu_memory_allocated_mb,
            "gpu_memory_reserved_mb": self.gpu_memory_reserved_mb,
        }


@dataclass
class SAMBoxResult:
    masks: np.ndarray
    scores: np.ndarray
    timings: SAMInferenceTimings


@dataclass(frozen=True)
class SAMImageEmbedding:
    """Reusable encoder output plus the sizes required for mask projection."""

    features: torch.Tensor
    original_size: tuple[int, int]
    input_size: tuple[int, int]


class _StageTimer:
    def __init__(self, manager: "SAMModelManager", timings: SAMInferenceTimings, field: str):
        self.manager = manager
        self.timings = timings
        self.field = field
        self.started = 0.0

    def __enter__(self) -> None:
        if self.manager.profile:
            self.manager._synchronize()
            self.started = time.perf_counter()

    def __exit__(self, *_args: object) -> None:
        if self.manager.profile:
            self.manager._synchronize()
            elapsed = (time.perf_counter() - self.started) * 1000.0
            setattr(self.timings, self.field, getattr(self.timings, self.field) + elapsed)


class SAMModelManager:
    """Own one SAM model and reuse it for image and video inference.

    ``run_inference`` is serialized because SAM's predictor state (the current
    image embedding and image dimensions) is mutable.  This prevents concurrent
    requests from replacing one another's embeddings while still allowing the
    same GPU allocation and checkpoint to serve every request in the process.
    """

    def __init__(
        self,
        checkpoint: str,
        model_type: str = "vit_l",
        device: str = "cuda",
        use_fp16: bool = True,
        *,
        profile: bool = False,
    ) -> None:
        if model_type not in SAM_MODEL_TYPES:
            raise ValueError(f"Unknown SAM model_type '{model_type}', expected one of {SAM_MODEL_TYPES}")
        self.checkpoint = resolve_model_path(checkpoint)
        self.model_type = model_type
        self.device = torch.device(device)
        self.use_fp16 = bool(use_fp16 and self.device.type == "cuda")
        self.profile = profile
        self._model = None
        self._predictor = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.RLock()
        self.model_loading_ms = 0.0
        if use_fp16 and self.device.type != "cuda":
            logger.info("SAM FP16 requested on %s; using FP32 because autocast is CUDA-only", self.device)

    def load_model(self):
        """Load once, disable gradients, move to the target device, and cache."""
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is not None:
                return self._model
            if not os.path.isfile(self.checkpoint):
                raise FileNotFoundError(
                    f"SAM checkpoint not found: {self.checkpoint}\n"
                    "Download the matching checkpoint from "
                    "https://github.com/facebookresearch/segment-anything#model-checkpoints "
                    "or configure an existing checkpoint."
                )
            from segment_anything import SamPredictor, sam_model_registry

            started = time.perf_counter()
            model = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
            model.eval()
            model.requires_grad_(False)
            model.to(device=self.device)
            self._model = model
            self._predictor = SamPredictor(model)
            self.model_loading_ms = (time.perf_counter() - started) * 1000.0
            logger.info(
                "Loaded persistent SAM (%s) from %s on %s in %.1f ms (fp16=%s)",
                self.model_type,
                self.checkpoint,
                self.device,
                self.model_loading_ms,
                self.use_fp16,
            )
            return model

    def get_model(self):
        return self.load_model()

    def get_predictor(self):
        self.load_model()
        return self._predictor

    def _autocast(self):
        if self.use_fp16:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def _synchronize(self) -> None:
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    def _timer(self, timings: SAMInferenceTimings, field: str) -> _StageTimer:
        return _StageTimer(self, timings, field)

    @torch.inference_mode()
    def encode_image(self, image_rgb: np.ndarray, timings: SAMInferenceTimings | None = None):
        """Preprocess and encode one RGB image, leaving its embedding reusable."""
        predictor = self.get_predictor()
        timings = timings or SAMInferenceTimings()
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("SAM input must be an HxWx3 RGB image")

        with self._inference_lock, self._autocast():
            with self._timer(timings, "preprocessing_ms"):
                transformed = predictor.transform.apply_image(image_rgb)
                tensor = torch.as_tensor(transformed, device=self.device)
                tensor = tensor.permute(2, 0, 1).contiguous()[None, :, :, :]
                predictor.reset_image()
                predictor.original_size = image_rgb.shape[:2]
                predictor.input_size = tuple(tensor.shape[-2:])
                model_input = predictor.model.preprocess(tensor)
            with self._timer(timings, "image_encoder_ms"):
                predictor.features = predictor.model.image_encoder(model_input)
                predictor.is_image_set = True
            return SAMImageEmbedding(
                features=predictor.features,
                original_size=tuple(predictor.original_size),
                input_size=tuple(predictor.input_size),
            )

    def activate_embedding(self, embedding: SAMImageEmbedding) -> None:
        """Restore a cached embedding before decoding new prompts."""
        predictor = self.get_predictor()
        with self._inference_lock:
            predictor.features = embedding.features
            predictor.original_size = embedding.original_size
            predictor.input_size = embedding.input_size
            predictor.is_image_set = True

    def decode_embedding_boxes(
        self,
        embedding: SAMImageEmbedding,
        boxes: np.ndarray,
        *,
        batch_size: int = 16,
        timings: SAMInferenceTimings | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Atomically activate a cached image embedding and decode its boxes."""
        with self._inference_lock:
            self.activate_embedding(embedding)
            return self.decode_boxes(boxes, batch_size=batch_size, timings=timings)

    @torch.inference_mode()
    def decode_boxes(
        self,
        boxes: np.ndarray,
        *,
        batch_size: int = 16,
        timings: SAMInferenceTimings | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Decode box prompts against the embedding from ``encode_image``."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        predictor = self.get_predictor()
        if not predictor.is_image_set:
            raise RuntimeError("Call encode_image before decode_boxes")
        timings = timings or SAMInferenceTimings()
        boxes_np = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        if len(boxes_np) == 0:
            height, width = predictor.original_size
            return np.empty((0, height, width), dtype=bool), np.empty((0,), dtype=np.float32)

        with self._inference_lock, self._autocast():
            boxes_gpu = torch.as_tensor(boxes_np, device=self.device)
            transformed_boxes = predictor.transform.apply_boxes_torch(boxes_gpu, predictor.original_size)
            mask_chunks: list[np.ndarray] = []
            score_chunks: list[np.ndarray] = []
            for start in range(0, len(boxes_np), batch_size):
                chunk = transformed_boxes[start : start + batch_size]
                with self._timer(timings, "prompt_encoder_ms"):
                    sparse_embeddings, dense_embeddings = predictor.model.prompt_encoder(
                        points=None,
                        boxes=chunk,
                        masks=None,
                    )
                with self._timer(timings, "mask_decoder_ms"):
                    low_res_masks, iou_predictions = predictor.model.mask_decoder(
                        image_embeddings=predictor.features,
                        image_pe=predictor.model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                    )
                with self._timer(timings, "postprocessing_ms"):
                    masks = predictor.model.postprocess_masks(
                        low_res_masks,
                        predictor.input_size,
                        predictor.original_size,
                    )
                    masks = masks > predictor.model.mask_threshold
                    mask_chunks.append(masks.squeeze(1).cpu().numpy())
                    score_chunks.append(iou_predictions.squeeze(1).float().cpu().numpy())
        return np.concatenate(mask_chunks), np.concatenate(score_chunks)

    def run_inference(
        self,
        image_rgb: np.ndarray,
        boxes: np.ndarray,
        *,
        batch_size: int = 16,
    ) -> SAMBoxResult:
        """Encode an image once and decode all its box prompts."""
        self.load_model()
        timings = SAMInferenceTimings(model_loading_ms=self.model_loading_ms)
        with self._inference_lock:
            if self.profile:
                self._synchronize()
                if self.device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats(self.device)
            started = time.perf_counter()
            self.encode_image(image_rgb, timings)
            masks, scores = self.decode_boxes(boxes, batch_size=batch_size, timings=timings)
            if self.profile:
                self._synchronize()
            timings.total_inference_ms = (time.perf_counter() - started) * 1000.0
            if self.device.type == "cuda" and torch.cuda.is_available():
                timings.gpu_memory_allocated_mb = torch.cuda.max_memory_allocated(self.device) / 1024**2
                timings.gpu_memory_reserved_mb = torch.cuda.max_memory_reserved(self.device) / 1024**2
        return SAMBoxResult(masks=masks, scores=scores, timings=timings)

    @contextmanager
    def inference_context(self) -> Iterator[None]:
        """Lock and precision context for callers such as automatic-mask generation."""
        with self._inference_lock, torch.inference_mode(), self._autocast():
            yield


_MANAGERS: dict[tuple[str, str, str, bool], SAMModelManager] = {}
_MANAGERS_LOCK = threading.Lock()


def get_sam_model_manager(
    checkpoint: str,
    model_type: str = "vit_l",
    device: str = "cuda",
    use_fp16: bool = True,
    *,
    eager: bool = True,
) -> SAMModelManager:
    """Return the process-wide manager for one model/device/precision tuple."""
    resolved = resolve_model_path(checkpoint)
    key = (resolved, model_type, str(torch.device(device)), bool(use_fp16 and torch.device(device).type == "cuda"))
    with _MANAGERS_LOCK:
        manager = _MANAGERS.get(key)
        if manager is None:
            manager = SAMModelManager(resolved, model_type, device, use_fp16)
            _MANAGERS[key] = manager
    if eager:
        manager.load_model()
    return manager


def clear_sam_model_cache() -> None:
    """Clear manager references (primarily useful for tests and controlled refreshes)."""
    with _MANAGERS_LOCK:
        _MANAGERS.clear()
