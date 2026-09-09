"""Validated SAM2.1 deployment configuration.

Only the variant selected by ``FRUIT_PIPELINE_SAM2_MODEL`` is resolved and
loaded.  Keeping this catalog in one place prevents the API, downloader and
runtime from disagreeing about checkpoint/config pairs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from fruit_pipeline.utils.env import env_flag


@dataclass(frozen=True)
class SAM2Variant:
    name: str
    config: str
    filename: str
    url: str
    sha256: str


_BASE = "https://dl.fbaipublicfiles.com/segment_anything_2/092824"
SAM2_VARIANTS: dict[str, SAM2Variant] = {
    "sam2.1_hiera_tiny": SAM2Variant(
        "sam2.1_hiera_tiny", "configs/sam2.1/sam2.1_hiera_t.yaml",
        "sam2.1_hiera_tiny.pt", f"{_BASE}/sam2.1_hiera_tiny.pt",
        "7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69",
    ),
    "sam2.1_hiera_small": SAM2Variant(
        "sam2.1_hiera_small", "configs/sam2.1/sam2.1_hiera_s.yaml",
        "sam2.1_hiera_small.pt", f"{_BASE}/sam2.1_hiera_small.pt",
        "6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38",
    ),
    "sam2.1_hiera_base_plus": SAM2Variant(
        "sam2.1_hiera_base_plus", "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "sam2.1_hiera_base_plus.pt", f"{_BASE}/sam2.1_hiera_base_plus.pt",
        "a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5",
    ),
    "sam2.1_hiera_large": SAM2Variant(
        "sam2.1_hiera_large", "configs/sam2.1/sam2.1_hiera_l.yaml",
        "sam2.1_hiera_large.pt", f"{_BASE}/sam2.1_hiera_large.pt",
        "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318",
    ),
}
SAM2_MODEL_NAMES = tuple(SAM2_VARIANTS)
SAM2_PRECISIONS = ("bf16", "fp16", "fp32")
SAM2_RUNTIMES = ("pytorch", "hybrid", "tensorrt")


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    value = int(os.getenv(name, str(default)))
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    value = float(os.getenv(name, str(default)))
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


@dataclass(frozen=True)
class SAM2Config:
    model_name: str = "sam2.1_hiera_base_plus"
    checkpoint: str | None = None
    config_file: str | None = None
    device: str = "cuda"
    precision: str = "bf16"
    vos_optimized: bool = False
    runtime: str = "pytorch"
    tensorrt_engine_dir: str = "/models/tensorrt"
    tensorrt_precision: str = "fp16"
    tensorrt_workspace_bytes: int = 4 * 1024**3
    tensorrt_allow_fallback: bool = True

    refresh_seconds: float = 10.0
    refresh_processed_frames: int = 30
    refresh_jitter_seconds: float = 1.0
    min_refresh_interval_seconds: float = 2.0
    max_refresh_queue: int = 32
    max_cameras_per_gpu: int = 8
    max_active_objects_per_camera: int = 256
    max_total_active_objects: int = 1024
    max_concurrent_discoveries: int = 1
    max_gpu_queue: int = 64
    camera_idle_timeout_seconds: float = 60.0
    max_frame_history: int = 32
    missing_grace_refreshes: int = 2
    match_mask_iou: float = 0.35
    match_box_iou: float = 0.20
    match_centroid_distance: float = 2.0

    points_per_side: int = 32
    points_per_batch: int = 64
    pred_iou_thresh: float = 0.88
    stability_score_thresh: float = 0.95
    box_nms_thresh: float = 0.7
    min_mask_region_area: int = 30
    require_cuda_extension: bool = False

    def __post_init__(self) -> None:
        if self.model_name not in SAM2_VARIANTS:
            raise ValueError(f"Unknown SAM2 model '{self.model_name}'; expected one of {SAM2_MODEL_NAMES}")
        if self.precision not in SAM2_PRECISIONS:
            raise ValueError(f"SAM2 precision must be one of {SAM2_PRECISIONS}")
        if self.runtime not in SAM2_RUNTIMES:
            raise ValueError(f"SAM2 runtime must be one of {SAM2_RUNTIMES}")
        if self.runtime == "tensorrt" and self.tensorrt_precision != "fp16":
            raise ValueError("The experimental SAM2 TensorRT runtime supports FP16 only")
        for name in ("match_mask_iou", "match_box_iou"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between 0 and 1")

    @property
    def variant(self) -> SAM2Variant:
        return SAM2_VARIANTS[self.model_name]

    @property
    def resolved_checkpoint(self) -> str:
        return str(Path(self.checkpoint or f"/models/{self.variant.filename}").expanduser())

    @property
    def resolved_config(self) -> str:
        return self.config_file or self.variant.config

    def discovery_metadata(self) -> dict[str, object]:
        return {
            "points_per_side": self.points_per_side,
            "points_per_batch": self.points_per_batch,
            "pred_iou_thresh": self.pred_iou_thresh,
            "stability_score_thresh": self.stability_score_thresh,
            "box_nms_thresh": self.box_nms_thresh,
            "min_mask_region_area": self.min_mask_region_area,
        }

    @classmethod
    def from_env(cls) -> "SAM2Config":
        model = os.getenv("FRUIT_PIPELINE_SAM2_MODEL", "sam2.1_hiera_base_plus")
        return cls(
            model_name=model,
            checkpoint=os.getenv("FRUIT_PIPELINE_SAM2_CHECKPOINT") or None,
            config_file=os.getenv("FRUIT_PIPELINE_SAM2_CONFIG") or None,
            device=os.getenv("FRUIT_PIPELINE_DEVICE", "cuda"),
            precision=os.getenv("SAM2_PRECISION", "bf16").lower(),
            vos_optimized=env_flag("SAM2_VOS_OPTIMIZED", False),
            runtime=os.getenv("SAM2_RUNTIME", "pytorch").lower(),
            tensorrt_engine_dir=os.getenv("SAM2_TENSORRT_ENGINE_DIR", "/models/tensorrt"),
            tensorrt_precision=os.getenv("SAM2_TENSORRT_PRECISION", "fp16").lower(),
            tensorrt_workspace_bytes=_env_int("SAM2_TENSORRT_WORKSPACE_BYTES", 4 * 1024**3, 1),
            tensorrt_allow_fallback=env_flag("SAM2_TENSORRT_ALLOW_FALLBACK", True),
            refresh_seconds=_env_float("SAM2_REFRESH_SECONDS", 10.0),
            refresh_processed_frames=_env_int("SAM2_REFRESH_PROCESSED_FRAMES", 30),
            refresh_jitter_seconds=_env_float("SAM2_REFRESH_JITTER_SECONDS", 1.0),
            min_refresh_interval_seconds=_env_float("SAM2_MIN_REFRESH_INTERVAL_SECONDS", 2.0),
            max_refresh_queue=_env_int("SAM2_MAX_REFRESH_QUEUE", 32, 1),
            max_cameras_per_gpu=_env_int("SAM2_MAX_CAMERAS_PER_GPU", 8, 1),
            max_active_objects_per_camera=_env_int("SAM2_MAX_ACTIVE_OBJECTS_PER_CAMERA", 256, 1),
            max_total_active_objects=_env_int("SAM2_MAX_TOTAL_ACTIVE_OBJECTS", 1024, 1),
            max_concurrent_discoveries=_env_int("SAM2_MAX_CONCURRENT_DISCOVERIES", 1, 1),
            max_gpu_queue=_env_int("SAM2_MAX_GPU_QUEUE", 64, 1),
            camera_idle_timeout_seconds=_env_float("SAM2_CAMERA_IDLE_TIMEOUT_SECONDS", 60.0, 1),
            max_frame_history=_env_int("SAM2_MAX_FRAME_HISTORY", 32, 2),
            missing_grace_refreshes=_env_int("SAM2_MISSING_GRACE_REFRESHS", 2),
            require_cuda_extension=env_flag("SAM2_REQUIRE_CUDA_EXTENSION", False),
        )
