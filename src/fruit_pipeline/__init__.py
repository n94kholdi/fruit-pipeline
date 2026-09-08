"""Pretrained-only detection and segmentation for densely packed fruit images.

Pipeline stages live in focused detection, segmentation, and visualization
subpackages. They are orchestrated by :mod:`fruit_pipeline.pipeline` and
exposed through :mod:`fruit_pipeline.cli`.
"""

__version__ = "0.1.0"
from fruit_pipeline.integrated_pipeline import (
    FrameResult,
    IntegratedFruitSizingPipeline,
    IntegratedPipelineConfig,
    MediaResult,
)

__all__ = [
    "FrameResult",
    "IntegratedFruitSizingPipeline",
    "IntegratedPipelineConfig",
    "MediaResult",
]
