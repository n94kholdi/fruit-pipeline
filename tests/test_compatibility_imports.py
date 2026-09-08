"""The old flat module paths remain available during the src-layout migration."""

from fruit_pipeline.detect import detect_tiled as legacy_detect_tiled
from fruit_pipeline.detection.merging import merge_detections
from fruit_pipeline.detection.tiling import detect_tiled
from fruit_pipeline.merge import merge_detections as legacy_merge_detections


def test_flat_module_compatibility_imports_point_to_canonical_functions():
    assert legacy_detect_tiled is detect_tiled
    assert legacy_merge_detections is merge_detections
