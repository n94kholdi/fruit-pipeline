from pathlib import Path

from fruit_pipeline.utils.paths import MODELS_DIR, resolve_model_path


def test_resolve_model_path_finds_existing_models_dir_file():
    assert Path(resolve_model_path("yolo11x.pt")) == MODELS_DIR / "yolo11x.pt"


def test_resolve_model_path_keeps_explicit_models_relative_path():
    assert Path(resolve_model_path("models/yolo11x.pt")) == MODELS_DIR / "yolo11x.pt"


def test_resolve_model_path_anchors_missing_bare_filename_under_models():
    resolved = Path(resolve_model_path("nonexistent_weights.pt"))
    assert resolved.parent == MODELS_DIR
    assert resolved.name == "nonexistent_weights.pt"
