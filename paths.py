"""Project-local path helpers."""

from __future__ import annotations

from pathlib import Path

# Repo root (this package maps fruit_pipeline -> ".").
_PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = _PROJECT_ROOT / "models"


def resolve_model_path(path: str) -> str:
    """Resolve a weights/checkpoint path, preferring ``models/`` in this repo.

    - Existing file at ``path`` (relative to cwd or absolute) -> use it.
    - Bare filename (``yolo11x.pt``) or missing ``models/...`` -> try
      ``<repo>/models/<basename>`` if that file exists.
    - Otherwise anchor bare filenames under ``models/`` so Ultralytics (etc.)
      download into ``models/`` instead of the process cwd.
    """
    p = Path(path)
    if p.is_file():
        return str(p.resolve())

    by_name = MODELS_DIR / p.name
    if by_name.is_file():
        return str(by_name)

    if not p.is_absolute() and p.parent in (Path("."), Path("models")):
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        return str(by_name.resolve())

    return str(p)
