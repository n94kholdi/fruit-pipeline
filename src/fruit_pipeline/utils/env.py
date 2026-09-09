"""Conventional boolean environment-variable helpers."""

from __future__ import annotations

import os


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
