import argparse
from pathlib import Path

import pytest

from fruit_pipeline.inference import confidence_threshold, find_images


def test_confidence_threshold_accepts_bounds():
    assert confidence_threshold("0") == 0.0
    assert confidence_threshold("0.35") == 0.35
    assert confidence_threshold("1") == 1.0


@pytest.mark.parametrize("value", ["-0.1", "1.1"])
def test_confidence_threshold_rejects_out_of_range_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        confidence_threshold(value)


def test_find_images_is_non_recursive_and_sorted(tmp_path: Path):
    (tmp_path / "b.JPG").touch()
    (tmp_path / "a.png").touch()
    (tmp_path / "notes.txt").touch()
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "hidden.jpg").touch()

    assert [path.name for path in find_images(tmp_path)] == ["a.png", "b.JPG"]
