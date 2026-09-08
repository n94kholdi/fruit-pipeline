import sys
from types import SimpleNamespace

import pytest

from fruit_pipeline.segmentation.sam_manager import (
    SAMModelManager,
    clear_sam_model_cache,
    env_flag,
    get_sam_model_manager,
)


class _FakeModel:
    def __init__(self):
        self.calls = []

    def eval(self):
        self.calls.append("eval")
        return self

    def requires_grad_(self, enabled):
        self.calls.append(("requires_grad", enabled))
        return self

    def to(self, *, device):
        self.calls.append(("to", str(device)))
        return self


class _FakePredictor:
    def __init__(self, model):
        self.model = model


def test_manager_loads_checkpoint_once_and_disables_gradients(tmp_path, monkeypatch):
    checkpoint = tmp_path / "sam.pth"
    checkpoint.touch()
    loads = []

    def construct(*, checkpoint):
        loads.append(checkpoint)
        return _FakeModel()

    monkeypatch.setitem(
        sys.modules,
        "segment_anything",
        SimpleNamespace(SamPredictor=_FakePredictor, sam_model_registry={"vit_l": construct}),
    )
    manager = SAMModelManager(str(checkpoint), device="cpu", use_fp16=True)

    first = manager.load_model()
    second = manager.load_model()

    assert first is second
    assert loads == [str(checkpoint)]
    assert first.calls == ["eval", ("requires_grad", False), ("to", "cpu")]
    assert manager.use_fp16 is False


def test_process_cache_reuses_manager_without_loading(tmp_path):
    clear_sam_model_cache()
    checkpoint = tmp_path / "sam.pth"

    first = get_sam_model_manager(str(checkpoint), device="cpu", eager=False)
    second = get_sam_model_manager(str(checkpoint), device="cpu", eager=False)

    assert first is second


@pytest.mark.parametrize(
    ("value", "expected"),
    [("true", True), ("YES", True), ("0", False), ("off", False)],
)
def test_env_flag_parses_boolean_values(monkeypatch, value, expected):
    monkeypatch.setenv("SAM_TEST_FLAG", value)
    assert env_flag("SAM_TEST_FLAG", not expected) is expected


def test_env_flag_rejects_ambiguous_value(monkeypatch):
    monkeypatch.setenv("SAM_TEST_FLAG", "sometimes")
    with pytest.raises(ValueError, match="SAM_TEST_FLAG"):
        env_flag("SAM_TEST_FLAG", True)
