import pytest

from fruit_pipeline.utils.env import env_flag


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
