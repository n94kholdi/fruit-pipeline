from types import SimpleNamespace

import numpy as np
import pytest

from fruit_pipeline.rfdetr_diagnostics import (
    RuntimeConfig,
    assert_nms_unreachable,
    assert_predictions_not_capped,
    box_diagnostics,
    build_sweep_configs,
    read_runtime_config,
    rectangle_union_area,
    score_statistics,
)


class _PlainModule:
    num_queries = 300

    def named_modules(self):
        return [("", self), ("decoder", object())]


class _Postprocess:
    num_select = 275

    def forward(self, outputs):
        return outputs


class _Model:
    def __init__(self, module=None, postprocess=None):
        self.model_config = SimpleNamespace(patch_size=14, num_windows=4)
        self.model = SimpleNamespace(
            model=module or _PlainModule(),
            inference_model=None,
            postprocess=postprocess or _Postprocess(),
            resolution=560,
        )
        self._is_optimized_for_inference = False

    def predict(self, image):
        return image


def test_reads_values_from_active_runtime_objects():
    assert read_runtime_config(_Model()) == RuntimeConfig(
        num_queries=300,
        num_select=275,
        native_resolution=560,
        patch_size=14,
        num_windows=4,
    )


def test_nms_assert_checks_runtime_module_graph():
    class NMS:
        pass

    class Module(_PlainModule):
        def named_modules(self):
            return [("", self), ("post.nms", NMS())]

    assert_nms_unreachable(_Model())
    with pytest.raises(AssertionError, match="NMS is reachable"):
        assert_nms_unreachable(_Model(module=Module()))


def test_cap_assertion_is_loud_only_at_equality():
    assert_predictions_not_capped(299, 300)
    with pytest.raises(AssertionError, match="CAP BINDING"):
        assert_predictions_not_capped(300, 300)


def test_rectangle_union_area_does_not_double_count_overlap():
    assert rectangle_union_area([[0, 0, 4, 4], [2, 0, 6, 4]]) == 24.0


def test_box_diagnostics_containment_area_ratios_and_coverage_gap():
    diagnostics = box_diagnostics(
        [
            [0, 0, 10, 10],
            [1, 1, 3, 3],
            [6, 6, 8, 8],
            [20, 20, 22, 22],
        ]
    )
    assert diagnostics["containment_counts"] == [2, 0, 0, 0]
    assert diagnostics["max_containment_count"] == 2
    assert diagnostics["collapse_candidate_count"] == 1
    assert diagnostics["area_ratio_p050"] == 1.0
    assert diagnostics["coverage_gap"] == pytest.approx(0.92)


def test_empty_box_diagnostics_are_defined():
    diagnostics = box_diagnostics([])
    assert diagnostics["max_containment_count"] == 0
    assert diagnostics["collapse_candidate_count"] == 0
    assert np.isnan(diagnostics["coverage_gap"])


def test_score_statistics_contains_min_max_and_all_deciles():
    stats = score_statistics([0.0, 1.0])
    assert stats["score_min"] == 0.0
    assert stats["score_p050"] == 0.5
    assert stats["score_max"] == 1.0
    assert len([key for key in stats if key.startswith("score_p")]) == 11


def test_sweep_clamps_num_select_and_uses_runtime_divisor():
    runtime = RuntimeConfig(400, 250, 560, 14, 4)
    configs = build_sweep_configs(runtime, baseline_threshold=0.25, max_resolution=672)
    num_select_configs = [config for config in configs if config.axis == "num_select"]
    assert [(config.requested_value, config.num_select) for config in num_select_configs] == [
        (300, 300),
        (600, 400),
        (900, 400),
        ("num_queries", 400),
    ]
    assert [config.resolution for config in configs if config.axis == "resolution"] == [560, 616, 672]


def test_resolution_step_must_match_actual_model_divisor():
    runtime = RuntimeConfig(300, 300, 560, 14, 4)
    with pytest.raises(ValueError, match="multiple of 56"):
        build_sweep_configs(runtime, 0.25, 672, resolution_step=64)
