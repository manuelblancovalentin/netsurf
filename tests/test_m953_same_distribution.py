from __future__ import annotations

from pathlib import Path

import pytest

from netsurf.adapters.registry import create_adapter
from netsurf.conditional_config import load_conditional
from netsurf.conditional_config import ConditionalConfigurationError
from netsurf.samples import require_disjoint


ROOT = Path(__file__).resolve().parents[1]
K6 = ROOT / "config/examples/m953_kws_restricted_k6_smoke.toml"
K8 = ROOT / "config/examples/m953_kws_restricted_k8_template.toml"


def test_schema_61_examples_validate_and_expose_cost_before_execution():
    six = load_conditional(K6, check_inputs=False)
    eight = load_conditional(K8, check_inputs=False)
    assert six.schema_version == eight.schema_version == "6.1.0"
    assert six.samples.calibration_role == "conditional_calibration"
    assert six.samples.model_selection_role == "conditional_model_selection"
    assert six.samples.evaluation_role == "conditional_final_evaluation"
    assert six.operation_counts() == {
        "conditional_requested_evaluations": 38,
        "restricted_oracle_requested_evaluations": 327,
        "total_requested_evaluations": 365,
    }
    assert eight.operation_counts()["total_requested_evaluations"] == 1755


@pytest.mark.skipif(
    not Path("/projects/kona-assets/manuelbv/locks/mlperf-tiny/kws.lock.json").is_file(),
    reason="pinned KWS assets are absent",
)
def test_pinned_kws_same_distribution_windows_are_deterministic_and_disjoint():
    tensorflow = pytest.importorskip("tensorflow")
    if tensorflow.__version__ != "2.14.1":
        pytest.skip("KWS reference contract requires TensorFlow 2.14.1")
    configuration = load_conditional(K6)
    adapter = create_adapter(configuration, None)
    adapter.load()
    observed = []
    for role, expected_start in (
        ("conditional_calibration", 1000),
        ("conditional_model_selection", 2024),
        ("conditional_final_evaluation", 3048),
    ):
        first_x, first_y, first = adapter.sample_partition(role, 2)
        second_x, second_y, second = adapter.sample_partition(role, 2)
        assert first.indices == (expected_start, expected_start + 1)
        assert first.sample_sha256 == second.sample_sha256
        assert (first_x == second_x).all()
        assert (first_y == second_y).all()
        observed.append(first)
    require_disjoint(observed)


def test_same_distribution_window_bounds_reject_overlap_by_construction():
    configuration = load_conditional(K6, check_inputs=False)
    assert configuration.samples.calibration_count <= 1024
    assert configuration.samples.model_selection_count <= 1024
    assert configuration.samples.evaluation_count <= 1842


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("calibration_offset = 0", "calibration_offset = 1024"),
        ("model_selection_offset = 0", "model_selection_offset = 1024"),
        ("evaluation_offset = 0", "evaluation_offset = 1842"),
    ),
)
def test_schema_61_rejects_sample_windows_past_fixed_boundaries(tmp_path, field, replacement):
    path = tmp_path / "outside-window.toml"
    path.write_text(K6.read_text(encoding="utf-8").replace(field, replacement), encoding="utf-8")
    with pytest.raises(ConditionalConfigurationError, match="window capacity"):
        load_conditional(path, check_inputs=False)


def test_schema_61_rejects_non_target_population_role(tmp_path):
    text = K6.read_text(encoding="utf-8").replace(
        'calibration_role = "conditional_calibration"',
        'calibration_role = "calibration"',
    )
    path = tmp_path / "bad.toml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConditionalConfigurationError, match="same-distribution roles"):
        load_conditional(path, check_inputs=False)
