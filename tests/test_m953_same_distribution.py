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
M954_CONTROL = ROOT / "config/examples/m954_kws_restricted_k8_control.toml"
M954_CONTEXT = ROOT / "config/examples/m954_kws_s5_context_k8.toml"
M954_SMOKE = ROOT / "config/examples/m954_kws_context_k6_smoke.toml"
M955_K6 = ROOT / "config/examples/m955_kws_paired_swap_k6_validation.toml"
M955_K8 = ROOT / "config/examples/m955_kws_s5_context_k8.toml"


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


def test_schema_62_examples_validate_costs_and_fresh_final_indices():
    control = load_conditional(M954_CONTROL, check_inputs=False)
    context = load_conditional(M954_CONTEXT, check_inputs=False)
    smoke = load_conditional(M954_SMOKE, check_inputs=False)
    assert control.schema_version == context.schema_version == smoke.schema_version == "6.2.0"
    assert control.protection_context.mode == "none"
    assert context.protection_context.mode == "verified_global_mask"
    assert context.protection_context.expected_population_size == 176128
    assert context.protection_context.expected_global_cardinality == 22016
    assert control.samples.evaluation_offset == context.samples.evaluation_offset == 6
    assert control.operation_counts()["total_requested_evaluations"] == 1755
    assert context.operation_counts()["total_requested_evaluations"] == 1755
    assert smoke.operation_counts()["total_requested_evaluations"] == 365


def test_schema_62_rejects_reuse_of_exposed_final_examples(tmp_path):
    path = tmp_path / "reuse.toml"
    path.write_text(
        M954_CONTROL.read_text(encoding="utf-8").replace(
            "evaluation_offset = 6", "evaluation_offset = 0"
        ), encoding="utf-8",
    )
    with pytest.raises(ConditionalConfigurationError, match="previously exposed"):
        load_conditional(path, check_inputs=False)


def test_schema_63_separates_exhaustive_selection_and_paired_final_costs():
    k6 = load_conditional(M955_K6, check_inputs=False)
    k8 = load_conditional(M955_K8, check_inputs=False)
    assert k6.schema_version == k8.schema_version == "6.3.0"
    assert k6.samples.evaluation_offset == k8.samples.evaluation_offset == 12
    assert k6.restricted_oracle.selection_seeds == (955311, 955312)
    assert k8.operation_counts() == {
        "conditional_requested_evaluations": 210,
        "restricted_oracle_requested_evaluations": 2313,
        "paired_swap_authoritative_evaluations_upper_bound": 768,
        "total_authoritative_evaluations_upper_bound": 3291,
        "total_requested_evaluations": 3291,
        "model_selection_outside_fields": 8,
        "full_enumeration_reference_fields": 1,
        "analytical_mask_risk_integrations": 252,
        "analytical_pairwise_mask_differences": 3402,
        "paired_swap_whole_path_observations": 192,
    }


def test_schema_63_rejects_exposed_final_indices_and_seed_reuse(tmp_path):
    text = M955_K6.read_text(encoding="utf-8")
    exposed = tmp_path / "exposed.toml"
    exposed.write_text(text.replace("evaluation_offset = 12", "evaluation_offset = 11"), encoding="utf-8")
    with pytest.raises(ConditionalConfigurationError, match="previously exposed"):
        load_conditional(exposed, check_inputs=False)
    reused = tmp_path / "reused.toml"
    reused.write_text(text.replace("seed = 955501", "seed = 955401"), encoding="utf-8")
    with pytest.raises(ConditionalConfigurationError, match="must be independent"):
        load_conditional(reused, check_inputs=False)
