from __future__ import annotations

from pathlib import Path
from dataclasses import replace

import numpy as np
import pytest

from netsurf.adapters.registry import create_adapter
from netsurf.curvature import DenseReluDenseCurvature
from netsurf.incremental import DenseReluDenseIncrementalEvaluator, FaultState
from netsurf.quantization import BitCodec, FixedPointFormat
from netsurf.search import proposal_influence, sparse_conditional_swap_score
from netsurf.search_config import SearchConfigurationError, load_search
from netsurf.search_engine import execute_search
from netsurf.search_reporting import paired_search_comparisons, summarize_search_trials
from netsurf.study_config import load_study
from netsurf.topology import DenseReluDenseTopologyEvaluator


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def curvature_fixture():
    config = load_study(ROOT / "config/experiments/numpy_qpolar_cancellation.toml")
    codec = BitCodec(FixedPointFormat.parse(config.benchmark.quantization))
    adapter = create_adapter(config, codec); adapter.load()
    inputs, labels, _ = adapter.sample_partition("evaluation", 4)
    evaluator = DenseReluDenseIncrementalEvaluator(adapter, codec)
    topology = DenseReluDenseTopologyEvaluator(evaluator, inputs, FaultState.from_bits())
    return adapter, topology, DenseReluDenseCurvature(topology), labels


def test_output_local_curvature_matches_direct_softmax_quadratic(curvature_fixture):
    adapter, topology, curvature, _ = curvature_fixture
    bits = [bit for bit in adapter.bit_identities() if bit.parameter == "dense1/kernel"]
    left, right = bits[0], bits[-1]
    a, b = curvature.coordinate(left), curvature.coordinate(right)
    dl = topology.response(left).exact_delta
    dr = topology.response(right).exact_delta
    h = curvature.inputs["dense1"]
    expected = dl * dr * h[:, a.input_index] * h[:, b.input_index] * curvature.output_q[:, a.output_index, b.output_index]
    np.testing.assert_allclose(curvature.local_per_sample(left, right), expected, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("parameter", ["dense0/kernel", "dense0/bias", "dense1/kernel", "dense1/bias"])
def test_kronecker_covariance_error_identity(curvature_fixture, parameter):
    adapter, _, curvature, _ = curvature_fixture
    bits = [bit for bit in adapter.bit_identities() if bit.parameter == parameter]
    left, right = bits[0], bits[-1]
    actual = curvature.local(left, right) - curvature.kron(left, right)
    assert actual == pytest.approx(curvature.covariance_error(left, right), abs=1e-14)


def test_stratified_factors_cover_all_samples_and_are_deterministic(curvature_fixture):
    _, _, curvature, _ = curvature_fixture
    first = curvature.stratified_factors("dense0", 2, minimum_samples=2)
    replay = curvature.stratified_factors("dense0", 2, minimum_samples=2)
    assert sum(value.sample_count for value in first.factors) == len(curvature.cache.logits)
    assert sum(first.probabilities) == pytest.approx(1.0)
    assert first.boundaries == replay.boundaries
    for left, right in zip(first.factors, replay.factors):
        np.testing.assert_array_equal(left.activation_factor, right.activation_factor)
        np.testing.assert_array_equal(left.curvature_factor, right.curvature_factor)


def test_cross_layer_local_curvature_is_rejected(curvature_fixture):
    adapter, _, curvature, _ = curvature_fixture
    hidden = next(bit for bit in adapter.bit_identities() if bit.parameter == "dense0/kernel")
    output = next(bit for bit in adapter.bit_identities() if bit.parameter == "dense1/kernel")
    with pytest.raises(ValueError, match="cross-layer"):
        curvature.local(hidden, output)


def test_sparse_mask_dependent_score_includes_remaining_faultable_neighbors():
    from netsurf.identities import BitIdentity
    a, b, q, protected_other = [BitIdentity("p", index, 0) for index in range(4)]
    universe = (a, b, q, protected_other)
    protected = {a, protected_other}
    interactions = {(a, q): 4.0, (b, q): -2.0, (a, b): 999.0, (a, protected_other): 888.0}
    score = sparse_conditional_swap_score(a, b, protected, universe, interactions, probabilities=0.1)
    # Direct a,b and already-protected neighbors are excluded.  Only q remains.
    assert score == pytest.approx(0.1 * (0.1 * 4.0 - 0.1 * -2.0))


def test_polarity_ablation_changes_constructed_proposal_distribution():
    before = np.asarray([0.55, 0.25, 0.15, 0.05])
    after = np.asarray([0.10, 0.20, 0.30, 0.40])
    result = proposal_influence(before, after, top_k=2)
    assert result.changed_fraction > 0
    assert result.top_k_overlap < 1
    assert result.total_variation > 0
    assert result.jensen_shannon_divergence > 0
    zero = proposal_influence(before, before, top_k=2)
    assert zero.changed_fraction == 0
    assert zero.jensen_shannon_divergence == pytest.approx(0.0)


def test_finite_polar_and_local_quadratic_are_distinct_finite_perturbation_levels(curvature_fixture):
    adapter, topology, curvature, labels = curvature_fixture
    output = [bit for bit in adapter.bit_identities() if bit.parameter == "dense1/kernel" and bit.bit_index == 5]
    pair = topology.pair(output[0], output[-1], labels)
    local = curvature.local(output[0], output[-1])
    assert np.isfinite(local)
    assert np.all(np.isfinite(pair.finite_polar_per_sample))
    # This test documents the levels rather than assuming a sign-bit Taylor
    # approximation is numerically identical to the finite interaction.
    assert pair.structurally_exact


@pytest.mark.parametrize("name", [
    "numpy_topology_polar_exact.toml",
    "mnist_topology_exact_validation.toml",
    "mnist_topology_polar_smoke.toml",
])
def test_schema_42_maintained_configs_validate(name):
    configuration = load_search(ROOT / "config/experiments" / name, check_inputs=False)
    assert configuration.schema_version == "4.2.0"
    assert configuration.topology.interaction_guidance == "mask_dependent_sparse_v1"
    assert all(isinstance(value, int) for value in configuration.initial_mask.budgets_bits)


def test_schema_42_rejects_unregistered_interaction_model(tmp_path):
    source = (ROOT / "config/experiments/numpy_topology_polar_exact.toml").read_text()
    source = source.replace('interaction_model = "kronecker"', 'interaction_model = "invented"', 1)
    path = tmp_path / "invalid.toml"; path.write_text(source)
    with pytest.raises(SearchConfigurationError, match="unsupported interaction model"):
        load_search(path, check_inputs=False)


def test_schema_42_multiple_budgets_remain_distinct_execution_conditions():
    configuration = load_search(ROOT / "config/experiments/numpy_topology_polar_exact.toml")
    configuration = replace(
        configuration,
        initial_mask=replace(configuration.initial_mask, budget_bits=1, budgets_bits=(1, 2)),
        methods=(configuration.methods[0],),
        polar=replace(configuration.polar, enabled=False, pair_budget=0),
        search=replace(configuration.search, iterations=0, exact_enumeration=False),
        final_evaluation=replace(configuration.final_evaluation, contexts=2, conditional_contexts=2),
    )
    outcome = execute_search(configuration)
    assert {member.protection_budget_bits for member in outcome.replications} == {1, 2}
    assert len({member.replication_name for member in outcome.replications}) == 2
    assert all({int(row["protection_budget_bits"]) for row in member.final_trials} == {member.protection_budget_bits} for member in outcome.replications)


def test_multi_budget_reporting_never_pools_or_cross_pairs_conditions():
    rows = []
    for budget, base in ((1, 0.2), (2, 0.8)):
        for strategy, offset in (("static_gradient", 0.0), ("finite_polar", 0.1)):
            rows.append({"replication": f"budget_{budget}_replication_0", "logical_ber": 0.05,
                         "protection_budget_bits": budget, "strategy": strategy, "trial": 0,
                         "schedule_sha256": f"schedule-{budget}", "accuracy": base + offset,
                         "categorical_crossentropy": 1.0 - base - offset})
    summaries = summarize_search_trials(rows)
    assert {(value["protection_budget_bits"], value["strategy"]) for value in summaries} == {
        (1, "static_gradient"), (1, "finite_polar"), (2, "static_gradient"), (2, "finite_polar")}
    paired = paired_search_comparisons(rows)
    assert len(paired) == 4  # two metrics at each of two budgets
    assert {value["protection_budget_bits"] for value in paired} == {1, 2}
    assert all(value["matched_pairs"] == 1 for value in paired)
