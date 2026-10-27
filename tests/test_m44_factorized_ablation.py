from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from netsurf.adapters.registry import create_adapter
from netsurf.api import regenerate_report
from netsurf.curvature import DenseReluDenseCurvature
from netsurf.factorized import FactorCalibration, FactorCalibrationCache, FactorizedBitTable, FactorizedInteractionAggregate
from netsurf.incremental import DenseReluDenseIncrementalEvaluator, FaultState
from netsurf.identities import BitIdentity
from netsurf.quantization import BitCodec, FixedPointFormat
from netsurf.search import enrich_sign_bit_probabilities, shuffle_factor_scores
from netsurf.search_config import SearchConfigurationError, load_search
from netsurf.search_engine import execute_search
from netsurf.search_bundle import create_search_bundle
from netsurf.result_bundle import read_result_bundle
from netsurf.study_config import load_study
from netsurf.topology import DenseReluDenseTopologyEvaluator


ROOT = Path(__file__).resolve().parents[1]


def _m44_text() -> str:
    text = (ROOT / "config/experiments/numpy_conditional_screening_exact.toml").read_text()
    text = text.replace('checkpoint = "../../netsurf/benchmarks/data/numpy_reference_checkpoint.json"',
                        f'checkpoint = "{ROOT / "netsurf/benchmarks/data/numpy_reference_checkpoint.json"}"')
    text = text.replace('dataset = "../../netsurf/benchmarks/data/numpy_reference_dataset.json"',
                        f'dataset = "{ROOT / "netsurf/benchmarks/data/numpy_reference_dataset.json"}"')
    text = text.replace('schema_version = "4.3.0"', 'schema_version = "4.4.0"')
    text = text.replace('scientific_protocol = "mask-aware-conditional-screening/v1"',
                        'scientific_protocol = "factorized-guidance-ablation/v1"')
    roles = {
        "static_gradient": ("gradient", "static"),
        "marginal_screening": ("conditional_marginal", "controlled"),
        "corrected_screening": ("conditional_marginal", "controlled"),
        "dynamic_kronecker": ("kronecker", "controlled"),
        "dynamic_stratified_kronecker": ("stratified_kronecker", "controlled"),
        "hybrid_factorized_screening": ("kronecker", "controlled"),
        "conditional_greedy": ("quality_reference", "reference"),
    }
    for name, (proposal, role) in roles.items():
        marker = f'name = "{name}"'
        text = text.replace(marker, marker + f'\nproposal_model = "{proposal}"\ncomparison_role = "{role}"')
    extra_methods = '''
[[methods]]
name = "gradient_screening"
strategy = "corrected_screening"
guidance = ["gradient_global"]
interaction_model = "none"
proposal_model = "gradient"
comparison_role = "controlled"

[[methods]]
name = "gradient_sign_enriched"
strategy = "corrected_screening"
guidance = ["gradient_global"]
interaction_model = "none"
proposal_model = "gradient_sign_enriched"
sign_bit_weight = 4.0
comparison_role = "controlled"

[[methods]]
name = "uniform_exploration"
strategy = "corrected_screening"
guidance = ["gradient_global"]
interaction_model = "none"
proposal_model = "uniform"
comparison_role = "controlled"

[[methods]]
name = "shuffled_kronecker"
strategy = "factorized_screening"
guidance = ["gradient_global"]
interaction_model = "factorized_kronecker"
proposal_model = "shuffled_kronecker"
aggregate_shuffle_mode = "layer_bit_position"
comparison_role = "controlled"
aggregate_population = "full"
dynamic_aggregate = true

'''
    text = text.replace('[[methods]]\nname = "conditional_greedy"', extra_methods + '[[methods]]\nname = "conditional_greedy"')
    insertion = '''
[proposal_control]
candidate_universe = "shared_candidate_pool"
verification_budget = 2
shared_backgrounds = true
enforce_equal_iterations = true

[factor_cache]
enabled = true
scope = "run"
vectorized_queries = true

'''
    return text.replace("[fault]\n", insertion + "[fault]\n")


def _curvature_fixture():
    config = load_study(ROOT / "config/experiments/numpy_qpolar_cancellation.toml")
    codec = BitCodec(FixedPointFormat.parse(config.benchmark.quantization))
    adapter = create_adapter(config, codec); adapter.load()
    inputs, _, _ = adapter.sample_partition("evaluation", 4)
    evaluator = DenseReluDenseIncrementalEvaluator(adapter, codec)
    topology = DenseReluDenseTopologyEvaluator(evaluator, inputs, FaultState.from_bits())
    return adapter, DenseReluDenseCurvature(topology)


def test_schema_44_requires_explicit_fairness_controls(tmp_path):
    path = tmp_path / "m44.toml"; path.write_text(_m44_text())
    configuration = load_search(path)
    assert configuration.proposal_control.verification_budget == 2
    assert configuration.factor_cache.vectorized_queries is True
    assert {method.comparison_role for method in configuration.methods} == {"controlled", "reference", "static"}
    broken = tmp_path / "broken.toml"
    broken.write_text(_m44_text().replace("shared_backgrounds = true", "shared_backgrounds = false"))
    with pytest.raises(SearchConfigurationError, match="shared_backgrounds"):
        load_search(broken)


def test_sign_enrichment_preserves_population_and_increases_sign_mass():
    bits = tuple(BitIdentity("dense0/kernel", word, position) for word in range(4) for position in range(6))
    probabilities = np.full(len(bits), 1 / len(bits))
    enriched = enrich_sign_bit_probabilities(probabilities, bits, sign_bit_index=5, weight=4.0)
    sign = np.asarray([bit.bit_index == 5 for bit in bits])
    assert len(enriched) == len(bits) and np.all(enriched > 0) and enriched.sum() == pytest.approx(1.0)
    assert enriched[sign].sum() > probabilities[sign].sum()


def test_structural_shuffle_is_deterministic_and_preserves_each_stratum_distribution():
    adapter, _ = _curvature_fixture(); bits = tuple(adapter.bit_identities())
    values = np.arange(len(bits), dtype=float)
    left = shuffle_factor_scores(values, bits, mode="layer_bit_position", seed=17)
    right = shuffle_factor_scores(values, bits, mode="layer_bit_position", seed=17)
    assert np.array_equal(left, right)
    for layer in {bit.parameter.split("/", 1)[0] for bit in bits}:
        for position in range(6):
            indexes = [index for index, bit in enumerate(bits) if bit.parameter.startswith(layer + "/") and bit.bit_index == position]
            assert sorted(left[indexes]) == sorted(values[indexes])
    assert np.any(left != values)


def test_vectorized_factor_scores_and_updates_match_historical_scalar_path():
    adapter, curvature = _curvature_fixture(); universe = tuple(adapter.bit_identities())
    protected = frozenset(universe[:8]); candidates = universe[8:40]
    scalar = FactorizedInteractionAggregate(curvature, universe, protected, probabilities=.03)
    table = FactorizedBitTable.build(curvature, universe, key="table")
    calibration = FactorCalibration.build(curvature, key="calibration", stratified=False, strata=1, minimum_samples=2)
    vectorized = FactorizedInteractionAggregate(curvature, universe, protected, probabilities=.03,
                                                bit_table=table, calibration=calibration)
    assert vectorized.query_many(candidates) == pytest.approx([scalar.query(bit) for bit in candidates], abs=1e-12)
    vectorized.swap(universe[0], universe[20]); scalar.swap(universe[0], universe[20])
    assert vectorized.query_many(candidates) == pytest.approx([scalar.query(bit) for bit in candidates], abs=1e-12)
    assert vectorized.reconstruction_error() < 1e-12


def test_factor_cache_exact_keys_reuse_and_mismatch_without_stale_reuse():
    cache = FactorCalibrationCache(); built = []
    def builder(value):
        return lambda: built.append(value) or value
    first, hit = cache.table("checkpoint-a/partition-a", builder("first"))
    again, second_hit = cache.table("checkpoint-a/partition-a", builder("wrong"))
    changed, changed_hit = cache.table("checkpoint-a/partition-b", builder("changed"))
    assert (first, hit, again, second_hit, changed, changed_hit) == ("first", False, "first", True, "changed", False)
    assert built == ["first", "changed"]


def test_controlled_methods_share_pool_backgrounds_and_verification_budget(tmp_path):
    path = tmp_path / "m44.toml"; path.write_text(_m44_text())
    outcome = execute_search(load_search(path))
    controlled = [value for value in outcome.search_results if value.proposal_provenance["comparison_role"] == "controlled"]
    assert len({value.proposal_provenance["candidate_pool_sha256"] for value in controlled}) == 1
    assert {value.proposals_evaluated for value in controlled} == {4}
    schedules = {tuple(step.background_schedule_sha256 for step in value.steps) for value in controlled}
    assert len(schedules) == 1
    factorized = [value for value in controlled if value.proposal_provenance["factorized"]]
    assert any(row["calibration_cache_hit"] for value in factorized for row in value.factorized_rows)


def test_schema_44_payload_repeats_and_saved_bundle_report_regenerates(tmp_path):
    path = tmp_path / "m44.toml"; path.write_text(_m44_text())
    configuration = load_search(path)
    first = execute_search(configuration); second = execute_search(configuration)
    assert first.scientific_payload_sha256 == second.scientific_payload_sha256
    configured = replace(configuration, output=replace(configuration.output, root=tmp_path, plots=False))
    bundle_path = create_search_bundle(replace(first, configuration=configured), repository_root=ROOT, run_id="m44-bounded")
    assert (bundle_path / "measurements/controlled_proposals.csv").is_file()
    bundle = read_result_bundle(bundle_path, verify=True)
    report = regenerate_report(bundle, tmp_path / "derived", plots=False)
    assert report.is_file() and "conditional" in report.read_text().lower()
