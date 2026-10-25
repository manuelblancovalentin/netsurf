from __future__ import annotations

from itertools import combinations
from pathlib import Path
from dataclasses import replace

import numpy as np
import pytest

from netsurf.adapters.registry import create_adapter
from netsurf.conditional_risk import compare_protection_masks, deterministic_swap_path
from netsurf.incremental import DenseReluDenseIncrementalEvaluator, FaultState, metrics_from_logits
from netsurf.quantization import BitCodec, FixedPointFormat
from netsurf.search import sample_full_fault_states
from netsurf.search_bundle import create_search_bundle
from netsurf.search_config import SearchConfigurationError, SearchReplications, load_search
from netsurf.search_engine import execute_search
from netsurf.experiment_config import OutputConfiguration
from netsurf.result_bundle import read_result_bundle
from netsurf.progress import ProgressTracker, search_operation_plan
from netsurf.api import regenerate_report
from netsurf.statistics import summarize
from netsurf.study_config import load_study


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "config/experiments/numpy_qpolar_cancellation.toml"
CONTROLLED = ROOT / "config/experiments/numpy_pgcs_conditional_validation.toml"


@pytest.fixture
def reference():
    configuration = load_study(REFERENCE)
    codec = BitCodec(FixedPointFormat.parse(configuration.benchmark.quantization))
    adapter = create_adapter(configuration, codec); adapter.load()
    inputs, labels, _ = adapter.sample_partition("evaluation", 4)
    return adapter, inputs, labels, DenseReluDenseIncrementalEvaluator(adapter, codec)


def _all_states(bits):
    return tuple(FaultState.from_bits(selected) for count in range(len(bits)+1) for selected in combinations(bits, count))


def test_deterministic_path_is_legal_complete_and_stable(reference):
    adapter, *_ = reference
    bits = tuple(adapter.bit_identities())
    left = frozenset((bits[7], bits[2], bits[15]))
    right = frozenset((bits[3], bits[11], bits[20]))
    path = deterministic_swap_path(left, right)
    assert path == deterministic_swap_path(left, right)
    assert {step.removed for step in path} == left-right
    assert {step.added for step in path} == right-left
    assert len(path) == len(left-right)
    current = set(left)
    for step in path:
        assert step.removed in current and step.added not in current
        current.remove(step.removed); current.add(step.added)
    assert frozenset(current) == right
    with pytest.raises(ValueError, match="equal-cardinality"):
        deterministic_swap_path(left, frozenset(bits[:2]))


def test_telescoping_conditional_identity_matches_exact_enumeration(reference):
    adapter, inputs, labels, evaluator = reference
    universe = tuple(adapter.bit_identities())[:6]
    baseline, target = frozenset(universe[:2]), frozenset(universe[3:5])
    states = _all_states(universe)
    result = compare_protection_masks(
        evaluator, inputs, labels, universe, states, baseline, target,
        probabilities=0.5, confidence=0.95,
    )

    def exact_risk(mask):
        return np.mean([
            metrics_from_logits(evaluator.full_forward(inputs, FaultState(state.bits-mask)).logits, labels).categorical_crossentropy
            for state in states
        ])

    expected = exact_risk(target)-exact_risk(baseline)
    assert result.conditional_loss_difference["mean"] == pytest.approx(expected, abs=1e-14)
    assert result.direct_loss_difference["mean"] == pytest.approx(expected, abs=1e-14)
    assert len(result.swaps) == 2
    assert result.context_count == 64


def test_identical_masks_have_exact_zero_effect(reference):
    adapter, inputs, labels, evaluator = reference
    universe = tuple(adapter.bit_identities())[:8]
    mask = frozenset(universe[:2])
    states = sample_full_fault_states(universe, probability=0.2, count=9, seed=17)
    result = compare_protection_masks(evaluator, inputs, labels, universe, states, mask, mask, probabilities=0.2)
    assert result.swaps == ()
    assert result.conditional_loss_difference["mean"] == 0
    assert result.direct_loss_difference["mean"] == 0


def test_unequal_bit_probabilities_match_weighted_exact_risk(reference):
    adapter, inputs, labels, evaluator = reference
    universe = tuple(adapter.bit_identities())[:4]
    probabilities = {bit: value for bit, value in zip(universe, (0.1, 0.2, 0.3, 0.4))}
    baseline, target = frozenset(universe[:2]), frozenset(universe[2:])
    weighted_conditional = weighted_direct = total = 0.0
    exact = {"left": 0.0, "right": 0.0}
    for state in _all_states(universe):
        weight = np.prod([probabilities[bit] if bit in state.bits else 1-probabilities[bit] for bit in universe])
        result = compare_protection_masks(evaluator, inputs, labels, universe, (state,), baseline, target, probabilities=probabilities)
        weighted_conditional += weight*result.conditional_loss_difference["mean"]
        weighted_direct += weight*result.direct_loss_difference["mean"]
        for name, mask in (("left",baseline),("right",target)):
            exact[name] += weight*metrics_from_logits(evaluator.full_forward(inputs,FaultState(state.bits-mask)).logits,labels).categorical_crossentropy
        total += weight
    assert total == pytest.approx(1)
    expected = exact["right"]-exact["left"]
    assert weighted_conditional == pytest.approx(expected, abs=1e-14)
    assert weighted_direct == pytest.approx(expected, abs=1e-14)


def test_context_is_cluster_unit_and_conditional_fixture_reduces_variance(reference):
    adapter, inputs, labels, evaluator = reference
    universe = tuple(adapter.bit_identities())
    baseline = frozenset((universe[5], universe[11]))
    target = frozenset((universe[23], universe[41]))
    states = sample_full_fault_states(universe, probability=0.02, count=400, seed=991)
    result = compare_protection_masks(evaluator, inputs, labels, universe, states, baseline, target, probabilities=0.02)
    assert result.conditional_loss_difference["count"] == 400
    assert result.conditional_loss_difference["standard_deviation"] < result.direct_loss_difference["standard_deviation"]
    known = summarize((-2.0, -1.0, 1.0, 2.0), confidence=0.95)
    assert known.mean == 0
    assert known.standard_error == pytest.approx(np.std((-2,-1,1,2),ddof=1)/2)


def test_controlled_configuration_rejects_nonindependent_final_seeds(tmp_path):
    text=CONTROLLED.read_text().replace("conditional_seed = 45104","conditional_seed = 45103")
    path=tmp_path/"bad.toml"; path.write_text(text)
    with pytest.raises(SearchConfigurationError,match="must be independent"):
        load_search(path,check_inputs=False)


def test_controlled_methods_share_design_and_final_schedules_are_independent():
    outcome=execute_search(load_search(CONTROLLED))
    assert {result.initial_mask for result in outcome.search_results} == {outcome.search_results[0].initial_mask}
    assert all(len(result.best_mask)==2 for result in outcome.search_results)
    schedules=[result.steps[0].background_schedule_sha256 for result in outcome.search_results if result.steps]
    assert len(set(schedules))==1
    natural=outcome.scientific_payload["final_fault_states"]
    conditional=outcome.scientific_payload["conditional_final_fault_states"]
    assert natural != conditional
    assert len(outcome.conditional_comparisons)==len(outcome.search_results)
    assert all(value.context_count==32 for value in outcome.conditional_comparisons)
    pool_keys={bit.key() for bit in outcome.candidate_pool}
    assert any(any(bit not in pool_keys for bit in state) for state in conditional)


def test_controlled_progress_and_multiber_plan_are_complete():
    config=load_search(CONTROLLED); events=[]
    tracker=ProgressTracker(config.name,search_operation_plan(config),events.append)
    execute_search(config,_tracker=tracker); tracker.finish()
    assert events[-1].completed==events[-1].total
    assert any(event.stage=="Held-out conditional evaluation" for event in events)
    research=load_search(ROOT/"config/experiments/mnist_pgcs_matched_research.toml")
    assert research.fault_probabilities==(0.001,0.01)
    assert len(research.methods)==8


def test_controlled_multiber_execution_keeps_conditions_separate():
    config=replace(load_search(CONTROLLED),fault_probabilities=(0.05,0.10))
    outcome=execute_search(config)
    assert len(outcome.replications)==2
    assert {member.logical_ber for member in outcome.replications}=={0.05,0.10}
    assert len({member.replication_name for member in outcome.replications})==2
    for member in outcome.replications:
        assert {float(row["logical_ber"]) for row in member.final_trials}=={member.logical_ber}


def test_replicated_bundle_aggregates_every_conditional_context(tmp_path):
    config=load_search(CONTROLLED)
    config=replace(config,replications=SearchReplications(2,1000,1000,0),output=OutputConfiguration(tmp_path/"runs",True,True))
    outcome=execute_search(config)
    path=create_search_bundle(outcome,repository_root=ROOT,run_id="controlled")
    bundle=read_result_bundle(path)
    assert bundle.scientific_payload_sha256==outcome.scientific_payload_sha256
    for row in bundle.summary["conditional_results"]:
        assert row["background_context_count"]==64
        assert row["replication_count"]==2
        assert row["sampling_unit"].startswith("one whole-path")
    assert (path/"measurements/conditional_context_contributions.csv").is_file()
    assert (path/"measurements/proposal_diagnostics.csv").is_file()
    assert (path/"plots/conditional_vs_direct_standard_error.png").is_file()
    assert "Independent conditional-risk comparison" in (path/"report.md").read_text()
    regenerated=regenerate_report(bundle,tmp_path/"derived",plots=True)
    assert "Independent conditional-risk comparison" in regenerated.read_text()
    assert (tmp_path/"derived/plots/conditional_vs_direct_standard_error.png").is_file()
