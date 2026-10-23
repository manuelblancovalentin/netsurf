from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from netsurf.adapters.registry import create_adapter
from netsurf.api import regenerate_report
from netsurf.empirical import uniform_candidates
from netsurf.estimators_m3 import per_sample_first_order_family
from netsurf.identities import BitIdentity
from netsurf.quantization import BitCodec, FixedPointFormat
from netsurf.result_bundle import read_result_bundle
from netsurf.study_bundle import create_study_bundle
from netsurf.study_config import StudyExecution, StudyPolicy, StudyProtection, StudyReplications, load_study
from netsurf.study_engine import execute_study


ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "config/experiments/numpy_qpolar_cancellation.toml"


class _PerExampleAdapter:
    def __init__(self):
        self.calls = 0

    def values(self):
        return {"weight": np.asarray([0.0])}

    def per_example_gradients(self, inputs, labels):
        self.calls += 1
        return {"weight": np.asarray([[-2.0], [-1.0], [1.0], [6.0]])}


def test_positive_absolute_and_signed_mean_scores_are_distinct_and_shared():
    adapter = _PerExampleAdapter()
    codec = BitCodec(FixedPointFormat.parse("q<6,2,1>"))
    identity = BitIdentity("weight", 0, 0)
    outputs = per_sample_first_order_family(
        adapter, np.zeros((4, 1)), np.zeros(4, dtype=int), (identity,), codec,
        requested=("per_sample_first_order_magnitude", "per_sample_positive_damage"),
    )
    delta = float(codec.perturbation_matrix(np.asarray([0.0]))[0, 0])
    absolute = outputs["per_sample_first_order_magnitude"].scores[identity]
    positive = outputs["per_sample_positive_damage"].scores[identity]
    signed_mean_magnitude = abs(np.mean([-2.0, -1.0, 1.0, 6.0]) * delta)
    assert adapter.calls == 1
    assert absolute == 2.5 * delta
    assert positive == 1.75 * delta
    assert signed_mean_magnitude == 1.0 * delta
    assert absolute >= positive >= signed_mean_magnitude


def test_uniform_candidate_sampling_is_reproducible_and_population_uniform():
    identities = tuple(BitIdentity("w", index, bit) for index in range(20) for bit in range(6))
    first = uniform_candidates(identities, 24, seed=17)
    replay = uniform_candidates(identities, 24, seed=17)
    other = uniform_candidates(identities, 24, seed=18)
    assert first == replay
    assert tuple(value.identity for value in first) != tuple(value.identity for value in other)
    assert len({value.identity for value in first}) == 24
    assert {value.inclusion_rule for value in first} == {"uniform_without_replacement"}
    assert {value.sampling_probability for value in first} == {0.2}


def test_controlled_signed_activation_cancellation_and_noncancellation():
    configuration = load_study(CONTROL)
    signed = execute_study(configuration)
    scores = {value.name: value.scores for value in signed.estimator_outputs}
    assert scores["clean_activation_qpolargrad"] != scores["uncancelled_activation_drift"]
    comparison = next(
        row for row in signed.estimator_agreement
        if row["left"] == "clean_activation_qpolargrad" and row["right"] == "uncancelled_activation_drift"
    )
    assert not comparison["identical_order"]
    assert comparison["changed_average_rank_count"] > 0

    nonnegative_path = ROOT / "netsurf/benchmarks/data/numpy_activation_nonnegative_dataset.json"
    nonnegative = execute_study(replace(
        configuration,
        benchmark=replace(configuration.benchmark, dataset=nonnegative_path),
    ))
    scores = {value.name: value.scores for value in nonnegative.estimator_outputs}
    assert scores["clean_activation_qpolargrad"] == scores["uncancelled_activation_drift"]


def test_replications_separate_candidate_and_hutchinson_randomization(tmp_path):
    configuration = load_study(CONTROL)
    hutchinson = replace(
        configuration.estimators[0],
        name="hutchinson_diagonal_second_order",
        implementation_version="rademacher-hvp-diagonal/v1",
        result_name="hutchinson_p2",
        seed=41,
        probes=2,
    )
    configuration = replace(
        configuration,
        name="numpy_m31_replicated_test",
        candidates=replace(configuration.candidates, strategy="uniform", count=18, seed=91),
        estimators=(hutchinson,),
        replications=StudyReplications(2, 1, 7, 0, 11),
        output=replace(configuration.output, root=tmp_path / "runs", plots=False),
    )
    first = execute_study(configuration)
    replay = execute_study(configuration)
    assert first.scientific_payload_sha256 == replay.scientific_payload_sha256
    assert len(first.replications) == 2
    assert first.replications[0].candidate_selection != first.replications[1].candidate_selection
    assert first.replications[0].estimator_outputs[0].metadata["seed"] == 41
    assert first.replications[1].estimator_outputs[0].metadata["seed"] == 48
    assert all(row["statistical_unit"] == "independent configured replication" for row in first.replication_aggregate)

    destination = create_study_bundle(first, repository_root=ROOT, run_id="replicated")
    bundle = read_result_bundle(destination)
    assert bundle.summary["replication_count"] == 2
    assert (destination / "measurements/replication_aggregate.csv").is_file()
    report = regenerate_report(bundle, tmp_path / "regenerated", plots=True)
    assert "Independent replications" in report.read_text(encoding="utf-8")
    assert (report.parent / "plots/predicted_vs_measured_damage.png").is_file()


def test_hutchinson_probe_seed_is_reproducible():
    configuration = load_study(CONTROL)
    codec = BitCodec(FixedPointFormat.parse(configuration.benchmark.quantization))
    adapter = create_adapter(configuration, codec)
    adapter.load()
    inputs, labels, _ = adapter.sample_partition("ranking", 4)
    identities = adapter.bit_identities()[:12]
    from netsurf.estimators_m3 import hutchinson_second_order

    first = hutchinson_second_order(adapter, inputs, labels, identities, codec, probes=3, seed=101)
    replay = hutchinson_second_order(adapter, inputs, labels, identities, codec, probes=3, seed=101)
    other = hutchinson_second_order(adapter, inputs, labels, identities, codec, probes=3, seed=102)
    assert first.scores == replay.scores
    assert first.scores != other.scores


def test_multi_ber_conditions_use_matched_schedules_within_each_condition():
    configuration = load_study(CONTROL)
    estimator = replace(
        configuration.estimators[0], name="msb_first",
        implementation_version="encoded-bit-index/v1", result_name="msb_first",
    )
    configuration = replace(
        configuration,
        scientific_protocol="multi-coverage-protection/v1",
        study_type="protection",
        estimators=(estimator,),
        policies=(StudyPolicy("msb+ordered", "msb_first", ("bit_significance_desc", "canonical"), None, True),),
        protection=StudyProtection((0.1,), "perfect_event_suppression", (0.01, 0.02)),
        execution=StudyExecution(2, 0.95, 501, 601),
    )
    outcome = execute_study(configuration)
    assert outcome.scientific_payload["checks"]["matched_schedules"] is True
    names = {row.policy for row in outcome.trials}
    assert any(name.startswith("ber=0.01/") for name in names)
    assert any(name.startswith("ber=0.02/") for name in names)
    assert len(outcome.fault_schedules) == 4
