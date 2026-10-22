"""Execution engine for schema-3 empirical and composable-policy studies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .adapters.registry import create_adapter
from .analysis_m3 import mask_composition, paired_policy_comparisons, ranking_quality
from .empirical import SingleBitMeasurement, measure_single_bits, select_all, stratified_candidates
from .estimators_m3 import EstimatorOutput, compute_estimator
from .experiment_engine import TrialMeasurement, _arrays_digest, _evaluate_faulted
from .faults import FaultModel, FaultSchedule
from .identities import BitIdentity
from .manifests import digest_object, sha256_file
from .ordering import compose_ordering
from .protection import ProtectionMask, ProtectionPolicy
from .quantization import BitCodec, FixedPointFormat
from .ranking import Ranking
from .samples import SamplePartition, require_disjoint
from .statistics import summarize
from .study_config import StudyConfiguration


M3_MEASUREMENT_SCHEMA_VERSION = "3.0.0"


@dataclass(frozen=True)
class StudyOutcome:
    configuration: StudyConfiguration
    scientific_payload: Mapping[str, Any]
    partitions: Mapping[str, SamplePartition]
    estimator_outputs: tuple[EstimatorOutput, ...]
    empirical_measurements: tuple[SingleBitMeasurement, ...]
    ranking_metrics: tuple[Mapping[str, Any], ...]
    trials: tuple[TrialMeasurement, ...]
    protection_results: tuple[Mapping[str, Any], ...]
    paired_comparisons: Mapping[str, Any]
    composition: tuple[Mapping[str, Any], ...]
    rankings: Mapping[str, tuple[tuple[Any, ...], ...]]
    protection_masks: Mapping[str, tuple[tuple[Any, ...], ...]]
    fault_schedules: tuple[Mapping[str, Any], ...]
    candidate_selection: tuple[Mapping[str, Any], ...]

    @property
    def scientific_payload_sha256(self) -> str:
        return digest_object(self.scientific_payload)


def _finite(value: Any) -> Any:
    """Make computed data strict-JSON compatible without altering finite values."""

    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _finite(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_finite(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    return value


def _basic_estimator(name: str, identities: Sequence[BitIdentity], *, seed: int) -> EstimatorOutput:
    if name == "msb_first":
        return EstimatorOutput(
            name, "encoded-bit-index/v1",
            {identity: float(identity.bit_index) for identity in identities}, None, 0.0,
            {"expression": "encoded bit index; larger is more significant"},
        )
    if name == "random":
        generator = np.random.default_rng(int(seed))
        return EstimatorOutput(
            name, "seeded-uniform-score/v1",
            {identity: float(value) for identity, value in zip(sorted(identities), generator.random(len(identities)))},
            None, 0.0, {"seed": int(seed), "purpose": "negative-control ranking score"},
        )
    raise ValueError(name)


def _summaries(rows: Sequence[TrialMeasurement], masks: Mapping[str, ProtectionMask], confidence: float):
    result = []
    for name, mask in masks.items():
        selected = [row for row in rows if row.policy == name]
        accuracy = summarize([row.accuracy for row in selected], confidence=confidence, bounds=(0.0, 1.0)).to_dict()
        loss = summarize([row.categorical_crossentropy for row in selected], confidence=confidence).to_dict()
        result.append({
            "policy": name,
            "protected_bits": len(mask.protected),
            "coverage": mask.coverage,
            "mask_sha256": mask.digest(),
            "accuracy": accuracy,
            "categorical_crossentropy": loss,
            "mean_applied_events": float(np.mean([row.applied_events for row in selected])),
            "mean_suppressed_events": float(np.mean([row.suppressed_events for row in selected])),
            "minimum_accuracy": float(min(row.accuracy for row in selected)),
            "minimum_accuracy_trial": int(min(selected, key=lambda row: row.accuracy).trial),
        })
    return tuple(result)


def execute_study(configuration: StudyConfiguration, *, adapter=None) -> StudyOutcome:
    """Execute a validated M3 study without writing any files."""

    codec = BitCodec(FixedPointFormat.parse(configuration.benchmark.quantization))
    benchmark = adapter or create_adapter(configuration, codec)
    benchmark.load()
    ranking_x, ranking_y, ranking_partition = benchmark.sample_partition("ranking", configuration.samples.ranking_count)
    empirical_x, empirical_y, empirical_partition = benchmark.sample_partition("empirical_validation", configuration.samples.empirical_count)
    evaluation_x, evaluation_y, evaluation_partition = benchmark.sample_partition("evaluation", configuration.samples.evaluation_count)
    partitions = {"ranking": ranking_partition, "empirical_validation": empirical_partition, "evaluation": evaluation_partition}
    if configuration.samples.require_disjoint:
        require_disjoint(tuple(partitions.values()))

    original = benchmark.values()
    identities = tuple(benchmark.bit_identities())
    if len(set(identities)) != len(identities):
        raise RuntimeError("eligible bit identities are not unique")
    outputs: list[EstimatorOutput] = []
    for specification in configuration.estimators:
        output = (
            _basic_estimator(specification.name, identities, seed=specification.seed)
            if specification.name in {"msb_first", "random"}
            else compute_estimator(
                specification.name, benchmark, ranking_x, ranking_y, identities, codec,
                seed=specification.seed, probes=specification.probes,
            )
        )
        if output.implementation_version != specification.implementation_version:
            raise ValueError(
                f"estimator {specification.name} implementation version mismatch: "
                f"configuration requested {specification.implementation_version!r}, code provides {output.implementation_version!r}"
            )
        outputs.append(output)
    by_name = {output.name: output for output in outputs}

    empirical: tuple[SingleBitMeasurement, ...] = ()
    candidate_rows: tuple[Mapping[str, Any], ...] = ()
    metrics: tuple[Mapping[str, Any], ...] = ()
    if configuration.study_type in {"single_bit", "estimator_validation"}:
        if configuration.candidates.strategy == "all":
            candidates = select_all(identities)
        else:
            score_output = by_name.get(configuration.candidates.score_estimator or "")
            candidates = stratified_candidates(
                identities, configuration.candidates.count, seed=configuration.candidates.seed,
                scores=None if score_output is None else score_output.scores,
                quantile_count=configuration.candidates.score_quantiles,
            )
        candidate_rows = tuple({
            "parameter": value.identity.parameter, "flat_index": value.identity.flat_index,
            "bit_index": value.identity.bit_index, "inclusion_rule": value.inclusion_rule,
            "sampling_probability": value.sampling_probability, "score_quantile": value.score_quantile,
        } for value in candidates)
        empirical = measure_single_bits(
            benchmark, empirical_x, empirical_y, candidates, codec,
            evaluation_sample_sha256=empirical_partition.sample_sha256,
        )
        metrics = ranking_quality(empirical, outputs)

    trial_rows: list[TrialMeasurement] = []
    masks: dict[str, ProtectionMask] = {}
    schedules: list[FaultSchedule] = []
    compositions: tuple[Mapping[str, Any], ...] = ()
    protection_results: tuple[Mapping[str, Any], ...] = ()
    comparisons: Mapping[str, Any] = {}
    if configuration.study_type == "protection":
        universe = frozenset(identities)
        masks["no_protection"] = ProtectionMask("control_none", frozenset(), len(identities), 0.0, {"control": True})
        masks["full_protection"] = ProtectionMask("control_full", universe, len(identities), float(len(identities)), {"control": True})
        for policy in configuration.policies:
            output = by_name[policy.estimator]
            ranking = Ranking(output.name, output.scores, output.metadata)
            ordering = compose_ordering(policy.name, ranking, identities, benchmark.parameter_specs(), policy.priority, seed=policy.seed)
            for coverage in configuration.protection.coverages:
                name = f"{policy.name}@{coverage:.6f}"
                mask = ProtectionPolicy.ordered(ordering, coverage)
                masks[name] = ProtectionMask(name, mask.protected, mask.universe_size, mask.budget, {**mask.metadata, "prespecified": policy.prespecified})
        fault_model = FaultModel(
            target=configuration.fault.target, location=configuration.fault.location,
            persistence=configuration.fault.persistence, distribution=configuration.fault.distribution,
            probability=configuration.fault.probability,
        )
        state_before = _arrays_digest(benchmark.values())
        for trial in range(configuration.execution.trials):
            schedule = FaultSchedule.generate(
                fault_model, identities, seed=configuration.execution.fault_seed + trial,
                inference_count=configuration.samples.evaluation_count,
            )
            schedules.append(schedule)
            for name, mask in masks.items():
                accuracy, loss, accounting = _evaluate_faulted(benchmark, evaluation_x, evaluation_y, original, schedule, mask, codec)
                trial_rows.append(TrialMeasurement(
                    trial, name, schedule.digest(), accuracy, loss, accounting.schedule_hash,
                    accounting.proposed_events, accounting.applied_events,
                    accounting.suppressed_events, accounting.unique_applied_bits,
                ))
        if _arrays_digest(benchmark.values()) != state_before:
            raise RuntimeError("model state was not restored after protection trials")
        protection_results = _summaries(trial_rows, masks, configuration.execution.confidence)
        accuracy_vectors = {name: [row.accuracy for row in trial_rows if row.policy == name] for name in masks}
        loss_vectors = {name: [row.categorical_crossentropy for row in trial_rows if row.policy == name] for name in masks}
        comparisons = {
            "accuracy": paired_policy_comparisons(accuracy_vectors, confidence=configuration.execution.confidence, seed=configuration.execution.permutation_seed),
            "categorical_crossentropy": paired_policy_comparisons(loss_vectors, confidence=configuration.execution.confidence, seed=configuration.execution.permutation_seed),
            "multiplicity": "Holm adjustment within each reported metric family",
        }
        compositions = mask_composition({name: tuple(mask.protected) for name, mask in masks.items()}, identities)

    exact_rankings = {
        output.name: tuple((identity.parameter, identity.flat_index, identity.bit_index, float(output.scores[identity])) for identity in sorted(identities))
        for output in outputs
    }
    exact_masks = {name: tuple((identity.parameter, identity.flat_index, identity.bit_index) for identity in sorted(mask.protected)) for name, mask in masks.items()}
    exact_schedules = tuple({
        "trial": trial, "seed": schedule.seed, "sha256": schedule.digest(),
        "events": tuple((event.inference_index, event.bit.parameter, event.bit.flat_index, event.bit.bit_index) for event in schedule.events),
    } for trial, schedule in enumerate(schedules))
    clean = benchmark.evaluate(evaluation_x, evaluation_y)
    quantized = benchmark.quantized_values()
    empirical_payload = [{key: value for key, value in measurement.to_dict().items() if key != "runtime_seconds"} for measurement in empirical]
    metric_payload = [{key: value for key, value in row.items() if key != "runtime_seconds"} for row in metrics]
    scientific_payload = _finite({
        "schema": "netsurf-m3-study/v1",
        "scientific_protocol": configuration.scientific_protocol,
        "benchmark": {
            "id": configuration.benchmark.id,
            "checkpoint_sha256": sha256_file(configuration.benchmark.checkpoint),
            "dataset_sha256": sha256_file(configuration.benchmark.dataset),
            "eligible_parameters": sum(value.size for value in benchmark.parameter_specs()),
            "eligible_bits": len(identities),
            "partitions": {name: value.to_dict() for name, value in partitions.items()},
        },
        "quantization": codec.format.to_dict(),
        "estimator_definitions": {output.name: {"implementation_version": output.implementation_version, **dict(output.metadata)} for output in outputs},
        "candidate_selection": list(candidate_rows),
        "empirical_measurements": empirical_payload,
        "ranking_metrics": metric_payload,
        "clean_evaluation": asdict(clean),
        "fault_model": asdict(FaultModel(configuration.fault.target, configuration.fault.location, configuration.fault.persistence, configuration.fault.distribution, configuration.fault.probability)),
        "protection_results": list(protection_results),
        "paired_comparisons": comparisons,
        "mask_composition": list(compositions),
        "exact_artifact_sha256": {
            "rankings": digest_object(exact_rankings), "protection_masks": digest_object(exact_masks), "fault_schedules": digest_object(exact_schedules),
        },
        "checks": {
            "partitions_disjoint": configuration.samples.require_disjoint,
            "model_state_restored": _arrays_digest(benchmark.values()) == _arrays_digest(original),
            "matched_schedules": all(len({row.schedule_sha256 for row in trial_rows if row.trial == trial}) == 1 for trial in range(configuration.execution.trials)) if trial_rows else "not_applicable",
            "quantized_state_exact": all(np.array_equal(original[name], quantized[name]) for name in original),
        },
    })
    if _arrays_digest(benchmark.values()) != _arrays_digest(original):
        raise RuntimeError("model state changed during study")
    return StudyOutcome(
        configuration, scientific_payload, partitions, tuple(outputs), empirical, tuple(_finite(metrics)),
        tuple(trial_rows), tuple(_finite(protection_results)), _finite(comparisons), tuple(_finite(compositions)),
        exact_rankings, exact_masks, exact_schedules, candidate_rows,
    )
