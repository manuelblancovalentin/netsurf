"""Benchmark-neutral execution engine for versioned NetSuRF experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Any, Mapping

import numpy as np

from .adapters.protocol import BenchmarkAdapter
from .adapters.registry import create_adapter
from .experiment_config import ExperimentConfiguration
from .faults import FaultModel, FaultSchedule, apply_per_inference_schedule, apply_persistent_schedule
from .identities import BitIdentity
from .manifests import digest_object, sha256_file
from .protection import ProtectionMask, ProtectionPolicy
from .quantization import BitCodec, FixedPointFormat
from .ranking import (
    FirstOrderEstimator,
    GradientMagnitudeEstimator,
    LegacyQPolarGradEstimator,
    MSBEstimator,
    RandomEstimator,
    Ranking,
)
from .statistics import summarize


@dataclass(frozen=True)
class TrialMeasurement:
    trial: int
    policy: str
    schedule_sha256: str
    accuracy: float
    categorical_crossentropy: float
    schedule_hash: str
    proposed_events: int
    applied_events: int
    suppressed_events: int
    unique_applied_bits: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentOutcome:
    configuration: ExperimentConfiguration
    scientific_payload: Mapping[str, Any]
    trials: tuple[TrialMeasurement, ...]
    rankings: Mapping[str, list[list[Any]]]
    protection_masks: Mapping[str, list[list[Any]]]
    fault_schedules: tuple[Mapping[str, Any], ...]
    estimator_runtimes_seconds: Mapping[str, float] = field(default_factory=dict)

    @property
    def scientific_payload_sha256(self) -> str:
        return digest_object(self.scientific_payload)


def _arrays_digest(values: Mapping[str, np.ndarray]) -> str:
    payload = []
    for name, value in sorted(values.items()):
        array = np.asarray(value)
        payload.append(
            {
                "name": name,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "bytes_sha256": __import__("hashlib").sha256(
                    array.tobytes(order="C")
                ).hexdigest(),
            }
        )
    return digest_object(payload)


def _codec_exhaustive(codec: BitCodec) -> bool:
    format_ = codec.format
    encoded = np.arange(1 << format_.total_bits, dtype=np.uint64)
    decoded = codec.decode(encoded)
    if not np.array_equal(codec.encode(decoded), encoded):
        return False
    for bit in range(format_.total_bits):
        expected = codec.decode(codec.flip(encoded, bit)) - decoded
        actual = codec.perturbation(decoded, bit)
        if not np.array_equal(expected, actual):
            return False
    return True


def _construct_rankings(
    configuration: ExperimentConfiguration,
    adapter: BenchmarkAdapter,
    identities: tuple[BitIdentity, ...],
    ranking_inputs: np.ndarray,
    ranking_labels: np.ndarray,
    checkpoint_state: Mapping[str, np.ndarray],
    codec: BitCodec,
) -> tuple[tuple[Ranking, ...], Mapping[str, Ranking], Mapping[str, float]]:
    requested = {value.name for value in configuration.estimators}
    needs_gradients = bool(
        requested
        & {
            "gradient_magnitude",
            "first_order_gradient_delta",
            "legacy_qpolargrad",
        }
    )
    gradients = (
        adapter.gradients(ranking_inputs, ranking_labels) if needs_gradients else {}
    )
    deltas = (
        {
            name: codec.perturbation_matrix(value)
            for name, value in checkpoint_state.items()
        }
        if "first_order_gradient_delta" in requested
        else {}
    )
    result = []
    by_result_name: dict[str, Ranking] = {}
    runtimes: dict[str, float] = {}
    for specification in configuration.estimators:
        started = perf_counter()
        if specification.name == "legacy_qpolargrad":
            source = specification.activation_source or "clean"
            impacts = adapter.polar_impacts(ranking_inputs, activation_source=source)
            ranking = LegacyQPolarGradEstimator().estimate(
                identities,
                gradients,
                impacts,
                activation_source=source,
            )
        elif specification.name == "msb_first":
            ranking = MSBEstimator().estimate(identities)
        elif specification.name == "random":
            ranking = RandomEstimator(
                configuration.execution.random_ranking_seed
            ).estimate(identities)
        elif specification.name == "gradient_magnitude":
            ranking = GradientMagnitudeEstimator().estimate(identities, gradients)
        elif specification.name == "first_order_gradient_delta":
            ranking = FirstOrderEstimator().estimate(identities, gradients, deltas)
        else:  # Configuration validation makes this unreachable.
            raise ValueError(f"unsupported estimator: {specification.name}")
        result.append(ranking)
        by_result_name[specification.result_name] = ranking
        runtimes[specification.result_name] = perf_counter() - started
    return tuple(result), by_result_name, runtimes


def _construct_masks(
    configuration: ExperimentConfiguration,
    rankings: Mapping[str, Ranking],
) -> dict[str, ProtectionMask]:
    masks = {}
    for specification in configuration.estimators:
        ranking = rankings[specification.result_name]
        if specification.policy == "global_ranking":
            mask = ProtectionPolicy.global_ranking(
                ranking, configuration.protection.coverage
            )
        elif specification.policy == "position_first":
            mask = ProtectionPolicy.position_first(
                ranking, configuration.protection.coverage
            )
        else:
            raise ValueError(f"unsupported protection policy: {specification.policy}")
        masks[specification.result_name] = mask
    return masks


def _evaluate_faulted(
    adapter: BenchmarkAdapter,
    inputs: np.ndarray,
    labels: np.ndarray,
    checkpoint_state: Mapping[str, np.ndarray],
    schedule: FaultSchedule,
    mask: ProtectionMask,
    codec: BitCodec,
) -> tuple[float, float, Any]:
    if schedule.model.persistence == "persistent_trial":
        corrupted, accounting = apply_persistent_schedule(checkpoint_state, schedule, mask, codec)
        with adapter.temporary_values(corrupted):
            evaluation = adapter.evaluate(inputs, labels)
        return evaluation.accuracy, evaluation.categorical_crossentropy, accounting
    if schedule.model.persistence != "per_inference":
        raise ValueError(f"unsupported fault persistence: {schedule.model.persistence}")
    accuracies, losses = [], []
    proposed = applied = suppressed = 0
    unique: set[BitIdentity] = set()
    for index in range(len(labels)):
        corrupted, accounting = apply_per_inference_schedule(
            checkpoint_state, schedule, mask, codec, index
        )
        with adapter.temporary_values(corrupted):
            evaluation = adapter.evaluate(inputs[index:index + 1], labels[index:index + 1])
        accuracies.append(evaluation.accuracy)
        losses.append(evaluation.categorical_crossentropy)
        proposed += accounting.proposed_events
        applied += accounting.applied_events
        suppressed += accounting.suppressed_events
        unique.update(
            event.bit for event in schedule.events
            if event.inference_index == index and event.bit not in mask.protected
        )
    from .faults import InjectionReport

    return float(np.mean(accuracies)), float(np.mean(losses)), InjectionReport(
        schedule.digest(), proposed, applied, suppressed, len(unique)
    )


def execute_experiment(
    configuration: ExperimentConfiguration,
    *,
    adapter: BenchmarkAdapter | None = None,
) -> ExperimentOutcome:
    """Execute a validated experiment and return typed, unsaved measurements."""

    codec = BitCodec(FixedPointFormat.parse(configuration.benchmark.quantization))
    benchmark_adapter = adapter or create_adapter(configuration, codec)
    benchmark_adapter.load()
    if configuration.schema_version == "2.0.0":
        evaluation_inputs, evaluation_labels = benchmark_adapter.dataset(
            configuration.benchmark.evaluation_samples, role="evaluation"
        )
        ranking_inputs, ranking_labels = benchmark_adapter.dataset(
            configuration.benchmark.ranking_samples, role="ranking"
        )
    else:
        evaluation_inputs, evaluation_labels = benchmark_adapter.dataset(
            configuration.benchmark.evaluation_samples
        )
        ranking_inputs, ranking_labels = benchmark_adapter.dataset(
            configuration.benchmark.ranking_samples
        )
    checkpoint_state = benchmark_adapter.values()
    checkpoint_baseline = benchmark_adapter.evaluate(
        evaluation_inputs, evaluation_labels
    )
    quantized_state = benchmark_adapter.quantized_values()
    quantization_errors = {
        name: float(np.max(np.abs(checkpoint_state[name] - quantized_state[name])))
        for name in checkpoint_state
    }

    with benchmark_adapter.temporary_values(checkpoint_state):
        identities = tuple(benchmark_adapter.bit_identities())
        rankings, rankings_by_result, estimator_runtimes = _construct_rankings(
            configuration,
            benchmark_adapter,
            identities,
            ranking_inputs,
            ranking_labels,
            checkpoint_state,
            codec,
        )
        masks = _construct_masks(configuration, rankings_by_result)
        modern = configuration.schema_version == "2.0.0"
        if modern:
            universe = frozenset(identities)
            masks = {
                "no_protection": ProtectionMask("control_none", frozenset(), len(identities), 0.0, {"control": True}),
                **masks,
                "full_protection": ProtectionMask(
                    "control_full", universe, len(identities), float(len(identities)), {"control": True}
                ),
            }
        fault_model = FaultModel(
            target=configuration.fault.target,
            location=configuration.fault.location,
            persistence=configuration.fault.persistence,
            distribution=configuration.fault.distribution,
            probability=configuration.fault.probability,
        )
        rows: list[TrialMeasurement] = []
        schedules: list[FaultSchedule] = []
        pre_trials_state = _arrays_digest(benchmark_adapter.values())
        for trial in range(configuration.execution.trials):
            schedule = FaultSchedule.generate(
                fault_model,
                identities,
                seed=configuration.execution.fault_seed + trial,
                inference_count=configuration.benchmark.evaluation_samples,
            )
            schedules.append(schedule)
            for policy_name, mask in masks.items():
                accuracy, loss, accounting = _evaluate_faulted(
                    benchmark_adapter, evaluation_inputs, evaluation_labels,
                    checkpoint_state, schedule, mask, codec,
                )
                rows.append(
                    TrialMeasurement(
                        trial,
                        policy_name,
                        schedule.digest(),
                        accuracy,
                        loss,
                        accounting.schedule_hash,
                        accounting.proposed_events,
                        accounting.applied_events,
                        accounting.suppressed_events,
                        accounting.unique_applied_bits,
                    )
                )
        post_trials_state = _arrays_digest(benchmark_adapter.values())

    if _arrays_digest(benchmark_adapter.values()) != _arrays_digest(checkpoint_state):
        raise RuntimeError("checkpoint state was not restored after experiment")

    policy_results: dict[str, Any] = {}
    for policy_name, mask in masks.items():
        selected = [row for row in rows if row.policy == policy_name]
        policy_results[policy_name] = {
            "protected_bits": len(mask.protected),
            "coverage": mask.coverage,
            "mask_sha256": mask.digest(),
            "accuracy": summarize(
                [row.accuracy for row in selected],
                confidence=configuration.execution.confidence,
                bounds=(0.0, 1.0),
            ).to_dict(),
            "categorical_crossentropy": summarize(
                [row.categorical_crossentropy for row in selected],
                confidence=configuration.execution.confidence,
            ).to_dict(),
            "mean_applied_events": float(
                np.mean([row.applied_events for row in selected])
            ),
            "mean_suppressed_events": float(
                np.mean([row.suppressed_events for row in selected])
            ),
        }

    exact_rankings = {
        ranking.estimator: [
            [identity.parameter, identity.flat_index, identity.bit_index, float(score)]
            for identity, score in sorted(ranking.scores.items())
        ]
        for ranking in rankings
    }
    exact_masks = {
        policy_name: [
            [identity.parameter, identity.flat_index, identity.bit_index]
            for identity in sorted(mask.protected)
        ]
        for policy_name, mask in masks.items()
    }
    exact_schedules = tuple(
        {
            "trial": trial,
            "seed": schedule.seed,
            "sha256": schedule.digest(),
            "events": [
                [
                    event.inference_index,
                    event.bit.parameter,
                    event.bit.flat_index,
                    event.bit.bit_index,
                ]
                for event in schedule.events
            ],
        }
        for trial, schedule in enumerate(schedules)
    )
    schedule_by_trial = {
        trial: {row.schedule_sha256 for row in rows if row.trial == trial}
        for trial in range(configuration.execution.trials)
    }
    qpolar_name = next(
        (
            value.result_name
            for value in configuration.estimators
            if value.name == "legacy_qpolargrad"
        ),
        None,
    )
    msb_name = next(
        (
            value.result_name
            for value in configuration.estimators
            if value.name == "msb_first"
        ),
        None,
    )
    full_msb_equivalent = False
    if qpolar_name is not None and msb_name is not None:
        full_tier = 1.0 / codec.format.total_bits
        full_msb_equivalent = (
            ProtectionPolicy.position_first(
                rankings_by_result[qpolar_name], full_tier
            ).protected
            == ProtectionPolicy.global_ranking(
                rankings_by_result[msb_name], full_tier
            ).protected
        )

    scientific_payload: dict[str, Any] = {
        "schema": (
            "netsurf-quantized-weight-experiment/v2"
            if configuration.schema_version == "2.0.0"
            else "netsurf-v2-mnist-lite-milestone/v1"
        ),
        "benchmark": {
            "id": configuration.benchmark.id,
            "checkpoint_sha256": sha256_file(configuration.benchmark.checkpoint),
            "dataset_sha256": sha256_file(configuration.benchmark.dataset),
            "split": configuration.benchmark.split,
            "sample_count": configuration.benchmark.evaluation_samples,
            "ranking_sample_count": configuration.benchmark.ranking_samples,
            "normalization": {
                "source_range": [0, 255],
                "target_range": [codec.format.value_min, codec.format.value_max],
            },
            "trainable_scalars": sum(
                specification.size
                for specification in benchmark_adapter.parameter_specs()
            ),
            "logical_bits": len(identities),
        },
        "quantization": codec.format.to_dict(),
        "fault_model": asdict(fault_model),
        "configuration": {
            "trials": configuration.execution.trials,
            "ber": configuration.fault.probability,
            "coverage": configuration.protection.coverage,
            "random_ranking_seed": configuration.execution.random_ranking_seed,
            "fault_seed_first": configuration.execution.fault_seed,
            "fault_seed_last": configuration.execution.fault_seed
            + configuration.execution.trials
            - 1,
        },
        "baseline": {"checkpoint": asdict(checkpoint_baseline)},
        "quantization_verification": {
            "maximum_absolute_quantization_error": max(quantization_errors.values()),
            "per_parameter_maximum_absolute_error": quantization_errors,
            "encoded_values_round_trip": all(
                np.array_equal(
                    codec.encode(codec.decode(codec.encode(value))),
                    codec.encode(value),
                )
                for value in checkpoint_state.values()
            ),
        },
        "rankings": {
            ranking.estimator: {"sha256": ranking.digest(), **ranking.metadata}
            for ranking in rankings
        },
        "policies": policy_results,
        "schedule_hashes": [schedule.digest() for schedule in schedules],
        "exact_artifact_sha256": {
            "rankings": digest_object(exact_rankings),
            "protection_masks": digest_object(exact_masks),
            "fault_schedules": digest_object(exact_schedules),
        },
        "checks": {
            "codec_exhaustive": _codec_exhaustive(codec),
            "full_msb_mask_equivalent": full_msb_equivalent,
            "state_restored": pre_trials_state == post_trials_state,
            "matched_schedules": all(
                len(hashes) == 1 for hashes in schedule_by_trial.values()
            ),
        },
    }
    if modern:
        comparisons = {}
        unprotected = [row for row in rows if row.policy == "no_protection"]
        for specification in configuration.estimators:
            protected = [row for row in rows if row.policy == specification.result_name]
            comparisons[specification.result_name] = {
                "accuracy_difference": summarize(
                    [p.accuracy - u.accuracy for p, u in zip(protected, unprotected)],
                    confidence=configuration.execution.confidence,
                    bounds=(-1.0, 1.0),
                ).to_dict(),
                "crossentropy_difference": summarize(
                    [p.categorical_crossentropy - u.categorical_crossentropy for p, u in zip(protected, unprotected)],
                    confidence=configuration.execution.confidence,
                ).to_dict(),
                "pairing": "same fault schedule within trial",
            }
        scientific_payload["paired_comparisons"] = comparisons
        scientific_payload["benchmark"]["quantization_backend"] = configuration.benchmark.quantization_backend
        scientific_payload["benchmark"]["injectable_parameters"] = list(configuration.benchmark.injectable_parameters)
        if configuration.benchmark.adapter == "mnist_quantized":
            scientific_payload["benchmark"]["execution_backend"] = (
                "tensorflow-cpu-eager-onednn-disabled/v1"
            )
        if qpolar_name is None:
            scientific_payload["checks"]["full_msb_mask_equivalent"] = "not_applicable"
    return ExperimentOutcome(
        configuration,
        scientific_payload,
        tuple(rows),
        exact_rankings,
        exact_masks,
        exact_schedules,
        estimator_runtimes,
    )
