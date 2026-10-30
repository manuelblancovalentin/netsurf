"""Exact isolated and common-background conditional bit-damage evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import math
from time import monotonic
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .adapters.registry import create_adapter
from .conditional_config import ConditionalConfiguration
from .faults import FaultEvent, FaultModel, FaultSchedule, InjectionReport
from .identities import BitIdentity
from .manifests import digest_object
from .protection import ProtectionMask
from .samples import SamplePartition, require_disjoint
from .statistics import summarize
from .task_metrics import evaluation_contract


@dataclass(frozen=True)
class ConditionalOutcome:
    configuration: ConditionalConfiguration
    scientific_payload: Mapping[str, Any]
    partitions: Mapping[str, SamplePartition]
    candidate_records: tuple[Mapping[str, Any], ...]
    background_records: tuple[Mapping[str, Any], ...]
    measurements: tuple[Mapping[str, Any], ...]
    summaries: tuple[Mapping[str, Any], ...]
    rank_comparisons: tuple[Mapping[str, Any], ...]
    runtime_seconds: float

    @property
    def scientific_payload_sha256(self) -> str:
        return digest_object(self.scientific_payload)


def _state_sha256(values: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        value = np.ascontiguousarray(values[name])
        digest.update(name.encode("utf-8")); digest.update(b"\0")
        digest.update(value.dtype.str.encode("ascii")); digest.update(b"\0")
        digest.update(repr(tuple(value.shape)).encode("ascii")); digest.update(b"\0")
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _model(probability: float) -> FaultModel:
    return FaultModel(
        "weight", "logical_trainable_parameter_bits", "persistent_trial",
        "independent_bernoulli", probability,
    )


def _schedule(
    probability: float,
    seed: int,
    events: Sequence[BitIdentity],
    universe_size: int,
    inference_count: int,
) -> FaultSchedule:
    ordered = tuple(FaultEvent(None, value) for value in sorted(set(events)))
    return FaultSchedule(_model(probability), int(seed), ordered, universe_size, inference_count)


def _master_backgrounds(
    universe: Sequence[BitIdentity], probability: float, count: int, seed: int,
) -> tuple[tuple[int, tuple[BitIdentity, ...]], ...]:
    result = []
    for field in range(count):
        field_seed = int(seed + field * 1_000_003)
        generator = np.random.default_rng(field_seed)
        selected = generator.random(len(universe)) < probability
        result.append((field_seed, tuple(value for value, hit in zip(universe, selected) if hit)))
    return tuple(result)


def _risk_value(evaluation, expected_metric: str) -> float:
    _primary_name, _primary, _primary_direction, loss_name, loss, loss_direction = (
        evaluation_contract(evaluation)
    )
    if loss_name != expected_metric or loss_direction != "minimize":
        raise ValueError(
            f"adapter exposes {loss_name}/{loss_direction}, expected {expected_metric}/minimize"
        )
    if not math.isfinite(loss):
        raise FloatingPointError(f"nonfinite task risk: {loss}")
    return float(loss)


def _average_ranks_descending(values: Mapping[BitIdentity, float]) -> dict[BitIdentity, float]:
    ordered = sorted(values, key=lambda identity: (-values[identity], identity))
    result: dict[BitIdentity, float] = {}
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        rank = 0.5 * ((start + 1) + end)
        for identity in ordered[start:end]:
            result[identity] = rank
        start = end
    return result


def _rank_summary(
    split: str,
    probability: float,
    isolated: Mapping[BitIdentity, float],
    conditional: Mapping[BitIdentity, float],
) -> dict[str, Any]:
    isolated_ranks = _average_ranks_descending(isolated)
    conditional_ranks = _average_ranks_descending(conditional)
    identities = sorted(isolated)
    reversals = comparable = 0
    for left_index, left in enumerate(identities):
        for right in identities[left_index + 1:]:
            isolated_difference = isolated[left] - isolated[right]
            conditional_difference = conditional[left] - conditional[right]
            if isolated_difference == 0.0 or conditional_difference == 0.0:
                continue
            comparable += 1
            reversals += int(isolated_difference * conditional_difference < 0.0)
    x = np.asarray([isolated_ranks[value] for value in identities], dtype=np.float64)
    y = np.asarray([conditional_ranks[value] for value in identities], dtype=np.float64)
    if np.array_equal(x, y):
        spearman: float | None = 1.0
    elif np.std(x) == 0.0 or np.std(y) == 0.0:
        # Correlation is mathematically undefined when either ranking is all ties.
        spearman = None
    else:
        spearman = float(np.corrcoef(x, y)[0, 1])
    return {
        "split": split,
        "logical_ber": probability,
        "candidate_count": len(identities),
        "strict_comparable_pairs": comparable,
        "strict_rank_reversals": reversals,
        "strict_rank_reversal_fraction": reversals / comparable if comparable else 0.0,
        "spearman_average_rank": spearman,
        "isolated_ranks": {value.key(): isolated_ranks[value] for value in identities},
        "conditional_ranks": {value.key(): conditional_ranks[value] for value in identities},
    }


def execute_conditional_damage(
    configuration: ConditionalConfiguration,
    *,
    adapter=None,
    progress: Callable[[str, int, int], None] | None = None,
) -> ConditionalOutcome:
    """Execute exact XOR damage under paired common Bernoulli backgrounds."""

    started = monotonic()
    benchmark = adapter or create_adapter(configuration, None)
    benchmark.load()
    calibration_x, calibration_y, calibration_partition = benchmark.sample_partition(
        "calibration", configuration.samples.calibration_count,
        offset=configuration.samples.calibration_offset,
    )
    evaluation_x, evaluation_y, evaluation_partition = benchmark.sample_partition(
        "final_evaluation", configuration.samples.evaluation_count,
        offset=configuration.samples.evaluation_offset,
    )
    partitions = {
        "calibration": calibration_partition,
        "held_out_evaluation": evaluation_partition,
    }
    if configuration.samples.require_disjoint:
        require_disjoint(tuple(partitions.values()))

    universe = tuple(benchmark.bit_identities())
    universe_set = set(universe)
    if len(universe_set) != len(universe):
        raise RuntimeError("eligible bit population contains duplicate identities")
    missing = tuple(value for value in configuration.candidates if value not in universe_set)
    if missing:
        raise ValueError(f"configured candidate is outside the eligible population: {missing[0].key()}")

    initial_state = benchmark.values()
    initial_state_sha256 = _state_sha256(initial_state)
    candidate_records = []
    for identity in configuration.candidates:
        original, flipped, original_value, flipped_value, _changed = benchmark.encoded_bit_transition(identity)
        candidate_records.append({
            "identity": identity.key(), "parameter": identity.parameter,
            "flat_index": identity.flat_index, "bit_index": identity.bit_index,
            "original_encoded": int(original), "flipped_encoded": int(flipped),
            "original_value": float(original_value), "flipped_value": float(flipped_value),
            "numerical_perturbation": float(flipped_value - original_value),
        })

    split_data = {
        "calibration": (
            calibration_x, calibration_y, calibration_partition,
            configuration.backgrounds.calibration_count,
            configuration.backgrounds.calibration_seed,
        ),
        "held_out_evaluation": (
            evaluation_x, evaluation_y, evaluation_partition,
            configuration.backgrounds.evaluation_count,
            configuration.backgrounds.evaluation_seed,
        ),
    }
    total = sum(
        len(configuration.candidates) * (1 + 2 * count) + 1
        for _name, (_x, _y, _partition, count, _seed) in split_data.items()
        for _probability in configuration.backgrounds.probabilities
    )
    completed = 0
    measurements = []
    backgrounds = []
    summaries = []
    rank_comparisons = []
    empty_mask = ProtectionMask("no_protection", frozenset(), len(universe), 0.0, {})

    for split, (inputs, labels, partition, field_count, split_seed) in split_data.items():
        evaluation_cache: dict[str, tuple[float, InjectionReport]] = {}

        def evaluate(schedule: FaultSchedule) -> tuple[float, InjectionReport]:
            key = schedule.digest()
            if key not in evaluation_cache:
                result, report = benchmark.evaluate_fault_schedule(inputs, labels, schedule, empty_mask)
                evaluation_cache[key] = (_risk_value(result, configuration.risk_metric), report)
            return evaluation_cache[key]

        isolated_by_candidate: dict[BitIdentity, float] = {}
        for probability_index, probability in enumerate(configuration.backgrounds.probabilities):
            probability_seed = split_seed + probability_index * 10_000_019
            master_fields = _master_backgrounds(universe, probability, field_count, probability_seed)
            empty = _schedule(probability, probability_seed - 1, (), len(universe), len(inputs))
            clean_risk, clean_report = evaluate(empty)
            completed += 1
            if progress is not None:
                progress(f"{split}: clean reference", completed, total)
            for field, (field_seed, master) in enumerate(master_fields):
                backgrounds.append({
                    "split": split, "logical_ber": probability, "field": field,
                    "seed": field_seed, "master_event_count": len(master),
                    "master_events": [value.key() for value in master],
                    "master_sha256": digest_object([value.key() for value in master]),
                    "sample_sha256": partition.sample_sha256,
                })

            for candidate in configuration.candidates:
                isolated_schedule = _schedule(
                    probability, probability_seed - 1, (candidate,), len(universe), len(inputs)
                )
                isolated_risk, isolated_report = evaluate(isolated_schedule)
                isolated_damage = isolated_risk - clean_risk
                isolated_by_candidate[candidate] = isolated_damage
                completed += 1
                if progress is not None:
                    progress(f"{split}: isolated {candidate.key()}", completed, total)

                conditional_values = []
                for field, (field_seed, master) in enumerate(master_fields):
                    selected_candidate = candidate in master
                    base_bits = tuple(value for value in master if value != candidate)
                    with_candidate_bits = (*base_bits, candidate)
                    base_schedule = _schedule(
                        probability, field_seed, base_bits, len(universe), len(inputs)
                    )
                    candidate_schedule = _schedule(
                        probability, field_seed, with_candidate_bits, len(universe), len(inputs)
                    )
                    base_risk, base_report = evaluate(base_schedule)
                    candidate_risk, candidate_report = evaluate(candidate_schedule)
                    conditional_damage = candidate_risk - base_risk
                    conditional_values.append(conditional_damage)
                    measurements.append({
                        "split": split, "sample_sha256": partition.sample_sha256,
                        "logical_ber": probability, "field": field, "field_seed": field_seed,
                        "candidate": candidate.key(), "parameter": candidate.parameter,
                        "flat_index": candidate.flat_index, "bit_index": candidate.bit_index,
                        "master_selected_candidate": selected_candidate,
                        "master_schedule_sha256": digest_object([value.key() for value in master]),
                        "background_schedule_sha256": base_schedule.digest(),
                        "candidate_schedule_sha256": candidate_schedule.digest(),
                        "clean_risk": clean_risk, "isolated_fault_risk": isolated_risk,
                        "isolated_damage": isolated_damage, "background_risk": base_risk,
                        "background_plus_candidate_risk": candidate_risk,
                        "conditional_damage": conditional_damage,
                        "background_proposed_events": base_report.proposed_events,
                        "background_applied_events": base_report.applied_events,
                        "background_suppressed_events": base_report.suppressed_events,
                        "background_unique_applied_bits": base_report.unique_applied_bits,
                        "candidate_proposed_events": candidate_report.proposed_events,
                        "candidate_applied_events": candidate_report.applied_events,
                        "candidate_suppressed_events": candidate_report.suppressed_events,
                        "candidate_unique_applied_bits": candidate_report.unique_applied_bits,
                        "empty_proposed_events": clean_report.proposed_events,
                        "isolated_proposed_events": isolated_report.proposed_events,
                    })
                    completed += 2
                    if progress is not None:
                        progress(f"{split}: field {field + 1}/{field_count}", completed, total)

                estimate = summarize(conditional_values, confidence=configuration.execution.confidence)
                summaries.append({
                    "split": split, "sample_sha256": partition.sample_sha256,
                    "logical_ber": probability, "candidate": candidate.key(),
                    "parameter": candidate.parameter, "flat_index": candidate.flat_index,
                    "bit_index": candidate.bit_index, "isolated_damage": isolated_damage,
                    "conditional_damage": estimate.to_dict(),
                    "background_field_count": field_count,
                    "uncertainty_unit": "independently seeded Bernoulli background field",
                    "candidate_sampling_uncertainty": "none: explicit prespecified fixed candidate list",
                    "ranking_mask_replications": 0,
                })

            conditional_by_candidate = {
                candidate: next(
                    row["conditional_damage"]["mean"] for row in summaries
                    if row["split"] == split and row["logical_ber"] == probability
                    and row["candidate"] == candidate.key()
                )
                for candidate in configuration.candidates
            }
            rank_comparisons.append(
                _rank_summary(split, probability, isolated_by_candidate, conditional_by_candidate)
            )

    final_state_sha256 = _state_sha256(benchmark.values())
    if final_state_sha256 != initial_state_sha256:
        raise RuntimeError("benchmark state differs after conditional-damage evaluation")

    payload = {
        "schema": "netsurf-exact-conditional-damage/v1",
        "scientific_protocol": configuration.scientific_protocol,
        "experiment": configuration.name,
        "benchmark": {
            "id": configuration.benchmark.id,
            "representation": benchmark.representation_manifest(),
            "task_metrics": {
                "risk": "categorical_crossentropy",
                "application_metric": "classification_accuracy",
                "authoritative_backend": configuration.benchmark.inference_backend,
            },
        },
        "partitions": {name: value.to_dict() for name, value in partitions.items()},
        "risk": {
            "name": configuration.risk_metric,
            "direction": configuration.risk_direction,
            "definition": "D_b(G)=R(G union {b})-R(G), with b excluded from G",
        },
        "candidate_population": {
            "selection": "explicit_prespecified",
            "count": len(configuration.candidates),
            "eligible_population_size": len(universe),
            "candidate_records_sha256": digest_object(candidate_records),
        },
        "background_design": {
            "distribution": "independent_bernoulli",
            "common_master_fields": True,
            "candidate_excluded_before_paired_evaluation": True,
            "probabilities": list(configuration.backgrounds.probabilities),
            "calibration_fields": configuration.backgrounds.calibration_count,
            "held_out_fields": configuration.backgrounds.evaluation_count,
            "background_records_sha256": digest_object(backgrounds),
        },
        "measurements_sha256": digest_object(measurements),
        "summaries": summaries,
        "rank_comparisons": rank_comparisons,
        "checks": {
            "partitions_disjoint": True,
            "exact_encoded_xor": True,
            "common_background_pairing": True,
            "candidate_excluded_from_background": True,
            "state_restored": final_state_sha256 == initial_state_sha256,
            "initial_state_sha256": initial_state_sha256,
            "final_state_sha256": final_state_sha256,
            "event_accounting": "proposed=applied+suppressed checked by adapter",
        },
        "uncertainty_scope": {
            "conditional_interval": "variation across independent Bernoulli background fields conditional on model, samples, candidates and probability",
            "evaluation_files": "fixed within each named split",
            "candidate_sampling": "none; fixed prespecified candidates",
            "ranker_replications": "not applicable; no mask is selected in Phase 1",
        },
    }
    return ConditionalOutcome(
        configuration, payload, partitions, tuple(candidate_records), tuple(backgrounds),
        tuple(measurements), tuple(summaries), tuple(rank_comparisons), monotonic() - started,
    )


def exact_bernoulli_risk(
    risk: Callable[[tuple[int, ...]], float], bit_count: int, probability: float,
    *, protected: frozenset[int] = frozenset(),
) -> float:
    """Integrate a Boolean fault-risk function exactly over unprotected bits."""

    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0,1]")
    available = tuple(index for index in range(bit_count) if index not in protected)
    value = 0.0
    for state in itertools.product((0, 1), repeat=len(available)):
        faults = tuple(available[index] for index, enabled in enumerate(state) if enabled)
        count = len(faults)
        mass = probability**count * (1.0 - probability) ** (len(available) - count)
        value += mass * float(risk(faults))
    return value


def exact_conditional_damage(
    risk: Callable[[tuple[int, ...]], float], bit_count: int, candidate: int, probability: float,
) -> float:
    """Return E_G[R(G union {b})-R(G)] exactly for a small Boolean oracle."""

    if not 0 <= candidate < bit_count:
        raise IndexError(candidate)
    others = tuple(index for index in range(bit_count) if index != candidate)
    value = 0.0
    for state in itertools.product((0, 1), repeat=len(others)):
        background = tuple(others[index] for index, enabled in enumerate(state) if enabled)
        count = len(background)
        mass = probability**count * (1.0 - probability) ** (len(others) - count)
        value += mass * (
            float(risk(tuple(sorted((*background, candidate))))) - float(risk(background))
        )
    return value
