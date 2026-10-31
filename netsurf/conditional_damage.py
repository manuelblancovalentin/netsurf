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
from .manifests import digest_object, sha256_file
from .protection import ProtectionMask
from .protection_context import verify_protection_context
from .paired_swap import paired_swap_field_contrast
from .samples import SamplePartition, require_disjoint
from .statistics import summarize
from .task_metrics import evaluation_contract
from .restricted_subcube import (
    analyze_restricted_response_tables, response_table_diagnostics,
    selection_stability_diagnostics, subcube_states,
    swap_contrast_from_response_table,
)


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
    oracle_response_records: tuple[Mapping[str, Any], ...]
    oracle_background_records: tuple[Mapping[str, Any], ...]
    oracle_derived_records: tuple[Mapping[str, Any], ...]
    oracle_summaries: tuple[Mapping[str, Any], ...]
    paired_swap_step_records: tuple[Mapping[str, Any], ...]
    paired_swap_field_records: tuple[Mapping[str, Any], ...]
    paired_swap_summaries: tuple[Mapping[str, Any], ...]
    protection_context_record: Mapping[str, Any]
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


def _sample_identity_records(
    inputs: np.ndarray, labels: np.ndarray, partition: SamplePartition,
) -> tuple[dict[str, Any], ...]:
    return tuple({
        "sample_id": f"{partition.source_split}:{index}",
        "source_split": partition.source_split,
        "original_index": int(index),
        "label": int(label),
        "encoded_input_sha256": hashlib.sha256(
            np.ascontiguousarray(sample).tobytes(order="C")
        ).hexdigest(),
    } for sample, label, index in zip(inputs, labels, partition.indices))


def _partition_diagnostics(labels: np.ndarray, partition: SamplePartition) -> dict[str, Any]:
    values, counts = np.unique(np.asarray(labels, dtype=np.int64), return_counts=True)
    indices = tuple(partition.indices)
    return {
        "role": partition.role,
        "source_split": partition.source_split,
        "count": len(indices),
        "first_index": indices[0],
        "last_index": indices[-1],
        "contiguous_source_indices": indices == tuple(range(indices[0], indices[-1] + 1)),
        "class_counts": {str(int(value)): int(count) for value, count in zip(values, counts)},
        "sample_sha256": partition.sample_sha256,
        "ordering_provenance": partition.provenance,
    }


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
    from .selection_preflight import (
        exposure_audit, load_frozen_selection, selection_contract, selection_summary, validate_workflow,
        reject_local_final_reuse,
    )
    if configuration.schema_version == "6.4.0":
        validate_workflow(dict(configuration.workflow), configuration.samples,
                          configuration.backgrounds, configuration.restricted_oracle,
                          configuration.paired_swap_evaluation, configuration.protection_context,
                          configuration.source_path)
        reject_local_final_reuse(configuration)
    replication_plan_record = None
    if configuration.schema_version == "6.5.0":
        from .replication_preflight import verify_replication_plan, load_replication_selection
        replication_plan_record = verify_replication_plan(configuration)
    campaign_record = None
    if configuration.schema_version == "6.6.0":
        from .cross_pool import verify_pool_configuration
        campaign_record = verify_pool_configuration(configuration, execution=True)
    mode = configuration.workflow.get("mode", "joint")
    frozen_source = load_frozen_selection(configuration) if mode == "final_only" else None
    exposure = exposure_audit(
        configuration.samples.evaluation_offset, configuration.samples.evaluation_count,
        configuration.workflow["purpose"],
    ) if configuration.workflow else None
    if replication_plan_record is not None:
        frozen_source = load_replication_selection(configuration)
        exposure = replication_plan_record["exposure_audit"]
    if campaign_record is not None:
        exposure = campaign_record["exposure_audit"]
    benchmark = adapter or create_adapter(configuration, None)
    benchmark.load()
    universe = tuple(benchmark.bit_identities())
    if campaign_record is not None and digest_object([b.key() for b in universe]) != campaign_record["universe_sha256"]:
        raise ValueError("campaign eligible population hash differs")
    universe_set = set(universe)
    if len(universe_set) != len(universe):
        raise RuntimeError("eligible bit population contains duplicate identities")
    missing = tuple(value for value in configuration.candidates if value not in universe_set)
    if missing:
        raise ValueError(f"configured candidate is outside the eligible population: {missing[0].key()}")
    context = verify_protection_context(
        configuration.protection_context, universe, configuration.candidates,
        expected_checkpoint_sha256=sha256_file(configuration.benchmark.checkpoint),
    )
    if (
        context.mode != "none"
        and len(context.original_inside_indices) != configuration.restricted_oracle.budget
    ):
        raise ValueError(
            "restricted budget must equal the source mask cardinality inside the candidate pool"
        )

    if frozen_source is not None:
        if frozen_source["protection_context_sha256"] != context.record["identity_sha256"]:
            raise ValueError("frozen protection context differs")
        if frozen_source["eligible_population_sha256"] != digest_object([b.key() for b in universe]):
            raise ValueError("frozen eligible bit population differs")

    sample_data = {}
    if mode == "joint":
        sample_data["calibration"] = benchmark.sample_partition(
            configuration.samples.calibration_role, configuration.samples.calibration_count,
            offset=configuration.samples.calibration_offset,
        )
    if mode != "selection_only":
        sample_data["held_out_evaluation"] = benchmark.sample_partition(
            configuration.samples.evaluation_role, configuration.samples.evaluation_count,
            offset=configuration.samples.evaluation_offset,
        )
        evaluation_x, evaluation_y, evaluation_partition = sample_data["held_out_evaluation"]
        if configuration.workflow and (
            tuple(evaluation_partition.indices) != tuple(exposure["requested_original_indices"])
            or evaluation_partition.source_split != "speech_commands_tfds_test_interleaved_blocks16"
        ):
            raise ValueError("actual final sample identities differ from the preflight exposure audit")
    model_selection_data = None
    if mode not in {"final_only", "replication_only"} and configuration.samples.model_selection_role is not None:
        model_selection_data = benchmark.sample_partition(
            configuration.samples.model_selection_role,
            configuration.samples.model_selection_count,
            offset=configuration.samples.model_selection_offset,
        )
        sample_data["model_selection"] = model_selection_data
    partitions = {name: data[2] for name, data in sample_data.items()}
    if campaign_record is not None and "joint_campaign" in campaign_record:
        from .multiber_campaign import verify_member, verify_partition
        parent = verify_member(configuration.workflow["campaign_manifest"], campaign_record)
        for name in ("model_selection", "held_out_evaluation"):
            if name in sample_data:
                verify_partition(parent, name, sample_data[name])
    if configuration.samples.require_disjoint:
        require_disjoint(tuple(partitions.values()))
    partition_diagnostics = {
        name: _partition_diagnostics(data[1], data[2]) for name, data in sample_data.items()
    }
    # Legacy joint diagnostics enumerate all labels. Never do this in selection-only.
    complete_window_diagnostics = (
        benchmark.same_distribution_window_diagnostics()
        if mode == "joint" and hasattr(benchmark, "same_distribution_window_diagnostics") else {}
    )
    if frozen_source is not None:
        source_ids = {row["sample_id"] for row in frozen_source["sample_identities"]}
        actual_final_ids = {row["sample_id"] for row in _sample_identity_records(
            evaluation_x, evaluation_y, evaluation_partition)}
        if source_ids & actual_final_ids:
            raise ValueError("frozen selection and actual final samples overlap")

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

    split_data = {}
    if mode == "joint":
        calibration_x, calibration_y, calibration_partition = sample_data["calibration"]
        split_data = {
            "calibration": (*sample_data["calibration"], configuration.backgrounds.calibration_count,
                            configuration.backgrounds.calibration_seed),
            "held_out_evaluation": (*sample_data["held_out_evaluation"], configuration.backgrounds.evaluation_count,
                                    configuration.backgrounds.evaluation_seed),
        }
    operation_counts = configuration.operation_counts()
    total = operation_counts.get(
        "total_authoritative_evaluations_upper_bound",
        operation_counts["total_requested_evaluations"],
    )
    completed = 0
    measurements = []
    backgrounds = []
    summaries = []
    rank_comparisons = []
    oracle_response_records: list[dict[str, Any]] = []
    oracle_background_records: list[dict[str, Any]] = []
    oracle_derived_records: list[dict[str, Any]] = []
    oracle_summaries: list[dict[str, Any]] = []
    paired_swap_step_records: list[dict[str, Any]] = []
    paired_swap_field_records: list[dict[str, Any]] = []
    paired_swap_summaries: list[dict[str, Any]] = []
    paired_input_records = []
    base_mask = ProtectionMask(
        "fixed_protection_context_outside_candidates", context.outside,
        len(universe), float(len(context.outside)),
        {"protection_context_sha256": context.record["identity_sha256"]},
    )
    background_universe = tuple(value for value in universe if value not in context.outside)

    for split, (inputs, labels, partition, field_count, split_seed) in split_data.items():
        evaluation_cache: dict[str, tuple[float, InjectionReport]] = {}

        def evaluate(schedule: FaultSchedule) -> tuple[float, InjectionReport]:
            key = schedule.digest()
            if key not in evaluation_cache:
                result, report = benchmark.evaluate_fault_schedule(inputs, labels, schedule, base_mask)
                evaluation_cache[key] = (_risk_value(result, configuration.risk_metric), report)
            return evaluation_cache[key]

        isolated_by_candidate: dict[BitIdentity, float] = {}
        for probability_index, probability in enumerate(configuration.backgrounds.probabilities):
            probability_seed = split_seed + probability_index * 10_000_019
            master_fields = _master_backgrounds(
                background_universe, probability, field_count, probability_seed
            )
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

    if configuration.restricted_oracle.enabled:
        if model_selection_data is None and mode not in {"final_only", "replication_only"}:
            raise ValueError("restricted oracle requires an explicit model-selection partition")
        if model_selection_data is not None:
            selection_x, selection_y, selection_partition = model_selection_data
        candidates = configuration.candidates
        candidate_set = set(candidates)
        outside_universe = tuple(
            value for value in universe
            if value not in candidate_set and value not in context.outside
        )
        states = () if mode == "replication_only" else subcube_states(len(candidates))

        for probability_index, probability in enumerate(configuration.backgrounds.probabilities):
            selection_cache: dict[str, tuple[float, InjectionReport]] = {}
            evaluation_cache: dict[str, tuple[float, InjectionReport]] = {}

            def exact_risk(inputs, labels, schedule, cache):
                key = schedule.digest()
                hit = key in cache
                if not hit:
                    result, report = benchmark.evaluate_fault_schedule(
                        inputs, labels, schedule, base_mask
                    )
                    if report.proposed_events != report.applied_events + report.suppressed_events:
                        raise RuntimeError(
                            "restricted-oracle event accounting violated: "
                            "proposed != applied + suppressed"
                        )
                    cache[key] = (_risk_value(result, configuration.risk_metric), report)
                risk, report = cache[key]
                return risk, report, hit

            if mode in {"final_only", "replication_only"}:
                frozen_masks = frozen_source["masks"][probability_index]
                selected = tuple(frozen_masks["selected"])
                isolated_mask = tuple(frozen_masks["isolated"])
                original_mask = tuple(frozen_masks["original"])
                if original_mask != context.original_inside_indices:
                    raise ValueError("frozen original mask differs from verified context")
                if frozen_source["protection_context_sha256"] != context.record["identity_sha256"]:
                    raise ValueError("frozen protection context differs")
                if frozen_source["eligible_population_sha256"] != digest_object([b.key() for b in universe]):
                    raise ValueError("frozen eligible bit population differs")
                comparison_indices = {
                    "restricted_selected_vs_original_s5": (original_mask, selected),
                    "restricted_selected_vs_exact_isolated": (isolated_mask, selected),
                    "exact_isolated_vs_original_s5": (original_mask, isolated_mask),
                }
            else:
                isolated_table: dict[tuple[int, ...], float] = {}
                for state in ((), *((index,) for index in range(len(candidates)))):
                    bits = tuple(candidates[index] for index in state)
                    schedule = _schedule(
                        probability,
                        configuration.restricted_oracle.calibration_seed - 1,
                        bits, len(universe), len(selection_x),
                    )
                    risk, report, cache_hit = exact_risk(
                        selection_x, selection_y, schedule, selection_cache
                    )
                    completed += 1
                    if progress is not None:
                        progress("restricted oracle: isolated model-selection states", completed, total)
                    isolated_table[state] = risk
                    oracle_response_records.append({
                        "phase": "model_selection_isolated", "logical_ber": probability,
                        "field": -1, "field_seed": schedule.seed,
                        "candidate_state": list(state),
                        "candidate_bits": [candidates[index].key() for index in state],
                        "outside_background_sha256": digest_object([]),
                        "risk": risk, "schedule_sha256": schedule.digest(),
                        "proposed_events": report.proposed_events,
                        "applied_events": report.applied_events,
                        "suppressed_events": report.suppressed_events,
                        "unique_applied_bits": report.unique_applied_bits,
                        "cache_hit": cache_hit,
                    })

                def response_tables(
                    phase, inputs, labels, partition, count, base_seeds, cache,
                ):
                    nonlocal completed
                    tables = []
                    total_fields = count * len(base_seeds)
                    global_field = 0
                    for replicate, base_seed in enumerate(base_seeds):
                        probability_seed = base_seed + probability_index * 10_000_019
                        fields = _master_backgrounds(
                            outside_universe, probability, count, probability_seed
                        )
                        for local_field, (field_seed, outside) in enumerate(fields):
                            field = global_field
                            global_field += 1
                            outside_sha256 = digest_object([value.key() for value in outside])
                            probability_law = {
                                "name": "independent_bernoulli",
                                "logical_ber": probability,
                                "outside_population_size": len(outside_universe),
                                "candidate_probability": probability,
                                "target_support": "all subsets of unprotected eligible coordinates",
                            }
                            law_sha256 = digest_object(probability_law)
                            pairing_key = digest_object({
                                "phase": phase, "logical_ber": probability, "field": field,
                                "replicate": replicate, "seed": field_seed,
                                "outside_sha256": outside_sha256,
                                "sample_sha256": partition.sample_sha256,
                                "protection_context_sha256": context.record["identity_sha256"],
                            })
                            oracle_background_records.append({
                                "phase": phase, "logical_ber": probability, "field": field,
                                "replicate": replicate, "local_field": local_field,
                                "seed": field_seed, "outside_event_count": len(outside),
                                "outside_events": [value.key() for value in outside],
                                "outside_sha256": outside_sha256,
                                "sample_sha256": partition.sample_sha256,
                                "pairing_key": pairing_key,
                                "probability_law": probability_law,
                                "probability_law_sha256": law_sha256,
                                "protection_context_sha256": context.record["identity_sha256"],
                            })
                            table = {}
                            for state in states:
                                state_bits = tuple(candidates[index] for index in state)
                                schedule = _schedule(
                                    probability, field_seed, (*outside, *state_bits),
                                    len(universe), len(inputs),
                                )
                                risk, report, cache_hit = exact_risk(inputs, labels, schedule, cache)
                                completed += 1
                                if progress is not None:
                                    progress(
                                        f"restricted oracle: {phase} field {field + 1}/{total_fields}",
                                        completed, total,
                                    )
                                table[state] = risk
                                oracle_response_records.append({
                                    "phase": phase, "logical_ber": probability,
                                    "field": field, "replicate": replicate,
                                    "local_field": local_field, "field_seed": field_seed,
                                    "candidate_state": list(state),
                                    "candidate_bits": [candidates[index].key() for index in state],
                                    "outside_background_sha256": outside_sha256,
                                    "pairing_key": pairing_key,
                                    "probability_law_sha256": law_sha256,
                                    "protection_context_sha256": context.record["identity_sha256"],
                                    "risk": risk, "schedule_sha256": schedule.digest(),
                                    "proposed_events": report.proposed_events,
                                    "applied_events": report.applied_events,
                                    "suppressed_events": report.suppressed_events,
                                    "unique_applied_bits": report.unique_applied_bits,
                                    "cache_hit": cache_hit,
                                })
                            diagnostics = response_table_diagnostics(
                                table, len(candidates), configuration.restricted_oracle.budget,
                                probability,
                            )
                            oracle_derived_records.append({
                                "phase": phase, "logical_ber": probability, "field": field,
                                "replicate": replicate, "local_field": local_field,
                                "pairing_key": pairing_key,
                                "response_table_sha256": digest_object([
                                    {"state": list(state), "risk": table[state]} for state in states
                                ]),
                                "conditional_marginals": {
                                    candidates[index].key(): value
                                    for index, value in diagnostics["conditional_marginals"].items()
                                },
                                "mask_risks": [{
                                    "indices": list(mask),
                                    "risk": risk,
                                } for mask, risk in sorted(diagnostics["mask_risks"].items())],
                                "pairwise_mask_differences": [{
                                    "left_indices": list(pair[0]),
                                    "right_indices": list(pair[1]),
                                    "left_minus_right": value,
                                } for pair, value in sorted(
                                    diagnostics["pairwise_mask_differences"].items()
                                )],
                            })
                            tables.append(table)
                    return tuple(tables)

                calibration_tables = response_tables(
                    "model_selection", selection_x, selection_y, selection_partition,
                    configuration.restricted_oracle.calibration_background_count,
                    configuration.restricted_oracle.selection_seeds, selection_cache,
                )
                if mode == "selection_only":
                    row = selection_summary(
                        calibration_tables, isolated_table, candidates,
                        configuration.restricted_oracle.budget, probability,
                        context.original_inside_indices,
                        configuration.restricted_oracle.selection_seeds,
                        configuration.restricted_oracle.calibration_background_count,
                    )
                    row["global_protection_cardinality"] = len(context.original)
                    oracle_summaries.append(row)
                    continue
                held_out_tables = response_tables(
                    "held_out_evaluation", evaluation_x, evaluation_y, evaluation_partition,
                    configuration.restricted_oracle.evaluation_background_count,
                    (configuration.restricted_oracle.evaluation_seed,), evaluation_cache,
                )
                analysis = analyze_restricted_response_tables(
                    calibration_tables, held_out_tables, isolated_table,
                    candidate_count=len(candidates),
                    budget=configuration.restricted_oracle.budget,
                    probability=probability,
                    original_mask=(
                        context.original_inside_indices if context.mode != "none" else None
                    ),
                )
                selected = tuple(analysis["selected_mask"])
                isolated_mask = tuple(analysis["isolated_mask"])
                original_mask = tuple(analysis["original_mask"])
                posthoc = tuple(analysis["posthoc_held_out_optimum_mask"])
                selected_risks = tuple(analysis["held_out_selected_field_risks"])
                isolated_risks = tuple(analysis["held_out_isolated_field_risks"])
                original_risks = tuple(analysis["held_out_original_field_risks"])
                posthoc_risks = tuple(analysis["held_out_posthoc_optimum_field_risks"])
                paired_selected_minus_isolated = tuple(
                    left - right for left, right in zip(selected_risks, isolated_risks)
                )
                paired_selected_minus_original = tuple(
                    left - right for left, right in zip(selected_risks, original_risks)
                )
                paired_isolated_minus_original = tuple(
                    left - right for left, right in zip(isolated_risks, original_risks)
                )
                posthoc_regret = tuple(
                    left - right for left, right in zip(selected_risks, posthoc_risks)
                )
                stability = selection_stability_diagnostics(
                    calibration_tables,
                    candidate_count=len(candidates),
                    budget=configuration.restricted_oracle.budget,
                    probability=probability,
                )
                comparison_indices = {
                    "restricted_selected_vs_original_s5": (original_mask, selected),
                    "restricted_selected_vs_exact_isolated": (isolated_mask, selected),
                    "exact_isolated_vs_original_s5": (original_mask, isolated_mask),
                }
                table_equivalence = []
                for field, table in enumerate(held_out_tables):
                    for comparison, (baseline_indices, target_indices) in comparison_indices.items():
                        derived = swap_contrast_from_response_table(
                            table, len(candidates), probability,
                            baseline_indices, target_indices,
                        )
                        table_equivalence.append({
                            "field": field,
                            "comparison": comparison,
                            **derived,
                        })

            if configuration.paired_swap_evaluation.enabled:
                paired_configuration = configuration.paired_swap_evaluation
                paired_fields = _master_backgrounds(
                    background_universe, probability,
                    paired_configuration.background_count,
                    paired_configuration.seed + probability_index * 10_000_019,
                )
                requested_for_probability = paired_configuration.background_count * sum(
                    2 * len(set(baseline_indices) - set(target_indices))
                    for comparison, (baseline_indices, target_indices) in comparison_indices.items()
                    if comparison in paired_configuration.comparisons
                )
                if probability == 0.0:
                    requested_for_probability = 0
                configured_upper = (
                    paired_configuration.background_count
                    * len(paired_configuration.comparisons)
                    * 2 * configuration.restricted_oracle.budget
                )
                total -= configured_upper - requested_for_probability
                paired_cache: dict[str, tuple[float, InjectionReport, FaultSchedule]] = {}
                input_loss_cache = {}
                probability_field_rows: dict[str, list[float]] = {
                    name: [] for name in paired_configuration.comparisons
                }
                probability_law = {
                    "name": "independent_bernoulli",
                    "logical_ber": probability,
                    "master_population_size": len(background_universe),
                    "step_background_rule": (
                        "master minus fixed protection minus current candidate mask "
                        "minus exchanged bits"
                    ),
                    "target_support": "all subsets of unprotected eligible coordinates",
                }
                law_sha256 = digest_object(probability_law)
                for field, (field_seed, master) in enumerate(paired_fields):
                    master_sha256 = digest_object([value.key() for value in master])
                    master_pairing_key = digest_object({
                        "phase": "paired_swap_final", "logical_ber": probability,
                        "field": field, "seed": field_seed,
                        "master_sha256": master_sha256,
                        "sample_sha256": evaluation_partition.sample_sha256,
                        "protection_context_sha256": context.record["identity_sha256"],
                    })

                    def evaluate_faults(faults: frozenset[BitIdentity]) -> float:
                        schedule = _schedule(
                            probability, field_seed, tuple(faults),
                            len(universe), len(evaluation_x),
                        )
                        key = schedule.digest()
                        if key not in paired_cache:
                            if configuration.workflow.get("record_per_input", False):
                                with benchmark.capture_input_losses() as captured:
                                    result, report = benchmark.evaluate_fault_schedule(
                                        evaluation_x, evaluation_y, schedule, base_mask)
                                if len(captured) != 1 or captured[0].shape != (len(evaluation_x),):
                                    raise RuntimeError("expected one authoritative per-input loss vector")
                                if not np.isfinite(captured[0]).all() or float(np.mean(captured[0])) != _risk_value(result, configuration.risk_metric):
                                    raise RuntimeError("per-input losses differ from authoritative scalar risk")
                                input_loss_cache[key] = captured[0]
                            else:
                                result, report = benchmark.evaluate_fault_schedule(
                                    evaluation_x, evaluation_y, schedule, base_mask
                                )
                            if report.proposed_events != report.applied_events + report.suppressed_events:
                                raise RuntimeError(
                                    "paired-swap event accounting violated: proposed != applied + suppressed"
                                )
                            paired_cache[key] = (
                                _risk_value(result, configuration.risk_metric), report, schedule
                            )
                        return paired_cache[key][0]

                    for comparison in paired_configuration.comparisons:
                        baseline_indices, target_indices = comparison_indices[comparison]
                        baseline_bits = frozenset(candidates[index] for index in baseline_indices)
                        target_bits = frozenset(candidates[index] for index in target_indices)
                        result = paired_swap_field_contrast(
                            evaluate_faults, master, baseline_bits, target_bits,
                            probability=probability, fixed_protected=context.outside,
                        )
                        comparison_pairing_key = digest_object({
                            "master_pairing_key": master_pairing_key,
                            "comparison": comparison,
                            "path": [{
                                "removed": step.removed.key(), "added": step.added.key(),
                            } for step in result["path"]],
                        })
                        probability_field_rows[comparison].append(
                            float(result["target_minus_baseline"])
                        )
                        paired_swap_field_records.append({
                            "logical_ber": probability,
                            "field": field,
                            "field_seed": field_seed,
                            "comparison": comparison,
                            "sample_sha256": evaluation_partition.sample_sha256,
                            "master_event_count": len(master),
                            "master_events": [value.key() for value in master],
                            "master_sha256": master_sha256,
                            "master_pairing_key": master_pairing_key,
                            "comparison_pairing_key": comparison_pairing_key,
                            "probability_law": probability_law,
                            "probability_law_sha256": law_sha256,
                            "protection_context_sha256": context.record["identity_sha256"],
                            "baseline_mask": [value.key() for value in sorted(baseline_bits)],
                            "target_mask": [value.key() for value in sorted(target_bits)],
                            "path_length": len(result["path"]),
                            "target_minus_baseline": result["target_minus_baseline"],
                            "authoritative_evaluation_requests": result["authoritative_evaluations"],
                            "uncertainty_unit": "independently seeded master Bernoulli field",
                        })
                        per_input = np.zeros(len(evaluation_x), dtype=np.float64)
                        for step in result["steps"]:
                            removed = step["removed"]
                            added = step["added"]
                            background = frozenset(step["background"])
                            removed_schedule = _schedule(
                                probability, field_seed, tuple(background | {removed}),
                                len(universe), len(evaluation_x),
                            )
                            added_schedule = _schedule(
                                probability, field_seed, tuple(background | {added}),
                                len(universe), len(evaluation_x),
                            )
                            removed_cached = paired_cache.get(removed_schedule.digest())
                            added_cached = paired_cache.get(added_schedule.digest())
                            if configuration.workflow.get("record_per_input", False):
                                per_input += probability * (input_loss_cache[removed_schedule.digest()]
                                                            - input_loss_cache[added_schedule.digest()])
                            removed_report = removed_cached[1] if removed_cached else None
                            added_report = added_cached[1] if added_cached else None
                            paired_swap_step_records.append({
                                "logical_ber": probability,
                                "field": field,
                                "field_seed": field_seed,
                                "comparison": comparison,
                                "comparison_pairing_key": comparison_pairing_key,
                                "step": step["step"],
                                "removed": removed.key(),
                                "added": added.key(),
                                "current_mask": [value.key() for value in step["current_mask"]],
                                "background_event_count": len(background),
                                "background_events": [value.key() for value in sorted(background)],
                                "background_sha256": digest_object([
                                    value.key() for value in sorted(background)
                                ]),
                                "removed_schedule_sha256": removed_schedule.digest(),
                                "added_schedule_sha256": added_schedule.digest(),
                                "removed_risk": step["removed_risk"],
                                "added_risk": step["added_risk"],
                                "removed_minus_added": step["removed_minus_added"],
                                "target_minus_baseline_contribution": step[
                                    "target_minus_baseline_contribution"
                                ],
                                "removed_proposed_events": (
                                    removed_report.proposed_events if removed_report else 0
                                ),
                                "removed_applied_events": (
                                    removed_report.applied_events if removed_report else 0
                                ),
                                "removed_suppressed_events": (
                                    removed_report.suppressed_events if removed_report else 0
                                ),
                                "added_proposed_events": (
                                    added_report.proposed_events if added_report else 0
                                ),
                                "added_applied_events": (
                                    added_report.applied_events if added_report else 0
                                ),
                                "added_suppressed_events": (
                                    added_report.suppressed_events if added_report else 0
                                ),
                            })
                            completed += int(step["authoritative_evaluations"])
                            if progress is not None:
                                progress(
                                    f"paired swap: {comparison} field {field + 1}/"
                                    f"{paired_configuration.background_count}",
                                    completed, total,
                                )
                        if configuration.workflow.get("record_per_input", False):
                            residual = float(np.mean(per_input)) - float(result["target_minus_baseline"])
                            if not np.isclose(residual, 0., rtol=0., atol=1e-12):
                                raise RuntimeError("per-input whole-path reduction disagrees with scalar contrast")
                            paired_input_records.append({"logical_ber": probability, "field": field,
                                "field_seed": field_seed, "comparison": comparison,
                                "master_pairing_key": master_pairing_key,
                                "contributions": per_input.tolist(), "mean_roundoff_residual": residual})
                for comparison, values in probability_field_rows.items():
                    baseline_indices, target_indices = comparison_indices[comparison]
                    paired_swap_summaries.append({
                        "logical_ber": probability,
                        "comparison": comparison,
                        "baseline_mask_indices": list(baseline_indices),
                        "target_mask_indices": list(target_indices),
                        "path": [{
                            "removed": step.removed.key(), "added": step.added.key(),
                        } for step in paired_swap_field_contrast(
                            lambda _faults: 0.0, (),
                            frozenset(candidates[index] for index in baseline_indices),
                            frozenset(candidates[index] for index in target_indices),
                            probability=0.0,
                        )["path"]],
                        "target_minus_baseline": summarize(
                            values, confidence=configuration.execution.confidence
                        ).to_dict(),
                        "master_background_fields": len(values),
                        "uncertainty_unit": "independently seeded master Bernoulli field",
                        "actual_unique_authoritative_inferences_shared_across_comparisons": len(
                            paired_cache
                        ),
                        "interpretation": (
                            "frozen-mask expected-risk contrast; whole telescoping path is summed "
                            "before uncertainty aggregation"
                        ),
                    })
            if mode in {"final_only", "replication_only"}:
                continue
            oracle_summaries.append({
                "logical_ber": probability,
                "candidate_count": len(candidates),
                "budget": configuration.restricted_oracle.budget,
                "selected_mask_indices": list(selected),
                "selected_mask_bits": [candidates[index].key() for index in selected],
                "isolated_mask_indices": list(isolated_mask),
                "isolated_mask_bits": [candidates[index].key() for index in isolated_mask],
                "original_mask_indices": list(original_mask),
                "original_mask_bits": [candidates[index].key() for index in original_mask],
                "global_protection_cardinality": len(context.outside) + len(selected),
                "global_protection_cardinality_preserved": (
                    context.mode == "none"
                    or len(context.outside) + len(selected) == len(context.original)
                ),
                "posthoc_held_out_optimum_indices": list(posthoc),
                "posthoc_held_out_optimum_bits": [candidates[index].key() for index in posthoc],
                "posthoc_held_out_optimum_is_selection_valid": False,
                "model_selection_risk": summarize(
                    analysis["calibration_field_risks"][selected],
                    confidence=configuration.execution.confidence,
                    bounds=(0.0, math.inf),
                ).to_dict(),
                "held_out_selected_risk": summarize(
                    selected_risks, confidence=configuration.execution.confidence,
                    bounds=(0.0, math.inf),
                ).to_dict(),
                "held_out_isolated_risk": summarize(
                    isolated_risks, confidence=configuration.execution.confidence,
                    bounds=(0.0, math.inf),
                ).to_dict(),
                "held_out_original_risk": summarize(
                    original_risks, confidence=configuration.execution.confidence,
                    bounds=(0.0, math.inf),
                ).to_dict(),
                "held_out_selected_minus_isolated": summarize(
                    paired_selected_minus_isolated,
                    confidence=configuration.execution.confidence,
                ).to_dict(),
                "held_out_selected_minus_original": summarize(
                    paired_selected_minus_original,
                    confidence=configuration.execution.confidence,
                ).to_dict(),
                "held_out_isolated_minus_original": summarize(
                    paired_isolated_minus_original,
                    confidence=configuration.execution.confidence,
                ).to_dict(),
                "posthoc_held_out_regret": summarize(
                    posthoc_regret, confidence=configuration.execution.confidence,
                ).to_dict(),
                "calibration_background_fields": len(calibration_tables),
                "held_out_background_fields": len(held_out_tables),
                "requested_state_evaluations": (
                    len(candidates) + 1
                    + (len(calibration_tables) + len(held_out_tables)) * len(states)
                ),
                "unique_model_selection_inferences": len(selection_cache),
                "unique_held_out_inferences": len(evaluation_cache),
                "isolated_damage_by_bit": {
                    candidates[index].key(): value
                    for index, value in analysis["isolated_damages"].items()
                },
                "eligible_masks": [{
                    "indices": list(mask),
                    "bits": [candidates[index].key() for index in mask],
                    "model_selection_mean_risk": risk,
                    "model_selection_regret": risk - min(
                        analysis["calibration_mean_risks"].values()
                    ),
                } for mask, risk in sorted(analysis["calibration_mean_risks"].items())],
                "model_selection_stability": stability,
                "response_table_swap_equivalence": {
                    "comparisons": table_equivalence,
                    "maximum_absolute_error": max(
                        (row["absolute_equivalence_error"] for row in table_equivalence),
                        default=0.0,
                    ),
                    "interpretation": (
                        "exact algebraic equality on the same saved response tables; "
                        "separate paired Monte Carlo fields have finite-sample differences"
                    ),
                },
                "interpretation": (
                    "selected mask is chosen only on model-selection samples/backgrounds; "
                    "held-out selected-minus-isolated is paired and confirmatory; posthoc "
                    "held-out optimum/regret is diagnostic and selection-biased"
                ),
            })

    final_state_sha256 = _state_sha256(benchmark.values())
    if final_state_sha256 != initial_state_sha256:
        raise RuntimeError("benchmark state differs after conditional-damage evaluation")

    payload = {
        "schema": {
            "6.0.0": "netsurf-exact-conditional-damage/v1",
            "6.1.0": "netsurf-exact-conditional-protection/v2",
            "6.2.0": "netsurf-contextual-exact-conditional-protection/v3",
            "6.3.0": "netsurf-paired-swap-held-out-protection/v4",
            "6.4.0": "netsurf-frozen-selection-protection/v5",
            "6.5.0": "netsurf-independent-frozen-mask-replication/v6",
            "6.6.0": "netsurf-cross-pool-frozen-protection/v7",
        }[configuration.schema_version],
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
        "partition_diagnostics": partition_diagnostics,
        "complete_window_diagnostics": complete_window_diagnostics,
        "sample_identities": {
            name: list(_sample_identity_records(*data)) for name, data in sample_data.items()
        },
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
        "protection_context": context.record,
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
        "restricted_oracle": {
            "enabled": configuration.restricted_oracle.enabled,
            "definition": (
                "exact 2^K candidate-subcube integration under Bernoulli candidate faults "
                "for each independently sampled outside-network background"
            ),
            "true_global_optimum_computed": False,
            "summaries": oracle_summaries,
            "response_records_sha256": digest_object(oracle_response_records),
            "background_records_sha256": digest_object(oracle_background_records),
            "derived_records_sha256": digest_object(oracle_derived_records),
            "response_table_reuse": (
                "all fixed-budget risks, conditional marginals and pairwise mask differences "
                "are derived from saved exact 2^K state responses without new inference"
            ),
        },
        "paired_swap_evaluation": {
            "enabled": configuration.paired_swap_evaluation.enabled,
            "implementation_version": "exact-paired-swap-contrast/v1",
            "identity": (
                "J_p(target)-J_p(baseline) = p E_G[R(G union removed)-"
                "R(G union added)], telescoped over a deterministic legal path"
            ),
            "selection_independence": (
                "all masks and paths are frozen from model-selection evidence before "
                "paired final fields are evaluated"
            ),
            "summaries": paired_swap_summaries,
            "field_records_sha256": digest_object(paired_swap_field_records),
            "step_records_sha256": digest_object(paired_swap_step_records),
        },
        "checks": {
            "partitions_disjoint": True,
            "exact_encoded_xor": True,
            "common_background_pairing": True,
            "candidate_excluded_from_background": True,
            "fixed_context_excluded_from_background": True,
            "global_cardinality_preserved": all(
                row["global_protection_cardinality_preserved"] for row in oracle_summaries
            ),
            "state_restored": final_state_sha256 == initial_state_sha256,
            "initial_state_sha256": initial_state_sha256,
            "final_state_sha256": final_state_sha256,
            "event_accounting": "proposed=applied+suppressed checked by adapter",
            "paired_swap_masks_frozen_before_final_evaluation": True,
        },
        "uncertainty_scope": {
            "conditional_interval": "variation across independent Bernoulli background fields conditional on model, samples, candidates and probability",
            "evaluation_files": "fixed within each named split",
            "candidate_sampling": "none; fixed prespecified candidates",
            "ranker_replications": (
                "one deterministic restricted mask selected on model-selection data"
                if configuration.restricted_oracle.enabled else
                "not applicable; no mask is selected in Phase 1"
            ),
            "restricted_oracle": (
                "paired outside-background fields; selected mask fixed using model-selection "
                "data before held-out evaluation"
            ),
            "paired_swap": (
                "each independently seeded master field contributes one whole-path contrast; "
                "swap steps and model-selection fields are not independent replications"
            ),
        },
    }
    if configuration.workflow:
        payload["workflow"] = {k: v for k, v in configuration.workflow.items() if k not in {"selection_bundle", "replication_plan", "campaign_manifest", "final_registry"}}
        payload["exposure_audit"] = exposure
        payload["candidate_population"]["selection"] = configuration.workflow["candidate_provenance"]
        payload["risk"]["definition"] = (
            "exact Bernoulli-integrated cross-entropy for every fixed-budget candidate mask"
            if mode == "selection_only" else "J_p(target)-J_p(baseline), exact paired-swap field estimator"
        )
        payload["background_design"]["calibration_fields"] = 0
        payload["background_design"]["held_out_fields"] = 0
        payload["operation_counts"] = operation_counts
        payload["checks"]["final_samples_accessed"] = mode != "selection_only"
        payload["checks"]["final_fields_generated"] = mode != "selection_only"
        if mode == "selection_only":
            frozen = {
                "schema_version": "1.0.0",
                "contract": selection_contract(configuration),
                "sample_identities": payload["sample_identities"]["model_selection"],
                "sample_partition": {**payload["partitions"]["model_selection"],
                                     "indices": list(partitions["model_selection"].indices)},
                "protection_context_sha256": context.record["identity_sha256"],
                "eligible_population_sha256": digest_object([b.key() for b in universe]),
                "selection_background_records_sha256": digest_object(oracle_background_records),
                "masks": [{
                    "logical_ber": row["logical_ber"],
                    "selected": row["selected_mask_indices"],
                    "isolated": row["isolated_mask_indices"],
                    "original": row["original_mask_indices"],
                } for row in oracle_summaries],
            }
            payload["frozen_selection"] = frozen
            payload["frozen_selection_sha256"] = digest_object(frozen)
        else:
            payload["frozen_selection"] = frozen_source
            payload["frozen_selection_sha256"] = digest_object(frozen_source)
            payload["comparison_family"] = {
                "primary": configuration.paired_swap_evaluation.comparisons[0],
                "secondary": list(configuration.paired_swap_evaluation.comparisons[1:]),
                "interval": "two-sided Student-t on whole-path field contrasts",
                "multiplicity": "Bonferroni simultaneous family intervals in addition to marginal intervals",
            }
            family_confidence = 1 - (1 - configuration.execution.confidence) / len(paired_swap_summaries)
            for row in paired_swap_summaries:
                values = [r["target_minus_baseline"] for r in paired_swap_field_records
                          if r["comparison"] == row["comparison"] and r["logical_ber"] == row["logical_ber"]]
                row["familywise_target_minus_baseline"] = summarize(values, confidence=family_confidence).to_dict()
    if campaign_record is not None:
        payload["campaign_binding"] = {"campaign_sha256": configuration.workflow["campaign_sha256"],
                                       "pool_id": configuration.workflow["pool_id"]}
        payload["uncertainty_scope"]["candidate_sampling"] = "status-stratified uniform pool; generalization uses all prespecified campaign pools as units"
        for row in oracle_summaries:
            row["interpretation"] = "independently probability-sampled fixed pool; selected on model-selection fields only; no final performance measured in selection mode"
    if replication_plan_record is not None:
        from .replication_statistics import summarize_replication
        from .replication_preflight import statistical_contract
        payload["replication_plan"] = replication_plan_record
        payload["replication_plan_sha256"] = digest_object(replication_plan_record)
        payload["comparison_family"] = statistical_contract()
        payload["replication_analysis"] = summarize_replication(
            paired_swap_field_records, configuration.paired_swap_evaluation.background_count)
    if configuration.workflow.get("record_per_input", False) and mode != "selection_only":
        payload["paired_input_diagnostics"] = {
            "schema_version": "1.0.0",
            "sample_identities": payload["sample_identities"]["held_out_evaluation"],
            "records": paired_input_records,
            "interpretation": "crossed field/input effects; no independence of field-times-input cells; scalar field estimates remain primary"}
    return ConditionalOutcome(
        configuration, payload, partitions, tuple(candidate_records), tuple(backgrounds),
        tuple(measurements), tuple(summaries), tuple(rank_comparisons),
        tuple(oracle_response_records), tuple(oracle_background_records),
        tuple(oracle_derived_records), tuple(oracle_summaries),
        tuple(paired_swap_step_records), tuple(paired_swap_field_records),
        tuple(paired_swap_summaries), context.record,
        monotonic() - started,
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
