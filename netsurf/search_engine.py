"""Schema-4 execution engine for bounded conditional protection search."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np

from .adapters.registry import create_adapter
from .estimators_m3 import EstimatorOutput, basic_estimator, compute_estimators
from .identities import BitIdentity
from .incremental import DenseReluDenseIncrementalEvaluator, FaultState, metrics_from_logits
from .manifests import digest_object, sha256_file
from .ordering import OrderedRanking, compose_ordering
from .polar import PolarPairMeasurement, interaction_map, measure_polar_pair
from .quantization import BitCodec, FixedPointFormat
from .ranking import Ranking
from .samples import SamplePartition, require_disjoint
from .search import (
    PriorComponent,
    ProposalPrior,
    RankingPrior,
    SearchResult,
    conditional_search,
    exhaustive_optimum,
    sample_full_fault_states,
)
from .search_config import SearchConfiguration
from .statistics import summarize


@dataclass(frozen=True)
class SearchOutcome:
    configuration: SearchConfiguration
    scientific_payload: Mapping[str, Any]
    partitions: Mapping[str, SamplePartition]
    estimator_outputs: tuple[EstimatorOutput, ...]
    orderings: Mapping[str, OrderedRanking]
    candidate_pool: tuple[BitIdentity, ...]
    polar_measurements: tuple[PolarPairMeasurement, ...]
    search_results: tuple[SearchResult, ...]
    final_trials: tuple[Mapping[str, Any], ...]
    convergence: tuple[Mapping[str, Any], ...]
    incremental_validation: tuple[Mapping[str, Any], ...]
    exact_enumeration: tuple[Mapping[str, Any], ...]
    exact_optimum_mask: frozenset[BitIdentity]
    exact_optimum_risk: float | None
    replication_name: str = "replication_0"
    replications: tuple["SearchOutcome", ...] = ()

    @property
    def scientific_payload_sha256(self) -> str:
        return digest_object(self.scientific_payload)


def _compute_outputs(configuration, benchmark, inputs, labels, identities, codec):
    advanced = tuple(value for value in configuration.estimators if value.name not in {"msb_first", "random"})
    advanced_outputs = iter(compute_estimators(advanced, benchmark, inputs, labels, identities, codec))
    outputs = []
    for specification in configuration.estimators:
        output = basic_estimator(specification.name, identities, seed=specification.seed) if specification.name in {"msb_first", "random"} else next(advanced_outputs)
        if output.implementation_version != specification.implementation_version:
            raise RuntimeError(f"estimator version mismatch for {specification.name}")
        if output.name != specification.result_name:
            output = replace(output, name=specification.result_name, metadata={**output.metadata, "estimator_definition": specification.name})
        outputs.append(output)
    return tuple(outputs)


def _candidate_pool(configuration, identities, orderings, initial, seed):
    count = min(configuration.search.candidate_pool_count, len(identities))
    if configuration.search.candidate_pool_strategy == "all":
        selected = list(identities)
    elif configuration.search.candidate_pool_strategy == "uniform":
        generator = np.random.default_rng(seed)
        selected = [identities[int(value)] for value in generator.choice(len(identities), count, replace=False)]
    else:
        selected, seen, position = [], set(), 0
        ordered = tuple(orderings.values())
        while len(selected) < count:
            for ordering in ordered:
                if position < len(ordering.identities) and ordering.identities[position] not in seen:
                    selected.append(ordering.identities[position]); seen.add(ordering.identities[position])
                    if len(selected) == count:
                        break
            position += 1
    # The fixed-budget initial state must be reachable within a restricted pool.
    selected = list(dict.fromkeys(selected))[:count]
    selected_set = set(selected)
    for bit in sorted(initial):
        if bit not in selected_set:
            replacement = next((index for index in range(len(selected) - 1, -1, -1) if selected[index] not in initial), None)
            if replacement is None:
                raise ValueError("candidate pool cannot contain the complete initial mask")
            selected_set.remove(selected[replacement]); selected[replacement] = bit; selected_set.add(bit)
    return tuple(sorted(selected_set))


def _matched_final_trials(evaluator, inputs, labels, states, results, replication, progress=None):
    rows = []
    for result in results:
        for trial, state in enumerate(states):
            active = FaultState(state.bits - result.best_mask)
            metrics = metrics_from_logits(evaluator.full_forward(inputs, active).logits, labels)
            rows.append({
                "replication": replication, "strategy": result.strategy, "trial": trial,
                "schedule_sha256": digest_object([bit.key() for bit in sorted(state.bits)]),
                "fault_count": len(state.bits), "active_fault_count": len(active.bits),
                "accuracy": metrics.accuracy, "categorical_crossentropy": metrics.categorical_crossentropy,
            })
            if progress is not None: progress()
    return tuple(rows)


def _heldout_convergence(evaluator, inputs, labels, states, result, replication, progress=None):
    masks = [(0, result.initial_mask)]
    current = set(result.initial_mask)
    for step in result.steps:
        if step.accepted:
            current.remove(step.removed); current.add(step.added)
            masks.append((step.iteration, frozenset(current)))
    rows = []
    cumulative_time = cumulative_work = cumulative_evaluations = 0.0
    step_by_iteration = {step.iteration: step for step in result.steps}
    for iteration, mask in masks:
        for index in range(1, iteration + 1):
            step = step_by_iteration.get(index)
            if step is not None:
                cumulative_time += step.wall_seconds
                cumulative_work += step.work_proxy_macs
                cumulative_evaluations += step.exact_evaluations
        losses = []
        for state in states:
            losses.append(metrics_from_logits(evaluator.full_forward(inputs, FaultState(state.bits - mask)).logits, labels).categorical_crossentropy)
            if progress is not None: progress()
        summary = summarize(losses).to_dict()
        rows.append({"replication": replication, "strategy": result.strategy, "iteration": iteration,
                     "exact_evaluations": int(cumulative_evaluations), "work_proxy_macs": int(cumulative_work),
                     "wall_seconds": cumulative_time, "heldout_loss_mean": summary["mean"],
                     "heldout_loss_ci_low": summary["confidence_interval"][0], "heldout_loss_ci_high": summary["confidence_interval"][1]})
        cumulative_time = cumulative_work = cumulative_evaluations = 0.0
    return tuple(rows)


def _incremental_validation(evaluator, inputs, labels, identities):
    cases = []
    groups = {
        "final_layer": tuple(bit for bit in identities if bit.parameter.startswith("dense1"))[:2],
        "hidden_layer": tuple(bit for bit in identities if bit.parameter.startswith("dense0"))[:2],
        "cross_layer": (next(bit for bit in identities if bit.parameter.startswith("dense0")), next(bit for bit in identities if bit.parameter.startswith("dense1"))),
    }
    background = FaultState(frozenset())
    cache = evaluator.cache(inputs, background)
    for name, changed in groups.items():
        target = FaultState(frozenset(changed))
        full_times, incremental_times = [], []
        full = incremental = None
        for _ in range(3):
            full = evaluator.full_forward(inputs, target); full_times.append(full.diagnostics.wall_seconds)
            incremental = evaluator.incremental_forward(cache, inputs, target); incremental_times.append(incremental.diagnostics.wall_seconds)
        full_metrics, incremental_metrics = metrics_from_logits(full.logits, labels), metrics_from_logits(incremental.logits, labels)
        cases.append({
            "case": name, "earliest_layer": incremental.diagnostics.earliest_affected_layer,
            "changed_bits": len(changed), "changed_parameters": incremental.diagnostics.changed_parameters,
            "relu_gate_changes": incremental.diagnostics.relu_gate_changes,
            "max_logit_absolute_error": float(np.max(np.abs(full.logits - incremental.logits))),
            "loss_absolute_error": abs(full_metrics.categorical_crossentropy - incremental_metrics.categorical_crossentropy),
            "prediction_mismatches": int(np.count_nonzero(np.argmax(full.logits, axis=1) != np.argmax(incremental.logits, axis=1))),
            "full_seconds_mean": float(np.mean(full_times)), "full_seconds_std": float(np.std(full_times, ddof=1)),
            "incremental_seconds_mean": float(np.mean(incremental_times)), "incremental_seconds_std": float(np.std(incremental_times, ddof=1)),
            "full_work_proxy_macs": full.diagnostics.work_proxy_macs,
            "incremental_work_proxy_macs": incremental.diagnostics.work_proxy_macs,
            "fallback": incremental.diagnostics.fallback,
        })
    return tuple(cases)


def _execute_replication(configuration: SearchConfiguration, replication_index: int, tracker=None):
    replication = f"replication_{replication_index}"
    context = {"replication": replication_index + 1, "replications": configuration.replications.count}
    if tracker is not None: tracker.update("Model loading", **context)
    codec = BitCodec(FixedPointFormat.parse(configuration.benchmark.quantization))
    benchmark = create_adapter(configuration, codec); benchmark.load()
    if tracker is not None:
        tracker.update("Model loading", advance=1, **context)
        tracker.update("Sample partitioning", **context)
    offset = replication_index * configuration.replications.sample_offset_stride
    ranking_x, ranking_y, ranking_partition = benchmark.sample_partition("ranking", configuration.samples.ranking_count, offset=offset)
    search_x, search_y, search_partition = benchmark.sample_partition("empirical_validation", configuration.samples.search_count, offset=offset)
    final_x, final_y, final_partition = benchmark.sample_partition("evaluation", configuration.samples.final_count, offset=offset)
    partitions = {"ranking": ranking_partition, "search": search_partition, "final_evaluation": final_partition}
    if configuration.samples.require_disjoint:
        require_disjoint(tuple(partitions.values()))
    if tracker is not None:
        tracker.update("Sample partitioning", advance=1, **context)
    identities = tuple(benchmark.bit_identities())
    if tracker is not None: tracker.update("Ranking and initialization", **context)
    outputs = _compute_outputs(configuration, benchmark, ranking_x, ranking_y, identities, codec)
    by_name = {value.name: value for value in outputs}
    orderings = {}
    for guide in configuration.guidance:
        output = by_name[guide.estimator]
        orderings[guide.name] = compose_ordering(guide.name, Ranking(output.name, output.scores, output.metadata), identities, benchmark.parameter_specs(), guide.priority, seed=guide.seed)
    initial_ordering = orderings[configuration.initial_mask.guidance]
    if configuration.initial_mask.budget_bits > len(identities):
        raise ValueError("protection budget exceeds eligible population")
    initial = frozenset(initial_ordering.identities[:configuration.initial_mask.budget_bits])
    search_seed = configuration.search.seed + replication_index * configuration.replications.search_seed_stride
    pool = _candidate_pool(configuration, identities, orderings, initial, search_seed)
    pool_set = set(pool)
    components = [PriorComponent(RankingPrior.from_ordering(OrderedRanking(value.name, value.estimator, value.criteria, tuple(bit for bit in value.identities if bit in pool_set), value.seed)), guide.weight) for guide, value in ((g, orderings[g.name]) for g in configuration.guidance) if guide.weight > 0]
    proposal = ProposalPrior(pool, components, uniform_exploration=configuration.search.uniform_exploration, rank_temperature=configuration.search.rank_temperature)
    if tracker is not None: tracker.update("Ranking and initialization", advance=1, **context)
    evaluator = DenseReluDenseIncrementalEvaluator(benchmark, codec, cache_memory_limit_bytes=int(configuration.search.cache_memory_mb * 1024**2))
    if tracker is not None: tracker.update("Incremental exactness validation", **context)
    validation = _incremental_validation(evaluator, search_x, search_y, identities)
    if tracker is not None: tracker.update("Incremental exactness validation", advance=1, **context)

    polar_values = []
    if configuration.polar.enabled and configuration.polar.pair_budget:
        if tracker is not None: tracker.update("Sparse polar calibration", **context)
        possible = list(combinations(pool, 2)); generator = np.random.default_rng(configuration.polar.seed + replication_index)
        if len(possible) > configuration.polar.pair_budget:
            possible = [possible[int(i)] for i in generator.choice(len(possible), configuration.polar.pair_budget, replace=False)]
        for pair_index, (left, right) in enumerate(possible):
            if configuration.polar.mode == "finite_clean":
                backgrounds = (FaultState(frozenset()),)
            else:
                backgrounds = sample_full_fault_states(tuple(bit for bit in identities if bit not in {left, right}), probability=configuration.fault.probability, count=configuration.polar.background_contexts, seed=configuration.polar.seed + replication_index * 1009 + pair_index)
            for background in backgrounds:
                polar_values.append(measure_polar_pair(evaluator, search_x, search_y, background, left, right))
                if tracker is not None: tracker.update("Sparse polar calibration", advance=1, **context)
    interactions = interaction_map(polar_values)
    if configuration.polar.mode == "local_quadratic":
        grouped = {}
        for value in polar_values:
            grouped.setdefault((value.left, value.right), []).append(value.local_quadratic_interaction)
        interactions = {key: float(np.mean(values)) for key, values in grouped.items()}

    final_seed = configuration.final_evaluation.seed + replication_index * configuration.replications.final_seed_stride
    final_states = sample_full_fault_states(identities, probability=configuration.fault.probability, count=configuration.final_evaluation.contexts, seed=final_seed)
    results = []
    for strategy_index, strategy in enumerate(configuration.search.strategies):
        def search_progress(value):
            if tracker is not None:
                tracker.update("Conditional search", advance=1, replication=replication_index + 1,
                               replications=configuration.replications.count, logical_ber=configuration.fault.probability,
                               policy=strategy, iteration=value["iteration"], iterations=value["iterations"],
                               protected_bits=value["protected_bits"], estimated_risk=value["estimated_risk"],
                               best_estimated_risk=value["best_estimated_risk"], accepted=value["accepted"],
                               rejected=value["rejected"], exact_evaluations=value["exact_evaluations"],
                               work_class=f"Search {strategy}")
        result = conditional_search(
            evaluator, search_x, search_y, identities, initial, strategy=strategy, proposal=proposal,
            probability=configuration.fault.probability, iterations=configuration.search.iterations,
            background_contexts=configuration.search.background_contexts, seed=search_seed + strategy_index * 100_003,
            final_contexts=configuration.final_evaluation.contexts, final_seed=final_seed,
            confidence=configuration.final_evaluation.confidence, candidate_pool=pool,
            greedy_add_candidates=configuration.search.greedy_add_candidates,
            greedy_remove_candidates=configuration.search.greedy_remove_candidates,
            initial_temperature=configuration.search.initial_temperature, cooling_rate=configuration.search.cooling_rate,
            polar_interactions=interactions, polar_strength=configuration.polar.strength,
            evaluator_mode=configuration.search.evaluator, final_fault_states=final_states, progress=search_progress,
            final_inputs=final_x, final_labels=final_y,
        )
        results.append(result)
    if tracker is not None:
        tracker.update("Held-out final evaluation", **context)
    trials = _matched_final_trials(evaluator, final_x, final_y, final_states, results, replication, None if tracker is None else lambda: tracker.update("Held-out final evaluation", advance=1, **context))
    maximum_states = sum(1 if value == "static" else 1 + configuration.search.iterations for value in configuration.search.strategies)
    actual_states = sum(1 + result.accepted for result in results)
    if tracker is not None:
        tracker.adjust_total((actual_states - maximum_states) * configuration.final_evaluation.contexts, stage="Held-out convergence")
        tracker.update("Held-out convergence", **context)
    convergence = tuple(row for result in results for row in _heldout_convergence(evaluator, final_x, final_y, final_states, result, replication, None if tracker is None else lambda: tracker.update("Held-out convergence", advance=1, **context)))
    exact_rows, exact_mask, exact_risk = (), frozenset(), None
    if configuration.search.exact_enumeration:
        if tracker is not None: tracker.update("Restricted exact enumeration", **context)
        exact_mask, exact_risk, exact_rows = exhaustive_optimum(evaluator, search_x, search_y, pool, budget=len(initial), probability=configuration.fault.probability)
        if tracker is not None: tracker.update("Restricted exact enumeration", advance=1, **context)
    payload = {
        "schema_version": configuration.schema_version, "scientific_protocol": configuration.scientific_protocol,
        "replication": replication,
        "benchmark": {"id": configuration.benchmark.id, "checkpoint_sha256": sha256_file(configuration.benchmark.checkpoint), "dataset_sha256": sha256_file(configuration.benchmark.dataset), "eligible_bits": len(identities), "quantization": configuration.benchmark.quantization},
        "partitions": {name: value.to_dict() for name, value in partitions.items()},
        "fault": {"probability": configuration.fault.probability, "persistence": configuration.fault.persistence, "distribution": configuration.fault.distribution},
        "initial_mask": [bit.key() for bit in sorted(initial)], "candidate_pool": [bit.key() for bit in pool],
        "rankings": {name: {"digest": value.digest(), "criteria": value.criteria, "estimator": value.estimator} for name, value in orderings.items()},
        "search": [{
            "strategy": value.strategy, "best_mask": [bit.key() for bit in sorted(value.best_mask)],
            "accepted": value.accepted, "rejected": value.rejected,
            "best_estimated_risk": value.best_estimated_risk, "final_risk": value.final_risk,
            "steps": [{
                **{key: item for key, item in step.to_dict().items() if key != "wall_seconds"},
                "background_fault_states": step.background_fault_states,
                "evaluated_candidates": [candidate.to_dict() for candidate in step.evaluated_candidates],
            } for step in value.steps],
        } for value in results],
        "final_fault_states": [[bit.key() for bit in sorted(value.bits)] for value in final_states],
        "exact_optimum": None if exact_risk is None else {"risk": exact_risk, "mask": [bit.key() for bit in sorted(exact_mask)]},
    }
    if tracker is not None: tracker.update("Scientific payload", advance=1, **context)
    return SearchOutcome(configuration, payload, partitions, outputs, orderings, pool, tuple(polar_values), tuple(results), trials, convergence, validation, tuple(exact_rows), exact_mask, exact_risk, replication)


def execute_search(configuration: SearchConfiguration, *, _tracker=None) -> SearchOutcome:
    members = tuple(_execute_replication(configuration, index, _tracker) for index in range(configuration.replications.count))
    if len(members) == 1:
        return members[0]
    payload = {"schema_version": configuration.schema_version, "scientific_protocol": configuration.scientific_protocol,
               "replications": [{"name": value.replication_name, "payload_sha256": value.scientific_payload_sha256} for value in members]}
    first = members[0]
    return replace(first, scientific_payload=payload, replications=members)
