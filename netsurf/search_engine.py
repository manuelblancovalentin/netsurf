"""Schema-4 execution engine for bounded conditional protection search."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations
from time import perf_counter
from typing import Any, Mapping, Sequence
import resource

import numpy as np

from .adapters.registry import create_adapter
from .conditional_risk import ConditionalMaskComparison, compare_protection_masks
from .curvature import DenseReluDenseCurvature
from .factorized import FactorizedInteractionAggregate
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
    conditional_screening_search,
    estimate_mask_risk,
    exhaustive_optimum,
    sample_full_fault_states,
)
from .search_config import SearchConfiguration
from .statistics import summarize
from .topology import DenseReluDenseTopologyEvaluator


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
    conditional_comparisons: tuple[ConditionalMaskComparison, ...]
    logical_ber: float
    replication_name: str = "replication_0"
    replications: tuple["SearchOutcome", ...] = ()
    topology_measurements: tuple[Mapping[str, Any], ...] = ()
    protection_budget_bits: int = 0
    timing_records: tuple[Mapping[str, Any], ...] = ()

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


def _matched_final_trials(evaluator, inputs, labels, states, results, replication, progress=None, logical_ber=None):
    rows = []
    for result in results:
        for trial, state in enumerate(states):
            active = FaultState(state.bits - result.best_mask)
            metrics = metrics_from_logits(evaluator.full_forward(inputs, active).logits, labels)
            rows.append({
                "replication": replication, "strategy": result.strategy, "trial": trial,
                **({"logical_ber": logical_ber} if logical_ber is not None else {}),
                "schedule_sha256": digest_object([bit.key() for bit in sorted(state.bits)]),
                "fault_count": len(state.bits), "active_fault_count": len(active.bits),
                "accuracy": metrics.accuracy, "categorical_crossentropy": metrics.categorical_crossentropy,
            })
            if progress is not None: progress()
    return tuple(rows)


def _heldout_convergence(evaluator, inputs, labels, states, result, replication, progress=None, logical_ber=None):
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
        rows.append({"replication": replication, "strategy": result.strategy, **({"logical_ber":logical_ber} if logical_ber is not None else {}), "iteration": iteration,
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


def _execute_replication(configuration: SearchConfiguration, replication_index: int, tracker=None, *, probability=None, condition_index=0):
    whole_started = perf_counter(); phase_started = whole_started; timing_records = []
    def finish_phase(phase, category, operation_count=0):
        nonlocal phase_started
        finished = perf_counter()
        timing_records.append({"phase": phase, "category": category, "exclusive_wall_seconds": finished-phase_started, "operation_count": int(operation_count), "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0})
        phase_started = finished
    controlled = configuration.schema_version in {"4.1.0", "4.2.0", "4.3.0"}
    probability = configuration.fault.probability if probability is None else float(probability)
    if configuration.schema_version in {"4.2.0", "4.3.0"}:
        replication = f"budget_{configuration.initial_mask.budget_bits}_ber_{condition_index}_replication_{replication_index}"
    else:
        replication = f"ber_{condition_index}_replication_{replication_index}" if controlled else f"replication_{replication_index}"
    context = {"replication": replication_index + 1, "replications": configuration.replications.count}
    if tracker is not None: tracker.update("Model loading", **context)
    codec = BitCodec(FixedPointFormat.parse(configuration.benchmark.quantization))
    benchmark = create_adapter(configuration, codec); benchmark.load()
    finish_phase("model_loading", "shared_preprocessing", 1)
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
    finish_phase("sample_partitioning", "shared_preprocessing", 3)
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
    finish_phase("ranking_and_initialization", "shared_preprocessing", len(outputs))
    if tracker is not None: tracker.update("Ranking and initialization", advance=1, **context)
    evaluator = DenseReluDenseIncrementalEvaluator(benchmark, codec, cache_memory_limit_bytes=int(configuration.search.cache_memory_mb * 1024**2))
    if tracker is not None: tracker.update("Incremental exactness validation", **context)
    validation = _incremental_validation(evaluator, search_x, search_y, identities)
    finish_phase("incremental_validation", "shared_preprocessing", len(validation))
    if tracker is not None: tracker.update("Incremental exactness validation", advance=1, **context)

    polar_values = []
    topology_rows = []
    if configuration.polar.enabled and configuration.polar.pair_budget:
        if tracker is not None: tracker.update("Sparse polar calibration", **context)
        generator = np.random.default_rng(configuration.polar.seed + replication_index)
        if configuration.schema_version in {"4.2.0", "4.3.0"}:
            hidden = tuple(bit for bit in identities if bit.parameter.startswith("dense0/"))
            output = tuple(bit for bit in identities if bit.parameter.startswith("dense1/"))
            pool_hidden = tuple(bit for bit in pool if bit.parameter.startswith("dense0/")) or hidden
            pool_output = tuple(bit for bit in pool if bit.parameter.startswith("dense1/")) or output
            def sampled_pairs(left_population, right_population, budget, *, distinct):
                selected = set(); maximum = len(left_population) * len(right_population)
                if distinct and left_population is right_population:
                    maximum = len(left_population) * (len(left_population) - 1) // 2
                target = min(int(budget), maximum)
                attempts = 0
                while len(selected) < target and attempts < max(100, target * 50):
                    left = left_population[int(generator.integers(len(left_population)))]
                    right = right_population[int(generator.integers(len(right_population)))]
                    attempts += 1
                    if distinct and left == right:
                        continue
                    selected.add(tuple(sorted((left, right))))
                return sorted(selected)
            possible = []
            possible.extend(sampled_pairs(pool_output, output, configuration.topology.output_layer_pair_budget, distinct=True))
            possible.extend(sampled_pairs(pool_hidden, hidden, configuration.topology.hidden_layer_pair_budget, distinct=True))
            # Cross-layer discovery is anchored alternately in the candidate
            # pool and sampled from the full opposite-layer fault population.
            possible.extend(sampled_pairs(pool_hidden, output, configuration.topology.cross_layer_pair_budget, distinct=False))
            if tracker is not None and len(possible) != configuration.polar.pair_budget:
                polar_contexts = 1 if configuration.polar.mode == "finite_clean" else configuration.polar.background_contexts
                tracker.adjust_total((len(possible) - configuration.polar.pair_budget) * polar_contexts,
                                     stage="Sparse polar calibration")
            shared_topology_backgrounds = (() if configuration.polar.mode == "finite_clean" else
                sample_full_fault_states(
                    identities, probability=probability,
                    count=configuration.polar.background_contexts,
                    seed=configuration.polar.seed + replication_index * 1009 + condition_index * 1_000_003,
                ))
        elif controlled:
            all_pairs = list(combinations(pool, 2))
            # Prioritize boundary swaps each configured guidance is likely to
            # propose, then fill the bounded sparse budget uniformly.  This
            # improves decision-level diagnostic coverage without claiming a
            # dense interaction model.
            prioritized=[]; seen=set()
            for method in configuration.methods:
                for guidance_name in method.guidance:
                    ordering=orderings[guidance_name]
                    protected=tuple(bit for bit in reversed(ordering.identities) if bit in initial)
                    additions=tuple(bit for bit in ordering.identities if bit in pool_set and bit not in initial)
                    for removed,added in zip(protected,additions):
                        key=tuple(sorted((removed,added)))
                        if key not in seen:
                            prioritized.append(key); seen.add(key)
                            if len(prioritized)>=configuration.polar.pair_budget: break
                    if len(prioritized)>=configuration.polar.pair_budget: break
                if len(prioritized)>=configuration.polar.pair_budget: break
            remaining=[pair for pair in all_pairs if tuple(sorted(pair)) not in seen]
            needed=configuration.polar.pair_budget-len(prioritized)
            sampled=[] if needed<=0 else [remaining[int(i)] for i in generator.choice(len(remaining),min(needed,len(remaining)),replace=False)]
            possible=prioritized+sampled
        else:
            all_pairs = list(combinations(pool, 2))
            possible=all_pairs
            if len(possible) > configuration.polar.pair_budget:
                possible = [possible[int(i)] for i in generator.choice(len(possible), configuration.polar.pair_budget, replace=False)]
        topology_cache = {}
        for pair_index, (left, right) in enumerate(possible):
            if configuration.polar.mode == "finite_clean":
                backgrounds = (FaultState(frozenset()),)
            elif configuration.schema_version in {"4.2.0", "4.3.0"}:
                backgrounds = tuple(FaultState(state.bits - {left, right}) for state in shared_topology_backgrounds)
            else:
                backgrounds = sample_full_fault_states(tuple(bit for bit in identities if bit not in {left, right}), probability=probability, count=configuration.polar.background_contexts, seed=configuration.polar.seed + replication_index * 1009 + condition_index * 1_000_003 + pair_index)
            for background in backgrounds:
                measured = measure_polar_pair(evaluator, search_x, search_y, background, left, right)
                polar_values.append(measured)
                if configuration.schema_version in {"4.2.0", "4.3.0"}:
                    if background not in topology_cache:
                        topology = DenseReluDenseTopologyEvaluator(
                            evaluator, search_x, background,
                            response_cache_limit=configuration.topology.cache_response_limit,
                        )
                        topology_cache[background] = (topology, DenseReluDenseCurvature(topology))
                    topology, curvature = topology_cache[background]
                    exact = topology.pair(left, right, search_y)
                    local = kron = stratified = None
                    if left.parameter.split("/", 1)[0] == right.parameter.split("/", 1)[0]:
                        local = curvature.local(left, right)
                        kron = curvature.kron(left, right)
                        factors = curvature.stratified_factors(
                            curvature.coordinate(left).layer,
                            configuration.topology.confidence_strata,
                            minimum_samples=configuration.topology.minimum_stratum_samples,
                        )
                        stratified = curvature.stratified_kron(left, right, factors)
                    topology_rows.append({
                        "left": left.key(), "right": right.key(),
                        "left_identity": left, "right_identity": right,
                        "background_fault_count": len(background.bits),
                        "topology_class": exact.topology_class,
                        "finite_polar": float(np.mean(exact.finite_polar_per_sample)),
                        "topology_corrected": float(np.mean(exact.exact_loss_interaction_per_sample)),
                        "topology_exact_interaction_error": float(np.mean(exact.exact_loss_interaction_per_sample)) - measured.exact_interaction,
                        "local_quadratic": local, "kronecker": kron,
                        "stratified_kronecker": stratified,
                        "logit_residual_mean_l2": float(np.mean(np.linalg.norm(exact.logit_residual, axis=1))),
                        "maximum_bound_violation": exact.maximum_bound_violation,
                        "structurally_exact": exact.structurally_exact,
                        "gate_changes": exact.left.relu_gate_changes + exact.right.relu_gate_changes,
                        "wall_seconds": exact.wall_seconds,
                    })
                if tracker is not None: tracker.update("Sparse polar calibration", advance=1, **context)
    interactions = interaction_map(polar_values)
    interaction_maps = {"finite_polar": interactions}
    if configuration.schema_version in {"4.2.0", "4.3.0"}:
        for model in configuration.topology.interaction_models:
            if model in {"factorized_kronecker", "factorized_stratified_kronecker"}:
                continue
            grouped = {}
            for row in topology_rows:
                value = row[model]
                if value is not None:
                    grouped.setdefault((row["left_identity"], row["right_identity"]), []).append(float(value))
            interaction_maps[model] = {key: float(np.mean(values)) for key, values in grouped.items()}
    if configuration.polar.mode == "local_quadratic":
        grouped = {}
        for value in polar_values:
            grouped.setdefault((value.left, value.right), []).append(value.local_quadratic_interaction)
        interactions = {key: float(np.mean(values)) for key, values in grouped.items()}
    finish_phase("interaction_calibration", "shared_preprocessing", len(polar_values))

    final_seed = configuration.final_evaluation.seed + replication_index * configuration.replications.final_seed_stride + condition_index * 10_000_019
    final_states = () if controlled else sample_full_fault_states(identities, probability=probability, count=configuration.final_evaluation.contexts, seed=final_seed)
    results = []
    for strategy_index, method in enumerate(configuration.methods):
        strategy = method.strategy
        method_iterations = configuration.search.iterations if method.iterations is None else method.iterations
        method_components = [component for component, guide in zip(components, (g for g in configuration.guidance if g.weight > 0)) if guide.name in method.guidance]
        method_proposal = ProposalPrior(pool, method_components, uniform_exploration=configuration.search.uniform_exploration, rank_temperature=configuration.search.rank_temperature)
        def search_progress(value):
            if tracker is not None:
                tracker.update("Conditional search", advance=1, replication=replication_index + 1,
                               replications=configuration.replications.count, logical_ber=configuration.fault.probability,
                               policy=method.name, iteration=value["iteration"], iterations=method_iterations,
                               protected_bits=value["protected_bits"], estimated_risk=value["estimated_risk"],
                               best_estimated_risk=value["best_estimated_risk"], accepted=value["accepted"],
                               rejected=value["rejected"], exact_evaluations=value["exact_evaluations"],
                               screened_candidates=value.get("screened_candidates"),
                               verified_swaps=value.get("verified_swaps"),
                               work_class=f"Search {method.name}")
        method_interactions = interactions
        method_strength = configuration.polar.strength
        interaction_guidance = "direct_pair_v1"
        if configuration.schema_version in {"4.2.0", "4.3.0"}:
            method_interactions = {} if method.interaction_model in {"none", "factorized_kronecker", "factorized_stratified_kronecker"} else dict(interaction_maps[method.interaction_model])
            method_strength = configuration.polar.strength if method.polarity_strength is None else method.polarity_strength
            if method.shuffle_interactions and method_interactions:
                keys = sorted(method_interactions)
                values = np.asarray([method_interactions[key] for key in keys])
                np.random.default_rng(search_seed + strategy_index * 100_003 + 17).shuffle(values)
                method_interactions = dict(zip(keys, map(float, values)))
            interaction_guidance = configuration.topology.interaction_guidance
        if configuration.schema_version == "4.3.0" and strategy in {"marginal_screening", "corrected_screening", "factorized_screening", "hybrid_screening"}:
            aggregate = None
            if method.interaction_model in {"factorized_kronecker", "factorized_stratified_kronecker"}:
                clean_topology = DenseReluDenseTopologyEvaluator(
                    evaluator, search_x, FaultState.from_bits(),
                    response_cache_limit=configuration.screening.response_cache_limit,
                )
                aggregate = FactorizedInteractionAggregate(
                    DenseReluDenseCurvature(clean_topology), identities, initial,
                    probabilities=probability,
                    stratified=method.interaction_model == "factorized_stratified_kronecker",
                    strata=configuration.topology.confidence_strata,
                    minimum_samples=configuration.topology.minimum_stratum_samples,
                    population=identities if method.aggregate_population == "full" else pool,
                )
            result = conditional_screening_search(
                evaluator, search_x, search_y, identities, initial, strategy=strategy,
                proposal=method_proposal, probability=probability, iterations=method_iterations,
                background_contexts=configuration.search.background_contexts,
                seed=search_seed + strategy_index * 100_003, candidate_pool=pool,
                removal_shortlist=configuration.screening.removal_shortlist,
                addition_shortlist=configuration.screening.addition_shortlist,
                verification_shortlist=configuration.screening.verification_shortlist,
                confidence=configuration.final_evaluation.confidence,
                final_contexts=configuration.final_evaluation.contexts, final_seed=final_seed,
                defer_final_evaluation=controlled, aggregate=aggregate,
                dynamic_aggregate=method.dynamic_aggregate,
                interaction_strength=configuration.screening.interaction_strength,
                polar_interactions=interaction_maps.get("finite_polar", {}) if strategy == "hybrid_screening" else None,
                polar_correction_budget=method.polar_correction_budget,
                response_cache_limit=configuration.screening.response_cache_limit,
                refresh_frequency=configuration.screening.refresh_frequency,
                progress=search_progress,
            )
            if aggregate is not None:
                count = len(result.factorized_rows)
                result = replace(result, factorized_rows=tuple({**row,
                    "factor_build_seconds": aggregate.timing.factor_build_seconds if index == 0 else 0.0,
                    "aggregate_build_seconds": aggregate.timing.aggregate_build_seconds if index == 0 else 0.0,
                    "aggregate_query_seconds": aggregate.timing.query_seconds if index == count - 1 else 0.0,
                    "aggregate_update_seconds": aggregate.timing.update_seconds if index == count - 1 else 0.0,
                    "aggregate_population": method.aggregate_population,
                    "dynamic_aggregate": method.dynamic_aggregate,
                    "timing_semantics": "shared_setup_attributed_to_first_row_and_cumulative_query_update_to_last_row",
                } for index, row in enumerate(result.factorized_rows)))
        else:
            result = conditional_search(
                evaluator, search_x, search_y, identities, initial, strategy=strategy, proposal=method_proposal,
                probability=probability, iterations=method_iterations,
                background_contexts=configuration.search.background_contexts, seed=search_seed + strategy_index * 100_003,
                final_contexts=configuration.final_evaluation.contexts, final_seed=final_seed,
                confidence=configuration.final_evaluation.confidence, candidate_pool=pool,
                greedy_add_candidates=configuration.search.greedy_add_candidates,
                greedy_remove_candidates=configuration.search.greedy_remove_candidates,
                initial_temperature=configuration.search.initial_temperature, cooling_rate=configuration.search.cooling_rate,
                polar_interactions=method_interactions, polar_strength=method_strength,
                evaluator_mode=configuration.search.evaluator, final_fault_states=None if controlled else final_states, progress=search_progress,
                final_inputs=None if controlled else final_x, final_labels=None if controlled else final_y,
                defer_final_evaluation=controlled,
                background_seed=search_seed if controlled else None,
                interaction_guidance=interaction_guidance,
            )
        if controlled:
            result = replace(result, strategy=method.name, proposal_provenance={**result.proposal_provenance,"method":method.name,"algorithm":method.strategy,"guidance":list(method.guidance)})
        results.append(result)
    finish_phase("search", "search_and_exact_verification", sum(len(value.steps) for value in results))
    conditional_comparisons = ()
    if controlled:
        # These fields are created only after all final masks are fixed.  They
        # are therefore structurally unavailable to proposal selection.
        final_states = sample_full_fault_states(identities, probability=probability, count=configuration.final_evaluation.contexts, seed=final_seed)
        conditional_seed = configuration.final_evaluation.conditional_seed + replication_index * configuration.replications.final_seed_stride + condition_index * 10_000_019
        conditional_states = sample_full_fault_states(identities, probability=probability, count=configuration.final_evaluation.conditional_contexts, seed=conditional_seed)
        evaluated_results = []
        comparisons = []
        for result in results:
            final_risk, _ = estimate_mask_risk(evaluator, final_x, final_y, identities, result.best_mask, probability=probability, contexts=len(final_states), seed=final_seed, confidence=configuration.final_evaluation.confidence, full_fault_states=final_states)
            evaluated_results.append(replace(result, final_risk=final_risk))
            def conditional_progress(completed, total, *, method_name=result.strategy):
                if tracker is not None:
                    tracker.update("Held-out conditional evaluation", advance=1, replication=replication_index+1,
                                   replications=configuration.replications.count, logical_ber=probability,
                                   policy=method_name, trial=completed, trials=total)
            comparisons.append(compare_protection_masks(
                evaluator, final_x, final_y, identities, conditional_states, initial, result.best_mask,
                probabilities=probability, confidence=configuration.final_evaluation.confidence,
                baseline_name="initial_static", target_name=result.strategy, mode=configuration.search.evaluator,
                progress=conditional_progress,
            ))
        results = evaluated_results
        conditional_comparisons = tuple(comparisons)
    if tracker is not None:
        tracker.update("Held-out final evaluation", **context)
    trials = tuple({**row, **({"protection_budget_bits": len(initial)} if configuration.schema_version in {"4.2.0", "4.3.0"} else {})}
                   for row in _matched_final_trials(evaluator, final_x, final_y, final_states, results, replication, None if tracker is None else lambda: tracker.update("Held-out final evaluation", advance=1, logical_ber=probability, **context), probability if controlled else None))
    maximum_states = sum(1 if value.strategy == "static" else 1 + (configuration.search.iterations if value.iterations is None else value.iterations) for value in configuration.methods)
    actual_states = sum(1 + result.accepted for result in results)
    if tracker is not None:
        tracker.adjust_total((actual_states - maximum_states) * configuration.final_evaluation.contexts, stage="Held-out convergence")
        tracker.update("Held-out convergence", **context)
    convergence = tuple({**row, **({"protection_budget_bits": len(initial)} if configuration.schema_version in {"4.2.0", "4.3.0"} else {})}
                        for result in results for row in _heldout_convergence(evaluator, final_x, final_y, final_states, result, replication, None if tracker is None else lambda: tracker.update("Held-out convergence", advance=1, logical_ber=probability, **context), probability if controlled else None))
    exact_rows, exact_mask, exact_risk = (), frozenset(), None
    if configuration.search.exact_enumeration:
        if tracker is not None: tracker.update("Restricted exact enumeration", **context)
        exact_mask, exact_risk, exact_rows = exhaustive_optimum(evaluator, search_x, search_y, pool, budget=len(initial), probability=configuration.fault.probability)
        if tracker is not None: tracker.update("Restricted exact enumeration", advance=1, **context)
    finish_phase("heldout_and_exact_evaluation", "final_heldout_evaluation", len(trials)+len(convergence)+len(conditional_comparisons)+len(exact_rows))
    def payload_candidate(candidate):
        row = candidate.to_dict()
        if configuration.schema_version == "4.3.0":
            # Proposal probabilities are derived ordering diagnostics.  NumPy's
            # vector normalization can vary in its final binary digit between
            # processes; fixed decimal canonicalization keeps the scientific
            # digest stable without changing selection or evaluation.
            for name in ("proposal_probability_remove", "proposal_probability_add"):
                row[name] = round(float(row[name]), 14)
        return row

    def payload_step(step):
        row = {key: item for key, item in step.to_dict().items() if key != "wall_seconds"}
        if configuration.schema_version == "4.3.0":
            for name in ("proposal_probability_remove", "proposal_probability_add"):
                row[name] = round(float(row[name]), 14)
        row["background_fault_states"] = step.background_fault_states
        row["evaluated_candidates"] = [payload_candidate(candidate) for candidate in step.evaluated_candidates]
        return row

    payload = {
        "schema_version": configuration.schema_version, "scientific_protocol": configuration.scientific_protocol,
        "replication": replication,
        "benchmark": {"id": configuration.benchmark.id, "checkpoint_sha256": sha256_file(configuration.benchmark.checkpoint), "dataset_sha256": sha256_file(configuration.benchmark.dataset), "eligible_bits": len(identities), "quantization": configuration.benchmark.quantization},
        "partitions": {name: value.to_dict() for name, value in partitions.items()},
        "fault": {"probability": probability, "persistence": configuration.fault.persistence, "distribution": configuration.fault.distribution},
        **({"protection_budget_bits": len(initial)} if configuration.schema_version in {"4.2.0", "4.3.0"} else {}),
        "initial_mask": [bit.key() for bit in sorted(initial)], "candidate_pool": [bit.key() for bit in pool],
        "rankings": {name: {"digest": value.digest(), "criteria": value.criteria, "estimator": value.estimator} for name, value in orderings.items()},
        "search": [{
            "strategy": value.strategy, "best_mask": [bit.key() for bit in sorted(value.best_mask)],
            "accepted": value.accepted, "rejected": value.rejected,
            "best_estimated_risk": value.best_estimated_risk, "final_risk": value.final_risk,
            "steps": [payload_step(step) for step in value.steps],
        } for value in results],
        "final_fault_states": [[bit.key() for bit in sorted(value.bits)] for value in final_states],
        **({"conditional_final_fault_states": [[bit.key() for bit in sorted(value.bits)] for value in conditional_states],
            "conditional_comparisons": [{key:item for key,item in value.to_dict().items() if key != "wall_seconds"} for value in conditional_comparisons]} if controlled else {}),
        **({"topology_interactions": [{key:value for key,value in row.items() if key not in {"left_identity","right_identity","wall_seconds"}} for row in topology_rows]} if configuration.schema_version in {"4.2.0", "4.3.0"} else {}),
        "exact_optimum": None if exact_risk is None else {"risk": exact_risk, "mask": [bit.key() for bit in sorted(exact_mask)]},
    }
    if tracker is not None: tracker.update("Scientific payload", advance=1, **context)
    finish_phase("scientific_payload", "reporting_preparation", 1)
    whole_elapsed = perf_counter() - whole_started
    exclusive_total = sum(float(value["exclusive_wall_seconds"]) for value in timing_records)
    timing_records.append({"phase":"whole_process","category":"whole_process_elapsed","exclusive_wall_seconds":whole_elapsed,"operation_count":0,"exclusive_phase_sum_seconds":exclusive_total,"uninstrumented_seconds":max(0.0,whole_elapsed-exclusive_total),"peak_rss_mb":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0})
    outcome = SearchOutcome(configuration, payload, partitions, outputs, orderings, pool, tuple(polar_values), tuple(results), trials, convergence, validation, tuple(exact_rows), exact_mask, exact_risk, conditional_comparisons, probability, replication, (), tuple(topology_rows), len(initial), tuple(timing_records))
    return outcome


def execute_search(configuration: SearchConfiguration, *, _tracker=None) -> SearchOutcome:
    budgets = configuration.initial_mask.budgets_bits or (configuration.initial_mask.budget_bits,)
    members = tuple(
        _execute_replication(
            replace(configuration, initial_mask=replace(configuration.initial_mask, budget_bits=budget)),
            index, _tracker, probability=probability, condition_index=condition_index,
        )
        for budget in budgets
        for condition_index, probability in enumerate(configuration.fault_probabilities)
        for index in range(configuration.replications.count)
    )
    if len(members) == 1:
        return members[0]
    payload = {"schema_version": configuration.schema_version, "scientific_protocol": configuration.scientific_protocol,
               "replications": [{"name": value.replication_name, **({"logical_ber":value.logical_ber} if configuration.schema_version in {"4.1.0", "4.2.0", "4.3.0"} else {}), **({"protection_budget_bits": value.protection_budget_bits} if configuration.schema_version in {"4.2.0", "4.3.0"} else {}), "payload_sha256": value.scientific_payload_sha256} for value in members]}
    first = members[0]
    return replace(first, configuration=configuration, scientific_payload=payload, replications=members)
