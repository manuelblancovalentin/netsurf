"""Fixed-budget conditional protection search using existing ranking outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import itertools
import math
from time import perf_counter
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from .identities import BitIdentity
from .incremental import (
    DenseReluDenseIncrementalEvaluator,
    FaultState,
    evaluate_conditional_swap,
    metrics_from_logits,
    sample_conditional_backgrounds,
)
from .statistics import summarize
from .manifests import digest_object
from .conditional_screening import screen_conditional_candidates
from .factorized import FactorizedInteractionAggregate


@dataclass(frozen=True)
class RankingPrior:
    name: str
    identities: tuple[BitIdentity, ...]
    implementation_version: str

    @classmethod
    def from_scores(cls, name: str, scores: Mapping[BitIdentity, float], implementation_version: str):
        ordered = tuple(sorted(scores, key=lambda bit: (-float(scores[bit]), bit)))
        return cls(name, ordered, implementation_version)

    @classmethod
    def from_ordering(cls, ordering, implementation_version: str = "composable-ordering/v1"):
        return cls(ordering.name, tuple(ordering.identities), implementation_version)


@dataclass(frozen=True)
class PriorComponent:
    prior: RankingPrior | None
    weight: float


class ProposalPrior:
    """Validated rank-mixture proposal with explicit full-population exploration."""

    implementation_version = "rank-mixture-with-uniform-exploration/v1"

    def __init__(self, universe: Sequence[BitIdentity], components: Sequence[PriorComponent], *, uniform_exploration: float, rank_temperature: float = 0.15):
        if not 0 < uniform_exploration <= 1:
            raise ValueError("uniform_exploration must be in (0,1]")
        if rank_temperature <= 0:
            raise ValueError("rank_temperature must be positive")
        self.universe = tuple(universe)
        if len(set(self.universe)) != len(self.universe):
            raise ValueError("proposal universe contains duplicate identities")
        if any(component.weight < 0 for component in components):
            raise ValueError("ranking mixture weights must be nonnegative")
        if sum(component.weight for component in components) <= 0 and uniform_exploration < 1:
            raise ValueError("at least one positive ranking component is required")
        for component in components:
            if component.prior is not None and set(component.prior.identities) != set(self.universe):
                raise ValueError(f"ranking prior {component.prior.name} does not cover the proposal universe")
        self.components = tuple(components)
        self.uniform_exploration = float(uniform_exploration)
        self.rank_temperature = float(rank_temperature)
        self._ranks = {
            component.prior.name: {bit: rank for rank, bit in enumerate(component.prior.identities)}
            for component in components if component.prior is not None
        }

    def _guided(self, eligible: Sequence[BitIdentity], *, removal: bool) -> np.ndarray:
        values = np.zeros(len(eligible), dtype=np.float64)
        component_total = sum(component.weight for component in self.components)
        if component_total == 0:
            return np.full(len(eligible), 1.0 / len(eligible))
        scale = max(1.0, self.rank_temperature * len(self.universe))
        for component in self.components:
            if component.weight == 0 or component.prior is None:
                continue
            ranks = np.asarray([self._ranks[component.prior.name][bit] for bit in eligible], dtype=np.float64)
            if removal:
                desirability = np.exp(-(len(self.universe) - 1 - ranks) / scale)
            else:
                desirability = np.exp(-ranks / scale)
            desirability /= desirability.sum()
            values += component.weight / component_total * desirability
        return values / values.sum()

    def probabilities(self, eligible: Sequence[BitIdentity], *, removal: bool = False, adjustment: Mapping[BitIdentity, float] | None = None) -> np.ndarray:
        eligible = tuple(eligible)
        if not eligible:
            raise ValueError("proposal eligible set is empty")
        guided = self._guided(eligible, removal=removal)
        probabilities = (1.0 - self.uniform_exploration) * guided + self.uniform_exploration / len(eligible)
        if adjustment:
            multipliers = np.asarray([max(0.0, float(adjustment.get(bit, 1.0))) for bit in eligible])
            if multipliers.sum() > 0:
                probabilities *= multipliers
                probabilities /= probabilities.sum()
        return probabilities

    def sample(self, eligible: Sequence[BitIdentity], *, generator: np.random.Generator, removal: bool = False, adjustment: Mapping[BitIdentity, float] | None = None):
        eligible = tuple(eligible)
        probabilities = self.probabilities(eligible, removal=removal, adjustment=adjustment)
        index = int(generator.choice(len(eligible), p=probabilities))
        return eligible[index], float(probabilities[index])

    def provenance(self):
        return {
            "implementation_version": self.implementation_version,
            "uniform_exploration": self.uniform_exploration,
            "rank_temperature": self.rank_temperature,
            "components": [
                {"ranking": None if value.prior is None else value.prior.name,
                 "implementation_version": None if value.prior is None else value.prior.implementation_version,
                 "weight": value.weight}
                for value in self.components
            ],
        }


@dataclass(frozen=True)
class ProposalInfluence:
    candidate_count: int
    changed_positions: int
    changed_fraction: float
    top_k: int
    top_k_overlap: float
    spearman_rank_correlation: float
    total_variation: float
    jensen_shannon_divergence: float


def sparse_conditional_swap_score(
    removed: BitIdentity,
    added: BitIdentity,
    protected: Iterable[BitIdentity],
    universe: Sequence[BitIdentity],
    interactions: Mapping[tuple[BitIdentity, BitIdentity], float],
    *,
    probabilities: float | Mapping[BitIdentity, float],
    isolated_damage: Mapping[BitIdentity, float] | None = None,
) -> float:
    """Second-order mask-dependent risk change over recorded sparse edges.

    Missing edges are deliberately omitted and are not asserted to be zero.
    The direct ``removed, added`` edge cancels from the swap identity.
    """

    protected = frozenset(protected)
    if removed not in protected or added in protected:
        raise ValueError("swap score requires protected removed and faultable added bits")
    probability = (lambda bit: float(probabilities[bit])) if isinstance(probabilities, Mapping) else (lambda bit: float(probabilities))
    damage = isolated_damage or {}
    result = probability(removed) * float(damage.get(removed, 0.0)) - probability(added) * float(damage.get(added, 0.0))
    left_sum = right_sum = 0.0
    for (first, second), interaction in interactions.items():
        if first == removed:
            other = second
            if other not in protected and other != added:
                left_sum += probability(other) * float(interaction)
        elif second == removed:
            other = first
            if other not in protected and other != added:
                left_sum += probability(other) * float(interaction)
        if first == added:
            other = second
            if other not in protected and other != removed:
                right_sum += probability(other) * float(interaction)
        elif second == added:
            other = first
            if other not in protected and other != removed:
                right_sum += probability(other) * float(interaction)
    result += probability(removed) * left_sum - probability(added) * right_sum
    return float(result)


def proposal_influence(before: np.ndarray, after: np.ndarray, *, top_k: int = 10) -> ProposalInfluence:
    """Compare proposal distributions before and after interaction guidance."""

    before = np.asarray(before, dtype=np.float64)
    after = np.asarray(after, dtype=np.float64)
    if before.shape != after.shape or before.ndim != 1 or len(before) == 0:
        raise ValueError("proposal distributions must be nonempty matching vectors")
    if np.any(before < 0) or np.any(after < 0) or not np.isclose(before.sum(), 1) or not np.isclose(after.sum(), 1):
        raise ValueError("proposal distributions must be normalized and nonnegative")
    order_before = np.argsort(-before, kind="stable")
    order_after = np.argsort(-after, kind="stable")
    ranks_before = np.empty(len(before), dtype=np.int64); ranks_before[order_before] = np.arange(len(before))
    ranks_after = np.empty(len(after), dtype=np.int64); ranks_after[order_after] = np.arange(len(after))
    changed = int(np.count_nonzero(ranks_before != ranks_after))
    k = min(max(1, int(top_k)), len(before))
    overlap = len(set(order_before[:k]) & set(order_after[:k])) / k
    if len(before) == 1:
        correlation = 1.0
    else:
        correlation = float(np.corrcoef(ranks_before, ranks_after)[0, 1])
    midpoint = 0.5 * (before + after)
    def kl(left, right):
        valid = left > 0
        return float(np.sum(left[valid] * np.log(left[valid] / right[valid])))
    return ProposalInfluence(len(before), changed, changed / len(before), k, overlap, correlation,
                             float(0.5 * np.sum(np.abs(before - after))),
                             0.5 * kl(before, midpoint) + 0.5 * kl(after, midpoint))

@dataclass(frozen=True)
class ProposalEvaluation:
    candidate_index: int
    removed: BitIdentity
    added: BitIdentity
    proposal_probability_remove: float
    proposal_probability_add: float
    risk_difference: float
    risk_ci_low: float
    risk_ci_high: float
    exact_evaluations: int
    work_proxy_macs: int
    gate_changes: int

    def to_dict(self):
        row = asdict(self)
        row["removed"] = self.removed.key(); row["added"] = self.added.key()
        return row


@dataclass(frozen=True)
class SearchStep:
    iteration: int
    strategy: str
    removed: BitIdentity | None
    added: BitIdentity | None
    proposal_probability_remove: float | None
    proposal_probability_add: float | None
    estimated_risk_difference: float
    estimated_risk_ci_low: float
    estimated_risk_ci_high: float
    accepted: bool
    acceptance_probability: float
    current_estimated_risk: float
    best_estimated_risk: float
    exact_evaluations: int
    work_proxy_macs: int
    gate_changes: int
    background_evaluations: int
    incremental_evaluations: int
    full_evaluations: int
    affected_layers: int
    reused_layers: int
    cache_hits: int
    cache_misses: int
    fallback_count: int
    background_schedule_sha256: str
    background_fault_states: tuple[tuple[str, ...], ...]
    evaluated_candidates: tuple[ProposalEvaluation, ...]
    wall_seconds: float

    def to_dict(self):
        row = asdict(self)
        row["removed"] = None if self.removed is None else self.removed.key()
        row["added"] = None if self.added is None else self.added.key()
        row.pop("evaluated_candidates", None)
        row.pop("background_fault_states", None)
        return row


@dataclass(frozen=True)
class SearchResult:
    strategy: str
    initial_mask: frozenset[BitIdentity]
    final_mask: frozenset[BitIdentity]
    best_mask: frozenset[BitIdentity]
    initial_risk: Mapping[str, object]
    final_risk: Mapping[str, object]
    best_estimated_risk: float
    steps: tuple[SearchStep, ...]
    accepted: int
    rejected: int
    proposals_generated: int
    proposals_evaluated: int
    exact_evaluations: int
    work_proxy_macs: int
    wall_seconds: float
    proposal_provenance: Mapping[str, object]
    proposal_influence: tuple[Mapping[str, object], ...] = ()
    screening_rows: tuple[Mapping[str, object], ...] = ()
    factorized_rows: tuple[Mapping[str, object], ...] = ()
    screening_context_rows: tuple[Mapping[str, object], ...] = ()


def estimate_mask_risk(
    evaluator: DenseReluDenseIncrementalEvaluator, inputs: np.ndarray, labels: np.ndarray,
    universe: Sequence[BitIdentity], protected: Iterable[BitIdentity], *, probability: float,
    contexts: int, seed: int, confidence: float = 0.95,
    full_fault_states: Sequence[FaultState] | None = None,
):
    protected = frozenset(protected)
    if full_fault_states is None:
        states = sample_conditional_backgrounds(
            universe, protected, (), probability=probability, count=contexts, seed=seed
        )
    else:
        states = tuple(FaultState(state.bits - protected) for state in full_fault_states)
    losses = [
        metrics_from_logits(evaluator.full_forward(inputs, state).logits, labels).categorical_crossentropy
        for state in states
    ]
    return summarize(losses, confidence=confidence).to_dict(), tuple(states)


def sample_full_fault_states(
    universe: Sequence[BitIdentity], *, probability: float, count: int, seed: int
) -> tuple[FaultState, ...]:
    """Draw mask-independent fault states for matched held-out comparisons."""

    return sample_conditional_backgrounds(
        universe, (), (), probability=probability, count=count, seed=seed
    )


def _polar_adjustment(
    candidates: Sequence[BitIdentity], removed: BitIdentity, interactions: Mapping[tuple[BitIdentity, BitIdentity], float] | None,
    strength: float,
) -> dict[BitIdentity, float] | None:
    if not interactions or strength == 0:
        return None
    values = []
    for candidate in candidates:
        value = interactions.get((removed, candidate), interactions.get((candidate, removed), 0.0))
        values.append(float(value))
    scale = max(np.std(values), 1e-12)
    # This is proposal guidance only: large positive finite interactions are
    # sampled more frequently for exact verification, not assumed beneficial.
    return {candidate: float(np.exp(np.clip(strength * value / scale, -20, 20))) for candidate, value in zip(candidates, values)}


def _mask_dependent_adjustment(
    candidates: Sequence[BitIdentity], removed: BitIdentity, current: frozenset[BitIdentity],
    universe: Sequence[BitIdentity], interactions: Mapping[tuple[BitIdentity, BitIdentity], float] | None,
    probability: float, strength: float,
) -> dict[BitIdentity, float] | None:
    if not interactions or strength == 0:
        return None
    adjacency: dict[BitIdentity, list[tuple[BitIdentity, float]]] = {}
    for (left, right), value in interactions.items():
        adjacency.setdefault(left, []).append((right, float(value)))
        adjacency.setdefault(right, []).append((left, float(value)))
    removed_edges = {other: value for other, value in adjacency.get(removed, ()) if other not in current}
    scores = []
    for candidate in candidates:
        left_sum = sum(value for other, value in removed_edges.items() if other != candidate)
        right_sum = sum(value for other, value in adjacency.get(candidate, ()) if other not in current and other != removed)
        scores.append(probability * probability * (left_sum - right_sum))
    scores = np.asarray(scores, dtype=np.float64)
    scale = max(float(np.std(scores)), 1e-12)
    # Lower predicted risk changes are more desirable additions.
    return {candidate: float(np.exp(np.clip(-strength * score / scale, -20, 20))) for candidate, score in zip(candidates, scores)}


def conditional_search(
    evaluator: DenseReluDenseIncrementalEvaluator,
    inputs: np.ndarray,
    labels: np.ndarray,
    universe: Sequence[BitIdentity],
    initial_mask: Iterable[BitIdentity],
    *,
    strategy: str,
    proposal: ProposalPrior,
    probability: float,
    iterations: int,
    background_contexts: int,
    seed: int,
    final_contexts: int,
    final_seed: int,
    confidence: float = 0.95,
    candidate_pool: Sequence[BitIdentity] | None = None,
    greedy_add_candidates: int = 8,
    greedy_remove_candidates: int = 4,
    initial_temperature: float = 0.01,
    cooling_rate: float = 0.95,
    polar_interactions: Mapping[tuple[BitIdentity, BitIdentity], float] | None = None,
    polar_strength: float = 1.0,
    evaluator_mode: str = "incremental",
    final_fault_states: Sequence[FaultState] | None = None,
    final_inputs: np.ndarray | None = None,
    final_labels: np.ndarray | None = None,
    progress: Callable[[Mapping[str, object]], None] | None = None,
    defer_final_evaluation: bool = False,
    background_seed: int | None = None,
    interaction_guidance: str = "direct_pair_v1",
) -> SearchResult:
    """Run a fixed-cardinality local search with exact conditional decisions."""
    supported = {"static", "conditional_greedy", "uniform_swap", "static_guided_swap", "polar_guided_swap", "simulated_annealing"}
    if strategy not in supported:
        raise ValueError(f"unsupported search strategy: {strategy}")
    if interaction_guidance not in {"direct_pair_v1", "mask_dependent_sparse_v1"}:
        raise ValueError(f"unsupported interaction guidance: {interaction_guidance}")
    if iterations < 0 or background_contexts < 1 or final_contexts < 1:
        raise ValueError("invalid search or context budget")
    universe = tuple(universe)
    universe_set = set(universe)
    pool = tuple(candidate_pool or universe)
    if not set(pool) <= universe_set:
        raise ValueError("candidate pool must be a subset of the eligible universe")
    current = frozenset(initial_mask)
    if not current <= universe_set or not current:
        raise ValueError("initial mask must be a nonempty subset of the universe")
    budget = len(current)
    generator = np.random.default_rng(int(seed))
    context_seed = int(seed if background_seed is None else background_seed)
    started = perf_counter()
    initial_risk, _ = estimate_mask_risk(
        evaluator, inputs, labels, universe, current, probability=probability,
        contexts=background_contexts, seed=context_seed + 1_000_000, confidence=confidence,
    )
    current_risk = float(initial_risk["mean"])
    best_risk, best_mask = current_risk, current
    steps: list[SearchStep] = []
    influence_rows: list[Mapping[str, object]] = []
    accepted = rejected = proposals_generated = proposals_evaluated = exact_evaluations = work = 0
    if strategy == "static":
        iterations = 0
    for iteration in range(iterations):
        iteration_started = perf_counter()
        protected = tuple(sorted(current))
        additions = tuple(bit for bit in pool if bit not in current)
        if not additions:
            break
        candidates = []
        if strategy == "conditional_greedy":
            remove_probabilities = proposal.probabilities(protected, removal=True)
            add_probabilities = proposal.probabilities(additions, removal=False)
            remove_order = np.argsort(-remove_probabilities)[:min(greedy_remove_candidates, len(protected))]
            add_order = np.argsort(-add_probabilities)[:min(greedy_add_candidates, len(additions))]
            for remove_index in remove_order:
                for add_index in add_order:
                    candidates.append((protected[int(remove_index)], additions[int(add_index)], float(remove_probabilities[int(remove_index)]), float(add_probabilities[int(add_index)])))
        else:
            if strategy == "uniform_swap":
                removed = protected[int(generator.integers(len(protected)))]
                added = additions[int(generator.integers(len(additions)))]
                q_remove, q_add = 1 / len(protected), 1 / len(additions)
            else:
                removed, q_remove = proposal.sample(protected, generator=generator, removal=True)
                adjustment = None
                if strategy == "polar_guided_swap":
                    adjustment = (_mask_dependent_adjustment(additions, removed, current, universe, polar_interactions, probability, polar_strength)
                                  if interaction_guidance == "mask_dependent_sparse_v1"
                                  else _polar_adjustment(additions, removed, polar_interactions, polar_strength))
                    before = proposal.probabilities(additions)
                    after = proposal.probabilities(additions, adjustment=adjustment)
                    influence = proposal_influence(before, after, top_k=min(10, len(additions)))
                    influence_rows.append({
                        "iteration": iteration + 1, "removed": removed.key(),
                        "interaction_guidance": interaction_guidance,
                        "polarity_strength": polar_strength, **asdict(influence),
                        "materially_influenced": influence.total_variation > 1e-12,
                    })
                added, q_add = proposal.sample(additions, generator=generator, adjustment=adjustment)
            candidates.append((removed, added, q_remove, q_add))
        proposals_generated += len(candidates)
        # One full Bernoulli background schedule is shared across every
        # alternative evaluated at this iteration. Candidate-specific
        # backgrounds are exact projections that remove the two swapped bits.
        base_backgrounds = sample_conditional_backgrounds(
            universe, current, (), probability=probability,
            count=background_contexts, seed=context_seed + iteration * 100_003,
        )
        background_rows = tuple(tuple(bit.key() for bit in sorted(state.bits)) for state in base_backgrounds)
        background_digest = digest_object(background_rows)
        evaluated = []
        evaluation_records = []
        shared_cache_store = {} if evaluator_mode == "incremental" else None
        for candidate_index, (removed, added, q_remove, q_add) in enumerate(candidates):
            backgrounds = tuple(FaultState(state.bits - {removed, added}) for state in base_backgrounds)
            estimate = evaluate_conditional_swap(
                evaluator, inputs, labels, backgrounds, removed, added,
                probability_removed=probability, probability_added=probability,
                confidence=confidence, mode=evaluator_mode, cache_store=shared_cache_store,
            )
            evaluated.append((estimate.risk_difference["mean"], removed, added, q_remove, q_add, estimate))
            interval = estimate.risk_difference["confidence_interval"]
            evaluation_records.append(ProposalEvaluation(
                candidate_index, removed, added, q_remove, q_add,
                float(estimate.risk_difference["mean"]), float(interval[0]), float(interval[1]),
                estimate.background_evaluations + estimate.incremental_evaluations + estimate.full_evaluations,
                estimate.work_proxy_macs, estimate.gate_changes,
            ))
        proposals_evaluated += len(evaluated)
        chosen = min(evaluated, key=lambda value: (value[0], value[1], value[2])) if strategy == "conditional_greedy" else evaluated[0]
        difference, removed, added, q_remove, q_add, estimate = chosen
        iteration_exact = sum(value[5].background_evaluations + value[5].incremental_evaluations + value[5].full_evaluations for value in evaluated)
        iteration_work = sum(value[5].work_proxy_macs for value in evaluated)
        iteration_gates = sum(value[5].gate_changes for value in evaluated)
        iteration_backgrounds = sum(value[5].background_evaluations for value in evaluated)
        iteration_incremental = sum(value[5].incremental_evaluations for value in evaluated)
        iteration_full = sum(value[5].full_evaluations for value in evaluated)
        iteration_affected = sum(value[5].affected_layers for value in evaluated)
        iteration_reused = sum(value[5].reused_layers for value in evaluated)
        iteration_hits = sum(value[5].cache_hits for value in evaluated)
        iteration_misses = sum(value[5].cache_misses for value in evaluated)
        iteration_fallbacks = sum(value[5].fallback_count for value in evaluated)
        exact_evaluations += iteration_exact
        work += iteration_work
        if strategy == "simulated_annealing":
            temperature = initial_temperature * cooling_rate ** iteration
            acceptance_probability = 1.0 if difference <= 0 else math.exp(-difference / max(temperature, 1e-15))
            accept = bool(generator.random() < acceptance_probability)
        else:
            acceptance_probability = 1.0 if difference < 0 else 0.0
            accept = difference < 0
        if accept:
            current = frozenset((set(current) - {removed}) | {added})
            if len(current) != budget:
                raise RuntimeError("search violated the fixed protection budget")
            current_risk += float(difference)
            accepted += 1
            if current_risk < best_risk:
                best_risk, best_mask = current_risk, current
        else:
            rejected += 1
        interval = estimate.risk_difference["confidence_interval"]
        step = SearchStep(
            iteration + 1, strategy, removed, added, q_remove, q_add, float(difference),
            float(interval[0]), float(interval[1]), accept, acceptance_probability,
            current_risk, best_risk, iteration_exact,
            iteration_work, iteration_gates,
            iteration_backgrounds, iteration_incremental, iteration_full,
            iteration_affected, iteration_reused, iteration_hits, iteration_misses,
            iteration_fallbacks, background_digest, background_rows,
            tuple(evaluation_records), perf_counter() - iteration_started,
        )
        steps.append(step)
        if progress:
            progress({
                "iteration": iteration + 1, "iterations": iterations, "strategy": strategy,
                "protected_bits": len(current), "estimated_risk": current_risk,
                "best_estimated_risk": best_risk, "accepted": accepted, "rejected": rejected,
                "exact_evaluations": exact_evaluations,
            })
    if defer_final_evaluation:
        final_risk = {}
    else:
        evaluation_inputs = inputs if final_inputs is None else final_inputs
        evaluation_labels = labels if final_labels is None else final_labels
        final_risk, _ = estimate_mask_risk(
            evaluator, evaluation_inputs, evaluation_labels, universe, best_mask, probability=probability,
            contexts=final_contexts, seed=final_seed, confidence=confidence,
            full_fault_states=final_fault_states,
        )
    return SearchResult(
        strategy, frozenset(initial_mask), current, best_mask, initial_risk, final_risk,
        best_risk, tuple(steps), accepted, rejected, proposals_generated,
        proposals_evaluated, exact_evaluations, work, perf_counter() - started,
        proposal.provenance(), tuple(influence_rows),
    )


def conditional_screening_search(
    evaluator: DenseReluDenseIncrementalEvaluator,
    inputs: np.ndarray,
    labels: np.ndarray,
    universe: Sequence[BitIdentity],
    initial_mask: Iterable[BitIdentity],
    *,
    strategy: str,
    proposal: ProposalPrior,
    probability: float,
    iterations: int,
    background_contexts: int,
    seed: int,
    candidate_pool: Sequence[BitIdentity],
    removal_shortlist: int,
    addition_shortlist: int,
    verification_shortlist: int,
    confidence: float,
    final_contexts: int,
    final_seed: int,
    defer_final_evaluation: bool = False,
    aggregate: FactorizedInteractionAggregate | None = None,
    dynamic_aggregate: bool = True,
    interaction_strength: float = 1.0,
    polar_interactions: Mapping[tuple[BitIdentity, BitIdentity], float] | None = None,
    polar_correction_budget: int = 0,
    response_cache_limit: int = 2048,
    refresh_frequency: int = 1,
    progress: Callable[[Mapping[str, object]], None] | None = None,
) -> SearchResult:
    """Reuse exact candidate marginals, then verify selected swaps exactly.

    ``marginal_screening`` verifies only the best ``C-D`` shortlist entry.
    Corrected/hybrid variants verify several shortlisted swaps with the exact
    three-state estimator; pair/polar values affect screening order only.
    """

    supported = {"marginal_screening", "corrected_screening", "factorized_screening", "hybrid_screening"}
    if strategy not in supported:
        raise ValueError(f"unsupported screening strategy: {strategy}")
    universe = tuple(universe); pool = tuple(candidate_pool); current = frozenset(initial_mask)
    if not current or not current <= set(universe) or not set(pool) <= set(universe):
        raise ValueError("invalid screening universe, pool, or initial mask")
    started = perf_counter(); context_seed = int(seed); budget = len(current)
    initial_risk, _ = estimate_mask_risk(evaluator, inputs, labels, universe, current,
                                         probability=probability, contexts=background_contexts,
                                         seed=context_seed + 1_000_000, confidence=confidence)
    current_risk = best_risk = float(initial_risk["mean"]); best_mask = current
    steps=[]; screening_rows=[]; screening_context_rows=[]; factorized_rows=[]; influence_rows=[]
    accepted = rejected = generated = evaluated_count = exact_count = work = 0
    cached_screening = cached_fields = cached_mask = None; cached_iteration = -1
    for iteration in range(iterations):
        iteration_started=perf_counter(); protected=tuple(sorted(current)); additions=tuple(bit for bit in pool if bit not in current)
        if not additions: break
        before_remove=proposal.probabilities(protected, removal=True); before_add=proposal.probabilities(additions)
        after_remove=before_remove.copy(); after_add=before_add.copy()
        if aggregate is not None:
            remove_terms=np.asarray([probability*aggregate.query(bit) for bit in protected]); add_terms=np.asarray([probability*aggregate.query(bit) for bit in additions])
            remove_scale=max(float(np.std(remove_terms)),1e-12); add_scale=max(float(np.std(add_terms)),1e-12)
            after_remove *= np.exp(np.clip(-interaction_strength*remove_terms/remove_scale,-20,20)); after_remove /= after_remove.sum()
            after_add *= np.exp(np.clip(interaction_strength*add_terms/add_scale,-20,20)); after_add /= after_add.sum()
            for population_name,before,after in (("removal",before_remove,after_remove),("addition",before_add,after_add)):
                influence=proposal_influence(before,after,top_k=min(10,len(before)))
                influence_rows.append({"iteration":iteration+1,"removed":"","population":population_name,"interaction_guidance":"full_population_factorized_v1","polarity_strength":interaction_strength,**asdict(influence),"materially_influenced":influence.total_variation>1e-12})
        remove_indices=np.argsort(-after_remove,kind="stable")[:min(removal_shortlist,len(protected))]
        add_indices=np.argsort(-after_add,kind="stable")[:min(addition_shortlist,len(additions))]
        remove_bits=tuple(protected[int(index)] for index in remove_indices); add_bits=tuple(additions[int(index)] for index in add_indices)
        refresh = (cached_screening is None or cached_mask != current or
                   iteration - cached_iteration >= refresh_frequency or
                   cached_screening.candidate_order != remove_bits + add_bits)
        if refresh:
            base_fields=sample_full_fault_states(universe,probability=probability,count=background_contexts,seed=context_seed+iteration*100_003)
            screened=screen_conditional_candidates(evaluator,inputs,labels,base_fields,current,remove_bits,add_bits,
                                                   probabilities=probability,confidence=confidence,response_cache_limit=response_cache_limit)
            cached_screening, cached_fields, cached_mask, cached_iteration = screened, base_fields, current, iteration
            for value in screened.removals+screened.additions:
                screening_rows.append({"iteration":iteration+1,"identity":value.identity.key(),"status":value.status,"weighted_mean":value.weighted_mean,"conditional_mean_loss_difference":value.conditional_mean_loss_difference,"standard_error":value.standard_error,"ci_low":value.confidence_interval[0],"ci_high":value.confidence_interval[1],"contexts":value.context_count,"examples":value.example_count,"response_count":value.response_count,"work_proxy_macs":value.work_proxy_macs,"exclusive_wall_seconds":value.exclusive_wall_seconds,"background_cache_hits":screened.background_cache_hits,"background_cache_misses":screened.background_cache_misses})
            for context_index, row in enumerate(screened.paired_weighted_contributions):
                for bit, contribution in zip(screened.candidate_order, row):
                    screening_context_rows.append({"iteration":iteration+1,"context":context_index,"identity":bit.key(),"status":"protected" if bit in current else "unprotected","weighted_loss_contribution":float(contribution)})
        else:
            screened, base_fields = cached_screening, cached_fields
        ranked=[]
        for removed in remove_bits:
            for added in add_bits:
                score=screened.marginal_swap_score(removed,added)
                ranked.append((score,removed,added,0.0))
        ranked.sort(key=lambda value:(value[0],value[1],value[2]))
        if polar_correction_budget and polar_interactions:
            corrected=[]
            for index,(score,removed,added,_) in enumerate(ranked):
                polar=0.0 if index >= polar_correction_budget else float(polar_interactions.get((removed,added),polar_interactions.get((added,removed),0.0)))
                corrected.append((score-probability*probability*polar,removed,added,polar))
            ranked=sorted(corrected,key=lambda value:(value[0],value[1],value[2]))
        generated += len(ranked)
        verify_count=1 if strategy=="marginal_screening" else min(verification_shortlist,len(ranked))
        verified=[]; records=[]; cache_store={}
        for candidate_index,(screen_score,removed,added,polar) in enumerate(ranked[:verify_count]):
            backgrounds=tuple(FaultState(state.bits-current-{removed,added}) for state in base_fields)
            estimate=evaluate_conditional_swap(evaluator,inputs,labels,backgrounds,removed,added,
                                               probability_removed=probability,probability_added=probability,
                                               confidence=confidence,cache_store=cache_store)
            verified.append((float(estimate.risk_difference["mean"]),removed,added,estimate,screen_score,polar))
            interval=estimate.risk_difference["confidence_interval"]
            records.append(ProposalEvaluation(candidate_index,removed,added,float(after_remove[protected.index(removed)]),float(after_add[additions.index(added)]),float(estimate.risk_difference["mean"]),float(interval[0]),float(interval[1]),estimate.background_evaluations+estimate.incremental_evaluations+estimate.full_evaluations,estimate.work_proxy_macs,estimate.gate_changes))
        evaluated_count += len(verified); chosen=min(verified,key=lambda value:(value[0],value[1],value[2])); difference,removed,added,estimate,screen_score,polar=chosen
        exact=sum(value[3].background_evaluations+value[3].incremental_evaluations+value[3].full_evaluations for value in verified); exact_count+=exact
        screening_work=screened.work_proxy_macs if refresh else 0
        iteration_work=sum(value[3].work_proxy_macs for value in verified)+screening_work; work+=iteration_work
        removed_aggregate = added_aggregate = removed_explicit = added_explicit = None
        if aggregate is not None:
            removed_aggregate = probability * aggregate.query(removed)
            added_aggregate = probability * aggregate.query(added)
            if len(aggregate.population) <= 2048:
                removed_explicit = probability * aggregate.explicit(removed)
                added_explicit = probability * aggregate.explicit(added)
        accept=difference<0; accepted+=int(accept); rejected+=int(not accept)
        if accept:
            current=frozenset((set(current)-{removed})|{added}); current_risk += difference
            if aggregate is not None and dynamic_aggregate: aggregate.swap(removed,added)
            if current_risk<best_risk: best_risk,best_mask=current_risk,current
        interval=estimate.risk_difference["confidence_interval"]
        rows=tuple(tuple(bit.key() for bit in sorted(state.bits)) for state in base_fields)
        step=SearchStep(iteration+1,strategy,removed,added,float(after_remove[protected.index(removed)]),float(after_add[additions.index(added)]),difference,float(interval[0]),float(interval[1]),accept,1.0 if accept else 0.0,current_risk,best_risk,exact,iteration_work,sum(value[3].gate_changes for value in verified),sum(value[3].background_evaluations for value in verified),sum(value[3].incremental_evaluations for value in verified),sum(value[3].full_evaluations for value in verified),sum(value[3].affected_layers for value in verified),sum(value[3].reused_layers for value in verified),sum(value[3].cache_hits for value in verified),sum(value[3].cache_misses for value in verified),sum(value[3].fallback_count for value in verified),digest_object(rows),rows,tuple(records),perf_counter()-iteration_started)
        steps.append(step)
        screening_seconds=screened.exclusive_wall_seconds if refresh else 0.0
        verification_seconds=sum(value[3].wall_seconds for value in verified)
        factorized_rows.append({"iteration":iteration+1,"strategy":strategy,"screening_score":screen_score,"polar_pair_correction":polar,"exact_risk_difference":difference,"removed_aggregate_score":removed_aggregate,"added_aggregate_score":added_aggregate,"removed_explicit_score":removed_explicit,"added_explicit_score":added_explicit,"aggregate_reconstruction_error":None if aggregate is None else aggregate.reconstruction_error(),"aggregate_population_coverage":None if aggregate is None else aggregate.population_coverage,"screening_refreshed":refresh,"screened_candidates":len(screened.candidate_order),"verified_swaps":len(verified),"conditional_screening_seconds":screening_seconds,"exact_verification_seconds":verification_seconds,"proposal_and_accounting_seconds":max(0.0,step.wall_seconds-screening_seconds-verification_seconds),"accepted":accept})
        if progress: progress({"iteration":iteration+1,"iterations":iterations,"strategy":strategy,"protected_bits":len(current),"estimated_risk":current_risk,"best_estimated_risk":best_risk,"accepted":accepted,"rejected":rejected,"exact_evaluations":exact_count,"screened_candidates":len(screened.candidate_order),"verified_swaps":len(verified)})
    final_risk={} if defer_final_evaluation else estimate_mask_risk(evaluator,inputs,labels,universe,best_mask,probability=probability,contexts=final_contexts,seed=final_seed,confidence=confidence)[0]
    provenance={**proposal.provenance(),"screening":"topology-conditional-marginal-screening/v1","corrected":strategy!="marginal_screening","factorized":aggregate is not None,"dynamic_aggregate":dynamic_aggregate,"polar_correction_budget":polar_correction_budget}
    return SearchResult(strategy,frozenset(initial_mask),current,best_mask,initial_risk,final_risk,best_risk,tuple(steps),accepted,rejected,generated,evaluated_count,exact_count,work,perf_counter()-started,provenance,tuple(influence_rows),tuple(screening_rows),tuple(factorized_rows),tuple(screening_context_rows))


def exhaustive_optimum(
    evaluator: DenseReluDenseIncrementalEvaluator, inputs: np.ndarray, labels: np.ndarray,
    candidate_universe: Sequence[BitIdentity], *, budget: int, probability: float,
) -> tuple[frozenset[BitIdentity], float, tuple[Mapping[str, object], ...]]:
    """Exact fault-distribution and mask enumeration for deliberately tiny pools."""
    candidates = tuple(candidate_universe)
    if len(candidates) > 20:
        raise ValueError("exact enumeration is limited to at most 20 candidate bits")
    if not 0 <= budget <= len(candidates):
        raise ValueError("invalid exact-enumeration budget")
    fault_states = []
    for count in range(len(candidates) + 1):
        for selected in itertools.combinations(candidates, count):
            weight = probability ** count * (1 - probability) ** (len(candidates) - count)
            fault_states.append((frozenset(selected), weight))
    rows = []
    best_mask, best_risk = None, float("inf")
    for mask in itertools.combinations(candidates, budget):
        protected = frozenset(mask)
        risk = 0.0
        for faults, weight in fault_states:
            active = FaultState(faults - protected)
            risk += weight * metrics_from_logits(evaluator.full_forward(inputs, active).logits, labels).categorical_crossentropy
        rows.append({"mask": tuple(bit.key() for bit in sorted(protected)), "risk": risk})
        if risk < best_risk:
            best_mask, best_risk = protected, risk
    return best_mask or frozenset(), float(best_risk), tuple(rows)
