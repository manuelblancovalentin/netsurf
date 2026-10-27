"""Reusable conditional bit screening for fixed protection masks.

The marginal estimates in this module are Monte Carlo quantities.  They are
kept separate from the exact three-state swap estimator: ``C_a-D_b`` omits the
pair correction and is therefore guidance, not an exact swap result.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from time import perf_counter
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from .identities import BitIdentity
from .incremental import DenseReluDenseIncrementalEvaluator, FaultState, metrics_from_logits
from .statistics import summarize
from .topology import DenseReluDenseTopologyEvaluator


def bit_probability(bit: BitIdentity, probabilities: float | Mapping[BitIdentity, float]) -> float:
    value = float(probabilities if isinstance(probabilities, (int, float)) else probabilities[bit])
    if not 0 <= value <= 1:
        raise ValueError("bit probabilities must be in [0,1]")
    return value


@dataclass(frozen=True)
class ExactSwapDecomposition:
    removal_cost: float
    addition_benefit: float
    pair_interaction: float
    marginal_swap_difference: float
    corrected_swap_difference: float
    direct_swap_difference: float


def enumerate_exact_swap_decomposition(
    universe: Sequence[BitIdentity],
    protected: Iterable[BitIdentity],
    removed: BitIdentity,
    added: BitIdentity,
    probabilities: float | Mapping[BitIdentity, float],
    loss: Callable[[frozenset[BitIdentity]], float],
) -> ExactSwapDecomposition:
    """Enumerate the exact conditional decomposition over a small universe."""

    universe = tuple(universe); protected = frozenset(protected)
    if removed not in protected or added in protected or removed == added:
        raise ValueError("decomposition requires a protected removal and unprotected addition")
    if not protected <= frozenset(universe):
        raise ValueError("protected set is outside the eligible universe")
    others = tuple(bit for bit in universe if bit not in protected and bit != added)
    pa, pb = bit_probability(removed, probabilities), bit_probability(added, probabilities)
    cost = benefit = interaction = direct = 0.0
    for flags in product((False, True), repeat=len(others)):
        state = frozenset(bit for bit, flag in zip(others, flags) if flag)
        weight = 1.0
        for bit, flag in zip(others, flags):
            p = bit_probability(bit, probabilities)
            weight *= p if flag else 1.0 - p
        lg = loss(state); lga = loss(state | {removed}); lgb = loss(state | {added}); lgab = loss(state | {removed, added})
        h = lgab - lga - lgb + lg
        # C_a integrates over b's Bernoulli event; D_b is conditioned on b=0.
        cost += weight * pa * ((1.0 - pb) * (lga - lg) + pb * (lgab - lgb))
        benefit += weight * pb * (lgb - lg)
        interaction += weight * h
        direct += weight * (pa * (lga - lg) - pb * (lgb - lg))
    marginal = cost - benefit
    corrected = marginal - pa * pb * interaction
    return ExactSwapDecomposition(cost, benefit, interaction, marginal, corrected, direct)


@dataclass(frozen=True)
class CandidateMarginal:
    identity: BitIdentity
    status: str
    weighted_mean: float
    conditional_mean_loss_difference: float
    standard_error: float
    confidence_interval: tuple[float, float]
    context_count: int
    example_count: int
    response_count: int
    work_proxy_macs: int
    exclusive_wall_seconds: float


@dataclass(frozen=True)
class ConditionalScreeningResult:
    protected: frozenset[BitIdentity]
    removals: tuple[CandidateMarginal, ...]
    additions: tuple[CandidateMarginal, ...]
    paired_weighted_contributions: np.ndarray
    candidate_order: tuple[BitIdentity, ...]
    background_cache_hits: int
    background_cache_misses: int
    fallback_count: int
    work_proxy_macs: int
    exclusive_wall_seconds: float
    implementation_version: str = "topology-conditional-marginal-screening/v1"

    def marginal_swap_score(self, removed: BitIdentity, added: BitIdentity) -> float:
        values = {value.identity: value.weighted_mean for value in self.removals + self.additions}
        return float(values[removed] - values[added])


def screen_conditional_candidates(
    evaluator: DenseReluDenseIncrementalEvaluator,
    inputs: np.ndarray,
    labels: np.ndarray,
    base_fault_fields: Sequence[FaultState],
    protected: Iterable[BitIdentity],
    removals: Sequence[BitIdentity],
    additions: Sequence[BitIdentity],
    *,
    probabilities: float | Mapping[BitIdentity, float],
    confidence: float = 0.95,
    response_cache_limit: int = 2048,
) -> ConditionalScreeningResult:
    """Screen candidate marginals with common random fields and exact responses.

    A context's unprotected fault field is shared.  Addition candidate ``b``
    uses its exact conditional projection ``F\\{b}``; equal projected states
    share one topology/background cache.
    """

    started = perf_counter(); protected = frozenset(protected)
    removals, additions = tuple(removals), tuple(additions)
    if not base_fault_fields or not removals or not additions:
        raise ValueError("screening requires contexts and nonempty removal/addition shortlists")
    if any(bit not in protected for bit in removals) or any(bit in protected for bit in additions):
        raise ValueError("candidate status is inconsistent with the protection mask")
    candidate_order = removals + additions
    contributions = np.empty((len(base_fault_fields), len(candidate_order)), dtype=np.float64)
    raw = np.empty_like(contributions)
    topology_cache: dict[FaultState, DenseReluDenseTopologyEvaluator] = {}
    cache_hits = cache_misses = fallbacks = work = responses = 0

    def topology(state: FaultState) -> DenseReluDenseTopologyEvaluator:
        nonlocal cache_hits, cache_misses
        if state in topology_cache:
            cache_hits += 1
            return topology_cache[state]
        cache_misses += 1
        value = DenseReluDenseTopologyEvaluator(
            evaluator, inputs, state, response_cache_limit=response_cache_limit,
        )
        topology_cache[state] = value
        return value

    for row, field in enumerate(base_fault_fields):
        faultable = FaultState(field.bits - protected)
        base_topology = topology(faultable)
        for column, bit in enumerate(removals):
            response = base_topology.response(bit); responses += 1; fallbacks += int(response.fallback)
            base_loss = metrics_from_logits(base_topology.cache.logits, labels).categorical_crossentropy if base_topology.cache is not None else metrics_from_logits(evaluator.full_forward(inputs, faultable).logits, labels).categorical_crossentropy
            faulted_loss = metrics_from_logits(base_topology.multi_toggle_logits((bit,)), labels).categorical_crossentropy
            raw[row, column] = faulted_loss - base_loss
            contributions[row, column] = bit_probability(bit, probabilities) * raw[row, column]
            work += evaluator.full_work_proxy_macs(len(inputs)) if response.fallback else len(inputs)
        for offset, bit in enumerate(additions, len(removals)):
            conditioned = faultable.without_bit(bit)
            item = topology(conditioned); response = item.response(bit); responses += 1; fallbacks += int(response.fallback)
            base_loss = metrics_from_logits(item.cache.logits, labels).categorical_crossentropy if item.cache is not None else metrics_from_logits(evaluator.full_forward(inputs, conditioned).logits, labels).categorical_crossentropy
            faulted_loss = metrics_from_logits(item.multi_toggle_logits((bit,)), labels).categorical_crossentropy
            raw[row, offset] = faulted_loss - base_loss
            contributions[row, offset] = bit_probability(bit, probabilities) * raw[row, offset]
            work += evaluator.full_work_proxy_macs(len(inputs)) if response.fallback else len(inputs)

    elapsed = perf_counter() - started
    marginal_rows = []
    for index, bit in enumerate(candidate_order):
        weighted = summarize(contributions[:, index], confidence=confidence)
        unweighted = summarize(raw[:, index], confidence=confidence)
        marginal_rows.append(CandidateMarginal(
            bit, "protected" if index < len(removals) else "unprotected",
            weighted.mean, unweighted.mean, weighted.standard_error,
            weighted.confidence_interval, len(base_fault_fields), len(inputs),
            len(base_fault_fields), work // max(1, len(candidate_order)), elapsed / len(candidate_order),
        ))
    return ConditionalScreeningResult(
        protected, tuple(marginal_rows[:len(removals)]), tuple(marginal_rows[len(removals):]),
        contributions, candidate_order, cache_hits, cache_misses, fallbacks, work, elapsed,
    )
