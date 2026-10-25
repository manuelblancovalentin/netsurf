"""Independent held-out comparisons between equal-budget protection masks.

The comparison is a telescoping sequence of legal swaps.  Each independent
base Bernoulli field contributes one *whole-path* paired observation, so
covariance among path steps is retained in the reported uncertainty.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Callable, Mapping, Sequence

import numpy as np

from .identities import BitIdentity
from .incremental import DenseReluDenseIncrementalEvaluator, FaultState, metrics_from_logits
from .manifests import digest_object
from .statistics import summarize


@dataclass(frozen=True)
class MaskSwap:
    index: int
    removed: BitIdentity
    added: BitIdentity

    def to_dict(self):
        return {"index": self.index, "removed": self.removed.key(), "added": self.added.key()}


@dataclass(frozen=True)
class ConditionalContextContribution:
    context: int
    schedule_sha256: str
    base_fault_count: int
    loss_difference: float
    zero_one_risk_difference: float
    direct_loss_difference: float
    direct_zero_one_risk_difference: float
    background_evaluations: int
    incremental_evaluations: int
    full_evaluations: int
    work_proxy_macs: int
    gate_changes: int
    cache_hits: int
    cache_misses: int
    fallback_count: int
    wall_seconds: float

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ConditionalMaskComparison:
    baseline_name: str
    target_name: str
    path_rule: str
    swaps: tuple[MaskSwap, ...]
    context_count: int
    input_sample_count: int
    conditional_loss_difference: Mapping[str, object]
    conditional_zero_one_risk_difference: Mapping[str, object]
    direct_loss_difference: Mapping[str, object]
    direct_zero_one_risk_difference: Mapping[str, object]
    contributions: tuple[ConditionalContextContribution, ...]
    exact_evaluations: int
    work_proxy_macs: int
    wall_seconds: float
    implementation_version: str = "telescoping-three-state-conditional-risk/v1"

    def to_dict(self):
        return {
            "baseline_name": self.baseline_name,
            "target_name": self.target_name,
            "path_rule": self.path_rule,
            "swaps": [value.to_dict() for value in self.swaps],
            "context_count": self.context_count,
            "input_sample_count": self.input_sample_count,
            "conditional_loss_difference": dict(self.conditional_loss_difference),
            "conditional_zero_one_risk_difference": dict(self.conditional_zero_one_risk_difference),
            "direct_loss_difference": dict(self.direct_loss_difference),
            "direct_zero_one_risk_difference": dict(self.direct_zero_one_risk_difference),
            "exact_evaluations": self.exact_evaluations,
            "work_proxy_macs": self.work_proxy_macs,
            "wall_seconds": self.wall_seconds,
            "implementation_version": self.implementation_version,
        }


def deterministic_swap_path(
    baseline: Sequence[BitIdentity] | frozenset[BitIdentity],
    target: Sequence[BitIdentity] | frozenset[BitIdentity],
) -> tuple[MaskSwap, ...]:
    """Pair sorted removals and additions to form a deterministic legal path."""
    left, right = frozenset(baseline), frozenset(target)
    if len(left) != len(right):
        raise ValueError("conditional mask comparison requires equal-cardinality masks")
    removed, added = tuple(sorted(left - right)), tuple(sorted(right - left))
    if len(removed) != len(added):
        raise RuntimeError("mask symmetric differences cannot form one-for-one swaps")
    path = tuple(MaskSwap(index, old, new) for index, (old, new) in enumerate(zip(removed, added), 1))
    current = set(left)
    for swap in path:
        if swap.removed not in current or swap.added in current:
            raise RuntimeError("constructed an illegal swap path")
        current.remove(swap.removed); current.add(swap.added)
    if frozenset(current) != right:
        raise RuntimeError("swap path does not reach the requested target mask")
    return path


def _probability(bit: BitIdentity, probabilities: float | Mapping[BitIdentity, float]) -> float:
    value = probabilities if isinstance(probabilities, (int, float)) else probabilities[bit]
    value = float(value)
    if not 0 <= value <= 1:
        raise ValueError("bit fault probabilities must be in [0,1]")
    return value


def compare_protection_masks(
    evaluator: DenseReluDenseIncrementalEvaluator,
    inputs: np.ndarray,
    labels: np.ndarray,
    universe: Sequence[BitIdentity],
    base_fault_fields: Sequence[FaultState],
    baseline: Sequence[BitIdentity] | frozenset[BitIdentity],
    target: Sequence[BitIdentity] | frozenset[BitIdentity],
    *,
    probabilities: float | Mapping[BitIdentity, float],
    confidence: float = 0.95,
    baseline_name: str = "static",
    target_name: str = "target",
    mode: str = "incremental",
    progress: Callable[[int, int], None] | None = None,
) -> ConditionalMaskComparison:
    """Compare masks using whole-path conditional and direct paired endpoints.

    ``base_fault_fields`` are mask-independent common Bernoulli fields.  For
    every path step the current protected set and both swapped bits are removed
    to form the valid conditional background.  A context's path steps are
    summed before statistical aggregation.
    """
    if mode not in {"incremental", "full"}:
        raise ValueError("conditional comparison mode must be incremental or full")
    universe_set = frozenset(universe)
    left, right = frozenset(baseline), frozenset(target)
    if not left <= universe_set or not right <= universe_set:
        raise ValueError("protection masks must be subsets of the eligible universe")
    if not base_fault_fields:
        raise ValueError("at least one held-out base fault field is required")
    if any(not state.bits <= universe_set for state in base_fault_fields):
        raise ValueError("base fault fields contain bits outside the eligible universe")
    path = deterministic_swap_path(left, right)
    started = perf_counter()
    rows = []
    for context, base_field in enumerate(base_fault_fields):
        context_started = perf_counter()
        current = set(left)
        loss_difference = zero_one_difference = 0.0
        background_count = incremental_count = full_count = work = gates = 0
        cache_hits = cache_misses = fallbacks = 0
        cache_store = {} if mode == "incremental" else None
        for swap in path:
            background = FaultState(base_field.bits - current - {swap.removed, swap.added})
            reused = bool(cache_store is not None and background in cache_store and cache_store[background] is not None)
            cache = cache_store.get(background) if reused else (evaluator.cache(inputs, background) if mode == "incremental" else None)
            if cache_store is not None and cache is not None and not reused:
                cache_store[background] = cache
            if cache is not None:
                base_logits = cache.logits
                removed_result = evaluator.incremental_forward(cache, inputs, background.with_bit(swap.removed))
                added_result = evaluator.incremental_forward(cache, inputs, background.with_bit(swap.added))
                incremental_count += 2; cache_hits += int(reused); cache_misses += int(not reused)
            else:
                base_result = evaluator.full_forward(inputs, background)
                removed_result = evaluator.full_forward(inputs, background.with_bit(swap.removed))
                added_result = evaluator.full_forward(inputs, background.with_bit(swap.added))
                base_logits = base_result.logits; full_count += 2; cache_misses += int(not reused)
            if not reused:
                background_count += 1
                work += evaluator.full_work_proxy_macs(len(inputs))
            base_metrics = metrics_from_logits(base_logits, labels)
            removed_metrics = metrics_from_logits(removed_result.logits, labels)
            added_metrics = metrics_from_logits(added_result.logits, labels)
            p_removed, p_added = _probability(swap.removed, probabilities), _probability(swap.added, probabilities)
            loss_difference += p_removed * (removed_metrics.categorical_crossentropy - base_metrics.categorical_crossentropy)
            loss_difference -= p_added * (added_metrics.categorical_crossentropy - base_metrics.categorical_crossentropy)
            base_error, removed_error, added_error = 1-base_metrics.accuracy, 1-removed_metrics.accuracy, 1-added_metrics.accuracy
            zero_one_difference += p_removed * (removed_error-base_error) - p_added * (added_error-base_error)
            work += removed_result.diagnostics.work_proxy_macs + added_result.diagnostics.work_proxy_macs
            gates += removed_result.diagnostics.relu_gate_changes + added_result.diagnostics.relu_gate_changes
            fallbacks += int(removed_result.diagnostics.fallback) + int(added_result.diagnostics.fallback)
            current.remove(swap.removed); current.add(swap.added)
        baseline_metrics = metrics_from_logits(evaluator.full_forward(inputs, FaultState(base_field.bits-left)).logits, labels)
        target_metrics = metrics_from_logits(evaluator.full_forward(inputs, FaultState(base_field.bits-right)).logits, labels)
        full_count += 2; work += 2 * evaluator.full_work_proxy_macs(len(inputs))
        rows.append(ConditionalContextContribution(
            context, digest_object([bit.key() for bit in sorted(base_field.bits)]), len(base_field.bits),
            loss_difference, zero_one_difference,
            target_metrics.categorical_crossentropy-baseline_metrics.categorical_crossentropy,
            (1-target_metrics.accuracy)-(1-baseline_metrics.accuracy),
            background_count, incremental_count, full_count, work, gates,
            cache_hits, cache_misses, fallbacks, perf_counter()-context_started,
        ))
        if progress is not None:
            progress(context+1, len(base_fault_fields))
    conditional_loss = [row.loss_difference for row in rows]
    conditional_zero_one = [row.zero_one_risk_difference for row in rows]
    direct_loss = [row.direct_loss_difference for row in rows]
    direct_zero_one = [row.direct_zero_one_risk_difference for row in rows]
    return ConditionalMaskComparison(
        baseline_name, target_name, "sorted_removed_then_sorted_added/v1", path,
        len(rows), len(inputs), summarize(conditional_loss, confidence).to_dict(),
        summarize(conditional_zero_one, confidence).to_dict(), summarize(direct_loss, confidence).to_dict(),
        summarize(direct_zero_one, confidence).to_dict(), tuple(rows),
        sum(row.background_evaluations+row.incremental_evaluations+row.full_evaluations for row in rows),
        sum(row.work_proxy_macs for row in rows), perf_counter()-started,
    )
