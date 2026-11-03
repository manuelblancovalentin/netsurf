"""Equal-query allocation search on a fixed, paired selection fault field.

The evaluator is supplied by a benchmark adapter and must apply the same
prespecified master fault fields and input cohort to every queried mask. This
module neither estimates susceptibility nor runs final evaluation during
selection. A query is one complete mask-risk measurement, not one bit flip.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
import time
from typing import Callable, Hashable, Sequence


Bit = Hashable
RiskEvaluator = Callable[[frozenset[Bit]], float]


@dataclass(frozen=True)
class SearchObservation:
    query: int
    candidate: tuple[Bit, ...]
    measured_risk: float
    accepted: bool


@dataclass(frozen=True)
class SearchResult:
    method: str
    seed: int
    selection_identity: str
    initial_mask: tuple[Bit, ...]
    selected_mask: tuple[Bit, ...]
    budget: int
    query_budget: int
    calls_per_query: int
    clip_inferences_per_query: int
    model_requests: int
    clip_inferences: int
    cost_type: str
    saved_table_lookups: int
    elapsed_seconds: float
    observations: tuple[SearchObservation, ...]


def _check(universe: Sequence[Bit], initial: Sequence[Bit], query_budget: int,
           selection_identity: str, calls_per_query: int,
           clip_inferences_per_query: int, cost_type: str) -> tuple[tuple[Bit, ...], frozenset[Bit]]:
    bits = tuple(universe)
    mask = frozenset(initial)
    if (not bits or len(set(bits)) != len(bits) or not mask or
        len(mask) != len(initial) or len(mask) >= len(bits) or not mask <= set(bits)):
        raise ValueError("unique universe and a nonempty, proper initial mask required")
    if (query_budget < 1 or not selection_identity or cost_type not in
        {"model_inference", "saved_table_lookup"} or
        (cost_type == "model_inference" and (calls_per_query < 1 or
         clip_inferences_per_query < calls_per_query)) or
        (cost_type == "saved_table_lookup" and
         (calls_per_query != 0 or clip_inferences_per_query != 0))):
        raise ValueError("positive fixed query cost and selection identity required")
    return bits, mask


def _measure(evaluate: RiskEvaluator, mask: frozenset[Bit]) -> float:
    value = float(evaluate(mask))
    if not math.isfinite(value):
        raise ValueError("risk evaluator must apply its prespecified finite failure penalty")
    return value


def search_allocation(universe: Sequence[Bit], initial: Sequence[Bit], *,
                      method: str, query_budget: int, seed: int,
                      selection_identity: str, calls_per_query: int,
                      clip_inferences_per_query: int,
                      evaluate: RiskEvaluator,
                      cost_type: str = "model_inference") -> SearchResult:
    """Use Q equal-cost queries, including the same initial-mask measurement.

    ``random_search`` samples uniform cardinality-B masks with replacement and
    retains the lowest measured risk. ``greedy_swap`` samples one protected and
    one unprotected coordinate uniformly at each step, accepting a strict
    measured improvement. Both use the same fixed selection field/input set.
    """
    if method not in {"random_search", "greedy_swap"}:
        raise ValueError("unknown allocation search method")
    bits, current = _check(universe, initial, query_budget, selection_identity,
                           calls_per_query, clip_inferences_per_query, cost_type)
    rng = random.Random(seed)
    started = time.perf_counter()
    current_risk = _measure(evaluate, current)
    observations = [SearchObservation(1, tuple(sorted(current)), current_risk, True)]
    for query in range(2, query_budget + 1):
        if method == "random_search":
            candidate = frozenset(rng.sample(bits, len(current)))
        else:
            protected = tuple(bit for bit in bits if bit in current)
            unprotected = tuple(bit for bit in bits if bit not in current)
            candidate = (current - {rng.choice(protected)}) | {rng.choice(unprotected)}
        risk = _measure(evaluate, candidate)
        accepted = risk < current_risk
        if accepted:
            current, current_risk = candidate, risk
        observations.append(SearchObservation(query, tuple(sorted(candidate)), risk, accepted))
    return SearchResult(method, seed, selection_identity, tuple(sorted(initial)),
        tuple(sorted(current)), len(current), query_budget, calls_per_query,
        clip_inferences_per_query, query_budget * calls_per_query,
        query_budget * clip_inferences_per_query,
        cost_type, query_budget if cost_type == "saved_table_lookup" else 0,
        time.perf_counter() - started, tuple(observations))


def evaluate_held_out(results: Sequence[SearchResult], static_masks: dict[str, Sequence[Bit]],
                      *, final_identity: str, evaluate: RiskEvaluator) -> dict[str, float]:
    """Measure every selected/static mask on one independent final field set."""
    if not results or not final_identity or any(
        item.selection_identity == final_identity for item in results):
        raise ValueError("final fields and inputs must be independent of selection")
    selection_identities = {item.selection_identity for item in results}
    initial_masks = {item.initial_mask for item in results}
    if len(selection_identities) != 1 or len(initial_masks) != 1:
        raise ValueError("search arms must share selection fields and initial mask")
    masks = {**{name: tuple(bits) for name, bits in static_masks.items()},
             **{f"{item.method}_seed{item.seed}": item.selected_mask for item in results}}
    if len(masks) != len(static_masks) + len(results):
        raise ValueError("duplicate search outcome identity")
    return {name: _measure(evaluate, frozenset(mask)) for name, mask in masks.items()}
