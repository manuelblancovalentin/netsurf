"""Exact bounded subcube evidence; never a whole-network optimality claim."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import Callable, Mapping, Sequence


@dataclass(frozen=True)
class RestrictedSubcubeResult:
    candidate_count: int
    budget: int
    probability: float
    selected_mask: tuple[int, ...]
    calibration_risks: Mapping[tuple[int, ...], float]
    held_out_risks: Mapping[tuple[int, ...], float]


def subcube_states(candidate_count: int) -> tuple[tuple[int, ...], ...]:
    if not 1 <= candidate_count <= 12:
        raise ValueError("restricted subcube requires 1 <= K <= 12")
    return tuple(
        tuple(index for index, enabled in enumerate(state) if enabled)
        for state in itertools.product((0, 1), repeat=candidate_count)
    )


def fixed_budget_masks(candidate_count: int, budget: int) -> tuple[tuple[int, ...], ...]:
    if not 0 <= budget <= candidate_count <= 12:
        raise ValueError("fixed budget requires 0 <= budget <= K <= 12")
    return tuple(itertools.combinations(range(candidate_count), budget))


def integrate_response_table(
    responses: Mapping[tuple[int, ...], float], candidate_count: int,
    probability: float, *, protected: tuple[int, ...] = (),
) -> float:
    """Integrate a complete exact response table under Bernoulli candidate faults."""

    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0,1]")
    states = subcube_states(candidate_count)
    if set(responses) != set(states):
        missing = set(states) - set(responses)
        extra = set(responses) - set(states)
        raise ValueError(f"response table is incomplete: missing={len(missing)}, extra={len(extra)}")
    protected_set = set(protected)
    if len(protected_set) != len(protected) or not protected_set <= set(range(candidate_count)):
        raise ValueError("protected indices must be unique candidate indices")
    available = tuple(index for index in range(candidate_count) if index not in protected_set)
    total = mass_total = 0.0
    for state in states:
        if any(index in protected_set for index in state):
            continue
        count = len(state)
        mass = probability**count * (1.0 - probability) ** (len(available) - count)
        mass_total += mass
        total += mass * float(responses[state])
    if abs(mass_total - 1.0) > 1e-12:
        raise ArithmeticError(f"Bernoulli probability mass does not normalize: {mass_total}")
    return total


def conditional_marginal_from_table(
    responses: Mapping[tuple[int, ...], float], candidate_count: int,
    probability: float, candidate: int, *, protected: tuple[int, ...] = (),
) -> float:
    """Integrate R(S union {b})-R(S) over all other unprotected candidates."""

    if not 0 <= candidate < candidate_count:
        raise IndexError(candidate)
    protected_set = set(protected)
    if len(protected_set) != len(protected):
        raise ValueError("protected indices must be unique")
    if not protected_set <= set(range(candidate_count)):
        raise ValueError("protected indices must be candidate indices")
    protected_set.discard(candidate)
    states = subcube_states(candidate_count)
    if set(responses) != set(states):
        raise ValueError("response table is incomplete")
    others = tuple(
        index for index in range(candidate_count)
        if index != candidate and index not in protected_set
    )
    value = mass_total = 0.0
    for enabled in itertools.product((0, 1), repeat=len(others)):
        base = tuple(others[index] for index, present in enumerate(enabled) if present)
        count = len(base)
        mass = probability**count * (1.0 - probability) ** (len(others) - count)
        value += mass * (responses[tuple(sorted((*base, candidate)))] - responses[base])
        mass_total += mass
    if abs(mass_total - 1.0) > 1e-12:
        raise ArithmeticError(f"conditional probability mass does not normalize: {mass_total}")
    return value


def response_table_diagnostics(
    responses: Mapping[tuple[int, ...], float], candidate_count: int,
    budget: int, probability: float,
) -> dict[str, object]:
    """Derive all fixed-budget risks, marginals and paired differences without inference."""

    masks = fixed_budget_masks(candidate_count, budget)
    risks = {
        mask: integrate_response_table(
            responses, candidate_count, probability, protected=mask
        ) for mask in masks
    }
    marginals = {
        index: conditional_marginal_from_table(
            responses, candidate_count, probability, index
        ) for index in range(candidate_count)
    }
    differences = {
        (left, right): risks[left] - risks[right]
        for left_index, left in enumerate(masks)
        for right in masks[left_index + 1:]
    }
    return {"mask_risks": risks, "conditional_marginals": marginals,
            "pairwise_mask_differences": differences}


def analyze_restricted_response_tables(
    calibration_tables: Sequence[Mapping[tuple[int, ...], float]],
    held_out_tables: Sequence[Mapping[tuple[int, ...], float]],
    isolated_table: Mapping[tuple[int, ...], float],
    *, candidate_count: int, budget: int, probability: float,
    original_mask: tuple[int, ...] | None = None,
) -> dict[str, object]:
    """Select on calibration tables and retain paired held-out field risks."""

    if not calibration_tables or not held_out_tables:
        raise ValueError("calibration and held-out response tables must be nonempty")
    masks = fixed_budget_masks(candidate_count, budget)
    calibration_fields = {
        mask: tuple(
            integrate_response_table(table, candidate_count, probability, protected=mask)
            for table in calibration_tables
        ) for mask in masks
    }
    calibration_means = {
        mask: sum(values) / len(values) for mask, values in calibration_fields.items()
    }
    selected = min(masks, key=lambda mask: (calibration_means[mask], mask))
    clean = float(isolated_table[()])
    isolated_damages = {
        index: float(isolated_table[(index,)]) - clean for index in range(candidate_count)
    }
    isolated_mask = tuple(sorted(
        sorted(range(candidate_count), key=lambda index: (-isolated_damages[index], index))[:budget]
    ))
    if original_mask is None:
        original_mask = isolated_mask
    if original_mask not in masks:
        raise ValueError("original mask must be an eligible fixed-budget candidate mask")
    held_out_selected = tuple(
        integrate_response_table(table, candidate_count, probability, protected=selected)
        for table in held_out_tables
    )
    held_out_isolated = tuple(
        integrate_response_table(table, candidate_count, probability, protected=isolated_mask)
        for table in held_out_tables
    )
    held_out_original = tuple(
        integrate_response_table(table, candidate_count, probability, protected=original_mask)
        for table in held_out_tables
    )
    # This optimum is diagnostic only: it is computed after exposing held-out responses.
    held_out_by_mask = {
        mask: tuple(
            integrate_response_table(table, candidate_count, probability, protected=mask)
            for table in held_out_tables
        ) for mask in masks
    }
    held_out_means = {
        mask: sum(values) / len(values) for mask, values in held_out_by_mask.items()
    }
    posthoc_held_out_optimum = min(masks, key=lambda mask: (held_out_means[mask], mask))
    return {
        "selected_mask": selected,
        "isolated_mask": isolated_mask,
        "original_mask": original_mask,
        "posthoc_held_out_optimum_mask": posthoc_held_out_optimum,
        "isolated_damages": isolated_damages,
        "calibration_field_risks": calibration_fields,
        "calibration_mean_risks": calibration_means,
        "held_out_selected_field_risks": held_out_selected,
        "held_out_isolated_field_risks": held_out_isolated,
        "held_out_original_field_risks": held_out_original,
        "held_out_posthoc_optimum_field_risks": held_out_by_mask[posthoc_held_out_optimum],
        "held_out_posthoc_optimum_is_selection_valid": False,
    }


def _expected_subcube_risk(
    risks: Sequence[Callable[[tuple[int, ...]], float]], candidate_count: int,
    probability: float, protected: tuple[int, ...],
) -> float:
    available = tuple(index for index in range(candidate_count) if index not in protected)
    total = 0.0
    for risk in risks:
        expectation = 0.0
        for state in itertools.product((0, 1), repeat=len(available)):
            faults = tuple(available[index] for index, enabled in enumerate(state) if enabled)
            count = len(faults)
            mass = probability**count * (1.0 - probability) ** (len(available) - count)
            expectation += mass * float(risk(faults))
        total += expectation
    return total / len(risks)


def select_restricted_subcube_mask(
    calibration_background_risks: Sequence[Callable[[tuple[int, ...]], float]],
    held_out_background_risks: Sequence[Callable[[tuple[int, ...]], float]],
    *, candidate_count: int, budget: int, probability: float,
) -> RestrictedSubcubeResult:
    """Select on calibration backgrounds and evaluate once on independent backgrounds."""

    if not calibration_background_risks or not held_out_background_risks:
        raise ValueError("calibration and held-out background risks must be nonempty")
    if not 1 <= budget < candidate_count <= 12:
        raise ValueError("restricted subcube requires 1 <= budget < K <= 12")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0,1]")
    masks = tuple(itertools.combinations(range(candidate_count), budget))
    calibration = {
        mask: _expected_subcube_risk(
            calibration_background_risks, candidate_count, probability, mask
        ) for mask in masks
    }
    selected = min(masks, key=lambda mask: (calibration[mask], mask))
    held_out = {
        mask: _expected_subcube_risk(
            held_out_background_risks, candidate_count, probability, mask
        ) for mask in masks
    }
    return RestrictedSubcubeResult(
        candidate_count, budget, probability, selected, calibration, held_out
    )
