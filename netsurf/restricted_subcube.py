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
