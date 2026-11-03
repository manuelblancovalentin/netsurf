"""Source-pinned PrioriFI probing order, without claiming protection effectiveness.

Upstream KastnerRG/priorifi commit 6e7637b92315e5df75bb6ce258972bdf1131bc20,
examples/smart-pixel/fault_injection_campaign.py. Its metric is a clean-vs-faulty
prediction-change fraction, not task loss. The allocation helper below is a
separately identified NetSuRF adaptation and requires complete calibration.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np

SOURCE_COMMIT = "6e7637b92315e5df75bb6ce258972bdf1131bc20"


def feedback_probe_order(bit_tiers: Sequence[Sequence[str]],
                         probe: Callable[[str], float], *, last_k: int = 5) -> list[tuple[str, float]]:
    """Probe first in each significance tier, then select the next tier by feedback.

    Within each tier, bits must already follow the published FKeras Hessian
    parameter order. Ties between tiers follow the pinned source's NumPy
    ``argsort`` behavior for the installed NumPy version; exact ties should be
    retained as a source-version-sensitive provenance detail.
    The callback must perform one isolated fault and restore clean state.
    """
    if last_k < 1 or not bit_tiers or any(not tier for tier in bit_tiers):
        raise ValueError("last_k must be positive and every supplied tier must be nonempty")
    tiers = [list(tier) for tier in bit_tiers]
    population = [bit for tier in tiers for bit in tier]
    if len(population) != len(set(population)):
        raise ValueError("duplicate encoded-bit identities")
    history: list[list[float]] = [[] for _ in tiers]
    observed: list[tuple[str, float]] = []

    def measure(index: int) -> None:
        bit = tiers[index].pop(0)
        metric = float(probe(bit))
        if not math.isfinite(metric):
            raise ValueError(f"nonfinite PrioriFI probe metric for {bit}")
        observed.append((bit, metric))
        history[index].append(metric)
        if len(history[index]) > last_k:
            history[index].pop(0)

    for index, tier in enumerate(tiers):
        if tier:
            measure(index)
    while any(tiers):
        # Reproduce the pinned source's unusual branch: the *remaining tier
        # length* determines median(last k) versus mean(all retained probes).
        # Empty tiers are skipped after ordering, as in the source.
        priority = [
            (np.median(history[i][-last_k:]) if len(tiers[i]) >= last_k
             else np.mean(history[i]))
            for i in range(len(tiers))
        ]
        index = next(
            int(i) for i in np.argsort(-np.asarray(priority)) if tiers[int(i)]
        )
        measure(index)
    assert len(observed) == len(population)
    return observed


def derived_allocation(probes: Sequence[tuple[str, float]], population: Sequence[str],
                       protected_bits: int) -> tuple[str, ...]:
    """NetSuRF adaptation: rank *fully measured* isolated change fractions.

    This is NOT a protection-allocation rule prescribed by PrioriFI. An
    incomplete feedback trace cannot produce a whole-universe mask here.
    """
    identity = list(population)
    if len(identity) != len(set(identity)) or not 0 <= protected_bits <= len(identity):
        raise ValueError("invalid protection population or budget")
    measured = dict(probes)
    if len(measured) != len(probes) or set(measured) != set(identity):
        raise ValueError("complete unique calibration measurement required for every eligible bit")
    if any(not math.isfinite(float(value)) for value in measured.values()):
        raise ValueError("nonfinite calibration metric")
    ordinal = {bit: index for index, bit in enumerate(identity)}
    order = sorted(identity, key=lambda bit: (-measured[bit], ordinal[bit]))
    return tuple(order[:protected_bits])
