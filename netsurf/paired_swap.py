"""Exact paired-swap contrasts for already frozen equal-budget masks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Hashable, Iterable, Sequence, TypeVar

from .manifests import digest_object


T = TypeVar("T", bound=Hashable)


@dataclass(frozen=True)
class SwapStep(Generic[T]):
    index: int
    removed: T
    added: T


def deterministic_swap_path(
    baseline: Iterable[T], target: Iterable[T],
) -> tuple[SwapStep[T], ...]:
    """Pair sorted removals/additions into a stable legal path."""

    left, right = frozenset(baseline), frozenset(target)
    if len(left) != len(right):
        raise ValueError("paired-swap comparison requires equal-cardinality masks")
    removed, added = tuple(sorted(left - right)), tuple(sorted(right - left))
    path = tuple(
        SwapStep(index, old, new)
        for index, (old, new) in enumerate(zip(removed, added), 1)
    )
    current = set(left)
    for step in path:
        if step.removed not in current or step.added in current:
            raise RuntimeError("constructed an illegal paired-swap path")
        current.remove(step.removed)
        current.add(step.added)
    if frozenset(current) != right:
        raise RuntimeError("paired-swap path does not reach target mask")
    return path


def paired_swap_field_contrast(
    evaluate: Callable[[frozenset[T]], float],
    master_field: Iterable[T],
    baseline: Iterable[T],
    target: Iterable[T],
    *, probability: float,
    fixed_protected: Iterable[T] = (),
) -> dict[str, object]:
    """Return one whole-path observation of ``J(target)-J(baseline)``.

    The master field supplies common Bernoulli draws.  At each legal swap the
    current protected set and both exchanged bits are removed, leaving the
    correct background law over all other unprotected coordinates.  The
    callback must perform authoritative inference for exactly the supplied
    fault identities.
    """

    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0,1]")
    left, right = frozenset(baseline), frozenset(target)
    path = deterministic_swap_path(left, right)
    master = frozenset(master_field)
    fixed = frozenset(fixed_protected)
    if fixed & (left | right):
        raise ValueError("fixed protection and exchanged masks must be disjoint")
    current = set(left)
    contributions = []
    total = 0.0
    for step in path:
        background = master - fixed - current - {step.removed, step.added}
        if probability == 0.0:
            removed_risk = added_risk = 0.0
            evaluated = False
        else:
            removed_risk = float(evaluate(background | {step.removed}))
            added_risk = float(evaluate(background | {step.added}))
            evaluated = True
        difference = removed_risk - added_risk
        contribution = probability * difference
        total += contribution
        contributions.append({
            "step": step.index,
            "removed": step.removed,
            "added": step.added,
            "current_mask": tuple(sorted(current)),
            "background": tuple(sorted(background)),
            "background_sha256": digest_object([str(value) for value in sorted(background)]),
            "removed_risk": removed_risk,
            "added_risk": added_risk,
            "removed_minus_added": difference,
            "target_minus_baseline_contribution": contribution,
            "authoritative_evaluations": 2 if evaluated else 0,
        })
        current.remove(step.removed)
        current.add(step.added)
    return {
        "path": path,
        "steps": tuple(contributions),
        "target_minus_baseline": total,
        "authoritative_evaluations": sum(
            int(row["authoritative_evaluations"]) for row in contributions
        ),
    }


def exact_bernoulli_mask_risk(
    risk: Callable[[frozenset[int]], float], bit_count: int, probability: float,
    protected: Iterable[int],
) -> float:
    """Small-space reference integration used only by mathematical tests."""

    import itertools

    protected_set = frozenset(protected)
    if not protected_set <= frozenset(range(bit_count)):
        raise ValueError("protected mask is outside reference universe")
    available = tuple(index for index in range(bit_count) if index not in protected_set)
    total = 0.0
    for enabled in itertools.product((0, 1), repeat=len(available)):
        faults = frozenset(
            available[index] for index, present in enumerate(enabled) if present
        )
        count = len(faults)
        mass = probability**count * (1.0 - probability) ** (len(available) - count)
        total += mass * float(risk(faults))
    return total


def exact_paired_swap_contrast(
    risk: Callable[[frozenset[int]], float], bit_count: int, probability: float,
    baseline: Iterable[int], target: Iterable[int],
) -> float:
    """Exactly integrate the paired-swap identity over a small full space."""

    import itertools

    left, right = frozenset(baseline), frozenset(target)
    path = deterministic_swap_path(left, right)
    current = set(left)
    total = 0.0
    for step in path:
        others = tuple(
            index for index in range(bit_count)
            if index not in current and index != step.added
        )
        expectation = 0.0
        for enabled in itertools.product((0, 1), repeat=len(others)):
            background = frozenset(
                others[index] for index, present in enumerate(enabled) if present
            )
            count = len(background)
            mass = probability**count * (1.0 - probability) ** (len(others) - count)
            expectation += mass * (
                float(risk(background | {step.removed}))
                - float(risk(background | {step.added}))
            )
        total += probability * expectation
        current.remove(step.removed)
        current.add(step.added)
    return total
