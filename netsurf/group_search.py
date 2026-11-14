"""Greedy protection search over groups of bits (EXPLORATORY; plan review/GROUP_SEARCH_PLAN_v2_2026-11-14.md).

Side effect free, NumPy only.  A group is one layer's bit position (all bit-b cells of the layer's weights).  A mask is
an (n, 8) boolean array, True = protected.  The search is generic: it receives a function that, for a list of masks,
returns a (len(masks), K) array of accuracies (or any score, higher is better) measured on K shared fault fields, and a
second array of the same shape used to break ties (for example minus the cross-entropy).  Pairing is by field.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Group:
    gid: int
    layer: int
    bit: int
    lo: int          # first weight index of the layer
    hi: int          # one past the last


def groups(starts) -> list[Group]:
    """All layer x bit groups, in layer order then bit order (gid = 8 * layer + bit)."""
    starts = [int(s) for s in starts]
    return [Group(8 * li + b, li, b, starts[li], starts[li + 1]) for li in range(len(starts) - 1) for b in range(8)]


def protect(mask: np.ndarray, g: Group) -> np.ndarray:
    out = mask.copy()
    out[g.lo:g.hi, g.bit] = True
    return out


def unprotected(mask: np.ndarray, gs: list[Group]) -> list[Group]:
    return [g for g in gs if not mask[g.lo:g.hi, g.bit].all()]


def budget_counts(total_bits: int, fractions) -> dict:
    return {float(f): int(np.floor(f * total_bits)) for f in fractions}


@dataclass
class Round:
    index: int
    current: float
    candidates: list = field(default_factory=list)   # dicts: gid, size, gain, se, gain_per_bit, tie
    accepted: int = -1


def evaluate_round(index: int, mask: np.ndarray, gs: list[Group], measure) -> Round:
    """Measure the current mask and every unprotected group added to it on the same K fields; paired gains."""
    cand = unprotected(mask, gs)
    masks = [mask] + [protect(mask, g) for g in cand]
    score, tie = measure(masks)
    score, tie = np.asarray(score, dtype=np.float64), np.asarray(tie, dtype=np.float64)
    K = score.shape[1]
    r = Round(index, float(score[0].mean()))
    for j, g in enumerate(cand, start=1):
        d = score[j] - score[0]
        size = g.hi - g.lo
        r.candidates.append(dict(gid=g.gid, layer=g.layer, bit=g.bit, size=size, gain=float(d.mean()),
                                 se=float(d.std(ddof=1) / np.sqrt(K)) if K > 1 else float("nan"),
                                 gain_per_bit=float(d.mean() / size), tie=float((tie[j] - tie[0]).mean() / size)))
    if r.candidates:
        best = sorted(r.candidates, key=lambda c: (-c["gain_per_bit"], -c["tie"], c["gid"]))[0]
        r.accepted = best["gid"]
    return r


def path_masks(n: int, gs: list[Group], accepted: list[int], counts: dict, rng: np.random.Generator):
    """Nested budget masks from the accepted group order.

    For each budget: whole accepted groups while they fit, then the remaining bits drawn uniformly at random (without
    replacement) from the next accepted group.  Also returns the whole-groups-only mask of each budget.  The random draw
    for a budget's partial group is taken from one permutation per group, so the masks are nested across budgets."""
    by_id = {g.gid: g for g in gs}
    perms = {gid: rng.permutation(by_id[gid].hi - by_id[gid].lo) for gid in accepted}
    full, whole = {}, {}
    for f, count in sorted(counts.items()):
        mk = np.zeros((n, 8), dtype=bool)
        used, partial_done = 0, False
        for gid in accepted:
            g = by_id[gid]
            size = g.hi - g.lo
            if used + size <= count:
                mk[g.lo:g.hi, g.bit] = True
                used += size
                continue
            whole[f] = mk.copy()
            take = count - used
            mk[g.lo + perms[gid][:take], g.bit] = True
            used += take
            partial_done = True
            break
        if not partial_done:
            whole[f] = mk.copy()
        if used != count:
            raise ValueError(f"accepted groups hold only {used} bits, budget {f} needs {count}")
        full[f] = mk
    return full, whole
