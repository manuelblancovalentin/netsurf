"""Deterministic tests of netsurf.group_search (group-search plan v2, Phase 0)."""
import itertools

import numpy as np
import pytest

from netsurf.group_search import budget_counts, evaluate_round, groups, path_masks, protect, unprotected

STARTS = [0, 6, 10, 19]            # three toy layers of 6, 4 and 9 weights
N = STARTS[-1]


def test_groups_cover_every_bit_once():
    gs = groups(STARTS)
    assert len(gs) == 3 * 8
    cover = np.zeros((N, 8), dtype=int)
    for g in gs:
        cover[g.lo:g.hi, g.bit] += 1
    assert (cover == 1).all()
    mk = np.zeros((N, 8), dtype=bool)
    for g in gs:
        mk = protect(mk, g)
    assert mk.all() and unprotected(mk, gs) == []


def linear_measure(weights, K=4, noise=None):
    """Score = - sum of unprotected weights (per bit cell), identical on every field (plus optional field noise)."""
    def measure(masks):
        s = np.array([[-(weights * ~m).sum()] * K for m in masks], dtype=float)
        if noise is not None:
            s = s + noise[None, :K]
        return s, np.zeros_like(s)
    return measure


def test_round_picks_the_brute_force_best_gain_per_bit_and_pairs_by_field():
    rng = np.random.default_rng(0)
    w = rng.random((N, 8))
    gs = groups(STARTS)
    mk = np.zeros((N, 8), dtype=bool)
    r = evaluate_round(0, mk, gs, linear_measure(w, noise=rng.normal(0, 5, 4)))
    brute = max(gs, key=lambda g: (w[g.lo:g.hi, g.bit].sum() / (g.hi - g.lo), -g.gid))
    assert r.accepted == brute.gid
    # field noise is shared by all masks, so the paired gains are exact and their standard error is zero
    for c in r.candidates:
        g = gs[c["gid"]]
        assert c["gain"] == pytest.approx(w[g.lo:g.hi, g.bit].sum())
        assert c["se"] == pytest.approx(0.0, abs=1e-9)


def test_a_group_that_changes_nothing_has_gain_exactly_zero_and_protected_groups_are_not_offered():
    w = np.zeros((N, 8))
    w[:, 7] = 1.0
    gs = groups(STARTS)
    mk = protect(np.zeros((N, 8), dtype=bool), gs[7])            # layer 0, bit 7 already protected
    r = evaluate_round(1, mk, gs, linear_measure(w))
    assert 7 not in [c["gid"] for c in r.candidates]
    zero = [c for c in r.candidates if c["bit"] != 7]
    assert all(c["gain"] == 0.0 for c in zero)
    assert r.accepted in (15, 23)                                 # an unprotected bit-7 group


def test_two_group_toy_matches_brute_force_over_orders():
    starts = [0, 3]                                               # one layer, 3 weights; use only bits 0 and 1
    gs = [g for g in groups(starts) if g.bit in (0, 1)]
    vals = {(): 0.0, (0,): 0.5, (1,): 0.2, (0, 1): 0.9}

    def measure(masks):
        out = []
        for m in masks:
            key = tuple(b for b in (0, 1) if m[:, b].all())
            out.append([vals[key]] * 2)
        return np.array(out), np.zeros((len(masks), 2))
    r = evaluate_round(0, np.zeros((3, 8), dtype=bool), gs, measure)
    best = max((0, 1), key=lambda b: vals[(b,)])
    assert gs[[g.bit for g in gs].index(best)].gid == r.accepted


def test_path_masks_exact_counts_nested_and_reproducible():
    gs = groups(STARTS)
    total = N * 8
    counts = budget_counts(total, (0.125, 0.25, 0.375))
    order = [g.gid for g in gs]                                  # every group, in order
    full1, whole1 = path_masks(N, gs, order, counts, np.random.default_rng([20261213, 0, 1]))
    full2, _ = path_masks(N, gs, order, counts, np.random.default_rng([20261213, 0, 1]))
    fs = sorted(counts)
    for f in fs:
        assert full1[f].sum() == counts[f]
        assert whole1[f].sum() <= counts[f]
        assert np.array_equal(full1[f], full2[f])
        assert (whole1[f] <= full1[f]).all()
    for a, b in zip(fs, fs[1:]):
        assert (full1[a] <= full1[b]).all()                       # nested
    with pytest.raises(ValueError):
        path_masks(N, gs, order[:2], counts, np.random.default_rng(0))


def test_budget_counts_floor():
    assert budget_counts(176128, (0.125, 0.1875, 0.25, 0.375)) == {0.125: 22016, 0.1875: 33024, 0.25: 44032, 0.375: 66048}
