"""Deterministic tests of netsurf.group_search (group-search plan v2, Phase 0)."""
import itertools

import numpy as np
import pytest

from netsurf.group_search import (budget_counts, evaluate_round, evaluate_slice_round, groups, next_slices,
                                  path_masks, protect, protect_slice, slice_path_masks, slices, unprotected)

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


def test_slices_partition_groups_in_score_order():
    rng = np.random.default_rng(1)
    score = rng.random((N, 8))
    score[0:3, 5] = 0.5                                          # ties: storage order
    gs = groups(STARTS)
    sl = slices(gs, score, 4)
    assert [s.uid for s in sl] == list(range(len(gs) * 4))
    cover = np.zeros((N, 8), dtype=int)
    for g in gs:
        parts = [s for s in sl if s.gid == g.gid]
        sizes = [s.idx.size for s in parts]
        assert sum(sizes) == g.hi - g.lo and max(sizes) - min(sizes) <= 1
        cat = np.concatenate([s.idx for s in parts])
        assert np.all(np.diff(score[cat, g.bit]) <= 0)           # highest score first, across slices
        for s in parts:
            cover[s.idx, s.bit] += 1
    assert (cover == 1).all()
    assert list(slices(gs, score, 4)[0 * 4 + 0].idx[:0]) == []
    tied = [s for s in slices(groups([0, 3]), np.full((3, 8), 0.5), 1) if s.bit == 5][0]
    assert list(tied.idx) == [0, 1, 2]


def test_one_slice_per_group_reproduces_the_group_search():
    rng = np.random.default_rng(2)
    w = rng.random((N, 8))
    gs = groups(STARTS)
    sl = slices(gs, rng.random((N, 8)), 1)
    a, b = np.zeros((N, 8), dtype=bool), np.zeros((N, 8), dtype=bool)
    for i in range(6):
        ra = evaluate_round(i, a, gs, linear_measure(w))
        rb = evaluate_slice_round(i, b, sl, linear_measure(w))
        assert ra.accepted == rb.accepted
        assert [c["gain"] for c in ra.candidates] == [c["gain"] for c in rb.candidates]
        a, b = protect(a, gs[ra.accepted]), protect_slice(b, sl[rb.accepted])
        assert np.array_equal(a, b)


def test_a_later_slice_is_offered_only_after_the_earlier_one():
    rng = np.random.default_rng(3)
    gs = groups(STARTS)
    sl = slices(gs, rng.random((N, 8)), 3)
    mk = np.zeros((N, 8), dtype=bool)
    assert {s.part for s in next_slices(mk, sl)} == {0} and len(next_slices(mk, sl)) == len(gs)
    mk = protect_slice(mk, sl[4 * 3 + 0])                         # group 4, part 0
    nxt = {s.gid: s.part for s in next_slices(mk, sl)}
    assert nxt[4] == 1 and all(p == 0 for g, p in nxt.items() if g != 4)
    w = np.zeros((N, 8)); w[:, 2] = 1.0
    r = evaluate_slice_round(0, mk, sl, linear_measure(w))
    assert r.accepted in {s.uid for s in next_slices(mk, sl)}
    assert sl[r.accepted].bit == 2 and sl[r.accepted].part == 0


def test_slice_path_masks_exact_nested_and_filled_in_score_order():
    rng = np.random.default_rng(4)
    score = rng.random((N, 8))
    gs = groups(STARTS)
    sl = slices(gs, score, 4)
    counts = budget_counts(N * 8, (0.125, 0.25, 0.375))
    order = [s.uid for s in sl]
    full = slice_path_masks(N, sl, order, counts)
    fs = sorted(counts)
    for f in fs:
        assert full[f].sum() == counts[f]
    for a, b in zip(fs, fs[1:]):
        assert (full[a] <= full[b]).all()
    # the first budget ends inside some slice: its protected bits there are that slice's highest-scored weights
    k = counts[fs[0]]
    used = 0
    for s in sl:
        if used + s.idx.size > k:
            take = k - used
            assert full[fs[0]][s.idx[:take], s.bit].all() and not full[fs[0]][s.idx[take:], s.bit].any()
            break
        used += s.idx.size
    with pytest.raises(ValueError):
        slice_path_masks(N, sl, order[:3], counts)


def test_budget_counts_floor():
    assert budget_counts(176128, (0.125, 0.1875, 0.25, 0.375)) == {0.125: 22016, 0.1875: 33024, 0.25: 44032, 0.375: 66048}
