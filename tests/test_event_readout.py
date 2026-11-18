"""Deterministic tests of netsurf.event_readout (top-bit analysis plan, Phase 0)."""
import itertools

import numpy as np
import pytest

from netsurf.event_readout import (additive_scores, correct, event_sum_pmf, gaussian_correct_probability,
                                   lattice_correct_probability, runner_up, tie_threshold)


def brute_force_pmf(a, p):
    out = {}
    for bits in itertools.product((0, 1), repeat=len(a)):
        s = int(np.dot(bits, a))
        out[s] = out.get(s, 0.0) + p ** sum(bits) * (1 - p) ** (len(a) - sum(bits))
    return out


@pytest.mark.parametrize("p", [0.003, 0.1, 0.5])
def test_event_sum_pmf_matches_enumeration(p):
    a = [3, -2, 0, 5, 3, -7, 1, 1, -2, 12, 0, -1]
    lo, pmf, dropped = event_sum_pmf(a, p, eps=1e-15)
    ref = brute_force_pmf(a, p)
    got = {lo + j: v for j, v in enumerate(pmf) if v > 0}
    for s, v in ref.items():
        if v > 1e-12:
            assert got.get(s, 0.0) == pytest.approx(v, rel=1e-9, abs=1e-14)
    assert dropped < 1e-12
    assert pmf.sum() + dropped == pytest.approx(1.0, abs=1e-12)


def test_event_sum_pmf_order_invariant_and_empty():
    rng = np.random.default_rng(1)
    a = rng.integers(-20, 21, size=400)
    lo1, p1, _ = event_sum_pmf(a, 0.02)
    lo2, p2, _ = event_sum_pmf(a[::-1], 0.02)
    assert lo1 == lo2 and np.allclose(p1, p2, atol=1e-15)
    assert event_sum_pmf([0, 0], 0.3) == (0, pytest.approx(np.array([1.0])), 0.0)
    assert event_sum_pmf([4, -1], 0.0)[1].tolist() == [1.0]


def test_lattice_probability_matches_simulation_of_the_same_model():
    rng = np.random.default_rng(7)
    a = rng.integers(-15, 6, size=300)
    p, m0, thr, mu, var = 0.03, 6, 1, 0.4, 2.5
    prob, dropped = lattice_correct_probability(m0, a, p, thr, mu, var)
    draws = 200_000
    s = (rng.random((draws, a.size)) < p).astype(np.int64) @ a
    b = rng.normal(mu, np.sqrt(var), draws)
    sim = np.mean(m0 + s + b > thr - 0.5)
    se = np.sqrt(sim * (1 - sim) / draws)
    assert abs(prob - sim) < 4 * se + dropped
    # zero background variance: exact indicator on the lattice
    prob0, _ = lattice_correct_probability(m0, a, p, thr)
    s0 = (rng.random((draws, a.size)) < p).astype(np.int64) @ a
    sim0 = np.mean(m0 + s0 >= thr)
    assert abs(prob0 - sim0) < 4 * np.sqrt(sim0 * (1 - sim0) / draws)


def test_gaussian_readout_approaches_lattice_for_many_small_effects():
    rng = np.random.default_rng(3)
    a = rng.integers(-1, 2, size=5000)
    p, m0 = 0.3, 12
    exact, _ = lattice_correct_probability(m0, a, p, 1, 0.0, 4.0)
    approx = float(gaussian_correct_probability(m0, a, p, 1, 0.0, 4.0))
    assert approx == pytest.approx(exact, abs=0.01)
    # vectorised over clips equals per-clip calls
    A = rng.integers(-3, 4, size=(5, 50))
    m = np.array([1, 0, -2, 5, 3])
    thr = np.array([1, 0, 1, 1, 0])
    vec = gaussian_correct_probability(m, A, 0.1, thr, 0.5, 1.0)
    one = [float(gaussian_correct_probability(m[i], A[i], 0.1, thr[i], 0.5, 1.0)) for i in range(5)]
    assert np.allclose(vec, one)


def test_single_event_additive_scores_equal_the_measured_scores():
    rng = np.random.default_rng(5)
    clean = rng.integers(-128, 128, size=(9, 12))
    flipped = rng.integers(-128, 128, size=(1, 9, 12))
    labels = rng.integers(0, 12, size=9)
    assert np.array_equal(additive_scores(clean, flipped), flipped[0])
    assert np.array_equal(correct(additive_scores(clean, flipped), labels), correct(flipped[0], labels))
    bg = rng.integers(-128, 128, size=(9, 12))
    assert np.array_equal(additive_scores(clean, None, bg), bg)
    # two events add their changes and the result is clipped to INT8
    two = np.stack([clean + 100, clean + 100]).clip(-128, 127)
    assert additive_scores(clean, two).max() <= 127


def test_runner_up_and_tie_rule():
    clean = np.array([[5, 5, 1], [0, 7, 7], [3, 9, 1]])
    labels = np.array([1, 1, 0])
    k = runner_up(clean, labels)
    assert k.tolist() == [0, 2, 1]
    thr = tie_threshold(labels, k)
    assert thr.tolist() == [1, 0, 0]
    # margin 0 with label index lower than the rival is correct, as argmax decides
    m0 = clean[np.arange(3), labels] - clean[np.arange(3), k]
    assert ((m0 >= thr) == correct(clean, labels)).all()
