"""Deterministic tests for netsurf.rate_aware (Run R, Stage 0)."""
from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from netsurf.mean_variance import (bit_second_moments, code_jumps, displacement_moments, greedy_mean_variance_mask,
                                   top_budget, variance_term)
from netsurf.rate_aware import (COPIES_LADDER, IDEAL_LADDER, _int32_jumps, abft_decode, abft_pair_outcomes,
                                abft_residual, analytic_evaluator, descend, grouped_covariance_term, ladder_rates,
                                mc_displacement, mc_evaluator, mc_uniforms, nmr_residual, rate_moments,
                                rate_variance_term, secded_residual, shrink_mean_gradient, value_ranking_start)

RATES = (1e-4, 3e-3, 0.05, 0.3)


# ------------------------------------------------------------------------------------------- residual curves
@pytest.mark.parametrize("p", RATES)
@pytest.mark.parametrize("copies", (3, 5, 7))
def test_nmr_matches_enumeration(p, copies):
    exact = 0.0
    for pattern in itertools.product((0, 1), repeat=copies):
        k = sum(pattern)
        if k > copies // 2:
            exact += p ** k * (1 - p) ** (copies - k)
    assert nmr_residual(p, copies) == pytest.approx(exact, rel=1e-12)


def test_closed_forms_of_three_and_five_copies():
    for p in RATES:
        assert nmr_residual(p, 3) == pytest.approx(3 * p ** 2 - 2 * p ** 3, rel=1e-12)
        assert nmr_residual(p, 5) == pytest.approx(10 * p ** 3 - 15 * p ** 4 + 6 * p ** 5, rel=1e-12)


@pytest.mark.parametrize("p", RATES)
@pytest.mark.parametrize("n", (5, 9, 13))
def test_secded_matches_enumeration_of_a_short_word(p, n):
    """Bit 0 is a data bit; it stays wrong when it flipped and the word holds two or more errors."""
    exact = 0.0
    for pattern in itertools.product((0, 1), repeat=n):
        if pattern[0] == 1 and sum(pattern) >= 2:
            exact += p ** sum(pattern) * (1 - p) ** (n - sum(pattern))
    assert secded_residual(p, n) == pytest.approx(exact, rel=1e-12)


@pytest.mark.parametrize("p", (1e-3, 0.05, 0.2))
def test_abft_residual_matches_enumeration_of_a_small_channel(p):
    """G = 2 weights of 8 bits (16 bits): bit 0 of weight 1 stays wrong when it flipped and weight 2 is faulty;
    flips of other bits of weight 1 only merge into one faulty weight, which is repaired."""
    G = 2
    exact = 0.0
    for pattern in itertools.product((0, 1), repeat=8 * G):
        if pattern[0] == 1 and any(pattern[8:]):
            exact += p ** sum(pattern) * (1 - p) ** (8 * G - sum(pattern))
    assert abft_residual(p, G) == pytest.approx(exact, rel=1e-10)


def test_ladders():
    rho, step = ladder_rates(IDEAL_LADDER, 0.01)
    assert rho.tolist() == [0.01, 0.0] and step == 1.0
    rho, step = ladder_rates(COPIES_LADDER, 0.01)
    assert step == 2.0 and rho[1] == pytest.approx(3e-4 - 2e-6) and rho[2] < rho[1] < rho[0]


# ------------------------------------------------------------------------------------------- moments
def _toy(seed, n=40):
    rng = np.random.default_rng(seed)
    codes = rng.integers(-128, 128, size=n).astype(np.int8)
    jumps = code_jumps(codes) * rng.uniform(0.01, 0.1, size=n)[:, None]
    m2 = bit_second_moments(jumps, rng.uniform(0.1, 2.0, size=n))
    return rng, jumps, m2


def test_rate_moments_reduce_to_run_p_with_ideal_suppression():
    rng, jumps, m2 = _toy(1)
    mask = rng.random(jumps.shape) < 0.3
    rho, _ = ladder_rates(IDEAL_LADDER, 0.02)
    mu, var = rate_moments(jumps, mask.astype(np.int8), rho)
    mu0, var0 = displacement_moments(jumps, mask, 0.02)
    assert np.array_equal(mu, mu0) and np.array_equal(var, var0)
    assert rate_variance_term(m2, mask.astype(np.int8), rho) == variance_term(m2, mask, 0.02)


@pytest.mark.parametrize("p", (0.01, 0.2))
def test_rate_moments_match_enumeration_for_mixed_levels(p):
    jumps = code_jumps(np.array([-77], dtype=np.int8)).astype(np.float64)
    levels = np.array([[0, 1, 2, 0, 1, 2, 0, 1]], dtype=np.int8)
    rho, _ = ladder_rates(COPIES_LADDER, p)
    r = rho[levels[0]]
    mean = var = 0.0
    values = []
    for pattern in itertools.product((0, 1), repeat=8):
        prob = float(np.prod(np.where(np.array(pattern) == 1, r, 1 - r)))
        values.append((prob, float(np.dot(pattern, jumps[0]))))
    mean = sum(pr * v for pr, v in values)
    var = sum(pr * (v - mean) ** 2 for pr, v in values)
    mu, vv = rate_moments(jumps, levels, rho)
    assert mu[0] == pytest.approx(mean, rel=1e-12) and vv[0] == pytest.approx(var, rel=1e-10)


def test_shrinkage():
    out = shrink_mean_gradient(np.array([1.0, 1.0, -2.0, 0.0]), np.array([0.5, 2.0, 1.0, 1.0]))
    assert out.tolist() == pytest.approx([0.75, 0.0, -1.5, 0.0])


def test_grouped_covariance_term_matches_brute_force():
    """Two weights in one group, three images: compare with 1/2 E_x sum_{b != b'} Cov e_b e_b' summed directly."""
    rng = np.random.default_rng(4)
    G = rng.normal(size=(3, 2))
    jumps = code_jumps(np.array([5, -100], dtype=np.int8)).astype(np.float64) * 0.01
    cov_cross, cov_same = 0.3, -0.1
    e = G[:, :, None] * jumps[None]                                        # (images, weights, bits)
    e = e.reshape(3, -1)
    w = np.repeat([0, 1], 8)
    direct = 0.0
    for b in range(16):
        for bb in range(16):
            if b != bb:
                c = cov_same if w[b] == w[bb] else cov_cross
                direct += c * np.mean(e[:, b] * e[:, bb])
    got = grouped_covariance_term(G, jumps, np.array([0, 0]), cov_cross, cov_same)
    assert got == pytest.approx(0.5 * direct, rel=1e-10)


# ------------------------------------------------------------------------------------------- ABFT
def _brute_force_abft(q, checksum_bits=32):
    G = q.size
    jumps = code_jumps(q.astype(np.int8)).astype(np.int64)
    c1, c2 = int(q.sum()), int((np.arange(1, G + 1) * q).sum())
    out = dict(ww=dict(pairs=0, wrong=0, cancel=0), wc=dict(pairs=0, wrong=0, cancel=0))

    def judge(kind, faulty, c1f, c2f, touched):
        dec = abft_decode(faulty, c1f, c2f)
        changed = np.flatnonzero(dec != faulty)
        out[kind]["pairs"] += 1
        s1 = int(faulty.sum()) - c1f
        if s1 == 0:
            out[kind]["cancel"] += 1
        if changed.size and not set(changed.tolist()) <= set(touched):
            out[kind]["wrong"] += 1

    bits = [(i, a) for i in range(G) for a in range(8)]
    for (i, a), (k, b) in itertools.combinations(bits, 2):
        if i == k:
            continue
        f = q.astype(np.int64).copy()
        f[i] += jumps[i, a]; f[k] += jumps[k, b]
        judge("ww", f, c1, c2, (i, k))
    for (i, a) in bits:
        for which in (1, 2):
            for e in _int32_jumps(c1 if which == 1 else c2, checksum_bits):
                f = q.astype(np.int64).copy()
                f[i] += jumps[i, a]
                judge("wc", f, c1 + (e if which == 1 else 0), c2 + (e if which == 2 else 0), (i,))
    return out


@pytest.mark.parametrize("seed", (0, 1, 2))
def test_abft_pair_outcomes_match_a_brute_force_decoder(seed):
    q = np.random.default_rng(seed).integers(-128, 128, size=4)
    got = abft_pair_outcomes(q, checksum_bits=32)
    ref = _brute_force_abft(q)
    for kind in ("ww", "wc"):
        for key in ("pairs", "wrong", "cancel"):
            assert got[kind][key] == ref[kind][key], (kind, key)


def test_abft_single_faults_are_harmless():
    q = np.array([3, -7, 120, 0, 55], dtype=np.int64)
    c1, c2 = int(q.sum()), int((np.arange(1, 6) * q).sum())
    jumps = code_jumps(q.astype(np.int8)).astype(np.int64)
    for i in range(5):
        for a in range(8):
            f = q.copy(); f[i] += jumps[i, a]
            assert np.array_equal(abft_decode(f, c1, c2), q)
    for which in (1, 2):
        for e in _int32_jumps(c1 if which == 1 else c2):
            assert np.array_equal(abft_decode(q, c1 + (e if which == 1 else 0), c2 + (e if which == 2 else 0)), q)


# ------------------------------------------------------------------------------------------- MC
def test_mc_displacement_and_uniforms_are_deterministic_and_unbiased():
    _, jumps, _ = _toy(2, n=30)
    rho, _ = ladder_rates(COPIES_LADDER, 0.2)
    levels = np.random.default_rng(0).integers(0, 3, size=jumps.shape).astype(np.int8)
    u1, u2 = mc_uniforms(30, 4000, (7, 1)), mc_uniforms(30, 4000, (7, 1))
    assert np.array_equal(u1, u2)
    X = np.stack([mc_displacement(jumps, levels, rho, u) for u in u1])
    mu, var = rate_moments(jumps, levels, rho)
    se = np.sqrt(var / X.shape[0])
    assert np.all(np.abs(X.mean(0) - mu) < 5 * se + 1e-12)


def test_mc_evaluator_with_quadratic_loss_matches_analytic_in_expectation():
    """For L = 1/2 |x|^2 (unit curvature) J = 1/2 |mu|^2 + 1/2 sum var exactly; MC must agree within its error."""
    _, jumps, _ = _toy(3, n=25)
    m2 = jumps * jumps                                            # F = 1
    rho, _ = ladder_rates(COPIES_LADDER, 0.1)
    levels = np.random.default_rng(1).integers(0, 3, size=jumps.shape).astype(np.int8)

    def mean_term(x):
        return float(0.5 * x @ x), x.copy()
    a_val, _, _ = analytic_evaluator(jumps, m2, rho, mean_term)(levels)
    u = mc_uniforms(25, 3000, (9,))
    m_val, _, parts = mc_evaluator(jumps, m2, rho, mean_term, u)(levels)
    assert abs(m_val - a_val) < 5 * parts["mc_se"]


# ------------------------------------------------------------------------------------------- search
def test_descend_reproduces_run_p_greedy_with_ideal_suppression():
    rng = np.random.default_rng(5)
    n, budget = 150, 240
    codes = np.where(rng.random(n) < 0.7, 0, rng.integers(-128, 128, size=n)).astype(np.int8)
    jumps = code_jumps(codes) * 0.05
    m2 = bit_second_moments(jumps, np.full(n, 0.05))

    def mean_term(mu):
        return float(0.5 * mu @ mu + 0.3 * mu.sum()), mu.copy() + 0.3

    p = 0.03
    start = top_budget(m2, budget)
    ref = greedy_mean_variance_mask(start, jumps, m2, [p], mean_term, min_swap=1, max_rounds=100)
    rho, step = ladder_rates(IDEAL_LADDER, p)
    got = descend(start.astype(np.int8), jumps, m2, rho, step, analytic_evaluator(jumps, m2, rho, mean_term),
                  min_swap=1, max_rounds=100)
    assert np.array_equal(got.levels.astype(bool), ref.mask)
    assert got.objective == ref.objective
    assert [t["accepted"] for t in got.trace] == [t["accepted"] for t in ref.trace]


@pytest.mark.parametrize("anneal", (False, True))
def test_descend_keeps_storage_exact_and_is_deterministic(anneal):
    rng, jumps, m2 = _toy(6, n=60)

    def mean_term(mu):
        return float(0.5 * mu @ mu + 0.2 * mu.sum()), mu.copy() + 0.2
    rho, step = ladder_rates(COPIES_LADDER, 0.05)
    start = value_ranking_start(jumps, m2, rho, step, np.full(60, 0.2), 150)
    assert int(start.sum()) == 150 and start.max() <= 2
    ev = analytic_evaluator(jumps, m2, rho, mean_term)
    a = descend(start, jumps, m2, rho, step, ev, min_swap=1, max_rounds=60, anneal=anneal, seed=3)
    b = descend(start, jumps, m2, rho, step, ev, min_swap=1, max_rounds=60, anneal=anneal, seed=3)
    assert np.array_equal(a.levels, b.levels) and a.objective == b.objective
    assert int(a.levels.sum()) == 150 and a.levels.min() >= 0 and a.levels.max() <= 2
    assert a.objective <= a.initial_objective
    for t in a.trace:
        assert t["storage"] == 150 * step
        assert sum(t["level_counts"]) == jumps.size
    assert "v_chosen_up" in a.trace[1] and "drift" in a.trace[0]
