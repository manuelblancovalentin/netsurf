"""Deterministic tests for netsurf.mean_variance (Run P, Phase 0 prerequisites)."""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from netsurf.mean_variance import (
    bit_second_moments,
    code_jumps,
    displacement_moments,
    greedy_mean_variance_mask,
    log_uniform_rates,
    mask_sha256,
    top_budget,
    variance_term,
)


def test_code_jumps_are_additive_for_every_code_and_flip_set():
    codes = np.arange(-128, 128, dtype=np.int16).astype(np.int8)
    jumps = code_jumps(codes).astype(np.int32)                          # (256, 8)
    flips = np.arange(256, dtype=np.uint8)
    bits = ((flips[:, None] >> np.arange(8, dtype=np.uint8)) & 1).astype(np.int32)  # (256, 8)
    after = (codes.view(np.uint8)[:, None] ^ flips[None, :]).view(np.int8).astype(np.int32)
    change = after - codes.astype(np.int32)[:, None]                   # (codes, flip sets)
    predicted = jumps @ bits.T
    assert np.array_equal(change, predicted)


def test_code_jumps_known_examples():
    # code 5 = 0b00000101: bit 0 set (jump -1), bit 1 clear (+2), sign clear (-128)
    j = code_jumps(np.array([5], dtype=np.int8))[0]
    assert j[0] == -1 and j[1] == 2 and j[2] == -4 and j[7] == -128
    # code -128 = 0b10000000: sign set, flipping it gives 0 (+128)
    assert code_jumps(np.array([-128], dtype=np.int8))[0][7] == 128


@pytest.mark.parametrize("p", [1e-3, 0.05, 0.3])
def test_displacement_moments_match_enumeration(p):
    rng = np.random.default_rng(7)
    codes = rng.integers(-128, 128, size=12).astype(np.int8)
    scale = rng.uniform(0.001, 0.1, size=12)
    jumps = code_jumps(codes) * scale[:, None]
    protected = rng.random((12, 8)) < 0.4
    mean, var = displacement_moments(jumps, protected, p)
    states = np.array(list(itertools.product((0, 1), repeat=8)), dtype=bool)  # (256, 8)
    for i in range(12):
        k = states.sum(axis=1)
        free = (~protected[i]).sum()
        valid = ~(states & protected[i]).any(axis=1)
        prob = np.where(valid, p ** k * (1 - p) ** (free - k), 0.0)
        assert np.isclose(prob.sum(), 1.0)
        disp = states.astype(float) @ jumps[i]
        m = (prob * disp).sum()
        v = (prob * (disp - m) ** 2).sum()
        assert np.isclose(mean[i], m, rtol=1e-12, atol=1e-15)
        assert np.isclose(var[i], v, rtol=1e-10, atol=1e-15)


def test_quadratic_loss_expectation_is_exact():
    """For a quadratic loss the mean term plus the diagonal-curvature variance term equals the exact expectation."""
    rng = np.random.default_rng(11)
    codes = rng.integers(-128, 128, size=2).astype(np.int8)
    jumps = code_jumps(codes) * np.array([0.02, 0.05])[:, None]
    protected = np.zeros((2, 8), dtype=bool)
    protected[0, [7, 3]] = True
    protected[1, [6]] = True
    a = rng.normal(size=2)
    H = np.array([[2.0, 0.7], [0.7, 1.3]])                                # off-diagonal curvature on purpose
    loss = lambda d: float(a @ d + 0.5 * d @ H @ d)                       # noqa: E731  (displacement from theta)
    p = 0.2
    free = [np.flatnonzero(~protected[i]) for i in range(2)]
    exact = 0.0
    for s0 in itertools.product((0, 1), repeat=free[0].size):
        for s1 in itertools.product((0, 1), repeat=free[1].size):
            d = np.array([np.dot(s0, jumps[0, free[0]]), np.dot(s1, jumps[1, free[1]])])
            prob = p ** (sum(s0) + sum(s1)) * (1 - p) ** (len(s0) + len(s1) - sum(s0) - sum(s1))
            exact += prob * loss(d)
    mu, _ = displacement_moments(jumps, protected, p)
    predicted = loss(mu) + variance_term(bit_second_moments(jumps, np.diag(H)), protected, p)
    assert np.isclose(predicted, exact, rtol=1e-12)


def test_log_uniform_rates():
    r = log_uniform_rates(3e-4, 3e-2, 8)
    assert r.size == 8 and np.isclose(r[0], 3e-4) and np.isclose(r[-1], 3e-2)
    assert np.allclose(np.diff(np.log(r)), np.log(100) / 7)


def test_top_budget_ties_follow_storage_order():
    s = np.array([[1.0, 3.0], [3.0, 2.0], [3.0, 0.0]])
    m = top_budget(s, 2)
    assert m.tolist() == [[False, True], [True, False], [False, False]]


def _zero_mean_term(mu):
    return 0.0, np.zeros_like(mu)


def test_variance_only_greedy_reaches_top_budget_and_keeps_budget():
    rng = np.random.default_rng(3)
    n, budget = 200, 300
    jumps = code_jumps(rng.integers(-128, 128, size=n).astype(np.int8)) * rng.uniform(0.01, 0.1, size=n)[:, None]
    m2 = bit_second_moments(jumps, rng.uniform(0.1, 2.0, size=n))
    start = top_budget(rng.random((n, 8)), budget)
    result = greedy_mean_variance_mask(start, jumps, m2, [0.01], _zero_mean_term, min_swap=1, max_rounds=200)
    assert int(result.mask.sum()) == budget
    assert np.array_equal(result.mask, top_budget(m2, budget))
    objectives = [t["objective"] for t in result.trace if t["accepted"]]
    assert all(b < a for a, b in zip(objectives, objectives[1:]))


def test_mean_term_is_reduced_and_result_is_deterministic():
    """With a quadratic mean term 1/2 |mu|^2 the greedy mask lowers the objective below the variance-only mask."""
    rng = np.random.default_rng(5)
    n, budget = 150, 240
    codes = np.where(rng.random(n) < 0.7, 0, rng.integers(-128, 128, size=n)).astype(np.int8)  # lopsided, like pruning
    jumps = code_jumps(codes) * 0.05
    m2 = bit_second_moments(jumps, np.full(n, 0.05))

    def mean_term(mu):
        return float(0.5 * mu @ mu), mu.copy()

    rates = log_uniform_rates(1e-2, 1e-1, 4)
    start = top_budget(m2, budget)
    a = greedy_mean_variance_mask(start, jumps, m2, rates, mean_term, min_swap=1, max_rounds=100)
    b = greedy_mean_variance_mask(start, jumps, m2, rates, mean_term, min_swap=1, max_rounds=100)
    assert a.objective < a.initial_objective
    assert mask_sha256(a.mask) == mask_sha256(b.mask)
    assert int(a.mask.sum()) == budget
