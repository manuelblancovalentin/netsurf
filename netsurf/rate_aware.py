"""Rate-aware protection assignment (Run R; code name ``rate_aware_mask``).

This is a new proposal that extends ``netsurf.mean_variance`` (Run P).  It is not qPolarGrad: the recovered legacy
expression is ``legacy_qpolargrad`` and is neither used nor redefined here.  The module is numpy-only and has no side
effects on import.  Notation follows ``RUN_R_RATE_AWARE_PLAN_2026-11-12.md`` (Section 2.0).

Protection is a *ladder* of levels with equal storage steps, for example ``none -> ideal`` (step 1 bit) or
``none -> 3 copies -> 5 copies`` (step 2 bits).  Every move changes one bit by one level, so a swap of ``m`` step-ups
against ``m`` step-downs preserves the storage budget exactly.  Level ``s`` has residual flip probability
``rho_s(p)``; bit ``b`` at level ``A(b)`` flips with ``r_b = rho_{A(b)}(p)``.

Objective (plan, Section 2.2):

    J_hat(A; p) = [L(theta + mu) - L(theta)] + 1/2 sum_w F_w V_w
    mu_w = sum_{b in w} r_b Delta_b,   V_w = sum_{b in w} r_b (1 - r_b) Delta_b^2

The first term is supplied by the caller through ``mean_term(mu) -> (L(theta+mu) - L(theta), dL/dw at theta+mu)``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

__all__ = [
    "Level", "IDEAL_LADDER", "COPIES_LADDER", "nmr_residual", "secded_residual", "abft_residual",
    "ladder_rates", "rate_moments", "rate_variance_term", "shrink_mean_gradient", "grouped_covariance_term",
    "abft_decode", "abft_pair_outcomes", "mc_uniforms", "mc_displacement", "analytic_evaluator", "mc_evaluator",
    "step_changes", "SearchResult", "descend", "value_ranking_start", "rate_aware_assignment",
    "per_input_evaluator", "select_on_heldout",
]

MeanTerm = Callable[[np.ndarray], "tuple[float, np.ndarray]"]
Evaluator = Callable[[np.ndarray], "tuple[float, np.ndarray, dict]"]


# ----------------------------------------------------------------------------------------------- residual curves
def nmr_residual(p: float, copies: int) -> float:
    """Probability that a majority vote over ``copies`` (odd) independent copies is wrong."""
    if copies < 1 or copies % 2 == 0:
        raise ValueError("copies must be a positive odd number")
    need = copies // 2 + 1
    return float(sum(math.comb(copies, k) * p ** k * (1.0 - p) ** (copies - k) for k in range(need, copies + 1)))


def secded_residual(p: float, n: int) -> float:
    """Data bit still wrong after SEC-DED over an ``n``-bit word (double errors left, miscorrection ignored)."""
    return float(p * (1.0 - (1.0 - p) ** (n - 1)))


def abft_residual(p: float, G: int, checksum_bits: int = 0) -> float:
    """Data bit still wrong after one-weight ABFT repair in a channel of ``G`` INT8 weights.

    ``checksum_bits = 0`` is the paper's model (checksums never flip).  With flipping checksum words, any checksum
    flip together with one faulty weight also prevents the correct repair, so the other bits are ``8(G-1) +
    checksum_bits``.  Wrong repairs are not part of this curve (see ``abft_pair_outcomes``)."""
    return float(p * (1.0 - (1.0 - p) ** (8 * (G - 1) + checksum_bits)))


@dataclass(frozen=True)
class Level:
    name: str
    cost: float                               # extra storage bits per data bit at this level
    residual: Callable[[float], float]        # rho_s(p)


IDEAL_LADDER = (Level("none", 0.0, lambda p: p), Level("ideal", 1.0, lambda p: 0.0))
COPIES_LADDER = (Level("none", 0.0, lambda p: p), Level("3 copies", 2.0, lambda p: nmr_residual(p, 3)),
                 Level("5 copies", 4.0, lambda p: nmr_residual(p, 5)))


def ladder_rates(ladder: Sequence[Level], p: float) -> tuple[np.ndarray, float]:
    """``rho_s(p)`` per level and the common storage step; raises if the steps are unequal."""
    costs = np.array([lv.cost for lv in ladder], dtype=np.float64)
    steps = np.diff(costs)
    if steps.size == 0 or not np.allclose(steps, steps[0]) or steps[0] <= 0:
        raise ValueError("a ladder needs at least two levels with equal, positive storage steps")
    return np.array([float(lv.residual(p)) for lv in ladder], dtype=np.float64), float(steps[0])


# ----------------------------------------------------------------------------------------------- moments
def rate_moments(jumps: np.ndarray, levels: np.ndarray, rho: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact mean and variance of each weight's displacement with per-bit rates ``rho[levels]``; shapes (n, 8)."""
    j = np.asarray(jumps, dtype=np.float64)
    mean, var = np.zeros(j.shape[0]), np.zeros(j.shape[0])
    for s, r in enumerate(rho):
        sel = levels == s
        mean = mean + r * np.where(sel, j, 0.0).sum(axis=1)
        var = var + r * (1.0 - r) * np.where(sel, j * j, 0.0).sum(axis=1)
    return mean, var


def rate_variance_term(m2: np.ndarray, levels: np.ndarray, rho: np.ndarray) -> float:
    """``1/2 sum_b r_b (1 - r_b) m2_b`` with ``m2_b = F_w Delta_b^2``."""
    total = 0.0
    for s, r in enumerate(rho):
        total += float(0.5 * r * (1.0 - r) * np.where(levels == s, m2, 0.0).sum())
    return total


def shrink_mean_gradient(gbar: np.ndarray, se: np.ndarray) -> np.ndarray:
    """``gbar * max(0, 1 - se^2 / gbar^2)``; zero where ``gbar`` is zero."""
    g = np.asarray(gbar, dtype=np.float64)
    s2 = np.asarray(se, dtype=np.float64) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = np.where(g != 0.0, np.maximum(0.0, 1.0 - s2 / (g * g)), 0.0)
    return g * factor


def grouped_covariance_term(per_example: np.ndarray, jumps: np.ndarray, weight_group: np.ndarray,
                            cov_cross: float, cov_same: float) -> float:
    """``1/2 E_x sum_{b != b', same group} Cov(xi_b, xi_b') e_{x,b} e_{x,b'}`` for one constant covariance between
    bits of different weights of a group (``cov_cross``) and one between bits of the same weight (``cov_same``).

    ``per_example`` is (images, weights) of gradients, ``jumps`` (weights, 8), ``weight_group`` (weights,) the group
    index of every weight (code word or ABFT channel), all for the bits under that scheme."""
    g = np.asarray(per_example, dtype=np.float64)
    J = np.asarray(jumps, dtype=np.float64)
    D, Q = J.sum(axis=1), (J * J).sum(axis=1)                     # sum and sum of squares of jumps per weight
    gD, g2Q, g2DD = g * D, g * g * Q, g * g * (D * D - Q)          # per image and weight
    groups = np.asarray(weight_group)
    ngroups = int(groups.max()) + 1
    s_all = np.zeros((g.shape[0], ngroups))
    np.add.at(s_all.T, groups, gD.T)
    sq = np.zeros((g.shape[0], ngroups))
    np.add.at(sq.T, groups, g2Q.T)
    all_pairs = (s_all ** 2 - sq).sum(axis=1)                       # every ordered pair b != b' within a group
    same_pairs = g2DD.sum(axis=1)                                   # ordered pairs within one weight
    cross_pairs = all_pairs - same_pairs
    return float(0.5 * np.mean(cov_cross * cross_pairs + cov_same * same_pairs))


# ----------------------------------------------------------------------------------------------- ABFT
def _wrap_int8(v):
    return ((np.asarray(v, dtype=np.int64) + 128) % 256) - 128


def abft_decode(q_faulty: np.ndarray, c1: int, c2: int) -> np.ndarray:
    """Decoder of plan Section 2.2 on one channel of integer codes (indices 1..G); ``c1``, ``c2`` are the stored
    (possibly faulty) checksums.  Returns the codes after the decoder, as int64 in the INT8 range."""
    q = np.asarray(q_faulty, dtype=np.int64).copy()
    G = q.size
    s1 = int(q.sum()) - int(c1)
    s2 = int((np.arange(1, G + 1) * q).sum()) - int(c2)
    if s1 != 0 and s2 % s1 == 0 and 1 <= s2 // s1 <= G:
        ell = s2 // s1
        q[ell - 1] = _wrap_int8(q[ell - 1] - s1)
    return q


def _int32_jumps(value: int, bits: int = 32) -> np.ndarray:
    """Change of a two's-complement ``bits``-bit word when each single bit flips."""
    u = value & ((1 << bits) - 1)
    out = np.empty(bits, dtype=np.int64)
    for d in range(bits):
        f = u ^ (1 << d)
        out[d] = (f - (1 << bits) if f >= 1 << (bits - 1) else f) - value
    return out


def abft_pair_outcomes(codes: np.ndarray, checksum_bits: int = 32) -> dict:
    """Outcome of every pair of single-bit faults in one ABFT channel (plan Section 2.2), counted exactly.

    Pairs counted: two data bits in different weights (``ww``); one data bit and one bit of either checksum word
    (``wc``).  For each kind: ``pairs``, ``wrong`` (decoder changes a healthy weight), ``cancel`` (sigma_1 = 0, left
    undetected) and, for wrong repairs, the per-target sums of the displacement and of its square (codes units)."""
    q = np.asarray(codes, dtype=np.int64).reshape(-1)
    G = q.size
    from .mean_variance import code_jumps                                        # local import keeps __init__ light
    d = code_jumps(q.astype(np.int8)).astype(np.int64).reshape(-1)               # (8G,)
    idx = np.repeat(np.arange(1, G + 1), 8)
    I, K = np.triu_indices(d.size, 1)
    keep = idx[I] != idx[K]
    I, K = I[keep], K[keep]
    out = dict(G=G)

    def classify(s1, s2, faulty_a, faulty_b, delta_target_base):
        nz = s1 != 0
        ok = np.zeros(s1.shape, dtype=bool)
        ok[nz] = s2[nz] % s1[nz] == 0
        ell = np.zeros(s1.shape, dtype=np.int64)
        ell[ok] = s2[ok] // s1[ok]
        wrong = ok & (ell >= 1) & (ell <= G) & (ell != faulty_a) & (ell != faulty_b)
        # a "repair" by a multiple of 256 wraps back to the same INT8 code and changes nothing
        pos = np.flatnonzero(wrong)
        changes = (_wrap_int8(q[ell[pos] - 1] - s1[pos]) - q[ell[pos] - 1]) != 0
        wrong[pos[~changes]] = False
        tgt = ell[wrong]
        before = q[tgt - 1]
        moved = _wrap_int8(before - s1[wrong]) - before
        sum_d = np.zeros(G); sum_d2 = np.zeros(G)
        np.add.at(sum_d, tgt - 1, moved.astype(np.float64))
        np.add.at(sum_d2, tgt - 1, moved.astype(np.float64) ** 2)
        return dict(pairs=int(s1.size), wrong=int(wrong.sum()), cancel=int((~nz).sum()),
                    sum_displacement=sum_d, sum_displacement2=sum_d2)

    s1 = d[I] + d[K]
    s2 = idx[I] * d[I] + idx[K] * d[K]
    out["ww"] = classify(s1, s2, idx[I], idx[K], None)
    if checksum_bits:
        c1 = int(q.sum()); c2 = int((np.arange(1, G + 1) * q).sum())
        e1, e2 = _int32_jumps(c1, checksum_bits), _int32_jumps(c2, checksum_bits)
        Dw, E1 = np.meshgrid(d, e1, indexing="ij"); Jw = np.repeat(idx, 1)[:, None] * np.ones_like(E1)
        r1 = classify((Dw - E1).ravel(), (Jw * Dw).ravel(), Jw.ravel(), Jw.ravel(), None)
        Dw, E2 = np.meshgrid(d, e2, indexing="ij"); Jw = np.repeat(idx, 1)[:, None] * np.ones_like(E2)
        r2 = classify(Dw.ravel(), (Jw * Dw - E2).ravel(), Jw.ravel(), Jw.ravel(), None)
        out["wc"] = {k: (r1[k] + r2[k]) for k in r1}
    return out


# ----------------------------------------------------------------------------------------------- MC correction
def mc_uniforms(n: int, K: int, seed: Sequence[int]) -> np.ndarray:
    """``K`` fields of one uniform number per stored data bit, shape (K, n, 8), from ``default_rng(seed)``."""
    return np.random.default_rng(list(seed)).random((K, n, 8), dtype=np.float32)


def mc_displacement(jumps: np.ndarray, levels: np.ndarray, rho: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Displacement of every weight in one field: bit ``b`` flips when ``u_b < r_b``."""
    r = np.asarray(rho, dtype=np.float64)[levels]
    return np.where(u < r, np.asarray(jumps, dtype=np.float64), 0.0).sum(axis=1)


def analytic_evaluator(jumps, m2, rho, mean_term: MeanTerm) -> Evaluator:
    """``A -> (J_hat, gradient at theta + mu, parts)``."""
    def evaluate(levels):
        mu, _ = rate_moments(jumps, levels, rho)
        loss_increase, grad = mean_term(mu)
        var = rate_variance_term(m2, levels, rho)
        value = 0.0
        value += float(loss_increase) + var
        return value / 1, np.asarray(grad, dtype=np.float64).reshape(-1), dict(
            drift=float(loss_increase), variance=var, mc_correction=0.0, mu_norm=float(np.linalg.norm(mu)))
    return evaluate


def mc_evaluator(jumps, m2, rho, mean_term: MeanTerm, uniforms: np.ndarray, refresh: int = 1) -> Evaluator:
    """``A -> (J_hat + C_hat, g_hat, parts)``.  Every ``refresh``-th call recomputes the correction on the assignment
    being evaluated: ``C_hat = mean_k[L(theta+X_k)] - L(theta+mu) - variance term`` and the gradient correction
    ``mean_k grad(theta+X_k) - grad(theta+mu)``; between refreshes both are reused (control-variate form).  With
    ``refresh = 1`` the value is the MC average loss increase and ``g_hat`` the MC mean gradient."""
    state = dict(calls=0, c=0.0, dg=None, c_se=float("nan"))

    def evaluate(levels):
        mu, _ = rate_moments(jumps, levels, rho)
        drift, grad = mean_term(mu)
        grad = np.asarray(grad, dtype=np.float64).reshape(-1)
        var = rate_variance_term(m2, levels, rho)
        if state["calls"] % refresh == 0:
            losses, grads = [], []
            for u in uniforms:
                lk, gk = mean_term(mc_displacement(jumps, levels, rho, u))
                losses.append(float(lk)); grads.append(np.asarray(gk, dtype=np.float64).reshape(-1))
            losses = np.array(losses)
            state["c"] = float(losses.mean() - drift - var)
            state["c_se"] = float(losses.std(ddof=1) / math.sqrt(losses.size)) if losses.size > 1 else float("nan")
            state["dg"] = np.mean(grads, axis=0) - grad
        state["calls"] += 1
        return float(drift) + var + state["c"], grad + state["dg"], dict(
            drift=float(drift), variance=var, mc_correction=state["c"], mc_se=state["c_se"],
            mu_norm=float(np.linalg.norm(mu)))
    return evaluate


def per_input_evaluator(jumps, rho, losses_term, gsq) -> Evaluator:
    """Validated variant (v2): ``A -> (J_hat, gradient at theta + mu, parts)`` with the objective kept per calibration
    input, ``J_x = [L_x(theta+mu) - L_x(theta)] + 1/2 sum_w g_{x,w}^2 V_w``, so that a swap can be accepted only when
    its gain is resolved over inputs (``descend(..., accept_z=z)``).  ``losses_term(mu)`` returns the per-input loss
    increase (N,) and the mean gradient at theta + mu; ``gsq`` (N, weights) holds the squared per-input gradients at
    theta.  The mean of ``J_x`` equals the analytic ``J_hat`` with ``F_w = mean_x g_{x,w}^2``."""
    gsq = np.asarray(gsq)

    def evaluate(levels):
        mu, V = rate_moments(jumps, levels, rho)
        per_loss, grad = losses_term(mu)
        per_var = 0.5 * (gsq @ np.asarray(V, dtype=gsq.dtype)).astype(np.float64)
        per_input = np.asarray(per_loss, dtype=np.float64) + per_var
        return float(per_input.mean()), np.asarray(grad, dtype=np.float64).reshape(-1), dict(
            drift=float(np.mean(per_loss)), variance=float(per_var.mean()), mc_correction=0.0,
            mu_norm=float(np.linalg.norm(mu)), per_input=per_input)
    return evaluate


def select_on_heldout(candidates: dict, score: Callable[[np.ndarray], float]) -> tuple[str, dict]:
    """Safeguard of the validated variant: score every candidate assignment on inputs that the search did not use
    and return the name of the lowest, with all scores."""
    scores = {k: float(score(v)) for k, v in candidates.items()}
    return min(scores, key=scores.get), scores


# ----------------------------------------------------------------------------------------------- search
def step_changes(jumps, m2, rho, step_cost, g, levels):
    """Predicted change of the objective per storage bit for one step up (``up``, +inf at the top level) and for one
    step down (``down``, +inf at level 0), from the first-order formula of plan Section 2.3; ``v_b = -up``."""
    J = np.asarray(jumps, dtype=np.float64)
    top = len(rho) - 1
    up = np.full(J.shape, np.inf)
    down = np.full(J.shape, np.inf)
    for s in range(top):
        rs, rt = float(rho[s]), float(rho[s + 1])
        dr = rs - rt
        change = np.zeros(J.shape)
        change += -dr * J * g[:, None] - 0.5 * dr * (1.0 - rs - rt) * m2
        change /= step_cost
        up = np.where(levels == s, change, up)
        down = np.where(levels == s + 1, -change, down)
    return up, down


def _ordered(values, candidates, count):
    v = values[candidates]
    return candidates[np.lexsort((candidates, v))[:count]]


def _sampled(values, candidates, count, T, rng):
    """``count`` candidates sampled without replacement with probability proportional to exp(-value / T)."""
    keys = -values[candidates] / T + rng.gumbel(size=candidates.size)
    return candidates[np.argsort(-keys, kind="stable")[:count]]


@dataclass
class SearchResult:
    levels: np.ndarray
    objective: float
    initial_objective: float
    trace: list = field(default_factory=list)


def _stats(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return dict(n=0, sum=0.0, mean=float("nan"), std=float("nan"))
    return dict(n=int(x.size), sum=float(x.sum()), mean=float(x.mean()), std=float(x.std()))


def descend(initial, jumps, m2, rho, step_cost, evaluate: Evaluator, *, max_rounds=40, min_swap=16,
            initial_swap=None, se=None, anneal=False, anneal_factor=0.9, seed=0, accept_z=None) -> SearchResult:
    """Budget-preserving swap descent (plan Section 2.3) from ``initial`` levels (n, 8).

    Each round scores every step up and down with the gradient returned by ``evaluate`` (shrunk with ``se`` when
    given), swaps the ``m`` best step-ups against the ``m`` cheapest step-downs, and accepts when the recomputed
    objective is lower; otherwise ``m`` is halved.  With ``anneal`` the moves are sampled with probability
    proportional to exp(v / T) and a worse objective is accepted with probability exp(-dJ / T); T is set by the first
    rejected step and multiplied by ``anneal_factor`` every round.  With ``accept_z`` (validated variant v2; needs an
    evaluator that returns ``per_input``) a swap is accepted only if the mean per-input change d of the objective is
    negative and mean(d) + accept_z * sd(d) / sqrt(N) < 0, so gains carried by a few calibration inputs are refused.  The best assignment seen is returned.  With
    two levels, ``anneal=False`` and ``se=None`` this is Run P's ``greedy_mean_variance_mask`` for one rate.

    The trace keeps, per round: objective and its parts, predicted change, swap size, temperature, storage used,
    count of bits per level, the rates rho, and summary statistics (sum, mean, std) of v_b over all possible
    step-ups and step-downs and over the chosen ones."""
    levels = np.array(initial, dtype=np.int8, copy=True)
    rho = np.asarray(rho, dtype=np.float64)
    budget_steps = int(levels.sum(dtype=np.int64))
    m = int(initial_swap) if initial_swap is not None else max(min_swap, budget_steps // 16)
    rng = np.random.default_rng(seed)
    current, g, parts = evaluate(levels)
    cur_inputs = parts.pop("per_input", None)
    if accept_z is not None and cur_inputs is None:
        raise ValueError("accept_z needs an evaluator that returns per_input")
    start, best, best_levels, T = current, current, levels.copy(), None
    counts = lambda L: [int((L == s).sum()) for s in range(len(rho))]  # noqa: E731
    trace = [dict(round=0, objective=current, accepted=True, swap=0, predicted=0.0, temperature=None,
                  storage=float(budget_steps * step_cost), level_counts=counts(levels), rho=rho.tolist(), **parts)]
    for rnd in range(1, max_rounds + 1):
        if m < min_swap:
            break
        gs = shrink_mean_gradient(g, se) if se is not None else g
        up, down = step_changes(jumps, m2, rho, step_cost, gs, levels)
        flat_up, flat_down = up.reshape(-1), down.reshape(-1)
        can_up = np.flatnonzero(np.isfinite(flat_up))
        k = min(m, can_up.size, int(np.isfinite(flat_down).sum()))
        if k == 0:
            break
        if anneal and T:
            gain = _sampled(flat_up, can_up, k, T, rng)
        else:
            gain = _ordered(flat_up, can_up, k)
        can_down = np.flatnonzero(np.isfinite(flat_down))
        can_down = np.setdiff1d(can_down, gain, assume_unique=True) if len(rho) > 2 else can_down
        k = min(k, can_down.size)
        gain = gain[:k]
        cost = _sampled(flat_down, can_down, k, T, rng) if (anneal and T) else _ordered(flat_down, can_down, k)
        predicted = float(flat_up[gain].sum() - (-flat_down)[cost].sum()) * step_cost
        row = dict(round=rnd, swap=int(k), predicted=predicted, temperature=T,
                   v_all_up=_stats(-flat_up), v_all_down=_stats(-flat_down),
                   v_chosen_up=_stats(-flat_up[gain]), v_chosen_down=_stats(-flat_down[cost]))
        if predicted >= 0.0 and not (anneal and T):
            trace.append(dict(row, objective=current, accepted=False, reason="no predicted improvement",
                              storage=float(levels.sum(dtype=np.int64) * step_cost), level_counts=counts(levels),
                              rho=rho.tolist()))
            m //= 2
            continue
        candidate = levels.reshape(-1).copy()
        candidate[gain] += 1
        candidate[cost] -= 1
        candidate = candidate.reshape(levels.shape)
        value, cand_g, cand_parts = evaluate(candidate)
        cand_inputs = cand_parts.pop("per_input", None)
        accepted = value < current
        if accept_z is not None:
            d = cand_inputs - cur_inputs
            bound = float(d.mean() + accept_z * d.std(ddof=1) / math.sqrt(d.size))
            row["gate_bound"] = bound
            accepted = bool(accepted and bound < 0)
        if not accepted and anneal:
            if T is None:
                T = float(value - current)
            elif rng.random() < math.exp(-(value - current) / T):
                accepted = True
        trace.append(dict(row, objective=value, accepted=bool(accepted), storage=float(candidate.sum(dtype=np.int64)
                          * step_cost), level_counts=counts(candidate), rho=rho.tolist(), **cand_parts))
        if accepted:
            levels, current, g = candidate, value, cand_g
            cur_inputs = cand_inputs
            if value < best:
                best, best_levels = value, candidate.copy()
        else:
            m //= 2
        if anneal and T:
            T *= anneal_factor
    if int(best_levels.sum(dtype=np.int64)) != budget_steps:
        raise AssertionError("budget not preserved")
    return SearchResult(levels=best_levels, objective=best, initial_objective=start, trace=trace)


def value_ranking_start(jumps, m2, rho, step_cost, g, budget_steps: int) -> np.ndarray:
    """Assignment from the values ``v_b`` at mu = 0 (gradient ``g`` at theta): steps taken in decreasing value, a
    bit's step s -> s+1 only after its step s-1 -> s; exactly ``budget_steps`` steps."""
    J = np.asarray(jumps, dtype=np.float64)
    levels = np.zeros(J.shape, dtype=np.int8)
    if len(rho) == 2:
        up, _ = step_changes(J, m2, rho, step_cost, g, levels)
        flat = up.reshape(-1)
        chosen = _ordered(flat, np.arange(flat.size), budget_steps)
        levels.reshape(-1)[chosen] = 1
        return levels
    import heapq
    values = []
    for s in range(len(rho) - 1):
        up, _ = step_changes(J, m2, rho, step_cost, g, np.full(J.shape, s, dtype=np.int8))
        values.append(up.reshape(-1))
    values = np.stack(values)                                 # (steps, bits): predicted change of step s -> s+1
    flat = levels.reshape(-1)
    heap = [(float(values[0, b]), b) for b in range(flat.size)]
    heapq.heapify(heap)
    taken = 0
    while taken < budget_steps:
        if not heap:
            raise AssertionError("could not place the budget")
        _, b = heapq.heappop(heap)
        flat[b] += 1
        taken += 1
        if flat[b] < values.shape[0]:
            heapq.heappush(heap, (float(values[flat[b], b]), b))
    return levels


def rate_aware_assignment(starts: Sequence[np.ndarray], jumps, m2, rho, step_cost, evaluate: Evaluator,
                          **kwargs) -> tuple[SearchResult, int]:
    """Descend from every start; return the result with the lowest objective and the index of its start."""
    results = [descend(s, jumps, m2, rho, step_cost, evaluate, **kwargs) for s in starts]
    i = int(np.argmin([r.objective for r in results]))
    return results[i], i
