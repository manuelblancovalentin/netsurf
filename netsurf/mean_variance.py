"""Mean-variance protection masks (Run P; proposed method, code name ``mean_variance_mask``).

This is a new proposal.  It is not qPolarGrad: the recovered legacy expression is ``legacy_qpolargrad`` and is
neither used nor redefined here.

Fault model.  Stored INT8 two's-complement codes; every eligible bit flips independently with probability ``p``
unless it is protected (ideal suppression).  In two's complement the code change caused by any set of flipped bits
of one word is the sum of the per-bit code jumps ``c_j = (1 - 2 a_j) 2^j`` for ``j < 7`` and
``c_7 = -(1 - 2 a_7) 128`` (``a_j`` the stored bit), so for weight ``i`` with decoded jumps ``Delta_b = s_i c_b``:

    mean displacement      mu_i  = p        * sum_{unprotected b of i} Delta_b
    displacement variance  var_i = p (1-p)  * sum_{unprotected b of i} Delta_b ** 2

Both are exact, and displacements of different weights are independent.

Objective.  A second-order expansion of the expected loss around the mean weights (the linear term vanishes there)
with the Fisher diagonal ``G2_i = E_x[g_i(x)^2]`` as the curvature gives

    J_p(P) = [L(theta + mu) - L(theta)] + 1/2 p (1-p) sum_{unprotected b} G2_i Delta_b ** 2 .

The first term is supplied by the caller (one forward and one backward pass of a differentiable model at the mean
weights); this module is numpy-only and has no side effects on import.  For a range of rates the objective is the
average of ``J_p`` over a fixed grid of rates.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

__all__ = [
    "code_jumps",
    "displacement_moments",
    "bit_second_moments",
    "variance_term",
    "log_uniform_rates",
    "top_budget",
    "mask_sha256",
    "MeanVarianceResult",
    "greedy_mean_variance_mask",
]

MeanTerm = Callable[[np.ndarray], "tuple[float, np.ndarray]"]


def code_jumps(codes: np.ndarray) -> np.ndarray:
    """Per-bit code jumps of INT8 codes, shape ``codes.shape + (8,)``, integer code units."""
    q = np.asarray(codes)
    if q.dtype != np.int8:
        raise TypeError("codes must be int8")
    bits = (q.view(np.uint8)[..., None] >> np.arange(8, dtype=np.uint8)) & 1
    sign = 1 - 2 * bits.astype(np.int16)
    weight = (2 ** np.arange(8, dtype=np.int16)).astype(np.int16)
    jumps = sign * weight
    jumps[..., 7] = -jumps[..., 7]
    return jumps


def displacement_moments(jumps: np.ndarray, protected: np.ndarray, p: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact mean and variance of each weight's displacement; ``jumps`` and ``protected`` have shape (n, 8)."""
    j = np.asarray(jumps, dtype=np.float64)
    keep = ~np.asarray(protected, dtype=bool)
    mean = p * np.where(keep, j, 0.0).sum(axis=1)
    var = p * (1.0 - p) * np.where(keep, j * j, 0.0).sum(axis=1)
    return mean, var


def bit_second_moments(jumps: np.ndarray, curvature: np.ndarray) -> np.ndarray:
    """``curvature_i * Delta_b**2`` per bit, shape (n, 8); with the Fisher diagonal this is ``E_x[(g_i Delta_b)^2]``."""
    j = np.asarray(jumps, dtype=np.float64)
    return np.asarray(curvature, dtype=np.float64)[:, None] * j * j


def variance_term(m2: np.ndarray, protected: np.ndarray, p: float) -> float:
    """``1/2 p (1-p) sum_{unprotected b} m2_b``."""
    keep = ~np.asarray(protected, dtype=bool)
    return float(0.5 * p * (1.0 - p) * np.where(keep, m2, 0.0).sum())


def log_uniform_rates(low: float, high: float, count: int) -> np.ndarray:
    """``count`` rates equally spaced in log between ``low`` and ``high`` (inclusive)."""
    if not (0.0 < low <= high < 1.0) or count < 1:
        raise ValueError("need 0 < low <= high < 1 and count >= 1")
    return np.exp(np.linspace(np.log(low), np.log(high), count))


def top_budget(score: np.ndarray, budget: int) -> np.ndarray:
    """Exactly ``budget`` highest scores; ties broken by flat storage order (lower index first)."""
    s = np.asarray(score, dtype=np.float64).reshape(-1)
    order = np.lexsort((np.arange(s.size), -s))
    mask = np.zeros(s.size, dtype=bool)
    mask[order[:budget]] = True
    return mask.reshape(np.shape(score))


def mask_sha256(mask: np.ndarray) -> str:
    return hashlib.sha256(np.packbits(np.asarray(mask, dtype=bool).reshape(-1), bitorder="little").tobytes()).hexdigest()


@dataclass
class MeanVarianceResult:
    mask: np.ndarray
    objective: float
    initial_objective: float
    trace: list[dict] = field(default_factory=list)


def _evaluate(mask, jumps, m2, rates, mean_term):
    """Objective averaged over rates, plus the per-rate gradients of the mean term at the mean weights."""
    total, grads = 0.0, []
    for p in rates:
        mu, _ = displacement_moments(jumps, mask, p)
        loss_increase, grad = mean_term(mu)
        grads.append(np.asarray(grad, dtype=np.float64).reshape(-1))
        total += float(loss_increase) + variance_term(m2, mask, p)
    return total / len(rates), grads


def _ordered(values: np.ndarray, candidates: np.ndarray, count: int) -> np.ndarray:
    """The ``count`` candidates with the smallest values; ties by flat index."""
    v = values[candidates]
    return candidates[np.lexsort((candidates, v))[:count]]


def greedy_mean_variance_mask(
    initial: np.ndarray,
    jumps: np.ndarray,
    m2: np.ndarray,
    rates: Sequence[float],
    mean_term: MeanTerm,
    *,
    max_rounds: int = 40,
    min_swap: int = 16,
    initial_swap: int | None = None,
) -> MeanVarianceResult:
    """Greedy budget-preserving swaps that lower the mean-variance objective.

    ``mean_term(mu)`` must return ``(L(theta + mu) - L(theta), dL/dw at theta + mu)`` for a per-weight mean
    displacement ``mu`` of shape (n,).  Each round predicts, to first order at the current mean weights, the change
    of the objective from protecting every unprotected bit and from releasing every protected bit; it swaps the
    ``m`` best additions with the ``m`` cheapest releases and accepts the swap only if the recomputed objective is
    lower; after a rejection ``m`` is halved.  The budget is preserved exactly and the procedure is deterministic.
    """
    mask = np.array(initial, dtype=bool, copy=True)
    jumps = np.asarray(jumps, dtype=np.float64)
    m2 = np.asarray(m2, dtype=np.float64)
    rates = tuple(float(r) for r in rates)
    if mask.shape != jumps.shape or m2.shape != jumps.shape or mask.ndim != 2 or mask.shape[1] != 8:
        raise ValueError("initial, jumps and m2 must all have shape (n, 8)")
    budget = int(mask.sum())
    m = int(initial_swap) if initial_swap is not None else max(min_swap, budget // 16)
    current, grads = _evaluate(mask, jumps, m2, rates, mean_term)
    start = current
    trace = [dict(round=0, objective=current, swap=0, accepted=True, predicted=0.0)]
    for rnd in range(1, max_rounds + 1):
        if m < min_swap:
            break
        add = np.zeros(jumps.shape)
        for p, g in zip(rates, grads):
            add += -p * jumps * g[:, None] - 0.5 * p * (1.0 - p) * m2
        add /= len(rates)
        flat_add = add.reshape(-1)
        flat_mask = mask.reshape(-1)
        free = np.flatnonzero(~flat_mask)
        held = np.flatnonzero(flat_mask)
        k = min(m, free.size, held.size)
        if k == 0:
            break
        gain = _ordered(flat_add, free, k)            # most negative predicted change when protected
        cost = _ordered(-flat_add, held, k)           # releasing a protected bit changes J by -add
        predicted = float(flat_add[gain].sum() - flat_add[cost].sum())
        if predicted >= 0.0:
            trace.append(dict(round=rnd, objective=current, swap=k, accepted=False, predicted=predicted,
                              reason="no predicted improvement"))
            m //= 2
            continue
        candidate = flat_mask.copy()
        candidate[gain] = True
        candidate[cost] = False
        candidate = candidate.reshape(mask.shape)
        value, cand_grads = _evaluate(candidate, jumps, m2, rates, mean_term)
        accepted = value < current
        trace.append(dict(round=rnd, objective=value, swap=k, accepted=bool(accepted), predicted=predicted))
        if accepted:
            mask, current, grads = candidate, value, cand_grads
        else:
            m //= 2
    if int(mask.sum()) != budget:
        raise AssertionError("budget not preserved")
    return MeanVarianceResult(mask=mask, objective=current, initial_objective=start, trace=trace)
