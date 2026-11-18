"""Low-bit background and top-bit events of stored INT8 weight faults at one linear layer (side-effect free).

A stored INT8 code is linear in its bits, so the displacement of a weight under a set of flips is the sum of the
displacements of its low-order bits (0..5, the "background") and of its two top bits (6 and 7, the "events").  This
module gives the exact per-bit displacement, the split of a fault field into the two parts, the exact mean and variance
of each part under independent flips with probability ``p`` on the unprotected bits, and two distances of a sample of
standardized values from a standard normal.
"""
from __future__ import annotations

from math import erf, sqrt

import numpy as np

LOW_BITS = (0, 1, 2, 3, 4, 5)
TOP_BITS = (6, 7)


def bit_deltas(codes: np.ndarray) -> np.ndarray:
    """(n, 8) change of the signed INT8 value of each code when bit j is flipped (bit 0 least significant)."""
    u = np.asarray(codes).astype(np.int8).view(np.uint8).astype(np.int64)
    bit = (u[:, None] >> np.arange(8)[None, :]) & 1
    weight = np.array([1, 2, 4, 8, 16, 32, 64, -128], dtype=np.int64)       # two's complement place values
    return (1 - 2 * bit) * weight[None, :]


def split_displacement(codes: np.ndarray, flips: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Displacement of each weight due to its flipped low bits and to its flipped top bits.

    ``flips`` is a boolean array (..., n, 8); the two results have shape (..., n) and sum to the full displacement.
    """
    d = bit_deltas(codes)
    f = np.asarray(flips, dtype=bool)
    low = (f[..., LOW_BITS] * d[:, LOW_BITS]).sum(-1)
    top = (f[..., TOP_BITS] * d[:, TOP_BITS]).sum(-1)
    return low, top


def moments(codes: np.ndarray, unprotected: np.ndarray, p: float, bits) -> tuple[np.ndarray, np.ndarray]:
    """Exact mean and variance (n,) of the displacement from the given bit positions, each unprotected bit flipping
    independently with probability ``p`` and no protected bit flipping."""
    d = bit_deltas(codes)[:, list(bits)].astype(np.float64)
    u = np.asarray(unprotected, dtype=bool)[:, list(bits)]
    return p * (d * u).sum(1), p * (1 - p) * (d * d * u).sum(1)


def excess_kurtosis(n: float, s1: float, s2: float, s3: float, s4: float) -> float:
    """Excess kurtosis from the count and the first four power sums of a sample."""
    m = s1 / n
    v = s2 / n - m * m
    m4 = s4 / n - 4 * m * s3 / n + 6 * m * m * s2 / n - 3 * m ** 4
    return float(m4 / (v * v) - 3.0)


def ks_to_normal(counts: np.ndarray, edges: np.ndarray) -> float:
    """Largest gap between the empirical distribution of a histogram and the standard normal, at the bin edges."""
    c = np.asarray(counts, dtype=np.float64)
    cdf = np.cumsum(c) / c.sum()
    ref = np.array([0.5 * (1 + erf(e / sqrt(2))) for e in np.asarray(edges)[1:]])
    return float(np.max(np.abs(cdf - ref)))


__all__ = ["LOW_BITS", "TOP_BITS", "bit_deltas", "split_displacement", "moments", "excess_kurtosis", "ks_to_normal"]
