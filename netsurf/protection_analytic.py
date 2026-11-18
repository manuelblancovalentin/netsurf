"""Closed-form residual error of protection mechanisms under independent stored-bit faults (side-effect free).

Every stored bit, redundancy included, flips independently with probability ``p``.  Each function returns the
probability that one protected data bit is still wrong after the mechanism acts, with the same repair rules as
``netsurf.nonideal_protection``:

* (2n+1)-modular redundancy: the majority vote is wrong when at least n+1 of the 2n+1 copies flip.
* A t-error-correcting code over ``k`` data bits with ``r`` check bits: a data bit stays wrong when it flips and at
  least t other bits of the codeword flip (miscorrection of more errors is ignored, as in the simulator).
  SEC-DED is t = 1 (the code also detects two errors, but detection does not repair them).
* ABFT per output channel of ``m`` weights of ``bits`` bits, fault-free checksums: a channel is repaired only when
  exactly one of its weights is faulty.

Storage costs are extra stored bits per protected data bit.  Voters, encoders, decoders, area and timing are not
modelled.
"""
from __future__ import annotations

from math import comb, ceil, log2


def binom_tail(n: int, k: int, p: float) -> float:
    """P(Binomial(n, p) >= k)."""
    if k <= 0:
        return 1.0
    return float(sum(comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(k, n + 1)))


def mr_residual(p: float, n: int) -> float:
    """(2n+1)-modular redundancy: wrong after voting when at least n+1 copies flip.  n=1 is TMR (3p^2 - 2p^3)."""
    return binom_tail(2 * n + 1, n + 1, p)


def mr_cost(n: int) -> float:
    return 2.0 * n


def bch_check_bits(k: int, t: int) -> int:
    """Check bits of a t-error-correcting, (t+1)-detecting extended binary BCH code over k data bits: t*m + 1 with
    m the smallest integer such that 2^m - 1 >= k + t*m.  For t = 1 this is the usual SEC-DED count
    (64 -> 8, 32 -> 7, 16 -> 6, 8 -> 5)."""
    m = 1
    while 2 ** m - 1 < k + t * m:
        m += 1
    return t * m + 1


def code_residual(p: float, k: int, t: int = 1, r: int | None = None) -> float:
    r = bch_check_bits(k, t) if r is None else r
    return p * binom_tail(k + r - 1, t, p)


def code_cost(k: int, t: int = 1, r: int | None = None) -> float:
    r = bch_check_bits(k, t) if r is None else r
    return r / k


def abft_residual(p: float, m: int, bits: int = 8) -> float:
    q = 1 - (1 - p) ** bits
    return p * (1 - (1 - q) ** (m - 1))


def abft_cost(m: int, bits: int = 8, checksum_bits: int = 64) -> float:
    return checksum_bits / (bits * m)


def expected_residual_faults(residual: float, bits: float) -> float:
    return residual * bits


def needed(p: float, bits: float, family: str, target: float = 1.0, k: int = 64, max_level: int = 12):
    """Smallest level of a family (n for MR, t for codes over k bits) whose expected residual faults over ``bits``
    protected bits is at most ``target``; returns (level, extra storage) or (None, None)."""
    for lvl in range(1, max_level + 1):
        if family == "mr":
            res, cost = mr_residual(p, lvl), mr_cost(lvl)
        else:
            res, cost = code_residual(p, k, lvl), code_cost(k, lvl)
        if res * bits <= target:
            return lvl, cost
    return None, None


__all__ = ["binom_tail", "mr_residual", "mr_cost", "bch_check_bits", "code_residual", "code_cost", "abft_residual",
           "abft_cost", "expected_residual_faults", "needed", "ceil", "log2"]
