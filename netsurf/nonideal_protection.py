"""Stored-bit fault patterns under non-ideal protection schemes (side-effect free, numpy only).

A stored INT8 weight word is handled as a ``uint8``; a fault pattern is the ``uint8`` word XORed onto it.  Every stored
bit flips independently with probability ``p``.  This includes the redundant copies of triplicated bits and the
check bits of codes: no protected bit is immune.

    tmr                  protected bits are stored three times and voted: a voted bit is wrong with 3p^2 - 2p^3
    secded               a single-error-correcting, double-error-detecting code over consecutive data bits: a codeword
                         with exactly one flipped bit (data or check) is repaired, one with two or more is left as it
                         is (miscorrection of three or more errors is not modelled, which favours the code)
    abft_ideal           two checksums per output channel, assumed fault-free: a channel with exactly one faulty
                         weight is repaired, one with more is left as it is
    abft_faulty          the same decoder with its real failure modes: the two 32-bit checksum words flip too, and a
                         syndrome that points at a healthy weight causes a wrong repair
"""
from __future__ import annotations

import numpy as np

POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
SECDED_CHECK_BITS = {64: 8, 32: 7, 16: 6, 8: 5}


def tmr_residual(p: float) -> float:
    """Probability that the majority of three independent copies of a bit is wrong."""
    return 3.0 * p * p - 2.0 * p ** 3


def raw_field(rng: np.random.Generator, n: int, p: float, chunk: int = 1 << 21) -> np.ndarray:
    """XOR words of ``n`` stored bytes whose 8 bits flip independently with probability ``p``."""
    out = np.empty(n, dtype=np.uint8)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        out[s:e] = np.packbits(rng.random((e - s, 8)) < p, axis=1, bitorder="little")[:, 0]
    return out


def pack_mask(mask: np.ndarray) -> np.ndarray:
    """(n, 8) boolean bit mask, bit 0 least significant, to one ``uint8`` per word."""
    return np.packbits(np.asarray(mask, dtype=bool), axis=1, bitorder="little")[:, 0]


def tmr(raw: np.ndarray, voted: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Unprotected bits take the raw field; protected (``mask``) bits take the field of voted errors."""
    return (raw & ~mask) | (voted & mask)


def secded(raw: np.ndarray, data_bits: int, p: float, rng: np.random.Generator) -> np.ndarray:
    """SEC-DED over consecutive ``data_bits`` of one tensor (the last codeword is padded with clean bits)."""
    if data_bits % 8 or data_bits not in SECDED_CHECK_BITS:
        raise ValueError("data_bits must be one of 8, 16, 32, 64")
    per = data_bits // 8
    pad = (-raw.size) % per
    errors = POP[np.concatenate([raw, np.zeros(pad, dtype=np.uint8)])].reshape(-1, per).sum(axis=1, dtype=np.int64)
    errors = errors + rng.binomial(SECDED_CHECK_BITS[data_bits], p, size=errors.size)
    return np.where(np.repeat(errors >= 2, per)[: raw.size], raw, 0).astype(np.uint8)


def _channels(a: np.ndarray, shape: tuple[int, ...], axis: int) -> np.ndarray:
    return np.moveaxis(np.asarray(a).reshape(shape), axis, 0).reshape(shape[axis], -1)


def _unchannels(mat: np.ndarray, shape: tuple[int, ...], axis: int) -> np.ndarray:
    moved = (shape[axis],) + tuple(s for i, s in enumerate(shape) if i != axis)
    return np.moveaxis(mat.reshape(moved), 0, axis).reshape(-1)


def abft_ideal(raw: np.ndarray, shape: tuple[int, ...], axis: int) -> np.ndarray:
    """Fault-free checksums: channels (along ``axis``) with exactly one faulty weight are repaired."""
    mat = _channels(raw, shape, axis)
    faulty = (mat != 0).sum(axis=1)
    return _unchannels(np.where((faulty == 1)[:, None], 0, mat).astype(np.uint8), shape, axis)


def _wrap32(x: np.ndarray) -> np.ndarray:
    return ((np.asarray(x, dtype=np.int64) + (1 << 31)) % (1 << 32)) - (1 << 31)


def abft_faulty(codes: np.ndarray, raw: np.ndarray, shape: tuple[int, ...], axis: int, p: float,
                rng: np.random.Generator) -> np.ndarray:
    """Checksum decoder with faulty checksums and wrong repairs.

    Per channel of G weights with integer codes q_i: C1 = sum q_i and C2 = sum i q_i (i = 1..G), stored as 32-bit
    two's-complement words whose bits flip with probability ``p``.  With the faulty codes q' the decoder forms
    s1 = sum q'_i - C1' and s2 = sum i q'_i - C2' (32-bit arithmetic).  If s1 = 0 it does nothing.  If s2 / s1 is an
    integer l in 1..G it subtracts s1 from weight l (8-bit wraparound).  Otherwise it leaves the channel as it is.
    Returns the XOR pattern of the stored weights after decoding.
    """
    q = _channels(np.asarray(codes).view(np.int8), shape, axis).astype(np.int64)
    qf = _channels((np.asarray(codes).view(np.uint8) ^ raw).view(np.int8), shape, axis).astype(np.int64)
    C, G = q.shape
    idx = np.arange(1, G + 1, dtype=np.int64)
    weights = (1 << np.arange(32, dtype=np.int64))

    def stored(value):
        flips = ((rng.random((C, 32)) < p) * weights).sum(axis=1)
        return _wrap32((_wrap32(value) % (1 << 32)) ^ flips)

    c1, c2 = stored(q.sum(axis=1)), stored((q * idx).sum(axis=1))
    s1 = _wrap32(qf.sum(axis=1) - c1)
    s2 = _wrap32((qf * idx).sum(axis=1) - c2)
    safe = np.where(s1 == 0, 1, s1)
    ell = s2 // safe
    act = (s1 != 0) & (s2 % safe == 0) & (ell >= 1) & (ell <= G)
    rows = np.flatnonzero(act)
    out = qf.copy()
    out[rows, ell[rows] - 1] -= s1[rows]
    out = ((out + 128) % 256) - 128
    xor = out.astype(np.int8).view(np.uint8) ^ q.astype(np.int8).view(np.uint8)
    return _unchannels(xor, shape, axis)


def storage_cost(scheme: str, *, share: float = 0.0, data_bits: int = 0, channels: int = 0, weight_bits: int = 0) -> float:
    """Extra stored bits as a fraction of the weight bits."""
    if scheme == "unprotected":
        return 0.0
    if scheme == "tmr":
        return 2.0 * share
    if scheme == "secded":
        return SECDED_CHECK_BITS[data_bits] / data_bits
    if scheme == "abft":
        return 64.0 * channels / weight_bits
    raise ValueError(scheme)
