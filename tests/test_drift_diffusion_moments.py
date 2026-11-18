"""Deterministic checks of the exact two-moment description of stored-bit faults (exploratory prototype)."""
import importlib.util
import itertools
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "workflows/drift_diffusion_prototype.py"
spec = importlib.util.spec_from_file_location("drift_diffusion_prototype", SCRIPT)
dd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dd)


def test_bit_jumps_are_exact_twos_complement_changes():
    codes = np.arange(-128, 128, dtype=np.int8)
    jumps = dd.bit_jumps(codes)
    for bit in range(7):
        assert set(np.abs(jumps[:, bit])) == {1 << bit}
    assert np.array_equal(jumps[:, 7], np.where(codes >= 0, -128, 128))
    flipped = (codes.view(np.uint8) ^ np.uint8(1 << 3)).view(np.int8)
    assert np.array_equal(jumps[:, 3], flipped.astype(np.int16) - codes)


def test_moments_match_exhaustive_enumeration_of_fault_patterns():
    rng = np.random.default_rng(7)
    codes = rng.integers(-128, 128, size=12, dtype=np.int8)
    protected = rng.random((12, 8)) < 0.3
    jumps = dd.bit_jumps(codes)
    d, v = dd.unprotected_moments(jumps, protected)
    p = 0.07
    for k in range(codes.size):
        free = [b for b in range(8) if not protected[k, b]]
        mean = second = 0.0
        for pattern in itertools.product((0, 1), repeat=len(free)):
            prob = np.prod([p if x else 1 - p for x in pattern])
            mask = sum(1 << b for b, x in zip(free, pattern) if x)
            faulted = int((np.uint8(codes[k]) ^ np.uint8(mask)).astype(np.int8))
            delta = faulted - int(codes[k])
            mean += prob * delta
            second += prob * delta * delta
        assert abs(mean - p * d[k]) < 1e-9
        assert abs((second - mean * mean) - p * (1 - p) * v[k]) < 1e-9


def test_drift_without_protection_is_a_shrink_and_sign_protection_pushes_outward():
    codes = np.arange(-128, 128, dtype=np.int8)
    jumps = dd.bit_jumps(codes)
    none = np.zeros((256, 8), dtype=bool)
    d, _ = dd.unprotected_moments(jumps, none)
    assert np.array_equal(d, -1 - 2 * codes.astype(np.int32))           # E[q'] = (1 - 2p) q - p
    sign_only = none.copy(); sign_only[:, 7] = True
    d_msb, _ = dd.unprotected_moments(jumps, sign_only)
    assert np.all(d_msb[codes >= 0] == 127 - 2 * codes[codes >= 0].astype(np.int32))
    small_pos = (codes >= 0) & (codes < 64); small_neg = (codes < 0) & (codes >= -64)
    assert np.all(d_msb[small_pos] > 0) and np.all(d_msb[small_neg] < 0)  # away from zero for small weights
