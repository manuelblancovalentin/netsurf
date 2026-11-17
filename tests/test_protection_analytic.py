"""Closed-form residual errors agree with the bit-level simulators of netsurf.nonideal_protection."""
import numpy as np

from netsurf import nonideal_protection as N
from netsurf import protection_analytic as A


def test_tmr_closed_form():
    for p in (1e-3, 1e-2, 0.1):
        assert np.isclose(A.mr_residual(p, 1), N.tmr_residual(p))
        assert np.isclose(A.mr_residual(p, 2), 10 * p**3 * (1 - p)**2 + 5 * p**4 * (1 - p) + p**5)


def test_secded_check_bits_match_the_simulator():
    for k, r in N.SECDED_CHECK_BITS.items():
        assert A.bch_check_bits(k, 1) == r


def test_secded_residual_matches_simulation():
    rng = np.random.default_rng(1)
    p, k = 0.02, 32
    raw = N.raw_field(rng, 2_000_000, p)
    res = N.secded(raw, k, p, np.random.default_rng(2))
    measured = N.POP[res].sum() / (8 * raw.size)
    assert abs(measured - A.code_residual(p, k)) < 0.03 * A.code_residual(p, k)


def test_abft_residual_matches_simulation():
    rng = np.random.default_rng(3)
    p, m, ch = 0.003, 64, 20_000
    raw = N.raw_field(rng, m * ch, p)
    res = N.abft_ideal(raw, (ch, m), 0)
    measured = N.POP[res].sum() / (8 * raw.size)
    assert abs(measured - A.abft_residual(p, m)) < 0.05 * A.abft_residual(p, m)


def test_needed_level_grows_with_rate_and_size():
    small = A.needed(1e-4, 1e5, "mr")[0]
    large = A.needed(1e-2, 1e10, "mr")[0]
    assert small is not None and large is not None and large > small
