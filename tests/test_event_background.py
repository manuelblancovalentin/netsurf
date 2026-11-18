"""Low-bit background and top-bit events: exact displacements and moments."""
import numpy as np

from netsurf import event_background as E


def test_bit_deltas_equal_the_xor_difference_for_every_code_and_bit():
    codes = np.arange(-128, 128, dtype=np.int8)
    d = E.bit_deltas(codes)
    for j in range(8):
        flipped = (codes.view(np.uint8) ^ np.uint8(1 << j)).view(np.int8)
        assert np.array_equal(d[:, j], flipped.astype(np.int64) - codes.astype(np.int64))


def test_low_and_top_parts_sum_to_the_full_displacement():
    rng = np.random.default_rng(0)
    codes = rng.integers(-128, 128, 500).astype(np.int8)
    flips = rng.random((7, 500, 8)) < 0.3
    low, top = E.split_displacement(codes, flips)
    xor = np.packbits(flips, axis=-1, bitorder="little")[..., 0]
    full = (codes.view(np.uint8)[None, :] ^ xor).view(np.int8).astype(np.int64) - codes.astype(np.int64)
    assert np.array_equal(low + top, full)
    assert np.abs(low).max() <= 63 and set(np.unique(np.abs(top))) <= {0, 64, 128, 192}


def test_moments_match_simulation_and_ignore_protected_bits():
    rng = np.random.default_rng(1)
    codes = rng.integers(-128, 128, 40).astype(np.int8)
    unprot = rng.random((40, 8)) < 0.7
    p = 0.05
    flips = (rng.random((40_000, 40, 8)) < p) & unprot[None]
    low, top = E.split_displacement(codes, flips)
    for part, bits in ((low, E.LOW_BITS), (top, E.TOP_BITS)):
        mu, var = E.moments(codes, unprot, p, bits)
        assert np.allclose(part.mean(0), mu, atol=4 * np.sqrt(var.max() / 40_000) + 1e-9)
        assert np.allclose(part.var(0), var, rtol=0.12, atol=1e-9)
    assert np.all(E.moments(codes, np.zeros((40, 8), bool), p, E.TOP_BITS)[1] == 0)


def test_variance_share_of_the_events_does_not_depend_on_the_rate():
    codes = np.random.default_rng(2).integers(-128, 128, 300).astype(np.int8)
    unprot = np.ones((300, 8), bool)
    shares = []
    for p in (1e-5, 1e-3, 1e-1):
        vl, vt = E.moments(codes, unprot, p, E.LOW_BITS)[1], E.moments(codes, unprot, p, E.TOP_BITS)[1]
        shares.append(vt.sum() / (vt.sum() + vl.sum()))
    assert np.allclose(shares, shares[0])
    assert np.isclose(shares[0], (64 ** 2 + 128 ** 2) / sum(4 ** j for j in range(8)))


def test_distances_from_normal():
    z = np.random.default_rng(3).standard_normal(400_000)
    k = E.excess_kurtosis(z.size, z.sum(), (z ** 2).sum(), (z ** 3).sum(), (z ** 4).sum())
    assert abs(k) < 0.05
    edges = np.linspace(-10, 10, 401)
    assert E.ks_to_normal(np.histogram(z, edges)[0], edges) < 0.005
    spike = np.zeros(400); spike[200] = 1.0                           # all mass at zero
    assert E.ks_to_normal(spike, edges) > 0.45
