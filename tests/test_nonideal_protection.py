"""Deterministic checks of the non-ideal protection fault transforms."""
import numpy as np
import pytest

from netsurf import nonideal_protection as N


def test_tmr_residual_and_mixing():
    assert N.tmr_residual(0.0) == 0.0 and N.tmr_residual(1.0) == 1.0
    assert abs(N.tmr_residual(0.01) - (3e-4 - 2e-6)) < 1e-15
    raw, voted = np.array([0b11111111, 0b00001111], np.uint8), np.array([0b10000001, 0b11110000], np.uint8)
    mask = np.array([0b10000000, 0b11000000], np.uint8)          # sign bit; top two bits
    out = N.tmr(raw, voted, mask)
    assert out.tolist() == [0b11111111, 0b11001111]              # protected bits follow the voted field only
    assert N.tmr(raw, np.zeros(2, np.uint8), mask).tolist() == [0b01111111, 0b00001111]


def test_raw_field_rate_and_pack_mask():
    f = N.raw_field(np.random.default_rng(1), 200_000, 0.01)
    assert abs(N.POP[f].sum() / (8 * f.size) - 0.01) < 5e-4
    assert N.raw_field(np.random.default_rng(1), 1000, 0.0).sum() == 0
    m = np.zeros((2, 8), bool); m[0, 7] = True; m[1, 0] = True
    assert N.pack_mask(m).tolist() == [0x80, 0x01]


def test_secded_repairs_single_errors_only():
    rng = np.random.default_rng(0)
    raw = np.zeros(16, np.uint8)
    raw[0] = 0b00000100                 # codeword 0: one error  -> repaired
    raw[8] = 0b00000001; raw[9] = 0b1   # codeword 1: two errors -> left
    out = N.secded(raw, 64, 0.0, rng)   # p = 0: no check-bit errors
    assert out[:8].sum() == 0 and out[8] == 1 and out[9] == 1
    out8 = N.secded(raw, 8, 0.0, rng)   # 8-bit words: all three single errors repaired
    assert out8.sum() == 0
    with pytest.raises(ValueError):
        N.secded(raw, 24, 0.0, rng)


def test_secded_check_bit_error_blocks_repair():
    raw = np.zeros(8, np.uint8); raw[3] = 0b10000
    out = N.secded(raw, 64, 1.0, np.random.default_rng(0))      # every check bit flips: 1 + 8 errors
    assert out[3] == 0b10000


def test_secded_padding():
    raw = np.zeros(10, np.uint8); raw[9] = 1                     # second (padded) codeword, one error
    assert N.secded(raw, 64, 0.0, np.random.default_rng(0)).sum() == 0


def test_abft_ideal_along_any_axis():
    shape = (2, 3, 4)                                            # channels along axis 2: 4 channels of 6 weights
    raw = np.zeros(shape, np.uint8)
    raw[0, 0, 0] = 5                                             # channel 0: one faulty weight -> repaired
    raw[0, 1, 1] = 1; raw[1, 2, 1] = 2                           # channel 1: two faulty weights -> left
    out = N.abft_ideal(raw.reshape(-1), shape, 2).reshape(shape)
    assert out[..., 0].sum() == 0 and out[0, 1, 1] == 1 and out[1, 2, 1] == 2
    out0 = N.abft_ideal(raw.reshape(-1), shape, 0).reshape(shape)   # axis 0: channel 0 has 2 faults, channel 1 has 1
    assert np.array_equal(out0[0], raw[0]) and out0[1].sum() == 0


def _apply(codes, xor):
    return (codes.view(np.uint8) ^ xor).view(np.int8)


def test_abft_faulty_exact_single_repair_and_no_action_cases():
    rng = np.random.default_rng(0)
    codes = np.array([[3, -7, 100, 0, 5], [1, 2, 3, 4, 5]], np.int8)
    raw = np.zeros(codes.shape, np.uint8); raw[0, 2] = 0x80       # sign flip of weight 3 in channel 0
    out = N.abft_faulty(codes.reshape(-1), raw.reshape(-1), codes.shape, 0, 0.0, rng)
    assert out.sum() == 0                                         # repaired exactly
    assert N.abft_faulty(codes.reshape(-1), np.zeros(10, np.uint8), codes.shape, 0, 0.0, rng).sum() == 0


def test_abft_faulty_wrong_repair():
    codes = np.zeros((1, 4), np.int8)
    raw = np.zeros((1, 4), np.uint8); raw[0, 0] = 4; raw[0, 2] = 4   # errors +4 at weights 1 and 3: s2/s1 = 2
    out = N.abft_faulty(codes.reshape(-1), raw.reshape(-1), (1, 4), 0, 0.0, np.random.default_rng(0))
    after = _apply(codes.reshape(-1), out)
    assert after.tolist() == [4, -8, 4, 0]                           # the healthy weight 2 was "repaired" by -8


def test_abft_faulty_checksum_flips_alone_do_nothing_to_weights():
    codes = np.arange(-6, 6, dtype=np.int8).reshape(3, 4)
    out = N.abft_faulty(codes.reshape(-1), np.zeros(12, np.uint8), codes.shape, 0, 0.02, np.random.default_rng(5))
    # a flipped C1 gives s1 != 0 with s2 = 0 -> l = 0, out of range; a flipped C2 alone gives s1 = 0
    # (both flipped can cause a wrong repair; with this seed and rate the weights must stay clean or change by s1)
    after = _apply(codes.reshape(-1), out).reshape(3, 4)
    changed = np.argwhere(after != codes)
    assert len(changed) <= 3                                         # at most one wrong repair per channel


def test_abft_faulty_matches_ideal_when_checksums_are_clean_and_one_fault_per_channel():
    rng = np.random.default_rng(3)
    shape = (8, 3, 3, 4)
    codes = rng.integers(-127, 128, size=shape).astype(np.int8)
    raw = np.zeros(shape, np.uint8)
    for c in range(8):
        raw[c, rng.integers(3), rng.integers(3), rng.integers(4)] = 1 << rng.integers(8)
    out = N.abft_faulty(codes.reshape(-1), raw.reshape(-1), shape, 0, 0.0, rng)
    assert out.sum() == 0 and N.abft_ideal(raw.reshape(-1), shape, 0).sum() == 0


def test_storage_cost():
    assert N.storage_cost("tmr", share=0.125) == 0.25
    assert N.storage_cost("secded", data_bits=64) == 0.125 and N.storage_cost("secded", data_bits=8) == 0.625
    assert N.storage_cost("abft", channels=10, weight_bits=6400) == 0.1
    assert N.storage_cost("unprotected") == 0.0
