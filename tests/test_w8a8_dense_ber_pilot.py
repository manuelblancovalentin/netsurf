"""Bounded exact-storage tests for the independent packed Bernoulli pilot."""

import hashlib

import numpy as np
import pytest

from netsurf.w8a8_dense_ber_pilot import packed_field, xor_packed_codes


def test_packed_bernoulli_reproducible_and_write_once(tmp_path):
    a = tmp_path / "a.u8"
    b = tmp_path / "b.u8"
    first = packed_field(a, 800, 0.1, 7123, chunk_codes=7)
    second = packed_field(b, 800, 0.1, 7123, chunk_codes=7)
    assert a.read_bytes() == b.read_bytes()
    assert first == second
    assert first["sha256"] == hashlib.sha256(a.read_bytes()).hexdigest()
    assert first["proposed"] == sum(int(x).bit_count() for x in a.read_bytes())
    with pytest.raises(FileExistsError):
        packed_field(a, 800, 0.1, 7123)


def test_packed_mask_bits_are_exact_xor_and_restore():
    signed = np.asarray([-128, -1, 0, 1, 127], dtype=np.int8)
    original = signed.copy()
    packed = np.asarray([0b10000001, 0b11111111, 0b10000000, 0b00000011, 0b00000001],
                        dtype=np.uint8)
    xor_packed_codes(signed, packed)
    expected = np.bitwise_xor(original.view(np.uint8), packed)
    assert np.array_equal(signed.view(np.uint8), expected)
    assert signed.tolist() == [1, 0, -128, 2, 126]
    xor_packed_codes(signed, packed)
    assert np.array_equal(signed, original)


def test_dense_field_bits_have_correct_word_and_bit_layout(tmp_path):
    path = tmp_path / "field.u8"
    field = packed_field(path, 16, 0.5, 17, chunk_codes=1)
    rng = np.random.default_rng(17)
    expected = rng.random((2, 8), dtype=np.float64) < 0.5
    actual = np.unpackbits(np.frombuffer(path.read_bytes(), dtype=np.uint8), bitorder="little")
    assert np.array_equal(actual, expected.ravel())
    assert field["bytes"] == 2


def test_invalid_dense_field_or_storage_rejected(tmp_path):
    with pytest.raises(ValueError):
        packed_field(tmp_path / "bad.u8", 9, 0.1, 17)
    with pytest.raises(ValueError):
        xor_packed_codes(np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.uint8))
    with pytest.raises(ValueError):
        xor_packed_codes(np.zeros(2, dtype=np.int8), np.zeros(3, dtype=np.uint8))
