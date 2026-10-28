from __future__ import annotations

import numpy as np
import pytest

from netsurf.affine_quantization import (
    AffineQuantization,
    affine_bit_transition,
    signed_xor,
)


def test_per_tensor_affine_dequantization_and_zero_point():
    quantization = AffineQuantization((0.25,), (-3,), 0)
    encoded = np.asarray([-128, -3, 0, 127], dtype=np.int8)
    assert np.array_equal(
        quantization.dequantize(encoded),
        np.asarray([-31.25, 0.0, 0.75, 32.5]),
    )


def test_per_channel_affine_dequantization_uses_quantized_dimension():
    quantization = AffineQuantization((0.5, 2.0), (0, -1), 1)
    encoded = np.asarray([[2, 2], [-2, -2]], dtype=np.int8)
    assert np.array_equal(
        quantization.dequantize(encoded),
        np.asarray([[1.0, 6.0], [-1.0, -2.0]]),
    )
    assert quantization.scalar_parameters(encoded.shape, 1) == (2.0, -1)


def test_signed_xor_is_exact_for_sign_and_multiple_bits():
    value = np.asarray([0], dtype=np.int8)
    assert int(signed_xor(value, 0, 7)[0]) == -128
    assert int(signed_xor(np.asarray([-1], dtype=np.int8), 0, 7)[0]) == 127
    changed = signed_xor(value, 0, 0)
    changed = signed_xor(changed, 0, 1)
    assert int(changed[0]) == 3

    wide = np.asarray([0], dtype=np.int32)
    assert int(signed_xor(wide, 0, 31)[0]) == -(1 << 31)
    with pytest.raises(IndexError):
        signed_xor(wide, 0, 32)


def test_affine_transition_reports_encoded_and_physical_states():
    quantization = AffineQuantization((0.125,), (-2,), 0)
    original, flipped, original_value, flipped_value, changed = affine_bit_transition(
        np.asarray([0], dtype=np.int8), quantization, 0, 7
    )
    assert (original, flipped) == (0, -128)
    assert original_value == 0.25
    assert flipped_value == -15.75
    assert int(changed[0]) == -128

