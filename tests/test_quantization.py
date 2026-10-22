import numpy as np

from netsurf.quantization import BitCodec, FixedPointFormat


def test_q621_contract():
    format_ = FixedPointFormat.parse("q<6,2,1>")
    assert format_.fractional_bits == 3
    assert format_.step == 0.125
    assert format_.value_min == -4.0
    assert format_.value_max == 3.875


def test_q621_all_codes_and_flips_round_trip():
    codec = BitCodec(FixedPointFormat.parse("q<6,2,1>"))
    codes = np.arange(64, dtype=np.uint64)
    values = codec.decode(codes)
    np.testing.assert_array_equal(codec.encode(values), codes)
    for bit in range(6):
        expected = codec.decode(codec.flip(codes, bit)) - values
        np.testing.assert_array_equal(codec.perturbation(values, bit), expected)


def test_q621_sign_bit_direction_is_twos_complement():
    codec = BitCodec(FixedPointFormat.parse("q<6,2,1>"))
    assert codec.perturbation(np.array([0.0]), 5).item() == -4.0
    assert codec.perturbation(np.array([-4.0]), 5).item() == 4.0


def test_rounding_is_ties_to_even_and_overflow_saturates():
    codec = BitCodec(FixedPointFormat.parse("q<6,2,1>"))
    values = np.array([0.0625, 0.1875, -0.0625, -0.1875, 99.0, -99.0])
    np.testing.assert_array_equal(
        codec.quantize(values),
        np.array([0.0, 0.25, -0.0, -0.25, 3.875, -4.0]),
    )
