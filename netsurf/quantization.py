"""Exact fixed-point encoding and bit-flip semantics."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import numpy as np


_FORMAT = re.compile(r"^q<(?P<bits>\d+),(?P<integer>\d+),(?P<signed>[01])>$")


@dataclass(frozen=True)
class FixedPointFormat:
    """Saturating fixed-point format used by historical NetSuRF.

    ``integer_bits`` excludes the sign bit for signed formats, matching the
    recovered NetSuRF/QKeras notation. Rounding uses NumPy's ties-to-even rule.
    """

    total_bits: int
    integer_bits: int
    signed: bool

    def __post_init__(self) -> None:
        if self.total_bits <= 0:
            raise ValueError("total_bits must be positive")
        sign_bits = 1 if self.signed else 0
        if self.integer_bits < 0 or self.integer_bits + sign_bits > self.total_bits:
            raise ValueError("integer_bits and sign bit exceed total_bits")

    @classmethod
    def parse(cls, value: str) -> "FixedPointFormat":
        match = _FORMAT.fullmatch(value.strip())
        if not match:
            raise ValueError(f"invalid fixed-point format: {value!r}")
        return cls(
            total_bits=int(match.group("bits")),
            integer_bits=int(match.group("integer")),
            signed=bool(int(match.group("signed"))),
        )

    @property
    def fractional_bits(self) -> int:
        return self.total_bits - self.integer_bits - (1 if self.signed else 0)

    @property
    def step(self) -> float:
        return 2.0 ** (-self.fractional_bits)

    @property
    def code_min(self) -> int:
        return -(1 << (self.total_bits - 1)) if self.signed else 0

    @property
    def code_max(self) -> int:
        return (1 << (self.total_bits - 1)) - 1 if self.signed else (1 << self.total_bits) - 1

    @property
    def value_min(self) -> float:
        return self.code_min * self.step

    @property
    def value_max(self) -> float:
        return self.code_max * self.step

    def bit_exponent(self, bit_index: int) -> int:
        if not 0 <= bit_index < self.total_bits:
            raise IndexError(bit_index)
        return bit_index - self.fractional_bits

    def to_dict(self) -> dict[str, Any]:
        return {
            "notation": str(self),
            "total_bits": self.total_bits,
            "integer_bits_excluding_sign": self.integer_bits,
            "fractional_bits": self.fractional_bits,
            "signed": self.signed,
            "rounding": "nearest_ties_to_even",
            "overflow": "saturate",
            "encoded_bit_order": "lsb_zero",
            "range": [self.value_min, self.value_max],
            "step": self.step,
        }

    def __str__(self) -> str:
        return f"q<{self.total_bits},{self.integer_bits},{int(self.signed)}>"


class BitCodec:
    """Vectorized two's-complement codec with exact XOR bit flips."""

    def __init__(self, format_: FixedPointFormat):
        self.format = format_
        self._mask = (1 << format_.total_bits) - 1

    def quantize(self, values: Any) -> np.ndarray:
        values_array = np.asarray(values, dtype=np.float64)
        codes = np.rint(values_array / self.format.step)
        codes = np.clip(codes, self.format.code_min, self.format.code_max)
        return (codes * self.format.step).astype(np.float64)

    def encode(self, values: Any) -> np.ndarray:
        quantized_codes = np.rint(self.quantize(values) / self.format.step).astype(np.int64)
        return np.bitwise_and(quantized_codes, self._mask).astype(np.uint64)

    def decode(self, encoded: Any) -> np.ndarray:
        raw = np.bitwise_and(np.asarray(encoded, dtype=np.uint64), np.uint64(self._mask))
        signed_codes = raw.astype(np.int64)
        if self.format.signed:
            sign = 1 << (self.format.total_bits - 1)
            signed_codes = np.where((signed_codes & sign) != 0, signed_codes - (1 << self.format.total_bits), signed_codes)
        return signed_codes.astype(np.float64) * self.format.step

    def flip(self, encoded: Any, bit_index: int) -> np.ndarray:
        self.format.bit_exponent(bit_index)
        return np.bitwise_xor(np.asarray(encoded, dtype=np.uint64), np.uint64(1 << bit_index))

    def perturbation(self, values: Any, bit_index: int) -> np.ndarray:
        encoded = self.encode(values)
        return self.decode(self.flip(encoded, bit_index)) - self.decode(encoded)

    def perturbation_matrix(self, values: Any) -> np.ndarray:
        encoded = self.encode(values)
        return np.stack(
            [self.decode(self.flip(encoded, bit)) - self.decode(encoded) for bit in range(self.format.total_bits)],
            axis=-1,
        )
