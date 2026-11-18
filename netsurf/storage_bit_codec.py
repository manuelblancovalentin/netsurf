"""Versioned exact stored-word XOR for IEEE binary32 and bfloat16.

Words are little-endian byte sequences.  This codec changes encoded storage,
never a floating-point approximation of the encoded transition.  INT8 keeps
using :mod:`netsurf.quantization` and is deliberately untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct


@dataclass(frozen=True)
class StorageBitCodec:
    precision: str
    version: str = "stored-xor/v1"

    def __post_init__(self) -> None:
        if self.precision not in {"FP32", "BF16"} or self.version != "stored-xor/v1":
            raise ValueError("only FP32/BF16 stored-xor/v1 is validated")

    @property
    def width(self) -> int:
        return 32 if self.precision == "FP32" else 16

    @property
    def byte_width(self) -> int:
        return self.width // 8

    def xor(self, raw: bytes, flat_index: int, bit_index: int) -> bytes:
        if len(raw) % self.byte_width:
            raise ValueError("storage byte count is not a whole number of words")
        if not 0 <= flat_index < len(raw) // self.byte_width:
            raise IndexError(flat_index)
        if not 0 <= bit_index < self.width:
            raise IndexError(bit_index)
        result = bytearray(raw)
        result[flat_index * self.byte_width + bit_index // 8] ^= 1 << (bit_index % 8)
        return bytes(result)

    def fields(self, raw_word: bytes) -> dict[str, int | str]:
        if len(raw_word) != self.byte_width:
            raise ValueError("expected one encoded word")
        word = int.from_bytes(raw_word, "little")
        mantissa_bits = 23 if self.precision == "FP32" else 7
        exponent = (word >> mantissa_bits) & 0xff
        mantissa = word & ((1 << mantissa_bits) - 1)
        category = ("nan" if mantissa else "infinity") if exponent == 255 else (
            ("subnormal" if mantissa else "zero") if exponent == 0 else "normal"
        )
        return {"sign": word >> (self.width - 1), "exponent": exponent,
                "significand": mantissa, "category": category}

    def value(self, raw_word: bytes) -> float:
        self.fields(raw_word)
        if self.precision == "FP32":
            return struct.unpack("<f", raw_word)[0]
        return struct.unpack("<f", b"\x00\x00" + raw_word)[0]

    def finite_displacement(self, raw_word: bytes, bit_index: int) -> float | None:
        """Return an ordering score only when both encoded values are finite."""
        original = self.value(raw_word)
        changed = self.value(self.xor(raw_word, 0, bit_index))
        if not math.isfinite(original) or not math.isfinite(changed):
            return None
        return abs(changed - original)
