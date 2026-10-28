"""Exact affine integer storage semantics for modern quantized runtimes.

The historical :class:`~netsurf.quantization.BitCodec` models one fixed-point
format shared by every tensor.  TFLite models instead store signed integer
codes plus per-tensor or per-channel affine metadata.  This module keeps those
representations separate and never treats scales or zero points as faultable
parameter bits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


_SIGNED_DTYPES = {np.dtype("int8"), np.dtype("int16"), np.dtype("int32"), np.dtype("int64")}


@dataclass(frozen=True)
class AffineQuantization:
    """TFLite affine dequantization metadata for one stored tensor."""

    scales: tuple[float, ...]
    zero_points: tuple[int, ...]
    quantized_dimension: int

    def __post_init__(self) -> None:
        if not self.scales or any(not np.isfinite(value) or value <= 0 for value in self.scales):
            raise ValueError("affine scales must be finite and positive")
        if len(self.zero_points) not in {1, len(self.scales)}:
            raise ValueError("zero_points must be scalar or match scales")
        if self.quantized_dimension < 0:
            raise ValueError("quantized_dimension must be nonnegative")

    def _broadcast(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(self.scales) == 1:
            return np.asarray(self.scales[0]), np.asarray(self.zero_points[0])
        if self.quantized_dimension >= values.ndim:
            raise ValueError("quantized_dimension exceeds tensor rank")
        if values.shape[self.quantized_dimension] != len(self.scales):
            raise ValueError("per-channel scale count does not match tensor shape")
        shape = [1] * values.ndim
        shape[self.quantized_dimension] = len(self.scales)
        scales = np.asarray(self.scales, dtype=np.float64).reshape(shape)
        zero_points = np.asarray(
            self.zero_points if len(self.zero_points) > 1 else self.zero_points * len(self.scales),
            dtype=np.float64,
        ).reshape(shape)
        return scales, zero_points

    def dequantize(self, encoded: Any) -> np.ndarray:
        values = np.asarray(encoded)
        if values.dtype not in _SIGNED_DTYPES:
            raise TypeError(f"affine encoded values must use a signed integer dtype, got {values.dtype}")
        scales, zero_points = self._broadcast(values)
        return (values.astype(np.float64) - zero_points) * scales

    def scalar_parameters(self, shape: tuple[int, ...], flat_index: int) -> tuple[float, int]:
        if not 0 <= flat_index < int(np.prod(shape, dtype=np.int64)):
            raise IndexError(flat_index)
        if len(self.scales) == 1:
            return self.scales[0], self.zero_points[0]
        coordinate = np.unravel_index(flat_index, shape, order="C")
        channel = coordinate[self.quantized_dimension]
        zero_point = self.zero_points[channel] if len(self.zero_points) > 1 else self.zero_points[0]
        return self.scales[channel], zero_point

    def to_dict(self) -> dict[str, object]:
        return {
            "scheme": "affine",
            "scales": list(self.scales),
            "zero_points": list(self.zero_points),
            "quantized_dimension": self.quantized_dimension,
        }


def signed_xor(values: Any, flat_index: int, bit_index: int) -> np.ndarray:
    """Return a copy after XOR of one bit in one signed integer scalar."""

    source = np.asarray(values)
    if source.dtype not in _SIGNED_DTYPES:
        raise TypeError(f"XOR storage requires a signed integer dtype, got {source.dtype}")
    width = source.dtype.itemsize * 8
    if not 0 <= bit_index < width:
        raise IndexError(bit_index)
    if not 0 <= flat_index < source.size:
        raise IndexError(flat_index)
    result = np.ascontiguousarray(source).copy()
    unsigned = result.view(np.dtype(f"uint{width}")).reshape(-1)
    unsigned[flat_index] ^= np.array(1 << bit_index, dtype=unsigned.dtype)
    return result


def affine_bit_transition(
    values: Any,
    quantization: AffineQuantization,
    flat_index: int,
    bit_index: int,
) -> tuple[int, int, float, float, np.ndarray]:
    """Return encoded and dequantized states for one exact XOR transition."""

    source = np.asarray(values)
    changed = signed_xor(source, flat_index, bit_index)
    original_code = int(source.reshape(-1)[flat_index])
    changed_code = int(changed.reshape(-1)[flat_index])
    scale, zero_point = quantization.scalar_parameters(tuple(source.shape), flat_index)
    original_value = (original_code - zero_point) * scale
    changed_value = (changed_code - zero_point) * scale
    return original_code, changed_code, original_value, changed_value, changed
