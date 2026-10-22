"""Stable identities for logical model parameters and their encoded bits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Tuple


@dataclass(frozen=True, order=True)
class BitIdentity:
    """One logical bit in one scalar parameter.

    ``flat_index`` is C-order within the named parameter. ``bit_index`` is
    zero-based from the least-significant encoded bit. These conventions are
    explicit and never inferred from display labels.
    """

    parameter: str
    flat_index: int
    bit_index: int

    def key(self) -> str:
        return f"{self.parameter}[{self.flat_index}]#b{self.bit_index}"


@dataclass(frozen=True)
class ParameterSpec:
    """Immutable description of one fault-addressable tensor."""

    name: str
    shape: Tuple[int, ...]
    trainable: bool = True
    pruned_flat_indices: Tuple[int, ...] = ()

    @property
    def size(self) -> int:
        result = 1
        for dimension in self.shape:
            result *= dimension
        return result

    def bits(self, width: int, *, include_pruned: bool = True) -> Iterator[BitIdentity]:
        pruned = set(self.pruned_flat_indices)
        for flat_index in range(self.size):
            if not include_pruned and flat_index in pruned:
                continue
            for bit_index in range(width):
                yield BitIdentity(self.name, flat_index, bit_index)


def enumerate_bits(
    parameters: Iterable[ParameterSpec], width: int, *, include_pruned: bool = True
) -> tuple[BitIdentity, ...]:
    return tuple(
        bit
        for parameter in parameters
        for bit in parameter.bits(width, include_pruned=include_pruned)
    )
