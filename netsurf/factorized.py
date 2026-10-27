"""Mask-aware full-population Kronecker interaction aggregates.

The matrix identities here are exact within the selected Kronecker GGN
surrogate. They are not exact finite-fault reliability interactions.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterable, Mapping, Sequence

import numpy as np

from .conditional_screening import bit_probability
from .curvature import DenseReluDenseCurvature, KroneckerFactors, StratifiedKroneckerFactors
from .identities import BitIdentity


@dataclass(frozen=True)
class AggregateTiming:
    factor_build_seconds: float
    aggregate_build_seconds: float
    query_seconds: float = 0.0
    update_seconds: float = 0.0


@dataclass
class _LayerAggregate:
    factors: tuple[KroneckerFactors, ...]
    weights: tuple[float, ...]
    moment: np.ndarray
    transformed: list[np.ndarray]


class FactorizedInteractionAggregate:
    """Dynamic within-layer interaction sum over every unprotected bit."""

    implementation_version = "full-vulnerable-population-kronecker-aggregate/v1"

    def __init__(
        self,
        curvature: DenseReluDenseCurvature,
        universe: Sequence[BitIdentity],
        protected: Iterable[BitIdentity],
        *,
        probabilities: float | Mapping[BitIdentity, float],
        stratified: bool = False,
        strata: int = 4,
        minimum_samples: int = 2,
        population: Iterable[BitIdentity] | None = None,
    ):
        factor_started = perf_counter()
        self.curvature = curvature; self.universe = tuple(universe); self.protected = set(protected)
        self.probabilities = probabilities
        if not self.protected <= set(self.universe):
            raise ValueError("protected mask is outside aggregate universe")
        self.population = frozenset(self.universe if population is None else population)
        if not self.population <= set(self.universe):
            raise ValueError("aggregate population is outside universe")
        self.stratified = bool(stratified)
        layer_factors: dict[str, tuple[tuple[KroneckerFactors, ...], tuple[float, ...]]] = {}
        for layer in ("dense0", "dense1"):
            if stratified:
                values = curvature.stratified_factors(layer, strata, minimum_samples=minimum_samples)
                layer_factors[layer] = (values.factors, values.probabilities)
            else:
                layer_factors[layer] = ((curvature.factors(layer),), (1.0,))
        factor_seconds = perf_counter() - factor_started
        aggregate_started = perf_counter(); self.layers: dict[str, _LayerAggregate] = {}
        for layer, (factors, weights) in layer_factors.items():
            moment = np.zeros((factors[0].curvature_factor.shape[0], factors[0].activation_factor.shape[0]), dtype=np.float64)
            for bit in self.population:
                coordinate = curvature.coordinate(bit)
                if coordinate.layer != layer or bit in self.protected:
                    continue
                delta = curvature.topology.response(bit).exact_delta
                moment[coordinate.output_index, coordinate.input_index] += bit_probability(bit, probabilities) * delta
            transformed = [factor.curvature_factor @ moment @ factor.activation_factor for factor in factors]
            self.layers[layer] = _LayerAggregate(factors, tuple(map(float, weights)), moment, transformed)
        self.timing = AggregateTiming(factor_seconds, perf_counter() - aggregate_started)

    def _self_term(self, bit: BitIdentity, factor: KroneckerFactors) -> float:
        coordinate = self.curvature.coordinate(bit); delta = self.curvature.topology.response(bit).exact_delta
        return (bit_probability(bit, self.probabilities) * delta * delta
                * factor.activation_factor[coordinate.input_index, coordinate.input_index]
                * factor.curvature_factor[coordinate.output_index, coordinate.output_index])

    def query(self, bit: BitIdentity) -> float:
        started = perf_counter(); coordinate = self.curvature.coordinate(bit); delta = self.curvature.topology.response(bit).exact_delta
        layer = self.layers[coordinate.layer]
        value = 0.0
        for weight, factor, transformed in zip(layer.weights, layer.factors, layer.transformed):
            item = delta * transformed[coordinate.output_index, coordinate.input_index]
            # Only this encoded event is excluded. Other bit positions in the
            # same scalar word remain distinct vulnerable events.
            if bit in self.population and bit not in self.protected:
                item -= self._self_term(bit, factor)
            value += weight * item
        self.timing = AggregateTiming(self.timing.factor_build_seconds, self.timing.aggregate_build_seconds,
                                      self.timing.query_seconds + perf_counter() - started, self.timing.update_seconds)
        return float(value)

    def explicit(self, bit: BitIdentity) -> float:
        coordinate = self.curvature.coordinate(bit); result = 0.0
        for other in self.population:
            if other == bit or other in self.protected or self.curvature.coordinate(other).layer != coordinate.layer:
                continue
            probability = bit_probability(other, self.probabilities)
            if self.stratified:
                layer = self.layers[coordinate.layer]
                interaction = sum(weight * self.curvature.kron(bit, other, factor)
                                  for weight, factor in zip(layer.weights, layer.factors))
            else:
                interaction = self.curvature.kron(bit, other, self.layers[coordinate.layer].factors[0])
            result += probability * interaction
        return float(result)

    def swap(self, removed: BitIdentity, added: BitIdentity) -> None:
        """Unprotect ``removed`` and protect ``added`` via rank-one updates."""
        if removed not in self.protected or added in self.protected:
            raise ValueError("aggregate swap requires protected removal and unprotected addition")
        started = perf_counter()
        for bit, sign in ((removed, 1.0), (added, -1.0)):
            if bit not in self.population:
                continue
            coordinate = self.curvature.coordinate(bit); delta = self.curvature.topology.response(bit).exact_delta
            coefficient = sign * bit_probability(bit, self.probabilities) * delta
            layer = self.layers[coordinate.layer]
            layer.moment[coordinate.output_index, coordinate.input_index] += coefficient
            for index, factor in enumerate(layer.factors):
                layer.transformed[index] += coefficient * np.outer(
                    factor.curvature_factor[:, coordinate.output_index],
                    factor.activation_factor[coordinate.input_index, :],
                )
        self.protected.remove(removed); self.protected.add(added)
        self.timing = AggregateTiming(self.timing.factor_build_seconds, self.timing.aggregate_build_seconds,
                                      self.timing.query_seconds, self.timing.update_seconds + perf_counter() - started)

    def reconstruction_error(self) -> float:
        """Rebuild every transformed matrix and return maximum drift."""
        maximum = 0.0
        for layer in self.layers.values():
            for factor, transformed in zip(layer.factors, layer.transformed):
                rebuilt = factor.curvature_factor @ layer.moment @ factor.activation_factor
                maximum = max(maximum, float(np.max(np.abs(transformed - rebuilt))))
        return maximum

    @property
    def population_coverage(self) -> float:
        return len(self.population) / len(self.universe)

