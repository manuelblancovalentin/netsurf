"""Mask-aware full-population Kronecker interaction aggregates.

The matrix identities here are exact within the selected Kronecker GGN
surrogate. They are not exact finite-fault reliability interactions.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Iterable, Mapping, Sequence

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


@dataclass(frozen=True)
class FactorizedBitTable:
    """Immutable vectorized metadata for one encoded-bit universe."""

    identities: tuple[BitIdentity, ...]
    layers: np.ndarray
    input_indices: np.ndarray
    output_indices: np.ndarray
    deltas: np.ndarray
    build_seconds: float
    key: str
    implementation_version: str = "factorized-bit-table/v1"

    @classmethod
    def build(cls, curvature: DenseReluDenseCurvature, universe: Sequence[BitIdentity], *, key: str):
        started = perf_counter(); identities = tuple(universe)
        layers = np.empty(len(identities), dtype=np.int8)
        inputs = np.empty(len(identities), dtype=np.int32)
        outputs = np.empty(len(identities), dtype=np.int32)
        deltas = np.empty(len(identities), dtype=np.float64)
        for index, bit in enumerate(identities):
            coordinate = curvature.coordinate(bit)
            layers[index] = 0 if coordinate.layer == "dense0" else 1
            inputs[index] = coordinate.input_index
            outputs[index] = coordinate.output_index
            deltas[index] = curvature.topology.response(bit).exact_delta
        for value in (layers, inputs, outputs, deltas): value.flags.writeable = False
        return cls(identities, layers, inputs, outputs, deltas, perf_counter()-started, key)


@dataclass(frozen=True)
class FactorCalibration:
    """Calibration-dependent factors, independent of mask and logical BER."""

    factors: Mapping[str, tuple[KroneckerFactors, ...]]
    weights: Mapping[str, tuple[float, ...]]
    key: str
    stratified: bool
    build_seconds: float
    implementation_version: str = "factorized-calibration/v1"

    @classmethod
    def build(cls, curvature: DenseReluDenseCurvature, *, key: str, stratified: bool,
              strata: int, minimum_samples: int):
        started = perf_counter(); factors = {}; weights = {}
        for layer in ("dense0", "dense1"):
            if stratified:
                value = curvature.stratified_factors(layer, strata, minimum_samples=minimum_samples)
                factors[layer], weights[layer] = value.factors, tuple(map(float, value.probabilities))
            else:
                factors[layer], weights[layer] = (curvature.factors(layer),), (1.0,)
        return cls(factors, weights, key, bool(stratified), perf_counter()-started)


class FactorCalibrationCache:
    """Run-local exact-key cache; mismatched scientific inputs are misses."""

    def __init__(self):
        self.tables: dict[str, FactorizedBitTable] = {}
        self.calibrations: dict[str, FactorCalibration] = {}
        self.table_hits = self.table_misses = self.calibration_hits = self.calibration_misses = 0

    def table(self, key: str, builder: Callable[[], FactorizedBitTable]):
        if key in self.tables:
            self.table_hits += 1; return self.tables[key], True
        self.table_misses += 1; value = builder(); self.tables[key] = value; return value, False

    def calibration(self, key: str, builder: Callable[[], FactorCalibration]):
        if key in self.calibrations:
            self.calibration_hits += 1; return self.calibrations[key], True
        self.calibration_misses += 1; value = builder(); self.calibrations[key] = value; return value, False

    def stats(self):
        return {"bit_table_hits":self.table_hits,"bit_table_misses":self.table_misses,
                "calibration_hits":self.calibration_hits,"calibration_misses":self.calibration_misses}


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
        bit_table: FactorizedBitTable | None = None,
        calibration: FactorCalibration | None = None,
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
        self.bit_table = bit_table
        self.optimized = bit_table is not None or calibration is not None
        if self.bit_table is not None and self.bit_table.identities != self.universe:
            raise ValueError("factorized bit table does not match the aggregate universe")
        self._index = {bit:index for index,bit in enumerate(self.universe)}
        self._probability_values = (None if self.bit_table is None else
            np.asarray([bit_probability(bit, probabilities) for bit in self.universe], dtype=np.float64))
        self._population_mask = (None if self.bit_table is None else
            np.asarray([bit in self.population for bit in self.universe]))
        self._protected_mask = (None if self.bit_table is None else
            np.asarray([bit in self.protected for bit in self.universe]))
        layer_factors: dict[str, tuple[tuple[KroneckerFactors, ...], tuple[float, ...]]] = {}
        if calibration is not None:
            if calibration.stratified != self.stratified:
                raise ValueError("factor calibration stratification does not match aggregate")
            for layer in ("dense0", "dense1"):
                layer_factors[layer] = (calibration.factors[layer], calibration.weights[layer])
            factor_seconds = 0.0
        else:
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
            if self.bit_table is None:
                # Preserve the schema-4.3 scalar construction order exactly.
                for bit in self.population:
                    coordinate = curvature.coordinate(bit)
                    if coordinate.layer != layer or bit in self.protected:
                        continue
                    delta = curvature.topology.response(bit).exact_delta
                    moment[coordinate.output_index, coordinate.input_index] += bit_probability(bit, probabilities) * delta
            else:
                population_mask = self._population_mask & ~self._protected_mask
                layer_value = 0 if layer == "dense0" else 1
                selected = population_mask & (self.bit_table.layers == layer_value)
                np.add.at(moment, (self.bit_table.output_indices[selected], self.bit_table.input_indices[selected]),
                          self._probability_values[selected] * self.bit_table.deltas[selected])
            transformed = [factor.curvature_factor @ moment @ factor.activation_factor for factor in factors]
            self.layers[layer] = _LayerAggregate(factors, tuple(map(float, weights)), moment, transformed)
        self.timing = AggregateTiming(factor_seconds, perf_counter() - aggregate_started)

    def _self_term(self, bit: BitIdentity, factor: KroneckerFactors) -> float:
        coordinate = self.curvature.coordinate(bit)
        delta = (self.curvature.topology.response(bit).exact_delta if self.bit_table is None
                 else self.bit_table.deltas[self._index[bit]])
        return (bit_probability(bit, self.probabilities) * delta * delta
                * factor.activation_factor[coordinate.input_index, coordinate.input_index]
                * factor.curvature_factor[coordinate.output_index, coordinate.output_index])

    def query(self, bit: BitIdentity) -> float:
        if self.bit_table is None:
            started = perf_counter(); coordinate = self.curvature.coordinate(bit)
            delta = self.curvature.topology.response(bit).exact_delta
            layer = self.layers[coordinate.layer]; value = 0.0
            for weight, factor, transformed in zip(layer.weights, layer.factors, layer.transformed):
                item = delta * transformed[coordinate.output_index, coordinate.input_index]
                if bit in self.population and bit not in self.protected:
                    item -= self._self_term(bit, factor)
                value += weight * item
            self.timing = AggregateTiming(self.timing.factor_build_seconds, self.timing.aggregate_build_seconds,
                                          self.timing.query_seconds + perf_counter() - started, self.timing.update_seconds)
            return float(value)
        return float(self.query_many((bit,))[0])

    def query_many(self, bits: Sequence[BitIdentity]) -> np.ndarray:
        """Vectorized aggregate-score lookup preserving exact scalar semantics."""
        if self.bit_table is None:
            return np.asarray([self.query(bit) for bit in bits], dtype=np.float64)
        started=perf_counter(); bits=tuple(bits)
        indices=np.asarray([self._index[bit] for bit in bits],dtype=np.int64)
        values=np.zeros(len(bits),dtype=np.float64)
        for layer_name,layer_value in (("dense0",0),("dense1",1)):
            positions=np.flatnonzero(self.bit_table.layers[indices]==layer_value)
            if not len(positions): continue
            selected=indices[positions]; layer=self.layers[layer_name]
            for weight,factor,transformed in zip(layer.weights,layer.factors,layer.transformed):
                items=self.bit_table.deltas[selected]*transformed[self.bit_table.output_indices[selected],self.bit_table.input_indices[selected]]
                subtract=self._population_mask[selected] & ~self._protected_mask[selected]
                if np.any(subtract):
                    chosen=selected[subtract]; ii=self.bit_table.input_indices[chosen]; jj=self.bit_table.output_indices[chosen]
                    items[subtract]-=(self._probability_values[chosen]*self.bit_table.deltas[chosen]**2
                                      *factor.activation_factor[ii,ii]*factor.curvature_factor[jj,jj])
                values[positions]+=weight*items
        self.timing = AggregateTiming(self.timing.factor_build_seconds, self.timing.aggregate_build_seconds,
                                      self.timing.query_seconds + perf_counter() - started, self.timing.update_seconds)
        return values

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
            coordinate = self.curvature.coordinate(bit)
            delta = (self.curvature.topology.response(bit).exact_delta if self.bit_table is None
                     else self.bit_table.deltas[self._index[bit]])
            coefficient = sign * bit_probability(bit, self.probabilities) * delta
            layer = self.layers[coordinate.layer]
            layer.moment[coordinate.output_index, coordinate.input_index] += coefficient
            for index, factor in enumerate(layer.factors):
                layer.transformed[index] += coefficient * np.outer(
                    factor.curvature_factor[:, coordinate.output_index],
                    factor.activation_factor[coordinate.input_index, :],
                )
        self.protected.remove(removed); self.protected.add(added)
        if self._protected_mask is not None:
            self._protected_mask[self._index[removed]]=False; self._protected_mask[self._index[added]]=True
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
