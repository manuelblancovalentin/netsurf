"""Layer-local and Kronecker-factored finite-bit interaction diagnostics.

These are generalized Gauss--Newton approximations around a declared
background.  They are not exact finite-fault Hessians.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .identities import BitIdentity
from .topology import DenseReluDenseTopologyEvaluator


@dataclass(frozen=True)
class LayerCoordinate:
    layer: str
    input_index: int
    output_index: int
    is_bias: bool


@dataclass(frozen=True)
class KroneckerFactors:
    layer: str
    activation_factor: np.ndarray
    curvature_factor: np.ndarray
    sample_count: int
    implementation_version: str = "kronecker-ggn/v1"


@dataclass(frozen=True)
class StratifiedKroneckerFactors:
    layer: str
    factors: tuple[KroneckerFactors, ...]
    probabilities: tuple[float, ...]
    boundaries: tuple[tuple[float, float], ...]
    implementation_version: str = "confidence-stratified-kronecker-ggn/v1"


class DenseReluDenseCurvature:
    """Input-dependent local curvature for the supported two-layer topology."""

    implementation_version = "dense-relu-dense-local-ggn/v1"

    def __init__(self, topology: DenseReluDenseTopologyEvaluator):
        if topology.cache is None:
            raise ValueError("local curvature requires a retained topology cache")
        self.topology = topology
        self.cache = topology.cache
        logits = self.cache.logits.astype(np.float64)
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        self.probabilities = np.exp(shifted)
        self.probabilities /= np.sum(self.probabilities, axis=1, keepdims=True)
        self.output_q = (np.eye(logits.shape[1])[None, :, :] * self.probabilities[:, :, None]
                         - self.probabilities[:, :, None] * self.probabilities[:, None, :])
        output_weights = self.cache.values["dense1/kernel"].astype(np.float64)
        gate = (self.cache.dense0_pre > 0).astype(np.float64)
        raw_hidden = np.einsum("hc,ncd,kd->nhk", output_weights, self.output_q, output_weights, optimize=True)
        self.hidden_b = raw_hidden * gate[:, :, None] * gate[:, None, :]
        self.inputs = {
            "dense0": np.concatenate((self.cache.inputs_flat.astype(np.float64), np.ones((len(logits), 1))), axis=1),
            "dense1": np.concatenate((self.cache.dense0_activation.astype(np.float64), np.ones((len(logits), 1))), axis=1),
        }
        self._factor_cache: dict[str, KroneckerFactors] = {}
        self._stratified_cache: dict[tuple[str, int, int], StratifiedKroneckerFactors] = {}

    def coordinate(self, identity: BitIdentity) -> LayerCoordinate:
        response = self.topology.response(identity)
        item = response.topology
        width = self.inputs["dense0" if item.layer == "hidden" else "dense1"].shape[1]
        # The explicit branch avoids giving bias bits a fictitious source.
        if item.parameter_class == "bias":
            input_index = width - 1
        else:
            input_index = int(item.source_index)
        return LayerCoordinate("dense0" if item.layer == "hidden" else "dense1", input_index, item.target_index, item.parameter_class == "bias")

    def _b(self, layer: str) -> np.ndarray:
        return self.hidden_b if layer == "dense0" else self.output_q

    def local_per_sample(self, left: BitIdentity, right: BitIdentity) -> np.ndarray:
        a, b = self.coordinate(left), self.coordinate(right)
        if a.layer != b.layer:
            raise ValueError("layer-local curvature is undefined for cross-layer pairs")
        dl = self.topology.response(left).exact_delta
        dr = self.topology.response(right).exact_delta
        activations = self.inputs[a.layer]
        curvature = self._b(a.layer)
        return dl * dr * activations[:, a.input_index] * activations[:, b.input_index] * curvature[:, a.output_index, b.output_index]

    def local(self, left: BitIdentity, right: BitIdentity) -> float:
        return float(np.mean(self.local_per_sample(left, right)))

    def factors(self, layer: str, indices: Sequence[int] | None = None) -> KroneckerFactors:
        if layer not in self.inputs:
            raise ValueError(f"unsupported layer: {layer}")
        if indices is None and layer in self._factor_cache:
            return self._factor_cache[layer]
        selected = np.arange(len(self.cache.logits)) if indices is None else np.asarray(indices, dtype=np.int64)
        if len(selected) == 0:
            raise ValueError("Kronecker factor sample set is empty")
        h = self.inputs[layer][selected]
        b = self._b(layer)[selected]
        result = KroneckerFactors(layer, np.einsum("ni,nj->ij", h, h) / len(h), np.mean(b, axis=0), len(selected))
        if indices is None:
            self._factor_cache[layer] = result
        return result

    def kron(self, left: BitIdentity, right: BitIdentity, factors: KroneckerFactors | None = None) -> float:
        a, b = self.coordinate(left), self.coordinate(right)
        if a.layer != b.layer:
            raise ValueError("Kronecker layer factors do not define cross-layer interactions")
        factors = self.factors(a.layer) if factors is None else factors
        if factors.layer != a.layer:
            raise ValueError("Kronecker factors belong to another layer")
        dl = self.topology.response(left).exact_delta
        dr = self.topology.response(right).exact_delta
        return float(dl * dr * factors.activation_factor[a.input_index, b.input_index] * factors.curvature_factor[a.output_index, b.output_index])

    def covariance_error(self, left: BitIdentity, right: BitIdentity) -> float:
        a, b = self.coordinate(left), self.coordinate(right)
        if a.layer != b.layer:
            raise ValueError("covariance identity is layer local")
        dl = self.topology.response(left).exact_delta
        dr = self.topology.response(right).exact_delta
        hh = self.inputs[a.layer][:, a.input_index] * self.inputs[a.layer][:, b.input_index]
        bb = self._b(a.layer)[:, a.output_index, b.output_index]
        return float(dl * dr * np.mean((hh - np.mean(hh)) * (bb - np.mean(bb))))

    def stratified_factors(self, layer: str, strata: int, *, minimum_samples: int = 2) -> StratifiedKroneckerFactors:
        key = (layer, int(strata), int(minimum_samples))
        if key in self._stratified_cache:
            return self._stratified_cache[key]
        if strata < 1:
            raise ValueError("strata must be positive")
        confidence = np.max(self.probabilities, axis=1)
        order = np.argsort(confidence, kind="stable")
        groups = [group for group in np.array_split(order, strata) if len(group) >= minimum_samples]
        if not groups:
            raise ValueError("all confidence strata are undersampled")
        # Merge rejected tail samples into the nearest retained stratum so no
        # calibration observation is silently discarded.
        used = set(int(i) for group in groups for i in group)
        missing = [int(i) for i in order if int(i) not in used]
        if missing:
            merged = list(groups)
            for index in missing:
                nearest = int(np.argmin([abs(float(confidence[index]) - float(np.mean(confidence[group]))) for group in merged]))
                merged[nearest] = np.append(merged[nearest], index)
            groups = merged
        factors = tuple(self.factors(layer, group) for group in groups)
        probabilities = tuple(len(group) / len(confidence) for group in groups)
        boundaries = tuple((float(np.min(confidence[group])), float(np.max(confidence[group]))) for group in groups)
        result = StratifiedKroneckerFactors(layer, factors, probabilities, boundaries)
        self._stratified_cache[key] = result
        return result

    def stratified_kron(self, left: BitIdentity, right: BitIdentity, factors: StratifiedKroneckerFactors) -> float:
        return float(sum(weight * self.kron(left, right, factor) for weight, factor in zip(factors.probabilities, factors.factors)))
