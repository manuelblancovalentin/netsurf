"""Exact topology-aware finite-fault responses for Dense--ReLU--Dense models.

This module is deliberately architecture specific.  It shares encoded-state
semantics with :mod:`netsurf.incremental` and never infers a numerical jump
from bit position alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterable

import numpy as np

from .identities import BitIdentity
from .incremental import DenseReluDenseIncrementalEvaluator, FaultState, ForwardCache, metrics_from_logits


@dataclass(frozen=True)
class BitTopology:
    layer: str
    parameter_class: str
    source_index: int | None
    target_index: int


@dataclass(frozen=True)
class TopologyResponse:
    identity: BitIdentity
    topology: BitTopology
    original_encoded: int
    flipped_encoded: int
    exact_delta: float
    logits_delta: np.ndarray
    hidden_delta: np.ndarray | None
    relu_gate_changes: int
    cache_hit: bool
    fallback: bool
    wall_seconds: float


@dataclass(frozen=True)
class TopologyPairResult:
    left: TopologyResponse
    right: TopologyResponse
    topology_class: str
    logit_residual: np.ndarray
    exact_logits: np.ndarray
    additive_logits: np.ndarray
    finite_polar_per_sample: np.ndarray
    exact_loss_interaction_per_sample: np.ndarray
    analytical_bound_per_sample: np.ndarray
    structurally_exact: bool
    numerically_close: bool
    wall_seconds: float

    @property
    def maximum_bound_violation(self) -> float:
        error = np.abs(self.exact_loss_interaction_per_sample - self.finite_polar_per_sample)
        return float(np.max(error - self.analytical_bound_per_sample))


def _logsumexp(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    maximum = np.max(values, axis=1, keepdims=True)
    return maximum[:, 0] + np.log(np.sum(np.exp(values - maximum), axis=1))


def finite_polar_per_sample(z: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Label-independent exact finite additive-logit polarity."""

    return _logsumexp(z + left + right) - _logsumexp(z + left) - _logsumexp(z + right) + _logsumexp(z)


class DenseReluDenseTopologyEvaluator:
    """On-demand exact response engine over one fixed background state."""

    implementation_version = "dense-relu-dense-topology-exact/v1"

    def __init__(
        self,
        evaluator: DenseReluDenseIncrementalEvaluator,
        inputs: np.ndarray,
        background: FaultState = FaultState(frozenset()),
        *,
        response_cache_limit: int | None = None,
    ):
        if any(bit in background.bits for bit in ()):  # pragma: no cover - documents state ownership
            raise AssertionError
        self.evaluator = evaluator
        self.inputs = np.asarray(inputs)
        self.background = background
        if response_cache_limit is not None and response_cache_limit < 0:
            raise ValueError("response_cache_limit must be nonnegative or None")
        self.response_cache_limit = response_cache_limit
        self._cache: ForwardCache | None = evaluator.cache(self.inputs, background)
        self._responses: dict[BitIdentity, TopologyResponse] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.fallbacks = 0

    @property
    def cache(self) -> ForwardCache | None:
        return self._cache

    @property
    def activation_bytes(self) -> int:
        return 0 if self._cache is None else self._cache.activation_bytes

    def _topology(self, identity: BitIdentity) -> BitTopology:
        shape = self.evaluator.base_values[identity.parameter].shape
        if identity.parameter == "dense0/kernel":
            source, target = np.unravel_index(identity.flat_index, shape)
            return BitTopology("hidden", "kernel", int(source), int(target))
        if identity.parameter == "dense0/bias":
            return BitTopology("hidden", "bias", None, int(identity.flat_index))
        if identity.parameter == "dense1/kernel":
            source, target = np.unravel_index(identity.flat_index, shape)
            return BitTopology("output", "kernel", int(source), int(target))
        if identity.parameter == "dense1/bias":
            return BitTopology("output", "bias", None, int(identity.flat_index))
        raise ValueError(f"unsupported topology parameter: {identity.parameter}")

    def _jump(self, identity: BitIdentity) -> tuple[int, int, float]:
        # A retained background cache already owns the exactly decoded words;
        # avoid decoding the full parameter population for each candidate.
        values = self._cache.values if self._cache is not None else self.evaluator.values_for_state(self.background)
        value = values[identity.parameter].reshape(-1)[identity.flat_index]
        encoded = self.evaluator.codec.encode(value)
        flipped = self.evaluator.codec.flip(encoded, identity.bit_index)
        return int(encoded), int(flipped), float(self.evaluator.codec.decode(flipped) - self.evaluator.codec.decode(encoded))

    def response(self, identity: BitIdentity) -> TopologyResponse:
        started = perf_counter()
        if identity in self.background.bits:
            raise ValueError("candidate response bit must be excluded from background")
        if identity in self._responses:
            self.cache_hits += 1
            return self._responses[identity]
        self.cache_misses += 1
        topology = self._topology(identity)
        encoded, flipped, delta = self._jump(identity)
        if self._cache is None:
            self.fallbacks += 1
            base = self.evaluator.full_forward(self.inputs, self.background).logits
            changed = self.evaluator.full_forward(self.inputs, self.background.with_bit(identity)).logits
            result = TopologyResponse(identity, topology, encoded, flipped, delta, changed - base, None, 0, False, True, perf_counter() - started)
            if self.response_cache_limit is None or len(self._responses) < self.response_cache_limit:
                self._responses[identity] = result
            return result
        cache = self._cache
        samples, outputs = cache.logits.shape
        hidden_delta = None
        gate_changes = 0
        if topology.layer == "hidden":
            if topology.parameter_class == "kernel":
                pre_delta = delta * cache.inputs_flat[:, topology.source_index]
            else:
                pre_delta = np.full(samples, delta, dtype=np.result_type(cache.dense0_pre, np.float64))
            original_pre = cache.dense0_pre[:, topology.target_index]
            hidden_delta = np.maximum(original_pre + pre_delta, 0) - np.maximum(original_pre, 0)
            gate_changes = int(np.count_nonzero((original_pre > 0) != (original_pre + pre_delta > 0)))
            logits_delta = hidden_delta[:, None] * cache.values["dense1/kernel"][topology.target_index, :][None, :]
        else:
            logits_delta = np.zeros((samples, outputs), dtype=np.result_type(cache.logits, np.float64))
            if topology.parameter_class == "kernel":
                logits_delta[:, topology.target_index] = delta * cache.dense0_activation[:, topology.source_index]
            else:
                logits_delta[:, topology.target_index] = delta
        result = TopologyResponse(identity, topology, encoded, flipped, delta, logits_delta, hidden_delta, gate_changes, True, False, perf_counter() - started)
        if self.response_cache_limit is None or len(self._responses) < self.response_cache_limit:
            self._responses[identity] = result
        return result

    @staticmethod
    def classify(left: BitTopology, right: BitTopology) -> str:
        if left.layer == right.layer == "output":
            return "output_output_additive"
        if left.layer == right.layer == "hidden":
            return "hidden_same_neuron" if left.target_index == right.target_index else "hidden_different_neurons_additive"
        hidden, output = (left, right) if left.layer == "hidden" else (right, left)
        if output.parameter_class == "bias":
            return "hidden_output_bias_additive"
        return "cross_layer_same_neuron" if hidden.target_index == output.source_index else "cross_layer_different_neurons_additive"

    def pair(self, left_identity: BitIdentity, right_identity: BitIdentity, labels: np.ndarray) -> TopologyPairResult:
        started = perf_counter()
        if left_identity == right_identity:
            raise ValueError("pair identities must be distinct")
        left, right = self.response(left_identity), self.response(right_identity)
        topology_class = self.classify(left.topology, right.topology)
        if self._cache is None:
            z0 = self.evaluator.full_forward(self.inputs, self.background).logits
            exact_logits = self.evaluator.full_forward(self.inputs, self.background.with_bit(left_identity).with_bit(right_identity)).logits
            residual = exact_logits - z0 - left.logits_delta - right.logits_delta
        else:
            z0 = self._cache.logits
            residual = np.zeros_like(z0, dtype=np.result_type(z0, np.float64))
            if topology_class == "hidden_same_neuron":
                j = left.topology.target_index
                a = self._cache.dense0_pre[:, j]
                # Reconstruct each exact preactivation jump from its response metadata.
                def v(response: TopologyResponse):
                    return (response.exact_delta * self._cache.inputs_flat[:, response.topology.source_index]
                            if response.topology.parameter_class == "kernel"
                            else np.full(len(a), response.exact_delta))
                vp, vq = v(left), v(right)
                relu_residual = np.maximum(a + vp + vq, 0) - np.maximum(a + vp, 0) - np.maximum(a + vq, 0) + np.maximum(a, 0)
                residual = relu_residual[:, None] * self._cache.values["dense1/kernel"][j, :][None, :]
            elif topology_class == "cross_layer_same_neuron":
                hidden, output = (left, right) if left.topology.layer == "hidden" else (right, left)
                residual[:, output.topology.target_index] = output.exact_delta * hidden.hidden_delta
            exact_logits = z0 + left.logits_delta + right.logits_delta + residual
        additive = z0 + left.logits_delta + right.logits_delta
        polar = finite_polar_per_sample(z0, left.logits_delta, right.logits_delta)
        labels = np.asarray(labels, dtype=np.int64)
        exact_loss = (metrics_from_logits(exact_logits, labels).per_example_loss
                      - metrics_from_logits(z0 + left.logits_delta, labels).per_example_loss
                      - metrics_from_logits(z0 + right.logits_delta, labels).per_example_loss
                      + metrics_from_logits(z0, labels).per_example_loss)
        bound = np.sqrt(2.0) * np.linalg.norm(residual.astype(np.float64), axis=1)
        exact_structure = topology_class.endswith("additive")
        close = bool(np.allclose(exact_loss, polar, rtol=1e-10, atol=1e-12))
        return TopologyPairResult(left, right, topology_class, residual, exact_logits, additive, polar, exact_loss, bound, exact_structure, close, perf_counter() - started)

    def multi_toggle_logits(self, toggles: Iterable[BitIdentity]) -> np.ndarray:
        """Exact arbitrary add/remove XOR update, including the cross-layer product."""

        target = self.background.toggle(*tuple(toggles))
        if self._cache is None:
            self.fallbacks += 1
            return self.evaluator.full_forward(self.inputs, target).logits
        # The existing incremental implementation evaluates precisely
        # V^G dh + dV h^G + dV dh + dc from the decoded target words.
        return self.evaluator.incremental_forward(self._cache, self.inputs, target).logits


def benchmark_topology_engines(
    evaluator: DenseReluDenseIncrementalEvaluator,
    inputs: np.ndarray,
    labels: np.ndarray,
    candidates: Iterable[BitIdentity],
) -> dict[str, object]:
    """Bounded end-to-end benchmark including cache construction.

    The returned timings are observations for this workload and make no
    architectural speedup claim.
    """

    candidates = tuple(candidates)
    if not candidates:
        raise ValueError("at least one benchmark candidate is required")
    background = FaultState.from_bits()
    started = perf_counter()
    cache = evaluator.cache(inputs, background)
    cache_seconds = perf_counter() - started
    topology = DenseReluDenseTopologyEvaluator(evaluator, inputs, background)

    started = perf_counter()
    full = [evaluator.full_forward(inputs, background.with_bit(bit)).logits for bit in candidates]
    full_seconds = perf_counter() - started
    started = perf_counter()
    incremental = [evaluator.incremental_forward(cache, inputs, background.with_bit(bit)).logits for bit in candidates]
    incremental_seconds = perf_counter() - started
    started = perf_counter()
    responses = [topology.response(bit).logits_delta for bit in candidates]
    topology_seconds = perf_counter() - started
    base = evaluator.full_forward(inputs, background).logits
    maximum_full_error = max(float(np.max(np.abs(base + response - reference))) for response, reference in zip(responses, full))
    maximum_incremental_error = max(float(np.max(np.abs(reference - exact))) for reference, exact in zip(incremental, full))
    pair_count = len(candidates) // 2
    topology_pair_seconds = full_pair_seconds = 0.0
    if pair_count:
        pairs = tuple(zip(candidates[0:2 * pair_count:2], candidates[1:2 * pair_count:2]))
        started = perf_counter()
        for left, right in pairs:
            topology.pair(left, right, labels)
        topology_pair_seconds = perf_counter() - started
        from .polar import measure_polar_pair
        started = perf_counter()
        for left, right in pairs:
            measure_polar_pair(evaluator, inputs, labels, background, left, right)
        full_pair_seconds = perf_counter() - started
    return {
        "candidate_count": len(candidates), "sample_count": len(inputs),
        "cache_construction_seconds": cache_seconds,
        "cache_activation_bytes": 0 if cache is None else cache.activation_bytes,
        "full_seconds": full_seconds, "incremental_seconds": incremental_seconds,
        "topology_seconds": topology_seconds,
        "full_responses_per_second": len(candidates) / full_seconds if full_seconds else float("inf"),
        "incremental_responses_per_second": len(candidates) / incremental_seconds if incremental_seconds else float("inf"),
        "topology_responses_per_second": len(candidates) / topology_seconds if topology_seconds else float("inf"),
        "pair_count": pair_count,
        "existing_pair_measurements_per_second": pair_count / full_pair_seconds if full_pair_seconds else None,
        "topology_pair_measurements_per_second": pair_count / topology_pair_seconds if topology_pair_seconds else None,
        "topology_cache_hits": topology.cache_hits, "topology_cache_misses": topology.cache_misses,
        "topology_fallbacks": topology.fallbacks,
        "maximum_full_logit_absolute_error": maximum_full_error,
        "maximum_incremental_logit_absolute_error": maximum_incremental_error,
        "loss_checksum": float(sum(metrics_from_logits(value, labels).categorical_crossentropy for value in full)),
    }
