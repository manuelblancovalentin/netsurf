"""Exact layer-aware inference updates for supported Dense-ReLU-Dense models.

The evaluator shares NetSuRF's :class:`BitCodec` and :class:`BitIdentity`
contracts.  A fault state is always an XOR set relative to the immutable base
checkpoint; it is never represented as accumulated floating-point deltas.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterable, Mapping, Sequence

import numpy as np

from .identities import BitIdentity
from .quantization import BitCodec
from .statistics import summarize


SUPPORTED_PARAMETERS = (
    "dense0/kernel", "dense0/bias", "dense1/kernel", "dense1/bias",
)


@dataclass(frozen=True)
class FaultState:
    bits: frozenset[BitIdentity]

    @classmethod
    def from_bits(cls, bits: Iterable[BitIdentity] = ()) -> "FaultState":
        return cls(frozenset(bits))

    def toggle(self, *bits: BitIdentity) -> "FaultState":
        result = set(self.bits)
        for bit in bits:
            if bit in result:
                result.remove(bit)
            else:
                result.add(bit)
        return FaultState(frozenset(result))

    def with_bit(self, bit: BitIdentity) -> "FaultState":
        return FaultState(self.bits | {bit})

    def without_bit(self, bit: BitIdentity) -> "FaultState":
        return FaultState(self.bits - {bit})


@dataclass(frozen=True)
class ForwardCache:
    state: FaultState
    inputs_flat: np.ndarray
    values: Mapping[str, np.ndarray]
    dense0_pre: np.ndarray
    dense0_activation: np.ndarray
    logits: np.ndarray
    activation_bytes: int


@dataclass(frozen=True)
class IncrementalDiagnostics:
    evaluator: str
    earliest_affected_layer: int | None
    affected_layers: int
    reused_activation_layers: int
    changed_bits: int
    changed_parameters: int
    relu_gate_changes: int
    relu_gate_change_rate: float
    cache_hit: bool
    fallback: bool
    fallback_reason: str | None
    work_proxy_macs: int
    wall_seconds: float


@dataclass(frozen=True)
class InferenceResult:
    logits: np.ndarray
    diagnostics: IncrementalDiagnostics


@dataclass(frozen=True)
class MetricResult:
    accuracy: float
    categorical_crossentropy: float
    per_example_loss: np.ndarray
    predictions: np.ndarray


@dataclass(frozen=True)
class ConditionalSwapEstimate:
    removed: BitIdentity
    added: BitIdentity
    probability_removed: float
    probability_added: float
    context_count: int
    risk_difference: Mapping[str, object]
    removed_fault_effect: Mapping[str, object]
    added_fault_effect: Mapping[str, object]
    background_evaluations: int
    incremental_evaluations: int
    full_evaluations: int
    gate_changes: int
    work_proxy_macs: int
    affected_layers: int
    reused_layers: int
    cache_hits: int
    cache_misses: int
    fallback_count: int
    wall_seconds: float
    estimator: str = "three-state matched conditional swap/v1"


def metrics_from_logits(logits: np.ndarray, labels: np.ndarray) -> MetricResult:
    logits = np.asarray(logits)
    labels = np.asarray(labels, dtype=np.int64)
    shifted = logits.astype(np.float64) - np.max(logits, axis=1, keepdims=True)
    log_normalizer = np.log(np.sum(np.exp(shifted), axis=1))
    losses = log_normalizer - shifted[np.arange(len(labels)), labels]
    predictions = np.argmax(logits, axis=1)
    return MetricResult(
        float(np.mean(predictions == labels)), float(np.mean(losses)),
        losses, predictions,
    )


class DenseReluDenseIncrementalEvaluator:
    """Exact state-transition evaluator for explicit two-layer benchmarks."""

    implementation_version = "dense-relu-dense-exact-incremental/v1"

    def __init__(self, adapter, codec: BitCodec, *, cache_memory_limit_bytes: int = 256 * 1024 * 1024):
        if cache_memory_limit_bytes < 0:
            raise ValueError("cache_memory_limit_bytes must be nonnegative")
        values = adapter.values()
        if tuple(values) != SUPPORTED_PARAMETERS and set(values) != set(SUPPORTED_PARAMETERS):
            raise ValueError(
                "incremental inference supports explicit dense0/dense1 kernel and bias parameters only"
            )
        if values["dense0/kernel"].ndim != 2 or values["dense1/kernel"].ndim != 2:
            raise ValueError("incremental inference requires two rank-2 Dense kernels")
        if values["dense0/kernel"].shape[1] != values["dense1/kernel"].shape[0]:
            raise ValueError("dense0 output width must equal dense1 input width")
        self.adapter = adapter
        self.codec = codec
        self.base_values = {name: np.asarray(value).copy() for name, value in values.items()}
        self.base_encoded = {name: codec.encode(value).reshape(-1) for name, value in self.base_values.items()}
        self.cache_memory_limit_bytes = int(cache_memory_limit_bytes)
        self.identity_set = frozenset(adapter.bit_identities())

    def _validate_state(self, state: FaultState) -> None:
        unknown = state.bits - self.identity_set
        if unknown:
            raise ValueError(f"fault state contains {len(unknown)} ineligible bit identities")

    def values_for_state(self, state: FaultState) -> dict[str, np.ndarray]:
        """Decode the exact checkpoint XOR state, grouping bits by scalar word."""
        self._validate_state(state)
        encoded = {name: value.copy() for name, value in self.base_encoded.items()}
        for identity in sorted(state.bits):
            encoded[identity.parameter][identity.flat_index] ^= np.uint64(1 << identity.bit_index)
        return {
            name: self.codec.decode(codes).reshape(self.base_values[name].shape).astype(self.base_values[name].dtype)
            for name, codes in encoded.items()
        }

    @staticmethod
    def _flat(inputs: np.ndarray, width: int, dtype) -> np.ndarray:
        flat = np.asarray(inputs, dtype=dtype).reshape(len(inputs), -1)
        if flat.shape[1] != width:
            raise ValueError(f"input has {flat.shape[1]} features; expected {width}")
        return flat

    @staticmethod
    def _forward_values(flat: np.ndarray, values: Mapping[str, np.ndarray]):
        pre = flat @ values["dense0/kernel"] + values["dense0/bias"]
        hidden = np.maximum(pre, np.asarray(0, dtype=pre.dtype))
        logits = hidden @ values["dense1/kernel"] + values["dense1/bias"]
        return pre, hidden, logits

    def full_forward(self, inputs: np.ndarray, state: FaultState = FaultState(frozenset())) -> InferenceResult:
        started = perf_counter()
        values = self.values_for_state(state)
        flat = self._flat(inputs, values["dense0/kernel"].shape[0], values["dense0/kernel"].dtype)
        _, _, logits = self._forward_values(flat, values)
        samples = len(flat)
        full_macs = samples * (
            values["dense0/kernel"].size + values["dense1/kernel"].size
        )
        return InferenceResult(logits, IncrementalDiagnostics(
            "full_forward_reference/v1", 0 if state.bits else None, 2 if state.bits else 0,
            0, len(state.bits), len({(bit.parameter, bit.flat_index) for bit in state.bits}),
            0, 0.0, False, False, None, full_macs, perf_counter() - started,
        ))

    def full_work_proxy_macs(self, sample_count: int) -> int:
        """Dense multiply-accumulate count for one complete forward pass."""

        return int(sample_count) * (
            self.base_values["dense0/kernel"].size + self.base_values["dense1/kernel"].size
        )

    def cache(self, inputs: np.ndarray, state: FaultState = FaultState(frozenset())) -> ForwardCache | None:
        values = self.values_for_state(state)
        flat = self._flat(inputs, values["dense0/kernel"].shape[0], values["dense0/kernel"].dtype)
        pre, hidden, logits = self._forward_values(flat, values)
        activation_bytes = flat.nbytes + pre.nbytes + hidden.nbytes + logits.nbytes
        if activation_bytes > self.cache_memory_limit_bytes:
            return None
        return ForwardCache(state, flat, values, pre, hidden, logits, activation_bytes)

    @staticmethod
    def _changed_scalars(changed: Sequence[BitIdentity]):
        return {(identity.parameter, identity.flat_index) for identity in changed}

    @staticmethod
    def _sparse_affine_delta(inputs: np.ndarray, delta: np.ndarray, changed_indices: set[int]) -> tuple[np.ndarray, int]:
        output = np.zeros((len(inputs), delta.shape[1]), dtype=np.result_type(inputs, delta))
        for flat_index in sorted(changed_indices):
            source, destination = np.unravel_index(flat_index, delta.shape)
            value = delta[source, destination]
            if value != 0:
                output[:, destination] += inputs[:, source] * value
        return output, len(inputs) * len(changed_indices)

    def incremental_forward(self, cache: ForwardCache | None, inputs: np.ndarray, target: FaultState) -> InferenceResult:
        started = perf_counter()
        if cache is None:
            result = self.full_forward(inputs, target)
            return InferenceResult(result.logits, IncrementalDiagnostics(
                self.implementation_version, result.diagnostics.earliest_affected_layer,
                result.diagnostics.affected_layers, 0, result.diagnostics.changed_bits,
                result.diagnostics.changed_parameters, 0, 0.0, False, True,
                "activation cache exceeds configured memory limit", result.diagnostics.work_proxy_macs,
                perf_counter() - started,
            ))
        self._validate_state(target)
        changed = tuple(sorted(cache.state.bits ^ target.bits))
        if not changed:
            return InferenceResult(cache.logits.copy(), IncrementalDiagnostics(
                self.implementation_version, None, 0, 2, 0, 0, 0, 0.0, True,
                False, None, 0, perf_counter() - started,
            ))
        target_values = self.values_for_state(target)
        deltas = {name: target_values[name] - cache.values[name] for name in SUPPORTED_PARAMETERS}
        scalar_changes = self._changed_scalars(changed)
        changed_by_parameter = {
            name: {flat_index for parameter, flat_index in scalar_changes if parameter == name}
            for name in SUPPORTED_PARAMETERS
        }
        layer0_changed = bool(changed_by_parameter["dense0/kernel"] or changed_by_parameter["dense0/bias"])
        layer1_changed = bool(changed_by_parameter["dense1/kernel"] or changed_by_parameter["dense1/bias"])
        work = 0
        gate_changes = 0
        if layer0_changed:
            delta_pre, count = self._sparse_affine_delta(
                cache.inputs_flat, deltas["dense0/kernel"], changed_by_parameter["dense0/kernel"]
            )
            work += count
            for index in changed_by_parameter["dense0/bias"]:
                delta_pre[:, index] += deltas["dense0/bias"].reshape(-1)[index]
            target_pre = cache.dense0_pre + delta_pre
            target_hidden = np.maximum(target_pre, np.asarray(0, dtype=target_pre.dtype))
            delta_hidden = target_hidden - cache.dense0_activation
            gate_changes = int(np.count_nonzero((cache.dense0_pre > 0) != (target_pre > 0)))
            # W1^G delta_h is required even when its ReLU gate is unchanged.
            delta_logits = delta_hidden @ cache.values["dense1/kernel"]
            work += len(cache.inputs_flat) * cache.values["dense1/kernel"].size
            if layer1_changed:
                sparse_old, count = self._sparse_affine_delta(
                    cache.dense0_activation, deltas["dense1/kernel"], changed_by_parameter["dense1/kernel"]
                )
                sparse_cross, cross_count = self._sparse_affine_delta(
                    delta_hidden, deltas["dense1/kernel"], changed_by_parameter["dense1/kernel"]
                )
                delta_logits += sparse_old + sparse_cross
                work += count + cross_count
                for index in changed_by_parameter["dense1/bias"]:
                    delta_logits[:, index] += deltas["dense1/bias"].reshape(-1)[index]
            logits = cache.logits + delta_logits
            earliest, affected, reused = 0, 2, 0
        else:
            delta_logits, count = self._sparse_affine_delta(
                cache.dense0_activation, deltas["dense1/kernel"], changed_by_parameter["dense1/kernel"]
            )
            work += count
            for index in changed_by_parameter["dense1/bias"]:
                delta_logits[:, index] += deltas["dense1/bias"].reshape(-1)[index]
            logits = cache.logits + delta_logits
            earliest, affected, reused = 1, 1, 1
        return InferenceResult(logits, IncrementalDiagnostics(
            self.implementation_version, earliest, affected, reused, len(changed),
            len(scalar_changes), gate_changes,
            gate_changes / cache.dense0_pre.size if layer0_changed else 0.0,
            True, False, None, work, perf_counter() - started,
        ))


def evaluate_conditional_swap(
    evaluator: DenseReluDenseIncrementalEvaluator,
    inputs: np.ndarray,
    labels: np.ndarray,
    backgrounds: Sequence[FaultState],
    removed: BitIdentity,
    added: BitIdentity,
    *,
    probability_removed: float,
    probability_added: float,
    confidence: float = 0.95,
    mode: str = "incremental",
    cache_store: dict[FaultState, ForwardCache | None] | None = None,
) -> ConditionalSwapEstimate:
    """Evaluate the exact three-state conditional identity on matched contexts."""
    if removed == added:
        raise ValueError("swap bits must be distinct")
    if not 0 <= probability_removed <= 1 or not 0 <= probability_added <= 1:
        raise ValueError("bit probabilities must be in [0,1]")
    if mode not in {"incremental", "full"}:
        raise ValueError("conditional evaluator mode must be incremental or full")
    started = perf_counter()
    risk_differences, removed_effects, added_effects = [], [], []
    backgrounds_count = incremental_count = full_count = gate_changes = work = 0
    affected_layers = reused_layers = cache_hits = cache_misses = fallback_count = 0
    for background in backgrounds:
        if removed in background.bits or added in background.bits:
            raise ValueError("conditional background must exclude both swapped bits")
        reused_cache = (
            cache_store is not None and background in cache_store
            and cache_store[background] is not None
        )
        if reused_cache:
            cache = cache_store[background]
        else:
            cache = evaluator.cache(inputs, background) if mode == "incremental" else None
            if cache_store is not None and cache is not None:
                cache_store[background] = cache
        if mode == "incremental" and cache is not None:
            cache_hits += int(reused_cache)
            cache_misses += int(not reused_cache)
            background_logits = cache.logits
            removed_result = evaluator.incremental_forward(cache, inputs, background.with_bit(removed))
            added_result = evaluator.incremental_forward(cache, inputs, background.with_bit(added))
            incremental_count += 2
        else:
            cache_misses += int(not reused_cache)
            background_result = evaluator.full_forward(inputs, background)
            removed_result = evaluator.full_forward(inputs, background.with_bit(removed))
            added_result = evaluator.full_forward(inputs, background.with_bit(added))
            background_logits = background_result.logits
            full_count += 2  # The separately counted background is also full-forward.
        if not reused_cache:
            backgrounds_count += 1
            work += evaluator.full_work_proxy_macs(len(inputs))
        base_loss = metrics_from_logits(background_logits, labels).categorical_crossentropy
        removed_loss = metrics_from_logits(removed_result.logits, labels).categorical_crossentropy
        added_loss = metrics_from_logits(added_result.logits, labels).categorical_crossentropy
        removed_effect = removed_loss - base_loss
        added_effect = added_loss - base_loss
        removed_effects.append(removed_effect)
        added_effects.append(added_effect)
        risk_differences.append(probability_removed * removed_effect - probability_added * added_effect)
        gate_changes += removed_result.diagnostics.relu_gate_changes + added_result.diagnostics.relu_gate_changes
        work += removed_result.diagnostics.work_proxy_macs + added_result.diagnostics.work_proxy_macs
        affected_layers += removed_result.diagnostics.affected_layers + added_result.diagnostics.affected_layers
        reused_layers += removed_result.diagnostics.reused_activation_layers + added_result.diagnostics.reused_activation_layers
        fallback_count += int(removed_result.diagnostics.fallback) + int(added_result.diagnostics.fallback)
    if not backgrounds:
        raise ValueError("at least one conditional background is required")
    return ConditionalSwapEstimate(
        removed, added, float(probability_removed), float(probability_added), len(backgrounds),
        summarize(risk_differences, confidence=confidence).to_dict(),
        summarize(removed_effects, confidence=confidence).to_dict(),
        summarize(added_effects, confidence=confidence).to_dict(),
        backgrounds_count, incremental_count, full_count, gate_changes, work,
        affected_layers, reused_layers, cache_hits, cache_misses, fallback_count,
        perf_counter() - started,
    )


def sample_conditional_backgrounds(
    universe: Sequence[BitIdentity], protected: Iterable[BitIdentity], excluded: Iterable[BitIdentity],
    *, probability: float, count: int, seed: int,
) -> tuple[FaultState, ...]:
    """Draw persistent Bernoulli backgrounds over eligible unprotected bits."""
    if not 0 <= probability <= 1 or count < 1:
        raise ValueError("probability must be in [0,1] and count must be positive")
    protected_set, excluded_set = set(protected), set(excluded)
    population = tuple(bit for bit in universe if bit not in protected_set and bit not in excluded_set)
    generator = np.random.default_rng(int(seed))
    result = []
    for _ in range(count):
        selected = generator.random(len(population)) < probability
        result.append(FaultState.from_bits(bit for bit, hit in zip(population, selected) if hit))
    return tuple(result)
