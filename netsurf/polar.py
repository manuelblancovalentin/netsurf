"""Bounded finite-polar interaction measurements for proposal guidance."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Iterable, Sequence

import numpy as np

from .identities import BitIdentity
from .incremental import DenseReluDenseIncrementalEvaluator, FaultState, metrics_from_logits


def _losses(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return metrics_from_logits(logits, labels).per_example_loss


def _layer(identity: BitIdentity) -> str:
    return identity.parameter.split("/", 1)[0]


@dataclass(frozen=True)
class PolarPairMeasurement:
    left: BitIdentity
    right: BitIdentity
    background_fault_count: int
    finite_polar_interaction: float
    local_quadratic_interaction: float
    exact_interaction: float
    interaction_prediction_error: float
    interaction_sign_agreement: bool
    logit_nonadditivity_mean_l2: float
    analytical_error_bound: float
    bound_satisfied: bool
    cancellation_score: float | None
    logit_sign_agreement_frequency: float
    logit_sign_opposition_frequency: float
    left_weight_delta: float
    right_weight_delta: float
    left_original_encoded: int
    right_original_encoded: int
    left_activation_mean: float
    left_activation_std: float
    right_activation_mean: float
    right_activation_std: float
    relu_gate_changes: int
    layer_relationship: str
    wall_seconds: float
    implementation_version: str = "finite-additive-logit-polar/v1"

    def to_dict(self):
        result = asdict(self)
        result["left"] = self.left.key()
        result["right"] = self.right.key()
        return result


def _activation_statistics(cache, identity: BitIdentity) -> tuple[float, float]:
    activation = cache.inputs_flat if _layer(identity) == "dense0" else cache.dense0_activation
    if identity.parameter.endswith("/bias"):
        values = np.ones(len(activation), dtype=np.float64)
    else:
        source_index = np.unravel_index(identity.flat_index, cache.values[identity.parameter].shape)[0]
        values = activation[:, source_index]
    return float(np.mean(values)), float(np.std(values))


def _weight_delta(evaluator, background: FaultState, identity: BitIdentity) -> tuple[float, int]:
    values = evaluator.values_for_state(background)
    flat = values[identity.parameter].reshape(-1)
    encoded = evaluator.codec.encode(flat[identity.flat_index])
    flipped = evaluator.codec.flip(encoded, identity.bit_index)
    return float(evaluator.codec.decode(flipped) - evaluator.codec.decode(encoded)), int(encoded)


def measure_polar_pair(
    evaluator: DenseReluDenseIncrementalEvaluator,
    inputs: np.ndarray,
    labels: np.ndarray,
    background: FaultState,
    left: BitIdentity,
    right: BitIdentity,
) -> PolarPairMeasurement:
    """Measure additive-logit polar and exact four-state interactions."""
    if left == right:
        raise ValueError("polar pair identities must be distinct")
    if left in background.bits or right in background.bits:
        raise ValueError("polar background must exclude both pair bits")
    started = perf_counter()
    cache = evaluator.cache(inputs, background)
    if cache is None:
        z0 = evaluator.full_forward(inputs, background).logits
        left_result = evaluator.full_forward(inputs, background.with_bit(left))
        right_result = evaluator.full_forward(inputs, background.with_bit(right))
        both_result = evaluator.full_forward(inputs, background.with_bit(left).with_bit(right))
    else:
        z0 = cache.logits
        left_result = evaluator.incremental_forward(cache, inputs, background.with_bit(left))
        right_result = evaluator.incremental_forward(cache, inputs, background.with_bit(right))
        both_result = evaluator.incremental_forward(cache, inputs, background.with_bit(left).with_bit(right))
    za, zb, zab = left_result.logits, right_result.logits, both_result.logits
    ua, ub = za - z0, zb - z0
    additive = z0 + ua + ub
    polar_per_example = _losses(additive, labels) - _losses(za, labels) - _losses(zb, labels) + _losses(z0, labels)
    exact_per_example = _losses(zab, labels) - _losses(za, labels) - _losses(zb, labels) + _losses(z0, labels)
    residual = zab - additive
    residual_norm = np.linalg.norm(residual.astype(np.float64), axis=1)
    error = float(np.mean(exact_per_example - polar_per_example))
    bound = float(np.sqrt(2.0) * np.mean(residual_norm))
    shifted = z0.astype(np.float64) - np.max(z0, axis=1, keepdims=True)
    probabilities = np.exp(shifted); probabilities /= probabilities.sum(axis=1, keepdims=True)
    local = np.sum(probabilities * ua * ub, axis=1) - np.sum(probabilities * ua, axis=1) * np.sum(probabilities * ub, axis=1)
    denominator = float(np.mean(np.sum(ua.astype(np.float64) ** 2, axis=1)) + np.mean(np.sum(ub.astype(np.float64) ** 2, axis=1)))
    cancellation = None if denominator == 0 else float(np.mean(np.sum((ua + ub).astype(np.float64) ** 2, axis=1)) / denominator)
    products = ua.astype(np.float64) * ub.astype(np.float64)
    nonzero = (ua != 0) & (ub != 0)
    agreement = float(np.mean(products[nonzero] > 0)) if np.any(nonzero) else 0.0
    opposition = float(np.mean(products[nonzero] < 0)) if np.any(nonzero) else 0.0
    if cache is None:
        cache = evaluator.cache(inputs, background)
        if cache is None:
            # Statistics do not affect inference; compute a transient cache
            # under an unlimited local evaluator only when diagnostics were requested.
            transient = DenseReluDenseIncrementalEvaluator(evaluator.adapter, evaluator.codec, cache_memory_limit_bytes=2**63 - 1)
            cache = transient.cache(inputs, background)
    left_mean, left_std = _activation_statistics(cache, left)
    right_mean, right_std = _activation_statistics(cache, right)
    left_delta, left_encoded = _weight_delta(evaluator, background, left)
    right_delta, right_encoded = _weight_delta(evaluator, background, right)
    finite = float(np.mean(polar_per_example)); exact = float(np.mean(exact_per_example))
    return PolarPairMeasurement(
        left, right, len(background.bits), finite, float(np.mean(local)), exact,
        error, bool(np.sign(finite) == np.sign(exact)), float(np.mean(residual_norm)),
        bound, abs(error) <= bound + 1e-12, cancellation, agreement, opposition,
        left_delta, right_delta, left_encoded, right_encoded,
        left_mean, left_std, right_mean, right_std,
        left_result.diagnostics.relu_gate_changes + right_result.diagnostics.relu_gate_changes + both_result.diagnostics.relu_gate_changes,
        "within_layer" if _layer(left) == _layer(right) else "cross_layer",
        perf_counter() - started,
    )


def measure_sparse_interactions(
    evaluator: DenseReluDenseIncrementalEvaluator,
    inputs: np.ndarray,
    labels: np.ndarray,
    pairs: Sequence[tuple[BitIdentity, BitIdentity]],
    backgrounds: Sequence[FaultState],
    *,
    progress=None,
) -> tuple[PolarPairMeasurement, ...]:
    if not pairs or not backgrounds:
        return ()
    result = []
    total = len(pairs) * len(backgrounds)
    for background_index, background in enumerate(backgrounds):
        for pair_index, (left, right) in enumerate(pairs):
            value = measure_polar_pair(evaluator, inputs, labels, background, left, right)
            result.append(value)
            if progress:
                progress(len(result), total, value)
    return tuple(result)


def interaction_map(measurements: Iterable[PolarPairMeasurement]) -> dict[tuple[BitIdentity, BitIdentity], float]:
    grouped: dict[tuple[BitIdentity, BitIdentity], list[float]] = {}
    for value in measurements:
        key = (value.left, value.right)
        grouped.setdefault(key, []).append(value.finite_polar_interaction)
    return {key: float(np.mean(values)) for key, values in grouped.items()}
