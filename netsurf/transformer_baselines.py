"""Bounded estimator and allocation primitives for verified storage bits.

These scores are separate from the fault law and task-risk evaluator.  A score
does not by itself establish protection benefit; held-out paired trials do.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import math

import numpy as np

from .identities import BitIdentity
from .storage_bit_codec import StorageBitCodec


class UnsupportedEstimator(ValueError):
    pass


def random_scores(bits: Sequence[BitIdentity], seed: int) -> dict[BitIdentity, float]:
    if len(set(bits)) != len(bits):
        raise ValueError("duplicate candidate bits")
    generator = np.random.default_rng(seed)
    return {bit: float(value) for bit, value in zip(bits, generator.random(len(bits)))}


def significance_scores(bits: Sequence[BitIdentity]) -> dict[BitIdentity, float]:
    """MSB-first stored-position order, with stable identity tie-breaking."""
    return {bit: float(bit.bit_index) for bit in bits}


def _transition(bit: BitIdentity, raw: Mapping[str, bytes], codec: StorageBitCodec) -> tuple[float, float]:
    if bit.parameter not in raw:
        raise ValueError(f"candidate absent from raw storage: {bit.parameter}")
    start = bit.flat_index * codec.byte_width
    word = raw[bit.parameter][start:start + codec.byte_width]
    if len(word) != codec.byte_width:
        raise ValueError(f"candidate outside raw storage: {bit.key()}")
    before = codec.value(word)
    after = codec.value(codec.xor(word, 0, bit.bit_index))
    return before, after


def numerical_displacement_scores(bits: Sequence[BitIdentity], raw: Mapping[str, bytes],
                                  codec: StorageBitCodec) -> dict[BitIdentity, float]:
    scores = {}
    for bit in bits:
        before, after = _transition(bit, raw, codec)
        if not math.isfinite(before) or not math.isfinite(after):
            raise UnsupportedEstimator(f"nonfinite numerical displacement at {bit.key()}")
        scores[bit] = abs(after - before)
    return scores


def gradient_scores(bits: Sequence[BitIdentity], raw: Mapping[str, bytes], codec: StorageBitCodec,
                    per_example_gradients: Mapping[str, np.ndarray], *,
                    policy: str) -> dict[BitIdentity, float]:
    if policy not in {"global_absolute", "per_example_absolute", "per_example_positive"}:
        raise UnsupportedEstimator(f"unsupported gradient policy: {policy}")
    scores = {}
    for bit in bits:
        before, after = _transition(bit, raw, codec)
        if not math.isfinite(before) or not math.isfinite(after):
            raise UnsupportedEstimator(f"nonfinite first-order transition at {bit.key()}")
        gradients = np.asarray(per_example_gradients[bit.parameter], dtype=np.float64)
        if gradients.ndim < 2 or bit.flat_index >= int(np.prod(gradients.shape[1:])):
            raise ValueError(f"per-example gradient shape mismatch: {bit.key()}")
        values = gradients.reshape(gradients.shape[0], -1)[:, bit.flat_index]
        if not np.all(np.isfinite(values)):
            raise UnsupportedEstimator(f"nonfinite gradients at {bit.key()}")
        delta = after - before
        if policy == "global_absolute":
            score = abs(float(np.mean(values)) * delta)
        elif policy == "per_example_absolute":
            score = float(np.mean(np.abs(values * delta)))
        else:
            score = float(np.mean(np.maximum(values * delta, 0)))
        scores[bit] = score
    return scores


def exact_isolated_scores(bits: Sequence[BitIdentity], clean_risk: float,
                          evaluate_one_bit: Callable[[BitIdentity], float]) -> dict[BitIdentity, float]:
    """Task-risk difference on declared calibration samples; caller owns exposure."""
    if not math.isfinite(clean_risk):
        raise ValueError("clean risk must be finite")
    scores = {}
    for bit in bits:
        value = evaluate_one_bit(bit)
        if not math.isfinite(value):
            raise ValueError("isolated evaluator must apply the prespecified finite failure penalty")
        scores[bit] = value - clean_risk
    return scores


def select_equal_budget(scores: Mapping[BitIdentity, float], budget: int) -> frozenset[BitIdentity]:
    if not 0 <= budget <= len(scores) or any(not math.isfinite(v) for v in scores.values()):
        raise ValueError("invalid budget or nonfinite score")
    return frozenset(sorted(scores, key=lambda bit: (-scores[bit], bit))[:budget])
