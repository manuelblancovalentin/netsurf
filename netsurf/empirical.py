"""Exact single-bit susceptibility measurements and candidate sampling."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Mapping, Sequence

import numpy as np

from .identities import BitIdentity
from .quantization import BitCodec


def layer_name(parameter: str) -> str:
    return parameter.split("/", 1)[0]


def parameter_class(parameter: str) -> str:
    return parameter.rsplit("/", 1)[-1]


@dataclass(frozen=True)
class CandidateSelection:
    identity: BitIdentity
    inclusion_rule: str
    sampling_probability: float
    score_quantile: int | None = None


@dataclass(frozen=True)
class SingleBitMeasurement:
    parameter: str
    flat_index: int
    bit_index: int
    layer: str
    parameter_class: str
    original_encoded: int
    flipped_encoded: int
    numerical_perturbation: float
    clean_loss: float
    faulted_loss: float
    signed_loss_difference: float
    positive_loss_damage: float
    absolute_loss_difference: float
    clean_accuracy: float
    faulted_accuracy: float
    accuracy_damage: float
    prediction_change_count: int | None
    evaluation_sample_sha256: str
    runtime_seconds: float
    inclusion_rule: str
    sampling_probability: float
    score_quantile: int | None

    @property
    def identity(self) -> BitIdentity:
        return BitIdentity(self.parameter, self.flat_index, self.bit_index)

    def to_dict(self):
        return asdict(self)


def select_all(identities: Sequence[BitIdentity]) -> tuple[CandidateSelection, ...]:
    return tuple(CandidateSelection(value, "all", 1.0) for value in identities)


def uniform_candidates(
    identities: Sequence[BitIdentity], count: int, *, seed: int
) -> tuple[CandidateSelection, ...]:
    """Uniform sample without replacement from the complete eligible population."""

    universe = tuple(sorted(identities))
    if not 1 <= count <= len(universe):
        raise ValueError("candidate count must be within the eligible population")
    generator = np.random.default_rng(int(seed))
    selected = generator.choice(len(universe), size=count, replace=False)
    probability = count / len(universe)
    return tuple(
        CandidateSelection(universe[int(index)], "uniform_without_replacement", probability)
        for index in selected
    )


def stratified_candidates(
    identities: Sequence[BitIdentity],
    count: int,
    *,
    seed: int,
    scores: Mapping[BitIdentity, float] | None = None,
    quantile_count: int = 4,
) -> tuple[CandidateSelection, ...]:
    """Deterministically sample across layer, class, bit, and score quantile.

    Sampling is uniform without replacement inside each stratum. Returned
    probabilities are conditional stratum inclusion probabilities; metrics over
    this selected set must remain labelled sample-limited.
    """

    universe = tuple(sorted(identities))
    if not 1 <= count <= len(universe):
        raise ValueError("candidate count must be within the eligible population")
    quantiles: dict[BitIdentity, int | None] = {identity: None for identity in universe}
    if scores is not None:
        if set(scores) != set(universe):
            raise ValueError("candidate score population differs from eligible population")
        ordered = sorted(universe, key=lambda value: (float(scores[value]), value))
        for index, identity in enumerate(ordered):
            quantiles[identity] = min(quantile_count - 1, index * quantile_count // len(ordered))
    strata: dict[tuple[str, str, int, int | None], list[BitIdentity]] = {}
    for identity in universe:
        key = (layer_name(identity.parameter), parameter_class(identity.parameter), identity.bit_index, quantiles[identity])
        strata.setdefault(key, []).append(identity)
    generator = np.random.default_rng(int(seed))
    keys = sorted(strata)
    shuffled: dict[tuple[str, str, int, int | None], list[BitIdentity]] = {}
    for key in keys:
        values = list(strata[key])
        generator.shuffle(values)
        shuffled[key] = values
    selected: list[BitIdentity] = []
    positions = {key: 0 for key in keys}
    while len(selected) < count:
        progressed = False
        for key in keys:
            position = positions[key]
            if position < len(shuffled[key]):
                selected.append(shuffled[key][position])
                positions[key] += 1
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            break
    per_stratum_selected: dict[tuple[str, str, int, int | None], int] = {}
    for identity in selected:
        key = (layer_name(identity.parameter), parameter_class(identity.parameter), identity.bit_index, quantiles[identity])
        per_stratum_selected[key] = per_stratum_selected.get(key, 0) + 1
    result = []
    for identity in selected:
        key = (layer_name(identity.parameter), parameter_class(identity.parameter), identity.bit_index, quantiles[identity])
        probability = per_stratum_selected[key] / len(strata[key])
        result.append(CandidateSelection(identity, "stratified_layer_class_bit_score_quantile", probability, quantiles[identity]))
    return tuple(result)


def measure_single_bits(
    adapter,
    inputs: np.ndarray,
    labels: np.ndarray,
    candidates: Sequence[CandidateSelection],
    codec: BitCodec | None,
    *,
    evaluation_sample_sha256: str,
    progress=None,
) -> tuple[SingleBitMeasurement, ...]:
    """Measure exact isolated encoded-bit effects with restoration after each bit."""

    baseline = adapter.evaluate(inputs, labels)
    baseline_predictions = adapter.predict_classes(inputs) if hasattr(adapter, "predict_classes") else None
    if progress is not None:
        progress(0, len(candidates), None)
    original = adapter.values()
    measurements: list[SingleBitMeasurement] = []
    for candidate in candidates:
        identity = candidate.identity
        transition = getattr(adapter, "encoded_bit_transition", None)
        if callable(transition):
            (
                original_encoded,
                flipped_encoded,
                original_value,
                flipped_value,
                changed,
            ) = transition(identity)
        else:
            if codec is None:
                raise ValueError("empirical injection requires a codec or adapter bit transition")
            changed = adapter.values()
            flat = changed[identity.parameter].reshape(-1)
            original_encoded = int(codec.encode(flat[identity.flat_index]))
            flipped_encoded = int(codec.flip(np.uint64(original_encoded), identity.bit_index))
            original_value = float(codec.decode(np.uint64(original_encoded)))
            flipped_value = float(codec.decode(np.uint64(flipped_encoded)))
            flat[identity.flat_index] = flipped_value
        started = perf_counter()
        with adapter.temporary_values(changed):
            faulted = adapter.evaluate(inputs, labels)
            prediction_changes = None
            if baseline_predictions is not None:
                prediction_changes = int(np.sum(adapter.predict_classes(inputs) != baseline_predictions))
        elapsed = perf_counter() - started
        signed_loss = faulted.categorical_crossentropy - baseline.categorical_crossentropy
        measurement = SingleBitMeasurement(
                identity.parameter,
                identity.flat_index,
                identity.bit_index,
                layer_name(identity.parameter),
                parameter_class(identity.parameter),
                original_encoded,
                flipped_encoded,
                flipped_value - original_value,
                baseline.categorical_crossentropy,
                faulted.categorical_crossentropy,
                signed_loss,
                max(0.0, signed_loss),
                abs(signed_loss),
                baseline.accuracy,
                faulted.accuracy,
                baseline.accuracy - faulted.accuracy,
                prediction_changes,
                evaluation_sample_sha256,
                elapsed,
                candidate.inclusion_rule,
                candidate.sampling_probability,
                candidate.score_quantile,
            )
        measurements.append(measurement)
        if progress is not None:
            progress(len(measurements), len(candidates), measurement)
    restored = adapter.values()
    if any(not np.array_equal(original[name], restored[name]) for name in original):
        raise RuntimeError("model state changed during empirical single-bit sweep")
    return tuple(measurements)
