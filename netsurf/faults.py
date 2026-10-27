"""Explicit, replayable logical fault schedules and exact event accounting."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable, Mapping, Sequence

import numpy as np

from .identities import BitIdentity
from .protection import ProtectionMask
from .quantization import BitCodec


@dataclass(frozen=True)
class FaultModel:
    target: str
    location: str
    persistence: str
    distribution: str
    probability: float
    multiplicity: str = "independent_multi_bit"
    correlation: str = "none"
    recovery: str = "restore_after_trial"

    def __post_init__(self) -> None:
        if self.target not in {"weight", "activation"}:
            raise ValueError(f"unsupported target: {self.target}")
        if self.persistence not in {"per_inference", "persistent_trial", "bounded_activation"}:
            raise ValueError(f"unsupported persistence: {self.persistence}")
        if self.distribution != "independent_bernoulli":
            raise ValueError("v2 foundation currently supports independent_bernoulli only")
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("probability must be in [0, 1]")


@dataclass(frozen=True, order=True)
class FaultEvent:
    inference_index: int | None
    bit: BitIdentity

    def key(self) -> str:
        window = "persistent" if self.inference_index is None else str(self.inference_index)
        return f"{window}:{self.bit.key()}"


@dataclass(frozen=True)
class FaultSchedule:
    model: FaultModel
    seed: int
    events: tuple[FaultEvent, ...]
    universe_size: int
    inference_count: int

    @classmethod
    def generate(
        cls,
        model: FaultModel,
        universe: Sequence[BitIdentity],
        *,
        seed: int,
        inference_count: int,
    ) -> "FaultSchedule":
        if inference_count <= 0:
            raise ValueError("inference_count must be positive")
        generator = np.random.default_rng(int(seed))
        events: list[FaultEvent] = []
        windows = inference_count if model.persistence == "per_inference" else 1
        for window in range(windows):
            selected = generator.random(len(universe)) < model.probability
            inference_index = window if model.persistence == "per_inference" else None
            events.extend(FaultEvent(inference_index, bit) for bit, hit in zip(universe, selected) if hit)
        return cls(model, int(seed), tuple(events), len(universe), inference_count)

    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps([event.key() for event in self.events], separators=(",", ":")).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class InjectionReport:
    schedule_hash: str
    proposed_events: int
    applied_events: int
    suppressed_events: int
    unique_applied_bits: int


def apply_persistent_schedule(
    values: Mapping[str, np.ndarray],
    schedule: FaultSchedule,
    protection: ProtectionMask,
    codec: BitCodec,
) -> tuple[dict[str, np.ndarray], InjectionReport]:
    """Return a corrupted copy for a persistent logical-weight schedule."""

    if schedule.model.persistence != "persistent_trial":
        raise ValueError("schedule is not persistent_trial")
    original_encoded = {name: codec.encode(value).reshape(-1).copy() for name, value in values.items()}
    flipped_encoded = {name: value.copy() for name, value in original_encoded.items()}
    applied: list[BitIdentity] = []
    suppressed = 0
    for event in schedule.events:
        identity = event.bit
        if identity in protection.protected:
            suppressed += 1
            continue
        if identity.parameter not in flipped_encoded:
            raise KeyError(f"unknown fault target {identity.parameter}")
        target = flipped_encoded[identity.parameter]
        if not 0 <= identity.flat_index < target.size:
            raise IndexError(identity)
        target[identity.flat_index] ^= np.uint64(1 << identity.bit_index)
        applied.append(identity)
    corrupted = {}
    for name, codes in flipped_encoded.items():
        original = np.asarray(values[name])
        delta = codec.decode(codes) - codec.decode(original_encoded[name])
        corrupted[name] = (original.reshape(-1) + delta).reshape(original.shape).astype(original.dtype)
    return corrupted, InjectionReport(
        schedule_hash=schedule.digest(),
        proposed_events=len(schedule.events),
        applied_events=len(applied),
        suppressed_events=suppressed,
        unique_applied_bits=len(set(applied)),
    )


def apply_persistent_schedule_encoded_state(
    values: Mapping[str, np.ndarray],
    schedule: FaultSchedule,
    protection: ProtectionMask,
    codec: BitCodec,
) -> tuple[dict[str, np.ndarray], InjectionReport]:
    """Apply XOR events to an exact encoded-state baseline.

    Unlike :func:`apply_persistent_schedule`, this function does not retain an
    off-lattice float shadow.  Every input tensor must already be exactly
    representable by ``codec`` and the returned state is the decoded XOR
    result.  This separate entry point keeps the recovered legacy
    delta-on-float semantics unchanged while making exact stored-state
    conformance testable.
    """

    if schedule.model.persistence != "persistent_trial":
        raise ValueError("schedule is not persistent_trial")
    encoded: dict[str, np.ndarray] = {}
    shapes: dict[str, tuple[int, ...]] = {}
    dtypes: dict[str, np.dtype] = {}
    for name, value in values.items():
        array = np.asarray(value)
        quantized = codec.quantize(array)
        if not np.array_equal(array.astype(np.float64), quantized):
            raise ValueError(f"{name} is not exactly representable by {codec.format}")
        encoded[name] = codec.encode(array).reshape(-1).copy()
        shapes[name] = tuple(array.shape)
        dtypes[name] = array.dtype

    applied: list[BitIdentity] = []
    suppressed = 0
    for event in schedule.events:
        identity = event.bit
        if identity in protection.protected:
            suppressed += 1
            continue
        if identity.parameter not in encoded:
            raise KeyError(f"unknown fault target {identity.parameter}")
        target = encoded[identity.parameter]
        if not 0 <= identity.flat_index < target.size:
            raise IndexError(identity)
        target[identity.flat_index] ^= np.uint64(1 << identity.bit_index)
        applied.append(identity)

    corrupted = {
        name: codec.decode(codes).reshape(shapes[name]).astype(dtypes[name])
        for name, codes in encoded.items()
    }
    return corrupted, InjectionReport(
        schedule_hash=schedule.digest(),
        proposed_events=len(schedule.events),
        applied_events=len(applied),
        suppressed_events=suppressed,
        unique_applied_bits=len(set(applied)),
    )


def apply_per_inference_schedule(
    values: Mapping[str, np.ndarray],
    schedule: FaultSchedule,
    protection: ProtectionMask,
    codec: BitCodec,
    inference_index: int,
) -> tuple[dict[str, np.ndarray], InjectionReport]:
    """Apply exactly one recorded per-inference window, independent of batching."""

    if schedule.model.persistence != "per_inference":
        raise ValueError("schedule is not per_inference")
    if not 0 <= inference_index < schedule.inference_count:
        raise IndexError(inference_index)
    selected = tuple(event for event in schedule.events if event.inference_index == inference_index)
    persistent = FaultSchedule(
        FaultModel(
            target=schedule.model.target,
            location=schedule.model.location,
            persistence="persistent_trial",
            distribution=schedule.model.distribution,
            probability=schedule.model.probability,
            multiplicity=schedule.model.multiplicity,
            correlation=schedule.model.correlation,
            recovery=schedule.model.recovery,
        ),
        schedule.seed,
        tuple(FaultEvent(None, event.bit) for event in selected),
        schedule.universe_size,
        1,
    )
    corrupted, report = apply_persistent_schedule(values, persistent, protection, codec)
    return corrupted, InjectionReport(
        schedule.digest(), report.proposed_events, report.applied_events,
        report.suppressed_events, report.unique_applied_bits,
    )
