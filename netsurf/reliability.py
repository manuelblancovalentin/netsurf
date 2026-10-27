"""Explicit application-level reliability measurements for fixed fault states."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .incremental import DenseReluDenseIncrementalEvaluator, FaultState, metrics_from_logits
from .pruning import topology


@dataclass(frozen=True)
class ReliabilityMeasurement:
    clean_accuracy: float
    clean_crossentropy: float
    faulty_accuracy: float
    faulty_crossentropy: float
    accuracy_degradation: float
    crossentropy_increase: float
    prediction_change_probability: float
    applied_fault_bits: int
    eligible_bits: int
    logical_ber_realization: float
    hidden_width: int
    parameter_count: int
    encoded_bit_count: int
    macs_per_inference: int
    encoded_memory_bytes: int
    interpretation: str = "fixed persistent logical parameter-fault state; not a physical SEU rate"

    def to_dict(self):
        return asdict(self)


def evaluate_persistent_fault_state(adapter, codec, inputs, labels, state: FaultState) -> ReliabilityMeasurement:
    """Measure one persistent XOR state against the same model's clean output."""

    evaluator = DenseReluDenseIncrementalEvaluator(adapter, codec)
    clean_logits = evaluator.full_forward(inputs).logits
    faulty_logits = evaluator.full_forward(inputs, state).logits
    clean = metrics_from_logits(clean_logits, labels)
    faulty = metrics_from_logits(faulty_logits, labels)
    structure = topology(adapter.values(), codec.format.total_bits)
    changed = float(np.mean(clean.predictions != faulty.predictions))
    return ReliabilityMeasurement(
        clean.accuracy, clean.categorical_crossentropy,
        faulty.accuracy, faulty.categorical_crossentropy,
        clean.accuracy - faulty.accuracy,
        faulty.categorical_crossentropy - clean.categorical_crossentropy,
        changed, len(state.bits), structure.encoded_bit_count,
        len(state.bits) / structure.encoded_bit_count if structure.encoded_bit_count else 0.0,
        structure.hidden_width, structure.parameter_count, structure.encoded_bit_count,
        structure.macs_per_inference, structure.encoded_memory_bytes,
    )
