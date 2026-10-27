"""Structured Dense--ReLU--Dense pruning with explicit physical topology changes.

The grouped Taylor score is an architecture-specific adaptation of the
GroupTaylorImportance workflow used by Vinck et al.  A hidden neuron is one
dependency group: its incoming kernel column, bias scalar, and outgoing kernel
row.  RAP is not implemented here; the separately named reliability-aware
composition is a NetSuRF research method with independently ablatable terms.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np

from .identities import BitIdentity
from .quantization import BitCodec


PARAMETERS = ("dense0/kernel", "dense0/bias", "dense1/kernel", "dense1/bias")


@dataclass(frozen=True)
class DenseTopology:
    input_features: int
    hidden_width: int
    output_features: int
    parameter_count: int
    encoded_bit_count: int
    macs_per_inference: int
    encoded_memory_bytes: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class ReliabilityAwareComponents:
    clean_preservation: tuple[float, ...]
    bit_susceptibility: tuple[float, ...]
    pruning_cost: tuple[float, ...]
    combined_prune_cost: tuple[float, ...]
    definition: str = "netsurf-grouped-clean-minus-susceptibility-plus-cost/v1"


def validate_dense_relu_dense(values: Mapping[str, np.ndarray]) -> DenseTopology:
    if set(values) != set(PARAMETERS):
        raise ValueError(f"expected parameters {PARAMETERS}, found {tuple(sorted(values))}")
    w0, b0 = np.asarray(values[PARAMETERS[0]]), np.asarray(values[PARAMETERS[1]])
    w1, b1 = np.asarray(values[PARAMETERS[2]]), np.asarray(values[PARAMETERS[3]])
    if w0.ndim != 2 or b0.ndim != 1 or w1.ndim != 2 or b1.ndim != 1:
        raise ValueError("Dense-ReLU-Dense parameters must have ranks 2,1,2,1")
    if w0.shape[1] != b0.size or w1.shape[0] != b0.size or w1.shape[1] != b1.size:
        raise ValueError(
            f"inconsistent hidden/output dimensions: {w0.shape}, {b0.shape}, {w1.shape}, {b1.shape}"
        )
    if b0.size < 1 or b1.size < 1:
        raise ValueError("hidden and output widths must be positive")
    parameters = sum(np.asarray(value).size for value in values.values())
    return DenseTopology(
        int(w0.shape[0]), int(b0.size), int(b1.size), int(parameters), 0,
        int(w0.shape[0] * b0.size + b0.size * b1.size), 0,
    )


def topology(values: Mapping[str, np.ndarray], total_bits: int) -> DenseTopology:
    base = validate_dense_relu_dense(values)
    bits = base.parameter_count * int(total_bits)
    return DenseTopology(
        base.input_features, base.hidden_width, base.output_features,
        base.parameter_count, bits, base.macs_per_inference, (bits + 7) // 8,
    )


def _group_arrays(values: Mapping[str, np.ndarray], hidden: int) -> tuple[np.ndarray, ...]:
    return (
        np.asarray(values["dense0/kernel"])[:, hidden],
        np.asarray(values["dense0/bias"])[hidden:hidden + 1],
        np.asarray(values["dense1/kernel"])[hidden, :],
    )


def grouped_magnitude_importance(
    values: Mapping[str, np.ndarray], *, norm: str = "l1"
) -> np.ndarray:
    """Return one physical dependency-group magnitude per hidden neuron."""

    shape = validate_dense_relu_dense(values)
    if norm not in {"l1", "l2"}:
        raise ValueError("magnitude norm must be 'l1' or 'l2'")
    scores = []
    for hidden in range(shape.hidden_width):
        vector = np.concatenate([value.reshape(-1) for value in _group_arrays(values, hidden)])
        scores.append(np.sum(np.abs(vector)) if norm == "l1" else np.sqrt(np.sum(vector * vector)))
    return np.asarray(scores, dtype=np.float64)


def grouped_taylor_importance(
    values: Mapping[str, np.ndarray], gradients: Mapping[str, np.ndarray]
) -> np.ndarray:
    """Sum ``abs(parameter * dL/dparameter)`` over each dependency group.

    This follows Torch-Pruning's elementwise first-order GroupTaylorImportance
    convention, adapted to Keras kernel layout and explicitly including the
    hidden bias and the downstream input-channel dependency.
    """

    shape = validate_dense_relu_dense(values)
    validate_dense_relu_dense(gradients)
    if any(np.asarray(values[name]).shape != np.asarray(gradients[name]).shape for name in PARAMETERS):
        raise ValueError("gradient shapes must equal parameter shapes")
    scores = []
    for hidden in range(shape.hidden_width):
        terms = []
        for value, gradient in zip(_group_arrays(values, hidden), _group_arrays(gradients, hidden)):
            terms.append(np.abs(np.asarray(value, dtype=np.float64) * np.asarray(gradient, dtype=np.float64)))
        scores.append(sum(float(np.sum(term)) for term in terms))
    return np.asarray(scores, dtype=np.float64)


def grouped_bit_susceptibility(
    scores: Mapping[BitIdentity, float], hidden_width: int, *, reduction: str = "mean"
) -> np.ndarray:
    """Aggregate already-defined bit susceptibility without redefining it."""

    grouped: list[list[float]] = [[] for _ in range(hidden_width)]
    for identity, score in scores.items():
        if identity.parameter == "dense0/kernel":
            hidden = identity.flat_index % hidden_width
        elif identity.parameter == "dense0/bias":
            hidden = identity.flat_index
        elif identity.parameter == "dense1/kernel":
            # The output width is inferred from flat-index runs per hidden row.
            continue
        else:
            continue
        if 0 <= hidden < hidden_width:
            grouped[hidden].append(float(score))
    # Add outgoing identities in a second pass after inferring output width.
    outgoing = [identity for identity in scores if identity.parameter == "dense1/kernel"]
    if outgoing:
        scalars = max(identity.flat_index for identity in outgoing) + 1
        if scalars % hidden_width:
            raise ValueError("outgoing bit identities do not match hidden width")
        output_width = scalars // hidden_width
        for identity in outgoing:
            grouped[identity.flat_index // output_width].append(float(scores[identity]))
    if reduction not in {"mean", "sum", "max"}:
        raise ValueError("susceptibility reduction must be mean, sum, or max")
    reduce = {"mean": np.mean, "sum": np.sum, "max": np.max}[reduction]
    return np.asarray([float(reduce(row)) if row else 0.0 for row in grouped], dtype=np.float64)


def _unit_scale(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    spread = float(np.max(array) - np.min(array)) if array.size else 0.0
    return np.zeros_like(array) if spread == 0 else (array - np.min(array)) / spread


def netsurf_reliability_aware_importance(
    clean_preservation: Sequence[float],
    bit_susceptibility: Sequence[float],
    pruning_cost: Sequence[float],
    *,
    clean_weight: float = 1.0,
    susceptibility_weight: float = 1.0,
    cost_weight: float = 0.0,
) -> ReliabilityAwareComponents:
    """Compose ablatable group terms; lower values are pruned first.

    High clean-preservation cost discourages removal. High susceptibility makes
    removal attractive, hence its negative sign. High released resource cost
    can make removal attractive when ``cost_weight`` is positive. This is a
    NetSuRF proposal, not RAP.
    """

    clean = _unit_scale(clean_preservation)
    susceptibility = _unit_scale(bit_susceptibility)
    cost = _unit_scale(pruning_cost)
    if not (clean.shape == susceptibility.shape == cost.shape):
        raise ValueError("reliability-aware component shapes must match")
    combined = clean_weight * clean - susceptibility_weight * susceptibility - cost_weight * cost
    return ReliabilityAwareComponents(
        tuple(map(float, clean)), tuple(map(float, susceptibility)), tuple(map(float, cost)),
        tuple(map(float, combined)),
    )


def select_pruned_neurons(scores: Sequence[float], remove_count: int) -> tuple[int, ...]:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not 0 <= remove_count < len(values):
        raise ValueError("remove_count must preserve at least one hidden neuron")
    if not np.all(np.isfinite(values)):
        raise ValueError("pruning scores must be finite")
    # Stable index is the explicit deterministic tie break.
    return tuple(map(int, np.lexsort((np.arange(len(values)), values))[:remove_count]))


def remove_hidden_neurons(
    values: Mapping[str, np.ndarray], remove: Sequence[int]
) -> dict[str, np.ndarray]:
    shape = validate_dense_relu_dense(values)
    removed = tuple(sorted(set(map(int, remove))))
    if len(removed) != len(tuple(remove)) or any(index < 0 or index >= shape.hidden_width for index in removed):
        raise ValueError("removed hidden-neuron indices must be unique and in range")
    if len(removed) >= shape.hidden_width:
        raise ValueError("structured pruning must retain at least one hidden neuron")
    keep = np.ones(shape.hidden_width, dtype=bool)
    keep[list(removed)] = False
    result = {
        "dense0/kernel": np.asarray(values["dense0/kernel"])[:, keep].copy(),
        "dense0/bias": np.asarray(values["dense0/bias"])[keep].copy(),
        "dense1/kernel": np.asarray(values["dense1/kernel"])[keep, :].copy(),
        "dense1/bias": np.asarray(values["dense1/bias"]).copy(),
    }
    validate_dense_relu_dense(result)
    return result


def paper_width_schedule(
    initial_width: int, *, iterations: int = 10, fraction: float = 0.149
) -> tuple[int, ...]:
    if initial_width < 1 or iterations < 1 or not 0.0 < fraction < 1.0:
        raise ValueError("invalid paper-fraction schedule")
    widths = []
    current = int(initial_width)
    # Round each ideal cumulative target rather than compounding rounded stage
    # widths.  This preserves the paper's stated 20% endpoint for
    # 0.851**10 while still producing legal integer widths for small models.
    for stage in range(1, iterations + 1):
        target = max(1, int(round(initial_width * (1.0 - fraction) ** stage)))
        if target >= current and current > 1:
            target = current - 1
        widths.append(target)
        current = target
    return tuple(widths)


def validate_target_widths(initial_width: int, targets: Sequence[int]) -> tuple[int, ...]:
    result = tuple(map(int, targets))
    if not result or any(value < 1 for value in result):
        raise ValueError("target widths must be a nonempty positive sequence")
    if any(right >= left for left, right in zip((initial_width, *result[:-1]), result)):
        raise ValueError("target widths must strictly decrease from the source width")
    return result


def exact_quantization(values: Mapping[str, np.ndarray], codec: BitCodec) -> dict[str, np.ndarray]:
    quantized = {name: codec.quantize(value).astype(np.float32) for name, value in values.items()}
    for name, value in quantized.items():
        if not np.array_equal(codec.decode(codec.encode(value)).astype(np.float32), value):
            raise RuntimeError(f"post-pruning quantization is not exact: {name}")
    return quantized


def checkpoint_pruning_contract(adapter) -> dict[str, object] | None:
    """Return M5-only topology/provenance fields for scientific manifests."""

    metadata = getattr(adapter, "metadata", {})
    if metadata.get("checkpoint_kind") not in {
        "physically_structured_pruned", "independently_trained_width_control",
    }:
        return None
    return {
        "checkpoint_kind": metadata["checkpoint_kind"],
        "method_classification": metadata["method_classification"],
        "source_checkpoint_sha256": metadata["source_checkpoint"]["sha256"],
        "pruning_scientific_payload_sha256": metadata["scientific_payload_sha256"],
        "topology": adapter.structural_metrics(),
    }
