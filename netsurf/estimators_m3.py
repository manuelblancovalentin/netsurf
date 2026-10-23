"""M3 estimator definitions with signed predictions kept separate from scores."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np

from .identities import BitIdentity
from .quantization import BitCodec


@dataclass(frozen=True)
class EstimatorOutput:
    name: str
    implementation_version: str
    scores: Mapping[BitIdentity, float]
    signed_predictions: Mapping[BitIdentity, float] | None
    runtime_seconds: float
    metadata: Mapping[str, Any]


def _parameter_value(matrix: Mapping[str, np.ndarray], identity: BitIdentity) -> float:
    return float(matrix[identity.parameter].reshape(-1)[identity.flat_index])


def _bit_value(matrix: Mapping[str, np.ndarray], identity: BitIdentity) -> float:
    value = matrix[identity.parameter]
    return float(value.reshape(-1, value.shape[-1])[identity.flat_index, identity.bit_index])


def _deltas(values, codec: BitCodec):
    return {name: codec.perturbation_matrix(value) for name, value in values.items()}


def first_order(adapter, inputs, labels, identities, codec: BitCodec) -> EstimatorOutput:
    started = perf_counter()
    gradients = adapter.gradients(inputs, labels)
    deltas = _deltas(adapter.values(), codec)
    signed = {
        identity: _parameter_value(gradients, identity) * _bit_value(deltas, identity)
        for identity in identities
    }
    return EstimatorOutput(
        "first_order_gradient_delta",
        "signed-gradient-times-exact-delta/v2",
        {identity: abs(value) for identity, value in signed.items()},
        signed,
        perf_counter() - started,
        {
            "signed_expression": "dL/dw * delta_b",
            "ranking_expression": "abs(dL/dw * delta_b)",
            "gradient_aggregation": "gradient of mean loss",
        },
    )


def gradient_magnitude(adapter, inputs, labels, identities) -> EstimatorOutput:
    started = perf_counter()
    gradients = adapter.gradients(inputs, labels)
    scores = {identity: abs(_parameter_value(gradients, identity)) for identity in identities}
    return EstimatorOutput(
        "gradient_magnitude",
        "absolute-mean-loss-gradient/v2",
        scores,
        None,
        perf_counter() - started,
        {"expression": "abs(dL/dw); copied to every bit of the scalar"},
    )


def per_sample_first_order_family(
    adapter, inputs, labels, identities, codec: BitCodec, *, requested: Sequence[str]
) -> Mapping[str, EstimatorOutput]:
    """Compute absolute and positive per-example scores from one Jacobian."""

    supported = {"per_sample_first_order_magnitude", "per_sample_positive_damage"}
    unknown = set(requested) - supported
    if unknown:
        raise ValueError(f"unsupported per-example first-order variants: {sorted(unknown)}")
    started = perf_counter()
    gradients = adapter.per_example_gradients(inputs, labels)
    deltas = _deltas(adapter.values(), codec)
    absolute_scores, positive_scores = {}, {}
    for identity in identities:
        per_example = gradients[identity.parameter].reshape(len(labels), -1)[:, identity.flat_index]
        changes = per_example * _bit_value(deltas, identity)
        absolute_scores[identity] = float(np.mean(np.abs(changes)))
        positive_scores[identity] = float(np.mean(np.maximum(0.0, changes)))
    elapsed = perf_counter() - started
    shared = {
        "shared_computation": "one per-example gradient Jacobian and one exact-delta matrix",
        "runtime_accounting": "shared wall time repeated for dependent rows; do not sum",
    } if len(requested) > 1 else {}
    outputs = {
        "per_sample_first_order_magnitude": EstimatorOutput(
            "per_sample_first_order_magnitude",
            "mean-absolute-per-example-gradient-times-exact-delta/v1",
            absolute_scores, None, elapsed,
            {**shared, "expression": "mean_x(abs(dell_x/dw * delta_b))", "warning": "distinct from abs(mean_x(dell_x/dw) * delta_b)"},
        ),
        "per_sample_positive_damage": EstimatorOutput(
            "per_sample_positive_damage",
            "mean-positive-per-example-gradient-times-exact-delta/v1",
            positive_scores, None, elapsed,
            {**shared, "expression": "mean_x(max(0, dell_x/dw * delta_b))", "warning": "first-order positive per-example approximation; not exact aggregate loss damage"},
        ),
    }
    return {name: outputs[name] for name in requested}


def per_sample_first_order(adapter, inputs, labels, identities, codec: BitCodec) -> EstimatorOutput:
    return per_sample_first_order_family(
        adapter, inputs, labels, identities, codec,
        requested=("per_sample_first_order_magnitude",),
    )["per_sample_first_order_magnitude"]


def qpolargrad(adapter, inputs, labels, identities, codec: BitCodec, *, variant: str) -> EstimatorOutput:
    sources = {
        "legacy_qpolargrad": "globally_attacked_first_call",
        "clean_activation_qpolargrad": "clean",
        "uncancelled_activation_drift": "clean_uncancelled",
    }
    if variant not in sources:
        raise ValueError(f"unknown qPolarGrad variant: {variant}")
    started = perf_counter()
    gradients = adapter.gradients(inputs, labels)
    impacts = adapter.polar_impacts(inputs, activation_source=sources[variant])
    scores = {
        identity: abs(_bit_value(impacts, identity) * _parameter_value(gradients, identity))
        for identity in identities
    }
    expressions = {
        "legacy_qpolargrad": "abs(mean_x(layer_input_i under global all-bit attack) * delta_b * dL/dw)",
        "clean_activation_qpolargrad": "abs(mean_x(clean layer_input_i) * delta_b * dL/dw)",
        "uncancelled_activation_drift": "mean_x(abs(clean layer_input_i * delta_b)) * abs(dL/dw)",
    }
    return EstimatorOutput(
        variant,
        {
            "legacy_qpolargrad": "recovered-dense-first-call/v1",
            "clean_activation_qpolargrad": "clean-dense-mean-cancelled/v1",
            "uncancelled_activation_drift": "clean-dense-mean-absolute/v1",
        }[variant],
        scores,
        None,
        perf_counter() - started,
        {
            "expression": expressions[variant],
            "activation_source": sources[variant],
            "warning": "research heuristic; gradient already contains the layer input factor",
        },
    )


def _unflatten(values: Mapping[str, np.ndarray], flat: np.ndarray) -> dict[str, np.ndarray]:
    result, offset = {}, 0
    for name, value in values.items():
        size = value.size
        result[name] = flat[offset:offset + size].reshape(value.shape)
        offset += size
    return result


def exact_diagonal_second_order(adapter, inputs, labels, identities, codec: BitCodec) -> EstimatorOutput:
    if not hasattr(adapter, "exact_hessian"):
        raise ValueError("exact diagonal Hessian is supported only by a tractable adapter")
    started = perf_counter()
    values = adapter.values()
    gradients = adapter.gradients(inputs, labels)
    hessian = adapter.exact_hessian(inputs, labels)
    diagonal = _unflatten(values, np.diag(hessian))
    deltas = _deltas(values, codec)
    signed = {}
    for identity in identities:
        delta = _bit_value(deltas, identity)
        signed[identity] = _parameter_value(gradients, identity) * delta + 0.5 * _parameter_value(diagonal, identity) * delta * delta
    return EstimatorOutput(
        "exact_diagonal_second_order",
        "finite-difference-exact-diagonal/v1",
        {identity: abs(value) for identity, value in signed.items()},
        signed,
        perf_counter() - started,
        {
            "signed_expression": "g_b*delta_b + 0.5*H_bb*delta_b^2",
            "hessian": "full symmetric finite-difference Hessian of analytic gradient",
            "warning": "local approximation may fail for large encoded perturbations",
        },
    )


def hutchinson_second_order(
    adapter,
    inputs,
    labels,
    identities,
    codec: BitCodec,
    *,
    probes: int,
    seed: int,
) -> EstimatorOutput:
    if probes < 1:
        raise ValueError("Hutchinson probes must be positive")
    started = perf_counter()
    values = adapter.values()
    generator = np.random.default_rng(int(seed))
    samples: list[dict[str, np.ndarray]] = []
    for _ in range(probes):
        vectors = {
            name: generator.choice(np.asarray([-1.0, 1.0]), size=value.shape)
            for name, value in values.items()
        }
        hvp = adapter.hessian_vector_product(inputs, labels, vectors)
        samples.append({name: vectors[name] * hvp[name] for name in values})
    diagonal = {name: np.mean([sample[name] for sample in samples], axis=0) for name in values}
    if probes > 1:
        standard_error = {
            name: np.std([sample[name] for sample in samples], axis=0, ddof=1) / np.sqrt(probes)
            for name in values
        }
        se_rms = float(np.sqrt(np.mean(np.concatenate([value.reshape(-1) for value in standard_error.values()]) ** 2)))
        se_max = float(max(np.max(np.abs(value)) for value in standard_error.values()))
    else:
        se_rms = se_max = float("nan")
    gradients = adapter.gradients(inputs, labels)
    deltas = _deltas(values, codec)
    signed = {}
    for identity in identities:
        delta = _bit_value(deltas, identity)
        signed[identity] = _parameter_value(gradients, identity) * delta + 0.5 * _parameter_value(diagonal, identity) * delta * delta
    return EstimatorOutput(
        "hutchinson_diagonal_second_order",
        "rademacher-hvp-diagonal/v1",
        {identity: abs(value) for identity, value in signed.items()},
        signed,
        perf_counter() - started,
        {
            "signed_expression": "g_b*delta_b + 0.5*diag(H)_b*delta_b^2",
            "diagonal_estimator": "mean_v(v * H v), independent Rademacher probes",
            "probes": probes,
            "seed": seed,
            "diagonal_standard_error_rms": se_rms,
            "diagonal_standard_error_max": se_max,
            "warning": "local approximation; probe uncertainty excludes model/data uncertainty",
        },
    )


def compute_estimator(
    name: str,
    adapter,
    inputs,
    labels,
    identities: Sequence[BitIdentity],
    codec: BitCodec,
    *,
    seed: int = 0,
    probes: int = 16,
) -> EstimatorOutput:
    if name == "first_order_gradient_delta":
        return first_order(adapter, inputs, labels, identities, codec)
    if name == "gradient_magnitude":
        return gradient_magnitude(adapter, inputs, labels, identities)
    if name == "per_sample_first_order_magnitude":
        return per_sample_first_order(adapter, inputs, labels, identities, codec)
    if name == "per_sample_positive_damage":
        return per_sample_first_order_family(
            adapter, inputs, labels, identities, codec,
            requested=("per_sample_positive_damage",),
        )["per_sample_positive_damage"]
    if name in {"legacy_qpolargrad", "clean_activation_qpolargrad", "uncancelled_activation_drift"}:
        return qpolargrad(adapter, inputs, labels, identities, codec, variant=name)
    if name == "exact_diagonal_second_order":
        return exact_diagonal_second_order(adapter, inputs, labels, identities, codec)
    if name == "hutchinson_diagonal_second_order":
        return hutchinson_second_order(adapter, inputs, labels, identities, codec, probes=probes, seed=seed)
    raise ValueError(f"unsupported M3 estimator: {name}")


def compute_estimators(specifications, adapter, inputs, labels, identities, codec: BitCodec):
    """Compute configured estimators while sharing per-example gradients."""

    family_names = {
        specification.name for specification in specifications
        if specification.name in {"per_sample_first_order_magnitude", "per_sample_positive_damage"}
    }
    family = per_sample_first_order_family(
        adapter, inputs, labels, identities, codec, requested=tuple(sorted(family_names))
    ) if family_names else {}
    result = []
    for specification in specifications:
        output = family.get(specification.name)
        if output is None:
            output = compute_estimator(
                specification.name, adapter, inputs, labels, identities, codec,
                seed=specification.seed, probes=specification.probes,
            )
        result.append(output)
    return tuple(result)
