"""Static, inspectable registry for tested NetSuRF benchmark architectures."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from ..paths import dataset_root, workspace_root


@dataclass(frozen=True)
class BenchmarkDescriptor:
    name: str
    adapter: str
    task: str
    architecture: str
    parameter_count: int | None
    quantization: str
    quantization_backend: str
    dataset: str
    checkpoint: str
    supported_estimators: tuple[str, ...]
    supported_fault_targets: tuple[str, ...]
    supported_persistence: tuple[str, ...]
    hardware_backends: tuple[str, ...]
    validation: str
    dependencies: tuple[str, ...]
    preparation_required: bool
    eligible_parameter_count: int | None = None
    eligible_bit_count: int | None = None

    def readiness(self) -> tuple[str, tuple[str, ...]]:
        reasons: list[str] = []
        for dependency in self.dependencies:
            if find_spec(dependency) is None:
                reasons.append(f"missing dependency: {dependency}")
        if self.name == "mnist_quantized":
            if not (dataset_root() / "mnist" / "mnist.npz").is_file():
                reasons.append("missing dataset: mnist/mnist.npz")
            if not (workspace_root() / "benchmarks" / "mnist_quantized" / "checkpoint.npz").is_file():
                reasons.append("missing checkpoint: run benchmarks prepare")
        if reasons:
            status = "missing dependency" if any("dependency" in value for value in reasons) else "missing checkpoint" if any("checkpoint" in value for value in reasons) else "missing dataset"
            return status, tuple(reasons)
        return ("ready" if self.validation == "scientifically validated" else "experimental"), ()

    def to_dict(self) -> dict[str, Any]:
        status, reasons = self.readiness()
        return {**asdict(self), "readiness": status, "readiness_reasons": list(reasons)}


_REGISTRY = {
    "numpy_activation_cancellation": BenchmarkDescriptor(
        "numpy_activation_cancellation", "numpy_activation_cancellation",
        "controlled signed-activation two-class classification",
        "2 inputs -> quantized Dense(3) -> ReLU -> quantized Dense(2) logits", 17,
        "q<6,2,1>", "netsurf_bitcodec_v1", "packaged signed or nonnegative 12-sample controls",
        "packaged deterministic JSON parameters",
        ("legacy_qpolargrad", "clean_activation_qpolargrad", "uncancelled_activation_drift", "first_order_gradient_delta", "per_sample_first_order_magnitude", "per_sample_positive_damage", "exact_diagonal_second_order", "hutchinson_diagonal_second_order"),
        ("weight",), ("persistent_trial", "per_inference"), (),
        "scientifically controlled unit benchmark", (), False,
    ),
    "numpy_reference": BenchmarkDescriptor(
        "numpy_reference", "numpy_reference", "two-class fixed synthetic classification",
        "2 inputs -> quantized Dense(3) -> ReLU -> quantized Dense(2) logits", 17,
        "q<6,2,1>", "netsurf_bitcodec_v1", "packaged fixed 8-sample dataset",
        "packaged deterministic JSON parameters", ("msb_first", "random", "first_order_gradient_delta", "per_sample_first_order_magnitude", "per_sample_positive_damage", "exact_diagonal_second_order", "hutchinson_diagonal_second_order"),
        ("weight",), ("persistent_trial", "per_inference"), (), "scientifically validated", (), False,
    ),
    "mnist_quantized": BenchmarkDescriptor(
        "mnist_quantized", "mnist_quantized", "MNIST 10-class image classification",
        "Flatten(28x28) -> Dense(32) -> ReLU -> Dense(10) logits", 25450,
        "q<6,2,1>", "netsurf_bitcodec_v1 over Keras-trained parameters", "/ml/datasets/mnist/mnist.npz",
        "workspace/benchmarks/mnist_quantized/checkpoint.npz", ("msb_first", "random", "first_order_gradient_delta", "per_sample_first_order_magnitude", "per_sample_positive_damage", "hutchinson_diagonal_second_order", "legacy_qpolargrad", "clean_activation_qpolargrad", "uncancelled_activation_drift"),
        ("weight",), ("persistent_trial", "per_inference"), ("hls4ml Vitis: planned validation", "hls4ml Catapult: dense path only, planned validation"),
        "bounded software integration validated; hardware equivalence not validated", ("tensorflow", "keras"), True,
    ),
    "legacy_mnist_lite": BenchmarkDescriptor(
        "legacy_mnist_lite", "legacy_mnist_lite", "historical MNIST compatibility",
        "recovered 2025 checkpoint", None, "q<6,2,1>", "QKeras/Keras v2 historical",
        "external archived MNIST", "external archived checkpoint",
        ("legacy_qpolargrad", "msb_first", "random", "first_order_gradient_delta"), ("weight",),
        ("persistent_trial",), (), "legacy compatibility only", ("tensorflow",), False,
    ),
    "mnist_lite_exact_kb_v1": BenchmarkDescriptor(
        "mnist_lite_exact_kb_v1", "mnist_lite_exact_kb_v1",
        "MNIST 10-class classification; recovered topology with a new v2 representation",
        "Flatten(28x28) -> Dense(10) -> float alpha/beta -> quantized ReLU -> Dense(10) -> softmax",
        7980, "q<6,2,1> kernels/biases; float32 alpha/beta auxiliaries",
        "netsurf_bitcodec_v1 over explicitly converted kernel/bias tensors",
        "/ml/datasets/mnist/mnist.npz",
        "external archived checkpoint ba48aee8... with deterministic in-memory conversion",
        (
            "msb_first", "random", "gradient_magnitude",
            "first_order_gradient_delta", "per_sample_first_order_magnitude",
            "per_sample_positive_damage", "hutchinson_diagonal_second_order",
        ),
        ("weight",), ("persistent_trial", "per_inference"), (),
        "bounded exact-kernel/bias inference validated; not a manuscript reproduction",
        ("tensorflow",), False, eligible_parameter_count=7960, eligible_bit_count=47760,
    ),
}


def benchmark_list() -> tuple[BenchmarkDescriptor, ...]:
    return tuple(_REGISTRY[name] for name in sorted(_REGISTRY))


def benchmark_info(name: str) -> BenchmarkDescriptor:
    try:
        return _REGISTRY[name]
    except KeyError as error:
        raise KeyError(f"unknown benchmark {name!r}; available: {', '.join(sorted(_REGISTRY))}") from error
