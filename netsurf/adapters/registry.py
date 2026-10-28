"""Explicit benchmark-adapter registry with no eager framework imports."""

from __future__ import annotations

from ..experiment_config import ExperimentConfiguration
from ..quantization import BitCodec
from .protocol import BenchmarkAdapter


def create_adapter(
    configuration: ExperimentConfiguration, codec: BitCodec | None
) -> BenchmarkAdapter:
    name = configuration.benchmark.adapter
    if name == "numpy_reference":
        from .numpy_reference import NumpyReferenceAdapter

        return NumpyReferenceAdapter(
            configuration.benchmark.checkpoint,
            configuration.benchmark.dataset,
            codec,
        )
    if name == "numpy_activation_cancellation":
        from .numpy_reference import NumpyActivationCancellationAdapter

        return NumpyActivationCancellationAdapter(
            configuration.benchmark.checkpoint,
            configuration.benchmark.dataset,
            codec,
        )
    if name == "mnist_quantized":
        from .mnist_quantized import MnistQuantizedAdapter

        return MnistQuantizedAdapter(
            configuration.benchmark.checkpoint,
            configuration.benchmark.dataset,
            codec,
        )
    if name == "legacy_mnist_lite":
        from .legacy_mnist import LegacyMnistLiteAdapter

        return LegacyMnistLiteAdapter(
            configuration.benchmark.checkpoint,
            configuration.benchmark.dataset,
            codec,
        )
    if name == "mnist_lite_exact_kb_v1":
        from .legacy_mnist import MnistLiteExactKernelBiasAdapter

        return MnistLiteExactKernelBiasAdapter(
            configuration.benchmark.checkpoint,
            configuration.benchmark.dataset,
            codec,
        )
    if name == "mlperf_tiny_ic_int8_v1":
        from .mlperf_tiny import MlperfTinyImageClassificationAdapter

        return MlperfTinyImageClassificationAdapter(
            configuration.benchmark.checkpoint,
            configuration.benchmark.dataset,
            codec,
            asset_lock=configuration.benchmark.asset_lock,
            inference_backend=configuration.benchmark.inference_backend,
            split=configuration.benchmark.split,
            injectable_parameters=configuration.benchmark.injectable_parameters,
        )
    raise ValueError(f"unsupported benchmark adapter: {name}")
