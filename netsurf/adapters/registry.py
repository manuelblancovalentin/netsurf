"""Explicit benchmark-adapter registry with no eager framework imports."""

from __future__ import annotations

from ..experiment_config import ExperimentConfiguration
from ..quantization import BitCodec
from .protocol import BenchmarkAdapter


def create_adapter(
    configuration: ExperimentConfiguration, codec: BitCodec
) -> BenchmarkAdapter:
    name = configuration.benchmark.adapter
    if name == "legacy_mnist_lite":
        from .legacy_mnist import LegacyMnistLiteAdapter

        return LegacyMnistLiteAdapter(
            configuration.benchmark.checkpoint,
            configuration.benchmark.dataset,
            codec,
        )
    raise ValueError(f"unsupported benchmark adapter: {name}")
