"""Optional adapters for historical frameworks and artifacts."""

from .legacy_mnist import LegacyMnistLiteAdapter, MnistLiteExactKernelBiasAdapter
from .protocol import BenchmarkAdapter

__all__ = ["BenchmarkAdapter", "LegacyMnistLiteAdapter", "MnistLiteExactKernelBiasAdapter"]
