"""Optional adapters for historical frameworks and artifacts."""

from .legacy_mnist import LegacyMnistLiteAdapter
from .protocol import BenchmarkAdapter

__all__ = ["BenchmarkAdapter", "LegacyMnistLiteAdapter"]
