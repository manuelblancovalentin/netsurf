"""Small independently verifiable quantized NumPy reference network."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterator, Mapping

import numpy as np

from ..identities import BitIdentity, ParameterSpec, enumerate_bits
from ..quantization import BitCodec


@dataclass(frozen=True)
class Evaluation:
    accuracy: float
    categorical_crossentropy: float
    sample_count: int


class NumpyReferenceAdapter:
    """Two-layer classifier whose live state is always exactly quantized."""

    def __init__(self, checkpoint: Path, dataset: Path, codec: BitCodec):
        self.checkpoint, self.dataset_path, self.codec = Path(checkpoint), Path(dataset), codec
        self._values: dict[str, np.ndarray] = {}
        self._inputs: np.ndarray | None = None
        self._labels: np.ndarray | None = None

    def load(self) -> None:
        checkpoint = json.loads(self.checkpoint.read_text(encoding="utf-8"))
        dataset = json.loads(self.dataset_path.read_text(encoding="utf-8"))
        if checkpoint.get("quantization") != str(self.codec.format):
            raise ValueError("checkpoint quantization does not match experiment")
        self._values = {name: self.codec.quantize(value) for name, value in checkpoint["parameters"].items()}
        self._inputs = np.asarray(dataset["inputs"], dtype=np.float64)
        self._labels = np.asarray(dataset["labels"], dtype=np.int64)

    def _require_loaded(self) -> None:
        if not self._values or self._inputs is None or self._labels is None:
            raise RuntimeError("adapter.load() must be called first")

    def dataset(self, count: int, *, role: str = "evaluation") -> tuple[np.ndarray, np.ndarray]:
        self._require_loaded()
        if role not in {"evaluation", "ranking"}:
            raise ValueError(f"unsupported dataset role: {role}")
        if not 1 <= count <= len(self._labels):
            raise ValueError(f"count must be in [1,{len(self._labels)}]")
        return self._inputs[:count].copy(), self._labels[:count].copy()

    def values(self) -> dict[str, np.ndarray]:
        self._require_loaded()
        return {name: value.copy() for name, value in self._values.items()}

    def parameter_specs(self) -> tuple[ParameterSpec, ...]:
        return tuple(ParameterSpec(name, tuple(value.shape)) for name, value in self.values().items())

    def bit_identities(self) -> tuple[BitIdentity, ...]:
        return enumerate_bits(self.parameter_specs(), self.codec.format.total_bits)

    @contextmanager
    def temporary_values(self, values: Mapping[str, np.ndarray]) -> Iterator[None]:
        before = self.values()
        expected = set(before)
        if set(values) != expected:
            raise ValueError(f"parameter mismatch: expected {sorted(expected)}")
        self._values = {name: np.asarray(values[name], dtype=np.float64).copy() for name in before}
        try:
            yield
        finally:
            self._values = before

    def quantized_values(self) -> dict[str, np.ndarray]:
        return {name: self.codec.quantize(value) for name, value in self.values().items()}

    def _forward(self, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hidden_pre = inputs @ self._values["dense0/kernel"] + self._values["dense0/bias"]
        hidden = np.maximum(hidden_pre, 0.0)
        logits = hidden @ self._values["dense1/kernel"] + self._values["dense1/bias"]
        return hidden, logits

    def evaluate(self, inputs: np.ndarray, labels: np.ndarray, *, batch_size: int = 256) -> Evaluation:
        _, logits = self._forward(np.asarray(inputs, dtype=np.float64))
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        probabilities = np.exp(shifted) / np.sum(np.exp(shifted), axis=1, keepdims=True)
        selected = np.maximum(probabilities[np.arange(len(labels)), labels], 1e-300)
        return Evaluation(float(np.mean(np.argmax(logits, axis=1) == labels)), float(np.mean(-np.log(selected))), len(labels))

    def gradients(self, inputs: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
        inputs = np.asarray(inputs, dtype=np.float64)
        hidden, logits = self._forward(inputs)
        probabilities = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        probabilities[np.arange(len(labels)), labels] -= 1.0
        dlogits = probabilities / len(labels)
        dense1_kernel = hidden.T @ dlogits
        dense1_bias = dlogits.sum(axis=0)
        dhidden = dlogits @ self._values["dense1/kernel"].T
        hidden_pre = inputs @ self._values["dense0/kernel"] + self._values["dense0/bias"]
        dhidden[hidden_pre <= 0.0] = 0.0
        return {"dense0/kernel": inputs.T @ dhidden, "dense0/bias": dhidden.sum(axis=0),
                "dense1/kernel": dense1_kernel, "dense1/bias": dense1_bias}

    def polar_impacts(self, inputs: np.ndarray, *, activation_source: str):
        raise ValueError("legacy_qpolargrad is not supported by numpy_reference")

    def exhaustive_single_bit_impacts(self, inputs: np.ndarray, labels: np.ndarray) -> dict[BitIdentity, float]:
        """Measured loss change for every eligible encoded bit."""
        baseline = self.evaluate(inputs, labels).categorical_crossentropy
        original = self.values()
        result: dict[BitIdentity, float] = {}
        for identity in self.bit_identities():
            changed = self.values()
            flat = changed[identity.parameter].reshape(-1)
            encoded = self.codec.encode(flat[identity.flat_index])
            flat[identity.flat_index] = self.codec.decode(self.codec.flip(encoded, identity.bit_index))
            with self.temporary_values(changed):
                result[identity] = self.evaluate(inputs, labels).categorical_crossentropy - baseline
        if any(not np.array_equal(original[name], self._values[name]) for name in original):
            raise RuntimeError("state changed during exhaustive injection")
        return result
