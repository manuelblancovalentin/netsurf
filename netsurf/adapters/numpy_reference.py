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
from ..samples import SamplePartition, sample_identity_digest


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

    def sample_partition(self, role: str, count: int, *, offset: int = 0):
        self._require_loaded()
        available = {
            "ranking": (0, 4),
            "empirical_validation": (1, 5, 6),
            "evaluation": (2, 3, 7),
        }
        if role not in available:
            raise ValueError(f"unsupported M3 sample role: {role}")
        if offset != 0:
            raise ValueError("numpy_reference has fixed partitions and supports offset=0 only")
        if not 1 <= count <= len(available[role]):
            raise ValueError(f"{role} count must be in [1,{len(available[role])}]")
        indices = available[role][:count]
        inputs = self._inputs[list(indices)].copy()
        labels = self._labels[list(indices)].copy()
        partition = SamplePartition(
            role,
            "packaged_fixed_all",
            indices,
            sample_identity_digest("packaged_fixed_all", indices, labels),
            "fixed reference samples; no model training performed",
        )
        return inputs, labels, partition

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

    def clean_activations(self, inputs: np.ndarray) -> dict[str, np.ndarray]:
        """Return explicit clean inputs to each dense layer."""

        inputs = np.asarray(inputs, dtype=np.float64)
        hidden, _ = self._forward(inputs)
        return {"dense0": inputs.copy(), "dense1": hidden.copy()}

    def evaluate(self, inputs: np.ndarray, labels: np.ndarray, *, batch_size: int = 256) -> Evaluation:
        _, logits = self._forward(np.asarray(inputs, dtype=np.float64))
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        probabilities = np.exp(shifted) / np.sum(np.exp(shifted), axis=1, keepdims=True)
        selected = np.maximum(probabilities[np.arange(len(labels)), labels], 1e-300)
        return Evaluation(float(np.mean(np.argmax(logits, axis=1) == labels)), float(np.mean(-np.log(selected))), len(labels))

    def predict_classes(self, inputs: np.ndarray) -> np.ndarray:
        return np.argmax(self._forward(np.asarray(inputs, dtype=np.float64))[1], axis=1)

    def gradients(self, inputs: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
        per_example = self.per_example_gradients(inputs, labels)
        return {name: np.mean(value, axis=0) for name, value in per_example.items()}

    def per_example_gradients(self, inputs: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
        inputs = np.asarray(inputs, dtype=np.float64)
        hidden, logits = self._forward(inputs)
        probabilities = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        probabilities[np.arange(len(labels)), labels] -= 1.0
        dlogits = probabilities
        dense1_kernel = hidden[:, :, None] * dlogits[:, None, :]
        dense1_bias = dlogits.copy()
        dhidden = dlogits @ self._values["dense1/kernel"].T
        hidden_pre = inputs @ self._values["dense0/kernel"] + self._values["dense0/bias"]
        dhidden[hidden_pre <= 0.0] = 0.0
        return {
            "dense0/kernel": inputs[:, :, None] * dhidden[:, None, :],
            "dense0/bias": dhidden,
            "dense1/kernel": dense1_kernel,
            "dense1/bias": dense1_bias,
        }

    def exact_hessian(self, inputs: np.ndarray, labels: np.ndarray, *, epsilon: float = 1e-5) -> np.ndarray:
        """Finite-difference the analytic mean-loss gradient on 17 parameters."""

        original = self.values()
        names = tuple(original)
        offsets = []
        for name in names:
            for flat_index in range(original[name].size):
                offsets.append((name, flat_index))
        hessian = np.empty((len(offsets), len(offsets)), dtype=np.float64)
        for column, (name, flat_index) in enumerate(offsets):
            plus, minus = self.values(), self.values()
            plus[name].reshape(-1)[flat_index] += epsilon
            minus[name].reshape(-1)[flat_index] -= epsilon
            with self.temporary_values(plus):
                high = self.gradients(inputs, labels)
            with self.temporary_values(minus):
                low = self.gradients(inputs, labels)
            high_flat = np.concatenate([high[value].reshape(-1) for value in names])
            low_flat = np.concatenate([low[value].reshape(-1) for value in names])
            hessian[:, column] = (high_flat - low_flat) / (2.0 * epsilon)
        if any(not np.array_equal(original[name], self._values[name]) for name in original):
            raise RuntimeError("state changed during exact Hessian calculation")
        return 0.5 * (hessian + hessian.T)

    def hessian_vector_product(self, inputs, labels, vectors):
        names = tuple(self._values)
        vector = np.concatenate([np.asarray(vectors[name]).reshape(-1) for name in names])
        product = self.exact_hessian(inputs, labels) @ vector
        result, offset = {}, 0
        for name in names:
            size = self._values[name].size
            result[name] = product[offset:offset + size].reshape(self._values[name].shape)
            offset += size
        return result


    def polar_impacts(self, inputs: np.ndarray, *, activation_source: str):
        inputs = np.asarray(inputs, dtype=np.float64)
        clean_hidden, _ = self._forward(inputs)
        if activation_source == "clean":
            dense_inputs = {"dense0": inputs, "dense1": clean_hidden}
            uncancelled = False
        elif activation_source == "globally_attacked_first_call":
            attacked = {}
            full_mask = np.uint64((1 << self.codec.format.total_bits) - 1)
            for name, value in self._values.items():
                attacked[name] = self.codec.decode(np.bitwise_xor(self.codec.encode(value), full_mask))
            hidden_pre = inputs @ attacked["dense0/kernel"] + attacked["dense0/bias"]
            dense_inputs = {"dense0": inputs, "dense1": np.maximum(hidden_pre, 0.0)}
            uncancelled = False
        elif activation_source == "clean_uncancelled":
            dense_inputs = {"dense0": inputs, "dense1": clean_hidden}
            uncancelled = True
        else:
            raise ValueError(f"unsupported qPolar activation source: {activation_source}")
        deltas = {name: self.codec.perturbation_matrix(value) for name, value in self._values.items()}
        impacts = {}
        for name, delta in deltas.items():
            if name.endswith("/bias"):
                impacts[name] = np.abs(delta) if uncancelled else delta
            else:
                activation = dense_inputs[name.split("/", 1)[0]]
                factor = np.mean(np.abs(activation), axis=0) if uncancelled else np.mean(activation, axis=0)
                impacts[name] = factor[:, None, None] * (np.abs(delta) if uncancelled else delta)
        return impacts

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


class NumpyActivationCancellationAdapter(NumpyReferenceAdapter):
    """Reference adapter with four-sample disjoint controlled partitions."""

    def sample_partition(self, role: str, count: int, *, offset: int = 0):
        self._require_loaded()
        available = {
            "ranking": tuple(range(0, 4)),
            "empirical_validation": tuple(range(4, 8)),
            "evaluation": tuple(range(8, 12)),
        }
        if role not in available:
            raise ValueError(f"unsupported M3 sample role: {role}")
        if offset != 0:
            raise ValueError("activation control has fixed partitions and supports offset=0 only")
        if not 1 <= count <= len(available[role]):
            raise ValueError(f"{role} count must be in [1,{len(available[role])}]")
        indices = available[role][:count]
        inputs = self._inputs[list(indices)].copy()
        labels = self._labels[list(indices)].copy()
        partition = SamplePartition(
            role, "packaged_activation_control", indices,
            sample_identity_digest("packaged_activation_control", indices, labels),
            "controlled activation-cancellation reference partition",
        )
        return inputs, labels, partition
