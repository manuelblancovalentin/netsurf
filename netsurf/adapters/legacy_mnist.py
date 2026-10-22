"""Read-only adapter for the recovered 2025 MNIST-lite checkpoint.

Importing this module does not import the historical package. Loading the
model does, and callers should isolate ``HOME`` because that
historical import initializes Nodus and user configuration as a side effect.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
import sys
from typing import Iterator, Mapping

import numpy as np

from ..identities import ParameterSpec, enumerate_bits
from ..quantization import BitCodec
from ..paths import legacy_source_root, legacy_workspace_root, qkeras_package_root


def _load_legacy_package():
    """Load the preserved package without replacing canonical ``netsurf``.

    Historical modules contain absolute ``import netsurf`` statements.  They
    are resolved against the legacy module only while its eager import runs;
    the canonical package is restored immediately afterward.
    """

    package_root = legacy_source_root().resolve()
    initializer = package_root / "__init__.py"
    if not initializer.is_file():
        raise FileNotFoundError(initializer)
    module_name = "_netsurf_legacy"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(
        module_name,
        initializer,
        submodule_search_locations=[str(package_root)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load historical NetSuRF from {initializer}")
    module = importlib.util.module_from_spec(spec)
    canonical = sys.modules.get("netsurf")
    qkeras_parent = str(qkeras_package_root().resolve().parent)
    added_qkeras_parent = qkeras_parent not in sys.path
    if added_qkeras_parent:
        sys.path.insert(0, qkeras_parent)
    side_effect_root = legacy_workspace_root().resolve()
    side_effect_root.mkdir(parents=True, exist_ok=True)
    previous_working_directory = Path.cwd()
    sys.modules[module_name] = module
    sys.modules["netsurf"] = module
    try:
        os.chdir(side_effect_root)
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    finally:
        os.chdir(previous_working_directory)
        if added_qkeras_parent:
            sys.path.remove(qkeras_parent)
        if canonical is None:
            sys.modules.pop("netsurf", None)
        else:
            sys.modules["netsurf"] = canonical
    return module


@dataclass(frozen=True)
class Evaluation:
    accuracy: float
    categorical_crossentropy: float
    sample_count: int


class LegacyMnistLiteAdapter:
    """TensorFlow adapter with explicit state snapshot/restore semantics."""

    def __init__(self, checkpoint: Path, dataset: Path, codec: BitCodec):
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.dataset_path = Path(dataset).expanduser().resolve()
        self.codec = codec
        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)
        if not self.dataset_path.is_file():
            raise FileNotFoundError(self.dataset_path)
        self._tf = None
        self.model = None

    def load(self) -> None:
        # Deliberately local: importing the historical package has filesystem
        # and DB side effects and must never occur merely by importing netsurf.
        import tensorflow as tf

        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(17_041_425)
        legacy = _load_legacy_package()
        quantizer = legacy.QuantizationScheme(str(self.codec.format))
        custom_objects = legacy.dnn.models.get_custom_objects(quantizer)
        self.model = tf.keras.models.load_model(
            str(self.checkpoint), custom_objects=custom_objects, compile=False
        )
        self._tf = tf

    def _require_loaded(self) -> None:
        if self.model is None or self._tf is None:
            raise RuntimeError("adapter.load() must be called first")

    def dataset(self, count: int) -> tuple[np.ndarray, np.ndarray]:
        if count <= 0 or count > 10_000:
            raise ValueError("count must be in [1, 10000]")
        with np.load(self.dataset_path) as data:
            raw = data["x_test"][:count].astype(np.float32)
            labels = data["y_test"][:count].astype(np.int64)
        low, high = self.codec.format.value_min, self.codec.format.value_max
        normalized = low + raw * ((high - low) / 255.0)
        return normalized[..., None].astype(np.float32), labels

    def values(self) -> dict[str, np.ndarray]:
        self._require_loaded()
        return {variable.name: variable.numpy().copy() for variable in self.model.trainable_variables}

    def parameter_specs(self) -> tuple[ParameterSpec, ...]:
        return tuple(ParameterSpec(name, tuple(value.shape)) for name, value in self.values().items())

    def bit_identities(self):
        return enumerate_bits(self.parameter_specs(), self.codec.format.total_bits)

    def set_values(self, values: Mapping[str, np.ndarray]) -> None:
        self._require_loaded()
        variables = {variable.name: variable for variable in self.model.trainable_variables}
        if set(values) != set(variables):
            missing = set(variables) - set(values)
            extra = set(values) - set(variables)
            raise ValueError(f"state mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
        for name, variable in variables.items():
            value = np.asarray(values[name])
            if tuple(value.shape) != tuple(variable.shape):
                raise ValueError(f"shape mismatch for {name}: {value.shape} != {tuple(variable.shape)}")
            variable.assign(value)

    @contextmanager
    def temporary_values(self, values: Mapping[str, np.ndarray]) -> Iterator[None]:
        before = self.values()
        try:
            self.set_values(values)
            yield
        finally:
            self.set_values(before)
        after = self.values()
        for name in before:
            if not np.array_equal(before[name], after[name]):
                raise RuntimeError(f"model state restoration failed for {name}")

    def quantized_values(self) -> dict[str, np.ndarray]:
        return {
            name: self.codec.quantize(value).astype(value.dtype)
            for name, value in self.values().items()
        }

    def evaluate(self, inputs: np.ndarray, labels: np.ndarray, *, batch_size: int = 256) -> Evaluation:
        self._require_loaded()
        predictions = self.model(inputs, training=False).numpy()
        selected = np.maximum(predictions[np.arange(labels.size), labels], 1e-12)
        return Evaluation(
            accuracy=float(np.mean(np.argmax(predictions, axis=1) == labels)),
            categorical_crossentropy=float(np.mean(-np.log(selected))),
            sample_count=int(labels.size),
        )

    def gradients(self, inputs: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
        """Gradient of held-out mean cross entropy plus registered model losses."""

        self._require_loaded()
        tf = self._tf
        targets = tf.one_hot(labels, depth=10)
        with tf.GradientTape() as tape:
            predictions = self.model(inputs, training=False)
            loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(targets, predictions))
            if self.model.losses:
                loss = loss + tf.add_n(self.model.losses)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        result: dict[str, np.ndarray] = {}
        for variable, gradient in zip(self.model.trainable_variables, gradients):
            if gradient is None:
                raise RuntimeError(f"no gradient for {variable.name}")
            result[variable.name] = gradient.numpy().copy()
        return result

    def layer_inputs(self, inputs: np.ndarray) -> dict[str, np.ndarray]:
        """Return the input activation used by each trainable parameter."""

        self._require_loaded()
        tf = self._tf
        layer_names = ("fc0", "alpha0", "fc_out")
        probe = tf.keras.Model(
            inputs=self.model.inputs,
            outputs=[self.model.get_layer(name).input for name in layer_names],
        )
        fc0_input, alpha_input, fc_out_input = [value.numpy() for value in probe(inputs, training=False)]
        return {
            "fc0/kernel:0": fc0_input,
            "fc0/bias:0": fc0_input,
            "alpha0/alpha:0": alpha_input,
            "alpha0/beta:0": alpha_input,
            "fc_out/kernel:0": fc_out_input,
            "fc_out/bias:0": fc_out_input,
        }

    def globally_attacked_layer_inputs(self, inputs: np.ndarray) -> dict[str, np.ndarray]:
        """First-call equivalent of the legacy all-bit qPolar activation attack.

        All encoded bits are complemented simultaneously, then model state is
        restored. This reproduces the globally-corrupted activation source but
        not the unintended cross-call accumulation in legacy ``*_N`` tensors.
        """

        attacked: dict[str, np.ndarray] = {}
        mask = np.uint64((1 << self.codec.format.total_bits) - 1)
        for name, value in self.values().items():
            encoded = self.codec.encode(value)
            decoded = self.codec.decode(np.bitwise_xor(encoded, mask)).astype(value.dtype)
            attacked[name] = decoded
        with self.temporary_values(attacked):
            return self.layer_inputs(inputs)

    def polar_impacts(
        self,
        inputs: np.ndarray,
        *,
        activation_source: str = "globally_attacked_first_call",
    ) -> dict[str, np.ndarray]:
        values = self.values()
        deltas = {name: self.codec.perturbation_matrix(value) for name, value in values.items()}
        if activation_source == "globally_attacked_first_call":
            activations = self.globally_attacked_layer_inputs(inputs)
        elif activation_source == "clean":
            activations = self.layer_inputs(inputs)
        else:
            raise ValueError(activation_source)

        impacts: dict[str, np.ndarray] = {}
        for name, delta in deltas.items():
            if name.endswith("kernel:0"):
                mean_input = np.mean(activations[name], axis=0).reshape(-1, 1)
                # Dense kernels use input-feature x output-channel layout.
                impacts[name] = mean_input[..., None] * delta
            elif name.endswith("alpha:0"):
                mean_input = np.mean(activations[name], axis=0)
                impacts[name] = mean_input[..., None] * delta
            else:
                impacts[name] = delta
        return impacts
