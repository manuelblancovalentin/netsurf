"""Read-only adapter for the recovered 2025 MNIST-lite checkpoint.

Importing this module does not import the historical package. Loading the
model does, and callers should isolate ``HOME`` because that
historical import initializes Nodus and user configuration as a side effect.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
from typing import Iterator, Mapping

import numpy as np

from ..identities import ParameterSpec, enumerate_bits
from ..quantization import BitCodec
from ..paths import legacy_source_root, legacy_workspace_root, qkeras_package_root
from ..samples import SamplePartition, sample_identity_digest


_MNIST_LITE_EXACT_CHECKPOINT_SHA256 = (
    "ba48aee8adbe7d04092d74f41958ec99112ae3e5f88ed56552ea9d1c4dba3006"
)
_MNIST_LITE_EXACT_DATASET_SHA256 = (
    "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    previous_home = os.environ.get("HOME")
    sys.modules[module_name] = module
    sys.modules["netsurf"] = module
    try:
        # Nodus and the historical package inspect HOME during import.  Keep
        # their unavoidable compatibility state inside the gitignored v2
        # workspace rather than the researcher's personal configuration.
        os.environ["HOME"] = str(side_effect_root)
        os.chdir(side_effect_root)
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    finally:
        os.chdir(previous_working_directory)
        if previous_home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = previous_home
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

    def _mnist_split(self, source_split: str) -> tuple[np.ndarray, np.ndarray]:
        with np.load(self.dataset_path) as data:
            raw = data[f"x_{source_split}"].astype(np.float32)
            labels = data[f"y_{source_split}"].astype(np.int64)
        low, high = self.codec.format.value_min, self.codec.format.value_max
        normalized = low + raw * ((high - low) / 255.0)
        return normalized[..., None].astype(np.float32), labels

    def sample_partition(self, role: str, count: int, *, offset: int = 0):
        """Return explicit nonoverlapping legacy-MNIST protocol partitions.

        The checkpoint's original training split is not recoverable from the
        model file.  These are v2 analysis partitions, not claims about the
        historical training protocol.
        """

        regions = {
            "ranking": ("train", 50_000, 60_000),
            "calibration": ("train", 50_000, 60_000),
            "empirical_validation": ("train", 40_000, 50_000),
            "evaluation": ("test", 0, 10_000),
            "final_evaluation": ("test", 0, 10_000),
        }
        if role not in regions:
            raise ValueError(f"unsupported legacy MNIST sample role: {role}")
        source_split, start, stop = regions[role]
        if offset < 0 or count <= 0 or start + offset + count > stop:
            raise ValueError(
                f"{role} requires offset >= 0, count > 0, and offset + count <= {stop - start}"
            )
        all_inputs, all_labels = self._mnist_split(source_split)
        indices = tuple(range(start + offset, start + offset + count))
        inputs = all_inputs[list(indices)].copy()
        labels = all_labels[list(indices)].copy()
        partition = SamplePartition(
            role=role,
            source_split=f"mnist_{source_split}",
            indices=indices,
            sample_sha256=sample_identity_digest(f"mnist_{source_split}", indices, labels),
            provenance=(
                "NetSuRF v2 identity-gate partition; the historical checkpoint's "
                "original training/calibration membership is unknown"
            ),
        )
        return inputs, labels, partition

    def dataset(
        self, count: int, *, role: str = "evaluation"
    ) -> tuple[np.ndarray, np.ndarray]:
        inputs, labels, _ = self.sample_partition(role, count)
        return inputs, labels

    def values(self) -> dict[str, np.ndarray]:
        self._require_loaded()
        return {variable.name: variable.numpy().copy() for variable in self.model.trainable_variables}

    def parameter_specs(self) -> tuple[ParameterSpec, ...]:
        return tuple(ParameterSpec(name, tuple(value.shape)) for name, value in self.values().items())

    def bit_identities(self):
        return enumerate_bits(self.parameter_specs(), self.codec.format.total_bits)

    @staticmethod
    def tensor_role(name: str) -> str:
        leaf = name.removesuffix(":0").rsplit("/", 1)[-1]
        return leaf if leaf in {"kernel", "bias", "alpha", "beta"} else "other"

    def exact_kernel_bias_values(self) -> dict[str, np.ndarray]:
        """Return the explicit hybrid conversion used by the identity gate.

        Kernel and bias tensors are q<6,2,1> values. Learned alpha/beta remain
        float shadows and are therefore not members of the exact encoded fault
        population. This is a conversion candidate, not the historical clean
        state and not a claim about deployment storage.
        """

        return {
            name: (
                self.codec.quantize(value).astype(value.dtype)
                if self.tensor_role(name) in {"kernel", "bias"}
                else value.copy()
            )
            for name, value in self.values().items()
        }

    def exact_kernel_bias_bit_identities(self):
        """Enumerate all allocated kernel/bias bits, including zero words."""

        parameters = tuple(
            ParameterSpec(name, tuple(value.shape))
            for name, value in self.values().items()
            if self.tensor_role(name) in {"kernel", "bias"}
        )
        return enumerate_bits(parameters, self.codec.format.total_bits)

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

    def predict_outputs(self, inputs: np.ndarray) -> np.ndarray:
        self._require_loaded()
        return self.model(inputs, training=False).numpy().copy()

    def predict_logits(self, inputs: np.ndarray) -> np.ndarray:
        """Return the recovered output-Dense values before final softmax."""

        self._require_loaded()
        probe = self._tf.keras.Model(
            inputs=self.model.inputs, outputs=self.model.get_layer("fc_out").output
        )
        return probe(inputs, training=False).numpy().copy()

    def predict_classes(self, inputs: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_outputs(inputs), axis=1)

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


def _state_sha256(values: Mapping[str, np.ndarray]) -> str:
    """Hash named tensor identity, dtype, shape, and C-order contents."""

    digest = hashlib.sha256()
    for name in sorted(values):
        value = np.ascontiguousarray(values[name])
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(value.dtype.str.encode("ascii") + b"\0")
        digest.update(repr(tuple(value.shape)).encode("ascii") + b"\0")
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


class MnistLiteExactKernelBiasAdapter(LegacyMnistLiteAdapter):
    """Recovered MNIST-lite with an explicit exact kernel/bias baseline.

    All allocated kernels and biases, including zero-valued words, are
    converted to q<6,2,1>. Learned alpha/beta remain uncorrupted float32
    auxiliaries. The adapter is a new v2 benchmark identity; it does not
    reproduce the historical float-shadow fault semantics.
    """

    def __init__(self, checkpoint: Path, dataset: Path, codec: BitCodec):
        super().__init__(checkpoint, dataset, codec)
        self._converted_state_sha256: str | None = None
        self._per_example_gradient_function = None

    def load(self) -> None:
        observed_checkpoint = _file_sha256(self.checkpoint)
        observed_dataset = _file_sha256(self.dataset_path)
        if observed_checkpoint != _MNIST_LITE_EXACT_CHECKPOINT_SHA256:
            raise ValueError(
                "mnist_lite_exact_kb_v1 checkpoint identity mismatch: "
                f"{observed_checkpoint}"
            )
        if observed_dataset != _MNIST_LITE_EXACT_DATASET_SHA256:
            raise ValueError(
                "mnist_lite_exact_kb_v1 dataset identity mismatch: "
                f"{observed_dataset}"
            )
        super().load()
        converted = super().exact_kernel_bias_values()
        self.set_values(converted)
        self._converted_state_sha256 = _state_sha256(converted)

    def parameter_specs(self) -> tuple[ParameterSpec, ...]:
        return tuple(
            ParameterSpec(name, tuple(value.shape))
            for name, value in self.values().items()
            if self.tensor_role(name) in {"kernel", "bias"}
        )

    def bit_identities(self):
        return enumerate_bits(self.parameter_specs(), self.codec.format.total_bits)

    def quantized_values(self) -> dict[str, np.ndarray]:
        # The live kernel/bias state is exact. Float auxiliaries are intentionally
        # retained rather than silently coerced through the common codec.
        return self.values()

    def representation_manifest(self) -> dict[str, object]:
        self._require_loaded()
        if self._converted_state_sha256 is None:
            raise RuntimeError("converted state digest is unavailable")
        values = self.values()
        eligible = [name for name in values if self.tensor_role(name) in {"kernel", "bias"}]
        auxiliaries = [name for name in values if self.tensor_role(name) in {"alpha", "beta"}]
        return {
            "identity": "mnist_lite_exact_kb_v1",
            "representation": "q<6,2,1> kernels/biases with float32 alpha/beta auxiliaries",
            "converted_state_sha256": self._converted_state_sha256,
            "eligible_tensor_names": eligible,
            "eligible_scalars": sum(values[name].size for name in eligible),
            "eligible_bits": len(self.bit_identities()),
            "uncorrupted_float_auxiliary_names": auxiliaries,
            "uncorrupted_float_auxiliary_scalars": sum(values[name].size for name in auxiliaries),
            "zero_valued_allocated_words_included": True,
        }

    def per_example_gradients(
        self, inputs: np.ndarray, labels: np.ndarray
    ) -> dict[str, np.ndarray]:
        self._require_loaded()
        tf = self._tf
        variables = tuple(self.model.trainable_variables)
        if self._per_example_gradient_function is None:
            input_shape = tuple(self.model.input_shape[1:])

            @tf.function(
                input_signature=(
                    tf.TensorSpec((None, *input_shape), tf.float32, name="inputs"),
                    tf.TensorSpec((None,), tf.int64, name="labels"),
                ),
                reduce_retracing=True,
            )
            def compiled(input_tensor, label_tensor):
                with tf.GradientTape() as tape:
                    predictions = self.model(input_tensor, training=False)
                    losses = tf.keras.losses.sparse_categorical_crossentropy(
                        label_tensor, predictions
                    )
                return tuple(
                    tape.jacobian(losses, variables, experimental_use_pfor=True)
                )

            self._per_example_gradient_function = compiled
        jacobians = self._per_example_gradient_function(
            tf.convert_to_tensor(np.asarray(inputs, dtype=np.float32)),
            tf.convert_to_tensor(np.asarray(labels, dtype=np.int64)),
        )
        return {
            variable.name: gradient.numpy().copy()
            for variable, gradient in zip(variables, jacobians)
        }

    def hessian_vector_product(
        self,
        inputs: np.ndarray,
        labels: np.ndarray,
        vectors: Mapping[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        self._require_loaded()
        tf = self._tf
        variables = tuple(self.model.trainable_variables)
        targets = tf.one_hot(labels, depth=10)
        with tf.GradientTape() as outer:
            with tf.GradientTape() as inner:
                predictions = self.model(inputs, training=False)
                loss = tf.reduce_mean(
                    tf.keras.losses.categorical_crossentropy(targets, predictions)
                )
                if self.model.losses:
                    loss = loss + tf.add_n(self.model.losses)
            gradients = inner.gradient(loss, variables)
            product = tf.add_n(
                [
                    tf.reduce_sum(
                        gradient
                        * tf.convert_to_tensor(vectors[variable.name], dtype=gradient.dtype)
                    )
                    for variable, gradient in zip(variables, gradients)
                ]
            )
        products = outer.gradient(product, variables)
        return {
            variable.name: value.numpy().copy()
            for variable, value in zip(variables, products)
        }
