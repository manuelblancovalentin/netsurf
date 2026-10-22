"""Modern Keras-trained MNIST benchmark with explicit fixed-point inference."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Iterator, Mapping

import numpy as np

from ..identities import BitIdentity, ParameterSpec, enumerate_bits
from ..manifests import digest_object, sha256_file
from ..paths import dataset_root, workspace_root
from ..quantization import BitCodec, FixedPointFormat


@dataclass(frozen=True)
class Evaluation:
    accuracy: float
    categorical_crossentropy: float
    sample_count: int


PARAMETER_NAMES = ("dense0/kernel", "dense0/bias", "dense1/kernel", "dense1/bias")


def _configure_tensorflow_runtime() -> None:
    """Freeze the numerical backend before TensorFlow initializes.

    oneDNN graph rewrites produced cross-process last-bit drift in losses and
    gradients on the validated host.  The benchmark is therefore explicitly a
    CPU eager-execution contract with those rewrites disabled.
    """

    required = {
        "TF_ENABLE_ONEDNN_OPTS": "0",
        "TF_DETERMINISTIC_OPS": "1",
        "CUDA_VISIBLE_DEVICES": "-1",
    }
    initialized = "tensorflow" in sys.modules or "keras" in sys.modules
    conflicts = {
        name: os.environ.get(name)
        for name, expected in required.items()
        if initialized and os.environ.get(name) != expected
    }
    if conflicts:
        raise RuntimeError(
            "mnist_quantized requires TensorFlow to be imported after "
            f"{required}; restart the process with that environment (found {conflicts})"
        )
    os.environ.update(required)


_configure_tensorflow_runtime()


def _keras_model():
    import keras

    inputs = keras.Input(shape=(28, 28, 1), name="image")
    value = keras.layers.Flatten(name="flatten")(inputs)
    value = keras.layers.Dense(32, activation="relu", name="dense0")(value)
    logits = keras.layers.Dense(10, name="dense1")(value)
    return keras.Model(inputs, logits, name="mnist_dense32_q621_source")


def _normalize(images: np.ndarray) -> np.ndarray:
    return (np.asarray(images, dtype=np.float32) / np.float32(255.0))[..., None]


def prepare_checkpoint(*, epochs: int = 3, train_samples: int = 20_000, seed: int = 20261022) -> Path:
    """Train once and atomically save exact quantized parameters under workspace/."""

    if epochs < 1 or not 1 <= train_samples <= 50_000:
        raise ValueError("epochs must be positive and train_samples must be in [1,50000]")
    import keras
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    source = dataset_root() / "mnist" / "mnist.npz"
    if not source.is_file():
        raise FileNotFoundError(source)
    with np.load(source) as data:
        train_x = _normalize(data["x_train"][:train_samples])
        train_y = data["y_train"][:train_samples].astype(np.int64)
        validation_x = _normalize(data["x_train"][50_000:60_000])
        validation_y = data["y_train"][50_000:60_000].astype(np.int64)
        test_x = _normalize(data["x_test"])
        test_y = data["y_test"].astype(np.int64)
    model = _keras_model()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    history = model.fit(
        train_x, train_y, validation_data=(validation_x, validation_y),
        epochs=epochs, batch_size=128, shuffle=False, verbose=2,
    )
    float_test = model.evaluate(test_x, test_y, batch_size=256, verbose=0, return_dict=True)
    codec = BitCodec(FixedPointFormat.parse("q<6,2,1>"))
    layers = (model.get_layer("dense0"), model.get_layer("dense1"))
    values = {
        "dense0/kernel": codec.quantize(layers[0].kernel.numpy()).astype(np.float32),
        "dense0/bias": codec.quantize(layers[0].bias.numpy()).astype(np.float32),
        "dense1/kernel": codec.quantize(layers[1].kernel.numpy()).astype(np.float32),
        "dense1/bias": codec.quantize(layers[1].bias.numpy()).astype(np.float32),
    }
    parameter_identity = [
        {"name": name, "shape": list(value.shape), "encoded_sha256": digest_object(codec.encode(value).tolist())}
        for name, value in values.items()
    ]
    metadata = {
        "schema_version": "1.0.0", "benchmark": "mnist_quantized",
        "architecture": "Flatten(28x28)-Dense(32)-ReLU-Dense(10 logits)",
        "quantization": codec.format.to_dict(), "quantization_backend": "netsurf_bitcodec_v1",
        "injectable_parameters": list(PARAMETER_NAMES), "training_seed": seed,
        "epochs": epochs, "train_samples": train_samples, "shuffle": False, "batch_size": 128,
        "training_split": "x_train[0:train_samples]", "ranking_split": "x_train[50000:60000]",
        "evaluation_split": "x_test", "preprocessing": "float32 pixels divided by 255; channel appended",
        "dataset_sha256": sha256_file(source), "float_test_metrics": {name: float(value) for name, value in float_test.items()},
        "training_history": {name: [float(item) for item in values_] for name, values_ in history.history.items()},
        "parameter_identity": parameter_identity,
    }
    output = workspace_root() / "benchmarks" / "mnist_quantized" / "checkpoint.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"checkpoint already exists and will not be overwritten: {output}")
    with tempfile.NamedTemporaryFile(prefix="checkpoint-", suffix=".npz", dir=output.parent, delete=False) as stream:
        temporary = Path(stream.name)
        np.savez_compressed(stream, **values, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(output)
    output.chmod(0o444)
    return output


class MnistQuantizedAdapter:
    """Fault adapter operating directly on decoded fixed-point checkpoint words."""

    def __init__(self, checkpoint: Path, dataset: Path, codec: BitCodec):
        self.checkpoint, self.dataset_path, self.codec = Path(checkpoint), Path(dataset), codec
        self._values: dict[str, np.ndarray] = {}
        self.metadata: dict[str, object] = {}

    def load(self) -> None:
        with np.load(self.checkpoint, allow_pickle=False) as checkpoint:
            self._values = {name: checkpoint[name].astype(np.float32).copy() for name in PARAMETER_NAMES}
            self.metadata = json.loads(str(checkpoint["metadata_json"]))
        if self.metadata["quantization"]["notation"] != str(self.codec.format):
            raise ValueError("checkpoint quantization does not match experiment")
        for name, value in self._values.items():
            if not np.array_equal(self.codec.decode(self.codec.encode(value)).astype(np.float32), value):
                raise ValueError(f"checkpoint parameter is not exactly representable: {name}")

    def _require_loaded(self) -> None:
        if not self._values:
            raise RuntimeError("adapter.load() must be called first")

    def dataset(self, count: int, *, role: str = "evaluation") -> tuple[np.ndarray, np.ndarray]:
        self._require_loaded()
        with np.load(self.dataset_path) as data:
            if role == "ranking":
                images, labels = data["x_train"][50_000:50_000 + count], data["y_train"][50_000:50_000 + count]
            elif role == "evaluation":
                images, labels = data["x_test"][:count], data["y_test"][:count]
            else:
                raise ValueError(f"unsupported dataset role: {role}")
        if len(labels) != count:
            raise ValueError(f"requested {count} {role} samples, found {len(labels)}")
        return _normalize(images), labels.astype(np.int64)

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
        if set(values) != set(before):
            raise ValueError("parameter identity mismatch")
        self._values = {name: np.asarray(values[name], dtype=np.float32).copy() for name in before}
        try:
            yield
        finally:
            self._values = before

    def quantized_values(self) -> dict[str, np.ndarray]:
        return {name: self.codec.quantize(value).astype(np.float32) for name, value in self.values().items()}

    @staticmethod
    def _forward_tensor(inputs, values):
        import tensorflow as tf

        flat = tf.reshape(inputs, (tf.shape(inputs)[0], 784))
        hidden = tf.nn.relu(tf.linalg.matmul(flat, values["dense0/kernel"]) + values["dense0/bias"])
        return tf.linalg.matmul(hidden, values["dense1/kernel"]) + values["dense1/bias"]

    def evaluate(self, inputs: np.ndarray, labels: np.ndarray, *, batch_size: int = 256) -> Evaluation:
        import tensorflow as tf

        values = {name: tf.convert_to_tensor(value) for name, value in self._values.items()}
        logits = self._forward_tensor(tf.convert_to_tensor(inputs), values)
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1, output_type=tf.int64), labels), tf.float32))
        return Evaluation(float(accuracy.numpy()), float(tf.reduce_mean(losses).numpy()), len(labels))

    def gradients(self, inputs: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
        import tensorflow as tf

        variables = {name: tf.Variable(value, trainable=True, name=name.replace("/", "_")) for name, value in self._values.items()}
        with tf.GradientTape() as tape:
            logits = self._forward_tensor(tf.convert_to_tensor(inputs), variables)
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        gradients = tape.gradient(loss, list(variables.values()))
        return {name: gradient.numpy().copy() for name, gradient in zip(variables, gradients)}

    def polar_impacts(self, inputs: np.ndarray, *, activation_source: str):
        raise ValueError("legacy_qpolargrad is not supported by mnist_quantized")
