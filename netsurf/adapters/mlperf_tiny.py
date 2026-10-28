"""Modern MLPerf Tiny TFLite adapters with no legacy runtime dependency."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import pickle
from pathlib import Path
import tarfile
from typing import Iterator, Mapping

import numpy as np

from ..faults import FaultSchedule, InjectionReport
from ..identities import BitIdentity, ParameterSpec
from ..protection import ProtectionMask
from ..samples import SamplePartition, sample_identity_digest
from ..tflite_storage import TFLiteModelStorage


MLPERF_TINY_COMMIT = "4addd0fa08d216e20637637874e084895f289da4"
MLPERF_TINY_IC_INT8_SHA256 = (
    "3c002613d1b2475eb51dd78dfb85a546c8ae658dee71cf6ade43b022fe205415"
)


def bounded_synthetic_inference_signature(payload) -> dict[str, object]:
    """Spawn-safe runtime check; never a substitute for task evaluation."""

    checkpoint, sample_count = payload
    adapter = MlperfTinyImageClassificationAdapter(
        Path(checkpoint), Path(checkpoint).parent
    )
    adapter.load()
    inputs = np.zeros((int(sample_count), 32, 32, 3), dtype=np.int8)
    outputs = adapter.predict_outputs(inputs)
    return {
        "sample_count": int(sample_count),
        "output_sha256": hashlib.sha256(outputs.tobytes(order="C")).hexdigest(),
        "checkpoint_sha256": adapter.storage.sha256,
        "eligible_bit_count": len(adapter.bit_identities()),
        "interpretation": "fixed synthetic input runtime determinism only",
    }


@dataclass(frozen=True)
class ClassificationEvaluation:
    accuracy: float
    categorical_crossentropy: float
    sample_count: int

    @property
    def metrics(self) -> Mapping[str, float]:
        return {
            "classification_accuracy": self.accuracy,
            "categorical_crossentropy": self.categorical_crossentropy,
        }


def _decode_cifar_batch(stream) -> tuple[np.ndarray, np.ndarray]:
    # CIFAR-10's official Python distribution is a pickle. Callers must verify
    # the enclosing dataset artifact before treating it as trusted input.
    document = pickle.load(stream, encoding="bytes")  # noqa: S301 - official format
    data = np.asarray(document[b"data"], dtype=np.uint8)
    labels = np.asarray(document.get(b"labels", document.get(b"fine_labels")), dtype=np.int64)
    if data.ndim != 2 or data.shape[1] != 3 * 32 * 32 or labels.shape != (data.shape[0],):
        raise ValueError("invalid CIFAR-10 Python batch layout")
    images = np.transpose(data.reshape(-1, 3, 32, 32), (0, 2, 3, 1))
    return np.ascontiguousarray(images), labels


def load_cifar10_python(path: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Load the official CIFAR-10 Python archive or extracted directory."""

    source = Path(path)
    names = ["test_batch"] if split == "test" else [f"data_batch_{index}" for index in range(1, 6)]
    images, labels = [], []
    if source.is_file() and tarfile.is_tarfile(source):
        with tarfile.open(source, "r:*") as archive:
            members = {Path(member.name).name: member for member in archive.getmembers() if member.isfile()}
            for name in names:
                if name not in members:
                    raise ValueError(f"CIFAR-10 archive is missing {name}")
                stream = archive.extractfile(members[name])
                if stream is None:
                    raise ValueError(f"cannot read CIFAR-10 member {name}")
                x, y = _decode_cifar_batch(stream)
                images.append(x)
                labels.append(y)
    elif source.is_dir():
        root = source / "cifar-10-batches-py" if (source / "cifar-10-batches-py").is_dir() else source
        for name in names:
            with (root / name).open("rb") as stream:
                x, y = _decode_cifar_batch(stream)
                images.append(x)
                labels.append(y)
    else:
        raise FileNotFoundError(source)
    return np.concatenate(images), np.concatenate(labels)


class MlperfTinyImageClassificationAdapter:
    """Official ResNet-8 INT8 TFLite inference and exact stored-bit faults.

    Gradients are deliberately unsupported: the official float HDF5 checkpoint
    has not yet been tensor-mapped and fault-equivalence-validated against this
    fused, per-channel-quantized TFLite graph.
    """

    def __init__(self, checkpoint: Path, dataset: Path, codec=None):
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.dataset_path = Path(dataset).expanduser().resolve()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)
        if not self.dataset_path.exists():
            raise FileNotFoundError(self.dataset_path)
        self.storage: TFLiteModelStorage | None = None
        self._model_content: bytes | None = None
        self._interpreter = None
        self._interpreter_backend: str | None = None
        self._dataset_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def load(self) -> None:
        self.storage = TFLiteModelStorage.from_path(
            self.checkpoint, expected_sha256=MLPERF_TINY_IC_INT8_SHA256
        )
        self._set_model_content(self.storage.reference_content)

    def _require_loaded(self) -> TFLiteModelStorage:
        if self.storage is None or self._model_content is None:
            raise RuntimeError("adapter.load() must be called first")
        return self.storage

    def _set_model_content(self, content: bytes) -> None:
        try:
            from ai_edge_litert.interpreter import Interpreter

            backend = "ai_edge_litert"
        except ImportError:
            import tensorflow as tf

            Interpreter = tf.lite.Interpreter
            backend = "tensorflow.lite (deprecated upstream fallback)"

        interpreter = Interpreter(model_content=bytes(content), num_threads=1)
        interpreter.allocate_tensors()
        input_detail = interpreter.get_input_details()
        output_detail = interpreter.get_output_details()
        if len(input_detail) != 1 or len(output_detail) != 1:
            raise ValueError("MLPerf Tiny image classification requires one input and one output")
        if tuple(input_detail[0]["shape"]) != (1, 32, 32, 3) or input_detail[0]["dtype"] != np.int8:
            raise ValueError("unexpected MLPerf Tiny image-classification input contract")
        if tuple(output_detail[0]["shape"]) != (1, 10) or output_detail[0]["dtype"] != np.int8:
            raise ValueError("unexpected MLPerf Tiny image-classification output contract")
        self._model_content = bytes(content)
        self._interpreter = interpreter
        self._interpreter_backend = backend

    @staticmethod
    def preprocess(images: np.ndarray) -> np.ndarray:
        values = np.asarray(images)
        if values.dtype != np.uint8 or values.ndim != 4 or tuple(values.shape[1:]) != (32, 32, 3):
            raise ValueError("CIFAR-10 inputs must be NHWC uint8 with shape (N,32,32,3)")
        return (values.astype(np.int16) - 128).astype(np.int8)

    def _split(self, split: str) -> tuple[np.ndarray, np.ndarray]:
        if split not in self._dataset_cache:
            images, labels = load_cifar10_python(self.dataset_path, split)
            self._dataset_cache[split] = (self.preprocess(images), labels)
        return self._dataset_cache[split]

    def sample_partition(self, role: str, count: int, *, offset: int = 0):
        regions = {
            "ranking": ("train", 0, 25_000),
            "calibration": ("train", 0, 25_000),
            "empirical_validation": ("train", 25_000, 50_000),
            "evaluation": ("test", 0, 10_000),
            "final_evaluation": ("test", 0, 10_000),
        }
        if role not in regions:
            raise ValueError(f"unsupported MLPerf Tiny CIFAR-10 sample role: {role}")
        split, start, stop = regions[role]
        if count <= 0 or offset < 0 or start + offset + count > stop:
            raise ValueError(f"invalid {role} offset/count within [{start},{stop})")
        inputs, labels = self._split(split)
        indices = tuple(range(start + offset, start + offset + count))
        selected_labels = labels[list(indices)].copy()
        return (
            inputs[list(indices)].copy(),
            selected_labels,
            SamplePartition(
                role,
                f"cifar10_{split}",
                indices,
                sample_identity_digest(f"cifar10_{split}", indices, selected_labels),
                "MLPerf Tiny pinned-source CIFAR-10 partition; train roles are disjoint from held-out test evaluation",
            ),
        )

    def dataset(self, count: int, *, role: str = "evaluation"):
        inputs, labels, _ = self.sample_partition(role, count)
        return inputs, labels

    def values(self) -> dict[str, np.ndarray]:
        storage = self._require_loaded()
        return storage.values(self._model_content)

    def quantized_values(self) -> dict[str, np.ndarray]:
        return self.values()

    def parameter_specs(self) -> tuple[ParameterSpec, ...]:
        return self._require_loaded().parameter_specs()

    def bit_identities(self) -> tuple[BitIdentity, ...]:
        return self._require_loaded().bit_identities()

    def msb_scores(
        self, identities: tuple[BitIdentity, ...]
    ) -> tuple[dict[BitIdentity, float], dict[str, object]]:
        """Return word-relative tiers: sign bit 0, then -1, -2, ...

        Raw bit indices cannot be compared between INT8 kernels and INT32
        biases: bit 31 and bit 7 are both sign bits of their respective stored
        words.  The relative score preserves that equivalence explicitly.
        """

        widths = {
            descriptor.parameter: descriptor.bit_width
            for descriptor in self._require_loaded().descriptors
        }
        scores = {
            identity: float(identity.bit_index - (widths[identity.parameter] - 1))
            for identity in identities
        }
        return scores, {
            "implementation_version": "relative-word-bit-index/v1",
            "expression": "bit_index - (stored_word_width - 1); sign bits tie at zero",
            "word_widths": {"kernel": 8, "bias": 32},
        }

    @contextmanager
    def temporary_values(self, values: Mapping[str, np.ndarray]) -> Iterator[None]:
        storage = self._require_loaded()
        before = self._model_content
        self._set_model_content(storage.model_content_from_values(values))
        try:
            yield
        finally:
            self._set_model_content(before)
        if self._model_content != before:
            raise RuntimeError("TFLite reference model was not restored")

    def _outputs(self, inputs: np.ndarray) -> np.ndarray:
        self._require_loaded()
        values = np.asarray(inputs)
        if values.dtype != np.int8 or values.ndim != 4 or tuple(values.shape[1:]) != (32, 32, 3):
            raise ValueError("preprocessed TFLite inputs must be int8 NHWC (N,32,32,3)")
        input_detail = self._interpreter.get_input_details()[0]
        output_detail = self._interpreter.get_output_details()[0]
        outputs = np.empty((len(values), 10), dtype=np.int8)
        for index, sample in enumerate(values):
            self._interpreter.set_tensor(input_detail["index"], sample[None, ...])
            self._interpreter.invoke()
            outputs[index] = self._interpreter.get_tensor(output_detail["index"])[0]
        return outputs

    def predict_outputs(self, inputs: np.ndarray) -> np.ndarray:
        encoded = self._outputs(inputs)
        return (encoded.astype(np.float64) + 128.0) / 256.0

    def predict_classes(self, inputs: np.ndarray) -> np.ndarray:
        return np.argmax(self._outputs(inputs), axis=1)

    def evaluate(
        self, inputs: np.ndarray, labels: np.ndarray, *, batch_size: int = 256
    ) -> ClassificationEvaluation:
        del batch_size  # The reference graph has batch 1; loop order is the execution contract.
        probabilities = self.predict_outputs(inputs)
        labels = np.asarray(labels, dtype=np.int64)
        if labels.shape != (len(probabilities),):
            raise ValueError("label count does not match inputs")
        selected = np.maximum(probabilities[np.arange(len(labels)), labels], 1e-12)
        return ClassificationEvaluation(
            float(np.mean(np.argmax(probabilities, axis=1) == labels)),
            float(np.mean(-np.log(selected))),
            int(len(labels)),
        )

    def encoded_bit_transition(self, identity: BitIdentity):
        return self._require_loaded().bit_transition(self.values(), identity)

    def evaluate_fault_schedule(
        self,
        inputs: np.ndarray,
        labels: np.ndarray,
        schedule: FaultSchedule,
        protection: ProtectionMask,
    ) -> tuple[ClassificationEvaluation, InjectionReport]:
        storage = self._require_loaded()
        if schedule.model.persistence == "persistent_trial":
            content, report = storage.apply_schedule(schedule, protection)
            before = self._model_content
            self._set_model_content(content)
            try:
                return self.evaluate(inputs, labels), report
            finally:
                self._set_model_content(before)
        if schedule.model.persistence != "per_inference":
            raise ValueError(f"unsupported fault persistence: {schedule.model.persistence}")
        accuracies, losses = [], []
        proposed = applied = suppressed = 0
        unique: set[BitIdentity] = set()
        before = self._model_content
        try:
            for index in range(len(labels)):
                content, report = storage.apply_schedule(
                    schedule, protection, inference_index=index
                )
                self._set_model_content(content)
                result = self.evaluate(inputs[index : index + 1], labels[index : index + 1])
                accuracies.append(result.accuracy)
                losses.append(result.categorical_crossentropy)
                proposed += report.proposed_events
                applied += report.applied_events
                suppressed += report.suppressed_events
                unique.update(
                    event.bit
                    for event in schedule.events
                    if event.inference_index == index and event.bit not in protection.protected
                )
        finally:
            self._set_model_content(before)
        return ClassificationEvaluation(
            float(np.mean(accuracies)), float(np.mean(losses)), len(labels)
        ), InjectionReport(schedule.digest(), proposed, applied, suppressed, len(unique))

    def representation_manifest(self) -> dict[str, object]:
        return {
            "identity": "mlperf_tiny_ic_resnet8_int8_v1",
            "suite_commit": MLPERF_TINY_COMMIT,
            "interpreter_backend": self._interpreter_backend,
            **self._require_loaded().manifest(),
        }

    def quantization_manifest(self) -> dict[str, object]:
        return {
            "scheme": "TFLite affine integer constant buffers",
            "kernel_dtype": "int8",
            "bias_dtype": "int32",
            "per_tensor_and_per_channel": True,
            "encoded_bit_order": "lsb_zero",
            "signed_encoding": "two's-complement XOR",
            "metadata_faultable": False,
        }

    def preprocessing_manifest(self) -> dict[str, object]:
        return {
            "source": "CIFAR-10 Python uint8 RGB NHWC",
            "operation": "int8 = uint8 - 128",
            "input_shape": [1, 32, 32, 3],
            "input_scale": 1.0,
            "input_zero_point": -128,
        }

    def task_metric_manifest(self) -> dict[str, object]:
        return {
            "task_type": "classification",
            "metrics": {
                "classification_accuracy": {"direction": "maximize", "units": "fraction"},
                "categorical_crossentropy": {
                    "direction": "minimize",
                    "units": "natural-log nats per sample",
                },
            },
            "future_metric_families": [
                "reconstruction_mse",
                "per_example_reconstruction_error",
                "anomaly_roc_auc",
                "negative_log_likelihood",
                "perplexity",
            ],
        }

    def quantization_verification(self) -> dict[str, object]:
        storage = self._require_loaded()
        return {
            "reference_bytes_immutable": True,
            "private_copy_patch_round_trip": storage.model_content_from_values(self.values())
            == storage.reference_content,
            "eligible_values_are_integer_storage_codes": True,
        }

    def gradients(self, inputs, labels):
        raise NotImplementedError(
            "gradient ranking is blocked until the official differentiable checkpoint is mapped "
            "and fault-equivalence-validated against the INT8 TFLite graph"
        )

    per_example_gradients = gradients
    hessian_vector_product = gradients
    polar_impacts = gradients
