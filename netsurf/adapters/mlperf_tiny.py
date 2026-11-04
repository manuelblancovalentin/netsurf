"""Modern MLPerf Tiny TFLite adapters with no legacy runtime dependency."""

from __future__ import annotations

from contextlib import contextmanager
import csv
from dataclasses import dataclass
import hashlib
import pickle
from pathlib import Path
import tarfile
from typing import Iterator, Mapping

import numpy as np

from ..asset_lock import AssetLockError, VerifiedAssetLock, load_verified_asset_lock
from ..faults import FaultSchedule, InjectionReport
from ..identities import BitIdentity, ParameterSpec
from ..protection import ProtectionMask
from ..samples import SamplePartition, sample_identity_digest
from ..tflite_storage import TFLiteModelStorage


MLPERF_TINY_COMMIT = "4addd0fa08d216e20637637874e084895f289da4"
MLPERF_TINY_IC_INT8_SHA256 = (
    "3c002613d1b2475eb51dd78dfb85a546c8ae658dee71cf6ade43b022fe205415"
)
REFERENCE_INFERENCE_BACKEND = "tensorflow_lite_builtin_ref_v1"
OPTIMIZED_INFERENCE_BACKEND = "tensorflow_lite_builtin_without_default_delegates_v1"
DEFAULT_INFERENCE_BACKEND = "tensorflow_lite_auto_v1"
_MODEL_RELATIVE_PATH = (
    "benchmark/training/image_classification/trained_models/pretrainedResnet_quant.tflite"
)
_OFFICIAL_INDEX_RELATIVE_PATH = (
    "benchmark/training/image_classification/perf_samples_idxs.npy"
)
_OFFICIAL_LABEL_RELATIVE_PATH = "benchmark/evaluation/datasets/ic01/y_labels.csv"


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


def _decode_cifar_batch(stream) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    # CIFAR-10's official Python distribution is a pickle. Callers must verify
    # the enclosing dataset artifact before treating it as trusted input.
    document = pickle.load(stream, encoding="bytes")  # noqa: S301 - official format
    data = np.asarray(document[b"data"], dtype=np.uint8)
    labels = np.asarray(document.get(b"labels", document.get(b"fine_labels")), dtype=np.int64)
    raw_names = document.get(b"filenames")
    if data.ndim != 2 or data.shape[1] != 3 * 32 * 32 or labels.shape != (data.shape[0],):
        raise ValueError("invalid CIFAR-10 Python batch layout")
    images = np.transpose(data.reshape(-1, 3, 32, 32), (0, 2, 3, 1))
    if raw_names is None or len(raw_names) != data.shape[0]:
        raise ValueError("CIFAR-10 Python batch lacks stable filenames")
    names = tuple(
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in raw_names
    )
    return np.ascontiguousarray(images), labels, names


def _load_cifar10_python_with_names(
    path: Path, split: str
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Load the official CIFAR-10 Python archive or extracted directory."""

    source = Path(path)
    names = ["test_batch"] if split == "test" else [f"data_batch_{index}" for index in range(1, 6)]
    images, labels, filenames = [], [], []
    if source.is_file() and tarfile.is_tarfile(source):
        with tarfile.open(source, "r:*") as archive:
            members = {Path(member.name).name: member for member in archive.getmembers() if member.isfile()}
            for name in names:
                if name not in members:
                    raise ValueError(f"CIFAR-10 archive is missing {name}")
                stream = archive.extractfile(members[name])
                if stream is None:
                    raise ValueError(f"cannot read CIFAR-10 member {name}")
                x, y, names_in_batch = _decode_cifar_batch(stream)
                images.append(x)
                labels.append(y)
                filenames.extend(names_in_batch)
    elif source.is_dir():
        root = source / "cifar-10-batches-py" if (source / "cifar-10-batches-py").is_dir() else source
        for name in names:
            with (root / name).open("rb") as stream:
                x, y, names_in_batch = _decode_cifar_batch(stream)
                images.append(x)
                labels.append(y)
                filenames.extend(names_in_batch)
    else:
        raise FileNotFoundError(source)
    return np.concatenate(images), np.concatenate(labels), tuple(filenames)


def load_cifar10_python(path: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Load the official CIFAR-10 Python archive or extracted directory."""

    images, labels, _ = _load_cifar10_python_with_names(path, split)
    return images, labels


class MlperfTinyImageClassificationAdapter:
    """Official ResNet-8 INT8 TFLite inference and exact stored-bit faults.

    Gradients are deliberately unsupported: the official float HDF5 checkpoint
    has not yet been tensor-mapped and fault-equivalence-validated against this
    fused, per-channel-quantized TFLite graph.
    """

    benchmark_id = "mlperf-tiny/ic"

    def __init__(
        self,
        checkpoint: Path,
        dataset: Path,
        codec=None,
        *,
        asset_lock: Path | None = None,
        inference_backend: str = REFERENCE_INFERENCE_BACKEND,
        split: str = "test",
        injectable_parameters: tuple[str, ...] = ("kernels", "biases"),
        require_reference_equivalence: bool = True,
    ):
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.dataset_path = Path(dataset).expanduser().resolve()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)
        if not self.dataset_path.exists():
            raise FileNotFoundError(self.dataset_path)
        if inference_backend not in {
            REFERENCE_INFERENCE_BACKEND,
            OPTIMIZED_INFERENCE_BACKEND,
            DEFAULT_INFERENCE_BACKEND,
        }:
            raise ValueError(f"unsupported TFLite inference backend: {inference_backend}")
        if split not in {"test", "official_subset_200"}:
            raise ValueError("MLPerf Tiny split must be 'test' or 'official_subset_200'")
        if tuple(injectable_parameters) not in {("kernels",), ("kernels", "biases")}:
            raise ValueError(
                "injectable_parameters must be ('kernels',) or ('kernels', 'biases')"
            )
        self.asset_lock_path = (
            None if asset_lock is None else Path(asset_lock).expanduser().resolve()
        )
        self.asset_lock: VerifiedAssetLock | None = None
        self.inference_backend = inference_backend
        self.split = split
        self.injectable_parameters = tuple(injectable_parameters)
        self.require_reference_equivalence = bool(require_reference_equivalence)
        self.storage: TFLiteModelStorage | None = None
        self._model_content: bytes | None = None
        self._interpreter = None
        self._interpreter_backend: str | None = None
        self._runtime_provenance: dict[str, object] = {}
        self._dataset_cache: dict[
            str, tuple[np.ndarray, np.ndarray, tuple[str, ...]]
        ] = {}
        self._official_index_cache: np.ndarray | None = None
        self._differentiable_companion = None

    def load(self) -> None:
        if self.asset_lock_path is not None:
            self.asset_lock = load_verified_asset_lock(str(self.asset_lock_path))
            locked_model = self.asset_lock.file(_MODEL_RELATIVE_PATH)
            if locked_model != self.checkpoint:
                raise AssetLockError(
                    f"checkpoint {self.checkpoint} is not the model resolved by {self.asset_lock_path}"
                )
            dataset_root = Path(
                self.asset_lock.document["resolved"]["datasets"]["cifar10"]
            ).resolve()
            archive = self.asset_lock.file("cifar-10-python.tar.gz")
            if self.dataset_path not in {dataset_root, archive} and dataset_root not in self.dataset_path.parents:
                raise AssetLockError(
                    f"dataset {self.dataset_path} is not resolved by {self.asset_lock_path}"
                )
        self.storage = TFLiteModelStorage.from_path(
            self.checkpoint,
            expected_sha256=MLPERF_TINY_IC_INT8_SHA256,
            eligible_roles=tuple(
                {"kernels": "kernel", "biases": "bias"}[value]
                for value in self.injectable_parameters
            ),
        )
        self._set_model_content(self.storage.reference_content)
        if self.asset_lock is not None and self.require_reference_equivalence:
            conformance = self.reference_conformance()
            if not conformance["exact_output_digest_match"]:
                raise RuntimeError(
                    "configured TFLite runtime is not byte-equivalent to the registered "
                    "TensorFlow 2.14.1 BUILTIN_REF backend: official 200-output digest "
                    f"{conformance['output_sha256']} != "
                    f"{conformance['expected_output_sha256']}; use the recorded reference "
                    "environment or an explicit diagnostic adapter with equivalence disabled"
                )

    def _require_loaded(self) -> TFLiteModelStorage:
        if self.storage is None or self._model_content is None:
            raise RuntimeError("adapter.load() must be called first")
        return self.storage

    def _set_model_content(self, content: bytes) -> None:
        import tensorflow as tf

        resolver_by_backend = {
            REFERENCE_INFERENCE_BACKEND: tf.lite.experimental.OpResolverType.BUILTIN_REF,
            OPTIMIZED_INFERENCE_BACKEND: (
                tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
            ),
            DEFAULT_INFERENCE_BACKEND: tf.lite.experimental.OpResolverType.AUTO,
        }
        resolver = resolver_by_backend[self.inference_backend]
        interpreter = tf.lite.Interpreter(
            model_content=bytes(content),
            experimental_delegates=[],
            experimental_op_resolver_type=resolver,
            num_threads=1,
        )
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
        self._interpreter_backend = self.inference_backend
        if self.inference_backend == DEFAULT_INFERENCE_BACKEND:
            delegates = "default delegates permitted; XNNPACK observed on Kona"
        else:
            delegates = "none"
        self._runtime_provenance = {
            "library": "tensorflow",
            "tensorflow_version": tf.__version__,
            "interpreter": "tf.lite.Interpreter",
            "operator_resolver": f"tf.lite.experimental.OpResolverType.{resolver.name}",
            "delegates": delegates,
            "threads": 1,
            "recorded_reference_tensorflow_version": "2.14.1",
            "recorded_runtime_version_match": tf.__version__ == "2.14.1",
            "reference_equivalence_enforced": self.require_reference_equivalence,
        }

    @staticmethod
    def preprocess(images: np.ndarray) -> np.ndarray:
        values = np.asarray(images)
        if values.dtype != np.uint8 or values.ndim != 4 or tuple(values.shape[1:]) != (32, 32, 3):
            raise ValueError("CIFAR-10 inputs must be NHWC uint8 with shape (N,32,32,3)")
        return (values.astype(np.int16) - 128).astype(np.int8)

    def _split(self, split: str) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
        if split not in self._dataset_cache:
            images, labels, filenames = _load_cifar10_python_with_names(
                self.dataset_path, split
            )
            self._dataset_cache[split] = (
                self.preprocess(images),
                labels,
                filenames,
            )
        return self._dataset_cache[split]

    def _official_indices(self) -> np.ndarray:
        if self.asset_lock is None:
            raise AssetLockError(
                "official_subset_200 requires a verified Kona Assets lock"
            )
        if self._official_index_cache is None:
            index_path = self.asset_lock.file(_OFFICIAL_INDEX_RELATIVE_PATH)
            indices = np.asarray(np.load(index_path, allow_pickle=False), dtype=np.int64)
            if indices.shape != (200,) or len(np.unique(indices)) != 200:
                raise AssetLockError("official performance index must contain 200 unique rows")
            if int(indices.min()) < 0 or int(indices.max()) >= 10_000:
                raise AssetLockError("official performance index is outside CIFAR-10 test")
            _, labels, filenames = self._split("test")
            label_path = self.asset_lock.file(_OFFICIAL_LABEL_RELATIVE_PATH)
            with label_path.open("r", encoding="utf-8", newline="") as stream:
                rows = tuple(csv.reader(stream))
            if len(rows) != 200:
                raise AssetLockError("official label index must contain 200 rows")
            for row_index, (dataset_index, row) in enumerate(zip(indices, rows)):
                if len(row) != 3:
                    raise AssetLockError(f"official label row {row_index} is malformed")
                expected_name = str(Path(filenames[int(dataset_index)]).with_suffix(".bin"))
                if row[0] != expected_name or int(row[2]) != int(labels[int(dataset_index)]):
                    raise AssetLockError(
                        f"official sample identity differs at selected row {row_index}"
                    )
            self._official_index_cache = indices
        return self._official_index_cache.copy()

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
        official = role in {"evaluation", "final_evaluation"} and self.split == "official_subset_200"
        if official:
            stop = 200
        if count <= 0 or offset < 0 or start + offset + count > stop:
            raise ValueError(f"invalid {role} offset/count within [{start},{stop})")
        inputs, labels, _ = self._split(split)
        if official:
            indices = tuple(
                int(value)
                for value in self._official_indices()[offset : offset + count]
            )
            dataset_identity = "cifar10_test_mlperf_tiny_ic01_official_200"
            notes = (
                "Official MLPerf Tiny ic01 performance subset in perf_samples_idxs.npy order; "
                "filenames and labels independently matched to y_labels.csv"
            )
        else:
            indices = tuple(range(start + offset, start + offset + count))
            dataset_identity = f"cifar10_{split}"
            notes = (
                "MLPerf Tiny pinned-source CIFAR-10 partition; train roles are disjoint "
                "from held-out test evaluation"
            )
        selected_labels = labels[list(indices)].copy()
        return (
            inputs[list(indices)].copy(),
            selected_labels,
            SamplePartition(
                role,
                dataset_identity,
                indices,
                sample_identity_digest(dataset_identity, indices, selected_labels),
                notes,
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

    def predict_encoded_outputs(self, inputs: np.ndarray) -> np.ndarray:
        """Return the exact INT8 output codes produced by the selected runtime."""

        return self._outputs(inputs)

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

    def reference_conformance(self, *, full_test: bool = False) -> dict[str, object]:
        """Measure clean outputs against the immutable lock's registered digest.

        The full test evaluates 10,000 images; the default is the bounded
        official 200-image gate.  This method never changes model state.
        """

        if self.asset_lock is None:
            raise AssetLockError("reference conformance requires a verified asset lock")
        if full_test:
            inputs, labels, _ = self._split("test")
            expected_key = "full_outputs_sha256"
            reference_key = "full_test_10000"
        else:
            inputs, labels, _ = self._split("test")
            indices = self._official_indices()
            inputs, labels = inputs[indices], labels[indices]
            expected_key = "subset_outputs_sha256"
            reference_key = "official_subset_200"
        outputs = self.predict_encoded_outputs(inputs)
        digest = hashlib.sha256(outputs.tobytes(order="C")).hexdigest()
        prediction = np.argmax(outputs, axis=1)
        reference = self.asset_lock.benchmark["reference_results"]
        expected_digest = reference["output_digests"][expected_key]
        expected_accuracy = float(reference[reference_key]["value"])
        accuracy = float(np.mean(prediction == labels))
        return {
            "scope": reference_key,
            "sample_count": int(len(labels)),
            "output_sha256": digest,
            "expected_output_sha256": expected_digest,
            "exact_output_digest_match": digest == expected_digest,
            "accuracy": accuracy,
            "expected_accuracy": expected_accuracy,
            "accuracy_match": accuracy == expected_accuracy,
            "backend": dict(self._runtime_provenance),
        }

    def encoded_bit_transition(self, identity: BitIdentity):
        return self._require_loaded().bit_transition(self.values(), identity)

    def bit_perturbations(self) -> dict[str, np.ndarray]:
        """Exact decoded XOR jumps for the selected affine storage population."""

        from ..mlperf_companion import affine_bit_perturbations

        storage = self._require_loaded()
        return affine_bit_perturbations(self.values(), storage.descriptors)

    def _companion(self):
        if self.injectable_parameters != ("kernels",):
            raise ValueError(
                "the M9 differentiable companion is validated for INT8 kernels only; "
                "INT32 biases require an independent mapping gate"
            )
        if self._differentiable_companion is None:
            from ..mlperf_companion import MlperfResnet8DifferentiableCompanion

            self._differentiable_companion = MlperfResnet8DifferentiableCompanion(
                self._require_loaded().reference_content
            )
        return self._differentiable_companion

    def companion_correspondence(
        self, inputs: np.ndarray, labels: np.ndarray | None = None
    ) -> dict[str, object]:
        """Measure, never assume, clean straight-through forward correspondence."""

        from ..mlperf_companion import CompanionCorrespondence

        exact = self.predict_encoded_outputs(inputs)
        approximate = self._companion().encoded_probabilities(inputs)
        difference = np.abs(
            approximate.astype(np.int16) - exact.astype(np.int16)
        )
        exact_classes = np.argmax(exact, axis=1)
        approximate_classes = np.argmax(approximate, axis=1)
        exact_accuracy = approximate_accuracy = None
        if labels is not None:
            label_values = np.asarray(labels, dtype=np.int64)
            exact_accuracy = float(np.mean(exact_classes == label_values))
            approximate_accuracy = float(np.mean(approximate_classes == label_values))
        result = CompanionCorrespondence(
            sample_count=len(exact),
            exact_code_fraction=float(np.mean(difference == 0)),
            exact_output_vector_fraction=float(np.mean(np.all(difference == 0, axis=1))),
            class_agreement=float(np.mean(exact_classes == approximate_classes)),
            probability_mae=float(np.mean(difference) / 256.0),
            probability_max_error=float(np.max(difference) / 256.0),
            authoritative_accuracy=exact_accuracy,
            companion_accuracy=approximate_accuracy,
        )
        return {
            **result.to_dict(),
            "acceptance_criteria": {
                "minimum_class_agreement": 0.95,
                "maximum_probability_mae": 0.01,
                "maximum_accuracy_difference": 0.02,
            },
            "accepted": bool(
                result.class_agreement >= 0.95
                and result.probability_mae <= 0.01
                and (
                    exact_accuracy is None
                    or abs(exact_accuracy - approximate_accuracy) <= 0.02
                )
            ),
            "warning": (
                "acceptance authorizes a straight-through ranking surrogate, not "
                "replacement of authoritative TFLite inference"
            ),
        }

    def companion_kernel_correspondence(self, hdf5_checkpoint: Path):
        """Return the per-layer HDF5-to-TFLite kernel mapping diagnostic."""

        return self._companion().keras_kernel_correspondence(hdf5_checkpoint)

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
        result = {
            "identity": "mlperf_tiny_ic_resnet8_int8_v1",
            "suite_commit": MLPERF_TINY_COMMIT,
            "interpreter_backend": self._interpreter_backend,
            "runtime": dict(self._runtime_provenance),
            "fault_population_assumption": (
                "equal per-bit Bernoulli probability across selected storage classes is a "
                "logical software model, not evidence of equal physical upset rates"
            ),
            **self._require_loaded().manifest(),
        }
        if self._differentiable_companion is not None:
            result["differentiable_companion"] = self._differentiable_companion.manifest()
        if self.asset_lock is not None:
            result["asset_lock"] = self.asset_lock.manifest()
        return result

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
            "selection": self.split,
        }

    def task_metric_manifest(self) -> dict[str, object]:
        return {
            "task_type": "classification",
            "metrics": {
                "classification_accuracy": {"direction": "maximize", "units": "fraction"},
                "categorical_crossentropy": {
                    "direction": "minimize",
                    "units": "natural-log nats per sample",
                    "definition": (
                        "mean -ln(max((int8_output[label]+128)/256, 1e-12)); "
                        "no renormalization after integer softmax"
                    ),
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
        return self._companion().gradients(inputs, labels)

    def per_example_gradients(self, inputs, labels):
        return self._companion().per_example_gradients(inputs, labels)

    def hessian_vector_product(self, inputs, labels, vectors):
        raise NotImplementedError("M9 does not validate a TFLite companion Hessian")

    def polar_impacts(self, inputs, activation_source):
        raise NotImplementedError("M9 does not define qPolarGrad for ResNet-8")
