"""Exact-bit adapters for pinned MLPerf Tiny KWS and ToyCar artifacts."""

from __future__ import annotations

from contextlib import contextmanager
import csv
from dataclasses import dataclass
import hashlib
import multiprocessing
import os
from pathlib import Path
import platform
import sys
from typing import Iterator, Mapping

import numpy as np

from ..asset_lock import (
    AssetLockError,
    VerifiedAssetLock,
    load_benchmark_asset_lock,
    sha256_path,
)
from ..faults import FaultSchedule, InjectionReport
from ..identities import BitIdentity, ParameterSpec
from ..mlperf_companion import affine_bit_perturbations
from ..protection import ProtectionMask
from ..samples import SamplePartition, sample_identity_digest
from ..tflite_storage import TFLiteModelStorage
from ..tfrecord import interleaved_examples
from .mlperf_tiny import (
    ClassificationEvaluation,
    REFERENCE_INFERENCE_BACKEND,
)


MLPERF_TINY_COMMIT = "4addd0fa08d216e20637637874e084895f289da4"
KWS_MODEL_SHA256 = "aeea436800704fce17b17292e4412630ad856e9d777c044c64ef748a880bd0ae"
AD_MODEL_SHA256 = "87cf24194ef93d1d9b11a591d805526b98008e351655d29883c825c9c106ba24"
KWS_MODEL_RELATIVE = "benchmark/training/keyword_spotting/trained_models/kws_ref_model.tflite"
KWS_INDEX_RELATIVE = "benchmark/evaluation/datasets/kws01/y_labels.csv"
AD_MODEL_RELATIVE = "benchmark/training/anomaly_detection/trained_models/ad01_int8.tflite"
AD_INDEX_RELATIVE = "benchmark/evaluation/datasets/ad01/y_labels.csv"
TOYCAR_PREPROCESSING_BLAS_THREADS = 2
TOYCAR_BOUNDED_EXPECTED_SCORES = (
    10.168487922577622,
    11.777840215450889,
)
TOYCAR_BOUNDED_EXPECTED_SCORE_SHA256 = (
    "cd135be4abd970a9856891741a8fb09b7f67af6bd0552a9c23be28c848ef9fb1"
)


@dataclass(frozen=True)
class AnomalyEvaluation:
    reconstruction_mse: float
    roc_auc: float
    sample_count: int
    window_count: int
    file_scores: tuple[float, ...]

    @property
    def metrics(self) -> Mapping[str, float]:
        return {
            "reconstruction_mse": self.reconstruction_mse,
            "roc_auc": self.roc_auc,
        }

    @property
    def primary_metric_name(self) -> str:
        return "roc_auc"

    @property
    def primary_metric_value(self) -> float:
        return self.roc_auc

    @property
    def loss_metric_name(self) -> str:
        return "reconstruction_mse"

    @property
    def loss_metric_value(self) -> float:
        return self.reconstruction_mse


class _ExactTFLiteKernelAdapter:
    benchmark_id = ""
    model_sha256 = ""
    model_relative_path = ""
    index_relative_path = ""
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]

    def __init__(
        self,
        checkpoint: Path,
        dataset: Path,
        codec=None,
        *,
        asset_lock: Path,
        inference_backend: str = REFERENCE_INFERENCE_BACKEND,
        split: str,
        injectable_parameters: tuple[str, ...] = ("kernels",),
        require_reference_equivalence: bool = True,
    ):
        del codec
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.dataset_path = Path(dataset).expanduser().resolve()
        self.asset_lock_path = Path(asset_lock).expanduser().resolve()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)
        if not self.dataset_path.exists():
            raise FileNotFoundError(self.dataset_path)
        if inference_backend != REFERENCE_INFERENCE_BACKEND:
            raise ValueError("M9.3 benchmarks require tensorflow_lite_builtin_ref_v1")
        if tuple(injectable_parameters) != ("kernels",):
            raise ValueError("M9.3 validates the kernel-only INT8 fault population")
        self.inference_backend = inference_backend
        self.split = split
        self.injectable_parameters = tuple(injectable_parameters)
        self.require_reference_equivalence = bool(require_reference_equivalence)
        self.asset_lock: VerifiedAssetLock | None = None
        self.storage: TFLiteModelStorage | None = None
        self._model_content: bytes | None = None
        self._interpreter = None
        self._runtime_provenance: dict[str, object] = {}
        self._companion_instance = None
        self._bounded_conformance: dict[str, object] | None = None

    def load(self) -> None:
        self.asset_lock = load_benchmark_asset_lock(
            str(self.asset_lock_path), self.benchmark_id, False
        )
        locked_datasets = {
            Path(value).expanduser().resolve()
            for value in (self.asset_lock.document.get("resolved", {}).get("datasets", {}) or {}).values()
        }
        if self.dataset_path not in locked_datasets:
            raise AssetLockError("dataset path differs from the pinned asset lock")
        locked_model = self.asset_lock.file(self.model_relative_path)
        # The same pinned inode can be exposed through /data and /mnt/raid0/ml
        # mount aliases on Kona.  Preserve path pinning for different files,
        # then enforce the recorded SHA-256 on the bytes below.
        if locked_model != self.checkpoint and not locked_model.samefile(self.checkpoint):
            raise AssetLockError("checkpoint path differs from the pinned asset lock")
        model_record = self.asset_lock.file_record(self.model_relative_path)
        if model_record.get("sha256") != self.model_sha256:
            raise AssetLockError("pinned model SHA-256 differs from the adapter contract")
        if sha256_path(self.checkpoint) != self.model_sha256:
            raise AssetLockError("checkpoint bytes differ from the pinned SHA-256")
        index_path = self.asset_lock.file(self.index_relative_path)
        index_record = self.asset_lock.file_record(self.index_relative_path)
        if sha256_path(index_path) != index_record.get("sha256"):
            raise AssetLockError("evaluation index differs from the pinned SHA-256")
        self.storage = TFLiteModelStorage.from_path(
            self.checkpoint,
            expected_sha256=self.model_sha256,
            eligible_roles=("kernel",),
        )
        self._set_model_content(self.storage.reference_content)
        if self.require_reference_equivalence:
            self._bounded_conformance = self._bounded_reference_conformance()

    def _require_loaded(self) -> TFLiteModelStorage:
        if self.storage is None or self._model_content is None:
            raise RuntimeError("adapter.load() must be called first")
        return self.storage

    def _set_model_content(self, content: bytes) -> None:
        import tensorflow as tf

        interpreter = tf.lite.Interpreter(
            model_content=bytes(content),
            experimental_delegates=[],
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
            num_threads=1,
        )
        interpreter.allocate_tensors()
        input_detail = interpreter.get_input_details()
        output_detail = interpreter.get_output_details()
        if len(input_detail) != 1 or len(output_detail) != 1:
            raise ValueError("pinned benchmark must have one input and one output")
        if tuple(input_detail[0]["shape"]) != (1, *self.input_shape) or input_detail[0]["dtype"] != np.int8:
            raise ValueError("unexpected pinned benchmark input contract")
        if tuple(output_detail[0]["shape"]) != (1, *self.output_shape) or output_detail[0]["dtype"] != np.int8:
            raise ValueError("unexpected pinned benchmark output contract")
        self._model_content = bytes(content)
        self._interpreter = interpreter
        self._runtime_provenance = {
            "library": "tensorflow",
            "tensorflow_version": tf.__version__,
            "interpreter": "tf.lite.Interpreter",
            "operator_resolver": "tf.lite.experimental.OpResolverType.BUILTIN_REF",
            "delegates": "none",
            "threads": 1,
            "recorded_reference_tensorflow_version": "2.14.1",
            "recorded_runtime_version_match": tf.__version__ == "2.14.1",
        }

    def _outputs(self, inputs: np.ndarray) -> np.ndarray:
        self._require_loaded()
        values = np.asarray(inputs)
        if values.dtype != np.int8 or tuple(values.shape[1:]) != self.input_shape:
            raise ValueError(f"inputs must be int8 with trailing shape {self.input_shape}")
        input_detail = self._interpreter.get_input_details()[0]
        output_detail = self._interpreter.get_output_details()[0]
        outputs = np.empty((len(values), *self.output_shape), dtype=np.int8)
        for index, sample in enumerate(values):
            self._interpreter.set_tensor(input_detail["index"], sample[None, ...])
            self._interpreter.invoke()
            outputs[index] = self._interpreter.get_tensor(output_detail["index"])[0]
        return outputs

    def values(self) -> dict[str, np.ndarray]:
        return self._require_loaded().values(self._model_content)

    def quantized_values(self) -> dict[str, np.ndarray]:
        return self.values()

    def parameter_specs(self) -> tuple[ParameterSpec, ...]:
        return self._require_loaded().parameter_specs()

    def bit_identities(self) -> tuple[BitIdentity, ...]:
        return self._require_loaded().bit_identities()

    def msb_scores(self, identities):
        return (
            {identity: float(identity.bit_index - 7) for identity in identities},
            {
                "implementation_version": "relative-word-bit-index/v1",
                "expression": "bit_index - 7 for the homogeneous INT8 kernel population",
                "word_widths": {"kernel": 8},
            },
        )

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

    def encoded_bit_transition(self, identity: BitIdentity):
        return self._require_loaded().bit_transition(self.values(), identity)

    def bit_perturbations(self) -> dict[str, np.ndarray]:
        storage = self._require_loaded()
        return affine_bit_perturbations(self.values(), storage.descriptors)

    def gradients(self, inputs, labels):
        return self._companion().gradients(inputs, labels)

    def per_example_gradients(self, inputs, labels):
        return self._companion().per_example_gradients(inputs, labels)

    def hessian_vector_product(self, inputs, labels, vectors):
        return self._companion().hessian_vector_product(inputs, labels, vectors)

    def polar_impacts(self, inputs, activation_source):
        raise NotImplementedError("M9.3 does not define qPolarGrad for these models")

    def evaluate_fault_schedule(self, inputs, labels, schedule, protection):
        storage = self._require_loaded()
        if schedule.model.persistence != "persistent_trial":
            raise ValueError(
                f"{self.benchmark_id} currently validates persistent_trial faults only"
            )
        content, report = storage.apply_schedule(schedule, protection)
        before = self._model_content
        self._set_model_content(content)
        try:
            result = self.evaluate(inputs, labels)
        finally:
            self._set_model_content(before)
        return result, report

    def representation_manifest(self) -> dict[str, object]:
        result = {
            "identity": self.benchmark_id,
            "suite_commit": MLPERF_TINY_COMMIT,
            "runtime": dict(self._runtime_provenance),
            "fault_population_assumption": (
                "equal logical per-bit Bernoulli probability over INT8 kernel storage; "
                "not a physical SEU-rate claim"
            ),
            **self._require_loaded().manifest(),
        }
        if self.asset_lock is not None:
            result["asset_lock"] = self.asset_lock.manifest()
        if self._companion_instance is not None:
            result["differentiable_companion"] = self._companion_instance.manifest()
        result["reference_equivalence_enforced"] = self.require_reference_equivalence
        if self._bounded_conformance is not None:
            result["bounded_reference_conformance"] = dict(self._bounded_conformance)
        return result

    def quantization_manifest(self) -> dict[str, object]:
        return {
            "scheme": "TFLite affine integer constant buffers",
            "eligible_storage": "INT8 kernels only",
            "excluded_storage": ["INT32 bias", "activations", "affine metadata"],
            "per_tensor_and_per_channel": True,
            "encoded_bit_order": "lsb_zero",
            "signed_encoding": "two's-complement XOR",
        }

    def dataset_identity_manifest(self) -> dict[str, object]:
        if self.asset_lock is None:
            raise RuntimeError("adapter is not loaded")
        benchmark = self.asset_lock.benchmark
        return {
            "dataset": dict(benchmark["dataset"]),
            "sample_index_sha256": benchmark["reference_results"]["sample_index_sha256"],
            "preprocessing_id": benchmark["reference_results"]["preprocessing_id"],
            "asset_lock_digest": self.asset_lock.digest,
            "local_path_excluded_from_scientific_identity": True,
        }

    def quantization_verification(self) -> dict[str, object]:
        storage = self._require_loaded()
        return {
            "reference_bytes_immutable": True,
            "private_copy_patch_round_trip": storage.model_content_from_values(self.values())
            == storage.reference_content,
            "eligible_values_are_integer_storage_codes": True,
        }

    def companion_correspondence(self, inputs: np.ndarray, labels=None) -> dict[str, object]:
        exact = self._outputs(inputs)
        approximate = self._companion().encoded_outputs(inputs)
        difference = np.abs(approximate.astype(np.int16) - exact.astype(np.int16))
        return {
            "sample_count": len(exact),
            "exact_code_fraction": float(np.mean(difference == 0)),
            "exact_output_vector_fraction": float(np.mean(np.all(difference == 0, axis=tuple(range(1, difference.ndim))))),
            "encoded_mae": float(np.mean(difference)),
            "encoded_max_error": int(np.max(difference)),
            "authoritative_inference": "TensorFlow 2.14.1 TFLite BUILTIN_REF",
            "companion_is_authoritative": False,
        }

    def _bounded_reference_conformance(self) -> dict[str, object]:
        raise NotImplementedError

    def _runtime_debug_identity(self) -> dict[str, object]:
        """Return diagnostic-only process identity, never scientific payload data."""

        import tensorflow as tf

        return {
            "pid": os.getpid(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "multiprocessing_start_method": multiprocessing.get_start_method(
                allow_none=True
            ),
            "tensorflow": tf.__version__,
            "numpy": np.__version__,
            "thread_environment": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                    "TF_NUM_INTRAOP_THREADS",
                    "TF_NUM_INTEROP_THREADS",
                    "TF_ENABLE_ONEDNN_OPTS",
                )
            },
        }


class MlperfTinyKwsDscnnAdapter(_ExactTFLiteKernelAdapter):
    benchmark_id = "mlperf-tiny/kws"
    model_sha256 = KWS_MODEL_SHA256
    model_relative_path = KWS_MODEL_RELATIVE
    index_relative_path = KWS_INDEX_RELATIVE
    input_shape = (49, 10, 1)
    output_shape = (12,)

    _M953_TEST_WINDOWS = {
        "conditional_calibration": (1000, 1024),
        "conditional_model_selection": (2024, 1024),
        "conditional_final_evaluation": (3048, 1842),
    }

    def __init__(self, *args, split="official_subset_1000", **kwargs):
        if split not in {"official_subset_1000", "test", "full_test_4890"}:
            raise ValueError("KWS split must be official_subset_1000, test, or full_test_4890")
        super().__init__(*args, split=split, **kwargs)
        self._record_cache: dict[str, list[tuple[np.ndarray, int]]] = {}
        self._record_iterators: dict[str, Iterator[dict]] = {}

    @staticmethod
    def _features(audio: np.ndarray) -> np.ndarray:
        import tensorflow as tf

        waveform = tf.cast(tf.constant(audio), tf.float32)
        waveform = waveform / tf.reduce_max(waveform)
        waveform = tf.pad(waveform, [[0, 16000 - tf.shape(waveform)[-1]]])
        padded = tf.pad(waveform, [[2, 2]])
        waveform = tf.slice(padded, [2], [16000])
        stft = tf.signal.stft(
            waveform, frame_length=480, frame_step=320,
            fft_length=None, window_fn=tf.signal.hann_window,
        )
        spectrogram = tf.abs(stft)
        weights = tf.signal.linear_to_mel_weight_matrix(
            40, stft.shape[-1], 16000, 20.0, 4000.0
        )
        mel = tf.tensordot(spectrogram, weights, 1)
        mel.set_shape(spectrogram.shape[:-1].concatenate(weights.shape[-1:]))
        mfcc = tf.signal.mfccs_from_log_mel_spectrograms(tf.math.log(mel + 1e-6))[..., :10]
        return tf.reshape(mfcc, (49, 10, 1)).numpy()

    def _records(self, split: str, stop: int):
        if self.asset_lock is None:
            raise RuntimeError("adapter is not loaded")
        paths = sorted(
            path for (asset, relative), path in self.asset_lock.resolved_files.items()
            if f"speech_commands-{split}.tfrecord" in relative
        )
        if not paths:
            raise AssetLockError(f"asset lock has no Speech Commands {split} shards")
        result = self._record_cache.setdefault(split, [])
        iterator = self._record_iterators.setdefault(
            split, iter(interleaved_examples(paths))
        )
        while len(result) < stop:
            try:
                example = next(iterator)
            except StopIteration:
                break
            audio = np.asarray(example["audio"], dtype=np.int16)
            label = int(np.asarray(example["label"]).reshape(-1)[0])
            result.append((audio, label))
        if len(result) < stop:
            raise ValueError(f"Speech Commands {split} contains fewer than {stop} examples")
        return result[:stop]

    def sample_partition(self, role: str, count: int, *, offset: int = 0):
        split_by_role = {
            "ranking": "train", "calibration": "train",
            "empirical_validation": "validation",
            "evaluation": "test", "final_evaluation": "test",
        }
        m953_window = self._M953_TEST_WINDOWS.get(role)
        if m953_window is not None:
            if self.split != "full_test_4890":
                raise ValueError(
                    f"KWS role {role!r} requires benchmark split full_test_4890"
                )
            start, capacity = m953_window
            if count < 1 or offset < 0 or offset + count > capacity:
                raise ValueError(
                    f"KWS role {role!r} permits relative indices [0,{capacity}); "
                    f"requested offset={offset}, count={count}"
                )
            split = "test"
            original_offset = start + offset
        elif role not in split_by_role or count < 1 or offset < 0:
            raise ValueError(f"unsupported or invalid KWS sample partition: {role}")
        else:
            split = split_by_role[role]
            original_offset = offset
            if split == "test" and self.split == "official_subset_1000" and offset + count > 1000:
                raise ValueError("official KWS evaluation subset contains 1000 samples")
        selected = self._records(split, original_offset + count)[original_offset:]
        features = np.stack([self._features(audio) for audio, _ in selected]).astype(np.float32)
        labels = np.asarray([label for _, label in selected], dtype=np.int64)
        input_detail = self._interpreter.get_input_details()[0]
        scale, zero_point = input_detail["quantization"]
        encoded = np.asarray(features / scale + zero_point, dtype=np.int8)
        indices = tuple(range(original_offset, original_offset + count))
        identity = f"speech_commands_tfds_{split}_interleaved_blocks16"
        if split == "test" and indices[-1] < 1000:
            index_path = self.asset_lock.file(self.index_relative_path)
            with index_path.open("r", encoding="utf-8", newline="") as stream:
                rows = tuple(csv.reader(stream))
            for local, index in enumerate(indices):
                if int(rows[index][2]) != int(labels[local]):
                    raise AssetLockError(f"KWS official label mismatch at test row {index}")
        return encoded, labels, SamplePartition(
            role, identity, indices,
            sample_identity_digest(identity, indices, labels),
            (
                "Pinned TFDS shard order: round-robin blocks of 16; exact TF signal MFCC "
                "preprocessing; M9.5.3 excludes prior official indices 0-999 and uses fixed "
                "same-distribution windows 1000:2024, 2024:3048, and 3048:4890"
                if m953_window is not None else
                "Pinned TFDS shard order: round-robin blocks of 16; exact TF signal MFCC preprocessing"
            ),
        )

    def same_distribution_window_diagnostics(self) -> dict[str, dict]:
        """Audit complete fixed M9.5 windows without using their outcomes for selection."""

        records = self._records("test", 4890)
        identity = "speech_commands_tfds_test_interleaved_blocks16"
        result = {}
        for role, (start, count) in self._M953_TEST_WINDOWS.items():
            labels = np.asarray(
                [label for _audio, label in records[start:start + count]], dtype=np.int64
            )
            values, counts = np.unique(labels, return_counts=True)
            indices = tuple(range(start, start + count))
            result[role] = {
                "source_split": identity,
                "selection_rule": f"contiguous pinned source-order indices [{start},{start + count})",
                "first_index": start,
                "last_index": start + count - 1,
                "count": count,
                "class_counts": {
                    str(int(value)): int(observed)
                    for value, observed in zip(values, counts)
                },
                "label_bound_sample_sha256": sample_identity_digest(identity, indices, labels),
                "ordering": "round-robin blocks of 16 over four hash-locked TFRecord shards",
            }
        return result

    def dataset(self, count: int, *, role="evaluation"):
        x, y, _ = self.sample_partition(role, count)
        return x, y

    def predict_encoded_outputs(self, inputs):
        return self._outputs(inputs)

    def predict_outputs(self, inputs):
        values = self._outputs(inputs)
        return (values.astype(np.float64) + 128.0) / 256.0

    def predict_classes(self, inputs):
        return np.argmax(self._outputs(inputs), axis=1)

    def evaluate(self, inputs, labels, *, batch_size=256):
        del batch_size
        probabilities = self.predict_outputs(inputs)
        labels = np.asarray(labels, dtype=np.int64)
        selected = np.maximum(probabilities[np.arange(len(labels)), labels], 1e-12)
        losses = -np.log(selected)
        capture = getattr(self, "_input_loss_capture", None)
        if capture is not None:
            capture.append(losses.copy())
        return ClassificationEvaluation(
            float(np.mean(np.argmax(probabilities, axis=1) == labels)),
            float(np.mean(losses)), len(labels),
        )

    @contextmanager
    def capture_input_losses(self):
        """Opt-in KWS diagnostics from the same authoritative inference calls.

        Neither the scalar reduction nor model restoration is changed. No output
        is captured outside this explicit context, including after exceptions.
        """
        if getattr(self, "_input_loss_capture", None) is not None:
            raise RuntimeError("nested input-loss capture is unsupported")
        captured = []
        self._input_loss_capture = captured
        try:
            yield captured
        finally:
            del self._input_loss_capture

    def _companion(self):
        if self._companion_instance is None:
            from ..mlperf_multibench_companion import MlperfKwsDscnnDifferentiableCompanion
            self._companion_instance = MlperfKwsDscnnDifferentiableCompanion(
                self._require_loaded().reference_content
            )
        return self._companion_instance

    def companion_correspondence(self, inputs: np.ndarray, labels=None) -> dict[str, object]:
        result = super().companion_correspondence(inputs, labels)
        exact_codes = self._outputs(inputs)
        approximate_codes = self._companion().encoded_outputs(inputs)
        if labels is not None:
            labels = np.asarray(labels, dtype=np.int64)
            exact = self.evaluate(inputs, labels)
            companion_losses = self._companion().losses(inputs, labels).numpy()
            result.update({
                "class_agreement": float(np.mean(
                    np.argmax(exact_codes, axis=1)
                    == np.argmax(approximate_codes, axis=1)
                )),
                "authoritative_accuracy": exact.accuracy,
                "companion_accuracy": float(np.mean(
                    np.argmax(approximate_codes, axis=1) == labels
                )),
                "authoritative_crossentropy": exact.categorical_crossentropy,
                "companion_surrogate_crossentropy": float(np.mean(companion_losses)),
                "loss_objectives_are_identical": False,
            })
        return result

    def reference_conformance(self, *, full_test=False):
        count = 4890 if full_test else 1000
        inputs, labels = self.dataset(count)
        outputs = self._outputs(inputs)
        reference = self.asset_lock.benchmark["reference_results"]
        key = "full_outputs_sha256" if full_test else "subset_outputs_sha256"
        expected_accuracy = float(reference["full_test_4890" if full_test else "official_subset_1000"]["value"])
        accuracy = float(np.mean(np.argmax(outputs, axis=1) == labels))
        digest = hashlib.sha256(outputs.tobytes(order="C")).hexdigest()
        return {
            "sample_count": count, "accuracy": accuracy,
            "expected_accuracy": expected_accuracy, "accuracy_match": accuracy == expected_accuracy,
            "output_sha256": digest,
            "expected_output_sha256": reference["output_digests"][key],
            "exact_output_digest_match": digest == reference["output_digests"][key],
        }

    def bounded_reference_diagnostics(self) -> dict[str, object]:
        """Replay the KWS gate and expose process-local byte identities."""

        inputs, labels, partition = self.sample_partition("evaluation", 16)
        outputs = self._outputs(inputs)
        digest = hashlib.sha256(outputs.tobytes(order="C")).hexdigest()
        expected = "b88c99a8768eb1fcf3b563246501d277dbe55543883ffd58122b138dca87ef42"
        return {
            "runtime": self._runtime_debug_identity(),
            "sample_ids": list(partition.indices),
            "sample_sha256": partition.sample_sha256,
            "preprocessed_int8_sha256": hashlib.sha256(
                inputs.tobytes(order="C")
            ).hexdigest(),
            "exact_tflite_output_sha256": digest,
            "expected_output_sha256": expected,
            "accuracy": float(np.mean(np.argmax(outputs, axis=1) == labels)),
            "exact_output_digest_match": digest == expected,
        }

    def _bounded_reference_conformance(self) -> dict[str, object]:
        diagnostics = self.bounded_reference_diagnostics()
        if not diagnostics["exact_output_digest_match"]:
            raise AssetLockError(
                "KWS bounded preprocessing/reference output digest differs from "
                "the full-conformance-derived gate"
            )
        return {
            "sample_count": 16,
            "sample_sha256": diagnostics["sample_sha256"],
            "output_sha256": diagnostics["exact_tflite_output_sha256"],
            "expected_output_sha256": diagnostics["expected_output_sha256"],
            "accuracy": diagnostics["accuracy"],
            "exact_output_digest_match": True,
        }

    def preprocessing_manifest(self):
        return {"id": "mlperf-tiny/kws01/tf-signal-mfcc-10x49/1", "input_shape": [1, 49, 10, 1], "int8_rounding": "truncation_toward_zero"}

    def task_metric_manifest(self):
        return {"task_type": "classification", "primary_metric": "classification_accuracy", "ranking_loss": "sparse_crossentropy_on_requantized_companion_logits", "exact_loss": "categorical_crossentropy_from_integer_softmax"}


class MlperfTinyToyCarAdapter(_ExactTFLiteKernelAdapter):
    benchmark_id = "mlperf-tiny/ad"
    model_sha256 = AD_MODEL_SHA256
    model_relative_path = AD_MODEL_RELATIVE
    index_relative_path = AD_INDEX_RELATIVE
    input_shape = (640,)
    output_shape = (640,)

    def __init__(self, *args, split="official_248_files", **kwargs):
        if split not in {"official_248_files", "test"}:
            raise ValueError("ToyCar split must be official_248_files or test")
        super().__init__(*args, split=split, **kwargs)

    @staticmethod
    def _vectors(path: Path) -> np.ndarray:
        import librosa
        import sys
        try:
            from threadpoolctl import threadpool_limits
        except ImportError as error:
            raise ImportError(
                "ToyCar exact preprocessing requires threadpoolctl so the pinned "
                "two-thread OpenBLAS reduction can be reproduced"
            ) from error

        audio, sample_rate = librosa.load(str(path), sr=None, mono=False)
        # The locked full-conformance scores were produced by the pinned NumPy
        # OpenBLAS build, whose compile-time default is two threads.  NetSuRF's
        # spawned workers otherwise reduce OpenBLAS to one thread and alter the
        # float64 reconstruction targets by a few ulps even though encoded
        # inputs and exact TFLite outputs remain byte-identical.  Scope the
        # original numerical contract to preprocessing only; TFLite remains a
        # one-thread BUILTIN_REF interpreter.
        with threadpool_limits(
            limits=TOYCAR_PREPROCESSING_BLAS_THREADS, user_api="blas"
        ):
            mel = librosa.feature.melspectrogram(
                y=audio, sr=sample_rate, n_fft=1024, hop_length=512,
                n_mels=128, power=2.0, pad_mode="reflect",
            )
        logmel = 10.0 * np.log10(mel + sys.float_info.epsilon)[:, 50:250]
        width = logmel.shape[1] - 4
        vectors = np.zeros((width, 640), dtype=np.float64)
        for step in range(5):
            vectors[:, 128 * step : 128 * (step + 1)] = logmel[:, step : step + width].T
        return vectors

    def _official_files(self):
        rows = tuple(csv.reader(self.asset_lock.file(self.index_relative_path).open("r", encoding="utf-8", newline="")))
        root = self.dataset_path / "dev_data_ToyCar" / "ToyCar" / "test"
        result = []
        for index, row in enumerate(rows):
            name, _, label, width, stride = row
            stem = name.replace("_hist_librosa.bin", "")
            if width != "2560" or stride != "512":
                raise AssetLockError("ToyCar evaluation window contract differs")
            path = root / f"{stem}.wav"
            if not path.is_file():
                raise AssetLockError(f"missing official ToyCar file: {path.name}")
            result.append((index, path, int(label)))
        return tuple(result)

    @staticmethod
    def _balanced_indices(records, count: int, offset: int):
        normal = [index for index, (_, _, label) in enumerate(records) if label == 0]
        anomaly = [index for index, (_, _, label) in enumerate(records) if label == 1]
        order = [value for pair in zip(normal, anomaly) for value in pair]
        order.extend(normal[len(anomaly):])
        order.extend(anomaly[len(normal):])
        if offset + count > len(order):
            raise ValueError("ToyCar evaluation selection exceeds 248 official files")
        return tuple(order[offset : offset + count])

    @staticmethod
    def _held_out_indices(records, role: str):
        """Disjoint, label-containing halves of the official development test."""

        normal = [index for index, (_, _, label) in enumerate(records) if label == 0]
        anomaly = [index for index, (_, _, label) in enumerate(records) if label == 1]
        parity = 0 if role == "empirical_validation" else 1
        selected_normal = normal[parity::2]
        selected_anomaly = anomaly[parity::2]
        order = [value for pair in zip(selected_normal, selected_anomaly) for value in pair]
        order.extend(selected_normal[len(selected_anomaly):])
        order.extend(selected_anomaly[len(selected_normal):])
        return tuple(order)

    def sample_partition(self, role: str, count: int, *, offset: int = 0):
        if count < 1 or offset < 0:
            raise ValueError("ToyCar sample count must be positive")
        if role in {"ranking", "calibration"}:
            root = self.dataset_path / "dev_data_ToyCar" / "ToyCar" / "train"
            records = tuple((index, path, 0) for index, path in enumerate(sorted(root.glob("*.wav"))))
            selected_positions = tuple(range(offset, offset + count))
            source = "dcase2020_toycar_development_train"
        elif role in {"empirical_validation", "evaluation", "final_evaluation"}:
            if count < 2:
                raise ValueError(
                    "ToyCar empirical/final evaluation requires at least one normal "
                    "and one anomalous file"
                )
            records = self._official_files()
            pool_role = (
                "empirical_validation"
                if role == "empirical_validation"
                else "evaluation"
            )
            pool = self._held_out_indices(records, pool_role)
            if offset + count > len(pool):
                raise ValueError(
                    f"ToyCar {role} selection exceeds its {len(pool)}-file "
                    "disjoint held-out half"
                )
            selected_positions = tuple(pool[offset : offset + count])
            source = "dcase2020_toycar_official_development_test"
        elif role == "reference_conformance":
            records = self._official_files()
            if count != len(records) or offset != 0:
                raise ValueError("ToyCar reference conformance requires all 248 files")
            selected_positions = tuple(range(len(records)))
            source = "dcase2020_toycar_official_development_test"
        else:
            raise ValueError(f"unsupported ToyCar sample role: {role}")
        if selected_positions and max(selected_positions) >= len(records):
            raise ValueError(f"ToyCar {role} selection exceeds available files")
        selected = [records[position] for position in selected_positions]
        vectors_by_file = [self._vectors(path) for _, path, _ in selected]
        vectors = np.concatenate(vectors_by_file, axis=0)
        input_detail = self._interpreter.get_input_details()[0]
        scale, zero_point = input_detail["quantization"]
        encoded = np.clip(np.round(vectors / scale + zero_point), -128, 127).astype(np.int8)
        file_ids = np.concatenate([
            np.full(len(values), local, dtype=np.float64)
            for local, values in enumerate(vectors_by_file)
        ])
        file_labels = np.concatenate([
            np.full(len(values), label, dtype=np.float64)
            for values, (_, _, label) in zip(vectors_by_file, selected)
        ])
        labels = np.column_stack((vectors, file_ids, file_labels))
        original_indices = tuple(int(records[position][0]) for position in selected_positions)
        digest_labels = tuple(int(value[2]) for value in selected)
        return encoded, labels, SamplePartition(
            role, source, original_indices,
            sample_identity_digest(source, original_indices, digest_labels),
            "Each configured sample is one audio file; arrays contain all 196 reconstruction windows per file",
        )

    def dataset(self, count: int, *, role="evaluation"):
        x, y, _ = self.sample_partition(role, count)
        return x, y

    def predict_encoded_outputs(self, inputs):
        return self._outputs(inputs)

    def predict_outputs(self, inputs):
        encoded = self._outputs(inputs)
        output_detail = self._interpreter.get_output_details()[0]
        scale, zero_point = output_detail["quantization"]
        return (encoded.astype(np.float64) - zero_point) * scale

    @staticmethod
    def _targets(labels):
        values = np.asarray(labels)
        if values.ndim != 2 or values.shape[1] != 642:
            raise ValueError("ToyCar labels must contain targets, file identity, and anomaly label")
        return values[:, :640], values[:, 640].astype(np.int64), values[:, 641].astype(np.int64)

    @staticmethod
    def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
        labels = np.asarray(labels, dtype=np.int64)
        if set(labels.tolist()) != {0, 1}:
            raise ValueError("ROC-AUC requires both normal and anomalous files")
        positive = scores[labels == 1]
        negative = scores[labels == 0]
        comparisons = (positive[:, None] > negative[None, :]).astype(np.float64)
        comparisons += 0.5 * (positive[:, None] == negative[None, :])
        return float(np.mean(comparisons))

    def evaluate(self, inputs, labels, *, batch_size=256):
        del batch_size
        targets, file_ids, file_labels_by_window = self._targets(labels)
        outputs = self.predict_outputs(inputs)
        window_errors = np.mean((targets - outputs) ** 2, axis=1)
        unique_files = np.unique(file_ids)
        scores = np.asarray([np.mean(window_errors[file_ids == value]) for value in unique_files])
        file_labels = np.asarray([file_labels_by_window[np.flatnonzero(file_ids == value)[0]] for value in unique_files])
        return AnomalyEvaluation(
            float(np.mean(window_errors)), self._roc_auc(file_labels, scores),
            len(unique_files), len(window_errors), tuple(float(value) for value in scores),
        )

    def file_score_records(
        self,
        inputs,
        labels,
        *,
        schedule: FaultSchedule | None = None,
        protection: ProtectionMask | None = None,
    ) -> tuple[tuple[dict[str, object], ...], InjectionReport | None]:
        """Return bounded file-level score evidence with optional exact faults.

        This is an opt-in diagnostic API.  Production bundles are unchanged;
        callers decide whether the additional per-file evidence is warranted.
        """

        def records():
            targets, file_ids, file_labels_by_window = self._targets(labels)
            outputs = self.predict_outputs(inputs)
            window_errors = np.mean((targets - outputs) ** 2, axis=1)
            result = []
            for file_id in np.unique(file_ids):
                selected = file_ids == file_id
                label = int(file_labels_by_window[np.flatnonzero(selected)[0]])
                score = float(np.mean(window_errors[selected]))
                result.append({
                    "file_id": int(file_id),
                    "label": label,
                    "score": score,
                    "score_float64_hex": score.hex(),
                    "window_count": int(np.sum(selected)),
                })
            return tuple(result)

        if schedule is None:
            if protection is not None:
                raise ValueError("protection requires a fault schedule")
            return records(), None
        if protection is None:
            raise ValueError("faulted file-score diagnostics require a protection mask")
        storage = self._require_loaded()
        content, report = storage.apply_schedule(schedule, protection)
        before = self._model_content
        self._set_model_content(content)
        try:
            result = records()
        finally:
            self._set_model_content(before)
        if self._model_content != before:
            raise RuntimeError("TFLite reference model was not restored")
        return result, report

    def _companion(self):
        if self._companion_instance is None:
            from ..mlperf_multibench_companion import MlperfToyCarDifferentiableCompanion
            self._companion_instance = MlperfToyCarDifferentiableCompanion(
                self._require_loaded().reference_content
            )
        return self._companion_instance

    def companion_correspondence(self, inputs: np.ndarray, labels=None) -> dict[str, object]:
        result = super().companion_correspondence(inputs, labels)
        if labels is not None:
            targets, file_ids, file_labels_by_window = self._targets(labels)
            authoritative_outputs = self.predict_outputs(inputs)
            authoritative_errors = np.mean(
                (targets - authoritative_outputs) ** 2, axis=1
            )
            companion_outputs = self._companion().forward(inputs).numpy().astype(np.float64)
            errors = np.mean((targets - companion_outputs) ** 2, axis=1)
            unique_files = np.unique(file_ids)
            scores = np.asarray([
                np.mean(errors[file_ids == value]) for value in unique_files
            ])
            file_labels = np.asarray([
                file_labels_by_window[np.flatnonzero(file_ids == value)[0]]
                for value in unique_files
            ])
            result.update({
                "authoritative_reconstruction_mse": float(np.mean(authoritative_errors)),
                "companion_reconstruction_mse": float(np.mean(errors)),
                "reconstruction_mse_absolute_difference": float(
                    abs(np.mean(errors) - np.mean(authoritative_errors))
                ),
                "loss_objectives_are_identical": True,
            })
            if set(file_labels.tolist()) == {0, 1}:
                authoritative_scores = np.asarray([
                    np.mean(authoritative_errors[file_ids == value])
                    for value in unique_files
                ])
                result.update({
                    "authoritative_roc_auc": self._roc_auc(
                        file_labels, authoritative_scores
                    ),
                    "companion_roc_auc": self._roc_auc(file_labels, scores),
                })
        return result

    def reference_conformance(self):
        inputs, labels = self.dataset(248, role="reference_conformance")
        result = self.evaluate(inputs, labels)
        scores = np.asarray(result.file_scores, dtype=np.float64)
        reference = self.asset_lock.benchmark["reference_results"]
        digest = hashlib.sha256(scores.tobytes(order="C")).hexdigest()
        expected = reference["output_digests"]["file_scores_sha256"]
        expected_auc = float(reference["official_248_files"]["value"])
        return {
            "file_count": 248, "window_count": result.window_count,
            "roc_auc": result.roc_auc, "expected_roc_auc": expected_auc,
            "roc_auc_match": result.roc_auc == expected_auc,
            "file_scores_sha256": digest, "expected_file_scores_sha256": expected,
            "exact_score_digest_match": digest == expected,
        }

    def bounded_reference_diagnostics(self) -> dict[str, object]:
        """Replay the exact two-file gate with process-local diagnostic evidence."""

        inputs, labels, partition = self.sample_partition("evaluation", 2)
        outputs = self.predict_encoded_outputs(inputs)
        result = self.evaluate(inputs, labels)
        scores = np.asarray(result.file_scores, dtype=np.float64)
        expected_scores = np.asarray(
            TOYCAR_BOUNDED_EXPECTED_SCORES, dtype=np.float64
        )
        differing = np.flatnonzero(
            scores.view(np.uint64) != expected_scores.view(np.uint64)
        )
        first_difference = None
        if len(differing):
            index = int(differing[0])
            first_difference = {
                "index": index,
                "observed": float(scores[index]),
                "observed_hex": float(scores[index]).hex(),
                "expected": float(expected_scores[index]),
                "expected_hex": float(expected_scores[index]).hex(),
                "absolute_difference": float(
                    abs(scores[index] - expected_scores[index])
                ),
            }
        digest = hashlib.sha256(scores.tobytes(order="C")).hexdigest()
        return {
            "runtime": self._runtime_debug_identity(),
            "preprocessing_blas_threads": TOYCAR_PREPROCESSING_BLAS_THREADS,
            "sample_ids": list(partition.indices),
            "sample_sha256": partition.sample_sha256,
            "preprocessed_int8_sha256": hashlib.sha256(
                inputs.tobytes(order="C")
            ).hexdigest(),
            "exact_tflite_output_sha256": hashlib.sha256(
                outputs.tobytes(order="C")
            ).hexdigest(),
            "file_scores": [float(value) for value in scores],
            "file_score_float64_hex": [float(value).hex() for value in scores],
            "file_scores_sha256": digest,
            "expected_file_scores_sha256": TOYCAR_BOUNDED_EXPECTED_SCORE_SHA256,
            "first_differing_value": first_difference,
            "window_count": result.window_count,
            "roc_auc": result.roc_auc,
            "exact_score_digest_match": (
                digest == TOYCAR_BOUNDED_EXPECTED_SCORE_SHA256
            ),
        }

    def _bounded_reference_conformance(self) -> dict[str, object]:
        diagnostics = self.bounded_reference_diagnostics()
        if not diagnostics["exact_score_digest_match"]:
            raise AssetLockError(
                "ToyCar bounded preprocessing/reference score digest differs "
                "from the full-conformance-derived gate; first difference: "
                f"{diagnostics['first_differing_value']}"
            )
        return {
            "file_count": 2,
            "window_count": diagnostics["window_count"],
            "sample_sha256": diagnostics["sample_sha256"],
            "preprocessed_int8_sha256": diagnostics[
                "preprocessed_int8_sha256"
            ],
            "exact_tflite_output_sha256": diagnostics[
                "exact_tflite_output_sha256"
            ],
            "file_scores_sha256": diagnostics["file_scores_sha256"],
            "expected_file_scores_sha256": diagnostics[
                "expected_file_scores_sha256"
            ],
            "preprocessing_blas_threads": diagnostics[
                "preprocessing_blas_threads"
            ],
            "roc_auc": diagnostics["roc_auc"],
            "exact_score_digest_match": True,
        }

    def preprocessing_manifest(self):
        return {
            "id": "mlperf-tiny/ad01/librosa-logmel-128x5/1",
            "input_shape": [1, 640],
            "rounding": "numpy_round_half_to_even",
            "file_windows": 196,
            "blas_threads": TOYCAR_PREPROCESSING_BLAS_THREADS,
            "blas_thread_reason": "pinned full-conformance float64 target identity",
        }

    def task_metric_manifest(self):
        return {
            "task_type": "anomaly_detection",
            "primary_metric": {"name": "roc_auc", "direction": "maximize", "unit": "fraction", "aggregation": "file_level"},
            "ranking_loss": {"name": "reconstruction_mse", "direction": "minimize", "aggregation": "mean_over_float_target_windows"},
            "alignment": "S4-S6 differentiate per-window reconstruction MSE; final ROC-AUC ranks file-level anomaly scores and is not differentiated",
            "forbidden_aliases": ["classification_accuracy", "categorical_crossentropy"],
        }
