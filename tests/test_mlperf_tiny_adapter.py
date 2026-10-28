from __future__ import annotations

import importlib
import os
from pathlib import Path
import pickle
import sys
import tomllib

import numpy as np
import pytest

from netsurf.experiment_config import ConfigurationError, validate_experiment_document
from netsurf.legacy_identity import load_benchmark_identity
from netsurf.study_config import StudyConfigurationError, load_study


def _modern_document():
    document = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "config/experiments/numpy_reference.toml").read_text(
            encoding="utf-8"
        )
    )
    document["benchmark"].update(
        {
            "adapter": "mlperf_tiny_ic_int8_v1",
            "id": "mlperf_tiny_ic_resnet8_int8_v1",
            "checkpoint": "/verified/pretrainedResnet_quant.tflite",
            "dataset": "/verified/cifar-10-python.tar.gz",
            "quantization": "tflite_affine_int8",
            "quantization_backend": "tflite_affine_integer_v1",
            "injectable_parameters": ["kernels", "biases"],
        }
    )
    document["estimators"] = [
        {
            "name": "msb_first",
            "implementation_version": "relative-word-bit-index/v1",
            "policy": "global_ranking",
            "result_name": "msb",
        }
    ]
    return document


def test_modern_adapter_import_has_no_deprecated_runtime_dependencies():
    importlib.import_module("netsurf.adapters.mlperf_tiny")
    forbidden = (
        "DEPRECATED",
        "_netsurf_legacy",
        "qkeras",
        "tensorflow_model_optimization",
    )
    loaded = tuple(sys.modules)
    assert not [name for name in loaded if name.startswith(forbidden)]


def test_mlperf_identity_records_dataset_and_gradient_blockers():
    identity = load_benchmark_identity("mlperf_tiny_ic_resnet8_int8_v1")
    assert identity["source"]["checkpoint_sha256"] == (
        "3c002613d1b2475eb51dd78dfb85a546c8ae658dee71cf6ade43b022fe205415"
    )
    assert identity["faultable_storage"]["eligible_scalars"] == 77_706
    assert identity["faultable_storage"]["eligible_bits"] == 629_952
    assert identity["task"]["measured_clean_performance"] is None
    assert "tensor-mapped" in identity["capabilities"]["gradient_blocker"]


def test_mlperf_configuration_accepts_only_exact_storage_contract():
    configuration = validate_experiment_document(
        _modern_document(), Path("/tmp/mlperf.toml"), check_inputs=False
    )
    assert configuration.benchmark.quantization_backend == "tflite_affine_integer_v1"
    assert configuration.benchmark.injectable_parameters == ("kernels", "biases")

    invalid = _modern_document()
    invalid["estimators"][0]["name"] = "first_order_gradient_delta"
    with pytest.raises(ConfigurationError, match="differentiable checkpoint"):
        validate_experiment_document(invalid, Path("/tmp/mlperf.toml"), check_inputs=False)

    invalid = _modern_document()
    invalid["estimators"][0]["policy"] = "position_first"
    with pytest.raises(ConfigurationError, match="INT8 kernels and INT32 biases"):
        validate_experiment_document(invalid, Path("/tmp/mlperf.toml"), check_inputs=False)


def test_schema3_accepts_bounded_empirical_contract_and_rejects_raw_bit_tiers(tmp_path):
    text = '''
schema_version = "3.1.0"
scientific_protocol = "single-bit-susceptibility/v1"
policies = []
[experiment]
name = "mlperf_bounded_empirical"
description = "Validation-only configuration fixture."
objective = "Exercise exact affine-bit empirical configuration."
[benchmark]
adapter = "mlperf_tiny_ic_int8_v1"
id = "mlperf_tiny_ic_resnet8_int8_v1"
checkpoint = "/verified/model.tflite"
dataset = "/verified/cifar.tar.gz"
split = "explicit_disjoint"
quantization = "tflite_affine_int8"
quantization_backend = "tflite_affine_integer_v1"
injectable_parameters = ["kernels", "biases"]
[samples]
ranking_count = 4
empirical_count = 4
evaluation_count = 4
require_disjoint = true
[candidates]
strategy = "uniform"
count = 8
seed = 71
score_quantiles = 1
[[estimators]]
name = "msb_first"
implementation_version = "relative-word-bit-index/v1"
[protection]
coverages = []
mitigation_model = "perfect_event_suppression"
[fault]
target = "weight"
location = "logical_trainable_parameter_bits"
persistence = "persistent_trial"
distribution = "independent_bernoulli"
probability = 0.0
[execution]
trials = 1
confidence = 0.95
fault_seed = 100
permutation_seed = 101
[output]
root = ""
report = true
plots = false
'''
    path = tmp_path / "empirical.toml"
    path.write_text(text, encoding="utf-8")
    configuration = load_study(path, check_inputs=False)
    assert configuration.candidates.strategy == "uniform"
    assert configuration.estimators[0].implementation_version == "relative-word-bit-index/v1"

    invalid = text.replace(
        'policies = []',
        '[[policies]]\nname = "raw"\nestimator = "msb_first"\npriority = ["bit_significance_desc"]\nprespecified = true',
        1,
    ).replace(
        'scientific_protocol = "single-bit-susceptibility/v1"',
        'scientific_protocol = "multi-coverage-protection/v1"',
        1,
    ).replace("coverages = []", "coverages = [0.1]", 1)
    path.write_text(invalid, encoding="utf-8")
    with pytest.raises(StudyConfigurationError, match="raw bit_significance_desc"):
        load_study(path, check_inputs=False)


def _write_batch(path: Path, count: int, label_offset: int) -> None:
    data = np.arange(count * 3 * 32 * 32, dtype=np.uint32).reshape(count, -1)
    payload = {
        b"data": (data % 256).astype(np.uint8),
        b"labels": [(label_offset + index) % 10 for index in range(count)],
    }
    with path.open("wb") as stream:
        pickle.dump(payload, stream)


def test_cifar_python_layout_and_integer_preprocessing(tmp_path):
    from netsurf.adapters.mlperf_tiny import (
        MlperfTinyImageClassificationAdapter,
        load_cifar10_python,
    )

    root = tmp_path / "cifar-10-batches-py"
    root.mkdir()
    for index in range(1, 6):
        _write_batch(root / f"data_batch_{index}", 2, index)
    _write_batch(root / "test_batch", 3, 0)
    images, labels = load_cifar10_python(tmp_path, "test")
    assert images.shape == (3, 32, 32, 3)
    assert labels.tolist() == [0, 1, 2]
    encoded = MlperfTinyImageClassificationAdapter.preprocess(images)
    assert encoded.dtype == np.int8
    assert int(encoded.reshape(-1)[0]) == -128
    assert int(encoded.min()) == -128
    assert int(encoded.max()) == 127


def _official_model() -> Path:
    configured = os.environ.get("NETSURF_MLPERF_TINY_IC_MODEL")
    if configured:
        return Path(configured)
    return Path(
        "/home/manuelbv/.cache/kona-assets/models/models--mlcommons--tiny/snapshots/"
        "4addd0fa08d216e20637637874e084895f289da4/benchmark/training/"
        "image_classification/trained_models/pretrainedResnet_quant.tflite"
    )


@pytest.mark.skipif(not _official_model().is_file(), reason="official pinned TFLite asset is absent")
def test_official_tflite_storage_identity_xor_protection_and_restore(tmp_path):
    from netsurf.adapters.mlperf_tiny import MlperfTinyImageClassificationAdapter
    from netsurf.faults import FaultEvent, FaultModel, FaultSchedule
    from netsurf.protection import ProtectionMask

    adapter = MlperfTinyImageClassificationAdapter(_official_model(), tmp_path)
    adapter.load()
    manifest = adapter.representation_manifest()
    assert manifest["reference_sha256"] == (
        "3c002613d1b2475eb51dd78dfb85a546c8ae658dee71cf6ade43b022fe205415"
    )
    assert manifest["eligible_tensor_count"] == 20
    assert manifest["eligible_scalar_count"] == 77706
    assert manifest["eligible_bit_count"] == 629952
    assert len(adapter.parameter_specs()) == 20

    identities = adapter.bit_identities()
    scores, metadata = adapter.msb_scores(identities)
    kernel_sign = next(bit for bit in identities if bit.bit_index == 7 and "/kernel" in bit.parameter)
    bias_sign = next(bit for bit in identities if bit.bit_index == 31 and "/bias" in bit.parameter)
    assert scores[kernel_sign] == scores[bias_sign] == 0.0
    assert metadata["implementation_version"] == "relative-word-bit-index/v1"

    zero_bit = None
    state = adapter.values()
    for bit in identities:
        if bit.bit_index == 0 and state[bit.parameter].reshape(-1)[bit.flat_index] == 0:
            zero_bit = bit
            break
    assert zero_bit is not None
    second_bit = type(zero_bit)(zero_bit.parameter, zero_bit.flat_index, 1)
    model = FaultModel("weight", "logical_trainable_parameter_bits", "persistent_trial", "independent_bernoulli", 0.0)
    schedule = FaultSchedule(
        model,
        91,
        (FaultEvent(None, zero_bit), FaultEvent(None, second_bit)),
        len(identities),
        1,
    )
    no_protection = ProtectionMask("none", frozenset(), len(identities), 0.0, {})
    content, report = adapter.storage.apply_schedule(schedule, no_protection)
    changed = adapter.storage.values(content)
    assert int(changed[zero_bit.parameter].reshape(-1)[zero_bit.flat_index]) == 3
    assert (report.proposed_events, report.applied_events, report.suppressed_events) == (2, 2, 0)

    partial = ProtectionMask("one", frozenset({zero_bit}), len(identities), 1.0, {})
    content, report = adapter.storage.apply_schedule(schedule, partial)
    changed = adapter.storage.values(content)
    assert int(changed[zero_bit.parameter].reshape(-1)[zero_bit.flat_index]) == 2
    assert (report.applied_events, report.suppressed_events) == (1, 1)

    full = ProtectionMask("all", frozenset(identities), len(identities), float(len(identities)), {})
    content, report = adapter.storage.apply_schedule(schedule, full)
    assert content == adapter.storage.reference_content
    assert (report.applied_events, report.suppressed_events) == (0, 2)
    assert all(np.array_equal(state[name], adapter.values()[name]) for name in state)

    synthetic = np.zeros((2, 32, 32, 3), dtype=np.int8)
    first = adapter.predict_outputs(synthetic)
    second = adapter.predict_outputs(synthetic)
    assert np.array_equal(first, second)
    assert first.shape == (2, 10)
    labels = np.argmax(first, axis=1)
    clean = adapter.evaluate(synthetic, labels)
    protected, protected_report = adapter.evaluate_fault_schedule(
        synthetic, labels, schedule, full
    )
    assert protected == clean
    assert protected_report.suppressed_events == 2
    _faulted, faulted_report = adapter.evaluate_fault_schedule(
        synthetic, labels, schedule, no_protection
    )
    assert faulted_report.applied_events == 2
    assert np.array_equal(adapter.predict_outputs(synthetic), first)
