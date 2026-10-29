from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pytest

from netsurf.faults import FaultEvent, FaultModel, FaultSchedule
from netsurf.derived_analysis import derive_task_protection_analysis
from netsurf.empirical import CandidateSelection, measure_single_bits
from netsurf.estimators_m3 import (
    basic_estimator,
    first_order,
    numerical_bit_jump,
    per_sample_first_order_family,
)
from netsurf.legacy_identity import load_benchmark_identity
from netsurf.benchmarks.registry import benchmark_info
from netsurf.api import _write_derived_analysis
from netsurf.protection import ProtectionMask
from netsurf.parallel import ExecutionResources, LogicalTask, run_process_tasks
from netsurf.reporting import render_study_report
from netsurf.study_config import StudyConfigurationError, load_study


ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path("/projects/kona-assets/manuelbv/locks/mlperf-tiny")
MODELS = Path(
    "/data/models/models--mlcommons--tiny/snapshots/"
    "4addd0fa08d216e20637637874e084895f289da4"
)
KWS_MODEL = MODELS / "benchmark/training/keyword_spotting/trained_models/kws_ref_model.tflite"
AD_MODEL = MODELS / "benchmark/training/anomaly_detection/trained_models/ad01_int8.tflite"
KWS_DATA = Path("/data/datasets/speech_commands/0.0.3")
AD_DATA = Path("/data/datasets/dcase2020-toycar")


def _parallel_bounded_conformance(kind: str):
    if kind == "kws":
        from netsurf.adapters.mlperf_tiny_multibench import MlperfTinyKwsDscnnAdapter

        adapter = MlperfTinyKwsDscnnAdapter(
            KWS_MODEL, KWS_DATA, asset_lock=ASSETS / "kws.lock.json"
        )
    elif kind == "toycar":
        from netsurf.adapters.mlperf_tiny_multibench import MlperfTinyToyCarAdapter

        adapter = MlperfTinyToyCarAdapter(
            AD_MODEL, AD_DATA, asset_lock=ASSETS / "ad.lock.json"
        )
    else:
        raise ValueError(kind)
    adapter.load()
    return adapter.bounded_reference_diagnostics()


def _study_text(adapter: str, identity: str, checkpoint: Path, dataset: Path, lock: Path, split: str) -> str:
    return f'''\
schema_version = "3.4.0"
scientific_protocol = "single-bit-susceptibility/v1"
policies = []
[experiment]
name = "m93_fixture"
description = "bounded configuration fixture"
objective = "validate M9.3 adapter contract"
[benchmark]
adapter = "{adapter}"
id = "{identity}"
checkpoint = "{checkpoint}"
dataset = "{dataset}"
asset_lock = "{lock}"
inference_backend = "tensorflow_lite_builtin_ref_v1"
split = "{split}"
quantization = "tflite_affine_int8"
quantization_backend = "tflite_affine_integer_v1"
injectable_parameters = ["kernels"]
[samples]
ranking_count = 2
empirical_count = 2
evaluation_count = 2
require_disjoint = true
[candidates]
strategy = "uniform"
count = 2
seed = 17
score_quantiles = 1
[[estimators]]
name = "numerical_bit_jump"
implementation_version = "absolute-affine-xor-jump/v1"
[protection]
coverages = []
mitigation_model = "perfect_event_suppression"
[fault]
target = "weight"
location = "logical_trainable_parameter_bits"
persistence = "persistent_trial"
distribution = "independent_bernoulli"
probability = 0.0001
[execution]
trials = 1
confidence = 0.95
fault_seed = 19
permutation_seed = 23
[replications]
count = 1
candidate_seed_stride = 0
hutchinson_seed_stride = 0
sample_offset_stride = 0
fault_seed_stride = 0
estimator_seed_stride = 0
[output]
root = ""
report = true
plots = false
'''


def test_m93_identity_manifests_separate_modern_models_and_vww_blocker():
    kws = load_benchmark_identity("mlperf_tiny_kws_dscnn_int8_v1")
    ad = load_benchmark_identity("mlperf_tiny_ad_toycar_int8_v1")
    vww = load_benchmark_identity("mlperf_tiny_vww_mobilenetv1_int8_candidate_v1")
    assert kws["source"]["checkpoint_sha256"].startswith("aeea4368")
    assert kws["graph"]["kernel_bits"] == 176_128
    assert ad["graph"]["kernel_bits"] == 2_113_536
    assert ad["task"]["final_metric"] == "file-level ROC-AUC"
    assert "classification" not in ad["task"]["ranking_loss"]
    assert vww["status"] == "blocked_dataset_provenance"
    assert "pixel identity" in vww["blocker"]
    status, reasons = benchmark_info(
        "mlperf_tiny_vww_mobilenetv1_int8_candidate_v1"
    ).readiness()
    assert status == "unsupported configuration"
    assert "provenance" in reasons[0]


def test_task_metric_aggregation_uses_every_replication_and_exact_pairs(tmp_path):
    rows = []
    for replication, baseline in (("replication_0", 0.2), ("replication_1", 0.8)):
        for trial in (0, 1):
            schedule = f"{replication}-{trial}"
            for condition, benefit in (
                ("no_protection", 0.0),
                ("s5@0.125000", 0.1),
                ("s2@0.125000", 0.02),
            ):
                rows.append({
                    "replication": replication, "trial": trial,
                    "policy": f"ber=0.0001/{condition}",
                    "schedule_sha256": schedule,
                    "primary_metric_name": "roc_auc",
                    "primary_metric_value": baseline + benefit,
                    "primary_metric_direction": "maximize",
                    "loss_metric_name": "reconstruction_mse",
                    "loss_metric_value": 2.0 - baseline - benefit,
                    "loss_metric_direction": "minimize",
                    "proposed_events": 10, "applied_events": 8,
                    "suppressed_events": 2, "unique_applied_bits": 8,
                    "ranking_mask_sha256": condition,
                    "protected_bits": 0 if condition == "no_protection" else 10,
                    "realized_coverage": 0.0 if condition == "no_protection" else 0.125,
                })
    result = derive_task_protection_analysis(
        rows, expected_replications=2, permutation_seed=41,
    )
    control = next(value for value in result["pooled"] if value["policy"] == "no_protection")
    assert control["primary_metric"]["mean"] == pytest.approx(0.5)
    assert control["fault_realization_count"] == 4
    assert control["replication_count"] == 2
    comparison = next(
        value for value in result["paired"]
        if value["left"] == "s2" and value["right"] == "s5"
        and value["metric"] == "roc_auc"
    )
    assert comparison["matched_pairs"] == 4
    assert comparison["difference"]["mean"] == pytest.approx(-0.08)
    assert len(result["replications"]) == 6
    report = render_study_report(
        {
            "clean_evaluation": {
                "roc_auc": 0.75, "reconstruction_mse": 1.25,
                "sample_count": 4, "window_count": 784,
            },
            "eligible_bits": 2_113_536,
            "empirical_candidate_count": 0,
            "estimator_agreement": [],
            "protection_results": result["pooled"],
            "paired_comparisons": result["paired"],
            "replication_protection_results": result["replications"],
            "replication_count": 2,
            "replication_aggregate": [],
            "estimator_runtimes_seconds": {},
            "checks": {},
            "scientific_payload_sha256": "0" * 64,
        },
        {
            "scientific_protocol": "multi-coverage-protection/v1",
            "experiment": {"name": "task_report", "description": "", "objective": ""},
            "benchmark": {
                "id": "mlperf_tiny_ad_toycar_int8_v1",
                "adapter": "mlperf_tiny_ad_int8_v1",
                "quantization": "tflite_affine_int8",
            },
            "samples": {
                "ranking_count": 2, "empirical_count": 2,
                "evaluation_count": 2, "require_disjoint": True,
            },
            "fault": {"probability": 0.0001, "persistence": "persistent_trial"},
            "protection": {"probabilities": [0.0001]},
            "execution": {"confidence": 0.95},
        },
    )
    assert "Mean ROC-AUC" in report
    assert "Mean reconstruction MSE" in report
    assert "Replication-level task summaries" in report
    assert "categorical cross-entropy" not in report
    _write_derived_analysis(tmp_path, result)
    table = (tmp_path / "measurements/protection_results.csv").read_text()
    assert "primary_metric_name" in table
    assert "loss_metric_name" in table
    assert "accuracy_mean" not in table


def test_schema34_accepts_kws_and_toycar_without_mislabeling_metrics(tmp_path):
    kws_path = tmp_path / "kws.toml"
    kws_path.write_text(_study_text(
        "mlperf_tiny_kws_int8_v1", "mlperf_tiny_kws_dscnn_int8_v1",
        KWS_MODEL, KWS_DATA, ASSETS / "kws.lock.json", "official_subset_1000",
    ))
    kws = load_study(kws_path, check_inputs=False)
    assert kws.schema_version == "3.4.0"
    assert kws.benchmark.injectable_parameters == ("kernels",)

    ad_path = tmp_path / "ad.toml"
    ad_text = _study_text(
        "mlperf_tiny_ad_int8_v1", "mlperf_tiny_ad_toycar_int8_v1",
        AD_MODEL, AD_DATA, ASSETS / "ad.lock.json", "official_248_files",
    )
    ad_path.write_text(ad_text)
    ad = load_study(ad_path, check_inputs=False)
    assert ad.benchmark.adapter == "mlperf_tiny_ad_int8_v1"
    ad_path.write_text(ad_text.replace('schema_version = "3.4.0"', 'schema_version = "3.3.0"'))
    with pytest.raises(StudyConfigurationError, match="task-specific"):
        load_study(ad_path, check_inputs=False)
    ad_path.write_text(ad_text.replace('persistence = "persistent_trial"', 'persistence = "per_inference"'))
    with pytest.raises(StudyConfigurationError, match="persistent_trial"):
        load_study(ad_path, check_inputs=False)


@pytest.mark.skipif(
    not (KWS_MODEL.is_file() and KWS_DATA.is_dir() and (ASSETS / "kws.lock.json").is_file()),
    reason="pinned KWS assets are absent",
)
def test_kws_exact_kernel_faults_depthwise_mapping_and_restoration():
    from netsurf.adapters.mlperf_tiny_multibench import MlperfTinyKwsDscnnAdapter

    adapter = MlperfTinyKwsDscnnAdapter(
        KWS_MODEL, KWS_DATA, asset_lock=ASSETS / "kws.lock.json"
    )
    adapter.load()
    assert len(adapter.parameter_specs()) == 10
    assert len(adapter.bit_identities()) == 176_128
    assert any(spec.shape == (1, 3, 3, 64) for spec in adapter.parameter_specs())
    inputs, labels, partition = adapter.sample_partition("evaluation", 2)
    assert partition.indices == (0, 1)
    clean_codes = adapter.predict_encoded_outputs(inputs)
    bits = adapter.bit_identities()
    same_word = (bits[0], bits[1])
    model = FaultModel(
        "weight", "logical_trainable_parameter_bits", "persistent_trial",
        "independent_bernoulli", 0.0,
    )
    schedule = FaultSchedule(
        model, 29, tuple(FaultEvent(None, bit) for bit in same_word),
        len(bits), len(labels),
    )
    none = ProtectionMask("none", frozenset(), len(bits), 0.0, {})
    full = ProtectionMask("full", frozenset(bits), len(bits), 1.0, {})
    _faulted, report = adapter.evaluate_fault_schedule(inputs, labels, schedule, none)
    assert report.proposed_events == 2 and report.applied_events == 2
    patched_content, _ = adapter.storage.apply_schedule(schedule, none)
    patched = adapter.storage.values(patched_content)
    original = adapter.values()[same_word[0].parameter].reshape(-1)[same_word[0].flat_index]
    changed = patched[same_word[0].parameter].reshape(-1)[same_word[0].flat_index]
    assert (int(changed) & 0xFF) == ((int(original) & 0xFF) ^ 0b11)
    protected, report = adapter.evaluate_fault_schedule(inputs, labels, schedule, full)
    assert report.suppressed_events == 2 and report.applied_events == 0
    assert protected == adapter.evaluate(inputs, labels)
    assert np.array_equal(adapter.predict_encoded_outputs(inputs), clean_codes)
    gradients = adapter.gradients(inputs[:1], labels[:1])
    assert set(gradients) == {spec.name for spec in adapter.parameter_specs()}
    assert all(np.all(np.isfinite(value)) for value in gradients.values())
    correspondence = adapter.companion_correspondence(inputs, labels)
    assert correspondence["encoded_max_error"] <= 2
    assert correspondence["class_agreement"] == 1.0
    assert correspondence["companion_is_authoritative"] is False
    selected = tuple(bits[index] for index in range(0, 64, 4))
    outputs = (
        basic_estimator("msb_first", selected),
        basic_estimator("random", selected, seed=11),
        numerical_bit_jump(adapter, selected),
        first_order(adapter, inputs[:1], labels[:1], selected, None),
        *per_sample_first_order_family(
            adapter, inputs, labels, selected, None,
            requested=("per_sample_first_order_magnitude", "per_sample_positive_damage"),
        ).values(),
    )
    assert all(set(output.scores) == set(selected) for output in outputs)
    assert all(np.all(np.isfinite(tuple(output.scores.values()))) for output in outputs)
    empirical = measure_single_bits(
        adapter, inputs, labels, (CandidateSelection(bits[9], "test", 1.0),), None,
        evaluation_sample_sha256=partition.sample_sha256,
    )
    assert len(empirical) == 1 and np.isfinite(empirical[0].signed_loss_difference)


@pytest.mark.skipif(
    not (
        AD_MODEL.is_file() and AD_DATA.is_dir() and (ASSETS / "ad.lock.json").is_file()
        and find_spec("librosa") is not None
    ),
    reason="pinned ToyCar assets or librosa are absent",
)
def test_toycar_task_metrics_companion_and_kernel_restoration():
    from netsurf.adapters.mlperf_tiny_multibench import MlperfTinyToyCarAdapter

    adapter = MlperfTinyToyCarAdapter(
        AD_MODEL, AD_DATA, asset_lock=ASSETS / "ad.lock.json"
    )
    adapter.load()
    assert len(adapter.parameter_specs()) == 10
    assert len(adapter.bit_identities()) == 2_113_536
    inputs, labels, partition = adapter.sample_partition("evaluation", 2)
    empirical_inputs, empirical_labels, empirical_partition = adapter.sample_partition(
        "empirical_validation", 2
    )
    assert set(partition.indices).isdisjoint(empirical_partition.indices)
    assert partition.source_split == empirical_partition.source_split
    evaluation = adapter.evaluate(inputs, labels)
    assert evaluation.primary_metric_name == "roc_auc"
    assert evaluation.loss_metric_name == "reconstruction_mse"
    assert adapter.evaluate(empirical_inputs, empirical_labels).primary_metric_name == "roc_auc"
    assert 0.0 <= evaluation.roc_auc <= 1.0
    manifest = adapter.task_metric_manifest()
    assert manifest["ranking_loss"]["name"] == "reconstruction_mse"
    assert manifest["primary_metric"]["name"] == "roc_auc"
    correspondence = adapter.companion_correspondence(inputs[:2], labels[:2])
    assert correspondence["encoded_max_error"] <= 8
    assert correspondence["reconstruction_mse_absolute_difference"] < 1.0
    assert correspondence["companion_is_authoritative"] is False
    gradients = adapter.gradients(inputs[:2], labels[:2])
    assert all(np.all(np.isfinite(value)) for value in gradients.values())
    bits = adapter.bit_identities()
    selected = tuple(bits[index] for index in range(0, 64, 4))
    outputs = (
        basic_estimator("msb_first", selected),
        basic_estimator("random", selected, seed=13),
        numerical_bit_jump(adapter, selected),
        first_order(adapter, inputs[:2], labels[:2], selected, None),
        *per_sample_first_order_family(
            adapter, inputs[:2], labels[:2], selected, None,
            requested=("per_sample_first_order_magnitude", "per_sample_positive_damage"),
        ).values(),
    )
    assert all(set(output.scores) == set(selected) for output in outputs)
    assert all(np.all(np.isfinite(tuple(output.scores.values()))) for output in outputs)
    clean = adapter.predict_encoded_outputs(inputs[:2])
    bit = bits[7]
    model = FaultModel(
        "weight", "logical_trainable_parameter_bits", "persistent_trial",
        "independent_bernoulli", 0.0,
    )
    schedule = FaultSchedule(model, 31, (FaultEvent(None, bit),), len(adapter.bit_identities()), 2)
    none = ProtectionMask("none", frozenset(), len(adapter.bit_identities()), 0.0, {})
    adapter.evaluate_fault_schedule(inputs, labels, schedule, none)
    assert np.array_equal(adapter.predict_encoded_outputs(inputs[:2]), clean)
    empirical = measure_single_bits(
        adapter, inputs, labels, (CandidateSelection(bits[11], "test", 1.0),), None,
        evaluation_sample_sha256=partition.sample_sha256,
    )
    assert empirical[0].primary_metric_name == "roc_auc"
    assert empirical[0].loss_metric_name == "reconstruction_mse"


@pytest.mark.skipif(
    not (
        KWS_MODEL.is_file() and KWS_DATA.is_dir()
        and AD_MODEL.is_file() and AD_DATA.is_dir()
        and (ASSETS / "kws.lock.json").is_file()
        and (ASSETS / "ad.lock.json").is_file()
        and find_spec("librosa") is not None
        and find_spec("threadpoolctl") is not None
    ),
    reason="pinned KWS/ToyCar assets or preprocessing dependencies are absent",
)
def test_parallel_kws_and_toycar_bounded_conformance_is_bitwise_deterministic():
    tasks = tuple(
        LogicalTask(f"{kind}:{index}", kind)
        for kind in ("kws", "toycar")
        for index in range(2)
    )
    result = run_process_tasks(
        tasks, _parallel_bounded_conformance, ExecutionResources(2, 1)
    )
    for kind, offset, match_key in (
        ("kws", 0, "exact_output_digest_match"),
        ("toycar", 2, "exact_score_digest_match"),
    ):
        left, right = result.values[offset : offset + 2]
        assert left[match_key] is True and right[match_key] is True
        assert left["sample_ids"] == right["sample_ids"]
        assert left["sample_sha256"] == right["sample_sha256"]
        assert left["preprocessed_int8_sha256"] == right["preprocessed_int8_sha256"]
        assert left["exact_tflite_output_sha256"] == right["exact_tflite_output_sha256"]
        assert left["runtime"]["pid"] != right["runtime"]["pid"]
        if kind == "toycar":
            assert left["file_scores_sha256"] == right["file_scores_sha256"]
            assert left["file_score_float64_hex"] == right["file_score_float64_hex"]
            assert left["first_differing_value"] is None
            assert left["preprocessing_blas_threads"] == 2
