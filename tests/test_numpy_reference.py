from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile

import numpy as np

from netsurf.adapters.numpy_reference import NumpyReferenceAdapter
from netsurf.benchmarks import benchmark_info, benchmark_list
from netsurf.doctor import doctor_report
from netsurf.experiment_config import load_experiment
from netsurf.experiment_engine import execute_experiment
from netsurf.faults import FaultEvent, FaultModel, FaultSchedule, apply_persistent_schedule
from netsurf.protection import ProtectionMask
from netsurf.quantization import BitCodec, FixedPointFormat


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/experiments/numpy_reference.toml"


def _adapter():
    configuration = load_experiment(CONFIG)
    codec = BitCodec(FixedPointFormat.parse(configuration.benchmark.quantization))
    adapter = NumpyReferenceAdapter(configuration.benchmark.checkpoint, configuration.benchmark.dataset, codec)
    adapter.load()
    return configuration, codec, adapter


def test_benchmark_registry_and_doctor_are_explicit():
    names = {item.name for item in benchmark_list()}
    assert {
        "numpy_reference",
        "mnist_quantized",
        "legacy_mnist_lite",
        "mnist_lite_exact_kb_v1",
    } <= names
    exact = benchmark_info("mnist_lite_exact_kb_v1").to_dict()
    assert exact["parameter_count"] == 7_980
    assert exact["eligible_parameter_count"] == 7_960
    assert exact["eligible_bit_count"] == 47_760
    descriptor = benchmark_info("numpy_reference")
    assert descriptor.parameter_count == 17
    assert descriptor.supported_fault_targets == ("weight",)
    report = doctor_report()
    assert report["packages"]["numpy"] is not None
    assert report["workspace"]["writable"] is True


def test_quantized_clean_inference_and_exact_bit_population():
    _, codec, adapter = _adapter()
    inputs, labels = adapter.dataset(8)
    result = adapter.evaluate(inputs, labels)
    assert result.accuracy == 1.0
    assert len(adapter.bit_identities()) == 17 * codec.format.total_bits
    for value in adapter.values().values():
        assert np.array_equal(codec.decode(codec.encode(value)), value)


def test_exhaustive_single_bit_injection_restores_state():
    _, _, adapter = _adapter()
    inputs, labels = adapter.dataset(8)
    before = adapter.values()
    impacts = adapter.exhaustive_single_bit_impacts(inputs, labels)
    assert len(impacts) == 102
    assert any(abs(value) > 0.0 for value in impacts.values())
    assert all(np.array_equal(before[name], adapter.values()[name]) for name in before)


def test_multiple_bits_use_xor_on_one_encoded_word():
    _, codec, adapter = _adapter()
    values = adapter.values()
    identity0, identity1 = adapter.bit_identities()[0], adapter.bit_identities()[1]
    model = FaultModel("weight", "logical_trainable_parameter_bits", "persistent_trial", "independent_bernoulli", 0.0)
    schedule = FaultSchedule(model, 1, (FaultEvent(None, identity0), FaultEvent(None, identity1)), len(adapter.bit_identities()), 1)
    mask = ProtectionMask("none", frozenset(), len(adapter.bit_identities()), 0.0, {})
    corrupted, report = apply_persistent_schedule(values, schedule, mask, codec)
    original_code = codec.encode(values[identity0.parameter].reshape(-1)[0])
    expected = codec.decode(np.bitwise_xor(original_code, np.uint64(0b11)))
    assert corrupted[identity0.parameter].reshape(-1)[0] == expected
    assert report.applied_events == 2
    assert report.unique_applied_bits == 2


def test_analytic_gradient_matches_finite_difference_away_from_relu_boundary():
    _, _, adapter = _adapter()
    inputs, labels = adapter.dataset(8, role="ranking")
    analytic = adapter.gradients(inputs, labels)["dense1/bias"][0]
    epsilon = 1e-5
    values = adapter.values()
    plus, minus = adapter.values(), adapter.values()
    plus["dense1/bias"][0] += epsilon
    minus["dense1/bias"][0] -= epsilon
    with adapter.temporary_values(plus):
        high = adapter.evaluate(inputs, labels).categorical_crossentropy
    with adapter.temporary_values(minus):
        low = adapter.evaluate(inputs, labels).categorical_crossentropy
    numerical = (high - low) / (2 * epsilon)
    assert np.isclose(analytic, numerical, rtol=1e-5, atol=1e-7)
    assert all(np.array_equal(values[name], adapter.values()[name]) for name in values)


def test_modern_engine_controls_pairing_and_reproducibility():
    configuration, codec, _ = _adapter()
    first = execute_experiment(configuration)
    second = execute_experiment(configuration)
    assert first.scientific_payload_sha256 == second.scientific_payload_sha256
    assert {row.policy for row in first.trials} == {
        "no_protection", "msb_first", "random", "first_order_gradient_delta", "full_protection"
    }
    assert first.scientific_payload["checks"]["matched_schedules"] is True
    assert first.scientific_payload["policies"]["full_protection"]["mean_applied_events"] == 0.0
    assert first.scientific_payload["policies"]["no_protection"]["mean_suppressed_events"] == 0.0


def test_per_inference_schedule_is_replayable_and_batch_independent():
    configuration = load_experiment(CONFIG)
    per_inference = replace(configuration, fault=replace(configuration.fault, persistence="per_inference"))
    first = execute_experiment(per_inference)
    second = execute_experiment(per_inference)
    assert first.scientific_payload_sha256 == second.scientific_payload_sha256
    assert all(event[0] is not None for schedule in first.fault_schedules for event in schedule["events"])
