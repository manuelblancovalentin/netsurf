from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from netsurf.adapters.mnist_quantized import MnistQuantizedAdapter
from netsurf.faults import FaultEvent, FaultModel, FaultSchedule, apply_persistent_schedule
from netsurf.curvature import DenseReluDenseCurvature
from netsurf.factorized import FactorizedInteractionAggregate
from netsurf.identities import BitIdentity
from netsurf.incremental import DenseReluDenseIncrementalEvaluator, FaultState
from netsurf.protection import ProtectionMask
from netsurf.pruning import (
    grouped_magnitude_importance,
    grouped_taylor_importance,
    netsurf_reliability_aware_importance,
    paper_width_schedule,
    remove_hidden_neurons,
    select_pruned_neurons,
    topology,
    validate_target_widths,
)
from netsurf.pruning_config import (
    FineTuning, PruningCalibration, PruningConfiguration, PruningConfigurationError,
    PruningOutput, PruningPlan, PruningSource, ReliabilityComposition, load_pruning,
)
from netsurf.pruning_pipeline import (
    _fine_tune, _independent_values, _parameter_identity, _save_checkpoint,
    inspect_pruned_checkpoint, prepare_pruned_checkpoint,
)
from netsurf.quantization import BitCodec, FixedPointFormat
from netsurf.reliability import evaluate_persistent_fault_state
from netsurf.topology import DenseReluDenseTopologyEvaluator


def small_values(hidden=3):
    return {
        "dense0/kernel": np.arange(2 * hidden, dtype=np.float32).reshape(2, hidden) / 8,
        "dense0/bias": np.arange(hidden, dtype=np.float32) / -8,
        "dense1/kernel": np.arange(hidden * 2, dtype=np.float32).reshape(hidden, 2) / 8,
        "dense1/bias": np.asarray([-0.25, 0.125], dtype=np.float32),
    }


def test_structural_pruning_physically_removes_complete_dependency_group():
    values = small_values()
    pruned = remove_hidden_neurons(values, (1,))
    assert pruned["dense0/kernel"].shape == (2, 2)
    assert pruned["dense0/bias"].shape == (2,)
    assert pruned["dense1/kernel"].shape == (2, 2)
    np.testing.assert_array_equal(pruned["dense0/kernel"], values["dense0/kernel"][:, [0, 2]])
    np.testing.assert_array_equal(pruned["dense1/kernel"], values["dense1/kernel"][[0, 2], :])
    metrics = topology(pruned, 6)
    assert metrics.parameter_count == 12
    assert metrics.encoded_bit_count == 72
    assert metrics.macs_per_inference == 8
    assert metrics.encoded_memory_bytes == 9


def test_group_importance_and_selection_are_deterministic_and_include_dependencies():
    values = small_values()
    gradients = {name: np.ones_like(value) for name, value in values.items()}
    taylor = grouped_taylor_importance(values, gradients)
    magnitude = grouped_magnitude_importance(values, norm="l1")
    np.testing.assert_allclose(taylor, magnitude)
    assert select_pruned_neurons([1.0, 1.0, 2.0], 2) == (0, 1)
    assert paper_width_schedule(100, iterations=10)[-1] == 20
    assert validate_target_widths(32, [24, 16, 8]) == (24, 16, 8)


def test_reliability_aware_components_are_independently_ablatable():
    clean = [1.0, 2.0, 3.0]
    susceptible = [3.0, 2.0, 1.0]
    cost = [1.0, 2.0, 3.0]
    clean_only = netsurf_reliability_aware_importance(clean, susceptible, cost, susceptibility_weight=0, cost_weight=0)
    susceptibility_only = netsurf_reliability_aware_importance(clean, susceptible, cost, clean_weight=0, cost_weight=0)
    assert clean_only.combined_prune_cost == clean_only.clean_preservation
    np.testing.assert_allclose(susceptibility_only.combined_prune_cost, -np.asarray(susceptibility_only.bit_susceptibility))
    assert clean_only.definition.startswith("netsurf-")
    first = _independent_values(4, 123)
    second = _independent_values(4, 123)
    assert all(np.array_equal(first[name], second[name]) for name in first)
    assert first["dense0/kernel"].shape == (784, 4)
    assert first["dense1/kernel"].shape == (4, 10)


def test_pruned_checkpoint_round_trip_bit_indexing_fault_mask_and_incremental_response(tmp_path):
    codec = BitCodec(FixedPointFormat.parse("q<6,2,1>"))
    rng = np.random.default_rng(91)
    values = {
        "dense0/kernel": codec.quantize(rng.normal(0, 0.2, (784, 2))).astype(np.float32),
        "dense0/bias": np.asarray([-0.25, 0.125], dtype=np.float32),
        "dense1/kernel": codec.quantize(rng.normal(0, 0.2, (2, 10))).astype(np.float32),
        "dense1/bias": np.zeros(10, dtype=np.float32),
    }
    shape = topology(values, 6)
    metadata = {
        "schema_version": "2.0.0", "checkpoint_kind": "physically_structured_pruned",
        "method_classification": "test fixture", "quantization": codec.format.to_dict(),
        "topology": shape.to_dict(), "parameter_identity": _parameter_identity(values, codec),
    }
    checkpoint = _save_checkpoint(tmp_path / "pruned.npz", values, metadata)
    report = inspect_pruned_checkpoint(checkpoint)
    assert report["topology"]["hidden_width"] == 2
    assert report["provenance_valid"]

    adapter = MnistQuantizedAdapter(checkpoint, tmp_path / "unused.npz", codec)
    adapter.load()
    assert len(adapter.bit_identities()) == shape.encoded_bit_count
    assert max(bit.flat_index for bit in adapter.bit_identities() if bit.parameter == "dense0/bias") == 1

    bit = BitIdentity("dense0/bias", 0, 5)
    model = FaultModel("weight", "logical_trainable_parameter_bits", "persistent_trial", "independent_bernoulli", 1.0)
    schedule = FaultSchedule(model, 1, (FaultEvent(None, bit),), shape.encoded_bit_count, 1)
    no_mask = ProtectionMask("none", frozenset(), shape.encoded_bit_count, 0, {})
    corrupted, accounting = apply_persistent_schedule(adapter.values(), schedule, no_mask, codec)
    assert corrupted["dense0/bias"][0] == pytest.approx(3.75)
    assert accounting.applied_events == 1
    protected = ProtectionMask("protected", frozenset((bit,)), shape.encoded_bit_count, 1, {})
    restored, accounting = apply_persistent_schedule(adapter.values(), schedule, protected, codec)
    np.testing.assert_array_equal(restored["dense0/bias"], adapter.values()["dense0/bias"])
    assert accounting.suppressed_events == 1

    inputs = rng.random((3, 28, 28, 1), dtype=np.float32)
    evaluator = DenseReluDenseIncrementalEvaluator(adapter, codec)
    state = FaultState.from_bits((bit,))
    incremental = evaluator.incremental_forward(evaluator.cache(inputs), inputs, state)
    full = evaluator.full_forward(inputs, state)
    np.testing.assert_allclose(incremental.logits, full.logits, rtol=1e-6, atol=1e-6)
    measurement = evaluate_persistent_fault_state(
        adapter, codec, inputs, np.asarray([0, 1, 2]), state
    )
    assert measurement.hidden_width == 2
    assert measurement.eligible_bits == shape.encoded_bit_count
    assert measurement.logical_ber_realization == pytest.approx(1 / shape.encoded_bit_count)
    assert 0.0 <= measurement.prediction_change_probability <= 1.0
    topology_evaluator = DenseReluDenseTopologyEvaluator(
        evaluator, inputs, FaultState.from_bits()
    )
    curvature = DenseReluDenseCurvature(topology_evaluator)
    universe = tuple(adapter.bit_identities())
    aggregate = FactorizedInteractionAggregate(
        curvature, universe, (), probabilities=0.001, population=universe[:48]
    )
    assert aggregate.query(universe[0]) == pytest.approx(aggregate.explicit(universe[0]), abs=1e-12)
    assert aggregate.population_coverage == pytest.approx(48 / len(universe))


def test_pruned_checkpoint_serialization_preserves_clean_inference(tmp_path):
    codec = BitCodec(FixedPointFormat.parse("q<6,2,1>"))
    rng = np.random.default_rng(7)
    values = {
        "dense0/kernel": codec.quantize(rng.normal(0, 0.1, (784, 2))).astype(np.float32),
        "dense0/bias": np.zeros(2, dtype=np.float32),
        "dense1/kernel": codec.quantize(rng.normal(0, 0.1, (2, 10))).astype(np.float32),
        "dense1/bias": np.zeros(10, dtype=np.float32),
    }
    inputs, labels = rng.random((4, 28, 28, 1), dtype=np.float32), np.asarray([0, 1, 2, 3])
    before = MnistQuantizedAdapter(tmp_path / "unused", tmp_path / "unused", codec)
    before._values = {name: value.copy() for name, value in values.items()}
    expected = before.evaluate(inputs, labels)
    shape = topology(values, 6)
    metadata = {
        "schema_version": "2.0.0", "checkpoint_kind": "physically_structured_pruned",
        "method_classification": "test fixture", "quantization": codec.format.to_dict(),
        "topology": shape.to_dict(), "parameter_identity": _parameter_identity(values, codec),
    }
    checkpoint = _save_checkpoint(tmp_path / "roundtrip.npz", values, metadata)
    after = MnistQuantizedAdapter(checkpoint, tmp_path / "unused", codec)
    after.load()
    actual = after.evaluate(inputs, labels)
    assert actual == expected


def test_schema5_configuration_validation_and_workspace_boundary(tmp_path, monkeypatch):
    import netsurf.pruning_config as module
    from netsurf import __main__ as cli
    from netsurf.experiment_config import load_schema

    monkeypatch.setattr(module, "workspace_root", lambda: tmp_path / "workspace")
    checkpoint, dataset = tmp_path / "source.npz", tmp_path / "mnist.npz"
    np.savez(checkpoint, **{"dense0/bias": np.zeros(32, dtype=np.float32)})
    dataset.write_bytes(b"dataset")
    output = tmp_path / "workspace" / "benchmarks" / "pruned.npz"
    config = tmp_path / "pruning.toml"
    config.write_text(f'''\
schema_version = "5.0.0"
scientific_protocol = "structured-hidden-neuron-pruning/v1"
[experiment]
name = "bounded_pruning"
description = "test"
[source]
adapter = "mnist_quantized"
checkpoint = "{checkpoint}"
dataset = "{dataset}"
quantization = "q<6,2,1>"
[calibration]
sample_count = 8
sample_offset = 0
[pruning]
method = "grouped_taylor"
schedule = "target_widths"
iterations = 1
step_fraction = 0.149
target_widths = [16]
magnitude_norm = "l1"
[reliability]
susceptibility_estimator = "first_order_gradient_delta"
susceptibility_reduction = "mean"
clean_weight = 1.0
susceptibility_weight = 1.0
cost_weight = 0.0
[fine_tuning]
enabled = false
mode = "quantization_aware"
epochs_per_stage = 0
batch_size = 32
learning_rate = 0.001
training_samples = 128
clean_loss_weight = 1.0
fault_loss_weight = 0.0
fault_probability = 0.001
seed = 17
[output]
checkpoint = "{output}"
''', encoding="utf-8")
    loaded = load_pruning(config)
    assert loaded.pruning.target_widths == (16,)
    assert loaded.output.checkpoint == output
    assert load_schema("pruning", "v5")["properties"]["schema_version"]["const"] == "5.0.0"
    args = cli._parser().parse_args(["pruning", "validate", str(config)])
    assert args.pruning_command == "validate"
    outside = config.read_text().replace(str(output), str(tmp_path / "outside.npz"))
    config.write_text(outside, encoding="utf-8")
    with pytest.raises(PruningConfigurationError, match="workspace"):
        load_pruning(config)


def test_bounded_pipeline_records_original_indices_and_provenance(tmp_path, monkeypatch):
    import netsurf.pruning_pipeline as module

    source, dataset = tmp_path / "source.npz", tmp_path / "dataset.npz"
    source.write_bytes(b"immutable-source")
    dataset.write_bytes(b"immutable-dataset")

    class FakeAdapter:
        def __init__(self, checkpoint, dataset_path, codec):
            self.codec = codec
            self._values = small_values()
            self.metadata = {"source": "fake-bounded-adapter"}

        def load(self):
            pass

        def values(self):
            return {name: value.copy() for name, value in self._values.items()}

        def sample_partition(self, role, count, offset=0):
            from netsurf.samples import SamplePartition
            return (
                np.zeros((count, 2), dtype=np.float32), np.zeros(count, dtype=np.int64),
                SamplePartition(role, "bounded", tuple(range(count)), "sample-hash", "bounded test"),
            )

        @contextmanager
        def temporary_values(self, values):
            before = self._values
            self._values = {name: value.copy() for name, value in values.items()}
            try:
                yield
            finally:
                self._values = before

        def gradients(self, inputs, labels):
            return {name: np.ones_like(value) for name, value in self._values.items()}

    monkeypatch.setattr(module, "MnistQuantizedAdapter", FakeAdapter)
    output = tmp_path / "workspace" / "pruned.npz"
    config = PruningConfiguration(
        "5.0.0", "structured-hidden-neuron-pruning/v1", "bounded", "bounded test",
        PruningSource("mnist_quantized", source, dataset, "q<6,2,1>"),
        PruningCalibration(2, 0),
        PruningPlan("grouped_taylor", "target_widths", 1, 0.149, (2,), "l1"),
        ReliabilityComposition("first_order_gradient_delta", "mean", 1.0, 1.0, 0.0),
        FineTuning(False, "quantization_aware", 0, 2, 0.001, 2, 1.0, 0.0, 0.001, 11),
        PruningOutput(output), tmp_path / "config.toml", "bounded-config",
    )
    assert prepare_pruned_checkpoint(config) == output
    with np.load(output, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
    stage = metadata["pruning_stages"][0]
    assert stage["removed_original_source_indices"] == stage["removed_stage_indices"]
    assert len(stage["retained_original_source_indices"]) == 2
    assert metadata["source_checkpoint"]["sha256"]
    assert metadata["scientific_payload_sha256"]
    assert metadata["topology"]["hidden_width"] == 2


@pytest.mark.parametrize(
    ("mode", "fault_weight"),
    (("quantization_aware", 0.0), ("netsurf_fault_augmented", 0.5)),
)
def test_bounded_fine_tuning_modes_finish_in_exact_six_bit_state(tmp_path, mode, fault_weight):
    pytest.importorskip("tensorflow")
    rng = np.random.default_rng(33)
    dataset = tmp_path / "mnist.npz"
    np.savez(
        dataset,
        x_train=rng.integers(0, 256, (4, 28, 28), dtype=np.uint8),
        y_train=np.asarray([0, 1, 2, 3], dtype=np.uint8),
    )
    codec = BitCodec(FixedPointFormat.parse("q<6,2,1>"))
    values = {
        "dense0/kernel": codec.quantize(rng.normal(0, 0.1, (784, 2))).astype(np.float32),
        "dense0/bias": np.zeros(2, dtype=np.float32),
        "dense1/kernel": codec.quantize(rng.normal(0, 0.1, (2, 10))).astype(np.float32),
        "dense1/bias": np.zeros(10, dtype=np.float32),
    }
    config = PruningConfiguration(
        "5.0.0", "structured-hidden-neuron-pruning/v1", "fine_tune", "bounded test",
        PruningSource("mnist_quantized", tmp_path / "source.npz", dataset, "q<6,2,1>"),
        PruningCalibration(2, 0),
        PruningPlan("grouped_taylor", "target_widths", 1, 0.149, (1,), "l1"),
        ReliabilityComposition("first_order_gradient_delta", "mean", 1.0, 1.0, 0.0),
        FineTuning(True, mode, 1, 2, 0.001, 4, 1.0, fault_weight, 0.01, 19),
        PruningOutput(tmp_path / "output.npz"), tmp_path / "config.toml", "bounded-config",
    )
    trained, history = _fine_tune(values, dataset, config, codec, stage=1)
    assert len(history) == 1
    for value in trained.values():
        np.testing.assert_array_equal(
            codec.decode(codec.encode(value)).astype(np.float32), value
        )
