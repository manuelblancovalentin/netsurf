"""Bounded, configuration-driven preparation of physically pruned checkpoints."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping

import numpy as np

from .adapters.mnist_quantized import MnistQuantizedAdapter, PARAMETER_NAMES, _import_ml_runtime, _normalize
from .identities import ParameterSpec, enumerate_bits
from .manifests import digest_object, sha256_file
from .pruning import (
    exact_quantization,
    grouped_bit_susceptibility,
    grouped_magnitude_importance,
    grouped_taylor_importance,
    netsurf_reliability_aware_importance,
    paper_width_schedule,
    remove_hidden_neurons,
    select_pruned_neurons,
    topology,
    validate_dense_relu_dense,
    validate_target_widths,
)
from .pruning_config import PruningConfiguration, load_pruning
from .quantization import BitCodec, FixedPointFormat
from .ranking import FirstOrderEstimator


def _parameter_identity(values: Mapping[str, np.ndarray], codec: BitCodec) -> list[dict[str, object]]:
    return [
        {
            "name": name,
            "shape": list(values[name].shape),
            "encoded_sha256": digest_object(codec.encode(values[name]).tolist()),
        }
        for name in PARAMETER_NAMES
    ]


def _first_order_scores(values, gradients, codec):
    specs = tuple(ParameterSpec(name, tuple(np.asarray(values[name]).shape)) for name in PARAMETER_NAMES)
    identities = enumerate_bits(specs, codec.format.total_bits)
    deltas = {name: codec.perturbation_matrix(value) for name, value in values.items()}
    return FirstOrderEstimator().estimate(identities, gradients, deltas).scores


def _fault_delta(values, codec: BitCodec, probability: float, generator) -> dict[str, np.ndarray]:
    result = {}
    for name, value in values.items():
        encoded = codec.encode(value)
        mask = np.zeros(encoded.shape, dtype=np.uint64)
        for bit in range(codec.format.total_bits):
            selected = generator.random(encoded.shape) < probability
            mask ^= selected.astype(np.uint64) << np.uint64(bit)
        result[name] = (codec.decode(encoded ^ mask) - codec.decode(encoded)).astype(np.float32)
    return result


def _fine_tune(values, dataset: Path, configuration: PruningConfiguration, codec: BitCodec, *, stage: int):
    tuning = configuration.fine_tuning
    if not tuning.enabled or tuning.epochs_per_stage == 0:
        return exact_quantization(values, codec), []
    _, tf = _import_ml_runtime()
    with np.load(dataset) as data:
        inputs = _normalize(data["x_train"][:tuning.training_samples])
        labels = data["y_train"][:tuning.training_samples].astype(np.int64)
    variables = {name: tf.Variable(np.asarray(values[name], dtype=np.float32), name=name.replace("/", "_")) for name in PARAMETER_NAMES}
    optimizer = tf.keras.optimizers.Adam(learning_rate=tuning.learning_rate)
    rng = np.random.default_rng(tuning.seed + stage)
    low, high, step = codec.format.value_min, codec.format.value_max, codec.format.step

    def ste_quantize(value):
        quantized = tf.clip_by_value(tf.round(value / step) * step, low, high)
        return value + tf.stop_gradient(quantized - value)

    def logits(batch, effective):
        flat = tf.reshape(batch, (tf.shape(batch)[0], 784))
        hidden = tf.nn.relu(tf.linalg.matmul(flat, effective["dense0/kernel"]) + effective["dense0/bias"])
        return tf.linalg.matmul(hidden, effective["dense1/kernel"]) + effective["dense1/bias"]

    history = []
    for epoch in range(tuning.epochs_per_stage):
        order = rng.permutation(len(labels))
        losses = []
        for start in range(0, len(order), tuning.batch_size):
            selected = order[start:start + tuning.batch_size]
            batch_x = tf.convert_to_tensor(inputs[selected], tf.float32)
            batch_y = tf.convert_to_tensor(labels[selected], tf.int64)
            with tf.GradientTape() as tape:
                clean_values = variables if tuning.mode == "standard" else {
                    name: ste_quantize(value) for name, value in variables.items()
                }
                clean_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=batch_y, logits=logits(batch_x, clean_values)
                ))
                loss = tuning.clean_loss_weight * clean_loss
                if tuning.mode == "netsurf_fault_augmented":
                    quantized_numpy = {
                        name: codec.quantize(value.numpy()).astype(np.float32)
                        for name, value in variables.items()
                    }
                    deltas = _fault_delta(quantized_numpy, codec, tuning.fault_probability, rng)
                    faulted_values = {
                        name: clean_values[name] + tf.constant(deltas[name], tf.float32)
                        for name in PARAMETER_NAMES
                    }
                    fault_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=batch_y, logits=logits(batch_x, faulted_values)
                    ))
                    loss = loss + tuning.fault_loss_weight * fault_loss
            gradients = tape.gradient(loss, tuple(variables.values()))
            optimizer.apply_gradients(zip(gradients, variables.values()))
            losses.append(float(loss.numpy()))
        history.append({"epoch": epoch + 1, "mean_training_objective": float(np.mean(losses))})
    trained = {name: value.numpy().copy() for name, value in variables.items()}
    return exact_quantization(trained, codec), history


def _independent_values(width: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    limit0 = np.sqrt(6.0 / (784 + width))
    limit1 = np.sqrt(6.0 / (width + 10))
    return {
        "dense0/kernel": rng.uniform(-limit0, limit0, (784, width)).astype(np.float32),
        "dense0/bias": np.zeros(width, dtype=np.float32),
        "dense1/kernel": rng.uniform(-limit1, limit1, (width, 10)).astype(np.float32),
        "dense1/bias": np.zeros(10, dtype=np.float32),
    }


def _save_checkpoint(path: Path, values, metadata) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"checkpoint already exists and will not be overwritten: {path}")
    with tempfile.NamedTemporaryFile(prefix="pruned-checkpoint-", suffix=".npz", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        np.savez_compressed(stream, **values, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
        stream.flush()
        os.fsync(stream.fileno())
    try:
        temporary.replace(path)
        path.chmod(0o444)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def prepare_pruned_checkpoint(configuration: PruningConfiguration | str | Path) -> Path:
    config = load_pruning(configuration) if isinstance(configuration, (str, Path)) else configuration
    codec = BitCodec(FixedPointFormat.parse(config.source.quantization))
    adapter = MnistQuantizedAdapter(config.source.checkpoint, config.source.dataset, codec)
    adapter.load()
    source_values = adapter.values()
    source_shape = validate_dense_relu_dense(source_values)
    calibration_x, calibration_y, calibration_partition = adapter.sample_partition(
        "ranking", config.calibration.sample_count, offset=config.calibration.sample_offset
    )
    if config.pruning.schedule == "paper_fraction":
        targets = paper_width_schedule(
            source_shape.hidden_width,
            iterations=config.pruning.iterations,
            fraction=config.pruning.step_fraction,
        )
    else:
        targets = validate_target_widths(source_shape.hidden_width, config.pruning.target_widths)

    stages = []
    if config.pruning.method == "independent_width":
        values = _independent_values(targets[0], config.fine_tuning.seed)
        values, history = _fine_tune(values, config.source.dataset, config, codec, stage=0)
        stages.append({
            "stage": 1, "source_width": source_shape.hidden_width, "target_width": targets[0],
            "removed_source_indices": [], "importance": "not_applicable_independent_training",
            "fine_tuning": history,
        })
        checkpoint_kind = "independently_trained_width_control"
    else:
        values = source_values
        checkpoint_kind = "physically_structured_pruned"
        active_source_indices = list(range(source_shape.hidden_width))
        for stage, target in enumerate(targets, start=1):
            with adapter.temporary_values(values):
                gradients = adapter.gradients(calibration_x, calibration_y)
            if config.pruning.method == "group_magnitude":
                scores = grouped_magnitude_importance(values, norm=config.pruning.magnitude_norm)
                components = None
            else:
                clean = grouped_taylor_importance(values, gradients)
                if config.pruning.method == "grouped_taylor":
                    scores, components = clean, None
                else:
                    bit_scores = _first_order_scores(values, gradients, codec)
                    susceptibility = grouped_bit_susceptibility(
                        bit_scores, len(clean), reduction=config.reliability.susceptibility_reduction
                    )
                    group_cost = np.full(len(clean), validate_dense_relu_dense(values).input_features + 1 + validate_dense_relu_dense(values).output_features)
                    components = netsurf_reliability_aware_importance(
                        clean, susceptibility, group_cost,
                        clean_weight=config.reliability.clean_weight,
                        susceptibility_weight=config.reliability.susceptibility_weight,
                        cost_weight=config.reliability.cost_weight,
                    )
                    scores = np.asarray(components.combined_prune_cost)
            current_width = validate_dense_relu_dense(values).hidden_width
            if target >= current_width:
                raise ValueError(f"target width {target} does not shrink current width {current_width}")
            removed = select_pruned_neurons(scores, current_width - target)
            removed_source = [active_source_indices[index] for index in removed]
            removed_set = set(removed)
            active_source_indices = [
                source_index for index, source_index in enumerate(active_source_indices)
                if index not in removed_set
            ]
            values = remove_hidden_neurons(values, removed)
            values, history = _fine_tune(values, config.source.dataset, config, codec, stage=stage)
            stages.append({
                "stage": stage, "source_width": current_width, "target_width": target,
                "removed_stage_indices": list(removed), "importance_scores": list(map(float, scores)),
                "removed_original_source_indices": removed_source,
                "retained_original_source_indices": list(active_source_indices),
                "importance_definition": (
                    "group-l1-or-l2/v1" if config.pruning.method == "group_magnitude"
                    else "sum-abs-parameter-times-mean-loss-gradient-over-hidden-dependency-group/v1"
                    if config.pruning.method == "grouped_taylor" else components.definition
                ),
                "reliability_components": None if components is None else {
                    "clean_preservation": list(components.clean_preservation),
                    "bit_susceptibility": list(components.bit_susceptibility),
                    "pruning_cost": list(components.pruning_cost),
                    "combined_prune_cost": list(components.combined_prune_cost),
                },
                "fine_tuning": history,
            })
    values = exact_quantization(values, codec)
    final_topology = topology(values, codec.format.total_bits)
    metadata = {
        "schema_version": "2.0.0",
        "checkpoint_kind": checkpoint_kind,
        "benchmark": "mnist_quantized",
        "architecture": f"Flatten(28x28)-Dense({final_topology.hidden_width})-ReLU-Dense(10 logits)",
        "quantization": codec.format.to_dict(),
        "quantization_backend": "netsurf_bitcodec_v1",
        "injectable_parameters": list(PARAMETER_NAMES),
        "source_checkpoint": {
            "path_recorded_from_configuration": str(config.source.checkpoint),
            "sha256": sha256_file(config.source.checkpoint),
            "metadata_sha256": digest_object(adapter.metadata),
        },
        "dataset": {"sha256": sha256_file(config.source.dataset)},
        "pruning_configuration": config.resolved_document(),
        "configuration_sha256": digest_object(config.resolved_document()),
        "method_classification": (
            "architecture-specific Vinck/Torch-Pruning adaptation" if config.pruning.method == "grouped_taylor"
            else "control" if config.pruning.method in {"group_magnitude", "independent_width"}
            else "NetSuRF-specific proposal; not RAP"
        ),
        "calibration_partition": calibration_partition.to_dict(),
        "pruning_stages": stages,
        "fine_tuning_seed": config.fine_tuning.seed,
        "topology": final_topology.to_dict(),
        "parameter_identity": _parameter_identity(values, codec),
        "scientific_payload_sha256": digest_object({
            "values": _parameter_identity(values, codec),
            "source_checkpoint_sha256": sha256_file(config.source.checkpoint),
            "dataset_sha256": sha256_file(config.source.dataset),
            "calibration": asdict(config.calibration),
            "pruning": {**asdict(config.pruning), "target_widths": list(config.pruning.target_widths)},
            "resolved_targets": list(targets),
            "reliability": asdict(config.reliability),
            "fine_tuning": asdict(config.fine_tuning),
        }),
    }
    return _save_checkpoint(config.output.checkpoint, values, metadata)


def inspect_pruned_checkpoint(path: str | Path) -> dict[str, object]:
    checkpoint = Path(path).expanduser().resolve()
    with np.load(checkpoint, allow_pickle=False) as data:
        if "metadata_json" not in data.files:
            raise ValueError("checkpoint has no metadata_json")
        metadata = json.loads(str(data["metadata_json"]))
        values = {name: data[name].astype(np.float32) for name in PARAMETER_NAMES}
    if metadata.get("checkpoint_kind") not in {"physically_structured_pruned", "independently_trained_width_control"}:
        raise ValueError("checkpoint is not an M5 structured-pruning artifact")
    codec = BitCodec(FixedPointFormat.parse(metadata["quantization"]["notation"]))
    exact_quantization(values, codec)
    shape = topology(values, codec.format.total_bits)
    expected = metadata.get("topology")
    if expected != shape.to_dict():
        raise ValueError("stored topology metadata does not match tensors")
    expected_identity = metadata.get("parameter_identity")
    if expected_identity != _parameter_identity(values, codec):
        raise ValueError("stored parameter identity hashes do not match tensors")
    return {
        "checkpoint": str(checkpoint), "sha256": sha256_file(checkpoint),
        "checkpoint_kind": metadata["checkpoint_kind"], "method_classification": metadata["method_classification"],
        "quantization_exact": True, "topology": shape.to_dict(), "provenance_valid": True,
    }
