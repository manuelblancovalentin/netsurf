"""Versioned configuration for physical hidden-neuron pruning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
import tomllib
from typing import Any, Mapping

import numpy as np

from .paths import workspace_root
from .quantization import FixedPointFormat


PRUNING_SCHEMA_VERSION = "5.0.0"
PRUNING_PROTOCOL = "structured-hidden-neuron-pruning/v1"


class PruningConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class PruningSource:
    adapter: str
    checkpoint: Path
    dataset: Path
    quantization: str


@dataclass(frozen=True)
class PruningCalibration:
    sample_count: int
    sample_offset: int


@dataclass(frozen=True)
class PruningPlan:
    method: str
    schedule: str
    iterations: int
    step_fraction: float
    target_widths: tuple[int, ...]
    magnitude_norm: str


@dataclass(frozen=True)
class ReliabilityComposition:
    susceptibility_estimator: str
    susceptibility_reduction: str
    clean_weight: float
    susceptibility_weight: float
    cost_weight: float


@dataclass(frozen=True)
class FineTuning:
    enabled: bool
    mode: str
    epochs_per_stage: int
    batch_size: int
    learning_rate: float
    training_samples: int
    clean_loss_weight: float
    fault_loss_weight: float
    fault_probability: float
    seed: int


@dataclass(frozen=True)
class PruningOutput:
    checkpoint: Path


@dataclass(frozen=True)
class PruningConfiguration:
    schema_version: str
    scientific_protocol: str
    name: str
    description: str
    source: PruningSource
    calibration: PruningCalibration
    pruning: PruningPlan
    reliability: ReliabilityComposition
    fine_tuning: FineTuning
    output: PruningOutput
    source_path: Path
    source_text: str

    def resolved_document(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scientific_protocol": self.scientific_protocol,
            "experiment": {"name": self.name, "description": self.description},
            "source": {
                **asdict(self.source),
                "checkpoint": str(self.source.checkpoint),
                "dataset": str(self.source.dataset),
            },
            "calibration": asdict(self.calibration),
            "pruning": {**asdict(self.pruning), "target_widths": list(self.pruning.target_widths)},
            "reliability": asdict(self.reliability),
            "fine_tuning": asdict(self.fine_tuning),
            "output": {"checkpoint": str(self.output.checkpoint)},
            "source_configuration": str(self.source_path),
        }


def _table(document: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = document.get(name)
    if not isinstance(value, dict):
        raise PruningConfigurationError(f"{name} must be a table")
    return value


def _keys(value: Mapping[str, Any], required: set[str], optional: set[str], name: str) -> None:
    missing = required - set(value)
    unknown = set(value) - required - optional
    if missing:
        raise PruningConfigurationError(f"{name} missing fields: {sorted(missing)}")
    if unknown:
        raise PruningConfigurationError(f"{name} unsupported fields: {sorted(unknown)}")


def _path(value: object, source: Path) -> Path:
    result = Path(str(value)).expanduser()
    return (source.parent / result if not result.is_absolute() else result).resolve()


def load_pruning(path: str | Path, *, check_inputs: bool = True) -> PruningConfiguration:
    source_path = Path(path).expanduser().resolve()
    try:
        source_text = source_path.read_text(encoding="utf-8")
        document = tomllib.loads(source_text)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise PruningConfigurationError(f"cannot load pruning configuration {source_path}: {error}") from error
    required = {
        "schema_version", "scientific_protocol", "experiment", "source", "calibration",
        "pruning", "reliability", "fine_tuning", "output",
    }
    _keys(document, required, set(), "root")
    if document["schema_version"] != PRUNING_SCHEMA_VERSION:
        raise PruningConfigurationError(f"unsupported pruning schema: {document['schema_version']!r}")
    if document["scientific_protocol"] != PRUNING_PROTOCOL:
        raise PruningConfigurationError(f"unsupported pruning protocol: {document['scientific_protocol']!r}")

    experiment = _table(document, "experiment")
    _keys(experiment, {"name", "description"}, set(), "experiment")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", str(experiment["name"])):
        raise PruningConfigurationError("experiment.name must be filesystem-safe")

    raw_source = _table(document, "source")
    _keys(raw_source, {"adapter", "checkpoint", "dataset", "quantization"}, set(), "source")
    if raw_source["adapter"] != "mnist_quantized":
        raise PruningConfigurationError("M5.0 pruning supports only the explicit mnist_quantized adapter")
    checkpoint, dataset = _path(raw_source["checkpoint"], source_path), _path(raw_source["dataset"], source_path)
    if check_inputs and (not checkpoint.is_file() or not dataset.is_file()):
        raise PruningConfigurationError(f"missing source checkpoint or dataset: {checkpoint}, {dataset}")
    FixedPointFormat.parse(str(raw_source["quantization"]))

    raw_calibration = _table(document, "calibration")
    _keys(raw_calibration, {"sample_count", "sample_offset"}, set(), "calibration")
    calibration = PruningCalibration(int(raw_calibration["sample_count"]), int(raw_calibration["sample_offset"]))
    if calibration.sample_count < 1 or calibration.sample_offset < 0 or calibration.sample_offset + calibration.sample_count > 2500:
        raise PruningConfigurationError("calibration sample_count/offset must fit the reserved 2500-sample ranking partition")

    raw_plan = _table(document, "pruning")
    _keys(raw_plan, {"method", "schedule", "iterations", "step_fraction", "target_widths", "magnitude_norm"}, set(), "pruning")
    method = str(raw_plan["method"])
    if method not in {"grouped_taylor", "group_magnitude", "netsurf_reliability_aware", "independent_width"}:
        raise PruningConfigurationError("unsupported pruning.method")
    schedule = str(raw_plan["schedule"])
    if schedule not in {"paper_fraction", "target_widths"}:
        raise PruningConfigurationError("pruning.schedule must be paper_fraction or target_widths")
    if not isinstance(raw_plan["target_widths"], list) or any(
        not isinstance(value, int) for value in raw_plan["target_widths"]
    ):
        raise PruningConfigurationError("pruning.target_widths must be an integer array")
    targets = tuple(map(int, raw_plan["target_widths"]))
    plan = PruningPlan(method, schedule, int(raw_plan["iterations"]), float(raw_plan["step_fraction"]), targets, str(raw_plan["magnitude_norm"]))
    if plan.iterations < 1 or not 0 < plan.step_fraction < 1 or plan.magnitude_norm not in {"l1", "l2"}:
        raise PruningConfigurationError("invalid pruning iterations, step_fraction, or magnitude_norm")
    if schedule == "target_widths" and not targets:
        raise PruningConfigurationError("target_widths schedule requires at least one target")
    if method == "independent_width" and (schedule != "target_widths" or len(targets) != 1):
        raise PruningConfigurationError("independent_width requires exactly one target_width")
    if check_inputs:
        try:
            with np.load(checkpoint, allow_pickle=False) as artifact:
                source_width = int(artifact["dense0/bias"].size)
        except (OSError, KeyError, ValueError) as error:
            raise PruningConfigurationError(f"cannot inspect source checkpoint topology: {error}") from error
        if schedule == "target_widths" and (
            any(right >= left for left, right in zip((source_width, *targets[:-1]), targets))
            or any(value < 1 for value in targets)
        ):
            raise PruningConfigurationError(
                f"target_widths must strictly decrease from source width {source_width}"
            )

    raw_reliability = _table(document, "reliability")
    _keys(raw_reliability, {"susceptibility_estimator", "susceptibility_reduction", "clean_weight", "susceptibility_weight", "cost_weight"}, set(), "reliability")
    reliability = ReliabilityComposition(
        str(raw_reliability["susceptibility_estimator"]), str(raw_reliability["susceptibility_reduction"]),
        float(raw_reliability["clean_weight"]), float(raw_reliability["susceptibility_weight"]),
        float(raw_reliability["cost_weight"]),
    )
    if reliability.susceptibility_estimator != "first_order_gradient_delta":
        raise PruningConfigurationError("M5.0 supports first_order_gradient_delta susceptibility only")
    if reliability.susceptibility_reduction not in {"mean", "sum", "max"}:
        raise PruningConfigurationError("unsupported susceptibility_reduction")
    if any(value < 0 for value in (reliability.clean_weight, reliability.susceptibility_weight, reliability.cost_weight)):
        raise PruningConfigurationError("reliability component weights must be nonnegative")

    raw_tuning = _table(document, "fine_tuning")
    _keys(raw_tuning, {"enabled", "mode", "epochs_per_stage", "batch_size", "learning_rate", "training_samples", "clean_loss_weight", "fault_loss_weight", "fault_probability", "seed"}, set(), "fine_tuning")
    if not isinstance(raw_tuning["enabled"], bool):
        raise PruningConfigurationError("fine_tuning.enabled must be boolean")
    tuning = FineTuning(
        bool(raw_tuning["enabled"]), str(raw_tuning["mode"]), int(raw_tuning["epochs_per_stage"]),
        int(raw_tuning["batch_size"]), float(raw_tuning["learning_rate"]), int(raw_tuning["training_samples"]),
        float(raw_tuning["clean_loss_weight"]), float(raw_tuning["fault_loss_weight"]),
        float(raw_tuning["fault_probability"]), int(raw_tuning["seed"]),
    )
    if tuning.mode not in {"standard", "quantization_aware", "netsurf_fault_augmented"}:
        raise PruningConfigurationError("unsupported fine_tuning.mode")
    if tuning.epochs_per_stage < 0 or tuning.batch_size < 1 or tuning.learning_rate <= 0 or not 1 <= tuning.training_samples <= 50_000:
        raise PruningConfigurationError("invalid fine-tuning numerical fields")
    if not 0 <= tuning.fault_probability <= 1 or tuning.clean_loss_weight < 0 or tuning.fault_loss_weight < 0:
        raise PruningConfigurationError("invalid fine-tuning loss weights or fault probability")
    if tuning.mode == "netsurf_fault_augmented" and tuning.fault_loss_weight <= 0:
        raise PruningConfigurationError("fault-augmented fine-tuning requires fault_loss_weight > 0")
    if method == "independent_width" and (not tuning.enabled or tuning.epochs_per_stage < 1):
        raise PruningConfigurationError("independent_width control requires enabled training with at least one epoch")

    raw_output = _table(document, "output")
    _keys(raw_output, {"checkpoint"}, set(), "output")
    output = PruningOutput(_path(raw_output["checkpoint"], source_path))
    if output.checkpoint.suffix != ".npz":
        raise PruningConfigurationError("output.checkpoint must end in .npz")
    try:
        output.checkpoint.relative_to(workspace_root())
    except ValueError as error:
        raise PruningConfigurationError(
            f"output.checkpoint must be under the NetSuRF workspace: {workspace_root()}"
        ) from error
    if check_inputs and output.checkpoint.exists():
        raise PruningConfigurationError(f"output checkpoint already exists and is immutable: {output.checkpoint}")

    return PruningConfiguration(
        PRUNING_SCHEMA_VERSION, PRUNING_PROTOCOL, str(experiment["name"]), str(experiment["description"]),
        PruningSource("mnist_quantized", checkpoint, dataset, str(raw_source["quantization"])),
        calibration, plan, reliability, tuning, output, source_path, source_text,
    )
