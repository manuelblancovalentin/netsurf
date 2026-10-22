"""Versioned experiment configuration loading and executable validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import resources
import json
from pathlib import Path
import re
try:
    import tomllib
except ImportError:  # Python 3.10 legacy adapter environment; already provides tomli.
    import tomli as tomllib
from typing import Any, Mapping

from .paths import workspace_root
from .quantization import FixedPointFormat


EXPERIMENT_SCHEMA_VERSION = "1.0.0"
MANIFEST_SCHEMA_VERSION = "1.0.0"
SUMMARY_SCHEMA_VERSION = "1.0.0"
MEASUREMENT_SCHEMA_VERSION = "1.0.0"
SUPPORTED_PROTOCOL = "legacy-mnist-persistent-weight/v1"


class ConfigurationError(ValueError):
    """A configuration is malformed or requests unsupported behavior."""


class UnsupportedSchemaVersion(ConfigurationError):
    """A serialized document uses a schema this package cannot interpret."""


@dataclass(frozen=True)
class BenchmarkConfiguration:
    adapter: str
    id: str
    checkpoint: Path
    dataset: Path
    split: str
    evaluation_samples: int
    ranking_samples: int
    quantization: str


@dataclass(frozen=True)
class EstimatorConfiguration:
    name: str
    implementation_version: str
    policy: str
    result_name: str
    activation_source: str | None = None


@dataclass(frozen=True)
class ProtectionConfiguration:
    coverage: float
    mitigation_model: str


@dataclass(frozen=True)
class FaultConfiguration:
    target: str
    location: str
    persistence: str
    distribution: str
    probability: float


@dataclass(frozen=True)
class ExecutionConfiguration:
    trials: int
    confidence: float
    random_ranking_seed: int
    fault_seed: int


@dataclass(frozen=True)
class OutputConfiguration:
    root: Path
    report: bool
    plots: bool


@dataclass(frozen=True)
class ExperimentConfiguration:
    schema_version: str
    scientific_protocol: str
    name: str
    description: str
    objective: str
    benchmark: BenchmarkConfiguration
    estimators: tuple[EstimatorConfiguration, ...]
    protection: ProtectionConfiguration
    fault: FaultConfiguration
    execution: ExecutionConfiguration
    output: OutputConfiguration
    source_path: Path
    source_text: str

    def resolved_document(self) -> dict[str, Any]:
        """Return the execution-ready document with explicit absolute paths."""

        return {
            "schema_version": self.schema_version,
            "scientific_protocol": self.scientific_protocol,
            "experiment": {
                "name": self.name,
                "description": self.description,
                "objective": self.objective,
            },
            "benchmark": {
                **asdict(self.benchmark),
                "checkpoint": str(self.benchmark.checkpoint),
                "dataset": str(self.benchmark.dataset),
            },
            "estimators": [asdict(value) for value in self.estimators],
            "protection": asdict(self.protection),
            "fault": asdict(self.fault),
            "execution": asdict(self.execution),
            "output": {
                "root": str(self.output.root),
                "report": self.output.report,
                "plots": self.output.plots,
            },
        }


def load_schema(kind: str, version: str = "v1") -> dict[str, Any]:
    """Load a packaged schema document; used by validators and tests."""

    resource = resources.files("netsurf.schemas").joinpath(kind, f"{version}.schema.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def template_text(name: str = "template") -> str:
    if name not in {"template", "smoke"}:
        raise ConfigurationError(f"unknown experiment template: {name}")
    return resources.files("netsurf.templates").joinpath(f"{name}.toml").read_text(
        encoding="utf-8"
    )


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ConfigurationError(f"{location} must be a table")
    return value


def _check_keys(value: Mapping[str, Any], required: set[str], optional: set[str], location: str):
    missing = required - set(value)
    unknown = set(value) - required - optional
    if missing:
        raise ConfigurationError(f"{location} missing required fields: {sorted(missing)}")
    if unknown:
        raise ConfigurationError(f"{location} has unsupported fields: {sorted(unknown)}")


def _resolve_path(value: str, source: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = source.parent / path
    return path.resolve()


def load_experiment(path: str | Path, *, check_inputs: bool = True) -> ExperimentConfiguration:
    """Load, version-check, and validate a TOML experiment configuration."""

    source = Path(path).expanduser().resolve()
    try:
        text = source.read_text(encoding="utf-8")
        document = tomllib.loads(text)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ConfigurationError(f"cannot load experiment configuration {source}: {error}") from error
    return validate_experiment_document(document, source, text, check_inputs=check_inputs)


def validate_experiment_document(
    document: Mapping[str, Any],
    source_path: Path,
    source_text: str = "",
    *,
    check_inputs: bool = True,
) -> ExperimentConfiguration:
    """Validate the supported executable subset of experiment schema v1."""

    schema_version = document.get("schema_version")
    if schema_version != EXPERIMENT_SCHEMA_VERSION:
        raise UnsupportedSchemaVersion(
            f"unsupported experiment schema_version {schema_version!r}; "
            f"supported: {EXPERIMENT_SCHEMA_VERSION}"
        )
    load_schema("experiment")
    _check_keys(
        document,
        {
            "schema_version",
            "scientific_protocol",
            "experiment",
            "benchmark",
            "estimators",
            "protection",
            "fault",
            "execution",
            "output",
        },
        set(),
        "root",
    )
    protocol = document["scientific_protocol"]
    if protocol != SUPPORTED_PROTOCOL:
        raise ConfigurationError(
            f"unsupported scientific_protocol {protocol!r}; supported: {SUPPORTED_PROTOCOL}"
        )

    experiment = _mapping(document["experiment"], "experiment")
    _check_keys(experiment, {"name", "description"}, {"objective"}, "experiment")
    name = experiment["name"]
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ConfigurationError("experiment.name must be a filesystem-safe identifier")

    benchmark = _mapping(document["benchmark"], "benchmark")
    _check_keys(
        benchmark,
        {
            "adapter",
            "id",
            "checkpoint",
            "dataset",
            "split",
            "evaluation_samples",
            "ranking_samples",
            "quantization",
        },
        set(),
        "benchmark",
    )
    if benchmark["adapter"] != "legacy_mnist_lite":
        raise ConfigurationError("only benchmark.adapter='legacy_mnist_lite' is implemented")
    checkpoint = _resolve_path(str(benchmark["checkpoint"]), source_path)
    dataset = _resolve_path(str(benchmark["dataset"]), source_path)
    if check_inputs:
        if not checkpoint.is_file():
            raise ConfigurationError(f"checkpoint does not exist: {checkpoint}")
        if not dataset.is_file():
            raise ConfigurationError(f"dataset does not exist: {dataset}")
    for field in ("evaluation_samples", "ranking_samples"):
        if not isinstance(benchmark[field], int) or not 1 <= benchmark[field] <= 10_000:
            raise ConfigurationError(f"benchmark.{field} must be an integer in [1, 10000]")
    FixedPointFormat.parse(str(benchmark["quantization"]))

    raw_estimators = document["estimators"]
    if not isinstance(raw_estimators, list) or not raw_estimators:
        raise ConfigurationError("estimators must be a nonempty array of tables")
    supported_estimators = {
        "legacy_qpolargrad",
        "msb_first",
        "random",
        "gradient_magnitude",
        "first_order_gradient_delta",
    }
    estimators = []
    result_names = set()
    for index, raw in enumerate(raw_estimators):
        value = _mapping(raw, f"estimators[{index}]")
        _check_keys(
            value,
            {"name", "implementation_version", "policy", "result_name"},
            {"activation_source"},
            f"estimators[{index}]",
        )
        if value["name"] not in supported_estimators:
            raise ConfigurationError(f"unsupported estimator: {value['name']}")
        if value["policy"] not in {"global_ranking", "position_first"}:
            raise ConfigurationError(f"unsupported protection policy: {value['policy']}")
        if value["result_name"] in result_names:
            raise ConfigurationError(f"duplicate estimator result_name: {value['result_name']}")
        result_names.add(value["result_name"])
        activation_source = value.get("activation_source")
        if value["name"] == "legacy_qpolargrad" and activation_source not in {
            "clean",
            "globally_attacked_first_call",
        }:
            raise ConfigurationError(
                "legacy_qpolargrad requires activation_source='clean' or "
                "'globally_attacked_first_call'"
            )
        estimators.append(
            EstimatorConfiguration(
                str(value["name"]),
                str(value["implementation_version"]),
                str(value["policy"]),
                str(value["result_name"]),
                None if activation_source is None else str(activation_source),
            )
        )

    protection = _mapping(document["protection"], "protection")
    _check_keys(protection, {"coverage", "mitigation_model"}, set(), "protection")
    coverage = float(protection["coverage"])
    if not 0.0 <= coverage <= 1.0:
        raise ConfigurationError("protection.coverage must be in [0,1]")
    if protection["mitigation_model"] != "perfect_event_suppression":
        raise ConfigurationError("only perfect_event_suppression is implemented")

    fault = _mapping(document["fault"], "fault")
    _check_keys(
        fault,
        {"target", "location", "persistence", "distribution", "probability"},
        set(),
        "fault",
    )
    expected_fault = {
        "target": "weight",
        "location": "logical_trainable_parameter_bits",
        "persistence": "persistent_trial",
        "distribution": "independent_bernoulli",
    }
    for field, expected in expected_fault.items():
        if fault[field] != expected:
            raise ConfigurationError(f"unsupported fault.{field}: {fault[field]!r}")
    probability = float(fault["probability"])
    if not 0.0 <= probability <= 1.0:
        raise ConfigurationError("fault.probability must be in [0,1]")

    execution = _mapping(document["execution"], "execution")
    _check_keys(
        execution,
        {"trials", "confidence", "random_ranking_seed", "fault_seed"},
        set(),
        "execution",
    )
    if not isinstance(execution["trials"], int) or execution["trials"] < 1:
        raise ConfigurationError("execution.trials must be a positive integer")
    confidence = float(execution["confidence"])
    if not 0.0 < confidence < 1.0:
        raise ConfigurationError("execution.confidence must be in (0,1)")
    for field in ("random_ranking_seed", "fault_seed"):
        if not isinstance(execution[field], int) or execution[field] < 0:
            raise ConfigurationError(f"execution.{field} must be a nonnegative integer")

    output = _mapping(document["output"], "output")
    _check_keys(output, {"report", "plots"}, {"root"}, "output")
    if not isinstance(output["report"], bool) or not isinstance(output["plots"], bool):
        raise ConfigurationError("output.report and output.plots must be booleans")
    raw_root = output.get("root", "")
    output_root = (
        workspace_root() / "runs"
        if raw_root == ""
        else _resolve_path(str(raw_root), source_path)
    ).resolve()

    return ExperimentConfiguration(
        EXPERIMENT_SCHEMA_VERSION,
        str(protocol),
        str(name),
        str(experiment["description"]),
        str(experiment.get("objective", "")),
        BenchmarkConfiguration(
            str(benchmark["adapter"]),
            str(benchmark["id"]),
            checkpoint,
            dataset,
            str(benchmark["split"]),
            int(benchmark["evaluation_samples"]),
            int(benchmark["ranking_samples"]),
            str(benchmark["quantization"]),
        ),
        tuple(estimators),
        ProtectionConfiguration(coverage, str(protection["mitigation_model"])),
        FaultConfiguration(
            str(fault["target"]),
            str(fault["location"]),
            str(fault["persistence"]),
            str(fault["distribution"]),
            probability,
        ),
        ExecutionConfiguration(
            int(execution["trials"]),
            confidence,
            int(execution["random_ranking_seed"]),
            int(execution["fault_seed"]),
        ),
        OutputConfiguration(output_root, output["report"], output["plots"]),
        source_path.resolve(),
        source_text,
    )
