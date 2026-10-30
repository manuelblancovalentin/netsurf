"""Strict configuration for exact conditional-damage studies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
import tomllib
from typing import Any, Mapping

from .experiment_config import BenchmarkConfiguration, OutputConfiguration
from .identities import BitIdentity
from .paths import workspace_root


CONDITIONAL_SCHEMA_VERSION = "6.0.0"
CONDITIONAL_PROTOCOL = "exact-conditional-damage/v1"


class ConditionalConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class ConditionalSamples:
    calibration_count: int
    evaluation_count: int
    calibration_offset: int
    evaluation_offset: int
    require_disjoint: bool


@dataclass(frozen=True)
class ConditionalBackgrounds:
    probabilities: tuple[float, ...]
    calibration_count: int
    evaluation_count: int
    calibration_seed: int
    evaluation_seed: int


@dataclass(frozen=True)
class ConditionalExecution:
    confidence: float


@dataclass(frozen=True)
class RestrictedOracleConfiguration:
    enabled: bool
    budget: int
    calibration_background_count: int
    evaluation_background_count: int
    calibration_seed: int
    evaluation_seed: int


@dataclass(frozen=True)
class ConditionalConfiguration:
    schema_version: str
    scientific_protocol: str
    name: str
    description: str
    objective: str
    benchmark: BenchmarkConfiguration
    samples: ConditionalSamples
    risk_metric: str
    risk_direction: str
    candidates: tuple[BitIdentity, ...]
    backgrounds: ConditionalBackgrounds
    execution: ConditionalExecution
    restricted_oracle: RestrictedOracleConfiguration
    output: OutputConfiguration
    source_path: Path
    source_text: str

    def resolved_document(self) -> dict[str, Any]:
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
                "asset_lock": str(self.benchmark.asset_lock),
            },
            "samples": asdict(self.samples),
            "risk": {"metric": self.risk_metric, "direction": self.risk_direction},
            "candidates": [
                {
                    "parameter": value.parameter,
                    "flat_index": value.flat_index,
                    "bit_index": value.bit_index,
                }
                for value in self.candidates
            ],
            "backgrounds": asdict(self.backgrounds),
            "execution": asdict(self.execution),
            "restricted_oracle": asdict(self.restricted_oracle),
            "output": asdict(self.output) | {"root": str(self.output.root)},
        }


def _table(document: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = document.get(name)
    if not isinstance(value, dict):
        raise ConditionalConfigurationError(f"{name} must be a table")
    return value


def _keys(value: Mapping[str, Any], required: set[str], optional: set[str], name: str) -> None:
    missing = required - set(value)
    unknown = set(value) - required - optional
    if missing:
        raise ConditionalConfigurationError(f"{name} missing fields: {sorted(missing)}")
    if unknown:
        raise ConditionalConfigurationError(f"{name} unsupported fields: {sorted(unknown)}")


def _path(value: str, source: Path) -> Path:
    path = Path(value).expanduser()
    return (source.parent / path if not path.is_absolute() else path).resolve()


def load_conditional(path: str | Path, *, check_inputs: bool = True) -> ConditionalConfiguration:
    source = Path(path).expanduser().resolve()
    try:
        text = source.read_text(encoding="utf-8")
        document = tomllib.loads(text)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ConditionalConfigurationError(f"cannot load conditional configuration {source}: {error}") from error

    _keys(
        document,
        {
            "schema_version", "scientific_protocol", "experiment", "benchmark",
            "samples", "risk", "candidates", "backgrounds", "execution", "output",
        },
        {"restricted_oracle"},
        "root",
    )
    if document["schema_version"] != CONDITIONAL_SCHEMA_VERSION:
        raise ConditionalConfigurationError(
            f"unsupported conditional schema: {document['schema_version']!r}"
        )
    if document["scientific_protocol"] != CONDITIONAL_PROTOCOL:
        raise ConditionalConfigurationError(
            f"unsupported conditional protocol: {document['scientific_protocol']!r}"
        )

    experiment = _table(document, "experiment")
    _keys(experiment, {"name", "description", "objective"}, set(), "experiment")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", str(experiment["name"])):
        raise ConditionalConfigurationError("experiment.name must be filesystem-safe")

    raw_benchmark = _table(document, "benchmark")
    _keys(
        raw_benchmark,
        {
            "adapter", "id", "checkpoint", "dataset", "asset_lock",
            "inference_backend", "split", "quantization", "quantization_backend",
            "injectable_parameters",
        },
        set(),
        "benchmark",
    )
    if raw_benchmark["adapter"] != "mlperf_tiny_kws_int8_v1":
        raise ConditionalConfigurationError(
            "schema 6.0 currently validates exact conditional damage only for mlperf_tiny_kws_int8_v1"
        )
    checkpoint = _path(str(raw_benchmark["checkpoint"]), source)
    dataset = _path(str(raw_benchmark["dataset"]), source)
    asset_lock = _path(str(raw_benchmark["asset_lock"]), source)
    if check_inputs and not (checkpoint.is_file() and dataset.exists() and asset_lock.is_file()):
        raise ConditionalConfigurationError(
            f"missing checkpoint, dataset, or asset lock: {checkpoint}, {dataset}, {asset_lock}"
        )
    if raw_benchmark["inference_backend"] != "tensorflow_lite_builtin_ref_v1":
        raise ConditionalConfigurationError("conditional damage requires TFLite BUILTIN_REF")
    if raw_benchmark["quantization_backend"] != "tflite_affine_integer_v1":
        raise ConditionalConfigurationError("conditional damage requires exact TFLite affine integer state")
    if tuple(raw_benchmark["injectable_parameters"]) != ("kernels",):
        raise ConditionalConfigurationError("M9.5 fault population is exactly INT8 kernels")
    benchmark = BenchmarkConfiguration(
        str(raw_benchmark["adapter"]), str(raw_benchmark["id"]), checkpoint, dataset,
        str(raw_benchmark["split"]), 0, 0, str(raw_benchmark["quantization"]),
        str(raw_benchmark["quantization_backend"]), tuple(raw_benchmark["injectable_parameters"]),
        asset_lock, str(raw_benchmark["inference_backend"]),
    )

    raw_samples = _table(document, "samples")
    _keys(
        raw_samples,
        {"calibration_count", "evaluation_count", "calibration_offset", "evaluation_offset", "require_disjoint"},
        set(), "samples",
    )
    samples = ConditionalSamples(
        int(raw_samples["calibration_count"]), int(raw_samples["evaluation_count"]),
        int(raw_samples["calibration_offset"]), int(raw_samples["evaluation_offset"]),
        bool(raw_samples["require_disjoint"]),
    )
    if samples.calibration_count < 1 or samples.evaluation_count < 1:
        raise ConditionalConfigurationError("sample counts must be positive")
    if samples.calibration_offset < 0 or samples.evaluation_offset < 0:
        raise ConditionalConfigurationError("sample offsets must be nonnegative")
    if not samples.require_disjoint:
        raise ConditionalConfigurationError("M9.5 requires disjoint calibration and held-out evaluation")

    raw_risk = _table(document, "risk")
    _keys(raw_risk, {"metric", "direction"}, set(), "risk")
    if (raw_risk["metric"], raw_risk["direction"]) != ("categorical_crossentropy", "minimize"):
        raise ConditionalConfigurationError(
            "KWS conditional damage currently supports categorical_crossentropy risk with direction=minimize"
        )

    raw_candidates = document["candidates"]
    if not isinstance(raw_candidates, list) or not 1 <= len(raw_candidates) <= 12:
        raise ConditionalConfigurationError("candidates must contain 1 to 12 explicit bit identities")
    candidates = []
    for index, value in enumerate(raw_candidates):
        if not isinstance(value, dict):
            raise ConditionalConfigurationError(f"candidates[{index}] must be a table")
        _keys(value, {"parameter", "flat_index", "bit_index"}, set(), f"candidates[{index}]")
        candidate = BitIdentity(str(value["parameter"]), int(value["flat_index"]), int(value["bit_index"]))
        if candidate.flat_index < 0 or not 0 <= candidate.bit_index < 8:
            raise ConditionalConfigurationError(f"invalid candidate identity: {candidate.key()}")
        candidates.append(candidate)
    if len(set(candidates)) != len(candidates):
        raise ConditionalConfigurationError("candidate identities must be unique")

    raw_backgrounds = _table(document, "backgrounds")
    _keys(
        raw_backgrounds,
        {"probabilities", "calibration_count", "evaluation_count", "calibration_seed", "evaluation_seed"},
        set(), "backgrounds",
    )
    probabilities = tuple(float(value) for value in raw_backgrounds["probabilities"])
    if not probabilities or len(set(probabilities)) != len(probabilities):
        raise ConditionalConfigurationError("background probabilities must be nonempty and unique")
    if any(not 0.0 <= value <= 1.0 for value in probabilities):
        raise ConditionalConfigurationError("background probabilities must lie in [0,1]")
    backgrounds = ConditionalBackgrounds(
        probabilities, int(raw_backgrounds["calibration_count"]),
        int(raw_backgrounds["evaluation_count"]), int(raw_backgrounds["calibration_seed"]),
        int(raw_backgrounds["evaluation_seed"]),
    )
    if backgrounds.calibration_count < 1 or backgrounds.evaluation_count < 1:
        raise ConditionalConfigurationError("background counts must be positive")
    if backgrounds.calibration_seed == backgrounds.evaluation_seed:
        raise ConditionalConfigurationError("calibration and evaluation background seeds must differ")

    raw_execution = _table(document, "execution")
    _keys(raw_execution, {"confidence"}, set(), "execution")
    execution = ConditionalExecution(float(raw_execution["confidence"]))
    if not 0.0 < execution.confidence < 1.0:
        raise ConditionalConfigurationError("execution.confidence must lie in (0,1)")

    raw_oracle = document.get("restricted_oracle", {})
    if not isinstance(raw_oracle, dict):
        raise ConditionalConfigurationError("restricted_oracle must be a table")
    _keys(
        raw_oracle, set(),
        {"enabled", "budget", "calibration_background_count", "evaluation_background_count", "calibration_seed", "evaluation_seed"},
        "restricted_oracle",
    )
    oracle = RestrictedOracleConfiguration(
        bool(raw_oracle.get("enabled", False)), int(raw_oracle.get("budget", 1)),
        int(raw_oracle.get("calibration_background_count", 1)),
        int(raw_oracle.get("evaluation_background_count", 1)),
        int(raw_oracle.get("calibration_seed", backgrounds.calibration_seed + 10_000_019)),
        int(raw_oracle.get("evaluation_seed", backgrounds.evaluation_seed + 10_000_019)),
    )
    if oracle.enabled and not 1 <= oracle.budget < len(candidates):
        raise ConditionalConfigurationError("restricted oracle budget must be in [1,K-1]")
    if oracle.enabled and len(candidates) > 12:
        raise ConditionalConfigurationError("restricted oracle is bounded to at most 12 candidates")
    if oracle.calibration_seed == oracle.evaluation_seed:
        raise ConditionalConfigurationError("restricted-oracle calibration/evaluation seeds must differ")
    if oracle.enabled:
        raise ConditionalConfigurationError(
            "restricted_oracle execution is not wired into schema-6 bundles; use the bounded "
            "select_restricted_subcube_mask API with separately constructed calibration and "
            "held-out background risks"
        )

    raw_output = _table(document, "output")
    _keys(raw_output, {"root", "report", "plots"}, set(), "output")
    root = str(raw_output["root"]).strip()
    if bool(raw_output["plots"]):
        raise ConditionalConfigurationError(
            "schema-6 does not yet define plot artifacts; output.plots must be false"
        )
    output = OutputConfiguration(
        workspace_root() / "runs" if not root else _path(root, source),
        bool(raw_output["report"]), bool(raw_output["plots"]),
    )
    return ConditionalConfiguration(
        CONDITIONAL_SCHEMA_VERSION, CONDITIONAL_PROTOCOL,
        str(experiment["name"]), str(experiment["description"]), str(experiment["objective"]),
        benchmark, samples, str(raw_risk["metric"]), str(raw_risk["direction"]),
        tuple(candidates), backgrounds, execution, oracle, output, source, text,
    )
