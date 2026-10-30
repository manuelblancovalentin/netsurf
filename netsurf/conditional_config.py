"""Strict configuration for exact conditional-damage studies."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
import re
import tomllib
from typing import Any, Mapping

from .experiment_config import BenchmarkConfiguration, OutputConfiguration
from .identities import BitIdentity
from .paths import workspace_root
from .protection_context import ProtectionContextConfiguration


CONDITIONAL_SCHEMA_VERSION = "6.0.0"
CONDITIONAL_SCHEMA_VERSION_6_1 = "6.1.0"
CONDITIONAL_SCHEMA_VERSION_6_2 = "6.2.0"
CONDITIONAL_SCHEMA_VERSION_6_3 = "6.3.0"
CONDITIONAL_PROTOCOL = "exact-conditional-damage/v1"
CONDITIONAL_PROTOCOL_6_1 = "exact-conditional-protection/v2"
CONDITIONAL_PROTOCOL_6_2 = "contextual-exact-conditional-protection/v3"
CONDITIONAL_PROTOCOL_6_3 = "paired-swap-held-out-protection/v4"

PAIRED_SWAP_COMPARISONS = (
    "restricted_selected_vs_original_s5",
    "restricted_selected_vs_exact_isolated",
    "exact_isolated_vs_original_s5",
)


class ConditionalConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class ConditionalSamples:
    calibration_count: int
    evaluation_count: int
    calibration_offset: int
    evaluation_offset: int
    require_disjoint: bool
    calibration_role: str = "calibration"
    model_selection_role: str | None = None
    model_selection_count: int = 0
    model_selection_offset: int = 0
    evaluation_role: str = "final_evaluation"


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
    calibration_replicate_seeds: tuple[int, ...] = ()

    @property
    def selection_seeds(self) -> tuple[int, ...]:
        return self.calibration_replicate_seeds or (self.calibration_seed,)


@dataclass(frozen=True)
class PairedSwapEvaluationConfiguration:
    enabled: bool = False
    background_count: int = 0
    seed: int = 0
    comparisons: tuple[str, ...] = ()


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
    protection_context: ProtectionContextConfiguration = field(
        default_factory=ProtectionContextConfiguration
    )
    paired_swap_evaluation: PairedSwapEvaluationConfiguration = field(
        default_factory=PairedSwapEvaluationConfiguration
    )

    def operation_counts(self) -> dict[str, int]:
        conditional = len(self.backgrounds.probabilities) * (
            len(self.candidates) * (1 + 2 * self.backgrounds.calibration_count) + 1
            + len(self.candidates) * (1 + 2 * self.backgrounds.evaluation_count) + 1
        )
        oracle = 0
        selection_fields = 0
        if self.restricted_oracle.enabled:
            selection_fields = (
                self.restricted_oracle.calibration_background_count
                * len(self.restricted_oracle.selection_seeds)
            )
            oracle = len(self.backgrounds.probabilities) * (
                len(self.candidates) + 1
                + (
                    selection_fields
                    + self.restricted_oracle.evaluation_background_count
                ) * (2 ** len(self.candidates))
            )
        paired_upper = 0
        if self.paired_swap_evaluation.enabled:
            paired_upper = len(self.backgrounds.probabilities) * (
                self.paired_swap_evaluation.background_count
                * len(self.paired_swap_evaluation.comparisons)
                * 2 * self.restricted_oracle.budget
            )
        if self.schema_version != CONDITIONAL_SCHEMA_VERSION_6_3:
            return {
                "conditional_requested_evaluations": conditional,
                "restricted_oracle_requested_evaluations": oracle,
                "total_requested_evaluations": conditional + oracle,
            }
        mask_count = math.comb(
            len(self.candidates), self.restricted_oracle.budget
        ) if self.restricted_oracle.enabled else 0
        table_fields = selection_fields + (
            self.restricted_oracle.evaluation_background_count
            if self.restricted_oracle.enabled else 0
        )
        return {
            "conditional_requested_evaluations": conditional,
            "restricted_oracle_requested_evaluations": oracle,
            "paired_swap_authoritative_evaluations_upper_bound": paired_upper,
            "total_authoritative_evaluations_upper_bound": conditional + oracle + paired_upper,
            "total_requested_evaluations": conditional + oracle + paired_upper,
            "model_selection_outside_fields": selection_fields,
            "full_enumeration_reference_fields": (
                self.restricted_oracle.evaluation_background_count
                if self.restricted_oracle.enabled else 0
            ),
            "analytical_mask_risk_integrations": (
                len(self.backgrounds.probabilities) * table_fields * mask_count
            ),
            "analytical_pairwise_mask_differences": (
                len(self.backgrounds.probabilities) * table_fields
                * math.comb(mask_count, 2) if mask_count >= 2 else 0
            ),
            "paired_swap_whole_path_observations": (
                len(self.backgrounds.probabilities)
                * self.paired_swap_evaluation.background_count
                * len(self.paired_swap_evaluation.comparisons)
                if self.paired_swap_evaluation.enabled else 0
            ),
        }

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
            "protection_context": self.protection_context.to_dict(),
            "paired_swap_evaluation": asdict(self.paired_swap_evaluation),
            "output": asdict(self.output) | {"root": str(self.output.root)},
            "operation_counts": self.operation_counts(),
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
        {"restricted_oracle", "protection_context", "paired_swap_evaluation"},
        "root",
    )
    schema_version = str(document["schema_version"])
    if schema_version not in {
        CONDITIONAL_SCHEMA_VERSION, CONDITIONAL_SCHEMA_VERSION_6_1,
        CONDITIONAL_SCHEMA_VERSION_6_2, CONDITIONAL_SCHEMA_VERSION_6_3,
    }:
        raise ConditionalConfigurationError(
            f"unsupported conditional schema: {document['schema_version']!r}"
        )
    expected_protocol = {
        CONDITIONAL_SCHEMA_VERSION: CONDITIONAL_PROTOCOL,
        CONDITIONAL_SCHEMA_VERSION_6_1: CONDITIONAL_PROTOCOL_6_1,
        CONDITIONAL_SCHEMA_VERSION_6_2: CONDITIONAL_PROTOCOL_6_2,
        CONDITIONAL_SCHEMA_VERSION_6_3: CONDITIONAL_PROTOCOL_6_3,
    }[schema_version]
    if document["scientific_protocol"] != expected_protocol:
        raise ConditionalConfigurationError(
            f"unsupported conditional protocol: {document['scientific_protocol']!r}"
        )
    if schema_version == CONDITIONAL_SCHEMA_VERSION_6_2 and "protection_context" not in document:
        raise ConditionalConfigurationError("schema 6.2 requires protection_context")
    if schema_version == CONDITIONAL_SCHEMA_VERSION_6_3 and not {
        "protection_context", "paired_swap_evaluation"
    } <= set(document):
        raise ConditionalConfigurationError(
            "schema 6.3 requires protection_context and paired_swap_evaluation"
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
    sample_required = {
        "calibration_count", "evaluation_count", "calibration_offset",
        "evaluation_offset", "require_disjoint",
    }
    sample_61 = {
        "calibration_role", "model_selection_role", "model_selection_count",
        "model_selection_offset", "evaluation_role",
    }
    _keys(raw_samples, sample_required | (sample_61 if schema_version in {
        CONDITIONAL_SCHEMA_VERSION_6_1, CONDITIONAL_SCHEMA_VERSION_6_2,
        CONDITIONAL_SCHEMA_VERSION_6_3,
    } else set()), set(), "samples")
    samples = ConditionalSamples(
        int(raw_samples["calibration_count"]), int(raw_samples["evaluation_count"]),
        int(raw_samples["calibration_offset"]), int(raw_samples["evaluation_offset"]),
        bool(raw_samples["require_disjoint"]),
        str(raw_samples.get("calibration_role", "calibration")),
        (str(raw_samples["model_selection_role"]) if "model_selection_role" in raw_samples else None),
        int(raw_samples.get("model_selection_count", 0)),
        int(raw_samples.get("model_selection_offset", 0)),
        str(raw_samples.get("evaluation_role", "final_evaluation")),
    )
    if samples.calibration_count < 1 or samples.evaluation_count < 1:
        raise ConditionalConfigurationError("sample counts must be positive")
    if samples.calibration_offset < 0 or samples.evaluation_offset < 0:
        raise ConditionalConfigurationError("sample offsets must be nonnegative")
    if not samples.require_disjoint:
        raise ConditionalConfigurationError("M9.5 requires disjoint calibration and held-out evaluation")
    if schema_version in {
        CONDITIONAL_SCHEMA_VERSION_6_1, CONDITIONAL_SCHEMA_VERSION_6_2,
        CONDITIONAL_SCHEMA_VERSION_6_3,
    }:
        expected_roles = (
            "conditional_calibration", "conditional_model_selection",
            "conditional_final_evaluation",
        )
        observed_roles = (
            samples.calibration_role, samples.model_selection_role, samples.evaluation_role,
        )
        if observed_roles != expected_roles:
            raise ConditionalConfigurationError(
                f"schema 6.1 requires same-distribution roles {expected_roles}, observed {observed_roles}"
            )
        if samples.model_selection_count < 1 or samples.model_selection_offset < 0:
            raise ConditionalConfigurationError("schema 6.1 model-selection count must be positive and offset nonnegative")
        if raw_benchmark["split"] != "full_test_4890":
            raise ConditionalConfigurationError("schema 6.1 KWS studies require benchmark.split=full_test_4890")
        window_capacities = {
            "calibration": 1024,
            "model-selection": 1024,
            "final-evaluation": 1842,
        }
        requested_windows = {
            "calibration": (samples.calibration_offset, samples.calibration_count),
            "model-selection": (samples.model_selection_offset, samples.model_selection_count),
            "final-evaluation": (samples.evaluation_offset, samples.evaluation_count),
        }
        for name, (offset, count) in requested_windows.items():
            if offset + count > window_capacities[name]:
                raise ConditionalConfigurationError(
                    f"schema 6.1 {name} offset+count exceeds its fixed "
                    f"same-distribution window capacity {window_capacities[name]}"
                )
        if schema_version == CONDITIONAL_SCHEMA_VERSION_6_2 and samples.evaluation_offset < 6:
            raise ConditionalConfigurationError(
                "schema 6.2 final evaluation excludes previously exposed relative indices 0-5; "
                "evaluation_offset must be at least 6"
            )
        if schema_version == CONDITIONAL_SCHEMA_VERSION_6_3 and samples.evaluation_offset < 12:
            raise ConditionalConfigurationError(
                "schema 6.3 final evaluation excludes previously exposed relative indices 0-11; "
                "evaluation_offset must be at least 12"
            )

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
    oracle_optional = {
        "enabled", "budget", "calibration_background_count",
        "evaluation_background_count", "calibration_seed", "evaluation_seed",
    }
    if schema_version == CONDITIONAL_SCHEMA_VERSION_6_3:
        oracle_optional.add("calibration_replicate_seeds")
    _keys(raw_oracle, set(), oracle_optional, "restricted_oracle")
    oracle = RestrictedOracleConfiguration(
        bool(raw_oracle.get("enabled", False)), int(raw_oracle.get("budget", 1)),
        int(raw_oracle.get("calibration_background_count", 1)),
        int(raw_oracle.get("evaluation_background_count", 1)),
        int(raw_oracle.get("calibration_seed", backgrounds.calibration_seed + 10_000_019)),
        int(raw_oracle.get("evaluation_seed", backgrounds.evaluation_seed + 10_000_019)),
        tuple(int(value) for value in raw_oracle.get("calibration_replicate_seeds", ())),
    )
    if oracle.enabled and not 1 <= oracle.budget < len(candidates):
        raise ConditionalConfigurationError("restricted oracle budget must be in [1,K-1]")
    if oracle.enabled and len(candidates) > 12:
        raise ConditionalConfigurationError("restricted oracle is bounded to at most 12 candidates")
    if oracle.calibration_seed == oracle.evaluation_seed:
        raise ConditionalConfigurationError("restricted-oracle calibration/evaluation seeds must differ")
    if len(set(oracle.selection_seeds)) != len(oracle.selection_seeds):
        raise ConditionalConfigurationError("model-selection replicate seeds must be unique")
    if (
        schema_version == CONDITIONAL_SCHEMA_VERSION_6_3
        and not oracle.calibration_replicate_seeds
    ):
        raise ConditionalConfigurationError(
            "schema 6.3 requires explicit model-selection calibration_replicate_seeds"
        )
    if oracle.evaluation_seed in oracle.selection_seeds:
        raise ConditionalConfigurationError(
            "model-selection and full-enumeration evaluation seeds must be independent"
        )
    if oracle.enabled and (
        oracle.calibration_background_count < 1 or oracle.evaluation_background_count < 1
    ):
        raise ConditionalConfigurationError("restricted-oracle background counts must be positive")
    if oracle.enabled and schema_version == CONDITIONAL_SCHEMA_VERSION:
        raise ConditionalConfigurationError(
            "restricted_oracle execution is not wired into schema-6 bundles; use the bounded "
            "select_restricted_subcube_mask API with separately constructed calibration and "
            "held-out background risks"
        )
    if schema_version in {CONDITIONAL_SCHEMA_VERSION_6_2, CONDITIONAL_SCHEMA_VERSION_6_3} and not oracle.enabled:
        raise ConditionalConfigurationError("schema 6.2+ requires restricted_oracle.enabled=true")

    raw_context = document.get("protection_context", {})
    if not isinstance(raw_context, dict):
        raise ConditionalConfigurationError("protection_context must be a table")
    if schema_version in {CONDITIONAL_SCHEMA_VERSION_6_2, CONDITIONAL_SCHEMA_VERSION_6_3}:
        _keys(
            raw_context,
            {"mode"},
            {
                "source_bundle", "artifact", "artifact_sha256",
                "scientific_payload_sha256", "mask_key",
                "expected_population_size", "expected_global_cardinality",
            },
            "protection_context",
        )
    elif raw_context:
        raise ConditionalConfigurationError("protection_context requires schema 6.2+")
    mode = str(raw_context.get("mode", "none"))
    if mode not in {"none", "verified_global_mask"}:
        raise ConditionalConfigurationError("protection_context.mode must be none or verified_global_mask")
    if mode == "verified_global_mask":
        required_context = {
            "source_bundle", "artifact", "artifact_sha256", "scientific_payload_sha256",
            "mask_key", "expected_population_size", "expected_global_cardinality",
        }
        missing_context = required_context - set(raw_context)
        if missing_context:
            raise ConditionalConfigurationError(
                f"verified protection context missing fields: {sorted(missing_context)}"
            )
    context = ProtectionContextConfiguration(
        mode=mode,
        source_bundle=(
            _path(str(raw_context["source_bundle"]), source)
            if raw_context.get("source_bundle") else None
        ),
        artifact=str(raw_context.get("artifact", "artifacts/protection_masks.json")),
        artifact_sha256=str(raw_context.get("artifact_sha256", "")),
        scientific_payload_sha256=str(raw_context.get("scientific_payload_sha256", "")),
        mask_key=str(raw_context.get("mask_key", "")),
        expected_population_size=int(raw_context.get("expected_population_size", 0)),
        expected_global_cardinality=int(raw_context.get("expected_global_cardinality", 0)),
    )
    if (
        mode == "verified_global_mask" and check_inputs
        and (context.source_bundle is None or not context.source_bundle.is_dir())
    ):
        raise ConditionalConfigurationError(
            f"protection-context source bundle is unavailable: {context.source_bundle}"
        )
    if mode == "verified_global_mask" and oracle.budget < 1:
        raise ConditionalConfigurationError("global-mask exchange requires a positive restricted budget")

    raw_paired = document.get("paired_swap_evaluation", {})
    if not isinstance(raw_paired, dict):
        raise ConditionalConfigurationError("paired_swap_evaluation must be a table")
    if schema_version == CONDITIONAL_SCHEMA_VERSION_6_3:
        _keys(
            raw_paired, {"enabled", "background_count", "seed", "comparisons"}, set(),
            "paired_swap_evaluation",
        )
    elif raw_paired:
        raise ConditionalConfigurationError("paired_swap_evaluation requires schema 6.3")
    paired = PairedSwapEvaluationConfiguration(
        bool(raw_paired.get("enabled", False)),
        int(raw_paired.get("background_count", 0)),
        int(raw_paired.get("seed", 0)),
        tuple(str(value) for value in raw_paired.get("comparisons", ())),
    )
    if schema_version == CONDITIONAL_SCHEMA_VERSION_6_3:
        if not paired.enabled or paired.background_count < 1:
            raise ConditionalConfigurationError(
                "schema 6.3 requires enabled paired-swap evaluation with positive backgrounds"
            )
        if not paired.comparisons or len(set(paired.comparisons)) != len(paired.comparisons):
            raise ConditionalConfigurationError("paired-swap comparisons must be nonempty and unique")
        unsupported = set(paired.comparisons) - set(PAIRED_SWAP_COMPARISONS)
        if unsupported:
            raise ConditionalConfigurationError(
                f"unsupported paired-swap comparisons: {sorted(unsupported)}"
            )
        independent_seeds = {
            backgrounds.calibration_seed, backgrounds.evaluation_seed,
            oracle.evaluation_seed, *oracle.selection_seeds,
        }
        if paired.seed in independent_seeds:
            raise ConditionalConfigurationError(
                "paired-swap final seed must be independent of all selection/reference seeds"
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
        schema_version, expected_protocol,
        str(experiment["name"]), str(experiment["description"]), str(experiment["objective"]),
        benchmark, samples, str(raw_risk["metric"]), str(raw_risk["direction"]),
        tuple(candidates), backgrounds, execution, oracle, output, source, text, context, paired,
    )
