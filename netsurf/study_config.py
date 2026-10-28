"""Strict schema-3 configuration for M3 empirical and policy studies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
import tomllib
from typing import Any, Mapping

from .experiment_config import BenchmarkConfiguration, FaultConfiguration, OutputConfiguration
from .paths import workspace_root
from .quantization import FixedPointFormat


STUDY_SCHEMA_VERSION = "3.1.0"
SUPPORTED_STUDY_SCHEMA_VERSIONS = {"3.0.0", STUDY_SCHEMA_VERSION}
SUPPORTED_PROTOCOLS = {
    "single-bit-susceptibility/v1": "single_bit",
    "estimator-validation/v1": "estimator_validation",
    "ranking-policy-comparison/v1": "protection",
    "multi-coverage-protection/v1": "protection",
}
SUPPORTED_ESTIMATORS = {
    "msb_first",
    "random",
    "gradient_magnitude",
    "first_order_gradient_delta",
    "per_sample_first_order_magnitude",
    "per_sample_positive_damage",
    "exact_diagonal_second_order",
    "hutchinson_diagonal_second_order",
    "legacy_qpolargrad",
    "clean_activation_qpolargrad",
    "uncancelled_activation_drift",
}
ESTIMATOR_IMPLEMENTATION_VERSIONS = {
    "msb_first": "encoded-bit-index/v1",
    "random": "seeded-uniform-score/v1",
    "gradient_magnitude": "absolute-mean-loss-gradient/v2",
    "first_order_gradient_delta": "signed-gradient-times-exact-delta/v2",
    "per_sample_first_order_magnitude": "mean-absolute-per-example-gradient-times-exact-delta/v1",
    "per_sample_positive_damage": "mean-positive-per-example-gradient-times-exact-delta/v1",
    "exact_diagonal_second_order": "finite-difference-exact-diagonal/v1",
    "hutchinson_diagonal_second_order": "rademacher-hvp-diagonal/v1",
    "legacy_qpolargrad": "recovered-dense-first-call/v1",
    "clean_activation_qpolargrad": "clean-dense-mean-cancelled/v1",
    "uncancelled_activation_drift": "clean-dense-mean-absolute/v1",
}


class StudyConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class StudySamples:
    ranking_count: int
    empirical_count: int
    evaluation_count: int
    require_disjoint: bool


@dataclass(frozen=True)
class CandidateConfiguration:
    strategy: str
    count: int
    seed: int
    score_estimator: str | None
    score_quantiles: int


@dataclass(frozen=True)
class StudyEstimator:
    name: str
    implementation_version: str
    seed: int
    probes: int
    result_name: str


@dataclass(frozen=True)
class StudyPolicy:
    name: str
    estimator: str
    priority: tuple[str, ...]
    seed: int | None
    prespecified: bool


@dataclass(frozen=True)
class StudyProtection:
    coverages: tuple[float, ...]
    mitigation_model: str
    probabilities: tuple[float, ...] = ()


@dataclass(frozen=True)
class StudyExecution:
    trials: int
    confidence: float
    fault_seed: int
    permutation_seed: int


@dataclass(frozen=True)
class StudyReplications:
    count: int
    candidate_seed_stride: int
    hutchinson_seed_stride: int
    sample_offset_stride: int
    fault_seed_stride: int


@dataclass(frozen=True)
class StudyConfiguration:
    schema_version: str
    scientific_protocol: str
    study_type: str
    name: str
    description: str
    objective: str
    benchmark: BenchmarkConfiguration
    samples: StudySamples
    candidates: CandidateConfiguration
    estimators: tuple[StudyEstimator, ...]
    policies: tuple[StudyPolicy, ...]
    protection: StudyProtection
    fault: FaultConfiguration
    execution: StudyExecution
    replications: StudyReplications
    output: OutputConfiguration
    source_path: Path
    source_text: str

    def resolved_document(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scientific_protocol": self.scientific_protocol,
            "study_type": self.study_type,
            "source_configuration": str(self.source_path),
            "experiment": {"name": self.name, "description": self.description, "objective": self.objective},
            "benchmark": {
                **asdict(self.benchmark),
                "checkpoint": str(self.benchmark.checkpoint),
                "dataset": str(self.benchmark.dataset),
                "asset_lock": (
                    None
                    if self.benchmark.asset_lock is None
                    else str(self.benchmark.asset_lock)
                ),
            },
            "samples": asdict(self.samples),
            "candidates": asdict(self.candidates),
            "estimators": [asdict(value) for value in self.estimators],
            "policies": [asdict(value) for value in self.policies],
            "protection": asdict(self.protection),
            "fault": asdict(self.fault),
            "execution": asdict(self.execution),
            "replications": asdict(self.replications),
            "output": {"root": str(self.output.root), "report": self.output.report, "plots": self.output.plots},
        }


def _table(document: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = document.get(name)
    if not isinstance(value, dict):
        raise StudyConfigurationError(f"{name} must be a table")
    return value


def _keys(value: Mapping[str, Any], required: set[str], optional: set[str], name: str):
    missing, unknown = required - set(value), set(value) - required - optional
    if missing:
        raise StudyConfigurationError(f"{name} missing fields: {sorted(missing)}")
    if unknown:
        raise StudyConfigurationError(f"{name} unsupported fields: {sorted(unknown)}")


def _path(value: str, source: Path) -> Path:
    path = Path(value).expanduser()
    return (source.parent / path if not path.is_absolute() else path).resolve()


def load_study(path: str | Path, *, check_inputs: bool = True) -> StudyConfiguration:
    source = Path(path).expanduser().resolve()
    try:
        text = source.read_text(encoding="utf-8")
        document = tomllib.loads(text)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise StudyConfigurationError(f"cannot load study configuration {source}: {error}") from error
    schema_version = document.get("schema_version")
    if schema_version not in SUPPORTED_STUDY_SCHEMA_VERSIONS:
        raise StudyConfigurationError(f"unsupported schema-3 study version: {schema_version!r}")
    protocol = document.get("scientific_protocol")
    if protocol not in SUPPORTED_PROTOCOLS:
        raise StudyConfigurationError(f"unsupported schema-3 protocol: {protocol!r}")
    optional_root = {"replications"} if schema_version == "3.1.0" else set()
    _keys(document, {"schema_version", "scientific_protocol", "experiment", "benchmark", "samples", "candidates", "estimators", "policies", "protection", "fault", "execution", "output"}, optional_root, "root")

    experiment = _table(document, "experiment")
    _keys(experiment, {"name", "description", "objective"}, set(), "experiment")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", str(experiment["name"])):
        raise StudyConfigurationError("experiment.name must be filesystem-safe")

    benchmark = _table(document, "benchmark")
    benchmark_required = {
        "adapter", "id", "checkpoint", "dataset", "split", "quantization",
        "quantization_backend", "injectable_parameters",
    }
    benchmark_optional = (
        {"asset_lock", "inference_backend"}
        if benchmark.get("adapter") == "mlperf_tiny_ic_int8_v1"
        else set()
    )
    _keys(benchmark, benchmark_required, benchmark_optional, "benchmark")
    if benchmark["adapter"] not in {
        "numpy_reference",
        "numpy_activation_cancellation",
        "mnist_quantized",
        "mnist_lite_exact_kb_v1",
        "mlperf_tiny_ic_int8_v1",
    }:
        raise StudyConfigurationError("unsupported schema-3 benchmark adapter")
    checkpoint, dataset = _path(str(benchmark["checkpoint"]), source), _path(str(benchmark["dataset"]), source)
    asset_lock = (
        _path(str(benchmark["asset_lock"]), source)
        if benchmark.get("asset_lock") is not None
        else None
    )
    if check_inputs and (
        not checkpoint.is_file()
        or not dataset.is_file()
        or (asset_lock is not None and not asset_lock.is_file())
    ):
        raise StudyConfigurationError(f"missing checkpoint or dataset: {checkpoint}, {dataset}")
    affine_tflite = benchmark["adapter"] == "mlperf_tiny_ic_int8_v1"
    if affine_tflite:
        if asset_lock is None:
            raise StudyConfigurationError("mlperf_tiny_ic_int8_v1 requires benchmark.asset_lock")
        if benchmark.get("inference_backend") != "tensorflow_lite_builtin_ref_v1":
            raise StudyConfigurationError(
                "mlperf_tiny_ic_int8_v1 requires "
                "inference_backend='tensorflow_lite_builtin_ref_v1'"
            )
        if benchmark["split"] not in {"official_subset_200", "test"}:
            raise StudyConfigurationError(
                "mlperf_tiny_ic_int8_v1 split must be 'official_subset_200' or 'test'"
            )
        if str(benchmark["quantization"]) != "tflite_affine_int8":
            raise StudyConfigurationError(
                "mlperf_tiny_ic_int8_v1 requires quantization='tflite_affine_int8'"
            )
        if str(benchmark["quantization_backend"]) != "tflite_affine_integer_v1":
            raise StudyConfigurationError(
                "mlperf_tiny_ic_int8_v1 requires "
                "quantization_backend='tflite_affine_integer_v1'"
            )
    else:
        FixedPointFormat.parse(str(benchmark["quantization"]))
    injectables = benchmark["injectable_parameters"]
    if not isinstance(injectables, list) or not injectables:
        raise StudyConfigurationError("benchmark.injectable_parameters must be nonempty")
    if benchmark["adapter"] == "mnist_lite_exact_kb_v1" and set(injectables) != {"kernels", "biases"}:
        raise StudyConfigurationError(
            "mnist_lite_exact_kb_v1 requires injectable_parameters=['kernels', 'biases']; "
            "alpha/beta remain uncorrupted float32 auxiliaries"
        )
    if benchmark["adapter"] == "mnist_lite_exact_kb_v1":
        if str(benchmark["quantization"]) != "q<6,2,1>":
            raise StudyConfigurationError("mnist_lite_exact_kb_v1 requires quantization='q<6,2,1>'")
        if str(benchmark["quantization_backend"]) != "netsurf_bitcodec_v1":
            raise StudyConfigurationError(
                "mnist_lite_exact_kb_v1 requires quantization_backend='netsurf_bitcodec_v1'"
            )
    if affine_tflite and tuple(injectables) not in {
        ("kernels",),
        ("kernels", "biases"),
    }:
        raise StudyConfigurationError(
            "mlperf_tiny_ic_int8_v1 injectable_parameters must be ['kernels'] or "
            "['kernels', 'biases']"
        )

    samples = _table(document, "samples")
    _keys(samples, {"ranking_count", "empirical_count", "evaluation_count", "require_disjoint"}, set(), "samples")
    for key in ("ranking_count", "empirical_count", "evaluation_count"):
        if not isinstance(samples[key], int) or samples[key] < 1:
            raise StudyConfigurationError(f"samples.{key} must be positive")
    if not isinstance(samples["require_disjoint"], bool):
        raise StudyConfigurationError("samples.require_disjoint must be boolean")

    candidates = _table(document, "candidates")
    _keys(candidates, {"strategy", "count", "seed", "score_quantiles"}, {"score_estimator"}, "candidates")
    if candidates["strategy"] not in {"all", "uniform", "stratified"}:
        raise StudyConfigurationError("candidates.strategy must be all, uniform, or stratified")
    if not isinstance(candidates["count"], int) or candidates["count"] < 1:
        raise StudyConfigurationError("candidates.count must be positive")
    if not isinstance(candidates["score_quantiles"], int) or candidates["score_quantiles"] < 1:
        raise StudyConfigurationError("candidates.score_quantiles must be positive")
    score_estimator = candidates.get("score_estimator")
    if schema_version == "3.0.0" and candidates["strategy"] == "uniform":
        raise StudyConfigurationError("uniform candidate sampling requires schema_version='3.1.0'")

    raw_estimators = document["estimators"]
    if not isinstance(raw_estimators, list) or not raw_estimators:
        raise StudyConfigurationError("estimators must be a nonempty array")
    estimators = []
    for index, value in enumerate(raw_estimators):
        if not isinstance(value, dict):
            raise StudyConfigurationError(f"estimators[{index}] must be a table")
        _keys(value, {"name", "implementation_version"}, {"seed", "probes", "result_name"} if schema_version == "3.1.0" else {"seed", "probes"}, f"estimators[{index}]")
        if value["name"] not in SUPPORTED_ESTIMATORS:
            raise StudyConfigurationError(f"unsupported estimator: {value['name']}")
        if affine_tflite and value["name"] not in {"msb_first", "random"}:
            raise StudyConfigurationError(
                f"estimator {value['name']!r} is blocked for mlperf_tiny_ic_int8_v1: "
                "no validated differentiable-to-TFLite parameter mapping exists"
            )
        if schema_version == "3.0.0" and value["name"] == "per_sample_positive_damage":
            raise StudyConfigurationError("per_sample_positive_damage requires schema_version='3.1.0'")
        expected_version = (
            "relative-word-bit-index/v1"
            if affine_tflite and value["name"] == "msb_first"
            else ESTIMATOR_IMPLEMENTATION_VERSIONS[value["name"]]
        )
        if value["implementation_version"] != expected_version:
            raise StudyConfigurationError(
                f"estimator {value['name']} requires implementation_version={expected_version!r}"
            )
        if int(value.get("probes", 16)) < 1:
            raise StudyConfigurationError("estimator probes must be positive")
        if value["name"] == "exact_diagonal_second_order" and benchmark["adapter"] not in {"numpy_reference", "numpy_activation_cancellation"}:
            raise StudyConfigurationError("exact diagonal Hessian is restricted to NumPy reference adapters")
        result_name = str(value.get("result_name", value["name"]))
        estimators.append(StudyEstimator(str(value["name"]), str(value["implementation_version"]), int(value.get("seed", 0)), int(value.get("probes", 16)), result_name))
    estimator_names = {value.result_name for value in estimators}
    if len(estimator_names) != len(estimators):
        raise StudyConfigurationError("estimator result_name values must be unique")
    if score_estimator is not None and score_estimator not in estimator_names:
        raise StudyConfigurationError("candidate score_estimator must be one of the configured estimators")

    raw_policies = document["policies"]
    if not isinstance(raw_policies, list):
        raise StudyConfigurationError("policies must be an array")
    policies = []
    for index, value in enumerate(raw_policies):
        if not isinstance(value, dict):
            raise StudyConfigurationError(f"policies[{index}] must be a table")
        _keys(value, {"name", "estimator", "priority", "prespecified"}, {"seed"}, f"policies[{index}]")
        if value["estimator"] not in estimator_names:
            raise StudyConfigurationError(f"policy estimator is not configured: {value['estimator']}")
        if not isinstance(value["priority"], list) or not value["priority"]:
            raise StudyConfigurationError("policy.priority must be a nonempty string array")
        supported_criteria = {"bit_significance_desc", "estimator_score_desc", "canonical", "reverse_canonical", "random", "layer_input_first", "layer_output_first", "stable_identity", "ud", "du"}
        unknown_criteria = set(map(str, value["priority"])) - supported_criteria
        if unknown_criteria:
            raise StudyConfigurationError(f"unsupported ordering criteria: {sorted(unknown_criteria)}")
        if affine_tflite and "bit_significance_desc" in value["priority"]:
            raise StudyConfigurationError(
                "mixed-width TFLite storage must order msb_first with "
                "priority=['estimator_score_desc', ...], not raw bit_significance_desc"
            )
        if "random" in value["priority"] and "seed" not in value:
            raise StudyConfigurationError("random ordering criterion requires policy seed")
        policies.append(StudyPolicy(str(value["name"]), str(value["estimator"]), tuple(map(str, value["priority"])), None if "seed" not in value else int(value["seed"]), bool(value["prespecified"])))
    if len({value.name for value in policies}) != len(policies):
        raise StudyConfigurationError("policy names must be unique")
    if SUPPORTED_PROTOCOLS[protocol] == "protection" and not policies:
        raise StudyConfigurationError("protection protocols require at least one policy")

    protection = _table(document, "protection")
    _keys(protection, {"coverages", "mitigation_model"}, {"probabilities"} if schema_version == "3.1.0" else set(), "protection")
    coverages = tuple(float(value) for value in protection["coverages"])
    if SUPPORTED_PROTOCOLS[protocol] == "protection" and (not coverages or any(not 0 <= value <= 1 for value in coverages)):
        raise StudyConfigurationError("protection protocols require coverages in [0,1]")
    if protection["mitigation_model"] != "perfect_event_suppression":
        raise StudyConfigurationError("only perfect_event_suppression is implemented")
    probabilities = tuple(float(value) for value in protection.get("probabilities", []))
    if any(not 0.0 <= value <= 1.0 for value in probabilities):
        raise StudyConfigurationError("protection.probabilities must be in [0,1]")

    fault = _table(document, "fault")
    _keys(fault, {"target", "location", "persistence", "distribution", "probability"}, set(), "fault")
    if (fault["target"], fault["location"], fault["distribution"]) != ("weight", "logical_trainable_parameter_bits", "independent_bernoulli"):
        raise StudyConfigurationError("unsupported fault semantics")
    if fault["persistence"] not in {"persistent_trial", "per_inference"}:
        raise StudyConfigurationError("unsupported fault persistence")
    if not 0.0 <= float(fault["probability"]) <= 1.0:
        raise StudyConfigurationError("fault.probability must be in [0,1]")

    execution = _table(document, "execution")
    _keys(execution, {"trials", "confidence", "fault_seed", "permutation_seed"}, set(), "execution")
    if int(execution["trials"]) < 1 or not 0 < float(execution["confidence"]) < 1:
        raise StudyConfigurationError("invalid execution trials or confidence")

    raw_replications = document.get("replications", {})
    if not isinstance(raw_replications, dict):
        raise StudyConfigurationError("replications must be a table")
    if raw_replications:
        _keys(raw_replications, {"count", "candidate_seed_stride", "hutchinson_seed_stride", "sample_offset_stride", "fault_seed_stride"}, set(), "replications")
    replications = StudyReplications(
        int(raw_replications.get("count", 1)), int(raw_replications.get("candidate_seed_stride", 0)),
        int(raw_replications.get("hutchinson_seed_stride", 0)), int(raw_replications.get("sample_offset_stride", 0)),
        int(raw_replications.get("fault_seed_stride", 0)),
    )
    if replications.count < 1 or any(value < 0 for value in asdict(replications).values()):
        raise StudyConfigurationError("replication count must be positive and strides nonnegative")

    output = _table(document, "output")
    _keys(output, {"report", "plots"}, {"root"}, "output")
    root = workspace_root() / "runs" if output.get("root", "") == "" else _path(str(output["root"]), source)

    return StudyConfiguration(
        str(schema_version), str(protocol), SUPPORTED_PROTOCOLS[protocol], str(experiment["name"]),
        str(experiment["description"]), str(experiment["objective"]),
        BenchmarkConfiguration(
            str(benchmark["adapter"]), str(benchmark["id"]), checkpoint, dataset,
            str(benchmark["split"]), int(samples["evaluation_count"]),
            int(samples["ranking_count"]), str(benchmark["quantization"]),
            str(benchmark["quantization_backend"]), tuple(map(str, injectables)),
            asset_lock,
            None if benchmark.get("inference_backend") is None else str(benchmark["inference_backend"]),
        ),
        StudySamples(int(samples["ranking_count"]), int(samples["empirical_count"]), int(samples["evaluation_count"]), bool(samples["require_disjoint"])),
        CandidateConfiguration(str(candidates["strategy"]), int(candidates["count"]), int(candidates["seed"]), None if score_estimator is None else str(score_estimator), int(candidates["score_quantiles"])),
        tuple(estimators), tuple(policies), StudyProtection(coverages, str(protection["mitigation_model"]), probabilities),
        FaultConfiguration(str(fault["target"]), str(fault["location"]), str(fault["persistence"]), str(fault["distribution"]), float(fault["probability"])),
        StudyExecution(int(execution["trials"]), float(execution["confidence"]), int(execution["fault_seed"]), int(execution["permutation_seed"])),
        replications,
        OutputConfiguration(root.resolve(), bool(output["report"]), bool(output["plots"])), source, text,
    )
