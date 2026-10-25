"""Strict versioned configuration for M4 conditional protection search."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
import tomllib
from typing import Any, Mapping

from .experiment_config import BenchmarkConfiguration, FaultConfiguration, OutputConfiguration
from .paths import workspace_root
from .quantization import FixedPointFormat
from .study_config import ESTIMATOR_IMPLEMENTATION_VERSIONS, SUPPORTED_ESTIMATORS, StudyEstimator


SEARCH_SCHEMA_VERSION = "4.0.0"
SEARCH_PROTOCOL = "conditional-protection-search/v1"
CONTROLLED_SEARCH_SCHEMA_VERSION = "4.1.0"
CONTROLLED_SEARCH_PROTOCOL = "conditional-protection-search/v2"


class SearchConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class SearchSamples:
    ranking_count: int
    search_count: int
    final_count: int
    require_disjoint: bool


@dataclass(frozen=True)
class GuidanceConfiguration:
    name: str
    estimator: str
    priority: tuple[str, ...]
    weight: float
    seed: int | None


@dataclass(frozen=True)
class InitialMaskConfiguration:
    guidance: str
    budget_bits: int


@dataclass(frozen=True)
class SearchMethodConfiguration:
    name: str
    strategy: str
    guidance: tuple[str, ...]


@dataclass(frozen=True)
class ConditionalSearchConfiguration:
    strategies: tuple[str, ...]
    evaluator: str
    iterations: int
    candidate_pool_strategy: str
    candidate_pool_count: int
    uniform_exploration: float
    rank_temperature: float
    background_contexts: int
    seed: int
    greedy_add_candidates: int
    greedy_remove_candidates: int
    initial_temperature: float
    cooling_rate: float
    cache_memory_mb: float
    exact_enumeration: bool


@dataclass(frozen=True)
class PolarConfiguration:
    enabled: bool
    mode: str
    pair_budget: int
    background_contexts: int
    seed: int
    strength: float
    collect_cancellation_diagnostics: bool


@dataclass(frozen=True)
class FinalEvaluationConfiguration:
    contexts: int
    seed: int
    confidence: float
    conditional_contexts: int
    conditional_seed: int


@dataclass(frozen=True)
class SearchReplications:
    count: int
    search_seed_stride: int
    final_seed_stride: int
    sample_offset_stride: int


@dataclass(frozen=True)
class SearchConfiguration:
    schema_version: str
    scientific_protocol: str
    name: str
    description: str
    objective: str
    benchmark: BenchmarkConfiguration
    samples: SearchSamples
    estimators: tuple[StudyEstimator, ...]
    guidance: tuple[GuidanceConfiguration, ...]
    initial_mask: InitialMaskConfiguration
    methods: tuple[SearchMethodConfiguration, ...]
    search: ConditionalSearchConfiguration
    polar: PolarConfiguration
    fault: FaultConfiguration
    fault_probabilities: tuple[float, ...]
    final_evaluation: FinalEvaluationConfiguration
    replications: SearchReplications
    output: OutputConfiguration
    source_path: Path
    source_text: str

    def resolved_document(self) -> dict[str, Any]:
        controlled = self.schema_version == CONTROLLED_SEARCH_SCHEMA_VERSION
        fault = asdict(self.fault)
        if controlled:
            fault.pop("probability", None); fault["probabilities"] = list(self.fault_probabilities)
        final = asdict(self.final_evaluation)
        if controlled:
            final = {"natural_contexts": final["contexts"], "natural_seed": final["seed"],
                     "conditional_contexts": final["conditional_contexts"], "conditional_seed": final["conditional_seed"],
                     "confidence": final["confidence"]}
        else:
            final = {"contexts": final["contexts"], "seed": final["seed"], "confidence": final["confidence"]}
        return {
            "schema_version": self.schema_version, "scientific_protocol": self.scientific_protocol,
            "study_type": "conditional_search", "source_configuration": str(self.source_path),
            "experiment": {"name": self.name, "description": self.description, "objective": self.objective},
            "benchmark": {**asdict(self.benchmark), "checkpoint": str(self.benchmark.checkpoint), "dataset": str(self.benchmark.dataset)},
            "samples": asdict(self.samples), "estimators": [asdict(value) for value in self.estimators],
            "guidance": [asdict(value) for value in self.guidance], "initial_mask": asdict(self.initial_mask),
            **({"methods": [asdict(value) for value in self.methods]} if controlled else {}),
            "search": {key:value for key,value in asdict(self.search).items() if not controlled or key != "strategies"},
            "polar": asdict(self.polar), "fault": fault,
            "final_evaluation": final, "replications": asdict(self.replications),
            "output": {"root": str(self.output.root), "report": self.output.report, "plots": self.output.plots},
        }


def _table(document: Mapping[str, Any], name: str):
    value = document.get(name)
    if not isinstance(value, dict):
        raise SearchConfigurationError(f"{name} must be a table")
    return value


def _keys(value, required, optional, name):
    missing, unknown = set(required) - set(value), set(value) - set(required) - set(optional)
    if missing:
        raise SearchConfigurationError(f"{name} missing fields: {sorted(missing)}")
    if unknown:
        raise SearchConfigurationError(f"{name} unsupported fields: {sorted(unknown)}")


def _path(value: str, source: Path) -> Path:
    path = Path(value).expanduser()
    return (source.parent / path if not path.is_absolute() else path).resolve()


def load_search(path: str | Path, *, check_inputs: bool = True) -> SearchConfiguration:
    source = Path(path).expanduser().resolve()
    try:
        text = source.read_text(encoding="utf-8")
        document = tomllib.loads(text)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise SearchConfigurationError(f"cannot load search configuration {source}: {error}") from error
    schema_version = document.get("schema_version")
    controlled = schema_version == CONTROLLED_SEARCH_SCHEMA_VERSION
    if schema_version not in {SEARCH_SCHEMA_VERSION, CONTROLLED_SEARCH_SCHEMA_VERSION}:
        raise SearchConfigurationError(f"unsupported conditional-search schema: {document.get('schema_version')!r}")
    expected_protocol = CONTROLLED_SEARCH_PROTOCOL if controlled else SEARCH_PROTOCOL
    if document.get("scientific_protocol") != expected_protocol:
        raise SearchConfigurationError(f"unsupported conditional-search protocol: {document.get('scientific_protocol')!r}")
    required = {"schema_version", "scientific_protocol", "experiment", "benchmark", "samples", "estimators", "guidance", "initial_mask", "search", "polar", "fault", "final_evaluation", "replications", "output"}
    if controlled:
        required.add("methods")
    _keys(document, required, set(), "root")
    experiment = _table(document, "experiment"); _keys(experiment, {"name", "description", "objective"}, set(), "experiment")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", str(experiment["name"])):
        raise SearchConfigurationError("experiment.name must be filesystem-safe")
    benchmark = _table(document, "benchmark")
    _keys(benchmark, {"adapter", "id", "checkpoint", "dataset", "split", "quantization", "quantization_backend", "injectable_parameters"}, set(), "benchmark")
    if benchmark["adapter"] not in {"numpy_reference", "numpy_activation_cancellation", "mnist_quantized"}:
        raise SearchConfigurationError("M4 exact incremental evaluation supports NumPy reference and mnist_quantized adapters")
    checkpoint, dataset = _path(str(benchmark["checkpoint"]), source), _path(str(benchmark["dataset"]), source)
    if check_inputs and (not checkpoint.is_file() or not dataset.is_file()):
        raise SearchConfigurationError(f"missing checkpoint or dataset: {checkpoint}, {dataset}")
    FixedPointFormat.parse(str(benchmark["quantization"]))
    injectables = benchmark["injectable_parameters"]
    if not isinstance(injectables, list) or not injectables:
        raise SearchConfigurationError("benchmark.injectable_parameters must be nonempty")
    samples = _table(document, "samples"); _keys(samples, {"ranking_count", "search_count", "final_count", "require_disjoint"}, set(), "samples")
    if any(int(samples[name]) < 1 for name in ("ranking_count", "search_count", "final_count")):
        raise SearchConfigurationError("sample counts must be positive")
    estimators_raw = document["estimators"]
    if not isinstance(estimators_raw, list) or not estimators_raw:
        raise SearchConfigurationError("estimators must be a nonempty array")
    estimators = []
    for index, value in enumerate(estimators_raw):
        _keys(value, {"name", "implementation_version"}, {"seed", "probes", "result_name"}, f"estimators[{index}]")
        name = str(value["name"])
        if name not in SUPPORTED_ESTIMATORS or value["implementation_version"] != ESTIMATOR_IMPLEMENTATION_VERSIONS.get(name):
            raise SearchConfigurationError(f"unsupported estimator/version: {name}/{value['implementation_version']}")
        if int(value.get("probes", 16)) < 1:
            raise SearchConfigurationError("estimator probes must be positive")
        if name == "exact_diagonal_second_order" and benchmark["adapter"] not in {"numpy_reference", "numpy_activation_cancellation"}:
            raise SearchConfigurationError("exact diagonal Hessian is restricted to NumPy reference adapters")
        result_name = str(value.get("result_name", name))
        estimators.append(StudyEstimator(name, str(value["implementation_version"]), int(value.get("seed", 0)), int(value.get("probes", 16)), result_name))
    estimator_names = {value.result_name for value in estimators}
    if len(estimator_names) != len(estimators):
        raise SearchConfigurationError("estimator result_name values must be unique")
    guidance_raw = document["guidance"]
    if not isinstance(guidance_raw, list) or not guidance_raw:
        raise SearchConfigurationError("guidance must be a nonempty array")
    guidance = []
    supported_criteria = {"bit_significance_desc", "estimator_score_desc", "canonical", "reverse_canonical", "random", "layer_input_first", "layer_output_first", "stable_identity", "ud", "du"}
    for index, value in enumerate(guidance_raw):
        _keys(value, {"name", "estimator", "priority", "weight"}, {"seed"}, f"guidance[{index}]")
        if value["estimator"] not in estimator_names:
            raise SearchConfigurationError(f"guidance estimator is not configured: {value['estimator']}")
        priority = tuple(map(str, value["priority"]))
        if not priority or set(priority) - supported_criteria:
            raise SearchConfigurationError(f"unsupported guidance priority: {priority}")
        if "random" in priority and "seed" not in value:
            raise SearchConfigurationError("random guidance requires seed")
        if float(value["weight"]) < 0:
            raise SearchConfigurationError("guidance weights must be nonnegative")
        guidance.append(GuidanceConfiguration(str(value["name"]), str(value["estimator"]), priority, float(value["weight"]), None if "seed" not in value else int(value["seed"])))
    if len({value.name for value in guidance}) != len(guidance):
        raise SearchConfigurationError("guidance names must be unique")
    if sum(value.weight for value in guidance) <= 0:
        raise SearchConfigurationError("at least one guidance weight must be positive")
    initial = _table(document, "initial_mask"); _keys(initial, {"guidance", "budget_bits"}, set(), "initial_mask")
    if initial["guidance"] not in {value.name for value in guidance} or int(initial["budget_bits"]) < 1:
        raise SearchConfigurationError("initial mask requires known guidance and positive budget_bits")
    search = _table(document, "search")
    search_fields = {"evaluator", "iterations", "candidate_pool_strategy", "candidate_pool_count", "uniform_exploration", "rank_temperature", "background_contexts", "seed", "greedy_add_candidates", "greedy_remove_candidates", "initial_temperature", "cooling_rate", "cache_memory_mb", "exact_enumeration"}
    if not controlled:
        search_fields.add("strategies")
    _keys(search, search_fields, set(), "search")
    supported_strategies = {"static", "conditional_greedy", "uniform_swap", "static_guided_swap", "polar_guided_swap", "simulated_annealing"}
    guidance_names = {value.name for value in guidance}
    if controlled:
        methods_raw = document["methods"]
        if not isinstance(methods_raw, list) or not methods_raw:
            raise SearchConfigurationError("methods must be a nonempty array")
        methods = []
        for index, value in enumerate(methods_raw):
            _keys(value, {"name", "strategy", "guidance"}, set(), f"methods[{index}]")
            method_guidance = tuple(map(str, value["guidance"]))
            if value["strategy"] not in supported_strategies:
                raise SearchConfigurationError(f"unsupported method strategy: {value['strategy']}")
            if not method_guidance or set(method_guidance)-guidance_names:
                raise SearchConfigurationError(f"method references unsupported guidance: {method_guidance}")
            methods.append(SearchMethodConfiguration(str(value["name"]),str(value["strategy"]),method_guidance))
        if len({value.name for value in methods}) != len(methods):
            raise SearchConfigurationError("method names must be unique")
        strategies = tuple(value.strategy for value in methods)
    else:
        strategies = tuple(map(str, search["strategies"]))
        if not strategies or set(strategies) - supported_strategies:
            raise SearchConfigurationError(f"unsupported search strategies: {sorted(set(strategies)-supported_strategies)}")
        if len(set(strategies)) != len(strategies):
            raise SearchConfigurationError("search strategies must be unique")
        methods = [SearchMethodConfiguration(value,value,tuple(guidance_names)) for value in strategies]
    if search["evaluator"] not in {"incremental", "full"}:
        raise SearchConfigurationError("search.evaluator must be incremental or full")
    if search["candidate_pool_strategy"] not in {"all", "top_union", "uniform"}:
        raise SearchConfigurationError("unsupported candidate_pool_strategy")
    if int(search["candidate_pool_count"]) < int(initial["budget_bits"]):
        raise SearchConfigurationError("candidate pool cannot be smaller than protection budget")
    if not 0 < float(search["uniform_exploration"]) <= 1:
        raise SearchConfigurationError("uniform_exploration must be in (0,1]")
    if int(search["iterations"]) < 0 or int(search["background_contexts"]) < 1:
        raise SearchConfigurationError("invalid search iteration/context budget")
    if any(value != "static" for value in strategies) and int(search["iterations"]) < 1:
        raise SearchConfigurationError("non-static strategies require at least one iteration")
    if not 0 < float(search["cooling_rate"]) <= 1 or float(search["initial_temperature"]) <= 0:
        raise SearchConfigurationError("invalid annealing schedule")
    polar = _table(document, "polar")
    _keys(polar, {"enabled", "mode", "pair_budget", "background_contexts", "seed", "strength", "collect_cancellation_diagnostics"}, set(), "polar")
    if polar["mode"] not in {"finite_clean", "finite_background", "local_quadratic"}:
        raise SearchConfigurationError("unsupported polar guidance mode")
    if "polar_guided_swap" in strategies and not bool(polar["enabled"]):
        raise SearchConfigurationError("polar_guided_swap requires polar.enabled=true")
    if "polar_guided_swap" in strategies and int(polar["pair_budget"]) < 1:
        raise SearchConfigurationError("polar_guided_swap requires a positive polar pair_budget")
    if int(polar["pair_budget"]) < 0 or int(polar["background_contexts"]) < 1:
        raise SearchConfigurationError("invalid sparse polar budget")
    fault = _table(document, "fault")
    _keys(fault, {"target", "location", "persistence", "distribution", "probabilities" if controlled else "probability"}, set(), "fault")
    if (fault["target"], fault["location"], fault["persistence"], fault["distribution"]) != ("weight", "logical_trainable_parameter_bits", "persistent_trial", "independent_bernoulli"):
        raise SearchConfigurationError("M4 supports persistent independent logical weight-bit faults only")
    probabilities = tuple(map(float, fault["probabilities"])) if controlled else (float(fault["probability"]),)
    if not probabilities or len(set(probabilities)) != len(probabilities) or any(not 0 <= value <= 1 for value in probabilities):
        raise SearchConfigurationError("fault probabilities must be unique values in [0,1]")
    final = _table(document, "final_evaluation")
    _keys(final, {"natural_contexts", "natural_seed", "conditional_contexts", "conditional_seed", "confidence"} if controlled else {"contexts", "seed", "confidence"}, set(), "final_evaluation")
    natural_contexts = int(final["natural_contexts"] if controlled else final["contexts"])
    natural_seed = int(final["natural_seed"] if controlled else final["seed"])
    conditional_contexts = int(final["conditional_contexts"] if controlled else final["contexts"])
    conditional_seed = int(final["conditional_seed"] if controlled else final["seed"])
    if natural_contexts < 1 or conditional_contexts < 1 or not 0 < float(final["confidence"]) < 1:
        raise SearchConfigurationError("invalid final evaluation settings")
    if controlled and natural_seed == conditional_seed:
        raise SearchConfigurationError("natural and conditional final-evaluation seeds must be independent")
    reps = _table(document, "replications"); _keys(reps, {"count", "search_seed_stride", "final_seed_stride", "sample_offset_stride"}, set(), "replications")
    if int(reps["count"]) < 1 or any(int(reps[name]) < 0 for name in ("search_seed_stride", "final_seed_stride", "sample_offset_stride")):
        raise SearchConfigurationError("invalid replication plan")
    output = _table(document, "output"); _keys(output, {"report", "plots"}, {"root"}, "output")
    root = workspace_root() / "runs" if output.get("root", "") == "" else _path(str(output["root"]), source)
    search_config = ConditionalSearchConfiguration(
        strategies, str(search["evaluator"]), int(search["iterations"]), str(search["candidate_pool_strategy"]),
        int(search["candidate_pool_count"]), float(search["uniform_exploration"]), float(search["rank_temperature"]),
        int(search["background_contexts"]), int(search["seed"]), int(search["greedy_add_candidates"]),
        int(search["greedy_remove_candidates"]), float(search["initial_temperature"]), float(search["cooling_rate"]),
        float(search["cache_memory_mb"]), bool(search["exact_enumeration"]),
    )
    if search_config.exact_enumeration and search_config.candidate_pool_count > 20:
        raise SearchConfigurationError("exact enumeration is limited to candidate_pool_count <= 20")
    return SearchConfiguration(
        str(schema_version), expected_protocol, str(experiment["name"]), str(experiment["description"]), str(experiment["objective"]),
        BenchmarkConfiguration(str(benchmark["adapter"]), str(benchmark["id"]), checkpoint, dataset, str(benchmark["split"]), int(samples["final_count"]), int(samples["ranking_count"]), str(benchmark["quantization"]), str(benchmark["quantization_backend"]), tuple(map(str, injectables))),
        SearchSamples(int(samples["ranking_count"]), int(samples["search_count"]), int(samples["final_count"]), bool(samples["require_disjoint"])),
        tuple(estimators), tuple(guidance), InitialMaskConfiguration(str(initial["guidance"]), int(initial["budget_bits"])), tuple(methods),
        search_config, PolarConfiguration(bool(polar["enabled"]), str(polar["mode"]), int(polar["pair_budget"]), int(polar["background_contexts"]), int(polar["seed"]), float(polar["strength"]), bool(polar["collect_cancellation_diagnostics"])),
        FaultConfiguration(str(fault["target"]), str(fault["location"]), str(fault["persistence"]), str(fault["distribution"]), probabilities[0]), probabilities,
        FinalEvaluationConfiguration(natural_contexts, natural_seed, float(final["confidence"]), conditional_contexts, conditional_seed),
        SearchReplications(int(reps["count"]), int(reps["search_seed_stride"]), int(reps["final_seed_stride"]), int(reps["sample_offset_stride"])),
        OutputConfiguration(root.resolve(), bool(output["report"]), bool(output["plots"])), source, text,
    )
