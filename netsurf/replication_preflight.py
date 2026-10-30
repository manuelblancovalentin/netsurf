"""Schema 6.5: independently planned evaluation of the original M956 masks.

No selection, input loading, or random draws occur during plan construction.
The original 6.4 final-only validator is deliberately left unchanged.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import re

from .conditional_config import ConditionalConfigurationError, PAIRED_SWAP_COMPARISONS
from .manifests import digest_object, sha256_file
from .selection_preflight import load_frozen_selection, selection_contract


REFERENCE_PAYLOAD = "6acf2a45452432e64186f52c3ffa6dfd2f298765106c1282ee220ef5823f0ff7"
REFERENCE_ARTIFACT = "675ac2b3a46a1555fb0f4da86a0808c96ca2b4ad961e66b68a4284f97286b046"
MASKS = {"selected": [2, 6], "isolated": [2, 7], "original": [6, 7]}
LEDGER = Path(__file__).parent / "data/kws_replication_exposure_v1.json"
SPLIT = "speech_commands_tfds_test_interleaved_blocks16"


def statistical_contract():
    return {
        "version": "frozen-mask-mean-risk/v1",
        "primary": PAIRED_SWAP_COMPARISONS[0],
        "secondary": list(PAIRED_SWAP_COMPARISONS[1:]),
        "estimand": "target minus baseline expected categorical cross-entropy",
        "independent_unit": "whole-path master Bernoulli fault field",
        "confidence": 0.95,
        "primary_inference": "two-sided Student-t mean CI and unadjusted paired t-test p-value",
        "secondary_inference": "Holm over two secondary two-sided paired t-tests; Bonferroni simultaneous mean CIs",
        "tail_diagnostics": "signed counts; linear empirical quantiles; five worst adverse fields; top-absolute 1/5/10% signed contributions; leave-one-field-out influence",
        "assumptions": "independent identically distributed fields; Student-t inference is approximate for nonnormal heavy-tailed effects",
        "scope": "conditional on fixed checkpoint, exploratory candidate pool, masks and fixed clips; not input-sampling uncertainty",
        "no_outcome_adaptive_changes": True,
    }


def validate_replication_workflow(raw, samples, backgrounds, oracle, paired, context, source):
    required = {"mode", "purpose", "candidate_provenance", "selection_bundle",
                "selection_payload_sha256", "selection_artifact_sha256"}
    optional = {"replication_plan", "replication_plan_sha256", "record_per_input"}
    if not isinstance(raw, dict) or not required <= raw.keys() or raw.keys() - required - optional:
        raise ConditionalConfigurationError("replication workflow requires frozen selection identity; only replication_plan and replication_plan_sha256 are optional")
    if raw["mode"] != "replication_only" or raw["purpose"] not in {"engineering", "confirmation"}:
        raise ConditionalConfigurationError("schema 6.5 requires replication_only and an explicit purpose")
    if raw["candidate_provenance"] != "exploratory_frozen_m953_pool":
        raise ConditionalConfigurationError("replication cannot change candidate provenance")
    if (not paired.enabled or paired.background_count < 2 or paired.seed < 0
            or paired.comparisons != PAIRED_SWAP_COMPARISONS
            or backgrounds.probabilities != (0.01,) or not oracle.enabled or oracle.budget != 2
            or context.mode != "verified_global_mask"
            or context.expected_population_size != 176128 or context.expected_global_cardinality != 22016):
        raise ConditionalConfigurationError("replication requires original BER, budget, population and ordered three-comparison family")
    if raw["selection_payload_sha256"] != REFERENCE_PAYLOAD or raw["selection_artifact_sha256"] != REFERENCE_ARTIFACT:
        raise ConditionalConfigurationError("replication requires the original finalized M956 selection hashes")
    plan_fields = {"replication_plan", "replication_plan_sha256"}
    if bool(plan_fields & raw.keys()) and not plan_fields <= raw.keys():
        raise ConditionalConfigurationError("replication plan path and SHA-256 must be supplied together")
    if "record_per_input" in raw and type(raw["record_per_input"]) is not bool:
        raise ConditionalConfigurationError("record_per_input must be a boolean")
    workflow = dict(raw)
    for key in ("selection_bundle", "replication_plan"):
        if key in workflow:
            path = Path(workflow[key]).expanduser()
            workflow[key] = str((source.parent / path if not path.is_absolute() else path).resolve())
    if "replication_plan_sha256" in workflow and not re.fullmatch("[0-9a-f]{64}", str(workflow["replication_plan_sha256"])):
        raise ConditionalConfigurationError("replication_plan_sha256 must be an exact file SHA-256")
    replication_exposure_audit(samples, raw["purpose"])
    return workflow


def replication_exposure_audit(samples, purpose):
    ledger = json.loads(LEDGER.read_text())
    requested = list(range(3048 + samples.evaluation_offset,
                           3048 + samples.evaluation_offset + samples.evaluation_count))
    exposed = set(ledger["exposed_test_indices"])
    overlap = sorted(set(requested) & exposed)
    if purpose == "confirmation" and overlap:
        raise ConditionalConfigurationError(f"replication final sample exposure overlap: {overlap}")
    return {"ledger_sha256": sha256_file(LEDGER), "audit_scope": ledger["audit_scope"],
            "requested_original_indices": requested, "prior_overlap": overlap,
            "purpose": purpose, "clean_reference_inference_previously_performed": True}


def field_seeds(seed, count):
    return [seed + i * 1_000_003 for i in range(count)]


def verify_new_seeds(configuration, record):
    contract = record["contract"]
    old = set()
    for seed in contract["selection_seeds"]:
        old.update(field_seeds(seed, contract["selection_fields_per_seed"]))
    prior = contract["final_plan"]["paired"]
    old.update(field_seeds(prior["seed"], prior["background_count"]))
    # The audit retains actual seeds from historical engineering and final records too.
    old.update(json.loads(LEDGER.read_text())["historical_field_seeds"])
    new = field_seeds(configuration.paired_swap_evaluation.seed,
                      configuration.paired_swap_evaluation.background_count)
    if len(set(new)) != len(new) or min(new) < 0 or set(new) & old:
        raise ConditionalConfigurationError("new and historical derived field seeds overlap or are invalid")
    return {"new_derived_seeds": new, "historical_derived_seeds_sha256": digest_object(sorted(old)),
            "historical_seed_count": len(old), "overlap": [],
            "interpretation": "disjoint seeded PRNG streams, not a proof of statistical independence"}


def load_replication_selection(configuration):
    """Verify the original contract using its original final plan, not the new one."""
    w = validate_replication_workflow(dict(configuration.workflow), configuration.samples,
        configuration.backgrounds, configuration.restricted_oracle,
        configuration.paired_swap_evaluation, configuration.protection_context, configuration.source_path)
    if configuration.execution.confidence != .95 or len(configuration.candidates) != 8:
        raise ConditionalConfigurationError("replication freezes eight candidates and two-sided 95% inference")
    root = Path(w["selection_bundle"])
    try:
        record = json.loads((root / "artifacts/frozen_selection.json").read_text())
        old = record["contract"]["final_plan"]
        # This calls the unchanged strict 6.4 loader, including all original contract,
        # checksum, candidate, context, population, dataset-lock and model checks.
        original = replace(configuration,
            samples=replace(configuration.samples, evaluation_role=old["role"],
                            evaluation_offset=old["offset"], evaluation_count=old["count"]),
            paired_swap_evaluation=replace(configuration.paired_swap_evaluation,
                background_count=old["paired"]["background_count"], seed=old["paired"]["seed"]),
            workflow={**w, "purpose": old["purpose"]})
        record = load_frozen_selection(original)
        if record["masks"] != [{"logical_ber": .01, **MASKS}]:
            raise ValueError("frozen masks differ from selected (2,6), isolated (2,7), original (6,7)")
        verify_new_seeds(configuration, record)
        return record
    except (OSError, KeyError, ValueError) as error:
        raise ConditionalConfigurationError(f"cannot replicate frozen masks: {error}") from error


def reject_recorded_reuse(configuration):
    """Audit finalized metadata in inventoried roots plus the new output root.

    Failed/manual jobs and concurrent reservations remain an explicit limitation.
    Only small exposure artifacts/manifests are scanned, never prediction files.
    """
    if configuration.workflow["purpose"] != "confirmation":
        return
    ledger = json.loads(LEDGER.read_text())
    requested = set(replication_exposure_audit(configuration.samples, "confirmation")["requested_original_indices"])
    new_seeds = set(field_seeds(configuration.paired_swap_evaluation.seed, configuration.paired_swap_evaluation.background_count))
    roots = {Path(p) for p in ledger["run_roots"]} | {configuration.output.root}
    for root in sorted(roots):
        for path in sorted(root.glob("*/*/manifest.json")):
            if not (path.parent / "FINALIZED").is_file():
                continue
            manifest = json.loads(path.read_text())
            if manifest.get("benchmark", {}).get("adapter") != "mlperf_tiny_kws_int8_v1":
                continue
            overlap = requested & set(manifest.get("m956_final_original_indices", ()))
            if overlap:
                raise ConditionalConfigurationError(f"exposure overlap with finalized bundle {path.parent}: {sorted(overlap)}")
            plan = path.parent / "artifacts/replication_plan.json"
            if plan.is_file():
                previous = json.loads(plan.read_text())
                if new_seeds & set(previous["seed_audit"]["new_derived_seeds"]):
                    raise ConditionalConfigurationError(f"field seeds reused from finalized replication {path.parent}")


def replication_plan(configuration):
    record = load_replication_selection(configuration)
    reject_recorded_reuse(configuration)
    contract = selection_contract(configuration)
    # Keep the original construction identity and label the new final plan separately.
    original_contract = dict(record["contract"])
    original_contract.pop("final_plan")
    current_construction = dict(contract)
    final = current_construction.pop("final_plan")
    if current_construction != original_contract:
        raise ConditionalConfigurationError("replication construction identity differs")
    final["procedure"] = statistical_contract()
    counts = configuration.operation_counts()
    n = configuration.paired_swap_evaluation.background_count
    # All three actual masks differ by one exchange, versus the conservative two-step bound.
    exact_requests = n * 2 * sum(len(set(MASKS[a]) - set(MASKS[b])) for a, b in
                                (("selected", "original"), ("selected", "isolated"), ("isolated", "original")))
    return {
        "schema_version": "1.0.0", "protocol": "independent-frozen-mask-replication/v1",
        "experiment_name": configuration.name,
        "selection_payload_sha256": configuration.workflow["selection_payload_sha256"],
        "selection_artifact_sha256": configuration.workflow["selection_artifact_sha256"],
        "original_frozen_selection_sha256": digest_object(record),
        "construction_contract": original_contract, "frozen_masks": MASKS,
        "eligible_population_sha256": record["eligible_population_sha256"],
        "protection_context_sha256": record["protection_context_sha256"],
        "final_plan": final,
        "exposure_audit": replication_exposure_audit(configuration.samples, configuration.workflow["purpose"]),
        "seed_audit": verify_new_seeds(configuration, record),
        "fault_law": "independent persistent Bernoulli, ideal event suppression; existing exact paired-swap paths",
        "operation_counts": {**counts, "exact_path_requests_before_cache": exact_requests,
            "exact_path_clip_inferences_before_cache": exact_requests * configuration.samples.evaluation_count,
            "clip_inferences_upper_bound": counts["total_requested_evaluations"] * configuration.samples.evaluation_count,
            "setup_reference_clips_additional": 16},
        "per_input_contributions": {
            "enabled": configuration.workflow.get("record_per_input", False),
            "definition": "per-clip p times removed-minus-added CE, summed along each field path; same authoritative calls, scalar primary reduction unchanged",
            "uncertainty": "crossed inputs/fields, not independent field-times-input observations; no input-sampling CI asserted"},
        "publication_scope": "replication of an exploratory frozen eight-bit pool, not population-wide superiority",
    }


def freeze_replication_plan(configuration, destination):
    from .result_bundle import _write_json
    plan = replication_plan(configuration)
    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, plan)  # Exclusive creation and read-only: never replace an old plan.
    return {"replication_plan": str(path), "replication_plan_sha256": sha256_file(path),
            "plan_payload_sha256": digest_object(plan), "operation_counts": plan["operation_counts"]}


def verify_replication_plan(configuration, *, require_frozen=True):
    try:
        import scipy.stats  # Explicit t-inference dependency, checked before inference.
    except ImportError as error:
        raise ConditionalConfigurationError("replication inference requires SciPy; researcher command: python -m pip install 'scipy>=1.11,<2'") from error
    plan = replication_plan(configuration)
    w = configuration.workflow
    if "replication_plan" not in w:
        if require_frozen:
            raise ConditionalConfigurationError("freeze and bind a replication plan before execution (conditional freeze-replication)")
        return plan
    path = Path(w["replication_plan"])
    try:
        if sha256_file(path) != w["replication_plan_sha256"] or json.loads(path.read_text()) != plan:
            raise ValueError("frozen replication plan/hash differs from configuration or exposure ledger")
    except (OSError, ValueError) as error:
        raise ConditionalConfigurationError(str(error)) from error
    return plan
