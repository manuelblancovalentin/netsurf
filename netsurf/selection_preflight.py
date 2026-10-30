"""Frozen-selection boundaries and auditable prior-exposure gates (schema 6.4)."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import re

from .manifests import digest_object, sha256_file


LEDGER = Path(__file__).parent / "data/kws_prior_exposure_v1.json"


def exposure_audit(offset, count, purpose):
    ledger = json.loads(LEDGER.read_text())
    exposed = {s["original_index"] for r in ledger["runs"] for s in r["samples"]}
    requested = set(range(3048 + offset, 3048 + offset + count))
    overlap = sorted(requested & exposed)
    if purpose == "confirmation" and overlap:
        raise ValueError(f"final sample exposure overlap: original KWS indices {overlap}")
    return {"ledger_sha256": sha256_file(LEDGER), "audit_scope": ledger["audit_scope"],
            "requested_original_indices": sorted(requested), "prior_overlap": overlap,
            "purpose": purpose, "clean_reference_inference_previously_performed": True}


def reject_local_final_reuse(configuration):
    """Also reject already finalized 6.4 final samples under this output root.

    This is not a global dataset-usage registry or a cross-process reservation.
    No scientific measurements or model inputs are read during this check.
    """
    if configuration.workflow.get("purpose") != "confirmation":
        return
    from .conditional_config import ConditionalConfigurationError
    requested = set(range(3048 + configuration.samples.evaluation_offset,
                          3048 + configuration.samples.evaluation_offset + configuration.samples.evaluation_count))
    for path in configuration.output.root.glob("*/*/manifest.json"):
        if not (path.parent / "FINALIZED").is_file():
            continue
        manifest = json.loads(path.read_text())
        overlap = requested & set(manifest.get("m956_final_original_indices", ()))
        if overlap:
            raise ConditionalConfigurationError(
                f"final sample exposure overlap with finalized bundle {path.parent}: {sorted(overlap)}")


def validate_workflow(raw, samples, backgrounds, oracle, paired, context, source):
    from .conditional_config import ConditionalConfigurationError, PAIRED_SWAP_COMPARISONS
    required = {"mode", "purpose", "candidate_provenance"}
    optional = {"selection_bundle", "selection_payload_sha256", "selection_artifact_sha256"}
    if not isinstance(raw, dict) or not required <= raw.keys() or raw.keys() - required - optional:
        raise ConditionalConfigurationError("workflow requires mode, purpose, candidate_provenance; only selection source fields are optional")
    workflow = dict(raw)
    mode = workflow["mode"]
    if mode not in {"selection_only", "final_only"}:
        raise ConditionalConfigurationError("workflow.mode must be selection_only or final_only; joint execution remains schema 6.3")
    if workflow["purpose"] not in {"engineering", "confirmation"}:
        raise ConditionalConfigurationError("workflow.purpose must be engineering or confirmation")
    if workflow["candidate_provenance"] != "exploratory_frozen_m953_pool":
        raise ConditionalConfigurationError("schema 6.4 currently accepts only exploratory_frozen_m953_pool")
    if context.mode != "verified_global_mask" or len(backgrounds.probabilities) != 1:
        raise ConditionalConfigurationError("schema 6.4 requires one prespecified BER and a verified global mask")
    if paired.background_count < 2 or paired.comparisons != PAIRED_SWAP_COMPARISONS:
        raise ConditionalConfigurationError("freeze >=2 final fields and the ordered three-comparison family even in selection_only")
    # Audit actual derived field seeds, not just master-seed equality.
    seeds = [seed + field * 1_000_003 for seed in oracle.selection_seeds
             for field in range(oracle.calibration_background_count)]
    final_seeds = [paired.seed + field * 1_000_003 for field in range(paired.background_count)]
    if len(set(seeds)) != len(seeds) or set(seeds) & set(final_seeds):
        raise ConditionalConfigurationError("selection/final derived field seeds must be independent and unique")
    if any(seed < 0 for seed in seeds + final_seeds):
        raise ConditionalConfigurationError("field seeds must be nonnegative")
    if mode == "selection_only":
        if paired.enabled or raw.keys() & optional:
            raise ConditionalConfigurationError("selection_only must disable paired evaluation and cannot load a selection source")
    else:
        if not paired.enabled or paired.background_count < 2 or paired.comparisons != PAIRED_SWAP_COMPARISONS:
            raise ConditionalConfigurationError("final_only requires >=2 fields and the ordered three-comparison family (selected-original primary)")
        if not optional <= raw.keys():
            raise ConditionalConfigurationError("final_only requires an explicit selection bundle and both hashes")
        for key in ("selection_payload_sha256", "selection_artifact_sha256"):
            if not re.fullmatch("[0-9a-f]{64}", str(raw[key])):
                raise ConditionalConfigurationError(f"{key} must be a SHA-256 from a finalized selection artifact")
        path = Path(raw["selection_bundle"]).expanduser()
        workflow["selection_bundle"] = str((source.parent / path if not path.is_absolute() else path).resolve())
    try:
        exposure_audit(samples.evaluation_offset, samples.evaluation_count, workflow["purpose"])
    except ValueError as error:
        raise ConditionalConfigurationError(str(error)) from error
    return workflow


def selection_contract(configuration):
    """Scientific construction identity and final plan; excludes paths/presentation."""
    benchmark = configuration.resolved_document()["benchmark"]
    benchmark["injectable_parameters"] = list(benchmark["injectable_parameters"])
    for key in ("checkpoint", "dataset", "asset_lock"):
        benchmark.pop(key)
    context = configuration.protection_context.to_dict()
    context.pop("source_bundle")
    return {
        "benchmark": benchmark,
        "checkpoint_sha256": sha256_file(configuration.benchmark.checkpoint),
        "asset_lock_sha256": sha256_file(configuration.benchmark.asset_lock),
        "candidates": [b.key() for b in configuration.candidates],
        "candidate_provenance": configuration.workflow["candidate_provenance"],
        "budget": configuration.restricted_oracle.budget,
        "protection_context": context,
        "risk": {"metric": configuration.risk_metric, "direction": configuration.risk_direction},
        "probabilities": list(configuration.backgrounds.probabilities),
        "model_selection": {"role": configuration.samples.model_selection_role,
                            "count": configuration.samples.model_selection_count,
                            "offset": configuration.samples.model_selection_offset},
        "selection_fields_per_seed": configuration.restricted_oracle.calibration_background_count,
        "selection_seeds": list(configuration.restricted_oracle.selection_seeds),
        "final_plan": {
            "role": configuration.samples.evaluation_role,
            "offset": configuration.samples.evaluation_offset,
            "count": configuration.samples.evaluation_count,
            "paired": {**asdict(configuration.paired_swap_evaluation), "enabled": True,
                       "comparisons": list(configuration.paired_swap_evaluation.comparisons)},
            "confidence": configuration.execution.confidence,
            "purpose": configuration.workflow["purpose"],
            "procedure": "Student-t marginal; Bonferroni three-contrast family; whole-path master-field unit",
        },
    }


def load_frozen_selection(configuration):
    """Verify content and construction before any final data or fields are accessed."""
    from .conditional_bundle import verify_conditional_bundle
    from .conditional_config import ConditionalConfigurationError
    from .result_bundle import ResultBundleError
    workflow = configuration.workflow
    root = Path(workflow["selection_bundle"])
    try:
        manifest = verify_conditional_bundle(root)
        payload = json.loads((root / "scientific_payload.json").read_text())
        artifact = root / "artifacts/frozen_selection.json"
        record = json.loads(artifact.read_text())
        expected = workflow["selection_payload_sha256"]
        if (manifest["status"] != "complete" or digest_object(payload) != expected
                or manifest["scientific_payload_sha256"] != expected
                or (root / "FINALIZED").read_text().strip() != f"scientific_payload_sha256={expected}"
                or sha256_file(artifact) != workflow["selection_artifact_sha256"]
                or digest_object(record) != payload.get("frozen_selection_sha256")):
            raise ValueError("frozen selection content/hash mismatch")
        if payload.get("workflow", {}).get("mode") != "selection_only":
            raise ValueError("source must be a selection-only result, not a joint/final result")
        if record["contract"] != selection_contract(configuration):
            raise ValueError("frozen selection contract differs (model, data, context, pool, budget, metric, BER or selection plan)")
        if len(record["masks"]) != 1:
            raise ValueError("expected exactly one frozen BER allocation")
        for row in record["masks"]:
            if row["logical_ber"] != configuration.backgrounds.probabilities[0]:
                raise ValueError("frozen mask BER differs")
            for name in ("selected", "isolated", "original"):
                indices = row[name]
                if (len(indices) != configuration.restricted_oracle.budget
                        or indices != sorted(set(indices))
                        or any(type(i) is not int or not 0 <= i < len(configuration.candidates) for i in indices)):
                    raise ValueError(f"invalid frozen {name} mask")
        selected_ids = {s["sample_id"] for s in record["sample_identities"]}
        final_ids = {f"speech_commands_tfds_test_interleaved_blocks16:{i}" for i in range(
            3048 + configuration.samples.evaluation_offset,
            3048 + configuration.samples.evaluation_offset + configuration.samples.evaluation_count)}
        if selected_ids & final_ids:
            raise ValueError("frozen selection samples overlap final samples")
        return record
    except (OSError, KeyError, ValueError, ResultBundleError) as error:
        raise ConditionalConfigurationError(f"cannot use frozen selection: {error}") from error


def selection_summary(tables, isolated_table, candidates, budget, probability, original, seeds, per_seed):
    from .restricted_subcube import fixed_budget_masks, select_response_tables, selection_stability_diagnostics
    masks = fixed_budget_masks(len(candidates), budget)
    selection = select_response_tables(tables, isolated_table, candidate_count=len(candidates),
                                       budget=budget, probability=probability, original_mask=original)
    risks = {mask: list(values) for mask, values in selection["calibration_field_risks"].items()}
    means = selection["calibration_mean_risks"]
    selected, isolated = selection["selected_mask"], selection["isolated_mask"]
    damage = selection["isolated_damages"]
    stability = selection_stability_diagnostics(tables, candidate_count=len(candidates), budget=budget, probability=probability)
    stability["per_field_best_second_margin"] = [
        sorted(risks[m][i] for m in masks)[1] - min(risks[m][i] for m in masks)
        for i in range(len(tables))
    ]
    stability["replicates"] = [{
        "replicate": r, "seed": seed,
        **selection_stability_diagnostics(tables[r * per_seed:(r + 1) * per_seed],
            candidate_count=len(candidates), budget=budget, probability=probability),
    } for r, seed in enumerate(seeds)]
    return {"logical_ber": probability, "candidate_count": len(candidates), "budget": budget,
            "selected_mask_indices": list(selected), "isolated_mask_indices": list(isolated),
            "original_mask_indices": list(original),
            "eligible_masks": [{"indices": list(m), "field_risks": risks[m], "mean_risk": means[m]} for m in masks],
            "isolated_damages": [{"candidate": candidates[i].key(), "damage": damage[i]} for i in damage],
            "model_selection_stability": stability,
            "global_protection_cardinality_preserved": True,
            "interpretation": "exploratory fixed pool; selected on model-selection fields only; no final performance measured"}
