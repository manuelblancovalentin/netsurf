"""Immutable result bundles for schema-6 exact conditional-damage studies."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

from . import __version__
from .conditional_damage import ConditionalOutcome
from .manifests import git_provenance, runtime_provenance, sha256_file
from .result_bundle import ResultBundleError, _inventory, _write_bytes, _write_csv, _write_json


def _report(outcome: ConditionalOutcome) -> str:
    if outcome.configuration.workflow and outcome.configuration.schema_version in {"6.4.0", "6.5.0", "6.6.0", "6.7.0", "6.8.0"}:
        return _preflight_report(outcome)
    payload = outcome.scientific_payload
    interval_label = f"{100.0 * outcome.configuration.execution.confidence:g}% CI"
    lines = [
        f"# {outcome.configuration.name}", "",
        outcome.configuration.description, "",
        "## Scientific question", "", outcome.configuration.objective, "",
        "## Protocol", "",
        "For each prespecified candidate bit, NetSuRF measures the exact encoded-XOR risk "
        "difference `D_b(G) = R(G ∪ {b}) - R(G)`. The candidate is removed from each "
        "common Bernoulli background before the paired comparison. Calibration and held-out "
        "evaluation use disjoint samples and independently seeded background fields.", "",
        "The reported confidence intervals describe background-field variability conditional "
        "on this frozen model, fixed candidate list, fixed sample partition, and logical BER. "
        "They are not uncertainty over models, candidate sampling, or physical radiation.", "",
        "## Conditional-damage estimates", "",
        f"| Split | Logical BER | Candidate | Isolated damage | Conditional mean | {interval_label} | Negative fields |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    measurements = outcome.measurements
    for row in outcome.summaries:
        estimate = row["conditional_damage"]
        matching = [
            value["conditional_damage"] for value in measurements
            if value["split"] == row["split"]
            and value["logical_ber"] == row["logical_ber"]
            and value["candidate"] == row["candidate"]
        ]
        negative = sum(value < 0.0 for value in matching)
        low, high = estimate["confidence_interval"]
        lines.append(
            f"| {row['split']} | {row['logical_ber']:.3g} | `{row['candidate']}` | "
            f"{row['isolated_damage']:.8g} | {estimate['mean']:.8g} | "
            f"[{low:.8g}, {high:.8g}] | {negative}/{len(matching)} |"
        )
    lines.extend(["", "## Rank-order diagnostics", ""])
    for row in outcome.rank_comparisons:
        correlation = row["spearman_average_rank"]
        correlation_text = "undefined (all ties)" if correlation is None else f"{correlation:.6g}"
        lines.append(
            f"- {row['split']}, BER {row['logical_ber']:.3g}: "
            f"{row['strict_rank_reversals']}/{row['strict_comparable_pairs']} strict pairwise "
            f"reversals; average-rank Spearman {correlation_text}."
        )
    if outcome.oracle_summaries:
        lines.extend(["", "## Restricted protection oracle", ""])
        lines.extend([
            "The empirically optimized mask is selected exclusively on the model-selection "
            "partition and its independently seeded outside-background fields. Held-out "
            "selected-minus-isolated differences are paired by outside field. The reported "
            "held-out optimum is post-hoc diagnostic information and is not a valid selected "
            "policy or an unbiased confirmation estimate.", "",
            "| BER | K | Budget | Original mask | Selected mask | Isolated mask | Held-out selected risk | Selected - original | Selected - isolated | Post-hoc regret |",
            "|---:|---:|---:|---|---|---|---:|---:|---:|---:|",
        ])
        for row in outcome.oracle_summaries:
            lines.append(
                f"| {row['logical_ber']:.3g} | {row['candidate_count']} | {row['budget']} | "
                f"`{', '.join(row['original_mask_bits'])}` | "
                f"`{', '.join(row['selected_mask_bits'])}` | "
                f"`{', '.join(row['isolated_mask_bits'])}` | "
                f"{row['held_out_selected_risk']['mean']:.8g} | "
                f"{row['held_out_selected_minus_original']['mean']:.8g} | "
                f"{row['held_out_selected_minus_isolated']['mean']:.8g} | "
                f"{row['posthoc_held_out_regret']['mean']:.8g} |"
            )
        lines.extend(["", "### Model-selection stability", ""])
        for row in outcome.oracle_summaries:
            stability = row.get("model_selection_stability")
            if stability:
                final_margin = stability["cumulative_best_second_margin"][-1]
                lines.append(
                    f"- BER {row['logical_ber']:.3g}: {stability['field_count']} outside "
                    f"fields over {stability['eligible_mask_count']} masks; "
                    f"{stability['cumulative_switch_count']} cumulative winner changes; "
                    f"final best-to-second mean-risk margin {final_margin:.8g}."
                )
        lines.extend([
            "", "These diagnostics describe finite model-selection-field sensitivity. "
            "They are not uncertainty over final fields, fixed source inputs, or models.",
        ])
    if outcome.paired_swap_summaries:
        lines.extend([
            "", "## Efficient frozen-mask paired comparisons", "",
            "Each row uses independent master Bernoulli fields as its replication units. "
            "All steps in a deterministic cardinality-preserving swap path are summed before "
            "the confidence interval is calculated. Masks were frozen using model-selection "
            "data; final outcomes do not select the path or mask.", "",
            f"| BER | Comparison | Fields | Target - baseline risk | {interval_label} |",
            "|---:|---|---:|---:|---:|",
        ])
        for row in outcome.paired_swap_summaries:
            estimate = row["target_minus_baseline"]
            low, high = estimate["confidence_interval"]
            lines.append(
                f"| {row['logical_ber']:.3g} | `{row['comparison']}` | "
                f"{row['master_background_fields']} | {estimate['mean']:.8g} | "
                f"[{low:.8g}, {high:.8g}] |"
            )
    lines.extend([
        "", "## Limitations", "",
        "This bounded candidate study does not estimate a population-wide ranking or establish "
        "causal architectural mechanisms. The restricted oracle selects a mask only within the "
        "explicit candidate subcube; it does not optimize over the full network bit population. "
        "Logical BER is not a calibrated SEU rate, and exact TFLite XOR damage is authoritative; "
        "no float-shadow perturbation is used.", "",
        "Model-selection, final master-field, and fixed-input uncertainties are separate. "
        "A small cumulative selection margin or winner switching motivates a new prespecified "
        "selection run, not tuning with final outcomes.", "",
        "## Reproduction", "",
        f"`netsurf conditional run {outcome.configuration.source_path}`", "",
        f"Scientific payload SHA-256: `{outcome.scientific_payload_sha256}`", "",
    ])
    return "\n".join(lines)


def _preflight_report(outcome):
    if outcome.configuration.schema_version == "6.5.0":
        from .replication_statistics import replication_report
        return replication_report(outcome)
    mode = outcome.configuration.workflow["mode"]
    population_description = (
        "Probability-sampled S5-status-stratified candidate pool; pool-level inference requires the complete frozen campaign. "
        if "campaign_manifest" in outcome.configuration.workflow else
        "Frozen exploratory candidate pool, not a random population sample. ")
    lines = [f"# {outcome.configuration.name}", "", outcome.configuration.objective, "",
             f"Execution: `{mode}`; purpose: `{outcome.configuration.workflow['purpose']}`.", "",
             population_description + "All risk measurements "
             "use exact INT8 XOR and TFLite 2.14.1 BUILTIN_REF. Logical BER is not a physical SEU rate.", ""]
    cohort = outcome.configuration.schema_version == "6.8.0"
    risk_label = "1 − file ROC-AUC" if cohort else "cross-entropy"
    if cohort:
        lines += ["One risk is computed over complete normal/anomalous files under a persistent weight-fault field. "
                  "All 196 windows of each file contribute to its pinned anomaly score. Tied scores receive half credit. "
                  "MSE is diagnostic only, never the optimization objective. Files, windows and normal/anomalous pairs "
                  "are not independent fault trials; input-population uncertainty is not estimated.", "",
                  "Exact file IDs, labels, ordered windows, score bytes/hashes, integer-output hashes and ordering changes "
                  "are saved in `artifacts/cohort_scores.json`.", ""]
        evidence = outcome.scientific_payload["scalar_cohort_evidence"]
        lines += ["| Cohort | Normal / anomalous files | Windows | Clean ROC-AUC | Evaluated states | Max strict pair reversals |",
                  "|---|---:|---:|---:|---:|---:|"]
        for key, record in evidence["cohorts"].items():
            rows = [r for r in evidence["evaluations"] if r["cohort_sha256"] == key]
            clean = next((r for r in rows if r["fault"] is None), None)
            normal = sum(f["label"] == 0 for f in record["files"])
            auc = clean.get("file_roc_auc", 1-clean["risk"]) if clean is not None else "not measured"
            lines.append(f"| {record['role']} | {normal} / {len(record['files'])-normal} | {record['window_count']} | {auc} | {len(rows)} | {max((r['pair_ordering'].get('strict_reversals', 0) for r in rows), default=0)} |")
        lines += ["", "Ordering counts are descriptive: pairs share files and must not be treated as independent observations.", ""]
    if mode == "selection_only":
        lines += ["No final-evaluation samples were loaded and no final fields were generated. "
                  "No held-out mask performance is reported.", "",
                  "## All model-selection mask risks", "",
                  f"| Candidate mask | Mean {risk_label} | Fields |", "|---|---:|---:|"]
        for row in outcome.oracle_summaries:
            for mask in row["eligible_masks"]:
                lines.append(f"| {mask['indices']} | {mask['mean_risk']:.10g} | {len(mask['field_risks'])} |")
            stability = row["model_selection_stability"]
            lines += ["", f"Selected: {row['selected_mask_indices']}; isolated: {row['isolated_mask_indices']}; "
                      f"original S5: {row['original_mask_indices']}.",
                      f"Cumulative switches: {stability['cumulative_switch_count']}; final best/second margin: "
                      f"{stability['cumulative_best_second_margin'][-1]:.10g}.", "",
                      "Replicate winners (independent outside-field batches, not independent inputs/models):"]
            for rep in stability["replicates"]:
                lines.append(f"- Seed {rep['seed']}: {rep['cumulative_winners'][-1]}, "
                             f"{rep['field_count']} fields, margin {rep['cumulative_best_second_margin'][-1]:.10g}")
        lines += ["", "All individual-field and cumulative winners/margins, per-mask field risks, exact "
                  "response tables and outside events are retained in the artifacts. The winning mean "
                  "is selection-biased; pilot stability is not a power guarantee."]
    else:
        lines += ["Masks loaded from the explicitly hash-pinned finalized selection artifact; "
                  "selection was not rerun. Primary: restricted-selected minus original S5. Negative "
                  "differences favor the target. Secondary contrasts share the same master fields.", "",
                  f"| Contrast | Fields | Mean {risk_label} difference | Marginal CI | Family-wise CI |",
                  "|---|---:|---:|---|---|"]
        for row in outcome.paired_swap_summaries:
            e = row["target_minus_baseline"]
            lines.append(f"| {row['comparison']} | {row['master_background_fields']} | {e['mean']:.10g} | "
                         f"{e['confidence_interval']} | {row['familywise_target_minus_baseline']['confidence_interval']} |")
        lines += ["", "Student-t intervals use whole-path independent master fields, conditional on "
                  "fixed samples/model/masks; no input-sampling or candidate-population inference. "
                  "Bonferroni intervals cover the prespecified three-contrast family. No automatic "
                  "superiority claim or post-hoc final optimum is computed."]
    lines += ["", f"Operation plan: `{outcome.configuration.operation_counts()}`", "",
              f"Scientific payload SHA-256: `{outcome.scientific_payload_sha256}`", "",
              f"Reproduce: `netsurf conditional run {outcome.configuration.source_path}`", ""]
    return "\n".join(lines)


def create_conditional_bundle(
    outcome: ConditionalOutcome, *, repository_root: Path, run_id: str | None = None,
) -> Path:
    """Write a complete bundle through staging, then atomically finalize it."""

    now = datetime.now(timezone.utc)
    identifier = run_id or f"{now.strftime('%Y%m%dT%H%M%S.%fZ')}-{uuid.uuid4().hex[:8]}"
    root = outcome.configuration.output.root / outcome.configuration.name
    destination = root / identifier
    staging = root / f".{identifier}.incomplete"
    root.mkdir(parents=True, exist_ok=True)
    if destination.exists() or staging.exists():
        raise ResultBundleError(f"run directory already exists: {destination}")
    staging.mkdir()
    try:
        _write_bytes(staging / "config.toml", outcome.configuration.source_text.encode("utf-8"))
        _write_json(staging / "resolved_config.json", outcome.configuration.resolved_document())
        _write_json(staging / "scientific_payload.json", outcome.scientific_payload)
        if "frozen_selection" in outcome.scientific_payload:
            _write_json(staging / "artifacts/frozen_selection.json", outcome.scientific_payload["frozen_selection"])
        if "scalar_cohort_evidence" in outcome.scientific_payload:
            _write_json(staging / "artifacts/cohort_scores.json", outcome.scientific_payload["scalar_cohort_evidence"])
        _write_json(staging / "summary.json", {
            "schema_version": outcome.configuration.schema_version,
            "scientific_protocol": outcome.configuration.scientific_protocol,
            "experiment_name": outcome.configuration.name,
            "benchmark_id": outcome.configuration.benchmark.id,
            "risk_metric": outcome.configuration.risk_metric,
            "candidate_count": len(outcome.candidate_records),
            "background_field_count": len(outcome.background_records),
            "conditional_estimates": list(outcome.summaries),
            "rank_comparisons": list(outcome.rank_comparisons),
            "restricted_oracle": list(outcome.oracle_summaries),
            "paired_swap_evaluation": list(outcome.paired_swap_summaries),
            "protection_context": dict(outcome.protection_context_record),
            "scientific_payload_sha256": outcome.scientific_payload_sha256,
            **({"replication_analysis": outcome.scientific_payload["replication_analysis"]}
               if "replication_analysis" in outcome.scientific_payload else {}),
        })
        _write_csv(staging / "measurements/conditional_damage.csv", [
            "schema_version", "split", "sample_sha256", "logical_ber", "field", "field_seed",
            "candidate", "parameter", "flat_index", "bit_index", "master_selected_candidate",
            "master_schedule_sha256", "background_schedule_sha256", "candidate_schedule_sha256",
            "clean_risk", "isolated_fault_risk", "isolated_damage", "background_risk",
            "background_plus_candidate_risk", "conditional_damage", "background_proposed_events",
            "background_applied_events", "background_suppressed_events", "background_unique_applied_bits",
            "candidate_proposed_events", "candidate_applied_events", "candidate_suppressed_events",
            "candidate_unique_applied_bits", "empty_proposed_events", "isolated_proposed_events",
        ], ({"schema_version": outcome.configuration.schema_version, **row} for row in outcome.measurements))
        _write_csv(staging / "measurements/conditional_summary.csv", [
            "schema_version", "split", "sample_sha256", "logical_ber", "candidate", "parameter",
            "flat_index", "bit_index", "isolated_damage", "conditional_mean", "conditional_std",
            "conditional_se", "ci_low", "ci_high", "background_field_count", "uncertainty_unit",
        ], ({
            "schema_version": outcome.configuration.schema_version, "split": row["split"],
            "sample_sha256": row["sample_sha256"], "logical_ber": row["logical_ber"],
            "candidate": row["candidate"], "parameter": row["parameter"],
            "flat_index": row["flat_index"], "bit_index": row["bit_index"],
            "isolated_damage": row["isolated_damage"],
            "conditional_mean": row["conditional_damage"]["mean"],
            "conditional_std": row["conditional_damage"]["standard_deviation"],
            "conditional_se": row["conditional_damage"]["standard_error"],
            "ci_low": row["conditional_damage"]["confidence_interval"][0],
            "ci_high": row["conditional_damage"]["confidence_interval"][1],
            "background_field_count": row["background_field_count"],
            "uncertainty_unit": row["uncertainty_unit"],
        } for row in outcome.summaries))
        _write_json(staging / "artifacts/candidates.json", list(outcome.candidate_records))
        _write_json(staging / "artifacts/background_fields.json", list(outcome.background_records))
        _write_json(staging / "artifacts/rank_comparisons.json", list(outcome.rank_comparisons))
        if outcome.oracle_response_records:
            _write_csv(staging / "measurements/restricted_oracle_responses.csv", [
                "schema_version", "phase", "logical_ber", "field", "field_seed",
                "replicate", "local_field",
                "candidate_state", "candidate_bits", "outside_background_sha256",
                "pairing_key", "probability_law_sha256", "protection_context_sha256",
                "risk", "schedule_sha256", "proposed_events", "applied_events",
                "suppressed_events", "unique_applied_bits", "cache_hit",
            ], ({"schema_version": outcome.configuration.schema_version, **row}
                for row in outcome.oracle_response_records))
            _write_json(
                staging / "artifacts/restricted_oracle_backgrounds.json",
                list(outcome.oracle_background_records),
            )
            _write_json(
                staging / "artifacts/restricted_oracle_masks.json",
                list(outcome.oracle_summaries),
            )
            _write_json(
                staging / "artifacts/restricted_oracle_derived.json",
                list(outcome.oracle_derived_records),
            )
        if outcome.paired_swap_field_records:
            _write_csv(staging / "measurements/paired_swap_fields.csv", [
                "schema_version", "logical_ber", "field", "field_seed", "comparison",
                "sample_sha256", "master_event_count", "master_events", "master_sha256",
                "master_pairing_key", "comparison_pairing_key", "probability_law",
                "probability_law_sha256", "protection_context_sha256", "baseline_mask",
                "target_mask", "path_length", "target_minus_baseline",
                "authoritative_evaluation_requests", "uncertainty_unit",
            ], ({"schema_version": outcome.configuration.schema_version, **row}
                for row in outcome.paired_swap_field_records))
            _write_csv(staging / "measurements/paired_swap_steps.csv", [
                "schema_version", "logical_ber", "field", "field_seed", "comparison",
                "comparison_pairing_key", "step", "removed", "added", "current_mask",
                "background_event_count", "background_events", "background_sha256",
                "removed_schedule_sha256", "added_schedule_sha256", "removed_risk",
                "added_risk", "removed_minus_added", "target_minus_baseline_contribution",
                "removed_proposed_events", "removed_applied_events",
                "removed_suppressed_events", "added_proposed_events",
                "added_applied_events", "added_suppressed_events",
            ], ({"schema_version": outcome.configuration.schema_version, **row}
                for row in outcome.paired_swap_step_records))
            _write_json(
                staging / "artifacts/paired_swap_summaries.json",
                list(outcome.paired_swap_summaries),
            )
        _write_json(
            staging / "artifacts/protection_context.json",
            dict(outcome.protection_context_record),
        )
        if outcome.configuration.output.report:
            _write_bytes(staging / "report.md", _report(outcome).encode("utf-8"))
        manifest = {
            "schema_version": outcome.configuration.schema_version,
            "run_id": identifier, "status": "complete",
            "source": git_provenance(repository_root.resolve()),
            "runtime": runtime_provenance(),
            "versions": {"netsurf": __version__, "conditional_schema": outcome.configuration.schema_version,
                         "scientific_protocol": outcome.configuration.scientific_protocol},
            "benchmark": {
                **outcome.configuration.resolved_document()["benchmark"],
                "checkpoint_sha256": sha256_file(outcome.configuration.benchmark.checkpoint),
                "asset_lock_sha256": sha256_file(outcome.configuration.benchmark.asset_lock),
            },
            "execution": {
                "runtime_seconds": outcome.runtime_seconds,
                "calibration_background_seed": outcome.configuration.backgrounds.calibration_seed,
                "evaluation_background_seed": outcome.configuration.backgrounds.evaluation_seed,
                "model_selection_replicate_seeds": list(
                    outcome.configuration.restricted_oracle.selection_seeds
                ),
                "paired_swap_evaluation_seed": (
                    outcome.configuration.paired_swap_evaluation.seed
                    if outcome.configuration.paired_swap_evaluation.enabled else None
                ),
            },
            "outputs": _inventory(staging),
            "scientific_payload_sha256": outcome.scientific_payload_sha256,
        }
        if outcome.configuration.workflow:
            manifest["execution"] = {
                "runtime_seconds": outcome.runtime_seconds,
                "mode": outcome.configuration.workflow["mode"],
                "operation_counts": outcome.configuration.operation_counts(),
                "model_selection_replicate_seeds_executed": (
                    list(outcome.configuration.restricted_oracle.selection_seeds)
                    if outcome.configuration.workflow["mode"] == "selection_only" else []),
                "paired_final_seed_executed": (
                    outcome.configuration.paired_swap_evaluation.seed
                    if outcome.configuration.workflow["mode"] in {"final_only", "replication_only"} else None),
            }
            manifest["m956_final_original_indices"] = (
                list(outcome.partitions["held_out_evaluation"].indices)
                if "held_out_evaluation" in outcome.partitions else [])
        if outcome.configuration.schema_version == "6.5.0":
            _write_json(staging / "artifacts/replication_plan.json", outcome.scientific_payload["replication_plan"])
            _write_json(staging / "artifacts/replication_analysis.json", outcome.scientific_payload["replication_analysis"])
            if "paired_input_diagnostics" in outcome.scientific_payload:
                _write_json(staging / "artifacts/paired_input_contributions.json", outcome.scientific_payload["paired_input_diagnostics"])
            manifest["replication_plan_sha256"] = outcome.scientific_payload["replication_plan_sha256"]
            manifest["outputs"] = _inventory(staging)
        if "campaign_binding" in outcome.scientific_payload:
            manifest["campaign_binding"] = outcome.scientific_payload["campaign_binding"]
            if "paired_input_diagnostics" in outcome.scientific_payload:
                _write_json(staging / "artifacts/paired_input_contributions.json", outcome.scientific_payload["paired_input_diagnostics"])
            manifest["outputs"] = _inventory(staging)
        _write_json(staging / "manifest.json", manifest)
        _write_json(staging / "checksums.json", {
            "schema_version": outcome.configuration.schema_version,
            "files": {row["path"]: row["sha256"] for row in _inventory(staging)},
        })
        _write_bytes(staging / "FINALIZED", (
            f"scientific_payload_sha256={outcome.scientific_payload_sha256}\n"
        ).encode("ascii"))
        staging.rename(destination)
        return destination
    except BaseException:
        # Preserve incomplete evidence for diagnosis; never finalize or overwrite it.
        raise


def verify_conditional_bundle(path: str | Path) -> dict[str, object]:
    root = Path(path).expanduser().resolve()
    if not (root / "FINALIZED").is_file():
        raise ResultBundleError(f"not a finalized conditional bundle: {root}")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    checksums = json.loads((root / "checksums.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") not in {"6.0.0", "6.1.0", "6.2.0", "6.3.0", "6.4.0", "6.5.0", "6.6.0", "6.7.0", "6.8.0"}:
        raise ResultBundleError(f"unsupported conditional manifest: {manifest.get('schema_version')!r}")
    for relative, expected in checksums["files"].items():
        actual = sha256_file(root / relative)
        if actual != expected:
            raise ResultBundleError(f"checksum mismatch for {relative}: expected {expected}, got {actual}")
    if manifest["schema_version"] == "6.8.0":
        from .cohort_analysis import verify_score_evidence
        from .manifests import digest_object
        payload = json.loads((root / "scientific_payload.json").read_text())
        evidence = json.loads((root / "artifacts/cohort_scores.json").read_text())
        if (digest_object(evidence) != payload["scalar_cohort_evidence_sha256"]
                or evidence != payload["scalar_cohort_evidence"]
                or digest_object(payload) != manifest["scientific_payload_sha256"]
                or (root / "FINALIZED").read_text().strip() != f"scientific_payload_sha256={digest_object(payload)}"):
            raise ResultBundleError("scalar-cohort payload/evidence identity mismatch")
        verify_score_evidence(evidence)
    return manifest
