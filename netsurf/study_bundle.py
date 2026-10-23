"""Immutable schema-3 result bundles for M3 scientific studies."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import uuid

from . import __version__
from .derived_analysis import derive_protection_analysis
from .experiment_config import MANIFEST_SCHEMA_VERSION, SUMMARY_SCHEMA_VERSION
from .manifests import git_provenance, runtime_provenance, sha256_file
from .plotting import plot_study_results
from .reporting import render_study_report
from .result_bundle import ResultBundleError, _inventory, _write_bytes, _write_csv, _write_json
from .study_engine import StudyOutcome


def _trial_rows(outcome: StudyOutcome):
    members = outcome.replications or (outcome,)
    replicated = bool(outcome.replications)
    rows = []
    for member in members:
        mask_by_condition = {str(value["policy"]): value for value in member.protection_results}
        for value in member.trials:
            row = {
                **({"replication": member.replication_name} if replicated else {}),
                **value.to_dict(),
            }
            mask = mask_by_condition.get(value.policy)
            if mask:
                row["ranking_mask_sha256"] = mask.get("mask_sha256")
                row["protected_bits"] = mask.get("protected_bits")
                row["realized_coverage"] = mask.get("coverage")
            rows.append(row)
    return tuple(rows)


def _derived_protection(outcome: StudyOutcome):
    rows = _trial_rows(outcome)
    if not rows:
        return None
    config = outcome.configuration
    return derive_protection_analysis(
        rows, confidence=config.execution.confidence,
        permutation_seed=config.execution.permutation_seed,
        default_ber=(config.protection.probabilities or (config.fault.probability,))[0],
        expected_replications=config.replications.count,
    )


def _summary(outcome: StudyOutcome, derived=None):
    members = outcome.replications or (outcome,)
    primary = members[0]
    payload = primary.scientific_payload
    overall = [dict(row) for row in primary.ranking_metrics if row["scope"] == "overall"]
    runtimes = {value.name: value.runtime_seconds for value in primary.estimator_outputs}
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "study_schema_version": outcome.configuration.schema_version,
        "scientific_protocol": outcome.configuration.scientific_protocol,
        "experiment_name": outcome.configuration.name,
        "benchmark_id": outcome.configuration.benchmark.id,
        "eligible_bits": payload["benchmark"]["eligible_bits"],
        "trial_count": outcome.configuration.execution.trials if primary.trials else 0,
        "trial_count_per_replication": outcome.configuration.execution.trials if primary.trials else 0,
        "pooled_fault_trial_count": max((row["fault_realization_count"] for row in (derived or {}).get("pooled", [])), default=0),
        "clean_evaluation": payload["clean_evaluation"],
        "empirical_candidate_count": len(primary.empirical_measurements),
        "sample_limited": any(value.sampling_probability < 1.0 for value in primary.empirical_measurements),
        "ranking_metrics_overall": overall,
        "estimator_agreement": list(outcome.estimator_agreement),
        "protection_results": list((derived or {}).get("pooled", outcome.protection_results)),
        "replication_protection_results": list((derived or {}).get("replications", [])),
        "paired_comparisons": list((derived or {}).get("paired", [])) if derived else outcome.paired_comparisons,
        "paired_replication_summaries": list((derived or {}).get("paired_replications", [])),
        "statistical_unit": None if not derived else derived["statistical_unit"],
        "interval_interpretation": None if not derived else derived["interval_interpretation"],
        "mask_composition": list(outcome.composition),
        "estimator_runtimes_seconds": runtimes,
        "replication_count": len(members),
        "replication_aggregate": list(outcome.replication_aggregate),
        "checks": payload["checks"],
        "scientific_payload_sha256": outcome.scientific_payload_sha256,
    }


def _prediction_rows(outcome: StudyOutcome, *, replication: str | None = None):
    candidates = {value.identity for value in outcome.empirical_measurements}
    for output in outcome.estimator_outputs:
        for identity in sorted(output.scores):
            row = {
                "schema_version": outcome.configuration.schema_version,
                "estimator": output.name,
                "implementation_version": output.implementation_version,
                "parameter": identity.parameter,
                "flat_index": identity.flat_index,
                "bit_index": identity.bit_index,
                "score": output.scores[identity],
                "signed_prediction": None if output.signed_predictions is None else output.signed_predictions[identity],
                "empirical_candidate": identity in candidates,
            }
            if replication is not None:
                row["replication"] = replication
            yield row


def create_study_bundle(outcome: StudyOutcome, *, repository_root: Path, run_id: str | None = None) -> Path:
    """Atomically write and finalize a schema-3 study bundle."""

    config = outcome.configuration
    measurement_version = config.schema_version
    members = outcome.replications or (outcome,)
    replicated = bool(outcome.replications)
    now = datetime.now(timezone.utc)
    identifier = run_id or f"{now.strftime('%Y%m%dT%H%M%S.%fZ')}-{uuid.uuid4().hex[:8]}"
    experiment_root = config.output.root / config.name
    destination, staging = experiment_root / identifier, experiment_root / f".{identifier}.incomplete"
    experiment_root.mkdir(parents=True, exist_ok=True)
    if destination.exists() or staging.exists():
        raise ResultBundleError(f"run directory already exists: {destination}")
    staging.mkdir()
    _write_bytes(staging / "config.toml", config.source_text.encode("utf-8"))
    resolved = config.resolved_document()
    _write_json(staging / "resolved_config.json", resolved)
    _write_json(staging / "scientific_payload.json", outcome.scientific_payload)
    derived = _derived_protection(outcome)
    summary = _summary(outcome, derived)
    _write_json(staging / "summary.json", summary)
    _write_json(staging / "artifacts/sample_partitions.json", ({member.replication_name: {name: value.to_dict() for name, value in member.partitions.items()} for member in members} if replicated else {name: value.to_dict() for name, value in outcome.partitions.items()}))
    _write_json(staging / "artifacts/candidate_selection.json", ({member.replication_name: list(member.candidate_selection) for member in members} if replicated else list(outcome.candidate_selection)))
    _write_json(staging / "artifacts/protection_masks.json", ({member.replication_name: member.protection_masks for member in members} if replicated else outcome.protection_masks))
    _write_json(staging / "artifacts/fault_schedules.json", ({member.replication_name: member.fault_schedules for member in members} if replicated else outcome.fault_schedules))

    empirical_fields = ["schema_version", "parameter", "flat_index", "bit_index", "layer", "parameter_class", "original_encoded", "flipped_encoded", "numerical_perturbation", "clean_loss", "faulted_loss", "signed_loss_difference", "positive_loss_damage", "absolute_loss_difference", "clean_accuracy", "faulted_accuracy", "accuracy_damage", "prediction_change_count", "evaluation_sample_sha256", "runtime_seconds", "inclusion_rule", "sampling_probability", "score_quantile"]
    if replicated:
        empirical_fields.insert(1, "replication")
    if any(member.empirical_measurements for member in members):
        _write_csv(staging / "measurements/empirical_bits.csv", empirical_fields, ({"schema_version": measurement_version, **({"replication": member.replication_name} if replicated else {}), **value.to_dict()} for member in members for value in member.empirical_measurements))
    prediction_fields = ["schema_version", "estimator", "implementation_version", "parameter", "flat_index", "bit_index", "score", "signed_prediction", "empirical_candidate"]
    if replicated:
        prediction_fields.insert(1, "replication")
    prediction_rows = tuple(row for member in members for row in _prediction_rows(member, replication=member.replication_name if replicated else None))
    _write_csv(staging / "measurements/estimator_predictions.csv", prediction_fields, prediction_rows)
    if any(member.ranking_metrics for member in members):
        metric_fields = ["schema_version", "estimator", "implementation_version", "scope", "candidate_count", "spearman_positive_damage", "spearman_absolute_change", "kendall_positive_damage", "zero_score_fraction", "tied_score_fraction", "runtime_seconds", "sample_limited", "top_5_recall", "top_5_overlap", "top_10_recall", "top_10_overlap", "top_20_recall", "top_20_overlap", "signed_mae", "signed_rmse", "signed_bias"]
        if replicated:
            metric_fields.insert(1, "replication")
        _write_csv(staging / "measurements/ranking_metrics.csv", metric_fields, ({"schema_version": measurement_version, **({"replication": member.replication_name} if replicated else {}), **row} for member in members for row in member.ranking_metrics))
    if any(member.estimator_agreement for member in members):
        fields = ["schema_version", "left", "right", "candidate_count", "spearman", "identical_scores", "identical_order", "changed_average_rank_count", "maximum_average_rank_shift", "top_10_overlap", "top_10_recall"]
        if replicated:
            fields.insert(1, "replication")
        _write_csv(staging / "measurements/estimator_agreement.csv", fields, ({"schema_version": measurement_version, **({"replication": member.replication_name} if replicated else {}), **row} for member in members for row in member.estimator_agreement))
    if any(member.trials for member in members):
        fields = ["schema_version", "trial", "policy", "schedule_sha256", "accuracy", "categorical_crossentropy", "schedule_hash", "proposed_events", "applied_events", "suppressed_events", "unique_applied_bits"]
        if replicated:
            fields.insert(1, "replication")
        _write_csv(staging / "measurements/trials.csv", fields, ({"schema_version": measurement_version, **{key: value for key, value in row.items() if key in fields and key != "schema_version"}} for row in _trial_rows(outcome)))
    if derived:
        def aggregate_row(value, replication=None):
            accuracy, loss = value["accuracy"], value["categorical_crossentropy"]
            return {
                "schema_version": measurement_version, **({"replication": replication} if replication else {}),
                "logical_ber": value["logical_ber"], "condition": value["condition"], "policy": value["policy"],
                "coverage": value["coverage"], "realized_coverage": value["realized_coverage"], "protected_bits": value["protected_bits"], "mask_sha256": value["ranking_mask_sha256"], "trial_count": value["trial_count"],
                "fault_realization_count": value["fault_realization_count"], "replication_count": value["replication_count"],
                "ranking_mask_count": value["ranking_mask_count"],
                "accuracy_mean": accuracy["mean"], "accuracy_standard_deviation": accuracy["standard_deviation"],
                "accuracy_standard_error": accuracy["standard_error"], "accuracy_ci_low": accuracy["confidence_interval"][0],
                "accuracy_ci_high": accuracy["confidence_interval"][1], "accuracy_median": accuracy["median"],
                "accuracy_quantile_05": accuracy["quantile_05"], "accuracy_minimum": accuracy["minimum"],
                "loss_mean": loss["mean"], "loss_standard_deviation": loss["standard_deviation"],
                "loss_standard_error": loss["standard_error"], "loss_ci_low": loss["confidence_interval"][0],
                "loss_ci_high": loss["confidence_interval"][1], "loss_median": loss["median"],
                "loss_quantile_05": loss["quantile_05"], "loss_maximum": loss["maximum"],
                "mean_proposed_events": value["proposed_events"]["mean"], "proposed_events_q05": value["proposed_events"]["quantile_05"], "proposed_events_q95": value["proposed_events"]["quantile_95"],
                "mean_applied_events": value["applied_events"]["mean"], "applied_events_q05": value["applied_events"]["quantile_05"], "applied_events_q95": value["applied_events"]["quantile_95"],
                "mean_suppressed_events": value["suppressed_events"]["mean"], "suppressed_events_q05": value["suppressed_events"]["quantile_05"], "suppressed_events_q95": value["suppressed_events"]["quantile_95"],
                "mean_unique_applied_bits": value["unique_applied_bits"]["mean"], "unique_applied_bits_q05": value["unique_applied_bits"]["quantile_05"], "unique_applied_bits_q95": value["unique_applied_bits"]["quantile_95"],
            }
        aggregate_fields = [
            "schema_version", "logical_ber", "condition", "policy", "coverage", "realized_coverage", "protected_bits", "mask_sha256", "trial_count", "fault_realization_count",
            "replication_count", "ranking_mask_count", "accuracy_mean", "accuracy_standard_deviation", "accuracy_standard_error",
            "accuracy_ci_low", "accuracy_ci_high", "accuracy_median", "accuracy_quantile_05", "accuracy_minimum",
            "loss_mean", "loss_standard_deviation", "loss_standard_error", "loss_ci_low", "loss_ci_high", "loss_median",
            "loss_quantile_05", "loss_maximum", "mean_proposed_events", "proposed_events_q05", "proposed_events_q95",
            "mean_applied_events", "applied_events_q05", "applied_events_q95", "mean_suppressed_events", "suppressed_events_q05", "suppressed_events_q95",
            "mean_unique_applied_bits", "unique_applied_bits_q05", "unique_applied_bits_q95",
        ]
        _write_csv(staging / "measurements/protection_results.csv", aggregate_fields, (aggregate_row(value) for value in derived["pooled"]))
        _write_csv(staging / "measurements/replication_protection_results.csv", ["schema_version", "replication", *aggregate_fields[1:]], (aggregate_row(value, value["replication"]) for value in derived["replications"]))
    if outcome.composition:
        fields = ["schema_version", "policy", "dimension", "category", "protected_count", "protected_fraction_of_mask", "coverage_within_category", "eligible_count"]
        _write_csv(staging / "measurements/mask_composition.csv", fields, ({"schema_version": measurement_version, **row} for row in outcome.composition))
    if derived and derived["paired"]:
        paired_fields = ["schema_version", "logical_ber", "coverage", "metric", "comparison", "left", "right", "matched_pairs", "mean_difference", "standard_deviation", "ci_low", "ci_high", "p_value_raw", "p_value_holm", "test", "pairing_identity", "multiplicity_family"]
        _write_csv(staging / "measurements/paired_comparisons.csv", paired_fields, ({
            "schema_version": measurement_version, "logical_ber": value["logical_ber"], "coverage": value["coverage"],
            "metric": value["metric"], "comparison": value["comparison"], "left": value["left"], "right": value["right"],
            "matched_pairs": value["matched_pairs"], "mean_difference": value["difference"]["mean"],
            "standard_deviation": value["difference"]["standard_deviation"], "ci_low": value["difference"]["confidence_interval"][0],
            "ci_high": value["difference"]["confidence_interval"][1], "p_value_raw": value["p_value_raw"],
            "p_value_holm": value["p_value_holm"], "test": value["permutation_test"],
            "pairing_identity": value["pairing_identity"], "multiplicity_family": value["multiplicity_family"],
        } for value in derived["paired"]))
        _write_csv(staging / "measurements/paired_differences.csv", ["schema_version", "logical_ber", "coverage", "metric", "comparison", "left", "right", "replication", "trial", "schedule_sha256", "difference"], ({"schema_version": measurement_version, **value} for value in derived["paired_differences"]))
        _write_csv(staging / "measurements/paired_replication_summaries.csv", ["schema_version", "replication", "logical_ber", "coverage", "metric", "comparison", "left", "right", "matched_pairs", "mean_difference", "standard_deviation", "ci_low", "ci_high"], ({
            "schema_version": measurement_version, "replication": value["replication"], "logical_ber": value["logical_ber"],
            "coverage": value["coverage"], "metric": value["metric"], "comparison": value["comparison"],
            "left": value["left"], "right": value["right"], "matched_pairs": value["matched_pairs"],
            "mean_difference": value["difference"]["mean"], "standard_deviation": value["difference"]["standard_deviation"],
            "ci_low": value["difference"]["confidence_interval"][0], "ci_high": value["difference"]["confidence_interval"][1],
        } for value in derived["paired_replications"]))
    runtime_fields = ["schema_version", "estimator", "implementation_version", "runtime_seconds"]
    if replicated:
        runtime_fields.insert(1, "replication")
    _write_csv(staging / "measurements/estimator_runtime.csv", runtime_fields, ({
        "schema_version": measurement_version, **({"replication": member.replication_name} if replicated else {}), "estimator": value.name,
        "implementation_version": value.implementation_version, "runtime_seconds": value.runtime_seconds,
    } for member in members for value in member.estimator_outputs))
    if outcome.replication_aggregate:
        fields = ["schema_version", "estimator", "scope", "metric", "replication_count", "mean", "standard_deviation", "minimum", "maximum", "statistical_unit"]
        _write_csv(staging / "measurements/replication_aggregate.csv", fields, ({"schema_version": measurement_version, **row} for row in outcome.replication_aggregate))

    figures: tuple[str, ...] = ()
    if config.output.plots:
        empirical_rows = tuple(value.to_dict() for value in members[0].empirical_measurements)
        generated = plot_study_results(
            summary, staging / "plots", empirical_rows=empirical_rows,
            prediction_rows=prediction_rows,
            trial_rows=_trial_rows(outcome),
        )
        figures = tuple(path.relative_to(staging).as_posix() for path in generated)
    if config.output.report:
        _write_bytes(staging / "report.md", render_study_report(summary, resolved, figure_paths=figures).encode("utf-8"))
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION, "run_id": identifier, "status": "complete",
        "source": git_provenance(Path(repository_root).resolve()), "runtime": runtime_provenance(),
        "versions": {
            "netsurf": __version__, "experiment_schema": config.schema_version,
            "manifest_schema": MANIFEST_SCHEMA_VERSION, "summary_schema": SUMMARY_SCHEMA_VERSION,
            "measurement_schema": measurement_version,
            "scientific_protocol": config.scientific_protocol,
            "estimators": {value.name: value.implementation_version for value in outcome.estimator_outputs},
        },
        "benchmark": {
            **resolved["benchmark"], "checkpoint_sha256": sha256_file(config.benchmark.checkpoint),
            "dataset_sha256": sha256_file(config.benchmark.dataset),
            "sample_partitions": ({member.replication_name: {name: value.to_dict() for name, value in member.partitions.items()} for member in members} if replicated else {name: value.to_dict() for name, value in outcome.partitions.items()}),
        },
        "experiment": {**resolved["experiment"], "requested_configuration": "config.toml", "resolved_configuration": "resolved_config.json"},
        "execution": {
            "samples": resolved["samples"], "candidates": resolved["candidates"], "estimators": resolved["estimators"],
            "policies": resolved["policies"], "fault": resolved["fault"], "protection": resolved["protection"],
            "randomization": resolved["execution"], "started_utc": now.isoformat(),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        },
        "outputs": _inventory(staging), "scientific_payload_sha256": outcome.scientific_payload_sha256,
    }
    _write_json(staging / "manifest.json", manifest)
    _write_json(staging / "checksums.json", {"schema_version": "1.0.0", "files": {entry["path"]: entry["sha256"] for entry in _inventory(staging)}})
    _write_bytes(staging / "FINALIZED", f"scientific_payload_sha256={outcome.scientific_payload_sha256}\n".encode("ascii"))
    staging.rename(destination)
    return destination
