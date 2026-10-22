"""Immutable schema-3 result bundles for M3 scientific studies."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import uuid

from . import __version__
from .experiment_config import MANIFEST_SCHEMA_VERSION, SUMMARY_SCHEMA_VERSION
from .manifests import git_provenance, runtime_provenance, sha256_file
from .plotting import plot_study_results
from .reporting import render_study_report
from .result_bundle import ResultBundleError, _inventory, _write_bytes, _write_csv, _write_json
from .study_engine import M3_MEASUREMENT_SCHEMA_VERSION, StudyOutcome


def _summary(outcome: StudyOutcome):
    payload = outcome.scientific_payload
    overall = [dict(row) for row in outcome.ranking_metrics if row["scope"] == "overall"]
    runtimes = {value.name: value.runtime_seconds for value in outcome.estimator_outputs}
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "study_schema_version": outcome.configuration.schema_version,
        "scientific_protocol": outcome.configuration.scientific_protocol,
        "experiment_name": outcome.configuration.name,
        "benchmark_id": outcome.configuration.benchmark.id,
        "eligible_bits": payload["benchmark"]["eligible_bits"],
        "trial_count": outcome.configuration.execution.trials if outcome.trials else 0,
        "clean_evaluation": payload["clean_evaluation"],
        "empirical_candidate_count": len(outcome.empirical_measurements),
        "sample_limited": any(value.sampling_probability < 1.0 for value in outcome.empirical_measurements),
        "ranking_metrics_overall": overall,
        "protection_results": list(outcome.protection_results),
        "paired_comparisons": outcome.paired_comparisons,
        "mask_composition": list(outcome.composition),
        "estimator_runtimes_seconds": runtimes,
        "checks": payload["checks"],
        "scientific_payload_sha256": outcome.scientific_payload_sha256,
    }


def _prediction_rows(outcome: StudyOutcome):
    candidates = {value.identity for value in outcome.empirical_measurements}
    for output in outcome.estimator_outputs:
        for identity in sorted(output.scores):
            yield {
                "schema_version": M3_MEASUREMENT_SCHEMA_VERSION,
                "estimator": output.name,
                "implementation_version": output.implementation_version,
                "parameter": identity.parameter,
                "flat_index": identity.flat_index,
                "bit_index": identity.bit_index,
                "score": output.scores[identity],
                "signed_prediction": None if output.signed_predictions is None else output.signed_predictions[identity],
                "empirical_candidate": identity in candidates,
            }


def create_study_bundle(outcome: StudyOutcome, *, repository_root: Path, run_id: str | None = None) -> Path:
    """Atomically write and finalize a schema-3 study bundle."""

    config = outcome.configuration
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
    summary = _summary(outcome)
    _write_json(staging / "summary.json", summary)
    _write_json(staging / "artifacts/sample_partitions.json", {name: value.to_dict() for name, value in outcome.partitions.items()})
    _write_json(staging / "artifacts/candidate_selection.json", list(outcome.candidate_selection))
    _write_json(staging / "artifacts/protection_masks.json", outcome.protection_masks)
    _write_json(staging / "artifacts/fault_schedules.json", outcome.fault_schedules)

    empirical_fields = ["schema_version", "parameter", "flat_index", "bit_index", "layer", "parameter_class", "original_encoded", "flipped_encoded", "numerical_perturbation", "clean_loss", "faulted_loss", "signed_loss_difference", "positive_loss_damage", "absolute_loss_difference", "clean_accuracy", "faulted_accuracy", "accuracy_damage", "prediction_change_count", "evaluation_sample_sha256", "runtime_seconds", "inclusion_rule", "sampling_probability", "score_quantile"]
    if outcome.empirical_measurements:
        _write_csv(staging / "measurements/empirical_bits.csv", empirical_fields, ({"schema_version": M3_MEASUREMENT_SCHEMA_VERSION, **value.to_dict()} for value in outcome.empirical_measurements))
    prediction_fields = ["schema_version", "estimator", "implementation_version", "parameter", "flat_index", "bit_index", "score", "signed_prediction", "empirical_candidate"]
    prediction_rows = tuple(_prediction_rows(outcome))
    _write_csv(staging / "measurements/estimator_predictions.csv", prediction_fields, prediction_rows)
    if outcome.ranking_metrics:
        metric_fields = ["schema_version", "estimator", "implementation_version", "scope", "candidate_count", "spearman_positive_damage", "spearman_absolute_change", "kendall_positive_damage", "zero_score_fraction", "tied_score_fraction", "runtime_seconds", "sample_limited", "top_5_recall", "top_5_overlap", "top_10_recall", "top_10_overlap", "top_20_recall", "top_20_overlap", "signed_mae", "signed_rmse", "signed_bias"]
        _write_csv(staging / "measurements/ranking_metrics.csv", metric_fields, ({"schema_version": M3_MEASUREMENT_SCHEMA_VERSION, **row} for row in outcome.ranking_metrics))
    if outcome.trials:
        fields = ["schema_version", "trial", "policy", "schedule_sha256", "accuracy", "categorical_crossentropy", "schedule_hash", "proposed_events", "applied_events", "suppressed_events", "unique_applied_bits"]
        _write_csv(staging / "measurements/trials.csv", fields, ({"schema_version": M3_MEASUREMENT_SCHEMA_VERSION, **row.to_dict()} for row in outcome.trials))
    if outcome.protection_results:
        fields = ["schema_version", "policy", "protected_bits", "coverage", "mask_sha256", "accuracy_mean", "accuracy_ci_low", "accuracy_ci_high", "loss_mean", "loss_ci_low", "loss_ci_high", "mean_applied_events", "mean_suppressed_events", "minimum_accuracy", "minimum_accuracy_trial"]
        rows = []
        for value in outcome.protection_results:
            rows.append({
                "schema_version": M3_MEASUREMENT_SCHEMA_VERSION, "policy": value["policy"],
                "protected_bits": value["protected_bits"], "coverage": value["coverage"], "mask_sha256": value["mask_sha256"],
                "accuracy_mean": value["accuracy"]["mean"], "accuracy_ci_low": value["accuracy"]["confidence_interval"][0], "accuracy_ci_high": value["accuracy"]["confidence_interval"][1],
                "loss_mean": value["categorical_crossentropy"]["mean"], "loss_ci_low": value["categorical_crossentropy"]["confidence_interval"][0], "loss_ci_high": value["categorical_crossentropy"]["confidence_interval"][1],
                "mean_applied_events": value["mean_applied_events"], "mean_suppressed_events": value["mean_suppressed_events"],
                "minimum_accuracy": value["minimum_accuracy"], "minimum_accuracy_trial": value["minimum_accuracy_trial"],
            })
        _write_csv(staging / "measurements/protection_results.csv", fields, rows)
    if outcome.composition:
        fields = ["schema_version", "policy", "dimension", "category", "protected_count", "protected_fraction_of_mask", "coverage_within_category", "eligible_count"]
        _write_csv(staging / "measurements/mask_composition.csv", fields, ({"schema_version": M3_MEASUREMENT_SCHEMA_VERSION, **row} for row in outcome.composition))
    if outcome.paired_comparisons:
        fields = ["schema_version", "metric", "comparison", "left", "right", "mean_difference", "ci_low", "ci_high", "p_value_raw", "p_value_holm", "test"]
        rows = []
        for metric in ("accuracy", "categorical_crossentropy"):
            for name, value in outcome.paired_comparisons.get(metric, {}).items():
                rows.append({
                    "schema_version": M3_MEASUREMENT_SCHEMA_VERSION, "metric": metric, "comparison": name,
                    "left": value["left"], "right": value["right"], "mean_difference": value["difference"]["mean"],
                    "ci_low": value["difference"]["confidence_interval"][0], "ci_high": value["difference"]["confidence_interval"][1],
                    "p_value_raw": value["p_value_raw"], "p_value_holm": value["p_value_holm"], "test": value["permutation_test"],
                })
        _write_csv(staging / "measurements/paired_comparisons.csv", fields, rows)
    _write_csv(staging / "measurements/estimator_runtime.csv", ["schema_version", "estimator", "implementation_version", "runtime_seconds"], ({
        "schema_version": M3_MEASUREMENT_SCHEMA_VERSION, "estimator": value.name,
        "implementation_version": value.implementation_version, "runtime_seconds": value.runtime_seconds,
    } for value in outcome.estimator_outputs))

    figures: tuple[str, ...] = ()
    if config.output.plots:
        empirical_rows = tuple(value.to_dict() for value in outcome.empirical_measurements)
        generated = plot_study_results(
            summary, staging / "plots", empirical_rows=empirical_rows,
            prediction_rows=prediction_rows,
            trial_rows=tuple(value.to_dict() for value in outcome.trials),
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
            "measurement_schema": M3_MEASUREMENT_SCHEMA_VERSION,
            "scientific_protocol": config.scientific_protocol,
            "estimators": {value.name: value.implementation_version for value in outcome.estimator_outputs},
        },
        "benchmark": {
            **resolved["benchmark"], "checkpoint_sha256": sha256_file(config.benchmark.checkpoint),
            "dataset_sha256": sha256_file(config.benchmark.dataset),
            "sample_partitions": {name: value.to_dict() for name, value in outcome.partitions.items()},
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
