"""Small public API shared by command-line and notebook workflows."""

from __future__ import annotations

from pathlib import Path
import csv
import json
import resource
import tomllib
from time import monotonic

from .experiment_config import ExperimentConfiguration, load_experiment
from .derived_analysis import derive_protection_analysis
from .manifests import digest_object
from .parallel import ExecutionResources
from .experiment_engine import ExperimentOutcome, execute_experiment
from .plotting import plot_protection_results, plot_study_results
from .progress import ProgressTracker, experiment_operation_plan, search_operation_plan, study_operation_plan
from .reporting import render_study_report, write_report
from .result_bundle import ResultBundle, create_result_bundle, read_result_bundle
from .study_bundle import create_study_bundle
from .study_config import StudyConfiguration, load_study
from .study_engine import StudyOutcome, execute_study
from .search_bundle import create_search_bundle
from .search_config import SearchConfiguration, load_search
from .search_engine import execute_search
from .search_reporting import plot_search_results, render_search_report


def _write_derived_analysis(destination: Path, derived: dict) -> None:
    """Write corrected presentation tables without touching the source bundle."""

    measurement_root = destination / "measurements"
    measurement_root.mkdir(parents=True, exist_ok=True)

    def write(name, fields, rows):
        with (measurement_root / name).open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    aggregate_fields = [
        "replication", "logical_ber", "condition", "policy", "coverage", "realized_coverage", "protected_bits", "mask_sha256", "trial_count",
        "fault_realization_count", "replication_count", "ranking_mask_count", "accuracy_mean",
        "accuracy_standard_deviation", "accuracy_standard_error", "accuracy_ci_low", "accuracy_ci_high",
        "accuracy_median", "accuracy_quantile_05", "accuracy_minimum", "loss_mean",
        "loss_standard_deviation", "loss_standard_error", "loss_ci_low", "loss_ci_high",
        "loss_median", "loss_quantile_05", "loss_maximum", "mean_proposed_events", "proposed_events_q05", "proposed_events_q95",
        "mean_applied_events", "applied_events_q05", "applied_events_q95", "mean_suppressed_events", "suppressed_events_q05", "suppressed_events_q95",
        "mean_unique_applied_bits", "unique_applied_bits_q05", "unique_applied_bits_q95",
    ]

    def aggregate(value, replication=""):
        accuracy, loss = value["accuracy"], value["categorical_crossentropy"]
        return {
            "replication": replication, "logical_ber": value["logical_ber"], "condition": value["condition"],
            "policy": value["policy"], "coverage": value["coverage"], "realized_coverage": value["realized_coverage"],
            "protected_bits": value["protected_bits"], "mask_sha256": value["ranking_mask_sha256"], "trial_count": value["trial_count"],
            "fault_realization_count": value["fault_realization_count"], "replication_count": value["replication_count"],
            "ranking_mask_count": value["ranking_mask_count"], "accuracy_mean": accuracy["mean"],
            "accuracy_standard_deviation": accuracy["standard_deviation"], "accuracy_standard_error": accuracy["standard_error"],
            "accuracy_ci_low": accuracy["confidence_interval"][0], "accuracy_ci_high": accuracy["confidence_interval"][1],
            "accuracy_median": accuracy["median"], "accuracy_quantile_05": accuracy["quantile_05"], "accuracy_minimum": accuracy["minimum"],
            "loss_mean": loss["mean"], "loss_standard_deviation": loss["standard_deviation"], "loss_standard_error": loss["standard_error"],
            "loss_ci_low": loss["confidence_interval"][0], "loss_ci_high": loss["confidence_interval"][1],
            "loss_median": loss["median"], "loss_quantile_05": loss["quantile_05"], "loss_maximum": loss["maximum"],
            "mean_proposed_events": value["proposed_events"]["mean"], "proposed_events_q05": value["proposed_events"]["quantile_05"], "proposed_events_q95": value["proposed_events"]["quantile_95"],
            "mean_applied_events": value["applied_events"]["mean"], "applied_events_q05": value["applied_events"]["quantile_05"], "applied_events_q95": value["applied_events"]["quantile_95"],
            "mean_suppressed_events": value["suppressed_events"]["mean"], "suppressed_events_q05": value["suppressed_events"]["quantile_05"], "suppressed_events_q95": value["suppressed_events"]["quantile_95"],
            "mean_unique_applied_bits": value["unique_applied_bits"]["mean"], "unique_applied_bits_q05": value["unique_applied_bits"]["quantile_05"], "unique_applied_bits_q95": value["unique_applied_bits"]["quantile_95"],
        }

    write("protection_results.csv", aggregate_fields[1:], ({key: value for key, value in aggregate(row).items() if key != "replication"} for row in derived["pooled"]))
    write("replication_protection_results.csv", aggregate_fields, (aggregate(row, row["replication"]) for row in derived["replications"]))
    paired_fields = ["logical_ber", "coverage", "metric", "comparison", "left", "right", "matched_pairs", "mean_difference", "standard_deviation", "ci_low", "ci_high", "p_value_raw", "p_value_holm", "test", "pairing_identity", "multiplicity_family"]
    write("paired_comparisons.csv", paired_fields, ({
        "logical_ber": row["logical_ber"], "coverage": row["coverage"], "metric": row["metric"],
        "comparison": row["comparison"], "left": row["left"], "right": row["right"], "matched_pairs": row["matched_pairs"],
        "mean_difference": row["difference"]["mean"], "standard_deviation": row["difference"]["standard_deviation"],
        "ci_low": row["difference"]["confidence_interval"][0], "ci_high": row["difference"]["confidence_interval"][1],
        "p_value_raw": row["p_value_raw"], "p_value_holm": row["p_value_holm"], "test": row["permutation_test"],
        "pairing_identity": row["pairing_identity"], "multiplicity_family": row["multiplicity_family"],
    } for row in derived["paired"]))
    difference_fields = ["logical_ber", "coverage", "metric", "comparison", "left", "right", "replication", "trial", "schedule_sha256", "difference"]
    write("paired_differences.csv", difference_fields, derived["paired_differences"])
    replication_fields = ["replication", "logical_ber", "coverage", "metric", "comparison", "left", "right", "matched_pairs", "mean_difference", "standard_deviation", "ci_low", "ci_high"]
    write("paired_replication_summaries.csv", replication_fields, ({
        "replication": row["replication"], "logical_ber": row["logical_ber"], "coverage": row["coverage"],
        "metric": row["metric"], "comparison": row["comparison"], "left": row["left"], "right": row["right"],
        "matched_pairs": row["matched_pairs"], "mean_difference": row["difference"]["mean"],
        "standard_deviation": row["difference"]["standard_deviation"], "ci_low": row["difference"]["confidence_interval"][0],
        "ci_high": row["difference"]["confidence_interval"][1],
    } for row in derived["paired_replications"]))


def load_configuration(path: str | Path):
    """Load the explicitly versioned configuration supported by its schema."""

    source = Path(path).expanduser().resolve()
    with source.open("rb") as stream:
        version = tomllib.load(stream).get("schema_version")
    if version in {"3.0.0", "3.1.0", "3.2.0"}:
        return load_study(source)
    if version in {"4.0.0", "4.1.0", "4.2.0", "4.3.0", "4.4.0"}:
        return load_search(source)
    return load_experiment(source)


def configuration_operation_plan(configuration) -> dict[str, int]:
    """Return the same measured-work plan used by the scientific tracker."""
    if isinstance(configuration, SearchConfiguration):
        return search_operation_plan(configuration, include_bundle=True)
    if isinstance(configuration, StudyConfiguration):
        return study_operation_plan(configuration, include_bundle=True)
    if isinstance(configuration, ExperimentConfiguration):
        return experiment_operation_plan(configuration, include_bundle=True)
    raise TypeError(f"unsupported experiment configuration: {type(configuration).__name__}")


def run_experiment(
    configuration: ExperimentConfiguration | str | Path,
    *,
    repository_root: Path | None = None,
    progress=None,
    resources: ExecutionResources | None = None,
    execution_provenance_overrides: dict | None = None,
) -> Path:
    """Validate, execute, and atomically save an experiment."""
    root = repository_root or Path(__file__).resolve().parents[1]
    config = load_configuration(configuration) if isinstance(configuration, (str, Path)) else configuration
    requested = resources or ExecutionResources()
    requested.validate_capacity()
    started = monotonic()
    def provenance():
        value = requested.provenance(elapsed_seconds=monotonic() - started)
        value["coordinator_peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        value["maximum_completed_child_rss_mb"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024.0
        value.update(execution_provenance_overrides or {})
        return value
    if isinstance(config, SearchConfiguration):
        tracker = ProgressTracker(config.name, search_operation_plan(config, include_bundle=True), progress)
        try:
            outcome = execute_search(config, _tracker=tracker, resources=requested)
            tracker.update("Result bundle generation")
            units = (len(config.initial_mask.budgets_bits) or 1) * len(config.fault_probabilities) * config.replications.count
            execution = provenance()
            execution.update(outcome.execution_metrics or {})
            execution["effective_workers"] = min(requested.workers, units) if units > 1 else 1
            execution["parallel_boundary"] = "BER/budget/search-replication" if units > 1 else "sequential single logical unit"
            destination = create_search_bundle(outcome, repository_root=root, execution_provenance=execution)
            tracker.update("Result bundle generation", advance=1)
            tracker.finish()
            return destination
        except BaseException as error:
            tracker.fail(error)
            raise
    if isinstance(config, StudyConfiguration):
        tracker = ProgressTracker(config.name, study_operation_plan(config, include_bundle=True), progress)
        try:
            outcome = execute_study(config, _tracker=tracker, resources=requested)
            tracker.update("Result bundle generation")
            execution = provenance()
            execution.update(outcome.execution_metrics or {})
            execution["effective_workers"] = min(requested.workers, config.replications.count) if config.replications.count > 1 else 1
            execution["parallel_boundary"] = "independent configured replication" if config.replications.count > 1 else "sequential single replication"
            destination = create_study_bundle(outcome, repository_root=root, execution_provenance=execution)
            tracker.update("Result bundle generation", advance=1)
            tracker.finish()
            return destination
        except BaseException as error:
            tracker.fail(error)
            raise
    tracker = ProgressTracker(config.name, experiment_operation_plan(config, include_bundle=True), progress)
    try:
        outcome = execute_experiment(config, _tracker=tracker)
        tracker.update("Result bundle generation")
        execution = provenance()
        execution["effective_workers"] = 1
        execution["parallel_boundary"] = "sequential reference; no replicated top-level unit"
        destination = create_result_bundle(outcome, repository_root=root, execution_provenance=execution)
        tracker.update("Result bundle generation", advance=1)
        tracker.finish()
        return destination
    except BaseException as error:
        tracker.fail(error)
        raise


def regenerate_report(
    bundle: ResultBundle | str | Path,
    output: str | Path,
    *,
    plots: bool = False,
) -> Path:
    """Regenerate presentation artifacts without running scientific code."""
    result = read_result_bundle(bundle) if isinstance(bundle, (str, Path)) else bundle
    destination = Path(output).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    figures: tuple[str, ...] = ()
    is_study = result.resolved_config.get("schema_version") in {"3.0.0", "3.1.0", "3.2.0"}
    is_search = result.resolved_config.get("schema_version") in {"4.0.0", "4.1.0", "4.2.0", "4.3.0", "4.4.0"}
    summary = result.summary
    derived = None
    trial_rows = ()
    if is_study:
        trial_path = result.path / "measurements/trials.csv"
        if trial_path.is_file():
            with trial_path.open(newline="", encoding="utf-8") as stream:
                trial_rows = tuple(dict(row) for row in csv.DictReader(stream))
            mask_path = result.path / "artifacts/protection_masks.json"
            if mask_path.is_file():
                masks = json.loads(mask_path.read_text(encoding="utf-8"))
                replicated_masks = int(result.resolved_config.get("replications", {}).get("count", 1)) > 1
                eligible = int(result.summary.get("eligible_bits", 0))
                enriched = []
                for row in trial_rows:
                    condition = str(row["policy"]).split("/", 1)[-1]
                    replication = str(row.get("replication") or "replication_0")
                    collection = masks.get(replication, {}) if replicated_masks else masks
                    identities = collection.get(condition)
                    if identities is not None:
                        row = {**row, "ranking_mask_sha256": digest_object(identities), "protected_bits": len(identities), "realized_coverage": len(identities) / eligible if eligible else 0.0}
                    enriched.append(row)
                trial_rows = tuple(enriched)
            expected = int(result.resolved_config.get("replications", {}).get("count", 1))
            derived = derive_protection_analysis(
                trial_rows,
                confidence=float(result.resolved_config["execution"]["confidence"]),
                permutation_seed=int(result.resolved_config["execution"]["permutation_seed"]),
                default_ber=float((result.resolved_config["protection"].get("probabilities") or [result.resolved_config["fault"]["probability"]])[0]),
                expected_replications=expected,
            )
            summary = {
                **result.summary,
                "protection_results": derived["pooled"],
                "replication_protection_results": derived["replications"],
                "paired_comparisons": derived["paired"],
                "paired_replication_summaries": derived["paired_replications"],
                "pooled_fault_trial_count": max((row["fault_realization_count"] for row in derived["pooled"]), default=0),
                "statistical_unit": derived["statistical_unit"],
                "interval_interpretation": derived["interval_interpretation"],
            }
            (destination / "derived_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            _write_derived_analysis(destination, derived)
    if is_search:
        def search_rows(name):
            path = result.path / "measurements" / name
            if not path.is_file(): return ()
            with path.open(newline="", encoding="utf-8") as stream:
                return tuple(dict(row) for row in csv.DictReader(stream))
        figures = ()
        if plots:
            generated = plot_search_results(summary, destination / "plots", trials=search_rows("final_trials.csv"), convergence=search_rows("convergence.csv"), validation=search_rows("incremental_validation.csv"), polar=search_rows("polar_interactions.csv"), topology=search_rows("topology_interactions.csv"), search_steps=search_rows("search_steps.csv"), composition=search_rows("mask_composition.csv"), search_results=search_rows("search_results.csv"), proposal_diagnostics=search_rows("proposal_diagnostics.csv"), screening=search_rows("candidate_screening.csv"), factorized=search_rows("factorized_guidance.csv"), timing=search_rows("timing_phases.csv"))
            figures = tuple(path.relative_to(destination).as_posix() for path in generated)
        output_path = destination / "report.md"
        output_path.write_text(render_search_report(summary, result.resolved_config, figure_paths=figures), encoding="utf-8")
        return output_path
    if plots and is_study:
        def rows(name):
            path = result.path / "measurements" / name
            if not path.is_file():
                return ()
            with path.open(newline="", encoding="utf-8") as stream:
                return tuple(dict(row) for row in csv.DictReader(stream))
        empirical_rows = rows("empirical_bits.csv")
        prediction_rows = rows("estimator_predictions.csv")
        trial_rows = trial_rows or rows("trials.csv")
        # A bundle-level plot depicts one coherent replication; aggregate
        # dispersion is already reported from replication_aggregate.csv.
        # Never join identical bit identities across different replications.
        if any("replication" in row for row in empirical_rows + prediction_rows):
            names = sorted({row.get("replication", "") for row in empirical_rows + prediction_rows})
            selected = names[0]
            empirical_rows = tuple(row for row in empirical_rows if row.get("replication") == selected)
            prediction_rows = tuple(row for row in prediction_rows if row.get("replication") == selected)
        generated = plot_study_results(
            summary, destination / "plots",
            empirical_rows=empirical_rows, prediction_rows=prediction_rows,
            trial_rows=trial_rows,
        )
        figures = tuple(path.relative_to(destination).as_posix() for path in generated)
    elif plots:
        generated = plot_protection_results(result.summary, destination / "plots")
        figures = tuple(path.relative_to(destination).as_posix() for path in generated)
    if is_study:
        output_path = destination / "report.md"
        output_path.write_text(render_study_report(summary, result.resolved_config, figure_paths=figures), encoding="utf-8")
        return output_path
    return write_report(result.summary, result.resolved_config, destination / "report.md", figure_paths=figures)


__all__ = [
    "ExperimentConfiguration", "ExperimentOutcome", "StudyConfiguration", "StudyOutcome", "ResultBundle",
    "load_configuration", "configuration_operation_plan", "load_experiment", "load_study", "execute_study",
    "read_result_bundle", "regenerate_report", "run_experiment",
]
