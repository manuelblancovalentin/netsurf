from __future__ import annotations

from dataclasses import replace
import csv
from io import StringIO
from pathlib import Path

import pytest

from netsurf.derived_analysis import derive_protection_analysis
from netsurf.api import regenerate_report
from netsurf.progress import LiveProgressReporter, ProgressEvent
from netsurf.result_bundle import read_result_bundle
from netsurf.study_bundle import create_study_bundle
from netsurf.study_config import StudyExecution, StudyPolicy, StudyProtection, StudyReplications, load_study
from netsurf.study_engine import execute_study


ROOT = Path(__file__).resolve().parents[1]


def _row(replication, ber, trial, condition, accuracy, loss, schedule):
    return {
        "replication": replication, "trial": trial,
        "policy": f"ber={ber:.8g}/{condition}", "schedule_sha256": schedule,
        "accuracy": accuracy, "categorical_crossentropy": loss,
        "proposed_events": 10, "applied_events": 8 if condition != "full_protection" else 0,
        "suppressed_events": 2 if condition != "full_protection" else 10,
        "unique_applied_bits": 8 if condition != "full_protection" else 0,
    }


def test_pooled_reporting_uses_every_replication_and_exact_pair_identity():
    rows = []
    # The first replication is deliberately unlike the second. A
    # replication_0-only implementation produces 0.1 instead of 0.5.
    for replication, base in (("replication_0", 0.1), ("replication_1", 0.9)):
        for ber in (0.001, 0.01):
            for trial in (0, 1):
                schedule = f"{replication}-{ber}-{trial}"
                rows.extend((
                    _row(replication, ber, trial, "no_protection", base, 2.0 - base, schedule),
                    _row(replication, ber, trial, "gradient_global@0.100000", base + 0.05, 1.9 - base, schedule),
                    _row(replication, ber, trial, "msb+ordered@0.100000", base + 0.01, 1.95 - base, schedule),
                ))
    derived = derive_protection_analysis(rows, expected_replications=2, permutation_seed=7)
    assert derived["paired"] == derive_protection_analysis(rows, expected_replications=2, permutation_seed=7)["paired"]
    pooled = next(row for row in derived["pooled"] if row["logical_ber"] == 0.001 and row["policy"] == "no_protection")
    assert pooled["accuracy"]["mean"] == pytest.approx(0.5)
    assert pooled["fault_realization_count"] == 4
    assert pooled["replication_count"] == 2
    comparison = next(row for row in derived["paired"] if row["logical_ber"] == 0.001 and row["left"] == "gradient_global" and row["right"] == "msb+ordered" and row["metric"] == "accuracy")
    assert comparison["matched_pairs"] == 4
    assert comparison["difference"]["mean"] == pytest.approx(0.04)
    assert len([row for row in derived["paired_replications"] if row["logical_ber"] == 0.001 and row["comparison"] == comparison["comparison"] and row["metric"] == "accuracy"]) == 2


def test_replication_completeness_is_rejected_for_saved_bundle_reanalysis():
    rows = [_row("replication_0", 0.001, 0, "no_protection", 0.5, 1.0, "s0")]
    with pytest.raises(ValueError, match="requires 2"):
        derive_protection_analysis(rows, expected_replications=2)


def test_overall_eta_uses_monotonic_measured_work_and_completes_cleanly():
    stream = StringIO()
    times = iter((0.0, 0.0, 2.0, 4.0, 6.0))
    reporter = LiveProgressReporter(stream, interactive=False, refresh_seconds=0.0, clock=lambda: next(times))
    remaining = (("Fault evaluation", 4),)
    reporter(ProgressEvent("study", "Fault evaluation", 0, 4, work_class="Fault evaluation", remaining_work=remaining))
    reporter(ProgressEvent("study", "Fault evaluation", 1, 4, work_class="Fault evaluation", remaining_work=(("Fault evaluation", 3),)))
    reporter(ProgressEvent("study", "Fault evaluation", 2, 4, work_class="Fault evaluation", remaining_work=(("Fault evaluation", 2),)))
    reporter(ProgressEvent("study", "Fault evaluation", 4, 4, work_class="Fault evaluation", remaining_work=(("Fault evaluation", 0),)))
    text = stream.getvalue()
    assert "Estimating..." in text
    assert "ETA 00:0" in text
    assert "Completed in 00:06" in text
    assert "calculating" not in text


def test_rich_narrow_terminal_uses_live_rendering_without_custom_cursor_code():
    pytest.importorskip("rich")
    stream = StringIO()
    times = iter((0.0, 0.0, 1.0))
    reporter = LiveProgressReporter(stream, interactive=True, width=48, refresh_seconds=0.0, clock=lambda: next(times))
    reporter(ProgressEvent("narrow", "Fault evaluation", 0, 1, policy="a-very-long-policy-name-that-must-not-share-the-bar", work_class="Fault evaluation", remaining_work=(("Fault evaluation", 1),)))
    reporter(ProgressEvent("narrow", "Fault evaluation", 1, 1, policy="a-very-long-policy-name-that-must-not-share-the-bar", work_class="Fault evaluation", remaining_work=(("Fault evaluation", 0),)))
    assert reporter._rich_available
    assert "Completed in" in stream.getvalue()


def test_wide_ber_configuration_is_maintained_and_valid():
    configuration = load_study(ROOT / "config/experiments/mnist_policy_ber_wide_v1.toml")
    assert configuration.name == "mnist_policy_ber_wide_v1"
    assert configuration.protection.probabilities == (0.0001, 0.001, 0.01, 0.1)
    assert configuration.replications.count == 5


def test_replicated_bundle_writes_pooled_and_replication_tables(tmp_path):
    configuration = load_study(ROOT / "config/experiments/numpy_qpolar_cancellation.toml")
    estimator = replace(configuration.estimators[0], name="msb_first", implementation_version="encoded-bit-index/v1", result_name="msb_first")
    configuration = replace(
        configuration, name="replicated_bundle_reporting", study_type="protection",
        scientific_protocol="multi-coverage-protection/v1", estimators=(estimator,),
        policies=(StudyPolicy("msb+ordered", "msb_first", ("bit_significance_desc", "canonical"), None, True),),
        protection=StudyProtection((0.1,), "perfect_event_suppression", (0.01,)),
        execution=StudyExecution(2, 0.95, 701, 801),
        replications=StudyReplications(2, 0, 0, 0, 100),
        output=replace(configuration.output, root=tmp_path / "runs", plots=True),
    )
    outcome = execute_study(configuration)
    destination = create_study_bundle(outcome, repository_root=ROOT, run_id="pooled")
    bundle = read_result_bundle(destination)
    no_protection = next(row for row in bundle.summary["protection_results"] if row["policy"] == "no_protection")
    assert no_protection["fault_realization_count"] == 4
    assert no_protection["replication_count"] == 2
    assert no_protection["ranking_mask_count"] == 1
    assert len(bundle.summary["replication_protection_results"]) == 6
    assert (destination / "measurements/replication_protection_results.csv").is_file()
    assert (destination / "measurements/paired_differences.csv").is_file()
    assert (destination / "plots/trial_accuracy_distributions_ber_1e-02.png").is_file()
    with (destination / "measurements/protection_results.csv").open(newline="", encoding="utf-8") as stream:
        pooled_rows = tuple(csv.DictReader(stream))
    assert {int(row["fault_realization_count"]) for row in pooled_rows} == {4}
    with (destination / "measurements/paired_comparisons.csv").open(newline="", encoding="utf-8") as stream:
        paired_rows = tuple(csv.DictReader(stream))
    assert paired_rows and {int(row["matched_pairs"]) for row in paired_rows} == {4}
    payload_hash = bundle.scientific_payload_sha256
    report = regenerate_report(bundle, tmp_path / "derived", plots=True)
    assert report.is_file()
    assert (report.parent / "derived_summary.json").is_file()
    assert (report.parent / "measurements/protection_results.csv").is_file()
    assert read_result_bundle(destination).scientific_payload_sha256 == payload_hash
