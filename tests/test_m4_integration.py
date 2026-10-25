from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import pytest

from netsurf.api import regenerate_report
from netsurf.experiment_config import OutputConfiguration
from netsurf.progress import ProgressTracker, search_operation_plan
from netsurf.result_bundle import read_result_bundle
from netsurf.search_bundle import create_search_bundle
from netsurf.search_config import SearchConfigurationError, load_search
from netsurf.search_engine import execute_search


ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "config/experiments/numpy_pgcs_exact.toml"


def test_all_maintained_m4_configurations_validate():
    for name in (
        "numpy_pgcs_exact.toml", "mnist_incremental_validation.toml",
        "mnist_static_guided_search.toml", "mnist_polar_guided_search.toml",
    ):
        assert load_search(ROOT / "config/experiments" / name).schema_version == "4.0.0"


def test_polar_strategy_without_polar_is_rejected(tmp_path):
    text = CONTROL.read_text().replace("enabled = true", "enabled = false", 1)
    path = tmp_path / "bad.toml"; path.write_text(text)
    with pytest.raises(SearchConfigurationError, match="requires polar.enabled"):
        load_search(path, check_inputs=False)


def test_search_is_deterministic_uses_matched_heldout_schedules_and_disjoint_samples():
    config = load_search(CONTROL)
    first = execute_search(config); replay = execute_search(config)
    assert first.scientific_payload_sha256 == replay.scientific_payload_sha256
    partition_indices = [set(value.indices) for value in first.partitions.values()]
    assert all(left.isdisjoint(right) for index, left in enumerate(partition_indices) for right in partition_indices[index + 1:])
    by_strategy = {}
    for row in first.final_trials:
        by_strategy.setdefault(row["strategy"], []).append((row["trial"], row["schedule_sha256"], row["fault_count"]))
    assert len(by_strategy) > 1
    assert len({tuple(value) for value in by_strategy.values()}) == 1
    greedy = next(value for value in first.search_results if value.strategy == "conditional_greedy")
    assert all(len(step.evaluated_candidates) > 1 for step in greedy.steps)
    greedy_payload = next(value for value in first.scientific_payload["search"] if value["strategy"] == "conditional_greedy")
    assert greedy_payload["steps"][0]["background_fault_states"]


def test_search_progress_reaches_declared_total():
    config = load_search(CONTROL)
    events = []
    tracker = ProgressTracker(config.name, search_operation_plan(config), events.append)
    execute_search(config, _tracker=tracker); tracker.finish()
    assert events[-1].completed == events[-1].total
    assert any(event.iteration == config.search.iterations and event.exact_evaluations for event in events)


def test_schema4_bundle_verifies_and_report_regenerates(tmp_path):
    config = load_search(CONTROL)
    config = replace(config, output=OutputConfiguration(tmp_path / "runs", True, False))
    outcome = execute_search(config)
    path = create_search_bundle(outcome, repository_root=ROOT, run_id="bounded")
    result = read_result_bundle(path)
    assert result.scientific_payload_sha256 == outcome.scientific_payload_sha256
    assert (path / "measurements/final_trials.csv").is_file()
    assert (path / "measurements/mask_overlap.csv").is_file()
    with (path / "measurements/search_results.csv").open(newline="") as stream:
        search_rows = list(csv.DictReader(stream))
    assert search_rows
    assert all("iteration_to_best" in row and "seconds_to_best" in row for row in search_rows)
    with (path / "measurements/mask_overlap.csv").open(newline="") as stream:
        overlap_rows = list(csv.DictReader(stream))
    assert overlap_rows
    assert all(0.0 <= float(row["jaccard"]) <= 1.0 for row in overlap_rows)
    regenerated = regenerate_report(result, tmp_path / "derived", plots=False)
    report = regenerated.read_text()
    assert "Search decisions use matched three-state" in report
    assert "Accuracy q05" in report
