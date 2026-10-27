from __future__ import annotations

import pytest
from dataclasses import replace
from pathlib import Path

from netsurf.experiment_config import OutputConfiguration
from netsurf.search_bundle import _union_fields, create_search_bundle
from netsurf.search_config import load_search
from netsurf.search_engine import execute_search
from netsurf.timing import ExclusiveTimingLedger


ROOT = Path(__file__).resolve().parents[1]


class FakeClock:
    def __init__(self): self.value = 10.0
    def __call__(self): return self.value
    def advance(self, seconds): self.value += seconds


def test_exclusive_timing_matches_monotonic_elapsed_without_double_counting():
    clock = FakeClock(); ledger = ExclusiveTimingLedger(clock)
    clock.advance(0.5)  # explicitly uninstrumented setup
    with ledger.phase("ranking", "shared_preprocessing", operation_count=1): clock.advance(2.0)
    with ledger.phase("search", "search_proposal", operation_count=7): clock.advance(3.0)
    clock.advance(0.25)
    result = ledger.summary()
    assert result["whole_process_elapsed_seconds"] == pytest.approx(5.75)
    assert result["exclusive_phase_seconds"] == pytest.approx(5.0)
    assert result["uninstrumented_seconds"] == pytest.approx(0.75)
    assert sum(row.operation_count for row in result["records"]) == 8


def test_exclusive_timing_rejects_nested_intervals():
    clock = FakeClock(); ledger = ExclusiveTimingLedger(clock)
    with ledger.phase("outer", "setup"):
        with pytest.raises(RuntimeError, match="cannot be nested"):
            with ledger.phase("inner", "setup"):
                pass


def test_optional_measurement_columns_are_unioned_across_mixed_methods():
    assert _union_fields(({"strategy": "sparse", "score": 1}, {"strategy": "factorized", "population": "full"})) == ["strategy", "score", "population"]


def test_schema43_engine_phase_sum_does_not_exceed_whole_process_and_bundle_is_auditable(tmp_path):
    config = load_search(ROOT / "config/experiments/numpy_conditional_screening_exact.toml")
    config = replace(config, output=OutputConfiguration(tmp_path / "runs", True, True))
    outcome = execute_search(config)
    whole = next(row for row in outcome.timing_records if row["phase"] == "whole_process")
    phases = [row for row in outcome.timing_records if row["phase"] != "whole_process"]
    phase_sum = sum(float(row["exclusive_wall_seconds"]) for row in phases)
    assert phase_sum == pytest.approx(float(whole["exclusive_phase_sum_seconds"]))
    assert phase_sum <= float(whole["exclusive_wall_seconds"])
    assert float(whole["uninstrumented_seconds"]) >= 0.0
    assert {row["phase"] for row in phases} >= {
        "model_loading", "ranking_and_initialization", "search", "heldout_and_exact_evaluation"
    }
    bundle = create_search_bundle(outcome, repository_root=ROOT, run_id="timing-audit")
    assert (bundle / "measurements/candidate_screening_contexts.csv").is_file()
    assert (bundle / "measurements/timing_phases.csv").is_file()
    assert (bundle / "plots/screening_vs_exact_swap.png").is_file()
    assert (bundle / "plots/exclusive_phase_timing.png").is_file()
    report = (bundle / "report.md").read_text()
    assert "Mask-aware conditional screening" in report
    assert "Exclusive timing audit" in report
