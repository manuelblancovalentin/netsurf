from __future__ import annotations

import pytest

from netsurf.timing import ExclusiveTimingLedger


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
