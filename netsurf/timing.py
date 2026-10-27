"""Non-overlapping wall-clock accounting for scientific workflows."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Iterator


@dataclass(frozen=True)
class TimingRecord:
    phase: str
    category: str
    exclusive_wall_seconds: float
    operation_count: int


class ExclusiveTimingLedger:
    """Record mutually exclusive phase intervals against one monotonic clock.

    Nested phases are rejected because their totals would be ambiguous. Gaps
    are allowed and reported as uninstrumented process time.
    """

    def __init__(self, clock: Callable[[], float] = perf_counter):
        self._clock = clock
        self._started = float(clock())
        self._active: tuple[str, str, float] | None = None
        self._records: list[TimingRecord] = []

    @contextmanager
    def phase(self, name: str, category: str, *, operation_count: int = 0) -> Iterator[None]:
        if self._active is not None:
            raise RuntimeError("exclusive timing phases cannot be nested")
        started = float(self._clock()); self._active = (str(name), str(category), started)
        try:
            yield
        finally:
            active = self._active; self._active = None
            assert active is not None
            elapsed = max(0.0, float(self._clock()) - active[2])
            self._records.append(TimingRecord(active[0], active[1], elapsed, int(operation_count)))

    @property
    def records(self) -> tuple[TimingRecord, ...]:
        return tuple(self._records)

    def summary(self) -> dict[str, object]:
        if self._active is not None:
            raise RuntimeError("cannot summarize while a timing phase is active")
        whole = max(0.0, float(self._clock()) - self._started)
        exclusive = sum(value.exclusive_wall_seconds for value in self._records)
        tolerance = max(1e-12, whole * 1e-9)
        if exclusive > whole + tolerance:
            raise RuntimeError("exclusive phase time exceeds whole-process elapsed time")
        return {
            "whole_process_elapsed_seconds": whole,
            "exclusive_phase_seconds": exclusive,
            "uninstrumented_seconds": max(0.0, whole - exclusive),
            "records": self.records,
        }
