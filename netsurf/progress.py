"""Dependency-free presentation events for long-running experiments."""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import Callable, TextIO
import sys


@dataclass(frozen=True)
class ProgressEvent:
    experiment: str
    stage: str
    completed: int
    total: int
    stage_completed: int | None = None
    stage_total: int | None = None
    replication: int | None = None
    replications: int | None = None
    logical_ber: float | None = None
    coverage: float | None = None
    policy: str | None = None
    trial: int | None = None
    trials: int | None = None
    mean_accuracy: float | None = None
    mean_loss: float | None = None
    failed: bool = False
    message: str | None = None


ProgressCallback = Callable[[ProgressEvent], None]


class ProgressTracker:
    """Count declared operations without entering scientific serialization."""

    def __init__(self, experiment: str, total: int, callback: ProgressCallback | None):
        self.experiment = experiment
        self.total = max(1, int(total))
        self.completed = 0
        self.callback = callback
        self._last: ProgressEvent | None = None

    def adjust_total(self, delta: int) -> None:
        self.total = max(self.completed + 1, self.total + int(delta))

    def update(self, stage: str, *, advance: int = 0, **context) -> None:
        self.completed += int(advance)
        if self.completed > self.total:
            raise RuntimeError("progress accounting exceeded its declared operation total")
        event = ProgressEvent(
            self.experiment, stage, self.completed, self.total, **context
        )
        self._last = event
        if self.callback is not None:
            try:
                self.callback(event)
            except Exception:
                # Presentation must never interrupt or alter scientific work.
                self.callback = None

    def finish(self) -> None:
        if self.completed != self.total:
            raise RuntimeError(
                f"progress accounting ended at {self.completed}/{self.total} operations"
            )

    def fail(self, error: BaseException) -> None:
        if self.callback is not None:
            previous = self._last
            try:
                self.callback(ProgressEvent(
                    self.experiment,
                    "Failed" if previous is None else previous.stage,
                    self.completed,
                    self.total,
                    replication=None if previous is None else previous.replication,
                    replications=None if previous is None else previous.replications,
                    failed=True,
                    message=str(error),
                ))
            except Exception:
                pass


class LiveProgressReporter:
    """Render compact interactive output with a plain redirected fallback."""

    def __init__(
        self,
        stream: TextIO | None = None,
        *,
        refresh_seconds: float = 1.0,
        clock: Callable[[], float] = monotonic,
        interactive: bool | None = None,
    ):
        self.stream = stream or sys.stderr
        self.refresh_seconds = float(refresh_seconds)
        self.clock = clock
        self.interactive = self.stream.isatty() if interactive is None else interactive
        self.started = self.clock()
        self.last_rendered = -float("inf")
        self.last_stage: str | None = None
        self.stage_started = self.started

    @staticmethod
    def _duration(seconds: float) -> str:
        seconds = max(0, int(seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"

    def _eta(self, event: ProgressEvent, now: float) -> str:
        if (
            event.stage_completed is None or event.stage_total is None
            or event.stage_completed < 2 or event.stage_completed >= event.stage_total
        ):
            return "calculating..."
        rate = (now - self.stage_started) / event.stage_completed
        return f"~{self._duration(rate * (event.stage_total - event.stage_completed))} (stage)"

    def _line(self, event: ProgressEvent, now: float) -> str:
        fraction = 1.0 if event.total == 0 else min(1.0, event.completed / event.total)
        width = 20
        bar = "#" * int(fraction * width) + "-" * (width - int(fraction * width))
        details = []
        if event.replication is not None:
            details.append(f"rep {event.replication}/{event.replications}")
        if event.logical_ber is not None:
            details.append(f"BER {event.logical_ber:.8g}")
        if event.coverage is not None:
            details.append(f"coverage {100 * event.coverage:.3g}%")
        if event.policy is not None:
            details.append(f"policy {event.policy}")
        if event.trial is not None:
            details.append(f"trial {event.trial}/{event.trials}")
        if event.mean_accuracy is not None:
            details.append(f"mean acc {100 * event.mean_accuracy:.2f}%")
        if event.mean_loss is not None:
            details.append(f"mean loss {event.mean_loss:.4g}")
        status = " | ".join(details)
        suffix = f" | {status}" if status else ""
        failure = f" | error: {event.message}" if event.failed and event.message else ""
        return (
            f"NetSuRF | {event.experiment} | {event.stage} | "
            f"{event.completed}/{event.total} {100 * fraction:5.1f}% [{bar}]"
            f"{suffix} | elapsed {self._duration(now - self.started)} | ETA {self._eta(event, now)}{failure}"
        )

    def __call__(self, event: ProgressEvent) -> None:
        now = self.clock()
        stage_changed = event.stage != self.last_stage
        if stage_changed:
            self.stage_started = now
            self.last_stage = event.stage
        final = event.failed or event.completed >= event.total
        if not (stage_changed or final or now - self.last_rendered >= self.refresh_seconds):
            return
        line = self._line(event, now)
        if self.interactive:
            self.stream.write("\r\x1b[2K" + line)
            if final:
                self.stream.write("\n")
        else:
            self.stream.write(line + "\n")
        self.stream.flush()
        self.last_rendered = now


def study_operation_count(configuration, *, include_bundle: bool = False) -> int:
    """Return the exact number of coarse operations emitted by study_engine."""

    per_replication = 3  # model load, sample loading, estimator batch
    if configuration.study_type in {"single_bit", "estimator_validation"}:
        per_replication += 3 + configuration.candidates.count  # selection, baseline+candidates, analysis
    else:
        masks = 2 + len(configuration.policies) * len(configuration.protection.coverages)
        probabilities = len(configuration.protection.probabilities) or 1
        per_replication += 2  # mask construction and aggregation
        per_replication += probabilities * configuration.execution.trials * (1 + masks)
    per_replication += 2  # clean evaluation and payload finalization
    total = configuration.replications.count * per_replication
    if configuration.replications.count > 1:
        total += 1
    return total + int(include_bundle)


def experiment_operation_count(configuration, *, include_bundle: bool = False) -> int:
    masks = len(configuration.estimators) + (2 if configuration.schema_version == "2.0.0" else 0)
    total = 5  # model, data, clean baseline, estimator/masks, payload aggregation
    total += configuration.execution.trials * (1 + masks)
    return total + int(include_bundle)
