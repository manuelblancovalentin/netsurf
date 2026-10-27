"""Presentation-only progress events for long-running experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic
from typing import Callable, Mapping, Sequence, TextIO
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
    iteration: int | None = None
    iterations: int | None = None
    protected_bits: int | None = None
    estimated_risk: float | None = None
    best_estimated_risk: float | None = None
    accepted: int | None = None
    rejected: int | None = None
    exact_evaluations: int | None = None
    screened_candidates: int | None = None
    verified_swaps: int | None = None
    active_workers: int | None = None
    total_workers: int | None = None
    failed: bool = False
    message: str | None = None
    work_class: str | None = None
    remaining_work: tuple[tuple[str, int], ...] = ()


ProgressCallback = Callable[[ProgressEvent], None]


def forward_worker_event(tracker: "ProgressTracker", event: ProgressEvent) -> None:
    """Project worker-local detail onto coordinator-global work accounting."""

    names = (
        "stage_completed", "stage_total", "replication", "replications",
        "logical_ber", "coverage", "policy", "trial", "trials",
        "mean_accuracy", "mean_loss", "iteration", "iterations",
        "protected_bits", "estimated_risk", "best_estimated_risk", "accepted",
        "rejected", "exact_evaluations", "screened_candidates", "verified_swaps",
    )
    context = {name: getattr(event, name) for name in names if getattr(event, name) is not None}
    tracker.update(event.stage, **context)


class ProgressTracker:
    """Count declared operations without entering scientific serialization."""

    def __init__(self, experiment: str, total: int | Mapping[str, int], callback: ProgressCallback | None):
        self.experiment = experiment
        self.remaining = dict(total) if isinstance(total, Mapping) else {}
        self.total = max(1, sum(self.remaining.values()) if self.remaining else int(total))
        self.completed = 0
        self.callback = callback
        self._last: ProgressEvent | None = None

    def adjust_total(self, delta: int, *, stage: str = "Empirical bit injection") -> None:
        self.total = max(self.completed + 1, self.total + int(delta))
        if self.remaining:
            self.remaining[stage] = self.remaining.get(stage, 0) + int(delta)

    def update(self, stage: str, *, advance: int = 0, **context) -> None:
        work_class = context.pop("work_class", stage)
        self.completed += int(advance)
        if self.completed > self.total:
            raise RuntimeError("progress accounting exceeded its declared operation total")
        if self.remaining and advance:
            self.remaining[work_class] = max(0, self.remaining.get(work_class, 0) - int(advance))
        event = ProgressEvent(
            self.experiment, stage, self.completed, self.total,
            work_class=work_class, remaining_work=tuple(sorted(self.remaining.items())), **context
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
    """Render a responsive Rich live view or sparse redirected records."""

    def __init__(
        self,
        stream: TextIO | None = None,
        *,
        refresh_seconds: float = 1.0,
        clock: Callable[[], float] = monotonic,
        interactive: bool | None = None,
        width: int | None = None,
    ):
        self.stream = stream or sys.stderr
        self.refresh_seconds = float(refresh_seconds)
        self.clock = clock
        self.interactive = self.stream.isatty() if interactive is None else interactive
        self.width = width
        self.started = self.clock()
        self.last_rendered = -float("inf")
        self.last_stage: str | None = None
        self.last_event_time = self.started
        self.class_rates: dict[str, float] = {}
        self.smoothed_eta: float | None = None
        self._live = None
        self._console = None
        self._rich_available = False
        if self.interactive:
            try:
                from rich.console import Console
                from rich.live import Live

                self._console = Console(file=self.stream, width=width, force_terminal=True)
                self._live = Live(console=self._console, refresh_per_second=max(1, int(round(1 / max(self.refresh_seconds, 0.1)))), transient=False, auto_refresh=False)
                self._rich_available = True
            except ImportError:
                self.interactive = False
                self.stream.write(
                    "NetSuRF: Rich live progress is unavailable; using plain output. "
                    "Environment owner command: python -m pip install 'rich>=13,<16'\n"
                )
                self.stream.flush()

    @staticmethod
    def _duration(seconds: float) -> str:
        seconds = max(0, int(seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"

    def _observe(self, event: ProgressEvent, now: float) -> None:
        previous = getattr(self, "_previous", None)
        if previous is not None and event.completed > previous.completed:
            units = event.completed - previous.completed
            duration = max(0.0, now - self.last_event_time) / units
            name = event.work_class or event.stage
            prior = self.class_rates.get(name)
            self.class_rates[name] = duration if prior is None else 0.25 * duration + 0.75 * prior
        if previous is None or event.completed != previous.completed:
            self.last_event_time = now
        self._previous = event

    def _eta_seconds(self, event: ProgressEvent, now: float) -> float | None:
        if event.completed >= event.total:
            return 0.0
        remaining = dict(event.remaining_work)
        if not remaining:
            if event.completed < 2:
                return None
            elapsed = max(0.0, now - self.started)
            estimate = elapsed / event.completed * (event.total - event.completed)
        else:
            known = sum(count for name, count in remaining.items() if name in self.class_rates)
            total = sum(remaining.values())
            if known == 0 or known / max(1, total) < 0.8:
                return None
            estimate = sum(self.class_rates[name] * count for name, count in remaining.items() if name in self.class_rates)
        self.smoothed_eta = estimate if self.smoothed_eta is None else 0.30 * estimate + 0.70 * self.smoothed_eta
        return self.smoothed_eta

    def _eta(self, event: ProgressEvent, now: float) -> str:
        if event.completed >= event.total:
            return f"Completed in {self._duration(now - self.started)}"
        estimate = self._eta_seconds(event, now)
        return "Estimating..." if estimate is None else self._duration(estimate)

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
        if event.iteration is not None:
            details.append(f"iteration {event.iteration}/{event.iterations}")
        if event.screened_candidates is not None:
            details.append(f"screened {event.screened_candidates}")
        if event.verified_swaps is not None:
            details.append(f"verified {event.verified_swaps}")
        status = " | ".join(details)
        suffix = f" | {status}" if status else ""
        failure = f" | error: {event.message}" if event.failed and event.message else ""
        return (
            f"NetSuRF | {event.experiment} | {event.stage} | "
            f"{event.completed}/{event.total} {100 * fraction:5.1f}% [{bar}]"
            f"{suffix} | elapsed {self._duration(now - self.started)} | ETA {self._eta(event, now)}{failure}"
        )

    def _renderable(self, event: ProgressEvent, now: float):
        from rich.console import Group
        from rich.progress import BarColumn, Progress, TaskProgressColumn
        from rich.table import Table
        from rich.text import Text

        progress = Progress(BarColumn(bar_width=None), TaskProgressColumn(), expand=True)
        progress.add_task("Overall", total=max(1, event.total), completed=min(event.completed, event.total))
        header = Text(f"NetSuRF — {event.experiment}", style="bold", overflow="ellipsis", no_wrap=True)
        stage = Text(f"Stage: {event.stage}", overflow="ellipsis", no_wrap=True)
        timing = Table.grid(expand=True)
        timing.add_column(); timing.add_column(justify="right")
        timing.add_row(f"Elapsed: {self._duration(now - self.started)}", self._eta(event, now))
        details = Table.grid(padding=(0, 1))
        details.add_column(style="dim", no_wrap=True); details.add_column(overflow="ellipsis", no_wrap=True)
        if event.replication is not None:
            details.add_row("Replication", f"{event.replication} / {event.replications}")
        if event.logical_ber is not None:
            details.add_row("BER", f"{event.logical_ber:.3g}")
        if event.coverage is not None:
            details.add_row("Protection", f"{event.coverage:.1%}")
        if event.policy is not None:
            details.add_row("Policy", event.policy)
        if event.trial is not None:
            details.add_row("Trial", f"{event.trial} / {event.trials}")
        if event.iteration is not None:
            details.add_row("Iteration", f"{event.iteration} / {event.iterations}")
        if event.protected_bits is not None:
            details.add_row("Protected bits", str(event.protected_bits))
        if event.estimated_risk is not None:
            details.add_row("Estimated risk", f"{event.estimated_risk:.6g}")
        if event.best_estimated_risk is not None:
            details.add_row("Best estimate", f"{event.best_estimated_risk:.6g}")
        if event.accepted is not None:
            details.add_row("Accepted / rejected", f"{event.accepted} / {event.rejected}")
        if event.exact_evaluations is not None:
            details.add_row("Exact evaluations", str(event.exact_evaluations))
        if event.screened_candidates is not None:
            details.add_row("Screened candidates", str(event.screened_candidates))
        if event.verified_swaps is not None:
            details.add_row("Verified swaps", str(event.verified_swaps))
        if event.mean_accuracy is not None or event.mean_loss is not None:
            details.add_row("Mean accuracy", "—" if event.mean_accuracy is None else f"{event.mean_accuracy:.2%}")
            details.add_row("Mean loss", "—" if event.mean_loss is None else f"{event.mean_loss:.4g}")
        if event.failed and event.message:
            details.add_row("Error", event.message)
        return Group(header, stage, progress, timing, details)

    def __call__(self, event: ProgressEvent) -> None:
        now = self.clock()
        self._observe(event, now)
        stage_changed = event.stage != self.last_stage
        if stage_changed:
            self.last_stage = event.stage
        final = event.failed or event.completed >= event.total
        if not (stage_changed or final or now - self.last_rendered >= self.refresh_seconds):
            return
        if self.interactive and self._rich_available:
            if not self._live.is_started:
                self._live.start(refresh=True)
            self._live.update(self._renderable(event, now), refresh=True)
            if final:
                self._live.stop()
        else:
            self.stream.write(self._line(event, now) + "\n")
            self.stream.flush()
        self.last_rendered = now


@dataclass(frozen=True)
class ExperimentQueueItem:
    """One validated configuration in a sequential execution queue."""

    name: str
    source: str
    operation_plan: Mapping[str, int]


@dataclass
class _QueueState:
    item: ExperimentQueueItem
    status: str = "pending"
    completed: int = 0
    total: int = 1
    stage: str = "Pending"
    event: ProgressEvent | None = None
    started: float | None = None
    finished: float | None = None
    error: str | None = None
    result: str | None = None
    last_event_time: float | None = None
    previous_completed: int = 0
    remaining: dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        self.remaining = {str(key): int(value) for key, value in self.item.operation_plan.items()}
        self.total = max(1, sum(self.remaining.values()))


class HierarchicalProgressReporter:
    """One coordinated queue and current-experiment Rich Live display."""

    _STYLES = {
        "pending": ("○", "grey50"),
        "running": ("●", "yellow"),
        "completed": ("●", "green"),
        "failed": ("●", "red"),
    }

    def __init__(
        self,
        items: Sequence[ExperimentQueueItem],
        stream: TextIO | None = None,
        *,
        refresh_seconds: float = 1.0,
        clock: Callable[[], float] = monotonic,
        interactive: bool | None = None,
        width: int | None = None,
    ):
        if not items:
            raise ValueError("experiment queue cannot be empty")
        self.stream = stream or sys.stderr
        self.refresh_seconds = float(refresh_seconds)
        self.clock = clock
        self.interactive = self.stream.isatty() if interactive is None else interactive
        self.width = width
        self.started = self.clock()
        self.last_rendered = -float("inf")
        self.states = [_QueueState(item) for item in items]
        self.current_index: int | None = None
        self.class_rates: dict[str, float] = {}
        self.smoothed_queue_eta: float | None = None
        self.smoothed_task_eta: float | None = None
        self.active_workers = 0
        self.total_workers = 1
        self.completed_worker_tasks = 0
        self.worker_started_at = self.started
        self._live = None
        self._console = None
        self._rich_available = False
        self._closed = False
        if self.interactive:
            try:
                from rich.console import Console
                from rich.live import Live

                self._console = Console(file=self.stream, width=width, force_terminal=True)
                self._live = Live(
                    console=self._console,
                    refresh_per_second=max(1, int(round(1 / max(self.refresh_seconds, 0.1)))),
                    transient=False,
                    auto_refresh=False,
                )
                self._rich_available = True
            except ImportError:
                self.interactive = False
                self.stream.write(
                    "NetSuRF: Rich live progress is unavailable; using plain output. "
                    "Environment owner command: python -m pip install 'rich>=13,<16'\n"
                )
                self.stream.flush()

    @staticmethod
    def _duration(seconds: float) -> str:
        return LiveProgressReporter._duration(seconds)

    def callback(self, index: int) -> ProgressCallback:
        if not 0 <= index < len(self.states):
            raise IndexError(index)
        return lambda event: self.update(index, event)

    def set_worker_pool(self, active: int, total: int, *, completed_tasks: int | None = None) -> None:
        """Update coordinator-owned worker status without touching science."""

        self.active_workers = max(0, int(active))
        self.total_workers = max(1, int(total))
        if completed_tasks is not None:
            self.completed_worker_tasks = max(0, int(completed_tasks))
        self._render(self.clock())

    def start_experiment(self, index: int) -> None:
        state = self.states[index]
        if state.status != "pending":
            raise RuntimeError(f"experiment {index} is already {state.status}")
        now = self.clock(); self.current_index = index
        state.status = "running"; state.stage = "Starting"; state.started = now
        state.last_event_time = now; self.smoothed_task_eta = None
        self._render(now, force=True, plain_message=f"Starting {state.item.name} ({index + 1}/{len(self.states)})")

    def _observe(self, state: _QueueState, event: ProgressEvent, now: float) -> None:
        if event.completed > state.previous_completed and state.last_event_time is not None:
            units = event.completed - state.previous_completed
            seconds_per_unit = max(0.0, now - state.last_event_time) / units
            name = event.work_class or event.stage
            prior = self.class_rates.get(name)
            self.class_rates[name] = seconds_per_unit if prior is None else 0.25 * seconds_per_unit + 0.75 * prior
            state.last_event_time = now
        elif event.completed != state.previous_completed:
            state.last_event_time = now
        state.previous_completed = event.completed

    def update(self, index: int, event: ProgressEvent) -> None:
        state = self.states[index]
        now = self.clock()
        if state.status == "pending":
            self.start_experiment(index)
            now = self.clock()
        if state.status not in {"running", "failed"}:
            return
        self._observe(state, event, now)
        state.event = event; state.completed = int(event.completed); state.total = max(1, int(event.total))
        if event.active_workers is not None:
            self.active_workers = max(0, int(event.active_workers))
        if event.total_workers is not None:
            self.total_workers = max(1, int(event.total_workers))
        state.stage = event.stage; state.remaining = dict(event.remaining_work)
        if event.failed:
            state.status = "failed"; state.error = event.message or "experiment failed"; state.finished = now
        self._render(now, force=event.failed or event.stage != getattr(state, "_rendered_stage", None))
        state._rendered_stage = event.stage

    def complete_experiment(self, index: int, result: str | None = None) -> None:
        state = self.states[index]
        if state.status == "failed":
            return
        if state.status != "running":
            raise RuntimeError(f"experiment {index} is not running")
        now = self.clock(); state.status = "completed"; state.stage = "Completed"
        state.completed = state.total; state.finished = now; state.result = result
        state.remaining = {key: 0 for key in state.remaining}
        self._render(now, force=True, plain_message=f"Completed {state.item.name}: {state.completed}/{state.total}")

    def fail_experiment(self, index: int, error: BaseException | str) -> None:
        state = self.states[index]; now = self.clock()
        if state.status == "failed":
            return
        if state.status == "pending":
            state.started = now
        state.status = "failed"; state.stage = "Failed"; state.finished = now
        state.error = str(error); self.current_index = index
        self._render(now, force=True, plain_message=f"Failed {state.item.name}: {state.error}")

    def _task_eta_seconds(self, state: _QueueState, now: float) -> float | None:
        if state.status == "completed":
            return 0.0
        remaining = state.remaining
        if not remaining:
            if state.completed < 2 or state.started is None:
                return None
            return (now - state.started) / state.completed * max(0, state.total - state.completed)
        total = sum(max(0, value) for value in remaining.values())
        known = sum(max(0, value) for name, value in remaining.items() if name in self.class_rates)
        if total == 0:
            return 0.0
        if known / total < 0.5:
            return None
        estimate = sum(self.class_rates[name] * max(0, value) for name, value in remaining.items() if name in self.class_rates)
        self.smoothed_task_eta = estimate if self.smoothed_task_eta is None else 0.30 * estimate + 0.70 * self.smoothed_task_eta
        return self.smoothed_task_eta

    def _queue_eta_seconds(self, now: float) -> float | None:
        remaining: dict[str, int] = {}
        for state in self.states:
            if state.status in {"completed", "failed"}:
                continue
            values = state.remaining or dict(state.item.operation_plan)
            for name, count in values.items():
                remaining[name] = remaining.get(name, 0) + max(0, int(count))
        total = sum(remaining.values())
        if total == 0:
            return 0.0
        known = sum(count for name, count in remaining.items() if name in self.class_rates)
        if known / total < 0.5:
            return None
        estimate = sum(self.class_rates[name] * count for name, count in remaining.items() if name in self.class_rates)
        self.smoothed_queue_eta = estimate if self.smoothed_queue_eta is None else 0.30 * estimate + 0.70 * self.smoothed_queue_eta
        return self.smoothed_queue_eta

    def _condition(self, event: ProgressEvent | None) -> str:
        if event is None:
            return ""
        values = []
        if event.replication is not None:
            values.append(f"rep {event.replication}/{event.replications}")
        if event.logical_ber is not None:
            values.append(f"BER {event.logical_ber:.3g}")
        if event.coverage is not None:
            values.append(f"coverage {event.coverage:.1%}")
        if event.policy:
            values.append(event.policy)
        if event.trial is not None:
            values.append(f"trial {event.trial}/{event.trials}")
        if event.iteration is not None:
            values.append(f"iteration {event.iteration}/{event.iterations}")
        return " · ".join(values)

    def _renderable(self, now: float):
        from rich.console import Group
        from rich.progress import BarColumn, Progress, TaskProgressColumn
        from rich.table import Table
        from rich.text import Text

        queue = Table(title="NetSuRF experiment queue", expand=True, box=None, padding=(0, 1))
        queue.add_column("Status", width=9, no_wrap=True); queue.add_column("Experiment", ratio=2, overflow="ellipsis", no_wrap=True)
        queue.add_column("Work", justify="right", no_wrap=True)
        queue.add_column("Runtime", justify="right", no_wrap=True)
        queue.add_column("Stage", ratio=2, overflow="ellipsis", no_wrap=True)
        for state in self.states:
            symbol, style = self._STYLES[state.status]
            stage = state.error if state.status == "failed" else state.stage
            label = "done" if state.status == "completed" else state.status
            if state.started is None:
                runtime = "—"
            elif state.finished is not None:
                runtime = self._duration(state.finished - state.started)
            else:
                runtime = self._duration(now - state.started)
            queue.add_row(Text(f"{symbol} {label}", style=style), Text(state.item.name, style=style),
                          f"{state.completed}/{state.total}", runtime,
                          Text(stage, style=style, overflow="ellipsis", no_wrap=True))
        total = sum(state.total for state in self.states)
        completed = sum(state.completed for state in self.states)
        overall = Progress(BarColumn(bar_width=None), TaskProgressColumn(), expand=True)
        overall.add_task("Queue", total=max(1, total), completed=min(completed, total))
        pieces = [queue, overall]
        state = None if self.current_index is None else self.states[self.current_index]
        if state is not None and state.status in {"running", "failed"}:
            event = state.event
            detail = Table.grid(expand=True, padding=(0, 1)); detail.add_column(style="dim", no_wrap=True); detail.add_column(overflow="ellipsis", no_wrap=True)
            detail.add_row("Current", state.item.name)
            detail.add_row("Stage", state.stage)
            condition = self._condition(event)
            if condition: detail.add_row("Condition", condition)
            detail.add_row("Work", f"{state.completed} / {state.total}")
            detail.add_row("Workers", f"{self.active_workers} active / {self.total_workers} total")
            worker_elapsed = max(0.0, now - self.worker_started_at)
            if self.completed_worker_tasks and worker_elapsed:
                detail.add_row("Throughput", f"{self.completed_worker_tasks / worker_elapsed:.3g} tasks/s")
            elapsed = 0.0 if state.started is None else now - state.started
            task_eta = self._task_eta_seconds(state, now)
            queue_eta = self._queue_eta_seconds(now)
            detail.add_row("Elapsed", self._duration(elapsed))
            detail.add_row("Experiment ETA", "Estimating..." if task_eta is None else self._duration(task_eta))
            detail.add_row("Queue ETA", "Estimating..." if queue_eta is None else self._duration(queue_eta))
            if event is not None and event.mean_accuracy is not None:
                detail.add_row("Mean accuracy", f"{event.mean_accuracy:.2%}")
            if event is not None and event.mean_loss is not None:
                detail.add_row("Mean loss", f"{event.mean_loss:.4g}")
            if state.error: detail.add_row("Error", state.error)
            pieces.append(detail)
        return Group(*pieces)

    def _plain_line(self, now: float) -> str:
        state = None if self.current_index is None else self.states[self.current_index]
        completed_tasks = sum(item.status == "completed" for item in self.states)
        failed_tasks = sum(item.status == "failed" for item in self.states)
        if state is None:
            return f"NetSuRF queue: {completed_tasks} completed, {failed_tasks} failed, {len(self.states)} total"
        fraction = min(1.0, state.completed / max(1, state.total))
        eta = self._queue_eta_seconds(now)
        condition = self._condition(state.event)
        suffix = f" | {condition}" if condition else ""
        return (f"NetSuRF queue {self.current_index + 1}/{len(self.states)} | {state.item.name} | {state.stage} | "
                f"{state.completed}/{state.total} {fraction:.1%}{suffix} | elapsed {self._duration(now-self.started)} | "
                f"workers {self.active_workers}/{self.total_workers} | "
                f"queue ETA {'Estimating...' if eta is None else self._duration(eta)}")

    def _render(self, now: float, *, force: bool = False, plain_message: str | None = None) -> None:
        if self._closed or (not force and now - self.last_rendered < self.refresh_seconds):
            return
        if self.interactive and self._rich_available:
            if not self._live.is_started:
                self._live.start(refresh=True)
            self._live.update(self._renderable(now), refresh=True)
        else:
            self.stream.write((plain_message or self._plain_line(now)) + "\n"); self.stream.flush()
        self.last_rendered = now

    def finish(self) -> None:
        if self._closed:
            return
        now = self.clock()
        self._render(now, force=True)
        if self._live is not None and self._live.is_started:
            self._live.stop()
        completed = sum(state.status == "completed" for state in self.states)
        failed = sum(state.status == "failed" for state in self.states)
        pending = sum(state.status == "pending" for state in self.states)
        summary = f"NetSuRF summary: {completed} completed, {failed} failed, {pending} pending in {self._duration(now-self.started)}"
        if self._console is not None and self.interactive and self._rich_available:
            self._console.print(summary)
        else:
            self.stream.write(summary + "\n"); self.stream.flush()
        self._closed = True


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


def study_operation_plan(configuration, *, include_bundle: bool = False) -> dict[str, int]:
    """Return actual planned operation counts by measurable work class."""
    replications = configuration.replications.count
    plan = {"Model loading": replications, "Sample loading": replications, "Estimator computation": replications,
            "Clean evaluation": replications, "Scientific payload": replications}
    if configuration.study_type in {"single_bit", "estimator_validation"}:
        plan.update({"Candidate selection": replications, "Empirical bit injection": replications * (configuration.candidates.count + 1), "Ranking analysis": replications})
    else:
        probabilities = len(configuration.protection.probabilities) or 1
        masks = 2 + len(configuration.policies) * len(configuration.protection.coverages)
        plan.update({"Protection-mask construction": replications,
                     "Fault evaluation": replications * probabilities * configuration.execution.trials * (1 + masks),
                     "Statistical aggregation": replications})
    if replications > 1:
        plan["Replication aggregation"] = 1
    if include_bundle:
        plan["Result bundle generation"] = 1
    return plan


def experiment_operation_count(configuration, *, include_bundle: bool = False) -> int:
    masks = len(configuration.estimators) + (2 if configuration.schema_version == "2.0.0" else 0)
    total = 5  # model, data, clean baseline, estimator/masks, payload aggregation
    total += configuration.execution.trials * (1 + masks)
    return total + int(include_bundle)


def experiment_operation_plan(configuration, *, include_bundle: bool = False) -> dict[str, int]:
    masks = len(configuration.estimators) + (2 if configuration.schema_version == "2.0.0" else 0)
    plan = {"Model loading": 1, "Sample loading": 1, "Clean evaluation": 1,
            "Estimator and mask construction": 1,
            "Fault evaluation": configuration.execution.trials * (1 + masks),
            "Scientific aggregation": 1}
    if include_bundle:
        plan["Result bundle generation"] = 1
    return plan


def search_operation_plan(configuration, *, include_bundle: bool = False) -> dict[str, int]:
    """Count the complete bounded schema-4 workflow by measurable work class."""

    conditions = len(getattr(configuration, "fault_probabilities", (configuration.fault.probability,)))
    budget_count = len(getattr(configuration.initial_mask, "budgets_bits", ())) or 1
    replications = configuration.replications.count * conditions * budget_count
    methods = getattr(configuration, "methods", ())
    method_names = tuple(value.name for value in methods) or configuration.search.strategies
    method_algorithms = tuple(value.strategy for value in methods) or configuration.search.strategies
    plan = {
        "Model loading": replications,
        "Sample partitioning": replications,
        "Ranking and initialization": replications,
        "Incremental exactness validation": replications,
        "Held-out final evaluation": replications * len(method_names) * configuration.final_evaluation.contexts,
        "Scientific payload": replications,
    }
    if configuration.schema_version in {"4.1.0", "4.2.0", "4.3.0", "4.4.0"}:
        plan["Held-out conditional evaluation"] = replications * len(method_names) * configuration.final_evaluation.conditional_contexts
    polar_contexts = 1 if configuration.polar.mode == "finite_clean" else configuration.polar.background_contexts
    if configuration.polar.enabled:
        plan["Sparse polar calibration"] = replications * configuration.polar.pair_budget * polar_contexts
    for method, name, strategy in zip(methods, method_names, method_algorithms):
        if strategy != "static":
            iterations = configuration.search.iterations if getattr(method, "iterations", None) is None else method.iterations
            plan[f"Search {name}"] = replications * iterations
    # Upper bound; the engine reduces this after it knows how many proposals
    # were accepted, because only accepted mask states are plotted.
    states = sum(1 if strategy == "static" else 1 + (configuration.search.iterations if getattr(method, "iterations", None) is None else method.iterations) for method, strategy in zip(methods, method_algorithms))
    plan["Held-out convergence"] = replications * states * configuration.final_evaluation.contexts
    if configuration.search.exact_enumeration:
        plan["Restricted exact enumeration"] = replications
    if include_bundle:
        plan["Result bundle generation"] = 1
    return plan
