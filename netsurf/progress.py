"""Presentation-only progress events for long-running experiments."""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import Callable, Mapping, TextIO
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
    failed: bool = False
    message: str | None = None
    work_class: str | None = None
    remaining_work: tuple[tuple[str, int], ...] = ()


ProgressCallback = Callable[[ProgressEvent], None]


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
    if configuration.schema_version in {"4.1.0", "4.2.0", "4.3.0"}:
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
