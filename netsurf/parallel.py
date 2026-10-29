"""Spawn-safe deterministic execution of independent logical work units.

This module deliberately knows nothing about estimators, masks, or fault
sampling.  Scientific seeds and task identities are constructed by the
calling engine before work reaches a process pool; results are returned in
canonical task-id order regardless of completion order.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass
import multiprocessing as mp
import os
from pathlib import Path
from queue import Empty, Full
import resource
from time import monotonic
from typing import Any, Callable, Sequence


_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TF_NUM_INTRAOP_THREADS",
    "TF_NUM_INTEROP_THREADS",
)
_WORKER_EVENT_QUEUE = None
_WORKER_TASK_ID: str | None = None


class ParallelExecutionError(RuntimeError):
    """An authoritative logical work unit failed or disappeared."""

    def __init__(self, task_id: str, cause: BaseException):
        super().__init__(f"parallel task {task_id!r} failed: {cause}")
        self.task_id = task_id
        self.cause = cause


@dataclass(frozen=True)
class ExecutionResources:
    """Presentation/execution controls excluded from scientific payloads."""

    workers: int = 1
    threads_per_worker: int = 1

    def __post_init__(self) -> None:
        if self.workers < 1:
            raise ValueError("workers must be at least one")
        if self.threads_per_worker < 1:
            raise ValueError("threads_per_worker must be at least one")

    @property
    def available_cpus(self) -> int:
        return available_cpu_count()

    def validate_capacity(self) -> None:
        if self.workers * self.threads_per_worker > self.available_cpus:
            raise ValueError(
                f"requested {self.workers} workers x {self.threads_per_worker} threads "
                f"exceeds the {self.available_cpus} CPUs available to this process"
            )

    def provenance(self, *, elapsed_seconds: float | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {
            "process_start_method": "spawn" if self.workers > 1 else "none_sequential_reference",
            "requested_workers": self.workers,
            "threads_per_worker": self.threads_per_worker,
            "available_cpus": self.available_cpus,
            "oversubscription_prevented": True,
            "scientific_task_order": "canonical_logical_identity",
        }
        if elapsed_seconds is not None:
            result["coordinator_elapsed_seconds"] = float(elapsed_seconds)
        return result


@dataclass(frozen=True)
class LogicalTask:
    """One unique, schedule-independent unit of scientific work."""

    task_id: str
    payload: Any


@dataclass(frozen=True)
class TaskCompletion:
    task_id: str
    elapsed_seconds: float
    peak_rss_mb: float


@dataclass(frozen=True)
class ParallelResult:
    values: tuple[Any, ...]
    completions: tuple[TaskCompletion, ...]
    elapsed_seconds: float
    peak_aggregate_worker_rss_mb: float = 0.0
    peak_worker_rss_mb: float = 0.0


def available_cpu_count() -> int:
    """Return CPUs actually available through affinity/cgroup constraints."""

    if hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except OSError:
            pass
    return max(1, os.cpu_count() or 1)


@contextmanager
def _spawn_thread_environment(threads: int):
    """Set limits before spawn imports NumPy/TensorFlow, then restore parent."""

    previous = {name: os.environ.get(name) for name in _THREAD_ENVIRONMENT}
    for name in _THREAD_ENVIRONMENT:
        os.environ[name] = str(threads)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _initialize_worker(threads: int, event_queue=None) -> None:
    global _WORKER_EVENT_QUEUE
    _WORKER_EVENT_QUEUE = event_queue
    for name in _THREAD_ENVIRONMENT:
        os.environ[name] = str(threads)
    # These are the established deterministic CPU TensorFlow contract.  They
    # are set before a task imports an adapter; no framework import occurs here.
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")


def _execute_task(function: Callable[[Any], Any], task: LogicalTask):
    global _WORKER_TASK_ID
    _WORKER_TASK_ID = task.task_id
    started = monotonic()
    value = function(task.payload)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return task.task_id, value, monotonic() - started, rss


def emit_worker_progress(event: Any) -> None:
    """Best-effort bounded worker event; presentation can never block science."""

    if _WORKER_EVENT_QUEUE is None or _WORKER_TASK_ID is None:
        return
    try:
        _WORKER_EVENT_QUEUE.put_nowait((_WORKER_TASK_ID, event))
    except Full:
        pass


def _validate_tasks(tasks: Sequence[LogicalTask]) -> tuple[LogicalTask, ...]:
    values = tuple(tasks)
    identities = [task.task_id for task in values]
    if len(identities) != len(set(identities)):
        duplicates = sorted({value for value in identities if identities.count(value) > 1})
        raise ValueError(f"duplicate logical task identities: {duplicates}")
    return values


def _pool_rss_mb(executor: ProcessPoolExecutor) -> tuple[float, float]:
    """Best-effort Linux current RSS sample for aggregate-memory profiling."""

    values = []
    for process in (getattr(executor, "_processes", None) or {}).values():
        try:
            fields = (Path(f"/proc/{process.pid}/status").read_text(encoding="utf-8").splitlines())
            rss = next(line for line in fields if line.startswith("VmRSS:"))
            values.append(float(rss.split()[1]) / 1024.0)
        except (OSError, StopIteration, ValueError):
            continue
    return sum(values), max(values, default=0.0)


def run_process_tasks(
    tasks: Sequence[LogicalTask],
    function: Callable[[Any], Any],
    resources: ExecutionResources,
    *,
    on_started: Callable[[str], None] | None = None,
    on_completed: Callable[[TaskCompletion], None] | None = None,
    on_result: Callable[[str, Any, TaskCompletion], None] | None = None,
    on_pool_changed: Callable[[int, int], None] | None = None,
    on_failed: Callable[[str, BaseException], None] | None = None,
    on_progress: Callable[[str, Any], None] | None = None,
) -> ParallelResult:
    """Execute independent tasks and return values in declared canonical order.

    At most twice the worker count is submitted at once, bounding queued
    serialized state.  Any failed/missing/duplicate unit aborts the operation;
    callers therefore cannot finalize an incomplete scientific bundle.
    """

    ordered = _validate_tasks(tasks)
    resources.validate_capacity()
    started = monotonic()
    if not ordered:
        return ParallelResult((), (), 0.0)
    if resources.workers == 1:
        values, completions = [], []
        for task in ordered:
            if on_started is not None:
                on_started(task.task_id)
            if on_pool_changed is not None:
                on_pool_changed(1, 1)
            task_id, value, elapsed, rss = _execute_task(function, task)
            completion = TaskCompletion(task_id, elapsed, rss)
            values.append(value); completions.append(completion)
            if on_completed is not None:
                on_completed(completion)
            if on_result is not None:
                on_result(task_id, value, completion)
        if on_pool_changed is not None:
            on_pool_changed(0, 1)
        return ParallelResult(tuple(values), tuple(completions), monotonic() - started)

    context = mp.get_context("spawn")
    results: dict[str, Any] = {}
    metrics: dict[str, TaskCompletion] = {}
    pending: dict[Any, LogicalTask] = {}
    peak_aggregate_rss = 0.0
    peak_worker_rss = 0.0
    iterator = iter(ordered)
    executor: ProcessPoolExecutor | None = None
    # Keep presentation traffic bounded, but retain enough detail for short,
    # fast replications to finish before the coordinator's first drain.  The
    # former four-events-per-worker queue routinely discarded every fault-
    # evaluation update after setup events filled it.
    event_queue = (
        context.Queue(maxsize=max(256, resources.workers * 128))
        if on_progress is not None else None
    )

    def drain_events() -> None:
        if event_queue is None or on_progress is None:
            return
        while True:
            try:
                task_id, event = event_queue.get_nowait()
            except Empty:
                return
            on_progress(task_id, event)

    try:
        with _spawn_thread_environment(resources.threads_per_worker):
            executor = ProcessPoolExecutor(
                max_workers=resources.workers,
                mp_context=context,
                initializer=_initialize_worker,
                initargs=(resources.threads_per_worker, event_queue),
            )

            def submit_one(task: LogicalTask) -> None:
                if on_started is not None:
                    on_started(task.task_id)
                pending[executor.submit(_execute_task, function, task)] = task

            for _ in range(min(len(ordered), resources.workers * 2)):
                submit_one(next(iterator))
            exhausted = len(pending) == len(ordered)
            if on_pool_changed is not None:
                on_pool_changed(min(resources.workers, len(pending)), resources.workers)
            while pending:
                done, _ = wait(tuple(pending), timeout=0.1, return_when=FIRST_COMPLETED)
                aggregate_rss, worker_rss = _pool_rss_mb(executor)
                drain_events()
                peak_aggregate_rss = max(peak_aggregate_rss, aggregate_rss)
                peak_worker_rss = max(peak_worker_rss, worker_rss)
                for future in done:
                    task = pending.pop(future)
                    try:
                        task_id, value, elapsed, rss = future.result()
                    except BaseException as error:
                        if on_failed is not None:
                            on_failed(task.task_id, error)
                        for remaining in pending:
                            remaining.cancel()
                        raise ParallelExecutionError(task.task_id, error) from error
                    if task_id != task.task_id or task_id in results:
                        raise RuntimeError(f"invalid or duplicate worker result identity: {task_id!r}")
                    results[task_id] = value
                    completion = TaskCompletion(task_id, elapsed, rss)
                    metrics[task_id] = completion
                    if on_completed is not None:
                        on_completed(completion)
                    if on_result is not None:
                        on_result(task_id, value, completion)
                    if not exhausted:
                        try:
                            submit_one(next(iterator))
                        except StopIteration:
                            exhausted = True
                if on_pool_changed is not None:
                    on_pool_changed(min(resources.workers, len(pending)), resources.workers)
            executor.shutdown(wait=True)
            drain_events()
            executor = None
        missing = [task.task_id for task in ordered if task.task_id not in results]
        if missing:
            raise RuntimeError(f"missing worker results: {missing}")
        if event_queue is not None:
            event_queue.close(); event_queue.join_thread()
        return ParallelResult(
            tuple(results[task.task_id] for task in ordered),
            tuple(metrics[task.task_id] for task in ordered),
            monotonic() - started,
            peak_aggregate_rss,
            max(peak_worker_rss, max((value.peak_rss_mb for value in metrics.values()), default=0.0)),
        )
    except BaseException:
        if executor is not None:
            processes = tuple((getattr(executor, "_processes", None) or {}).values())
            executor.shutdown(wait=False, cancel_futures=True)
            # ProcessPoolExecutor has no public immediate-termination API in
            # Python 3.11.  Terminating live children prevents an interrupted
            # CLI from waiting on long numerical tasks during interpreter exit.
            for process in processes:
                if process.is_alive():
                    process.terminate()
        if event_queue is not None:
            event_queue.close(); event_queue.join_thread()
        raise


def peak_worker_rss_mb(result: ParallelResult) -> float:
    return max((value.peak_rss_mb for value in result.completions), default=0.0)
