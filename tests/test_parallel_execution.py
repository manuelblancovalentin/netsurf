from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import time
import hashlib

import pytest

from netsurf.parallel import (
    ExecutionResources,
    LogicalTask,
    ParallelExecutionError,
    emit_worker_progress,
    run_process_tasks,
)
from netsurf.study_config import StudyReplications, load_study
from netsurf.study_engine import execute_study
from netsurf.study_bundle import create_study_bundle
from netsurf.result_bundle import read_result_bundle
from netsurf.api import run_experiment
from netsurf.search_config import SearchReplications, load_search
from netsurf.search_engine import execute_search


ROOT = Path(__file__).resolve().parents[1]


def _delayed_square(payload):
    value, delay = payload
    time.sleep(delay)
    return value * value


def _thread_environment(_payload):
    return tuple(os.environ.get(name) for name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS",
    ))


def _fail_on_two(value):
    if value == 2:
        raise ValueError("deliberate worker failure")
    return value


def _independent_pruning_initialization(seed):
    from netsurf.pruning_pipeline import _independent_values

    values = _independent_values(4, seed)
    digest = hashlib.sha256()
    for name in sorted(values):
        digest.update(name.encode("utf-8")); digest.update(values[name].tobytes(order="C"))
    return digest.hexdigest()


def _work_with_progress(value):
    emit_worker_progress(("stage", value))
    time.sleep(0.05)
    return value


@pytest.mark.parametrize("workers", (1, 2, 4, 8))
def test_worker_count_and_completion_order_do_not_change_canonical_results(workers):
    tasks = tuple(LogicalTask(f"task:{index:02d}", (index, (7 - index) * 0.001)) for index in range(8))
    result = run_process_tasks(tasks, _delayed_square, ExecutionResources(workers, 1))
    assert result.values == tuple(index * index for index in range(8))
    assert tuple(value.task_id for value in result.completions) == tuple(task.task_id for task in tasks)


def test_thread_limits_are_applied_before_worker_numerical_imports():
    result = run_process_tasks(
        (LogicalTask("environment", None),), _thread_environment,
        ExecutionResources(2, 3),
    )
    assert result.values == (("3",) * 6,)


def test_worker_progress_is_routed_through_parent_callback():
    observed = []
    tasks = tuple(LogicalTask(f"unit:{value}", value) for value in range(3))
    run_process_tasks(
        tasks, _work_with_progress, ExecutionResources(2, 1),
        on_progress=lambda task_id, event: observed.append((task_id, event)),
    )
    assert set(observed) == {(f"unit:{value}", ("stage", value)) for value in range(3)}


def test_duplicate_task_ids_and_worker_failure_are_not_accepted():
    with pytest.raises(ValueError, match="duplicate logical task"):
        run_process_tasks(
            (LogicalTask("same", 1), LogicalTask("same", 2)),
            _fail_on_two, ExecutionResources(2, 1),
        )
    with pytest.raises(ParallelExecutionError, match="task:02"):
        run_process_tasks(
            tuple(LogicalTask(f"task:{value:02d}", value) for value in range(4)),
            _fail_on_two, ExecutionResources(2, 1),
        )


def test_independent_pruning_initializations_are_worker_invariant():
    tasks = tuple(LogicalTask(f"training:{seed}", seed) for seed in (11, 13, 17))
    sequential = run_process_tasks(tasks, _independent_pruning_initialization, ExecutionResources(1, 1))
    parallel = run_process_tasks(tasks, _independent_pruning_initialization, ExecutionResources(3, 1))
    assert parallel.values == sequential.values


def test_numpy_replication_payload_is_identical_with_two_workers(tmp_path):
    configuration = load_study(ROOT / "config/experiments/numpy_qpolar_cancellation.toml")
    configuration = replace(
        configuration,
        candidates=replace(configuration.candidates, strategy="uniform", count=12, seed=51),
        replications=StudyReplications(2, 3, 5, 0, 7),
        output=replace(configuration.output, root=tmp_path / "runs", report=False, plots=False),
    )
    reference = execute_study(configuration, resources=ExecutionResources(1, 1))
    parallel = execute_study(configuration, resources=ExecutionResources(2, 1))
    assert parallel.scientific_payload_sha256 == reference.scientific_payload_sha256
    assert [value.replication_name for value in parallel.replications] == [
        value.replication_name for value in reference.replications
    ]
    assert [value.candidate_selection for value in parallel.replications] == [
        value.candidate_selection for value in reference.replications
    ]
    assert [value.protection_masks for value in parallel.replications] == [
        value.protection_masks for value in reference.replications
    ]
    destination = create_study_bundle(
        parallel, repository_root=ROOT, run_id="parallel-provenance",
        execution_provenance={"requested_workers": 2, "threads_per_worker": 1},
    )
    bundle = read_result_bundle(destination)
    assert bundle.scientific_payload_sha256 == reference.scientific_payload_sha256
    assert bundle.manifest["execution"]["parallel_execution"]["requested_workers"] == 2
    api_config = replace(configuration, output=replace(configuration.output, root=tmp_path / "api-runs"))
    api_bundle = read_result_bundle(run_experiment(api_config, resources=ExecutionResources(2, 1)))
    assert api_bundle.scientific_payload_sha256 == reference.scientific_payload_sha256
    execution = api_bundle.manifest["execution"]["parallel_execution"]
    assert execution["effective_workers"] == 2
    assert execution["logical_task_count"] == 2
    assert execution["peak_aggregate_worker_rss_mb"] > 0


def test_conditional_search_fault_fields_masks_and_payload_are_worker_invariant():
    configuration = load_search(ROOT / "config/experiments/numpy_pgcs_conditional_validation.toml")
    configuration = replace(configuration, replications=SearchReplications(2, 101, 103, 0))
    reference = execute_search(configuration, resources=ExecutionResources(1, 1))
    parallel = execute_search(configuration, resources=ExecutionResources(2, 1))
    assert parallel.scientific_payload_sha256 == reference.scientific_payload_sha256
    assert [member.scientific_payload["final_fault_states"] for member in parallel.replications] == [
        member.scientific_payload["final_fault_states"] for member in reference.replications
    ]
    assert [[result.best_mask for result in member.search_results] for member in parallel.replications] == [
        [result.best_mask for result in member.search_results] for member in reference.replications
    ]
    assert parallel.execution_metrics["logical_task_count"] == 2
    assert parallel.execution_metrics["peak_aggregate_worker_rss_mb"] > 0
