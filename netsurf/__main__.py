"""Command-line interface for versioned NetSuRF experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .api import configuration_operation_plan, load_configuration, read_result_bundle, regenerate_report, run_experiment
from .benchmarks import benchmark_info, benchmark_list
from .doctor import doctor_report
from .experiment_config import ConfigurationError, load_experiment, template_text
from .paths import workspace_root
from .parallel import ExecutionResources, LogicalTask, ParallelExecutionError, run_process_tasks
from .plotting import PlottingDependencyError
from .progress import ExperimentQueueItem, HierarchicalProgressReporter
from .pruning_config import PruningConfigurationError, load_pruning
from .result_bundle import ResultBundleError
from .study_config import StudyConfigurationError
from .search_config import SearchConfigurationError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="netsurf")
    subparsers = parser.add_subparsers(dest="command", required=True)
    doctor = subparsers.add_parser("doctor", help="diagnose the software experiment environment")
    doctor.add_argument("--json", action="store_true")
    benchmarks = subparsers.add_parser("benchmarks", help="discover and prepare registered benchmarks")
    benchmark_commands = benchmarks.add_subparsers(dest="benchmark_command", required=True)
    benchmark_commands.add_parser("list", help="list registered benchmark readiness")
    benchmark_info_parser = benchmark_commands.add_parser("info", help="inspect one benchmark contract")
    benchmark_info_parser.add_argument("benchmark_name")
    benchmark_prepare_parser = benchmark_commands.add_parser("prepare", help="prepare a reusable benchmark checkpoint")
    benchmark_prepare_parser.add_argument("benchmark_name")
    benchmark_prepare_parser.add_argument("--epochs", type=int, default=3)
    benchmark_prepare_parser.add_argument("--train-samples", type=int, default=20000)
    pruning = subparsers.add_parser("pruning", help="prepare and inspect physical structured-pruning checkpoints")
    pruning_commands = pruning.add_subparsers(dest="pruning_command", required=True)
    pruning_validate = pruning_commands.add_parser("validate", help="validate a schema-5 pruning configuration")
    pruning_validate.add_argument("configuration", type=Path)
    pruning_validate.add_argument("--resolved", action="store_true")
    pruning_run = pruning_commands.add_parser("run", help="execute a bounded schema-5 pruning preparation")
    pruning_run.add_argument("configuration", type=Path)
    pruning_batch = pruning_commands.add_parser("batch", help="prepare independent pruning configurations in parallel")
    pruning_batch.add_argument("configurations", type=Path, nargs="+")
    pruning_batch.add_argument("--workers", type=int, default=1)
    pruning_batch.add_argument("--threads-per-worker", type=int, default=1)
    pruning_inspect = pruning_commands.add_parser("inspect", help="verify an immutable pruned checkpoint")
    pruning_inspect.add_argument("checkpoint", type=Path)
    initialize = subparsers.add_parser("init-experiment", help="create a validated experiment TOML template")
    initialize.add_argument("--output", type=Path, required=True)
    initialize.add_argument("--template", choices=("template", "smoke"), default="template")
    validate = subparsers.add_parser("validate", help="validate without running")
    validate.add_argument("configuration", type=Path)
    validate.add_argument("--resolved", action="store_true")
    run = subparsers.add_parser("run", help="execute and finalize a new result bundle")
    run.add_argument("configuration", type=Path)
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--threads-per-worker", type=int, default=1)
    progress = run.add_mutually_exclusive_group()
    progress.add_argument("--progress", dest="progress", action="store_true", help="show progress even when output is redirected")
    progress.add_argument("--no-progress", dest="progress", action="store_false", help="disable live progress")
    run.set_defaults(progress=None)
    batch = subparsers.add_parser("batch", help="execute multiple independent experiment configurations")
    batch.add_argument("configurations", type=Path, nargs="+")
    batch.add_argument("--workers", type=int, default=1)
    batch.add_argument("--threads-per-worker", type=int, default=1)
    batch_progress = batch.add_mutually_exclusive_group()
    batch_progress.add_argument("--progress", dest="progress", action="store_true", help="show line-oriented progress when redirected")
    batch_progress.add_argument("--no-progress", dest="progress", action="store_false", help="disable progress display")
    batch.set_defaults(progress=None)
    inspect = subparsers.add_parser("inspect", help="verify and summarize a finalized bundle")
    inspect.add_argument("run_directory", type=Path)
    report = subparsers.add_parser("report", help="regenerate a report from saved data")
    report.add_argument("run_directory", type=Path)
    report.add_argument("--output", type=Path)
    report.add_argument("--plots", action="store_true")
    return parser


_EXPECTED_RUN_ERRORS = (
    ConfigurationError, StudyConfigurationError, SearchConfigurationError, PruningConfigurationError,
    ResultBundleError, PlottingDependencyError, FileExistsError, OSError,
    KeyError, ValueError,
    ParallelExecutionError,
)


def _batch_experiment_worker(payload):
    source, threads_per_worker, pool_workers = payload
    try:
        destination = run_experiment(
            Path(source), resources=ExecutionResources(1, threads_per_worker),
            execution_provenance_overrides={
                "requested_workers": pool_workers,
                "effective_workers": pool_workers,
                "parallel_boundary": "independent batch configuration",
                "process_start_method": "spawn",
            },
        )
        result = read_result_bundle(destination)
        return {"configuration": source, "status": "completed", "run_directory": str(destination),
                "scientific_payload_sha256": result.scientific_payload_sha256}
    except _EXPECTED_RUN_ERRORS as error:
        return {"configuration": source, "status": "failed", "error": str(error)}


def _pruning_batch_worker(payload):
    source, _threads_per_worker = payload
    from .pruning_pipeline import prepare_pruned_checkpoint
    return str(prepare_pruned_checkpoint(load_pruning(Path(source))))


def _run_configurations(
    paths: list[Path], *, progress: bool | None, batch: bool,
    workers: int = 1, threads_per_worker: int = 1,
) -> int:
    resources = ExecutionResources(workers, threads_per_worker)
    resources.validate_capacity()
    loaded = []
    items = []
    for path in paths:
        source = path.expanduser().resolve()
        try:
            configuration = load_configuration(source)
            loaded.append((source, configuration, None))
            items.append(ExperimentQueueItem(configuration.name, str(source), configuration_operation_plan(configuration)))
        except _EXPECTED_RUN_ERRORS as error:
            loaded.append((source, None, error))
            items.append(ExperimentQueueItem(source.stem, str(source), {"Configuration validation": 1}))
    enabled = sys.stderr.isatty() if progress is None else progress
    reporter = HierarchicalProgressReporter(items, sys.stderr) if enabled else None
    results = []
    unexpected = None
    try:
        if batch and workers > 1:
            for index, (_source, _configuration, validation_error) in enumerate(loaded):
                if validation_error is not None:
                    if reporter is not None:
                        reporter.fail_experiment(index, validation_error)
                    results.append({"configuration": str(loaded[index][0]), "status": "failed", "error": str(validation_error)})
            valid = [(index, source) for index, (source, configuration, error) in enumerate(loaded) if configuration is not None and error is None]
            by_id = {f"configuration:{index:06d}": index for index, _source in valid}
            completed_tasks = 0

            def started(task_id):
                if reporter is not None:
                    reporter.start_experiment(by_id[task_id])

            def pool_changed(active, total):
                if reporter is not None:
                    reporter.set_worker_pool(active, total, completed_tasks=completed_tasks)

            def received(task_id, value, _completion):
                nonlocal completed_tasks
                completed_tasks += 1
                index = by_id[task_id]
                if reporter is not None:
                    if value["status"] == "completed":
                        reporter.complete_experiment(index, value.get("run_directory"))
                    else:
                        reporter.fail_experiment(index, value["error"])
                    reporter.set_worker_pool(0, workers, completed_tasks=completed_tasks)

            def worker_failed(task_id, error):
                if reporter is not None:
                    reporter.fail_experiment(by_id[task_id], error)

            tasks = tuple(LogicalTask(task_id, (str(loaded[index][0]), threads_per_worker, workers)) for task_id, index in by_id.items())
            if tasks:
                parallel = run_process_tasks(tasks, _batch_experiment_worker, resources,
                                             on_started=started, on_result=received, on_pool_changed=pool_changed,
                                             on_failed=worker_failed)
                results.extend(parallel.values)
            results.sort(key=lambda value: next(i for i, item in enumerate(loaded) if str(item[0]) == value["configuration"]))
        else:
            for index, (source, configuration, validation_error) in enumerate(loaded):
                if reporter is not None:
                    reporter.start_experiment(index)
                if validation_error is not None:
                    if reporter is not None:
                        reporter.fail_experiment(index, validation_error)
                    results.append({"configuration": str(source), "status": "failed", "error": str(validation_error)})
                    continue
                try:
                    run_arguments = {"progress": None if reporter is None else reporter.callback(index)}
                    if workers != 1 or threads_per_worker != 1:
                        run_arguments["resources"] = resources
                    destination = run_experiment(configuration, **run_arguments)
                    result = read_result_bundle(destination)
                    if reporter is not None:
                        reporter.complete_experiment(index, str(destination))
                    results.append({
                        "configuration": str(source), "status": "completed",
                        "run_directory": str(destination),
                        "scientific_payload_sha256": result.scientific_payload_sha256,
                    })
                except _EXPECTED_RUN_ERRORS as error:
                    if reporter is not None:
                        reporter.fail_experiment(index, error)
                    results.append({"configuration": str(source), "status": "failed", "error": str(error)})
                except BaseException as error:
                    if reporter is not None:
                        reporter.fail_experiment(index, error)
                    unexpected = error
                    raise
    finally:
        if reporter is not None:
            reporter.finish()
    if unexpected is not None:
        raise unexpected
    failed = sum(value["status"] == "failed" for value in results)
    completed = len(results) - failed
    if batch:
        print(json.dumps({"completed": completed, "failed": failed, "experiments": results}, indent=2, sort_keys=True))
        if reporter is None:
            print(f"NetSuRF summary: {completed} completed, {failed} failed, 0 pending", file=sys.stderr)
    elif results and results[0]["status"] == "completed":
        print(json.dumps({key: results[0][key] for key in ("run_directory", "scientific_payload_sha256")}, indent=2, sort_keys=True))
    elif results:
        print(f"netsurf: {results[0]['error']}", file=sys.stderr)
    return 2 if failed else 0


def _run_pruning_batch(paths: list[Path], *, workers: int, threads_per_worker: int) -> int:
    resources = ExecutionResources(workers, threads_per_worker)
    resources.validate_capacity()
    configurations = [load_pruning(path.expanduser().resolve()) for path in paths]
    destinations = [configuration.output.checkpoint for configuration in configurations]
    if len(destinations) != len(set(destinations)):
        raise PruningConfigurationError("pruning batch contains duplicate checkpoint output paths")
    tasks = tuple(LogicalTask(f"pruning:{index:06d}", (str(configuration.source_path), threads_per_worker))
                  for index, configuration in enumerate(configurations))
    items = tuple(ExperimentQueueItem(
        configuration.name, str(configuration.source_path),
        {"Sequential pruning/training trajectory": max(1, len(configuration.pruning.target_widths) or configuration.pruning.iterations)},
    ) for configuration in configurations)
    reporter = HierarchicalProgressReporter(items, sys.stderr) if sys.stderr.isatty() else None
    by_id = {task.task_id: index for index, task in enumerate(tasks)}
    completed = 0

    def started(task_id):
        if reporter is not None:
            reporter.start_experiment(by_id[task_id])

    def received(task_id, value, _completion):
        nonlocal completed
        completed += 1
        if reporter is not None:
            reporter.complete_experiment(by_id[task_id], value)
            reporter.set_worker_pool(0, workers, completed_tasks=completed)

    def pool_changed(active, total):
        if reporter is not None:
            reporter.set_worker_pool(active, total, completed_tasks=completed)

    def failed(task_id, error):
        if reporter is not None:
            reporter.fail_experiment(by_id[task_id], error)

    try:
        result = run_process_tasks(
            tasks, _pruning_batch_worker, resources, on_started=started,
            on_result=received, on_pool_changed=pool_changed, on_failed=failed,
        )
    finally:
        if reporter is not None:
            reporter.finish()
    print(json.dumps({"completed": len(result.values), "checkpoints": list(result.values)}, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            report = doctor_report()
            if args.json:
                print(json.dumps(report, indent=2, sort_keys=True))
            else:
                print(f"NetSuRF {report['netsurf']} | Python {report['python']}")
                for name, value in report["packages"].items():
                    print(f"{name}: {value or 'missing'}")
                print(f"dataset root: {report['dataset_root']['path']} ({'readable' if report['dataset_root']['readable'] else 'unreadable'})")
                print(f"workspace: {report['workspace']['path']} ({'writable' if report['workspace']['writable'] else 'not writable'})")
                for name, status in report["benchmarks"].items():
                    print(f"benchmark {name}: {status}")
                for warning in report["warnings"]:
                    print(f"warning: {warning}")
        elif args.command == "benchmarks":
            if args.benchmark_command == "list":
                for item in benchmark_list():
                    status, reasons = item.readiness()
                    detail = f" ({'; '.join(reasons)})" if reasons else ""
                    print(f"{item.name}: {status}{detail}")
            elif args.benchmark_command == "info":
                print(json.dumps(benchmark_info(args.benchmark_name).to_dict(), indent=2, sort_keys=True))
            elif args.benchmark_command == "prepare":
                if args.benchmark_name != "mnist_quantized":
                    raise ConfigurationError(f"benchmark {args.benchmark_name!r} does not require preparation")
                from .adapters.mnist_quantized import prepare_checkpoint

                print(prepare_checkpoint(epochs=args.epochs, train_samples=args.train_samples))
        elif args.command == "pruning":
            if args.pruning_command == "validate":
                config = load_pruning(args.configuration)
                print(json.dumps(config.resolved_document(), indent=2, sort_keys=True) if args.resolved else f"valid: {config.name} (schema {config.schema_version})")
            elif args.pruning_command == "run":
                from .pruning_pipeline import prepare_pruned_checkpoint

                print(prepare_pruned_checkpoint(load_pruning(args.configuration)))
            elif args.pruning_command == "batch":
                return _run_pruning_batch(args.configurations, workers=args.workers, threads_per_worker=args.threads_per_worker)
            elif args.pruning_command == "inspect":
                from .pruning_pipeline import inspect_pruned_checkpoint

                print(json.dumps(inspect_pruned_checkpoint(args.checkpoint), indent=2, sort_keys=True))
        elif args.command == "init-experiment":
            destination = args.output.expanduser().resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("x", encoding="utf-8") as stream:
                stream.write(template_text(args.template))
            print(destination)
        elif args.command == "validate":
            config = load_configuration(args.configuration)
            print(json.dumps(config.resolved_document(), indent=2, sort_keys=True) if args.resolved else f"valid: {config.name} (schema {config.schema_version})")
        elif args.command == "run":
            return _run_configurations([args.configuration], progress=args.progress, batch=False,
                                       workers=args.workers, threads_per_worker=args.threads_per_worker)
        elif args.command == "batch":
            return _run_configurations(args.configurations, progress=args.progress, batch=True,
                                       workers=args.workers, threads_per_worker=args.threads_per_worker)
        elif args.command == "inspect":
            result = read_result_bundle(args.run_directory)
            print(json.dumps({"run_directory": str(result.path), "run_id": result.manifest["run_id"], "experiment": result.summary["experiment_name"], "benchmark": result.summary["benchmark_id"], "trials": result.summary["trial_count"], "scientific_payload_sha256": result.scientific_payload_sha256, "checksums": "verified"}, indent=2, sort_keys=True))
        elif args.command == "report":
            result = read_result_bundle(args.run_directory)
            output = args.output or workspace_root() / "scratch" / "reports" / result.manifest["run_id"]
            print(regenerate_report(result, output, plots=args.plots))
        return 0
    except _EXPECTED_RUN_ERRORS as error:
        print(f"netsurf: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
