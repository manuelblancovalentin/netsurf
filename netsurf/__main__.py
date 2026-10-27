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
from .plotting import PlottingDependencyError
from .progress import ExperimentQueueItem, HierarchicalProgressReporter
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
    initialize = subparsers.add_parser("init-experiment", help="create a validated experiment TOML template")
    initialize.add_argument("--output", type=Path, required=True)
    initialize.add_argument("--template", choices=("template", "smoke"), default="template")
    validate = subparsers.add_parser("validate", help="validate without running")
    validate.add_argument("configuration", type=Path)
    validate.add_argument("--resolved", action="store_true")
    run = subparsers.add_parser("run", help="execute and finalize a new result bundle")
    run.add_argument("configuration", type=Path)
    progress = run.add_mutually_exclusive_group()
    progress.add_argument("--progress", dest="progress", action="store_true", help="show progress even when output is redirected")
    progress.add_argument("--no-progress", dest="progress", action="store_false", help="disable live progress")
    run.set_defaults(progress=None)
    batch = subparsers.add_parser("batch", help="execute multiple experiment configurations sequentially")
    batch.add_argument("configurations", type=Path, nargs="+")
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
    ConfigurationError, StudyConfigurationError, SearchConfigurationError,
    ResultBundleError, PlottingDependencyError, FileExistsError, OSError,
    KeyError, ValueError,
)


def _run_configurations(paths: list[Path], *, progress: bool | None, batch: bool) -> int:
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
        for index, (source, configuration, validation_error) in enumerate(loaded):
            if reporter is not None:
                reporter.start_experiment(index)
            if validation_error is not None:
                if reporter is not None:
                    reporter.fail_experiment(index, validation_error)
                results.append({"configuration": str(source), "status": "failed", "error": str(validation_error)})
                continue
            try:
                destination = run_experiment(configuration, progress=None if reporter is None else reporter.callback(index))
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
            return _run_configurations([args.configuration], progress=args.progress, batch=False)
        elif args.command == "batch":
            return _run_configurations(args.configurations, progress=args.progress, batch=True)
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
