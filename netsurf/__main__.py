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
from .conditional_config import ConditionalConfigurationError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="netsurf")
    subparsers = parser.add_subparsers(dest="command", required=True)
    scaling = subparsers.add_parser("scaling", help="M11 frozen outside-conditioned measurement studies")
    sc = scaling.add_subparsers(dest="scaling_command", required=True)
    sf = sc.add_parser("freeze", help="verify sources/exposure and freeze pools; no inference")
    sf.add_argument("request", type=Path)
    sf.add_argument("--output", type=Path, required=True)
    sv = sc.add_parser("validate")
    sv.add_argument("request", type=Path)
    sr = sc.add_parser("run")
    sr.add_argument("plan", type=Path)
    sr.add_argument("--pool", required=True)
    sr.add_argument("--arm", type=int, required=True)
    sr.add_argument("--phase", choices=("selection", "final"), required=True)
    sr.add_argument("--selection-registry", type=Path)
    sr.add_argument("--output", type=Path, required=True)
    ss = sc.add_parser("seal")
    ss.add_argument("plan", type=Path)
    ss.add_argument("selection_root", type=Path)
    ss.add_argument("--output", type=Path, required=True)
    sa = sc.add_parser("analyze", help="offline all-pool analysis, no inference")
    sa.add_argument("plan", type=Path)
    sa.add_argument("selection_registry", type=Path)
    sa.add_argument("final_root", type=Path)
    sa.add_argument("--output", type=Path, required=True)
    sa.add_argument("--no-plots", action="store_true")
    doctor = subparsers.add_parser("doctor", help="diagnose the software experiment environment")
    doctor.add_argument("--json", action="store_true")
    benchmarks = subparsers.add_parser("benchmarks", help="discover and prepare registered benchmarks")
    benchmark_commands = benchmarks.add_subparsers(dest="benchmark_command", required=True)
    benchmark_commands.add_parser("list", help="list registered benchmark readiness")
    capabilities = benchmark_commands.add_parser("capabilities", help="inspect the preserved 18-target capability portfolio")
    capabilities.add_argument("target", nargs="?")
    probe = benchmark_commands.add_parser("probe", help="bounded exact-bit engineering validation; no downloads or training")
    probe.add_argument("target")
    probe.add_argument("--output", type=Path, required=True)
    benchmark_info_parser = benchmark_commands.add_parser("info", help="inspect one benchmark contract")
    benchmark_info_parser.add_argument("benchmark_name")
    benchmark_prepare_parser = benchmark_commands.add_parser("prepare", help="prepare a reusable benchmark checkpoint")
    benchmark_prepare_parser.add_argument("benchmark_name")
    benchmark_prepare_parser.add_argument("--epochs", type=int, default=3)
    benchmark_prepare_parser.add_argument("--train-samples", type=int, default=20000)
    benchmark_identity_parser = benchmark_commands.add_parser(
        "identity", help="show a versioned recovered-benchmark identity manifest"
    )
    benchmark_identity_parser.add_argument("benchmark_name")
    benchmark_identity_parser.add_argument("--verify-files", action="store_true")
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
    mechanism = subparsers.add_parser(
        "mechanism-audit",
        help="derive memory-bounded exploratory mechanism diagnostics from finalized bundles",
    )
    mechanism.add_argument(
        "--benchmark", action="append", nargs=3, required=True,
        metavar=("NAME", "ESTIMATOR_BUNDLE", "PROTECTION_BUNDLE"),
        help="benchmark label and its finalized estimator/protection bundles",
    )
    mechanism.add_argument("--output", type=Path, required=True)
    mechanism.add_argument("--no-plots", action="store_true")
    mechanism.add_argument("--skip-checksums", action="store_true")
    mechanism.add_argument("--bootstrap-seed", type=int, default=930401)
    mechanism.add_argument("--bootstrap-replicates", type=int, default=1000)
    conditional = subparsers.add_parser(
        "conditional", help="validate or run an exact common-background conditional-damage study"
    )
    conditional_commands = conditional.add_subparsers(dest="conditional_command", required=True)
    conditional_validate = conditional_commands.add_parser("validate", help="validate schema-6 conditional-damage TOML")
    conditional_validate.add_argument("configuration", type=Path)
    conditional_validate.add_argument("--resolved", action="store_true")
    conditional_freeze = conditional_commands.add_parser("freeze-replication", help="write an immutable schema-6.5 replication plan without inference")
    conditional_freeze.add_argument("configuration", type=Path)
    conditional_freeze.add_argument("--output", type=Path, required=True)
    conditional_run = conditional_commands.add_parser("run", help="execute and finalize a bounded exact conditional-damage study")
    conditional_run.add_argument("configuration", type=Path)
    campaign = subparsers.add_parser("campaign", help="freeze and execute a prespecified cross-pool study")
    campaign_commands = campaign.add_subparsers(dest="campaign_command", required=True)
    freeze = campaign_commands.add_parser("freeze", help="audit assets/exposure and freeze all pools without inference")
    freeze.add_argument("configuration", type=Path)
    freeze.add_argument("--output", type=Path, required=True)
    validate_campaign = campaign_commands.add_parser("validate")
    validate_campaign.add_argument("manifest", type=Path)
    seal = campaign_commands.add_parser("seal", help="bind all finalized selection masks before any final run")
    seal.add_argument("manifest", type=Path)
    campaign_run = campaign_commands.add_parser("run")
    campaign_run.add_argument("manifest", type=Path)
    campaign_run.add_argument("--phase", choices=("selection", "final"), required=True)
    campaign_run.add_argument("--pools", nargs="+", help="operational batch of already frozen pool IDs")
    campaign_run.add_argument("--no-progress", action="store_true")
    aggregate = campaign_commands.add_parser("aggregate", help="offline analysis; requires all finalized pool results")
    aggregate.add_argument("manifest", type=Path)
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate.add_argument("--no-plots", action="store_true")
    multiber = subparsers.add_parser("multiber", help="prespecified joint KWS BER campaign")
    multiber_commands = multiber.add_subparsers(dest="multiber_command", required=True)
    mb_freeze = multiber_commands.add_parser("freeze", help="verify M9.6 pools and KWS identities; freeze all BER arms")
    mb_freeze.add_argument("--source", type=Path)
    mb_freeze.add_argument("--output", type=Path, required=True)
    mb_freeze.add_argument("--engineering", action="store_true")
    mb_freeze.add_argument("--higher-ber-pilot", action="store_true",
                           help="freeze a bounded, previously exposed KWS engineering study at 2, 5, 7.5 and 10 percent BER")
    mb_validate = multiber_commands.add_parser("validate")
    mb_validate.add_argument("manifest", type=Path)
    mb_run = multiber_commands.add_parser("run")
    mb_run.add_argument("manifest", type=Path)
    mb_run.add_argument("--phase", choices=("selection", "final"), required=True)
    mb_run.add_argument("--ber", nargs="+", type=float)
    mb_run.add_argument("--pools", nargs="+")
    mb_run.add_argument("--no-progress", action="store_true")
    mb_seal = multiber_commands.add_parser("seal")
    mb_seal.add_argument("manifest", type=Path)
    mb_aggregate = multiber_commands.add_parser("aggregate")
    mb_aggregate.add_argument("manifest", type=Path)
    mb_aggregate.add_argument("--output", type=Path, required=True)
    return parser


_EXPECTED_RUN_ERRORS = (
    ConfigurationError, StudyConfigurationError, SearchConfigurationError, PruningConfigurationError,
    ConditionalConfigurationError,
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
        if args.command == "scaling":
            from .scaling_study import freeze, load_request, execute, seal
            if args.scaling_command == "freeze":
                result = freeze(args.request, args.output)
            elif args.scaling_command == "validate":
                result = load_request(args.request)
            elif args.scaling_command == "seal":
                result = seal(args.plan, args.selection_root, args.output)
            elif args.scaling_command == "analyze":
                from .scaling_reporting import analyze
                result = analyze(args.plan,args.selection_registry,args.final_root,args.output,plots=not args.no_plots)
            else:
                result = execute(args.plan, args.pool, args.arm, args.phase, args.output,
                    selection_registry=args.selection_registry)
            print(json.dumps(result, indent=2, sort_keys=True))
        elif args.command == "doctor":
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
            if args.benchmark_command == "capabilities":
                from .benchmarks.capabilities import capability_record, capability_records
                print(json.dumps(capability_record(args.target) if args.target else capability_records(), indent=2, sort_keys=True))
            elif args.benchmark_command == "probe":
                from .benchmarks.capabilities import probe_capability
                receipt = probe_capability(args.target, args.output)
                print(json.dumps({"output": str(args.output.resolve()), "status": receipt["status"],
                    "scientific_payload_sha256": receipt["scientific_payload_sha256"], "execution":receipt["execution"]}, indent=2))
            elif args.benchmark_command == "list":
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
            elif args.benchmark_command == "identity":
                from .legacy_identity import load_benchmark_identity, verify_identity_files

                identity = load_benchmark_identity(args.benchmark_name)
                if args.verify_files:
                    if args.benchmark_name not in {"legacy_mnist_lite", "mnist_lite_exact_kb_v1"}:
                        raise ConfigurationError(
                            "--verify-files currently supports the MNIST-lite source identities"
                        )
                    identity = {
                        **identity,
                        "file_verification": verify_identity_files(identity),
                    }
                print(json.dumps(identity, indent=2, sort_keys=True))
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
        elif args.command == "mechanism-audit":
            from .mechanism_audit import MechanismBundleSpec, run_mechanism_audit

            specifications = tuple(
                MechanismBundleSpec(name, Path(estimator), Path(protection))
                for name, estimator, protection in args.benchmark
            )
            print(run_mechanism_audit(
                specifications, args.output,
                verify=not args.skip_checksums,
                plots=not args.no_plots,
                bootstrap_seed=args.bootstrap_seed,
                bootstrap_replicates=args.bootstrap_replicates,
            ))
        elif args.command == "campaign":
            from .cross_pool import freeze_campaign, validate_campaign, run_campaign, seal_selections
            if args.campaign_command == "freeze":
                result = freeze_campaign(args.configuration, args.output)
            elif args.campaign_command == "validate":
                result = validate_campaign(args.manifest)
            elif args.campaign_command == "seal":
                result = seal_selections(args.manifest)
            elif args.campaign_command == "run":
                result = run_campaign(args.manifest, args.phase, pool_ids=args.pools, progress=not args.no_progress)
            else:
                from .cross_pool_analysis import aggregate_campaign
                result = aggregate_campaign(args.manifest, args.output, plots=not args.no_plots)
            print(json.dumps(result, indent=2, sort_keys=True))
        elif args.command == "multiber":
            from .multiber_campaign import SOURCE, freeze_joint, load_joint, run_joint, seal_joint
            from .manifests import sha256_file
            if args.multiber_command == "freeze":
                result = freeze_joint(args.source or SOURCE, args.output, engineering=args.engineering,
                                      pilot=args.higher_ber_pilot)
            elif args.multiber_command == "validate":
                from .cross_pool import validate_campaign
                parent = load_joint(args.manifest)
                result = {"sha256": sha256_file(args.manifest), "arms": [validate_campaign(a["manifest"]) for a in parent["arms"]],
                          "operation_counts_total": parent["operation_counts_total"]}
            elif args.multiber_command == "seal":
                result = seal_joint(args.manifest)
            elif args.multiber_command == "run":
                result = run_joint(args.manifest, args.phase, levels=args.ber, pool_ids=args.pools,
                                   progress=not args.no_progress)
            else:
                from .multiber_analysis import aggregate_joint
                result = aggregate_joint(args.manifest, args.output)
            print(json.dumps(result, indent=2, sort_keys=True))
        elif args.command == "conditional":
            from .conditional_bundle import create_conditional_bundle
            from .conditional_config import load_conditional
            from .conditional_damage import execute_conditional_damage

            configuration = load_conditional(args.configuration, check_inputs=args.conditional_command != "freeze-replication")
            if args.conditional_command == "validate":
                print(
                    json.dumps(configuration.resolved_document(), indent=2, sort_keys=True)
                    if args.resolved else
                    f"valid: {configuration.name} (schema {configuration.schema_version})"
                )
            elif args.conditional_command == "freeze-replication":
                from .replication_preflight import freeze_replication_plan
                print(json.dumps(freeze_replication_plan(configuration, args.output), indent=2, sort_keys=True))
            elif args.conditional_command == "run":
                outcome = execute_conditional_damage(
                    configuration,
                    progress=lambda stage, completed, total: print(
                        f"conditional: {completed}/{total} | {stage}", file=sys.stderr
                    ),
                )
                destination = create_conditional_bundle(
                    outcome, repository_root=Path(__file__).resolve().parents[1]
                )
                print(json.dumps({
                    "run_directory": str(destination),
                    "scientific_payload_sha256": outcome.scientific_payload_sha256,
                }, indent=2, sort_keys=True))
        return 0
    except _EXPECTED_RUN_ERRORS as error:
        print(f"netsurf: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
