"""Command-line interface for versioned NetSuRF experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .api import read_result_bundle, regenerate_report, run_experiment
from .experiment_config import ConfigurationError, load_experiment, template_text
from .paths import workspace_root
from .plotting import PlottingDependencyError
from .result_bundle import ResultBundleError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="netsurf")
    subparsers = parser.add_subparsers(dest="command", required=True)
    initialize = subparsers.add_parser("init-experiment", help="create a validated experiment TOML template")
    initialize.add_argument("--output", type=Path, required=True)
    initialize.add_argument("--template", choices=("template", "smoke"), default="template")
    validate = subparsers.add_parser("validate", help="validate without running")
    validate.add_argument("configuration", type=Path)
    validate.add_argument("--resolved", action="store_true")
    run = subparsers.add_parser("run", help="execute and finalize a new result bundle")
    run.add_argument("configuration", type=Path)
    inspect = subparsers.add_parser("inspect", help="verify and summarize a finalized bundle")
    inspect.add_argument("run_directory", type=Path)
    report = subparsers.add_parser("report", help="regenerate a report from saved data")
    report.add_argument("run_directory", type=Path)
    report.add_argument("--output", type=Path)
    report.add_argument("--plots", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "init-experiment":
            destination = args.output.expanduser().resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("x", encoding="utf-8") as stream:
                stream.write(template_text(args.template))
            print(destination)
        elif args.command == "validate":
            config = load_experiment(args.configuration)
            print(json.dumps(config.resolved_document(), indent=2, sort_keys=True) if args.resolved else f"valid: {config.name} (schema {config.schema_version})")
        elif args.command == "run":
            destination = run_experiment(args.configuration)
            result = read_result_bundle(destination)
            print(json.dumps({"run_directory": str(destination), "scientific_payload_sha256": result.scientific_payload_sha256}, indent=2, sort_keys=True))
        elif args.command == "inspect":
            result = read_result_bundle(args.run_directory)
            print(json.dumps({"run_directory": str(result.path), "run_id": result.manifest["run_id"], "experiment": result.summary["experiment_name"], "benchmark": result.summary["benchmark_id"], "trials": result.summary["trial_count"], "scientific_payload_sha256": result.scientific_payload_sha256, "checksums": "verified"}, indent=2, sort_keys=True))
        elif args.command == "report":
            result = read_result_bundle(args.run_directory)
            output = args.output or workspace_root() / "scratch" / "reports" / result.manifest["run_id"]
            print(regenerate_report(result, output, plots=args.plots))
        return 0
    except (ConfigurationError, ResultBundleError, PlottingDependencyError, FileExistsError, OSError) as error:
        print(f"netsurf: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
