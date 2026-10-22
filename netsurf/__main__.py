"""Command-line entry point for bounded NetSuRF v2 workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .milestone import execute_repeated
from .paths import artifact_root, dataset_root


def main() -> int:
    parser = argparse.ArgumentParser(prog="netsurf")
    subparsers = parser.add_subparsers(dest="command", required=True)
    milestone = subparsers.add_parser("milestone", help="run the deterministic MNIST-lite milestone twice")
    milestone.add_argument("--repo", type=Path, default=Path.cwd())
    milestone.add_argument("--output-root", type=Path)
    milestone.add_argument("--checkpoint", type=Path)
    milestone.add_argument("--dataset", type=Path)
    milestone.add_argument("--samples", type=int, default=256)
    milestone.add_argument("--ranking-samples", type=int, default=256)
    milestone.add_argument("--trials", type=int, default=12)
    milestone.add_argument("--ber", type=float, default=0.001)
    milestone.add_argument("--coverage", type=float, default=0.10)
    milestone.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if args.command == "milestone":
        result = execute_repeated(
            args.repo,
            args.output_root or artifact_root() / "runs" / "mnist_lite_milestone",
            checkpoint=args.checkpoint,
            dataset=args.dataset or dataset_root() / "mnist" / "mnist.npz",
            repeats=args.repeats,
            sample_count=args.samples,
            ranking_count=args.ranking_samples,
            trials=args.trials,
            ber=args.ber,
            coverage=args.coverage,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
