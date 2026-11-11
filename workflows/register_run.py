#!/usr/bin/env python3
"""Register a run so that `monitor_runs.py` shows it.  Writes one JSON file into the registry; nothing else.

Examples
    # a NetSuRF engine run (progress log plus FINALIZED marker)
    register_run.py --id runX --label "Run X  ResNet-8" --kind engine --tmux runX \
        --log /path/workspace/logs/runX.log --finalized-glob "/path/workspace/runs/<experiment>/*/FINALIZED"
    # a run that writes one file per unit of work
    register_run.py --id runY --label "Run Y" --kind count --tmux runY --total 1152 --globs "/path/out/eval/**/*.json"
    # training jobs counted by a line per epoch
    register_run.py --id runZ --label "Run Z" --kind epochs --tmux runZ --total 2500 --log-glob "/path/logs/seed_*.log" \
        --unit-regex "^\\d+/\\d+ - " --done-glob "/path/out/seed_*/manifest.json" --done-count 5

`--started` is seconds since the epoch; by default it is now.  `--started-from PATH` uses that file's
modification time, for registering a run after the fact.  An existing entry is replaced only with --replace.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

REGISTRY = Path("/projects/netsurf/manuelbv/workspace/implementation-worktrees/paper1-tecs-revision-v1/workspace/run_registry")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--kind", required=True, choices=["engine", "count", "epochs"])
    ap.add_argument("--tmux"); ap.add_argument("--proc"); ap.add_argument("--note", default="")
    ap.add_argument("--started", type=float); ap.add_argument("--started-from")
    ap.add_argument("--log"); ap.add_argument("--finalized-glob")
    ap.add_argument("--globs", nargs="+"); ap.add_argument("--total", type=int)
    ap.add_argument("--done-log"); ap.add_argument("--done-text")
    ap.add_argument("--log-glob"); ap.add_argument("--unit-regex"); ap.add_argument("--done-glob"); ap.add_argument("--done-count", type=int)
    ap.add_argument("--replace", action="store_true")
    a = ap.parse_args()
    need = {"engine": ("log", "finalized_glob"), "count": ("globs", "total"), "epochs": ("log_glob", "unit_regex", "total", "done_glob", "done_count")}[a.kind]
    missing = [n for n in need if getattr(a, n) in (None, [])]
    if missing:
        ap.error(f"kind {a.kind} needs: {', '.join('--' + m.replace('_', '-') for m in missing)}")
    started = a.started if a.started is not None else os.path.getmtime(a.started_from) if a.started_from else time.time()
    entry = {k: v for k, v in vars(a).items() if v not in (None, [], "") and k not in ("replace", "started_from")}
    entry["started"] = started
    REGISTRY.mkdir(parents=True, exist_ok=True)
    path = REGISTRY / f"{a.id}.json"
    if path.exists() and not a.replace:
        raise SystemExit(f"{path} exists; use --replace")
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(entry, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)
    print("registered", path)


if __name__ == "__main__":
    main()
