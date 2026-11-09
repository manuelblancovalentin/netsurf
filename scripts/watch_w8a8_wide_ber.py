#!/usr/bin/env python3
"""Read-only live monitor for the three frozen W8A8 wide-BER campaigns.

Requires only Python's standard library. Rich is used when available in a TTY.
This program never imports NetSuRF, loads a model, or writes a campaign file.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time


DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "workspace/runs/w8a8_wide_ber_matched_v1"
TARGETS = ("distilbert_sst2_int8", "pythia160m_int8", "qwen3_0_6b_int8")
TMUX_SOCKET = "netsurf-w8a8-wide-ber"


@dataclass(frozen=True)
class Snapshot:
    target: str
    state: str
    fields: int
    total_fields: int
    evaluations: int
    total_evaluations: int
    observations: int
    total_observations: int
    elapsed: float | None
    eta: float | None
    detail: str = ""


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _tmux_sessions() -> dict[str, float]:
    try:
        result = subprocess.run(
            ["tmux", "-L", TMUX_SOCKET, "list-sessions", "-F", "#{session_name} #{session_created}"],
            capture_output=True, text=True, check=False, timeout=3,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {}
    if result.returncode:
        return {}
    sessions = {}
    for line in result.stdout.splitlines():
        try:
            name, created = line.rsplit(" ", 1)
            sessions[name] = float(created)
        except ValueError:
            continue
    return sessions


def _median_duration(values: list[float]) -> float | None:
    values = [v for v in values if math.isfinite(v) and v > 0]
    return statistics.median(values) if values else None


def inspect_run(root: Path, target: str, sessions: dict[str, float], now: float | None = None) -> Snapshot:
    """Count only persisted records; never infer completion from queue messages."""
    now = time.time() if now is None else now
    run = root / target
    plan_path = run / "plan.json"
    if not plan_path.is_file():
        return Snapshot(target, "Pending", 0, 0, 0, 0, 0, 0, None, None, "no frozen plan")
    try:
        plan = _read_json(plan_path)
        policy_names = [p["name"] for p in plan["policies"]]
        real_names = [p["name"] for p in plan["policies"] if p["kind"] != "full"]
        total_fields = int(plan["counts"]["fields"])
        total_evaluations = int(plan["counts"]["planned_faulted_model_evaluations"])
        total_observations = total_fields * len(policy_names)
        shards = int(plan["execution_shards"])
        fields_per_ber = int(plan["fields_per_ber"])
        fields = evaluations = observations = 0
        shard_evaluations = [0] * shards
        durations: list[float] = []
        by_ber: dict[int, list[float]] = {i: [] for i in range(len(plan["bers"]))}
        for arm in range(len(plan["bers"])):
            for index in range(fields_per_ber):
                directory = run / "fields" / f"arm_{arm:02d}" / f"field_{index:03d}"
                if not (directory / "field.json").is_file():
                    continue
                saved = 0
                for name in policy_names:
                    if (directory / f"{name}.json").is_file():
                        saved += 1
                observations += saved
                fields += saved == len(policy_names)
                for name in real_names:
                    record_path = directory / f"{name}.json"
                    if not record_path.is_file():
                        continue
                    evaluations += 1
                    shard_evaluations[index % shards] += 1
                    try:
                        duration = float(_read_json(record_path)["elapsed_seconds"])
                    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                        continue  # An in-progress/invalid record cannot inform ETA.
                    if math.isfinite(duration) and duration > 0:
                        durations.append(duration)
                        by_ber[arm].append(duration)
        marker = run / "FINALIZED"
        if marker.is_file() and (run / "manifest.json").is_file():
            state = "Finalized"
        elif evaluations == total_evaluations and fields == total_fields:
            state = "Finalizing" if target in sessions else "Awaiting finalization"
        elif target in sessions:
            state = "Running"
        else:
            state = "Stopped" if observations else "Pending"
        elapsed = max(0.0, now - sessions[target]) if target in sessions else None
        # Estimate the remaining evaluation work along the slowest of the four
        # deterministic shards. Prefer BER-specific timings once >= 5 exist.
        # Field generation, model startup and final integrity checks are excluded.
        eta = None
        if state == "Running" and len(durations) >= 10:
            fallback = _median_duration(durations)
            if fallback is not None:
                remaining = [0.0] * shards
                for arm in range(len(plan["bers"])):
                    typical = (_median_duration(by_ber[arm]) if len(by_ber[arm]) >= 5
                               else fallback) or fallback
                    for index in range(fields_per_ber):
                        directory = run / "fields" / f"arm_{arm:02d}" / f"field_{index:03d}"
                        missing = sum(not (directory / f"{name}.json").is_file()
                                      for name in real_names)
                        remaining[index % shards] += missing * typical
                eta = max(remaining)
        if state == "Finalized":
            eta = 0.0
        return Snapshot(target, state, fields, total_fields, evaluations,
                        total_evaluations, observations, total_observations,
                        elapsed, eta, "ETA excludes field generation and final checksum pass")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return Snapshot(target, "Error", 0, 0, 0, 0, 0, 0, None, None, str(exc))


def _duration(seconds: float | None, *, estimated: bool = False) -> str:
    if seconds is None:
        return "Estimating…" if estimated else "—"
    value = max(0, round(seconds))
    hours, remainder = divmod(value, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


def _plain_line(item: Snapshot) -> str:
    percent = 100 * item.evaluations / item.total_evaluations if item.total_evaluations else 0
    eta = _duration(item.eta, estimated=item.state == "Running")
    return (f"{item.target}: {item.state} | {item.evaluations}/{item.total_evaluations} "
            f"evals ({percent:.1f}%) | {item.fields}/{item.total_fields} paired fields | "
            f"{item.observations}/{item.total_observations} records | "
            f"elapsed {_duration(item.elapsed)} | eval ETA* {eta}")


def _rich_table(items: list[Snapshot], width: int = 100):
    from rich.progress import ProgressBar
    from rich.table import Table
    from rich.text import Text
    table = Table(title="NetSuRF · W8A8 wide-BER campaigns", expand=True, box=None,
                  show_header=True, pad_edge=False)
    table.add_column("Model", no_wrap=True, min_width=9)
    table.add_column("State", no_wrap=True, min_width=10)
    compact = width < 95
    table.add_column("Progress", min_width=10, ratio=1)
    table.add_column("Inference", justify="right", no_wrap=True)
    if not compact:
        table.add_column("Fields", justify="right", no_wrap=True)
    table.add_column("Elapsed", justify="right", no_wrap=True)
    table.add_column("ETA*", justify="right", no_wrap=True)
    colors = {"Running": "yellow", "Finalizing": "yellow", "Finalized": "green",
              "Stopped": "red", "Error": "red", "Pending": "dim",
              "Awaiting finalization": "red"}
    for item in items:
        label = item.target.replace("_int8", "").replace("distilbert_sst2", "DistilBERT")
        label = label.replace("pythia160m", "Pythia").replace("qwen3_0_6b", "Qwen")
        color = colors.get(item.state, "white")
        bar = ProgressBar(total=max(1, item.total_evaluations),
                          completed=item.evaluations, width=10 if compact else None)
        values = [label, Text(item.state, style=color), bar,
                  f"{item.evaluations}/{item.total_evaluations}"]
        if not compact:
            values.append(f"{item.fields}/{item.total_fields}")
        values.extend([_duration(item.elapsed),
                       _duration(item.eta, estimated=item.state == "Running")])
        table.add_row(*values)
    return table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--interval", type=float, default=3.0, help="refresh seconds (default: 3)")
    parser.add_argument("--once", action="store_true", help="print one snapshot and exit")
    parser.add_argument("--plain", action="store_true", help="force line-oriented output")
    args = parser.parse_args(argv)
    if args.interval < 0.5:
        parser.error("--interval must be at least 0.5 seconds")

    def snapshot() -> list[Snapshot]:
        sessions = _tmux_sessions()
        return [inspect_run(args.runs_root, target, sessions) for target in TARGETS]

    use_rich = not args.plain and not args.once and sys.stdout.isatty()
    if use_rich:
        try:
            from rich.console import Console, Group
            from rich.live import Live
            from rich.text import Text
        except ImportError:
            use_rich = False
    if use_rich:
        console = Console()
        try:
            with Live(console=console, auto_refresh=False, transient=False) as live:
                while True:
                    rows = snapshot()
                    note = Text("*ETA extrapolates saved inference timings; field generation and final verification add time.",
                                style="dim")
                    live.update(Group(_rich_table(rows, console.width), note), refresh=True)
                    time.sleep(args.interval)
        except KeyboardInterrupt:
            return 0
    else:
        try:
            while True:
                for item in snapshot():
                    print(_plain_line(item), flush=True)
                if args.once:
                    break
                print("*ETA excludes field generation and final verification.", flush=True)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
