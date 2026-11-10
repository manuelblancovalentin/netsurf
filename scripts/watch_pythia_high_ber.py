#!/usr/bin/env python3
"""Progress-only, read-only monitor for the clean-gated Pythia wide-BER run.

Run from this worktree with:
    python -m scripts.watch_pythia_high_ber

Ctrl-C exits the monitor; it does not stop the experiment. No model or NetSuRF
scientific module is imported. Rich is optional and used only on a real TTY.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
import time

from scripts.watch_w8a8_wide_ber import _duration, inspect_run


ROOT = Path(__file__).resolve().parents[1] / "workspace/runs/pythia_w8a8_high_ber_clean_gated_v2"
SOCKET = "netsurf-pythia-highber"
SESSION = "pythia-highber"


@dataclass(frozen=True)
class Progress:
    stage: str
    evaluations: int
    total_evaluations: int
    fields: int
    total_fields: int
    elapsed: float | None
    eta: float | None
    preflight_workers: int

    @property
    def percent(self) -> float:
        return 100 * self.evaluations / self.total_evaluations if self.total_evaluations else 0.0


def _session_start() -> float | None:
    try:
        result = subprocess.run(
            ["tmux", "-L", SOCKET, "list-sessions", "-F", "#{session_name} #{session_created}"],
            capture_output=True, text=True, check=False, timeout=3,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode:
        return None
    for line in result.stdout.splitlines():
        try:
            name, created = line.rsplit(" ", 1)
            if name == SESSION:
                return float(created)
        except ValueError:
            continue
    return None


def inspect(root: Path = ROOT, *, now: float | None = None) -> Progress:
    """Count only saved observations and completed clean-gated fields."""
    now = time.time() if now is None else now
    started = _session_start()
    sessions = {root.name: started} if started is not None else {}
    raw = inspect_run(root.parent, root.name, sessions, now=now)
    gates = sum(1 for path in (root / "fields").glob("arm_*/field_*/clean_gate.json")
                if path.is_file())
    preflight = sum((root / "preflight" / f"shard_{shard:02d}.json").is_file()
                    for shard in range(4))
    if (root / "FINALIZED").is_file() and (root / "manifest.json").is_file():
        stage = "Completed"
    elif not (root / "plan.json").is_file():
        stage = "Starting" if started is not None else "Not started"
    elif not (root / "preflight/SEALED.json").is_file():
        stage = "Clean preflight" if started is not None else "Stopped in preflight"
    elif raw.evaluations == raw.total_evaluations and gates == raw.total_fields:
        stage = "Finalizing" if started is not None else "Awaiting finalization"
    else:
        stage = "Fault evaluation" if started is not None else "Stopped"
    # The existing operation-aware ETA is for faulted inference only. During
    # setup/finalization or after tmux exits, no credible remaining ETA exists.
    eta = raw.eta if stage == "Fault evaluation" else None
    return Progress(stage, raw.evaluations, raw.total_evaluations,
                    gates, raw.total_fields, raw.elapsed, eta, preflight)


def plain_line(item: Progress) -> str:
    if item.stage in {"Starting", "Clean preflight", "Stopped in preflight"}:
        return f"Pythia high-BER | {item.stage} | clean workers {item.preflight_workers}/4"
    if item.stage == "Completed":
        return (f"Pythia high-BER | Completed | {item.evaluations:,}/{item.total_evaluations:,} "
                f"evaluations | {item.fields}/{item.total_fields} clean-gated fields")
    eta = _duration(item.eta, estimated=item.stage == "Fault evaluation")
    return (f"Pythia high-BER | {item.stage} | {item.percent:.1f}% | "
            f"{item.evaluations:,}/{item.total_evaluations:,} evaluations | "
            f"{item.fields}/{item.total_fields} clean-gated fields | "
            f"elapsed {_duration(item.elapsed)} | inference ETA {eta}")


def rich_panel(item: Progress, width: int):
    from rich.console import Group
    from rich.progress import ProgressBar
    from rich.table import Table
    from rich.text import Text

    table = Table(title="Pythia high-BER", box=None, show_header=False,
                  expand=True, pad_edge=False)
    table.add_column("", min_width=8, no_wrap=True)
    table.add_column("", ratio=1, overflow="ellipsis")
    color = "green" if item.stage == "Completed" else ("red" if item.stage.startswith("Stopped")
            or item.stage == "Awaiting finalization" else "yellow")
    table.add_row("Stage", Text(item.stage, style=color))
    if item.stage in {"Starting", "Clean preflight", "Stopped in preflight"}:
        table.add_row("Clean", f"{item.preflight_workers}/4 workers passed")
    else:
        bar_width = max(8, min(32, width - 38))
        table.add_row("Progress", ProgressBar(total=max(1, item.total_evaluations),
                                               completed=item.evaluations, width=bar_width))
        table.add_row("Evaluations", f"{item.evaluations:,}/{item.total_evaluations:,}  ({item.percent:.1f}%)")
        table.add_row("Fields", f"{item.fields}/{item.total_fields} clean-gated")
        if item.stage != "Completed":
            table.add_row("Elapsed", _duration(item.elapsed))
            table.add_row("ETA*", _duration(item.eta, estimated=item.stage == "Fault evaluation"))
    note = Text("*Inference ETA excludes final checksum pass.", style="dim")
    return Group(table, note) if item.stage == "Fault evaluation" else table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="print one progress snapshot and exit")
    parser.add_argument("--interval", type=float, default=10.0, help="refresh seconds (default: 10)")
    parser.add_argument("--plain", action="store_true", help="line-oriented output, no Rich display")
    args = parser.parse_args(argv)
    if args.interval < 1:
        parser.error("--interval must be at least one second")

    interactive = sys.stdout.isatty() and not args.plain and not args.once
    if interactive:
        try:
            from rich.console import Console
            from rich.live import Live
        except ImportError:
            interactive = False
    try:
        if interactive:
            console = Console()
            with Live(console=console, auto_refresh=False) as live:
                while True:
                    item = inspect()
                    live.update(rich_panel(item, console.width), refresh=True)
                    if item.stage == "Completed":
                        break
                    time.sleep(args.interval)
        else:
            while True:
                item = inspect()
                print(plain_line(item), flush=True)
                if args.once or item.stage == "Completed":
                    break
                time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
