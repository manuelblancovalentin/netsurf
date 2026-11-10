"""Read-only receipt-level progress for the frozen KWS higher-BER pilot.

Usage: /mnt/raid0/ml/conda_envs/manuelbv/envs/netsurf/bin/python
       scripts/watch_kws_higher_ber_pilot.py [--once]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1] / "workspace/runs/kws_higher_ber_pilot_v1"


def snapshot(root: Path = ROOT) -> dict:
    manifest = root / "campaign.json"
    if not manifest.is_file():
        return {"exists": False, "root": str(root)}
    parent = json.loads(manifest.read_text())
    rows = []
    durations = {"selection": [], "final": []}
    for arm in parent["arms"]:
        directory = Path(arm["manifest"]).parent
        row = {"ber": arm["logical_ber"], "selection": 0, "final": 0,
               "current": "", "failed": False}
        for phase in ("selection", "final"):
            for index in range(len(parent["source_m96"]["candidate_pools"])):
                pool = f"pool_{index:03d}"
                attempt = directory / "attempts" / f"{pool}_{phase}.json"
                receipt = directory / "receipts" / f"{pool}_{phase}.json"
                if receipt.is_file():
                    row[phase] += 1
                    if attempt.is_file():
                        durations[phase].append(max(0., receipt.stat().st_mtime - attempt.stat().st_mtime))
                elif attempt.is_file():
                    row["current"] = f"{phase} {pool}"
        rows.append(row)
    stage_file = root / "runner.stage"
    stage = stage_file.read_text().strip().split("|", 1)[-1] if stage_file.is_file() else "Not started"
    pid_file = root / "runner.pid"
    active = False
    if pid_file.is_file():
        try:
            os.kill(int(pid_file.read_text().strip()), 0)
            active = True
        except (ValueError, OSError):
            pass
    per_ber = len(parent["source_m96"]["candidate_pools"])
    complete = sum(row["selection"] + row["final"] for row in rows)
    total = len(rows) * per_ber * 2
    means = {name: statistics.mean(values) if values else None for name, values in durations.items()}
    if means["selection"] is not None and means["final"] is None:
        # Historical KWS final/selection pool-time ratio, only a provisional scale.
        means["final"] = means["selection"] * 1.5
    remaining = None if means["selection"] is None else sum(
        (per_ber - row["selection"]) * means["selection"]
        + (per_ber - row["final"]) * means["final"] for row in rows)
    return {"exists": True, "root": str(root), "rows": rows, "complete": complete,
            "total": total, "stage": stage, "active": active, "remaining_seconds": remaining,
            "mean_pool_seconds": means,
            "analysis_finalized": (root.parent / "kws_higher_ber_pilot_analysis_v1/FINALIZED").is_file()}


def duration(seconds: float | None) -> str:
    if seconds is None:
        return "Estimating..."
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def render(s: dict):
    from rich.console import Group
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, TextColumn
    from rich.table import Table
    if not s["exists"]:
        return Panel(f"Frozen campaign not found: {s['root']}")
    progress = Progress(TextColumn("Overall"), BarColumn(bar_width=None), TextColumn("{task.completed}/{task.total}"),
                        TextColumn("{task.percentage:>5.1f}%"), expand=True)
    progress.add_task("pool phases", completed=s["complete"], total=s["total"])
    table = Table(show_header=True, header_style="bold", box=None, expand=True)
    for name in ("BER", "Selection", "Final", "Current", "ETA (rough)"):
        table.add_column(name, overflow="ellipsis", no_wrap=True)
    per_ber = s["total"] // (2 * len(s["rows"]))
    means = s["mean_pool_seconds"]
    for row in s["rows"]:
        eta_seconds = ((per_ber - row["selection"]) * means["selection"]
                       + (per_ber - row["final"]) * means["final"]) if means["selection"] is not None else None
        eta = duration(eta_seconds)
        table.add_row(f"{row['ber']:.3g}", f"{row['selection']}/{per_ber}", f"{row['final']}/{per_ber}",
                      row["current"] or "—", eta)
    state = "[green]Finalized[/green]" if s["analysis_finalized"] else (
        "[red]Failed/stopped[/red]" if s["stage"].startswith("FAILED") or not s["active"] else "[yellow]Running[/yellow]")
    footer = f"{state}  Stage: {s['stage']}\nOverall ETA (rough): {duration(s['remaining_seconds'])}"
    return Panel(Group(progress, table, footer), title="KWS higher-BER exploratory pilot", border_style="blue")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=float, default=2.0)
    args = parser.parse_args()
    if args.interval < 0.5:
        parser.error("interval must be at least 0.5 seconds")
    try:
        from rich.console import Console
        from rich.live import Live
    except ImportError:
        Console = None
    if Console is None or not sys.stdout.isatty():
        while True:
            state = snapshot()
            print(json.dumps({k: state[k] for k in ("stage", "complete", "total", "remaining_seconds", "active", "analysis_finalized")}
                             if state["exists"] else state), flush=True)
            if args.once or (state["exists"] and (state["analysis_finalized"] or state["stage"].startswith("FAILED"))):
                return
            time.sleep(args.interval)
    console = Console()
    if args.once:
        console.print(render(snapshot()))
        return
    with Live(render(snapshot()), console=console, refresh_per_second=1, transient=False) as live:
        while True:
            time.sleep(args.interval)
            state = snapshot()
            live.update(render(state), refresh=True)
            if state["exists"] and (state["analysis_finalized"] or state["stage"].startswith("FAILED")):
                return


if __name__ == "__main__":
    main()
