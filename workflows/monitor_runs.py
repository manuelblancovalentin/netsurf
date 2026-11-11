#!/usr/bin/env python3
"""Read-only live monitor for Paper 1 Run A (TFLite confirmation) and Run B (W8A8 cap replay).

Usage:
    python monitor_runs.py            # live view, refreshes every 30 s, Ctrl-C to leave
    python monitor_runs.py --once     # one snapshot
It only reads logs and counts result files; it never touches a run.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

W = Path("/projects/netsurf/manuelbv/workspace/implementation-worktrees")
RUN_A = W / "paper1-runA-tflite-confirm"
RUN_B = W / "paper1-tecs-revision-v1/workspace/runs/w8a8_cap_replay_v1"
A_RUNS = [("Run A  ResNet-8", "runA_resnet8", "paper1_runA_resnet8_confirm_v1"),
          ("Run A  DS-CNN", "runA_kws", "paper1_runA_kws_confirm_v1")]
B_RUNS = [("Run B  Pythia-160M", "runB_pythia", "pythia160m_int8"),
          ("Run B  Qwen3-0.6B", "runB_qwen", "qwen3_0_6b_int8")]
LINE = re.compile(r"NetSuRF \| (?P<exp>[^|]+) \| (?P<stage>[^|]+) \| (?P<done>\d+)/(?P<total>\d+)\s+[\d.]+% .*?"
                  r"elapsed (?P<elapsed>[^|]+) \| ETA (?P<eta>[^|\n]+)")


def tmux_alive(session: str) -> bool:
    return subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0


def tail(path: Path, nbytes: int = 200_000) -> str:
    if not path.is_file():
        return ""
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - nbytes))
        return f.read().decode("utf-8", "replace")


def fmt(seconds: float | None) -> str:
    if seconds is None or seconds != seconds or seconds < 0:
        return "?"
    return str(timedelta(seconds=int(seconds)))


def run_a(label: str, session: str, name: str) -> dict:
    log = RUN_A / "workspace/logs" / f"{session}.log"
    text = tail(log).replace("\r", "\n")
    last = None
    for m in LINE.finditer(text):
        last = m
    finalized = any((RUN_A / "workspace/runs" / name).glob("*/FINALIZED"))
    row = dict(label=label, alive=tmux_alive(session), finalized=finalized, done=0, total=1,
               detail="waiting for first progress line", eta="?", elapsed="?", problem="")
    if last:
        row.update(done=int(last["done"]), total=max(1, int(last["total"])), detail=last["stage"].strip(),
                   eta=last["eta"].strip(), elapsed=last["elapsed"].strip())
    errors = [ln for ln in text.splitlines()[-40:] if re.search(r"Traceback|Error|error:", ln)]
    if errors and not finalized:
        row["problem"] = errors[-1][:110]
    if finalized:
        row.update(done=row["total"], detail="FINALIZED", eta="done")
    return row


def run_b(label: str, session: str, target: str) -> dict:
    root = RUN_B / target
    row = dict(label=label, alive=tmux_alive(session), finalized=False, done=0, total=4800,
               detail="starting (mask verification and model load)", eta="?", elapsed="?", problem="")
    if not (root / "plan.json").is_file():
        row["detail"] = "plan not frozen"
        return row
    plan = json.loads((root / "plan.json").read_text())
    row["total"] = plan["counts"]["observations"]
    count = mismatches = fields = 0
    first = last = None
    fdir = root / "fields"
    if fdir.is_dir():
        for arm in os.scandir(fdir):
            for field in os.scandir(arm.path):
                for item in os.scandir(field.path):
                    if item.name == "field_done.json":
                        fields += 1
                        try:
                            mismatches += json.loads(Path(item.path).read_text())["mismatches"]
                        except Exception:
                            pass
                        continue
                    count += 1
                    t = item.stat().st_mtime
                    first = t if first is None or t < first else first
                    last = t if last is None or t > last else last
    row["done"] = count
    if count:
        row["detail"] = f"{fields}/192 fields complete, {mismatches} replay mismatches"
        row["elapsed"] = fmt(time.time() - first)
        if last > first and count > 1:
            rate = (count - 1) / (last - first)
            row["eta"] = fmt((row["total"] - count) / rate) if rate > 0 else "?"
    text = tail(root.parent / f"{target}.log", 20_000)
    errors = [ln for ln in text.splitlines()[-30:] if re.search(r"Traceback|RuntimeError|ValueError", ln)]
    if errors:
        row["problem"] = errors[-1][:110]
    if mismatches:
        row["problem"] = (row["problem"] + f"  {mismatches} bitwise mismatches").strip()
    if count >= row["total"]:
        row.update(finalized=True, eta="done", detail=row["detail"] + " (all observations saved)")
    return row


def render() -> Group:
    rows = [run_a(*r) for r in A_RUNS] + [run_b(*r) for r in B_RUNS]
    table = Table(expand=True, show_lines=False, pad_edge=False)
    table.add_column("Run", no_wrap=True)
    table.add_column("Progress", ratio=3, min_width=18)
    table.add_column("%", justify="right", no_wrap=True)
    table.add_column("Done / total", justify="right", no_wrap=True)
    table.add_column("Elapsed", justify="right", no_wrap=True)
    table.add_column("ETA", justify="right", no_wrap=True)
    table.add_column("State", no_wrap=True)
    notes = []
    for r in rows:
        frac = r["done"] / r["total"]
        if r["finalized"]:
            state = Text("complete", style="bold green")
        elif r["alive"]:
            state = Text("running", style="cyan")
        else:
            state = Text("NOT RUNNING", style="bold red")
        table.add_row(r["label"], ProgressBar(total=r["total"], completed=r["done"], width=None, complete_style="green", finished_style="bold green"),
                      f"{100 * frac:5.1f}", f'{r["done"]:,} / {r["total"]:,}', r["elapsed"], r["eta"], state)
        notes.append(f'[bold]{r["label"]}[/bold]: {r["detail"]}' + (f'  [red]{r["problem"]}[/red]' if r["problem"] else ""))
    try:
        load = os.getloadavg()
        mem = next(ln for ln in open("/proc/meminfo") if ln.startswith("MemAvailable")).split()[1]
        host = f"load {load[0]:.1f} / {load[1]:.1f} / {load[2]:.1f}   available memory {int(mem) / 1048576:.0f} GiB"
    except Exception:
        host = ""
    done_all = all(r["finalized"] for r in rows)
    footer = "[bold green]All four runs are complete. Tell Claude to analyze them.[/bold green]" if done_all else \
        "A run that shows NOT RUNNING without 'complete' stopped early: check its log."
    return Group(Panel(table, title=f"Paper 1 overnight runs   {datetime.now():%Y-%m-%d %H:%M:%S}", subtitle=host),
                 Text.from_markup("\n".join(notes)), Text.from_markup(footer))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", type=float, default=30.0)
    args = ap.parse_args()
    console = Console()
    if args.once:
        console.print(render())
        return
    with Live(render(), console=console, refresh_per_second=1, screen=False) as live:
        try:
            while True:
                time.sleep(args.interval)
                live.update(render())
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
