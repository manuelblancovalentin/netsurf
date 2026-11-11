#!/usr/bin/env python3
"""Read-only live monitor for every registered NetSuRF run.

    python monitor_runs.py            # live view, refreshes every 30 s, Ctrl-C to leave
    python monitor_runs.py --once     # one snapshot
    python monitor_runs.py --active   # hide runs that are complete

Runs are not listed here.  Each run has one JSON file in the registry directory (REGISTRY below), written
by `register_run.py` when the run is launched, so a run started tomorrow shows up without touching this file.
The monitor only reads logs and counts result files; it never touches a run.

Registry entry fields
    id, label, started (epoch seconds), tmux (session name, optional), proc (pgrep -f pattern, optional), note
    kind = "engine":  log (NetSuRF progress log), finalized_glob (path pattern of the FINALIZED marker)
    kind = "count":   globs (list of file patterns), total, done_log + done_text (optional completion line)
    kind = "epochs":  log_glob, unit_regex (one match per finished unit), total, done_glob, done_count
    kind = "logsum":  log_glob, done_key (JSON field holding work done in the last JSON line of each log), total,
                      done_glob + done_count (completion files)
"""
from __future__ import annotations

import argparse
import glob
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

REGISTRY = Path("/projects/netsurf/manuelbv/workspace/implementation-worktrees/paper1-tecs-revision-v1/workspace/run_registry")
LINE = re.compile(r"NetSuRF[^|\n]*\| (?P<exp>[^|\n]+) \| (?P<stage>[^|\n]+) \| (?P<done>\d+)/(?P<total>\d+)\s+[\d.]+%"
                  r"[^\n]*?elapsed (?P<elapsed>[^|\n]+) \|[^\n]*?ETA (?P<eta>[^|\n]+)")


def tail(path, nbytes: int = 200_000) -> str:
    if not path or not Path(path).is_file():
        return ""
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        f.seek(max(0, f.tell() - nbytes))
        return f.read().decode("utf-8", "replace")


def fmt(seconds) -> str:
    if seconds is None or seconds != seconds or seconds < 0:
        return "?"
    return str(timedelta(seconds=int(seconds)))


def alive(entry: dict) -> bool:
    if entry.get("tmux") and subprocess.run(["tmux", "has-session", "-t", entry["tmux"]], capture_output=True).returncode == 0:
        return True
    if entry.get("proc"):
        out = subprocess.run(["pgrep", "-f", entry["proc"]], capture_output=True, text=True).stdout.split()
        return any(int(pid) != os.getpid() for pid in out)
    return False


def newest(paths):
    times = [os.path.getmtime(p) for p in paths if p and os.path.exists(p)]
    return max(times) if times else None


def status(entry: dict) -> dict:
    """done, total, complete, end time (when complete), detail, eta, problem."""
    kind = entry["kind"]; now = time.time(); start = float(entry.get("started") or now)
    row = dict(label=entry.get("label", entry["id"]), done=0, total=1, complete=False, end=None, detail=entry.get("note", ""), eta="?", problem="")
    if kind == "engine":
        text = tail(entry["log"]).replace("\r", "\n")
        last = None
        for m in LINE.finditer(text):
            last = m
        marks = glob.glob(entry["finalized_glob"])
        if last:
            row.update(done=int(last["done"]), total=max(1, int(last["total"])), detail=last["stage"].strip(), eta=last["eta"].strip())
        if marks:
            row.update(complete=True, done=row["total"], end=newest(marks), detail="finalized")
        elif re.search(r"Traceback|Error:", "\n".join(text.splitlines()[-40:])):
            row["problem"] = "error in log"
    elif kind == "count":
        files = [f for pattern in entry["globs"] for f in glob.glob(pattern, recursive=True)]
        row.update(done=len(files), total=max(1, int(entry["total"])))
        said_done = bool(entry.get("done_log")) and entry.get("done_text", "\0") in tail(entry["done_log"], 20_000)
        if len(files) >= row["total"] or said_done:
            row.update(complete=True, end=newest(files) or newest([entry.get("done_log")]))
        elif len(files) > 1:
            last = newest(files); rate = (len(files) - 1) / max(1.0, last - min(os.path.getmtime(f) for f in files))
            row["eta"] = fmt((row["total"] - len(files)) / rate)
    elif kind == "epochs":
        logs = glob.glob(entry["log_glob"])
        units = sum(len(re.findall(entry["unit_regex"], tail(p, 2_000_000), flags=re.M)) for p in logs)
        finished = glob.glob(entry["done_glob"])
        row.update(done=min(units, int(entry["total"])), total=max(1, int(entry["total"])),
                   detail=(entry.get("note", "") + f"  {len(finished)}/{entry['done_count']} finished").strip())
        if len(finished) >= int(entry["done_count"]):
            row.update(complete=True, done=row["total"], end=newest(finished))
        elif units:
            row["eta"] = fmt((now - start) / units * (row["total"] - units))
        if any("Traceback" in tail(p, 5000) for p in logs):
            row["problem"] = "error in a log"
    elif kind == "logsum":
        logs = glob.glob(entry["log_glob"]); units = 0
        for log in logs:
            for line in reversed(tail(log, 20_000).splitlines()):
                if f'"{entry["done_key"]}"' in line:
                    try:
                        units += int(json.loads(line)[entry["done_key"]])
                    except Exception:
                        pass
                    break
        finished = glob.glob(entry["done_glob"])
        if len(finished) >= int(entry["done_count"]):
            row.update(complete=True, done=int(entry["total"]), total=int(entry["total"]), end=newest(finished))
        else:
            row.update(done=min(units + 0, int(entry["total"])), total=int(entry["total"]),
                       detail=(entry.get("note", "") + f"  {len(finished)}/{entry['done_count']} finished").strip())
            if units:
                row["eta"] = fmt((now - start) / units * (int(entry["total"]) - units))
        if any("Traceback" in tail(log, 5000) for log in logs):
            row["problem"] = "error in a log"
    row["running"] = (not row["complete"]) and alive(entry)
    row["elapsed"] = fmt((row["end"] if row["complete"] and row["end"] else now) - start)   # frozen once complete
    if row["complete"]:
        row["eta"] = "done"
    row["started"] = start
    return row


def render(active_only: bool) -> Group:
    rows = []
    for path in sorted(REGISTRY.glob("*.json")):
        try:
            rows.append(status(json.loads(path.read_text())))
        except Exception as exc:                       # a half-written or malformed entry must not kill the view
            rows.append(dict(label=path.stem, done=0, total=1, complete=False, running=False, elapsed="?", eta="?",
                             detail="", problem=f"unreadable registry entry: {exc}", started=0))
    rows.sort(key=lambda r: (r["complete"], r["started"]))
    if active_only:
        rows = [r for r in rows if not r["complete"]]
    table = Table(expand=True, show_lines=False, pad_edge=False)
    table.add_column("Run", no_wrap=True, min_width=26)
    table.add_column("Progress", ratio=3, min_width=14)
    table.add_column("%", justify="right", no_wrap=True)
    table.add_column("Done / total", justify="right", no_wrap=True)
    table.add_column("Elapsed", justify="right", no_wrap=True)
    table.add_column("ETA", justify="right", no_wrap=True)
    table.add_column("State", no_wrap=True)
    notes = []
    for r in rows:
        state = Text("complete", style="bold green") if r["complete"] else Text("running", style="cyan") if r["running"] else Text("NOT RUNNING", style="bold red")
        table.add_row(r["label"], ProgressBar(total=r["total"], completed=r["done"], width=None, complete_style="green", finished_style="bold green"),
                      f'{100 * r["done"] / r["total"]:5.1f}', f'{r["done"]:,} / {r["total"]:,}', r["elapsed"], r["eta"], state)
        if not r["complete"] or r["problem"]:
            notes.append(f'[bold]{r["label"]}[/bold]: {r["detail"]}' + (f'  [red]{r["problem"]}[/red]' if r["problem"] else ""))
    try:
        load = os.getloadavg()
        mem = next(ln for ln in open("/proc/meminfo") if ln.startswith("MemAvailable")).split()[1]
        host = f"load {load[0]:.1f} / {load[1]:.1f} / {load[2]:.1f}   available memory {int(mem) / 1048576:.0f} GiB"
    except Exception:
        host = ""
    footer = "[bold green]Every registered run is complete.[/bold green]" if all(r["complete"] for r in rows) else \
        "A run that shows NOT RUNNING without 'complete' stopped early: check its log."
    return Group(Panel(table, title=f"NetSuRF runs   {datetime.now():%Y-%m-%d %H:%M:%S}", subtitle=host),
                 Text.from_markup("\n".join(notes)), Text.from_markup(footer))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--active", action="store_true", help="hide completed runs")
    ap.add_argument("--interval", type=float, default=30.0)
    args = ap.parse_args()
    console = Console()
    if args.once:
        console.print(render(args.active))
        return
    with Live(render(args.active), console=console, refresh_per_second=1, screen=False) as live:
        try:
            while True:
                time.sleep(args.interval)
                live.update(render(args.active))
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
