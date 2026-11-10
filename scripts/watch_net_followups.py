#!/usr/bin/env python3
"""Read-only, three-study follow-up monitor. No model or fault imports."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time


BASE = Path("/projects/netsurf/manuelbv/workspace/implementation-worktrees")
W8 = BASE / "w8a8-followup/workspace/runs"
KWS = BASE / "kws-transition-validation/workspace/runs"


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "Estimating..."
    minutes, remainder = divmod(int(max(0, seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{remainder:02d}" if hours else f"{minutes:02d}:{remainder:02d}"


def _stage(path: Path) -> tuple[str, float | None]:
    if not path.is_file():
        return "Pending", None
    timestamp, _, label = path.read_text().strip().partition("|")
    try:
        started = datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp()
    except ValueError:
        started = path.stat().st_mtime
    return label, max(0.0, time.time() - started)


def snapshot() -> list[dict]:
    q = W8 / "qwen_crossover_p05_p075_v1"
    p = W8 / "pythia_clean_shard_diagnostic_v1"
    k = KWS / "kws_transition_independent_pools_p05_v1"
    qplan = json.loads((q / "plan.json").read_text()) if (q / "plan.json").is_file() else None
    qpolicies = [item["name"] for item in qplan["policies"]] if qplan else []
    qtotal = (len(qplan["bers"]) * qplan["fields_per_ber"] * len(qpolicies)) if qplan else 144
    qcomplete = sum((directory / f"{policy}.json").is_file()
                    for directory in (q / "fields").glob("arm_*/field_*")
                    for policy in qpolicies)
    result = []
    for name, root, state, completed, total in (
        ("Qwen crossover", q, W8 / "qwen_followup.state", qcomplete, qtotal),
        ("Pythia clean", p, W8 / "pythia_followup.state",
         sum((p / f"worker_{i:02d}.json").is_file() for i in range(4)), 4),
        ("KWS new pools", k, KWS / "kws_transition_validation.state",
         sum((k / "receipts" / f"pool_{i:03d}_{phase}.json").is_file()
             for i in range(32) for phase in ("selection", "final")), 64),
    ):
        label, stage_elapsed = _stage(state)
        if (root / "FINALIZED").exists() or (name == "KWS new pools" and
                                            (KWS / "kws_transition_independent_pools_p05_analysis_v1/FINALIZED").exists()):
            status = "COMPLETED"
        elif label.startswith("FAILED"):
            status = "FAILED"
        elif state.exists():
            status = "RUNNING"
        else:
            status = "PENDING"
        # Work classes are not interchangeable. Only show an ETA after a
        # measured completion in the currently visible stage.
        eta = None
        if status == "RUNNING" and stage_elapsed is not None:
            if name == "Qwen crossover" and completed:
                eta = stage_elapsed * (total - completed) / completed
            elif name == "Pythia clean" and completed:
                eta = stage_elapsed * (total - completed) / completed
            elif name == "KWS new pools":
                selection = sum((k / "receipts" / f"pool_{i:03d}_selection.json").is_file()
                                for i in range(32))
                final = completed - selection
                if "FINAL:" in label and final:
                    eta = stage_elapsed * (32 - final) / final
                elif "SELECTION:" in label and selection:
                    eta = stage_elapsed * (32 - selection) / selection
        result.append({"name": name, "status": status, "stage": label,
                       "completed": completed, "total": total,
                       "stage_elapsed": _duration(stage_elapsed),
                       "stage_eta": _duration(eta) if status == "RUNNING" else "—"})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="print one plain-text record")
    args = parser.parse_args()
    if args.once or not sys.stdout.isatty():
        for row in snapshot():
            print(f"{row['name']}: {row['status']} {row['completed']}/{row['total']} | "
                  f"{row['stage']} | stage elapsed {row['stage_elapsed']} | stage ETA {row['stage_eta']}")
        return
    try:
        from rich.console import Console
        from rich.live import Live
        from rich.table import Table
    except ImportError:
        while True:
            for row in snapshot():
                print(f"{row['name']}: {row['status']} {row['completed']}/{row['total']} | "
                      f"{row['stage']} | ETA {row['stage_eta']}")
            time.sleep(2)
    console = Console()
    with Live(console=console, refresh_per_second=1, transient=False) as live:
        while True:
            table = Table(title="NetSuRF follow-ups", expand=True)
            for column in ("Study", "State", "Saved work", "Stage", "Elapsed", "ETA"):
                table.add_column(column, overflow="ellipsis", no_wrap=True)
            rows = snapshot()
            for row in rows:
                color = {"COMPLETED": "green", "RUNNING": "yellow", "FAILED": "red",
                         "PENDING": "grey62"}[row["status"]]
                table.add_row(row["name"], f"[{color}]{row['status']}[/{color}]",
                              f"{row['completed']}/{row['total']}", row["stage"],
                              row["stage_elapsed"], row["stage_eta"])
            live.update(table, refresh=True)
            if all(row["status"] in ("COMPLETED", "FAILED") for row in rows):
                break
            time.sleep(1)


if __name__ == "__main__":
    main()
