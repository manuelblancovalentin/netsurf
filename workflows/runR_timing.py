#!/usr/bin/env python3
"""Run R compute ledger: per phase, jobs finished, CPU hours, time per job and per unit (mask, evaluation), and wall
time, collected from the seconds every job records in its DONE / JSON files.  Read-only; prints a Markdown table and
optionally writes JSON.  Jobs are single-threaded unless noted, so job seconds are CPU seconds of one core.

Wall time is the span from the earliest job start (DONE mtime minus its seconds) to the latest DONE mtime.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
from pathlib import Path

RUNS = Path("/projects/netsurf/manuelbv/workspace/implementation-worktrees/paper1-tecs-revision-v1/workspace/runs")


def _done(pattern, unit_key=None, unit_name=None, units_per_job=None):
    rows = []
    for p in glob.glob(str(RUNS / pattern)):
        try:
            d = json.loads(Path(p).read_text())
        except (ValueError, OSError):
            continue
        s = d.get("seconds")
        if s is None:
            continue
        units = d.get(unit_key) if unit_key else units_per_job
        rows.append((float(s), os.path.getmtime(p), units))
    return rows, unit_name


def _per_mask(pattern, key):
    secs = []
    for p in glob.glob(str(RUNS / pattern)):
        d = json.loads(Path(p).read_text())
        recs = d.get(key, d if isinstance(d, list) else [d])
        secs += [float(r["seconds"]) for r in (recs if isinstance(recs, list) else [recs]) if "seconds" in r]
    return secs


PHASES = [
    ("Stage 1 masks (abs/pos, 8 budgets)", "paper1_runR_masks_v1/*/rep*/DONE", None, "masks", 16, "2 threads per job"),
    ("Stage 1 predictions (p*)", "paper1_runR_predictions_v1/jobs/*.json", None, None, None, ""),
    ("Stage 1 evaluation", "paper1_runR_stage1_crossover_v1/*/rep*/field*/DONE", "evaluations", "evaluations", None, ""),
    ("Stage 2 v14 mask extraction", "paper1_runR_stage2_masks_v1/*/DONE", None, None, None, ""),
    ("Stage 2 static + analytic masks", "paper1_runR_stage2_masks_v1/*/rep*/b*/DONE", None, "analytic masks", 8, ""),
    ("Stage 2 evaluation, static + analytic", "paper1_runR_stage2_eval_v1/main/*/rep*/field*/DONE", "evaluations", "evaluations", None, ""),
    ("Stage 2 MC-corrected masks", "paper1_runR_stage2_masks_v1/*/rep*/b*/mc_*/DONE", None, "MC masks", 1, ""),
    ("Stage 2 evaluation, MC", "paper1_runR_stage2_eval_v1/mc/*/rep*/field*/DONE", "evaluations", "evaluations", None, ""),
    ("DS-CNN prep: masks", "paper1_runR_kws_prep_v1/masks/rep*/masks.json", None, "masks", 16, ""),
    ("DS-CNN prep: predictions", "paper1_runR_kws_prep_v1/predict/rep*.json", None, None, None, ""),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    out = []
    print("| Phase | Jobs | CPU hours | Median per job (min) | Per unit (s) | Wall (h) | Note |")
    print("|---|---|---|---|---|---|---|")
    for label, pattern, unit_key, unit_name, per_job, note in PHASES:
        rows, _ = _done(pattern, unit_key, unit_name, per_job)
        if not rows:
            continue
        secs = [r[0] for r in rows]
        units = sum(r[2] for r in rows if r[2])
        start = min(m - s for s, m, _ in rows)
        wall = (max(m for _, m, _ in rows) - start) / 3600
        per_unit = sum(secs) / units if units else None
        rec = dict(phase=label, jobs=len(rows), cpu_hours=sum(secs) / 3600, median_job_minutes=statistics.median(secs) / 60,
                   unit=unit_name, units=units or None, seconds_per_unit=per_unit, wall_hours=wall, note=note)
        out.append(rec)
        pu = f"{per_unit:.1f} per {unit_name[:-1] if unit_name.endswith('s') else unit_name}" if per_unit else ""
        print(f"| {label} | {len(rows)} | {rec['cpu_hours']:.1f} | {rec['median_job_minutes']:.1f} | {pu} | {wall:.2f} | {note} |")
    an = _per_mask("paper1_runR_stage2_masks_v1/*/rep*/b*/build.json", "analytic")
    mc = _per_mask("paper1_runR_stage2_masks_v1/*/rep*/b*/mc_*/mc.json", "records")
    if an:
        print(f"\nAnalytic rate-aware mask, search only: median {statistics.median(an):.0f} s over {len(an)} masks")
    if a.json:
        a.json.write_text(json.dumps(dict(phases=out, analytic_mask_search_seconds=an), indent=1))


if __name__ == "__main__":
    main()
