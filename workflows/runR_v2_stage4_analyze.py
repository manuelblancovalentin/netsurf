#!/usr/bin/env python3
"""Amendment R-A5, rule R4d' (DS-CNN): the selected rate-aware mask (v2) against the five static masks, on the frozen
Stage 4 fields.  Committed before any Stage 4 outcome is read.  Write-once.

v2's accuracy in a cell is the accuracy of the mask chosen for it in `picks.json` (sha256 checked): the v1 rate-aware
mask built for the tested rate, abs1024 or pos1024, on the same fields.  Usable cell: the best arm (static, v1 or v2)
within 0.05 of the mean clean accuracy.  For every static arm, difference static - v2 paired on the field, two-sided
pooled-field sign-flip test (100,000 assignments, seed 20261122), Holm over usable cells x static arms; a violation is
a mean difference of 0.5 point or more with Holm p < 0.05.  Descriptive: v2 - v1 per usable cell.
Output: workspace/runs/paper1_runR_v2_stage4_analysis_v1/summary.json, cells.csv.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from runP_phase0 import PAPER, sha256_file
from runR_kws import STAGE4_FRACTIONS
from runR_stage1 import RATES
from runR_stage4_kws import EVAL_OUT, FIELDS, REPS, STATIC

PICKS = PAPER / "workspace/runs/paper1_runR_v2_picks_v1/picks.json"
PICKS_SHA = "8c1a864f5310903d941d2d2e4eae51c5479c1aed25b0d78fa680eda347618335"
OUTD = PAPER / "workspace/runs/paper1_runR_v2_stage4_analysis_v1"
MARGIN, GAP = 0.05, 0.5
SIGNS = np.random.default_rng(20261122).choice((-1.0, 1.0), size=(100_000, len(REPS) * FIELDS))


def signflip_p(d):
    d = np.asarray(d, dtype=float)
    if np.allclose(d, 0):
        return 1.0
    null = np.abs((SIGNS[:, : d.size] * d).mean(axis=1))
    return float((1 + (null >= abs(d.mean()) - 1e-15).sum()) / (SIGNS.shape[0] + 1))


def holm(ps):
    ps = np.asarray(ps, dtype=float)
    adj, running = np.empty(ps.size), 0.0
    for rank, i in enumerate(np.argsort(ps)):
        running = max(running, (ps.size - rank) * ps[i])
        adj[i] = min(1.0, running)
    return adj


def main(eval_out=EVAL_OUT, outd=OUTD):
    if hashlib.sha256(PICKS.read_bytes()).hexdigest() != PICKS_SHA:
        raise SystemExit("picks.json does not match its frozen hash")
    picks = json.loads(PICKS.read_text())["picks"]
    acc, clean, missing = defaultdict(dict), defaultdict(list), []
    for r in REPS:
        for f in range(FIELDS):
            d = eval_out / f"rep{r}" / f"field{f}"
            if not (d / "DONE").exists():
                missing.append(f"rep{r}/field{f}"); continue
            clean[r].append(json.loads((d / "field.json").read_text())["clean_accuracy"])
            with open(d / "results.csv") as fh:
                for row in csv.DictReader(fh):
                    p = round(float(row["ber"]), 8)
                    if row["built_for"] and round(float(row["built_for"]), 8) != p:
                        continue
                    arm = "ra_analytic" if row["arm"] == "ra_analytic" else row["arm"]
                    acc[(round(float(row["budget"]), 6), p, arm)][(r, f)] = float(row["accuracy"])
    if missing:
        raise SystemExit(f"{len(missing)} Stage 4 jobs are not finished")
    outd.mkdir(exist_ok=False)
    c0 = float(np.mean([np.mean(v) for v in clean.values()]))
    mean = lambda d: float(np.mean(list(d.values())))  # noqa: E731
    cells, kept = [], Counter()
    for fr in STAGE4_FRACTIONS:
        for p in RATES:
            k = lambda a: acc[(round(fr, 6), round(p, 8), a)]  # noqa: E731
            v2 = {(r, f): k(picks[f"kws/kws/rep{r}/{fr:.6f}/{p:g}"])[(r, f)] for r in REPS for f in range(FIELDS)}
            kept.update(picks[f"kws/kws/rep{r}/{fr:.6f}/{p:g}"] for r in REPS)
            best = max([mean(v2), mean(k("ra_analytic"))] + [mean(k(a)) for a in STATIC])
            if best < c0 - MARGIN:
                continue
            keys = sorted(v2)
            for a in STATIC + ("ra_analytic",):
                d = np.array([k(a)[x] - v2[x] for x in keys]) * 100
                cells.append(dict(budget=fr, ber=p, arm=a, diff=float(d.mean()), p_raw=signflip_p(d),
                                  mean_arm=mean(k(a)), mean_v2=mean(v2)))
    st = [c for c in cells if c["arm"] != "ra_analytic"]
    for c, a in zip(st, holm([c["p_raw"] for c in st])):
        c["p_holm"] = float(a); c["violation"] = bool(c["diff"] >= GAP and a < 0.05)
    viol = [c for c in st if c["violation"]]
    v1 = [c for c in cells if c["arm"] == "ra_analytic"]
    summary = dict(status="R-A5 rule R4d' on the frozen Stage 4 fields", picks_sha256=PICKS_SHA,
                   script_sha256=sha256_file(Path(__file__)), clean_accuracy=c0, choices=dict(kept),
                   usable_cells=len({(c["budget"], c["ber"]) for c in cells}),
                   R4d_prime=dict(holds=not viol, violations=[{k: c[k] for k in ("budget", "ber", "arm", "diff", "p_holm")} for c in viol]),
                   largest_gap_static_minus_v2={a: max([c["diff"] for c in st if c["arm"] == a], default=None) for a in STATIC},
                   v1_minus_v2=dict(median=float(np.median([c["diff"] for c in v1])) if v1 else None,
                                    min=float(min(c["diff"] for c in v1)) if v1 else None,
                                    max=float(max(c["diff"] for c in v1)) if v1 else None))
    (outd / "summary.json").write_text(json.dumps(summary, indent=1))
    with open(outd / "cells.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cells[0])); w.writeheader(); w.writerows(cells)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
