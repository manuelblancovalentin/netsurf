#!/usr/bin/env python3
"""Run P version 2, gate analysis (plan v2, Section 3).

For each checkpoint: within-cell Spearman correlation, across the seven saved rankings, between the held-out
objective (averaged over replications) and the measured mean accuracy loss, over decision-useful cells with
BER >= 1e-3; median over those cells.  Gate: median >= 0.7 on at least four of the five retrained seeds.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runP_phase0 import CHECKPOINTS, OUT_V2  # noqa: E402

SEEDS = [f"retrained_seed{s}" for s in (7101, 7102, 7103, 7104, 7105)]


def key(r, c, b):
    return r, round(float(c), 6), float(f"{float(b):.3g}")


def checkpoint(name):
    pred = defaultdict(list)
    for r in csv.DictReader(open(OUT_V2 / name / "heldout_objective.csv")):
        pred[key(r["ranking"], r["budget"], r["ber"])].append((float(r["heldout_mean_term"]), float(r["heldout_variance_term"])))
    pred = {k: np.mean(v, axis=0) for k, v in pred.items()}
    meas = {}
    for r in csv.DictReader(open(CHECKPOINTS[name]["measured"])):
        meas[key(r["method"], r["coverage"], r["ber"])] = (float(r["mean_accuracy_drop"]), r["decision_useful_primary"] in ("1", "True"))
    rankings = sorted({k[0] for k in pred})
    cells = sorted({(k[1], k[2]) for k in pred})
    out = []
    for cov, ber in cells:
        ks = [(r, cov, ber) for r in rankings]
        if ber < 1e-3 or not all(k in meas for k in ks) or not meas[ks[0]][1]:
            continue
        m = [meas[k][0] for k in ks]
        row = dict(coverage=cov, ber=ber, measured_spread=float(max(m) - min(m)))
        for label, f in (("objective", lambda v: v[0] + v[1]), ("mean_term", lambda v: v[0]), ("variance_term", lambda v: v[1])):
            rho = stats.spearmanr([f(pred[k]) for k in ks], m).statistic
            row[label] = None if rho is None or math.isnan(rho) else float(rho)
        out.append(row)
    med = {lab: (float(np.median([c[lab] for c in out if c[lab] is not None])) if out else None)
           for lab in ("objective", "mean_term", "variance_term")}
    return dict(cells=out, n_cells=len(out), median=med)


def main():
    res = {n: checkpoint(n) for n in CHECKPOINTS if (OUT_V2 / n / "heldout_objective.csv").is_file()}
    passing = [s for s in SEEDS if s in res and res[s]["median"]["objective"] is not None and res[s]["median"]["objective"] >= 0.7]
    complete = all(s in res for s in SEEDS)
    summary = dict(gate_complete=complete, seeds_passing=passing, gate_passed=complete and len(passing) >= 4,
                   checkpoints=res)
    (OUT_V2 / "v2_gate_analysis.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(dict(gate_complete=complete, gate_passed=summary["gate_passed"], seeds_passing=passing,
                          medians={n: dict(cells=r["n_cells"], **r["median"]) for n, r in res.items()}), indent=1))


if __name__ == "__main__":
    main()
