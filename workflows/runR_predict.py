#!/usr/bin/env python3
"""Run R, amendment R-A3: predicted switch rates p* at eight budgets, from the rebuilt masks.

Same computation and definition as `runPQ_crossover_theory.py` / `runPQ_crossover_plot.py --max-p 0.03`: the
objective J_p of `pos1024` and `abs1024` on the 512 held-out training images 28000-28511 on 21 rates from 1e-4 to
1e-1; p* is the rate (up to 3e-2) above which the positive mask has the higher predicted loss at every larger grid
rate, by log-linear interpolation, on the mean over replications.  No fault outcome is read.

  job --checkpoint C --rep R      one (checkpoint, replication), single-threaded by default
  summarize                       p* per checkpoint and budget -> predictions.json (write-once)

Output: workspace/runs/paper1_runR_predictions_v1/.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from runP_phase0 import HELDOUT_COUNT, HELDOUT_OFFSET, PAPER, Model, sha256_file
from runP_phase12 import unpack
from runPQ_crossover_theory import PGRID
from runR_stage1 import FRACTIONS, MASKS, NAMES, bkey, reps_for
from netsurf.mean_variance import bit_second_moments, displacement_moments, variance_term

OUT = PAPER / "workspace/runs/paper1_runR_predictions_v1"
MAX_P = 3e-2
FROZEN = PAPER / "workspace/runs/paper1_runPQ_crossover_theory_v1_analysis/summary_maxp0.03.json"


def job(name, rep, threads):
    target = OUT / "jobs" / f"{name}_rep{rep}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise SystemExit(f"{target} exists (write-once)")
    t0 = time.time()
    model = Model(name, threads)
    xh, yh, ih = model.images("empirical_validation", HELDOUT_COUNT, HELDOUT_OFFSET)
    assert ih[0] == 28000 and ih[-1] == 28511
    G = model.per_example(xh, yh)
    m2 = bit_second_moments(model.jumps, (G * G).mean(0))
    del G
    model.set_displacement(np.zeros(model.n))
    clean = model.loss_acc(xh, yh)[0]
    packed = np.load(MASKS / name / f"rep{rep}" / "masks.npz")
    rows = []
    for a in ("abs1024", "pos1024"):
        for f in FRACTIONS:
            mk = unpack(packed[f"{a}@{bkey(f)}"], model.n)
            for p in PGRID:
                mu, _ = displacement_moments(model.jumps, mk, float(p))
                model.set_displacement(mu)
                loss, _ = model.loss_acc(xh, yh)
                rows.append(dict(arm=a, budget=f, p=float(p), J=loss - clean + variance_term(m2, mk, float(p))))
    model.set_displacement(np.zeros(model.n))
    target.write_text(json.dumps(dict(checkpoint=name, rep=rep, rows=rows, seconds=round(time.time() - t0),
                                      script_sha256=sha256_file(Path(__file__))), indent=0))
    print(json.dumps(dict(job_done=f"{name}/rep{rep}", seconds=round(time.time() - t0))), flush=True)


def crossover(pg, D):
    keep = pg <= MAX_P * (1 + 1e-9)
    pg, D = pg[keep], D[keep]
    neg = np.where(D <= 0)[0]
    if neg.size == 0:
        return float(pg[0]), "below_grid"
    i = int(neg[-1])
    if i == len(pg) - 1:
        return float(pg[-1]), "above_grid"
    x0, x1 = math.log(pg[i]), math.log(pg[i + 1])
    return math.exp(x0 + (x1 - x0) * (-D[i]) / (D[i + 1] - D[i])), "interpolated"


def summarize():
    target = OUT / "predictions.json"
    if target.exists():
        raise SystemExit("predictions.json exists (write-once)")
    pg = np.array(PGRID, dtype=float)
    frozen = json.loads(FROZEN.read_text())["crossover"]
    out, check = {}, {}
    for name in NAMES:
        J = defaultdict(list)
        for rep in reps_for(name):
            for r in json.loads((OUT / "jobs" / f"{name}_rep{rep}.json").read_text())["rows"]:
                J[(r["arm"], r["budget"], r["p"])].append(r["J"])
        for f in FRACTIONS:
            D = np.array([np.mean(J[("pos1024", f, float(p))]) - np.mean(J[("abs1024", f, float(p))]) for p in pg])
            ps, how = crossover(pg, D)
            old = frozen.get(f"{name}/1024/{f}")
            out[f"{name}/{bkey(f)}"] = dict(p_star=old["p_star"] if old else ps, how=old["how"] if old else how,
                                           source="frozen (summary_maxp0.03.json)" if old else "amendment R-A3",
                                           recomputed_p_star=ps, recomputed_how=how, curve=[float(v) for v in D])
            if old:
                check[f"{name}/{bkey(f)}"] = dict(frozen=old["p_star"], recomputed=ps)
    target.write_text(json.dumps(dict(status="predictions for Run R Stage 1; no fault outcome read",
                                      frozen_source=str(FROZEN), max_p=MAX_P, pgrid=list(map(float, pg)),
                                      predictions=out, old_budget_recompute_check=check), indent=1))
    for name in NAMES:
        print(f"{name:<20}", " ".join(f"{out[f'{name}/{bkey(f)}']['p_star']:.2e}{'*' if out[f'{name}/{bkey(f)}']['how'] != 'interpolated' else ''}"
                                     for f in FRACTIONS))
    worst = max(abs(math.log(v["recomputed"] / v["frozen"])) for v in check.values())
    print("largest |log ratio| recomputed vs frozen on the old budgets:", round(worst, 4))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage", required=True)
    j = sub.add_parser("job")
    j.add_argument("--checkpoint", required=True, choices=NAMES)
    j.add_argument("--rep", type=int, required=True)
    j.add_argument("--threads", type=int, default=1)
    sub.add_parser("summarize")
    a = ap.parse_args()
    if a.stage == "job":
        job(a.checkpoint, a.rep, a.threads)
    else:
        summarize()
