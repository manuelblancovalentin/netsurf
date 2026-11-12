#!/usr/bin/env python3
"""Exploratory (12 November 2026): where does the two-term expected-loss model predict that the positive-gradient mask
stops being better than the absolute-gradient mask?

For each ResNet-8 checkpoint, replication with a finished Run Q job, budget and calibration size (512, 1024 images),
the expected loss change of a mask under independent stored-bit flips with probability p is predicted as

    J_p(mask) = [L(theta + mu_p) - L(theta)] + 1/2 p (1 - p) sum_{unprotected b} E_x[(g_i Delta_b)^2]

(mean displacement mu_p evaluated exactly in the differentiable companion; variance term with the Fisher diagonal),
on the 512 held-out training images 28000-28511 that Run P used.  No fault outcome is read: the prediction uses only
the masks and calibration-side images.  The predicted crossover is the rate where J_p(pos) - J_p(abs) changes sign.

Output: workspace/runs/paper1_runPQ_crossover_theory_v1/<checkpoint>.json (write-once per checkpoint).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from runP_phase0 import BUDGETS, HELDOUT_COUNT, HELDOUT_OFFSET, Model, sha256_file
from runP_phase12 import OUT as RUNP, unpack
from runQ_calibration import OUT as RUNQ
from netsurf.mean_variance import bit_second_moments, displacement_moments, variance_term

OUT = RUNP.parent / "paper1_runPQ_crossover_theory_v1"
PGRID = np.unique(np.round(np.concatenate([np.logspace(-4, -1, 19), [3e-3, 3e-2]]), 12))
ARMS = ("abs512", "pos512", "abs1024", "pos1024")


def run(name, threads, out_dir, max_reps=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{name}.json"
    if target.exists():
        raise SystemExit(f"{target} exists (write-once)")
    t0 = time.time()
    model = Model(name, threads)
    xh, yh, ih = model.images("empirical_validation", HELDOUT_COUNT, HELDOUT_OFFSET)
    assert ih[0] == 28000 and ih[-1] == 28511
    G = model.per_example(xh, yh)
    m2 = bit_second_moments(model.jumps, (G * G).mean(0))
    gbar = G.mean(0)
    del G
    model.set_displacement(np.zeros(model.n))
    clean_loss, clean_acc = model.loss_acc(xh, yh)
    reps = sorted(int(p.parent.name[3:]) for p in (RUNQ / name).glob("rep*/DONE"))[:max_reps]
    rows = []
    for rep in reps:
        packed = np.load(RUNQ / name / f"rep{rep}" / "masks.npz")
        for arm in ARMS:
            for b in BUDGETS:
                mk = unpack(packed[f"{arm}@{b}"], model.n)
                for p in PGRID:
                    mu, _ = displacement_moments(model.jumps, mk, float(p))
                    model.set_displacement(mu)
                    loss, acc = model.loss_acc(xh, yh)
                    rows.append(dict(rep=rep, arm=arm, budget=float(b), p=float(p), drift=loss - clean_loss,
                                     drift_linear=float(gbar @ mu), variance=variance_term(m2, mk, float(p)),
                                     accuracy_at_mean=acc))
        print(json.dumps(dict(checkpoint=name, rep_done=rep, seconds=round(time.time() - t0))), flush=True)
    model.set_displacement(np.zeros(model.n))
    target.write_text(json.dumps(dict(
        status="exploratory; prediction from masks and held-out training images only, no fault outcomes read",
        checkpoint=name, checkpoint_sha256=model.sha256, heldout_images=[ih[0], ih[-1]], clean_loss=clean_loss,
        clean_accuracy=clean_acc, reps=reps, pgrid=[float(p) for p in PGRID], rows=rows,
        seconds=round(time.time() - t0), script_sha256=sha256_file(Path(__file__))), indent=0))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--max-reps", type=int, default=None)
    a = ap.parse_args()
    run(a.checkpoint, a.threads, a.out, a.max_reps)
