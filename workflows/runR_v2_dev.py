#!/usr/bin/env python3
"""Development of the validated rate-aware mask (v2) on training-side data only; no test input, no fault outcome.

v2 = the v1 search with two changes: (i) a swap is accepted only if its per-input gain is resolved,
mean(d) + z sd(d)/sqrt(N) < 0 with z = 2 over the N calibration inputs; (ii) a safeguard picks, among the v2 mask,
abs1024 and pos1024, the one with the lowest predicted objective on a separate selection set.

Data roles per (model, replication): search = the 1,024 calibration inputs of v1; selection = 512 further training
inputs (ResNet-8: held-out training images from 30000 + 512 r; DS-CNN: validation clips from 1024 + 512 r); scoring = the
held-out set of the earlier check (ResNet-8: training images 28000-28511; DS-CNN: validation clips 0-511).

  job --model resnet8 --checkpoint C --budget B      (first Stage 2 replication of C)
  job --model kws --rep R --budget B
Output (write-once): workspace/runs/paper1_runR_v2_dev_v1/<model>_<ckpt or rep>_b<B>.json
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from runP_phase0 import HELDOUT_COUNT, HELDOUT_OFFSET, PAPER, ROOT, Model, sha256_file
from runP_phase12 import unpack
from runR_stage1 import bkey, reps_for
from netsurf.mean_variance import bit_second_moments, displacement_moments, mask_sha256, variance_term
from netsurf.rate_aware import (IDEAL_LADDER, analytic_evaluator, descend, ladder_rates, per_input_evaluator,
                                rate_aware_assignment, select_on_heldout, value_ranking_start)

OUT = PAPER / "workspace/runs/paper1_runR_v2_dev_v1"
RATES = (1e-3, 3e-3, 1e-2, 3e-2)
Z, MAX_ROUNDS, MIN_SWAP = 2.0, 40, 16
RUNS = PAPER / "workspace/runs"


def losses_term(m, x, y):
    tf = m.tf
    m.set_displacement(np.zeros(m.n))
    clean = m.comp.losses(x, y).numpy().astype(np.float64)

    def f(mu):
        m.set_displacement(mu)
        with tf.GradientTape() as tape:
            lv = m.comp.losses(x, y)
            loss = tf.reduce_mean(lv)
        grads = tape.gradient(loss, [m.comp.variables[n] for n in m.names])
        return (lv.numpy().astype(np.float64) - clean,
                np.concatenate([g.numpy().reshape(-1) for g in grads]).astype(np.float64))
    return f


def objective_fn(m, x, y):
    """Predicted objective of an ideal-suppression mask on (x, y), as in the held-out check."""
    G = m.per_example(x, y)
    m2 = bit_second_moments(m.jumps, (G * G).mean(0))
    del G
    m.set_displacement(np.zeros(m.n))
    clean = m.loss_acc(x, y)[0]

    def J(mask, p):
        mu, _ = displacement_moments(m.jumps, mask, float(p))
        m.set_displacement(mu)
        val = m.loss_acc(x, y)[0] - clean + variance_term(m2, mask, float(p))
        m.set_displacement(np.zeros(m.n))
        return val
    return J


def job(model, key, frac, threads):
    out = OUT / f"{model}_{key}_b{bkey(frac)}.json"
    OUT.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise SystemExit(f"{out} exists (write-once)")
    t0 = time.time()
    b = bkey(frac)
    if model == "resnet8":
        m = Model(key, threads)
        rep = reps_for(key)[0]
        xc, yc, _ = m.images("ranking", 1024, 1000 * rep)
        xs, ys, _ = m.images("empirical_validation", 512, 30000 + 512 * rep)
        xd, yd, idd = m.images("empirical_validation", HELDOUT_COUNT, HELDOUT_OFFSET)
        assert idd[0] == 28000
        base = RUNS / "paper1_runR_stage2_masks_v1" / key / f"rep{rep}" / f"b{b}"
        st = np.load(base / "static.npz")
        static = {a: unpack(st[a], m.n) for a in ("abs64", "abs1024", "pos1024", "squared1024", "fkeras", "msb")}
        v1 = np.load(base / "analytic.npz")
        v1 = {p: unpack(v1[f"{p:g}"], m.n) for p in RATES}
    else:
        from runR_kws import KwsModel, OUT as PREP
        m = KwsModel(threads)
        rep = int(key)
        xc, yc, _ = m.clips("ranking", 1024, 1024 * rep)
        xs, ys, _ = m.clips("empirical_validation", 512, 1024 + 512 * rep)
        xd, yd, idd = m.clips("empirical_validation", 512, 0)
        assert idd[0] == 0
        pa, sa = np.load(PREP / "masks" / f"rep{rep}" / "masks.npz"), np.load(PREP / "static" / f"rep{rep}" / "static.npz")
        static = {a: unpack(pa[f"{a}@{b}"], m.n) for a in ("abs1024", "pos1024")}
        static.update({a: unpack(sa[f"{a}@{b}"], m.n) for a in ("abs64", "msb", "fkeras")})
        v1 = np.load(RUNS / "paper1_runR_stage4_kws_masks_v1" / f"rep{rep}" / f"b{b}" / "analytic.npz")
        v1 = {p: unpack(v1[f"{p:g}"], m.n) for p in RATES}
    count = int(static["abs1024"].sum())
    # search-side statistics on the calibration inputs
    G = m.per_example(xc, yc)
    gbar = G.mean(0)
    se = G.std(0, ddof=1) / math.sqrt(G.shape[0])
    gsq = (G * G).astype(np.float32)
    m2 = bit_second_moments(m.jumps, gsq.mean(0).astype(np.float64))
    del G
    term = losses_term(m, xc, yc)
    J_sel, J_dev = objective_fn(m, xs, ys), objective_fn(m, xd, yd)
    abs_start = static["abs1024"].astype(np.int8)
    prev, rows = None, []
    for p in RATES:
        ts = time.time()
        rho, step = ladder_rates(IDEAL_LADDER, p)
        ev = per_input_evaluator(m.jumps, rho, term, gsq)
        starts = [abs_start, value_ranking_start(m.jumps, m2, rho, step, gbar, count)]
        if prev is not None:
            starts.append(prev)
        res, which = rate_aware_assignment(starts, m.jumps, m2, rho, step, ev, se=se, max_rounds=MAX_ROUNDS,
                                           min_swap=MIN_SWAP, accept_z=Z)
        prev = res.levels
        v2 = res.levels.astype(bool)
        pick, sel_scores = select_on_heldout({"v2": v2, "abs1024": static["abs1024"], "pos1024": static["pos1024"]},
                                             lambda mk: J_sel(mk, p))
        v2sel = {"v2": v2, "abs1024": static["abs1024"], "pos1024": static["pos1024"]}[pick]
        dev = {a: J_dev(mk, p) for a, mk in static.items()}
        dev.update(v1=J_dev(v1[p], p), v2=J_dev(v2, p), v2sel=J_dev(v2sel, p))
        rows.append(dict(rate=p, start=("abs1024", "value_ranking", "previous_rate")[which],
                         rounds=len(res.trace) - 1, accepted=sum(t.get("accepted", False) for t in res.trace[1:]),
                         changed_vs_abs1024=int((v2 & ~static["abs1024"]).sum()),
                         changed_v1_vs_abs1024=int((v1[p] & ~static["abs1024"]).sum()), count=count,
                         safeguard_pick=pick, selection_scores=sel_scores, dev=dev, v2_sha256=mask_sha256(v2),
                         seconds=round(time.time() - ts)))
        print(json.dumps({k: rows[-1][k] for k in ("rate", "start", "rounds", "accepted", "changed_vs_abs1024",
                                                    "changed_v1_vs_abs1024", "safeguard_pick", "seconds")}), flush=True)
    m.set_displacement(np.zeros(m.n))
    out.write_text(json.dumps(dict(model=model, key=key, rep=rep, budget=frac, z=Z, rows=rows,
                                   seconds=round(time.time() - t0), script_sha256=sha256_file(Path(__file__)),
                                   module_sha256=sha256_file(ROOT / "netsurf/rate_aware.py")), indent=0, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("job")
    j.add_argument("--model", choices=("resnet8", "kws"), required=True)
    j.add_argument("--checkpoint")
    j.add_argument("--rep", type=int)
    j.add_argument("--budget", type=float, required=True)
    j.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    job(a.model, a.checkpoint if a.model == "resnet8" else str(a.rep), a.budget, a.threads)
