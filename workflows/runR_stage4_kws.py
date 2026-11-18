#!/usr/bin/env python3
"""Run R, Stage 4 on the keyword-spotting DS-CNN (plan `RUN_R_STAGE4_DSCNN_*.md`).  Write-once phases:

  build --rep R --budget B
      Calibration statistics from the 1,024 training clips of draw R; the static masks of the budget (abs1024 and
      pos1024 from the prep run, abs64 / significance-first / FKeras from the prep `static` command, hash-checked);
      the analytic rate-aware mask for each of the twelve rates (same search and starts as Stage 2).
  eval --rep R --field F
      Engine check: replays one saved v14 DS-CNN trial (significance-first at 12.5%, BER 3e-3, v14 replication 1
      window) and must reproduce its accuracy exactly.  Then one fresh nested field, seeds [20261119, R, F], on test
      window R (800 clips from offset 800 R): every static arm and the analytic mask built for the tested rate.
  mc --rep R --budget B --rate P
      MC-corrected mask (K = 32, refresh 1, seeds [20261118, 9, R]); run only if rule R2c holds on ResNet-8.

Interpreter: the TensorFlow 2.14.1 reference environment.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np

from runP_phase0 import PAPER, ROOT, sha256_file
from runP_phase12 import Replayer, pack, unpack
from runR_kws import CAL_COUNT, OUT as PREP, STAGE4_FRACTIONS, KwsModel
from runR_stage1 import RATES, bkey, flips_sha
from netsurf.mean_variance import bit_second_moments, mask_sha256
from netsurf.rate_aware import (IDEAL_LADDER, analytic_evaluator, descend, ladder_rates, mc_evaluator, mc_uniforms,
                                rate_aware_assignment, value_ranking_start)

MASKS_OUT = PAPER / "workspace/runs/paper1_runR_stage4_kws_masks_v1"
EVAL_OUT = PAPER / "workspace/runs/paper1_runR_stage4_kws_eval_v1"
REPS = tuple(range(6))
FIELDS, WINDOW = 8, 800
EVAL_SEED, MC_SEED, CKPT_INDEX = 20261119, 20261118, 9
K, REFRESH, MAX_ROUNDS, MIN_SWAP = 32, 1, 40, 16
STATIC = ("abs1024", "pos1024", "abs64", "msb", "fkeras")


def pkey(p):
    return f"{p:g}"


def write_done(out, t0, **extra):
    (out / "DONE").write_text(json.dumps(dict(seconds=round(time.time() - t0), script_sha256=sha256_file(Path(__file__)),
                                              module_sha256=sha256_file(ROOT / "netsurf/rate_aware.py"), **extra)))


def calibration(m, rep):
    x, y, idx = m.clips("ranking", CAL_COUNT, CAL_COUNT * rep)
    assert idx[0] == CAL_COUNT * rep and len(idx) == CAL_COUNT
    return x, y


def load_static(m, rep, frac):
    b = bkey(frac)
    out = {}
    pa = np.load(PREP / "masks" / f"rep{rep}" / "masks.npz")
    pinfo = json.loads((PREP / "masks" / f"rep{rep}" / "masks.json").read_text())["masks"]
    st = np.load(PREP / "static" / f"rep{rep}" / "static.npz")
    sinfo = json.loads((PREP / "static" / f"rep{rep}" / "static.json").read_text())["masks"]
    for a in STATIC:
        k = f"{a}@{b}"
        src, info = (pa, pinfo) if a in ("abs1024", "pos1024") else (st, sinfo)
        mk = unpack(src[k], m.n)
        if mask_sha256(mk) != info[k]["sha256"]:
            raise SystemExit(f"{k} does not match its record")
        out[a] = mk
    return out


def build(rep, frac, threads, root, rates, max_rounds):
    out = root / f"rep{rep}" / f"b{bkey(frac)}"
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    m = KwsModel(threads)
    x, y = calibration(m, rep)
    G = m.per_example(x, y)
    g2, gbar = (G * G).mean(0), G.mean(0)
    se = G.std(0, ddof=1) / math.sqrt(G.shape[0])
    del G
    m2 = bit_second_moments(m.jumps, g2)
    np.savez(out / "calibration.npz", g2=g2, gbar=gbar, se=se)
    static = load_static(m, rep, frac)
    count = int(static["abs1024"].sum())
    m.set_displacement(np.zeros(m.n))
    clean, _ = m.loss_and_grad(x, y)
    term = m.mean_term(x, y, clean)
    abs_start = static["abs1024"].astype(np.int8)
    analytic, traces, records, prev = {}, {}, [], None
    for p in rates:
        ts = time.time()
        rho, step = ladder_rates(IDEAL_LADDER, p)
        ev = analytic_evaluator(m.jumps, m2, rho, term)
        starts = [abs_start, value_ranking_start(m.jumps, m2, rho, step, gbar, count)]
        if prev is not None:
            starts.append(prev)
        res, which = rate_aware_assignment(starts, m.jumps, m2, rho, step, ev, se=se, max_rounds=max_rounds,
                                           min_swap=MIN_SWAP)
        if int(res.levels.sum()) != count:
            raise AssertionError("budget not preserved")
        prev = res.levels
        analytic[pkey(p)] = res.levels.astype(bool)
        traces[pkey(p)] = res.trace
        records.append(dict(rate=p, start=("abs1024", "value_ranking", "previous_rate")[which], objective=res.objective,
                            initial_objective=res.initial_objective, rounds=len(res.trace) - 1,
                            sha256=mask_sha256(analytic[pkey(p)]),
                            changed_vs_abs1024=int((analytic[pkey(p)] & ~static["abs1024"]).sum()),
                            seconds=round(time.time() - ts)))
        print(json.dumps(records[-1]), flush=True)
    m.set_displacement(np.zeros(m.n))
    np.savez(out / "analytic.npz", **{k: pack(v) for k, v in analytic.items()})
    (out / "traces_analytic.json").write_text(json.dumps(traces, default=float))
    (out / "build.json").write_text(json.dumps(dict(analytic=records, count=count, rates=list(rates),
                                                    clean_calibration_loss=float(clean)), indent=0, default=float))
    write_done(out, t0)


def mc(rep, frac, p, threads, root, max_rounds):
    base = root / f"rep{rep}" / f"b{bkey(frac)}"
    out = base / f"mc_{pkey(p)}"
    out.mkdir(parents=False, exist_ok=False)
    t0 = time.time()
    m = KwsModel(threads)
    x, y = calibration(m, rep)
    cal = np.load(base / "calibration.npz")
    m2 = bit_second_moments(m.jumps, cal["g2"])
    m.set_displacement(np.zeros(m.n))
    clean, _ = m.loss_and_grad(x, y)
    term = m.mean_term(x, y, clean)
    u = mc_uniforms(m.n, K, (MC_SEED, CKPT_INDEX, rep))
    rho, step = ladder_rates(IDEAL_LADDER, p)
    start = unpack(np.load(base / "analytic.npz")[pkey(p)], m.n).astype(np.int8)
    res = descend(start, m.jumps, m2, rho, step, mc_evaluator(m.jumps, m2, rho, term, u, refresh=REFRESH),
                  se=cal["se"], max_rounds=max_rounds, min_swap=MIN_SWAP)
    m.set_displacement(np.zeros(m.n))
    mask = res.levels.astype(bool)
    np.savez(out / "mc.npz", mask=pack(mask))
    (out / "trace.json").write_text(json.dumps(res.trace, default=float))
    (out / "mc.json").write_text(json.dumps(dict(rate=p, objective=res.objective, initial_objective=res.initial_objective,
                                                 rounds=len(res.trace) - 1, sha256=mask_sha256(mask)), default=float))
    write_done(out, t0)


def evaluate(rep, fi, threads, root, out_root, rates, fractions):
    assert rep in REPS and 0 <= fi < FIELDS
    m = KwsModel(threads)
    t0 = time.time()
    masks = {}
    for f in fractions:
        base = root / f"rep{rep}" / f"b{bkey(f)}"
        if not (base / "DONE").exists():
            raise SystemExit(f"build is not finished for budget {bkey(f)}")
        for a, mk in load_static(m, rep, f).items():
            masks[(a, f, None)] = mk
        an = np.load(base / "analytic.npz")
        for k in an.files:
            masks[("ra_analytic", f, float(k))] = unpack(an[k], m.n)
    out = out_root / f"rep{rep}" / f"field{fi}"
    out.mkdir(parents=True, exist_ok=False)
    rp = Replayer(m)
    ref = json.loads((PREP / "engine_ref.json").read_text())
    xr, yr, ir = m.clips("evaluation", len(ref["evaluation_indices"]), ref["evaluation_indices"][0])
    assert list(ir) == list(ref["evaluation_indices"])
    acc, _, _, _ = rp.evaluate(rp.flips(ref["events"]), masks[("msb", 0.125, None)], xr, yr)
    check = dict(policy=ref["policy"], replay_accuracy=acc, saved_accuracy=ref["accuracy"], match=acc == ref["accuracy"])
    (out / "check.json").write_text(json.dumps(check, indent=0))
    if not check["match"]:
        raise SystemExit(f"engine check failed: {check}")
    xe, ye, ie = m.clips("evaluation", WINDOW, WINDOW * rep)
    assert ie[0] == WINDOW * rep and len(ie) == WINDOW
    none = np.zeros((m.n, 8), dtype=bool)
    clean_acc, clean_ce, _, _ = rp.evaluate(none, none, xe, ye)
    u = np.random.default_rng([EVAL_SEED, rep, fi]).random((m.n, 8))
    rows, log = [], dict(field=fi, seed=[EVAL_SEED, rep, fi], flips_sha256={})
    for p in rates:
        fl = u < p
        log["flips_sha256"][pkey(p)] = flips_sha(fl)
        for (a, f, pa), mk in masks.items():
            if pa is not None and pa != p:
                continue
            acc, ce, prop, app = rp.evaluate(fl, mk, xe, ye)
            rows.append(dict(rep=rep, field=fi, ber=p, budget=f, arm=a, built_for="" if pa is None else pa,
                             accuracy=acc, cross_entropy=ce, proposed=prop, applied=app))
        print(json.dumps(dict(job=f"rep{rep}/field{fi}", rate_done=p, seconds=round(time.time() - t0))), flush=True)
    with open(out / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (out / "field.json").write_text(json.dumps(dict(fields=[log], rates=list(rates), clean_accuracy=clean_acc,
                                                     clean_cross_entropy=clean_ce), indent=0))
    write_done(out, t0, evaluations=len(rows))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="phase", required=True)
    for ph in ("build", "mc", "eval"):
        s = sub.add_parser(ph)
        s.add_argument("--rep", type=int, required=True, choices=REPS)
        s.add_argument("--threads", type=int, default=1)
        s.add_argument("--masks-out", type=Path, default=MASKS_OUT)
        s.add_argument("--max-rounds", type=int, default=MAX_ROUNDS)
        s.add_argument("--rates", type=float, nargs="*", default=list(RATES))
        if ph in ("build", "mc"):
            s.add_argument("--budget", type=float, required=True, choices=STAGE4_FRACTIONS)
        if ph == "mc":
            s.add_argument("--rate", type=float, required=True)
        if ph == "eval":
            s.add_argument("--field", type=int, required=True)
            s.add_argument("--out", type=Path, default=EVAL_OUT)
            s.add_argument("--fractions", type=float, nargs="*", default=list(STAGE4_FRACTIONS))
    a = ap.parse_args()
    frozen = a.masks_out == MASKS_OUT or getattr(a, "out", None) == EVAL_OUT
    if frozen and (a.max_rounds != MAX_ROUNDS or tuple(a.rates) != RATES
                   or tuple(getattr(a, "fractions", STAGE4_FRACTIONS)) != STAGE4_FRACTIONS):
        raise SystemExit("the frozen output directories only take the frozen constants, rates and budgets")
    if a.phase == "build":
        build(a.rep, a.budget, a.threads, a.masks_out, tuple(a.rates), a.max_rounds)
    elif a.phase == "mc":
        mc(a.rep, a.budget, a.rate, a.threads, a.masks_out, a.max_rounds)
    else:
        evaluate(a.rep, a.field, a.threads, a.masks_out, a.out, tuple(a.rates), tuple(a.fractions))


if __name__ == "__main__":
    main()
