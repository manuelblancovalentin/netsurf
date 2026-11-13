#!/usr/bin/env python3
"""Run R, Stage 2 (plan Section 5, amendment R-A4, Stage 2 addendum): the rate-aware mask under ideal suppression.

Phases, each write-once, in the order they run:

  bundle --checkpoint C
      Extract the v14 masks of the paper (abs64 = per-example absolute gradient from 64 images, FKeras,
      significance-first) for every Stage 2 replication at the four budgets into one small file per checkpoint, so
      that later jobs do not parse the 1-2 GB bundle file.
  build --checkpoint C --rep R --budget B
      1,024 calibration images (training images from offset 1000 * rep, as abs1024 / pos1024).  Saves the per-weight
      calibration statistics, the static masks of the budget (abs1024 and pos1024 from paper1_runR_masks_v1,
      hash-checked; squared1024 built here; the v14 masks from the bundle phase) and the analytical rate-aware mask
      for every rate.  Rates are swept from the lowest up; at each rate the search descends from the absolute-gradient
      assignment, from the ranking by v_b at mu = 0, and (above the lowest rate) from the previous rate's result, and
      keeps the lowest objective.
  eval --arms main --checkpoint C --rep R --field F
      Static arms and the analytical rate-aware mask.  Engine check as in Stage 1, then one fresh nested field,
      seeds [20261117, ckpt, rep, field]: every arm at the rate it was built for; for the pinned and under-trained
      checkpoints also every rate-aware mask built for rate p_a tested at every other rate p_b.
  mc --checkpoint C --rep R --budget B --rate P
      The MC-corrected mask for one budget and rate: descent from the analytical mask of the same budget and rate with
      the MC objective (K = 32 fields, refresh 1, seeds [20261118, ckpt, rep], shared by all budgets and rates).
  eval --arms mc ...
      The MC-corrected masks on the same fields as `eval --arms main` (same seeds, so the comparison is paired).

No asset lock is changed.  Interpreter: the TensorFlow 2.14.1 reference environment.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np

from runP_phase0 import CHECKPOINTS, PAPER, RANKINGS, ROOT, Model, bundle_dir, load_masks, sha256_file
from runP_phase12 import Replayer, pack, unpack
from runQ_calibration import OUT as RUNQ
from runR_stage1 import MASKS as RUNR_MASKS, NAMES, bkey, flips_sha, reps_for
from netsurf.mean_variance import bit_second_moments, mask_sha256, top_budget
from netsurf.rate_aware import (IDEAL_LADDER, analytic_evaluator, descend, ladder_rates, mc_evaluator, mc_uniforms,
                                rate_aware_assignment, value_ranking_start)

MASKS_OUT = PAPER / "workspace/runs/paper1_runR_stage2_masks_v1"
EVAL_OUT = PAPER / "workspace/runs/paper1_runR_stage2_eval_v1"
RATES = (1e-4, 3e-4, 1e-3, 2e-3, 3e-3, 5e-3, 1e-2, 3e-2)
FRACTIONS = (0.0625, 0.125, 0.1875, 0.25)
FIELDS = 8
EVAL_SEED, MC_SEED = 20261117, 20261118
K, REFRESH, MAX_ROUNDS, MIN_SWAP = 32, 1, 40, 16
CROSS = ("pinned", "undertrained_e100")
STATIC = ("abs64", "abs1024", "pos1024", "squared1024", "fkeras", "msb")
BUNDLE_KEYS = {"abs64": RANKINGS["abs_grad"], "fkeras": RANKINGS["fkeras"], "msb": RANKINGS["msb"]}


def pkey(p):
    return f"{p:g}"


def calibration(model, rep):
    x, y, idx = model.images("ranking", 1024, 1000 * rep)
    assert idx[0] == 1000 * rep and len(idx) == 1024
    return x, y


def write_done(out, t0, **extra):
    (out / "DONE").write_text(json.dumps(dict(seconds=round(time.time() - t0), script_sha256=sha256_file(Path(__file__)),
                                              module_sha256=sha256_file(ROOT / "netsurf/rate_aware.py"), **extra)))


def budget_dir(root, name, rep, frac):
    return root / name / f"rep{rep}" / f"b{bkey(frac)}"


# ------------------------------------------------------------------------------------------------ bundle
def bundle(name, root):
    out = root / name
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    model = Model(name, 1)
    reps = reps_for(name)
    masks, _ = load_masks(model, bundle_dir(CHECKPOINTS[name]["bundle"]), reps)
    arrays, info = {}, {}
    for rep in reps:
        for f in FRACTIONS:
            for a, k in BUNDLE_KEYS.items():
                m = masks[rep][f"{k}@{bkey(f)}"]
                arrays[f"r{rep}:{a}@{bkey(f)}"] = pack(m)
                info[f"r{rep}:{a}@{bkey(f)}"] = dict(count=int(m.sum()), sha256=mask_sha256(m))
    np.savez(out / "bundle_masks.npz", **arrays)
    (out / "bundle_masks.json").write_text(json.dumps(info, indent=0))
    write_done(out, t0)


# ------------------------------------------------------------------------------------------------ build
def build(name, rep, frac, threads, root, rates, max_rounds):
    assert rep in reps_for(name)
    if not (root / name / "DONE").exists():
        raise SystemExit("the bundle phase is not finished for this checkpoint")
    b = bkey(frac)
    out = budget_dir(root, name, rep, frac)
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    model = Model(name, threads)
    x, y = calibration(model, rep)
    G = model.per_example(x, y)
    g2, gbar = (G * G).mean(0), G.mean(0)
    se = G.std(0, ddof=1) / math.sqrt(G.shape[0])
    del G
    m2 = bit_second_moments(model.jumps, g2)
    np.savez(out / "calibration.npz", g2=g2, gbar=gbar, se=se)
    count = int(round(frac * model.n * 8))
    rq = np.load(RUNR_MASKS / name / f"rep{rep}" / "masks.npz")
    rinfo = json.loads((RUNR_MASKS / name / f"rep{rep}" / "masks.json").read_text())["masks"]
    bz = np.load(root / name / "bundle_masks.npz")
    static = {}
    for a in ("abs1024", "pos1024"):
        mk = unpack(rq[f"{a}@{b}"], model.n)
        if mask_sha256(mk) != rinfo[f"{a}@{b}"]["sha256"]:
            raise SystemExit(f"{a}@{b} does not match its record")
        static[a] = mk
    static["squared1024"] = top_budget(m2, count)
    for a in BUNDLE_KEYS:
        static[a] = unpack(bz[f"r{rep}:{a}@{b}"], model.n)
    info = {a: dict(count=int(m.sum()), sha256=mask_sha256(m)) for a, m in static.items()}
    np.savez(out / "static.npz", **{a: pack(m) for a, m in static.items()})
    model.set_displacement(np.zeros(model.n))
    clean, _ = model.loss_and_grad(x, y)
    term = model.mean_term(x, y, clean)
    abs_start = static["abs1024"].astype(np.int8)
    analytic, traces, records, prev = {}, {}, [], None
    for p in rates:
        ts = time.time()
        rho, step = ladder_rates(IDEAL_LADDER, p)
        ev = analytic_evaluator(model.jumps, m2, rho, term)
        starts = [abs_start, value_ranking_start(model.jumps, m2, rho, step, gbar, count)]
        if prev is not None:
            starts.append(prev)
        res, which = rate_aware_assignment(starts, model.jumps, m2, rho, step, ev, se=se,
                                           max_rounds=max_rounds, min_swap=MIN_SWAP)
        if int(res.levels.sum()) != count:
            raise AssertionError("budget not preserved")
        prev = res.levels
        k = pkey(p)
        analytic[k] = res.levels.astype(bool)
        traces[k] = res.trace
        records.append(dict(rate=p, start=("abs1024", "value_ranking", "previous_rate")[which],
                            objective=res.objective, initial_objective=res.initial_objective,
                            rounds=len(res.trace) - 1, sha256=mask_sha256(analytic[k]),
                            changed_vs_abs1024=int((analytic[k] & ~static["abs1024"]).sum()),
                            seconds=round(time.time() - ts)))
        print(json.dumps(records[-1]), flush=True)
    model.set_displacement(np.zeros(model.n))
    np.savez(out / "analytic.npz", **{k: pack(v) for k, v in analytic.items()})
    (out / "traces_analytic.json").write_text(json.dumps(traces, default=float))
    (out / "build.json").write_text(json.dumps(dict(static=info, analytic=records, count=count, rates=list(rates),
                                                    max_rounds=max_rounds, clean_calibration_loss=float(clean)),
                                               indent=0, default=float))
    write_done(out, t0)


# ------------------------------------------------------------------------------------------------ mc
def mc(name, rep, frac, p, threads, root, max_rounds):
    base = budget_dir(root, name, rep, frac)
    if not (base / "DONE").exists():
        raise SystemExit("the build phase is not finished for this budget")
    out = base / f"mc_{pkey(p)}"
    out.mkdir(parents=False, exist_ok=False)
    t0 = time.time()
    model = Model(name, threads)
    x, y = calibration(model, rep)
    cal = np.load(base / "calibration.npz")
    m2 = bit_second_moments(model.jumps, cal["g2"])
    model.set_displacement(np.zeros(model.n))
    clean, _ = model.loss_and_grad(x, y)
    term = model.mean_term(x, y, clean)
    ck = NAMES.index(name)
    u = mc_uniforms(model.n, K, (MC_SEED, ck, rep))
    rho, step = ladder_rates(IDEAL_LADDER, p)
    start = unpack(np.load(base / "analytic.npz")[pkey(p)], model.n).astype(np.int8)
    res = descend(start, model.jumps, m2, rho, step, mc_evaluator(model.jumps, m2, rho, term, u, refresh=REFRESH),
                  se=cal["se"], max_rounds=max_rounds, min_swap=MIN_SWAP)
    model.set_displacement(np.zeros(model.n))
    mask = res.levels.astype(bool)
    np.savez(out / "mc.npz", mask=pack(mask))
    (out / "trace.json").write_text(json.dumps(res.trace, default=float))
    rec = dict(rate=p, objective=res.objective, initial_objective=res.initial_objective, rounds=len(res.trace) - 1,
               sha256=mask_sha256(mask), changed_vs_analytic=int((mask & ~start.astype(bool)).sum()), K=K,
               refresh=REFRESH, seed=[MC_SEED, ck, rep], max_rounds=max_rounds)
    (out / "mc.json").write_text(json.dumps(rec, indent=0, default=float))
    print(json.dumps(rec, default=float), flush=True)
    write_done(out, t0)


# ------------------------------------------------------------------------------------------------ eval
def evaluate(arms, name, rep, fi, threads, root, out_root, rates, fractions):
    assert rep in reps_for(name) and 0 <= fi < FIELDS
    masks = {}
    model = Model(name, threads)
    t0 = time.time()
    for f in fractions:
        base = budget_dir(root, name, rep, f)
        b = bkey(f)
        if arms == "main":
            if not (base / "DONE").exists():
                raise SystemExit(f"the build phase is not finished for budget {b}")
            st, an = np.load(base / "static.npz"), np.load(base / "analytic.npz")
            masks.update({(a, b, None): unpack(st[a], model.n) for a in STATIC})
            masks.update({("ra_analytic", b, float(k)): unpack(an[k], model.n) for k in an.files})
        else:
            for p in rates:
                d = base / f"mc_{pkey(p)}"
                if not (d / "DONE").exists():
                    raise SystemExit(f"the mc phase is not finished for budget {b}, rate {p:g}")
                masks[("ra_mc", b, p)] = unpack(np.load(d / "mc.npz")["mask"], model.n)
    out = out_root / arms / name / f"rep{rep}" / f"field{fi}"
    out.mkdir(parents=True, exist_ok=False)
    xe, ye, ie = model.images("evaluation", 1000, 1000 * rep)
    assert ie[0] == 1000 * rep and len(ie) == 1000
    rp = Replayer(model)
    with open(RUNQ / name / f"rep{rep}" / "replay.csv") as fh:
        ref = next(r for r in csv.DictReader(fh) if r["arm"] == "abs1024" and abs(float(r["budget"]) - 0.125) < 1e-12)
    spec = CHECKPOINTS[name]
    sched = next(s for s in json.loads((bundle_dir(spec["bundle"]) / "artifacts/fault_schedules.json").read_text())
                 [f"replication_{rep}"] if s["sha256"] == ref["schedule_sha256"])
    rq = np.load(RUNR_MASKS / name / f"rep{rep}" / "masks.npz")
    acc, _, _, _ = rp.evaluate(rp.flips(sched["events"]), unpack(rq["abs1024@0.125000"], model.n), xe, ye)
    check = dict(schedule_sha256=ref["schedule_sha256"], replay_accuracy=acc, saved_accuracy=float(ref["accuracy"]),
                 match=acc == float(ref["accuracy"]))
    (out / "check.json").write_text(json.dumps(check, indent=0))
    if not check["match"]:
        raise SystemExit(f"engine check failed: {check}")
    none = np.zeros((model.n, 8), dtype=bool)
    clean_acc, clean_ce, _, _ = rp.evaluate(none, none, xe, ye)
    ck = NAMES.index(name)
    u = np.random.default_rng([EVAL_SEED, ck, rep, fi]).random((model.n, 8))
    rows, log = [], dict(field=fi, seed=[EVAL_SEED, ck, rep, fi], flips_sha256={})
    for p in rates:
        fl = u < p
        log["flips_sha256"][pkey(p)] = flips_sha(fl)
        for (a, b, pa), mk in masks.items():
            if pa is not None and pa != p and name not in CROSS:
                continue
            acc, ce, prop, app = rp.evaluate(fl, mk, xe, ye)
            rows.append(dict(checkpoint=name, ckpt=ck, rep=rep, field=fi, ber=p, budget=float(b), arm=a,
                             built_for="" if pa is None else pa, accuracy=acc, cross_entropy=ce,
                             proposed=prop, applied=app))
        print(json.dumps(dict(job=f"{arms}/{name}/rep{rep}/field{fi}", rate_done=p, seconds=round(time.time() - t0))),
              flush=True)
    with open(out / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (out / "field.json").write_text(json.dumps(dict(fields=[log], rates=list(rates), clean_accuracy=clean_acc,
                                                     clean_cross_entropy=clean_ce), indent=0))
    write_done(out, t0, evaluations=len(rows))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="phase", required=True)
    for ph in ("bundle", "build", "mc", "eval"):
        s = sub.add_parser(ph)
        s.add_argument("--checkpoint", required=True, choices=NAMES)
        s.add_argument("--threads", type=int, default=1)
        s.add_argument("--masks-out", type=Path, default=MASKS_OUT)
        s.add_argument("--max-rounds", type=int, default=MAX_ROUNDS, help="smoke tests only")
        if ph != "bundle":
            s.add_argument("--rep", type=int, required=True)
        if ph in ("build", "mc"):
            s.add_argument("--budget", type=float, required=True, choices=FRACTIONS)
        if ph == "build":
            s.add_argument("--rates", type=float, nargs="*", default=list(RATES))
        if ph == "mc":
            s.add_argument("--rate", type=float, required=True)
        if ph == "eval":
            s.add_argument("--arms", required=True, choices=("main", "mc"))
            s.add_argument("--field", type=int, required=True)
            s.add_argument("--out", type=Path, default=EVAL_OUT)
            s.add_argument("--rates", type=float, nargs="*", default=list(RATES))
            s.add_argument("--fractions", type=float, nargs="*", default=list(FRACTIONS))
    a = ap.parse_args()
    frozen = a.masks_out == MASKS_OUT or getattr(a, "out", None) == EVAL_OUT
    if frozen and (a.max_rounds != MAX_ROUNDS or tuple(getattr(a, "rates", RATES)) != RATES
                   or tuple(getattr(a, "fractions", FRACTIONS)) != FRACTIONS
                   or (a.phase == "mc" and a.rate not in RATES)):
        raise SystemExit("the frozen output directories only take the frozen constants, rates and budgets")
    if a.phase == "bundle":
        bundle(a.checkpoint, a.masks_out)
    elif a.phase == "build":
        build(a.checkpoint, a.rep, a.budget, a.threads, a.masks_out, tuple(a.rates), a.max_rounds)
    elif a.phase == "mc":
        mc(a.checkpoint, a.rep, a.budget, a.rate, a.threads, a.masks_out, a.max_rounds)
    else:
        evaluate(a.arms, a.checkpoint, a.rep, a.field, a.threads, a.masks_out, a.out, tuple(a.rates),
                 tuple(a.fractions))


if __name__ == "__main__":
    main()
