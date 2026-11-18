#!/usr/bin/env python3
"""EXPLORATORY: greedy group-level protection search on the DS-CNN, every decision judged by network passes
(plan review/GROUP_SEARCH_PLAN_v2_2026-11-14.md).

  search    --kind mc|gauss --rate P   One path for replication 0 on the first --clips calibration clips.  Each round:
            K shared fault fields (mc: real flips, seeds [20261211, 0, rate index, round, draw]; gauss: Gaussian
            displacement of every unprotected bit, seeds [20261214, ...]), the current mask and every unprotected
            layer x bit group added to it, paired gain in accuracy per protected bit, ties by cross-entropy then group
            id (netsurf.group_search).  Runs until the largest budget is covered; nested budget masks with a random
            partial group (seed [20261213, 0, rate index]) and whole-groups-only masks.  Output <out>/<kind>_<rate>/:
            rounds.json, masks.npz (full@<budget>, whole@<budget>), DONE.  Progress: one JSON line per round.
  evaluate  --field F   The search masks in the Stage 4 protocol (TFLite replay after the engine check, test window 0,
            field seeds [20261119, 0, F]), each mask at the rate it was built for.  <out>/field<F>/results.csv.

Both write-once.  Progress lines carry done/total for the run monitor.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from site_paths import WT  # noqa: E402

RUNS = WT / "paper1-tecs-revision-v1/workspace/runs"
REP = 0
FRACTIONS = (0.125, 0.1875, 0.25, 0.375)
SEED_MC, SEED_GAUSS, SEED_FILL = 20261211, 20261214, 20261213
REFERENCE_TF = "2.14.1"      # Stage 4 and every locked run used it; TF 2.21 gives slightly different TFLite outputs


def require_reference_tf(tf):
    if tf.__version__ != REFERENCE_TF:
        raise SystemExit(f"TensorFlow {tf.__version__}: use the reference environment (TensorFlow {REFERENCE_TF}; "
                         "kona: /projects/kona-assets/manuelbv/.venv-mlperf/bin/python, Quest: netsurf-gpu)")


def log(**kw):
    print(json.dumps(dict(t=time.strftime("%H:%M:%S"), **kw)), flush=True)


def rate_index(p):
    from runR_stage1 import RATES
    idx = [i for i, r in enumerate(RATES) if abs(r - p) < 1e-12]
    if not idx:
        raise SystemExit(f"rate {p} is not on the Stage 1/4 grid {RATES}")
    return idx[0]


def cmd_search(a):
    from runP_phase12 import pack
    from runR_kws import CAL_COUNT, KwsModel
    from runR_stage1 import bkey
    from netsurf.group_search import budget_counts, evaluate_round, groups, path_masks, protect
    t0 = time.time()
    m = KwsModel(a.threads)
    tf = m.tf
    require_reference_tf(tf)
    x, y, idx = m.clips("ranking", a.clips, CAL_COUNT * REP)
    x, y = np.asarray(x), np.asarray(y).astype(np.int64)
    labels = tf.convert_to_tensor(y)
    p, ri = a.rate, rate_index(a.rate)
    out = a.out / f"{a.kind}_{p:g}"
    out.mkdir(parents=True, exist_ok=False)
    gs = groups(m.starts)
    J = m.jumps
    J2 = J * J
    counts = budget_counts(m.n * 8, FRACTIONS)
    target = max(counts.values())

    def run(disp):
        m.set_displacement(disp)
        logits, _ = m.comp.forward(x)
        acc = float(np.mean(logits.numpy().argmax(1) == y))
        ce = float(tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)).numpy())
        return acc, ce

    clean_acc, clean_ce = run(np.zeros(m.n))
    every = np.ones((m.n, 8), dtype=bool)
    probe = np.random.default_rng([SEED_MC, REP, ri, 0, 0]).random((m.n, 8)) < p
    assert run(np.where(probe & ~every, J, 0.0).sum(1)) == (clean_acc, clean_ce), "full protection must give the clean result"
    log(phase="search", kind=a.kind, rate=p, clips=len(y), K=a.K, clean_accuracy=clean_acc, done=0, total=target,
        seconds=round(time.time() - t0))

    mask = np.zeros((m.n, 8), dtype=bool)
    accepted, rounds, passes = [], [], 0
    t_search = time.time()
    while mask.sum() < target:
        ri_round = len(rounds)
        if a.max_rounds and ri_round >= a.max_rounds:
            break
        if a.kind == "mc":
            fields = [np.random.default_rng([SEED_MC, REP, ri, ri_round, d]).random((m.n, 8)) < p for d in range(a.K)]
        else:
            eps = [np.random.default_rng([SEED_GAUSS, REP, ri, ri_round, d]).standard_normal(m.n) for d in range(a.K)]

        def measure(masks):
            nonlocal passes
            S = np.empty((len(masks), a.K))
            T = np.empty_like(S)
            for j, mk in enumerate(masks):
                if a.kind == "gauss":
                    mu = p * np.where(mk, 0.0, J).sum(1)
                    sd = np.sqrt(np.maximum(p * (1 - p) * np.where(mk, 0.0, J2).sum(1), 0.0))
                for d in range(a.K):
                    disp = np.where(fields[d] & ~mk, J, 0.0).sum(1) if a.kind == "mc" else mu + sd * eps[d]
                    S[j, d], ce = run(disp)
                    T[j, d] = -ce
                    passes += 1
            return S, T

        tr = time.time()
        r = evaluate_round(ri_round, mask, gs, measure)
        g = gs[r.accepted]
        mask = protect(mask, g)
        accepted.append(r.accepted)
        best = next(c for c in r.candidates if c["gid"] == r.accepted)
        rec = asdict(r)
        rec.update(seconds=round(time.time() - tr, 1), protected=int(mask.sum()))
        rounds.append(rec)
        el = time.time() - t_search
        done = int(min(mask.sum(), target))
        log(round=ri_round, done=done, total=target, accepted=dict(layer=g.layer, bit=g.bit, size=g.hi - g.lo),
            gain=round(best["gain"], 5), se=round(best["se"], 5), current_accuracy=round(r.current, 4),
            round_seconds=rec["seconds"], seconds=round(el), eta_minutes=round(el / done * (target - done) / 60, 1))
        if a.max_round_seconds and rec["seconds"] > a.max_round_seconds:
            (out / "rounds.json").write_text(json.dumps(rounds))
            raise SystemExit(f"round {ri_round} took {rec['seconds']} s, more than the limit {a.max_round_seconds} s")
    m.set_displacement(np.zeros(m.n))
    (out / "rounds.json").write_text(json.dumps(rounds))
    meta = dict(kind=a.kind, rate=p, rate_index=ri, K=a.K, clips=len(y), calibration_first=idx[0], accepted=accepted,
                clean_accuracy=clean_acc, passes=passes, seconds=round(time.time() - t0, 1), budgets=counts)
    if mask.sum() >= target:
        full, whole = path_masks(m.n, gs, accepted, counts, np.random.default_rng([SEED_FILL, REP, ri]))
        np.savez(out / "masks.npz", **{f"full@{bkey(f)}": pack(v) for f, v in full.items()},
                 **{f"whole@{bkey(f)}": pack(v) for f, v in whole.items()})
        meta["whole_bits"] = {f"{f:g}": int(v.sum()) for f, v in whole.items()}
    else:
        meta["stopped_early"] = True
    (out / "DONE").write_text(json.dumps(meta, indent=1))
    log(stage="search done", rounds=len(rounds), passes=passes, seconds=round(time.time() - t0),
        seconds_per_pass=round((time.time() - t_search) / max(passes, 1), 4))


def cmd_evaluate(a):
    from runP_phase12 import Replayer, unpack
    from runR_kws import OUT as PREP, KwsModel
    from runR_stage4_kws import EVAL_SEED, WINDOW, load_static
    m = KwsModel(a.threads)
    require_reference_tf(m.tf)
    t0 = time.time()
    masks = []                                          # (arm, variant, budget, rate, mask)
    for d in sorted(a.search.glob("*_*")):
        if not (d / "masks.npz").exists():
            continue
        meta = json.loads((d / "DONE").read_text())
        z = np.load(d / "masks.npz")
        for k in z.files:
            variant, b = k.split("@")
            masks.append((f"group_{meta['kind']}", variant, float(b), float(meta["rate"]), unpack(z[k], m.n)))
    out = a.out / f"field{a.field}"
    out.mkdir(parents=True, exist_ok=False)
    rp = Replayer(m)
    ref = json.loads((PREP / "engine_ref.json").read_text())
    xr, yr, ir = m.clips("evaluation", len(ref["evaluation_indices"]), ref["evaluation_indices"][0])
    assert list(ir) == list(ref["evaluation_indices"])
    acc, _, _, _ = rp.evaluate(rp.flips(ref["events"]), load_static(m, REP, 0.125)["msb"], xr, yr)
    check = dict(replay_accuracy=acc, saved_accuracy=ref["accuracy"], match=acc == ref["accuracy"])
    (out / "check.json").write_text(json.dumps(check))
    if not check["match"]:
        raise SystemExit(f"engine check failed: {check}")
    xe, ye, ie = m.clips("evaluation", WINDOW, WINDOW * REP)
    assert ie[0] == WINDOW * REP and len(ie) == WINDOW
    u = np.random.default_rng([EVAL_SEED, REP, a.field]).random((m.n, 8))
    log(phase="evaluate", field=a.field, masks=len(masks), done=0, total=len(masks))
    rows = []
    for i, (arm, variant, b, p, mk) in enumerate(masks):
        acc, ce, prop, app = rp.evaluate(u < p, mk, xe, ye)
        rows.append(dict(rep=REP, field=a.field, ber=p, budget=b, arm=arm, variant=variant, bits=int(mk.sum()),
                         accuracy=acc, cross_entropy=ce, proposed=prop, applied=app))
        el = time.time() - t0
        log(field=a.field, done=i + 1, total=len(masks), seconds=round(el), eta_minutes=round(el / (i + 1) * (len(masks) - i - 1) / 60, 1))
    with open(out / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (out / "DONE").write_text(json.dumps(dict(seconds=round(time.time() - t0), evaluations=len(rows))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("search", "evaluate"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--kind", choices=("mc", "gauss"), default="mc")
    ap.add_argument("--rate", type=float, default=3e-3)
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--clips", type=int, default=512)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-rounds", type=int, default=0, help="stop after this many rounds (smoke tests)")
    ap.add_argument("--max-round-seconds", type=float, default=0, help="stop if one round takes longer (cost guard)")
    ap.add_argument("--field", type=int, default=0)
    ap.add_argument("--search", type=Path, default=RUNS / "paper1_group_search_kws_v1")
    a = ap.parse_args()
    {"search": cmd_search, "evaluate": cmd_evaluate}[a.command](a)


if __name__ == "__main__":
    main()
