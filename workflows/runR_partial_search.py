#!/usr/bin/env python3
"""Partial-group protection search on the DS-CNN (plan review/PARTIAL_GROUP_SEARCH_PLAN_2026-11-14.md).

Each layer x bit group is split into --parts slices in abs1024 order (highest per-weight score first); every round,
each group's next slice is a candidate, judged by real-flip network passes on K shared fault fields at the target rate
(paired gain in accuracy per protected bit, ties by cross-entropy then slice id).  The search therefore chooses only
how much of each group to protect; within a group the order is abs1024's.

  search    --rep R --rate P --parts S   One path on the first --clips calibration clips of replication R.  The abs
            scores are recomputed from the replication's 1,024 calibration clips and must rebuild its stored abs1024
            masks exactly.  Seeds [20261215, R, rate index, round, draw].  Output <out>/rep<R>_mc_<P>_s<S>/: rounds.json,
            masks.npz (mask@<budget>), scores_check.json, DONE.
  evaluate  --rep R --field F   Every finished path of replication R in the Stage 4 protocol (engine check, TFLite on
            test window R, field seeds [20261119, R, F]), each mask at its own rate.  <out>/rep<R>/field<F>/results.csv.

Write-once.  One JSON progress line per round with done/total (protected bits) for the run monitor.
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
from runR_group_search import FRACTIONS, log, rate_index, require_reference_tf  # noqa: E402

RUNS = WT / "paper1-tecs-revision-v1/workspace/runs"
SEED_PARTIAL = 20261215


def rebuild_differences(score, stored: dict, counts: dict) -> dict:
    """For each budget fraction, the bits of the stored top-budget mask that the recomputed score would not select
    (0 everywhere = exact rebuild)."""
    from netsurf.mean_variance import top_budget
    return {f"{f:g}": int((stored[f] & ~top_budget(score, counts[f])).sum()) for f in stored}


def cmd_search(a):
    from runP_phase12 import pack, unpack
    from runQ_calibration import scores
    from runR_kws import CAL_COUNT, OUT as PREP, KwsModel
    from runR_stage1 import bkey
    from netsurf.group_search import budget_counts, evaluate_slice_round, groups, protect_slice, slice_path_masks, slices
    t0 = time.time()
    m = KwsModel(a.threads)
    tf = m.tf
    require_reference_tf(tf)
    p, ri, rep = a.rate, rate_index(a.rate), a.rep
    out = a.out / f"rep{rep}_mc_{p:g}_s{a.parts}"
    out.mkdir(parents=True, exist_ok=False)
    counts = budget_counts(m.n * 8, FRACTIONS)
    target = max(counts.values())

    xc, yc, ic = m.clips("ranking", CAL_COUNT, CAL_COUNT * rep)
    assert ic[0] == CAL_COUNT * rep and len(ic) == CAL_COUNT
    log(phase="scores", rep=rep, stage="per-example gradients on the 1,024 calibration clips", done=0, total=target)
    S = scores(m.per_example(np.asarray(xc), np.asarray(yc).astype(np.int64)), m.jumps)["abs"]
    packed = np.load(PREP / "masks" / f"rep{rep}" / "masks.npz")
    stored = {f: unpack(packed[f"abs1024@{bkey(f)}"], m.n) for f in FRACTIONS}
    diff = rebuild_differences(S, stored, counts)
    (out / "scores_check.json").write_text(json.dumps(dict(rep=rep, differing_bits=diff), indent=1))
    log(stage="abs1024 rebuild", differing_bits=diff, seconds=round(time.time() - t0))
    if any(diff.values()):
        raise SystemExit(f"recomputed abs scores do not rebuild the stored abs1024 masks: {diff}")

    x, y, idx = m.clips("ranking", a.clips, CAL_COUNT * rep)
    x, y = np.asarray(x), np.asarray(y).astype(np.int64)
    labels = tf.convert_to_tensor(y)
    gs = groups(m.starts)
    sl = slices(gs, S, a.parts)
    J = m.jumps

    def run(disp):
        m.set_displacement(disp)
        logits, _ = m.comp.forward(x)
        acc = float(np.mean(logits.numpy().argmax(1) == y))
        ce = float(tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)).numpy())
        return acc, ce

    clean_acc, clean_ce = run(np.zeros(m.n))
    probe = np.random.default_rng([SEED_PARTIAL, rep, ri, 0, 0]).random((m.n, 8)) < p
    assert run(np.where(probe & ~np.ones((m.n, 8), dtype=bool), J, 0.0).sum(1)) == (clean_acc, clean_ce), \
        "full protection must give the clean result"
    log(phase="search", rep=rep, rate=p, parts=a.parts, clips=len(y), K=a.K, clean_accuracy=clean_acc, done=0,
        total=target, seconds=round(time.time() - t0))

    mask = np.zeros((m.n, 8), dtype=bool)
    accepted, rounds, passes = [], [], 0
    t_search = time.time()
    while mask.sum() < target:
        rr = len(rounds)
        if a.max_rounds and rr >= a.max_rounds:
            break
        fields = [np.random.default_rng([SEED_PARTIAL, rep, ri, rr, d]).random((m.n, 8)) < p for d in range(a.K)]

        def measure(masks):
            nonlocal passes
            A = np.empty((len(masks), a.K))
            T = np.empty_like(A)
            for j, mk in enumerate(masks):
                for d in range(a.K):
                    A[j, d], ce = run(np.where(fields[d] & ~mk, J, 0.0).sum(1))
                    T[j, d] = -ce
                    passes += 1
            return A, T

        tr = time.time()
        r = evaluate_slice_round(rr, mask, sl, measure)
        s = sl[r.accepted]
        mask = protect_slice(mask, s)
        accepted.append(r.accepted)
        best = next(c for c in r.candidates if c["uid"] == r.accepted)
        rec = asdict(r)
        rec.update(seconds=round(time.time() - tr, 1), protected=int(mask.sum()))
        rounds.append(rec)
        el = time.time() - t_search
        done = int(min(mask.sum(), target))
        log(round=rr, done=done, total=target, accepted=dict(layer=s.layer, bit=s.bit, part=s.part, size=int(s.idx.size)),
            gain=round(best["gain"], 5), se=round(best["se"], 5), candidates=len(r.candidates),
            current_accuracy=round(r.current, 4), round_seconds=rec["seconds"], seconds=round(el),
            eta_minutes=round(el / done * (target - done) / 60, 1))
        if a.max_round_seconds and rec["seconds"] > a.max_round_seconds:
            (out / "rounds.json").write_text(json.dumps(rounds))
            raise SystemExit(f"round {rr} took {rec['seconds']} s, more than the limit {a.max_round_seconds} s")
    m.set_displacement(np.zeros(m.n))
    (out / "rounds.json").write_text(json.dumps(rounds))
    meta = dict(rep=rep, rate=p, rate_index=ri, parts=a.parts, K=a.K, clips=len(y), calibration_first=int(idx[0]),
                accepted=accepted, clean_accuracy=clean_acc, passes=passes, seconds=round(time.time() - t0, 1),
                budgets=counts)
    if mask.sum() >= target:
        masks = slice_path_masks(m.n, sl, accepted, counts)
        np.savez(out / "masks.npz", **{f"mask@{bkey(f)}": pack(v) for f, v in masks.items()})
        meta["differing_from_abs1024"] = {f"{f:g}": int((v & ~stored[f]).sum()) for f, v in masks.items()}
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
    rep = a.rep
    masks = []
    for d in sorted(a.search.glob(f"rep{rep}_mc_*")):
        if not (d / "masks.npz").exists():
            continue
        meta = json.loads((d / "DONE").read_text())
        z = np.load(d / "masks.npz")
        for k in z.files:
            masks.append((f"partial_s{meta['parts']}", float(k.split("@")[1]), float(meta["rate"]), unpack(z[k], m.n)))
    if not masks:
        raise SystemExit(f"no finished search path for replication {rep} in {a.search}")
    out = a.out / f"rep{rep}" / f"field{a.field}"
    out.mkdir(parents=True, exist_ok=False)
    rp = Replayer(m)
    ref = json.loads((PREP / "engine_ref.json").read_text())
    xr, yr, ir = m.clips("evaluation", len(ref["evaluation_indices"]), ref["evaluation_indices"][0])
    assert list(ir) == list(ref["evaluation_indices"])
    acc, _, _, _ = rp.evaluate(rp.flips(ref["events"]), load_static(m, 0, 0.125)["msb"], xr, yr)
    check = dict(replay_accuracy=acc, saved_accuracy=ref["accuracy"], match=acc == ref["accuracy"])
    (out / "check.json").write_text(json.dumps(check))
    if not check["match"]:
        raise SystemExit(f"engine check failed: {check}")
    xe, ye, ie = m.clips("evaluation", WINDOW, WINDOW * rep)
    assert ie[0] == WINDOW * rep and len(ie) == WINDOW
    u = np.random.default_rng([EVAL_SEED, rep, a.field]).random((m.n, 8))
    log(phase="evaluate", rep=rep, field=a.field, masks=len(masks), done=0, total=len(masks))
    rows = []
    for i, (arm, b, p, mk) in enumerate(masks):
        acc, ce, prop, app = rp.evaluate(u < p, mk, xe, ye)
        rows.append(dict(rep=rep, field=a.field, ber=p, budget=b, arm=arm, bits=int(mk.sum()), accuracy=acc,
                         cross_entropy=ce, proposed=prop, applied=app))
        el = time.time() - t0
        log(field=a.field, done=i + 1, total=len(masks), seconds=round(el),
            eta_minutes=round(el / (i + 1) * (len(masks) - i - 1) / 60, 1))
    with open(out / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (out / "DONE").write_text(json.dumps(dict(seconds=round(time.time() - t0), evaluations=len(rows))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("search", "evaluate"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rep", type=int, required=True, choices=range(6))
    ap.add_argument("--rate", type=float, default=1e-2)
    ap.add_argument("--parts", type=int, default=4)
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--clips", type=int, default=512)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-rounds", type=int, default=0, help="stop after this many rounds (smoke tests)")
    ap.add_argument("--max-round-seconds", type=float, default=0, help="stop if one round takes longer (cost guard)")
    ap.add_argument("--field", type=int, default=0)
    ap.add_argument("--search", type=Path, default=RUNS / "paper1_partial_search_kws_v1")
    a = ap.parse_args()
    {"search": cmd_search, "evaluate": cmd_evaluate}[a.command](a)


if __name__ == "__main__":
    main()
