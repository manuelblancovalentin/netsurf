#!/usr/bin/env python3
"""Partial-group protection search on the DS-CNN (plan review/PARTIAL_GROUP_SEARCH_PLAN_2026-11-14.md).

Each layer x bit group is split into --parts slices in abs1024 order (highest per-weight score first); every round,
each group's next slice is a candidate, judged by real-flip network passes on K shared fault fields at the target rate
(paired gain in accuracy per protected bit, ties by cross-entropy then slice id).  The search therefore chooses only
how much of each group to protect; within a group the order is abs1024's.

  scores    --rep R   CPU only.  The abs scores recomputed from the replication's 1,024 calibration clips; they must
            rebuild its stored abs1024 masks exactly (the stored masks were built on CPU; on the GPU, summation order
            moves 85-100 near-tied bits).  <out>/rep<R>/abs.npy, check.json.
  search    --rep R --rate P --parts S   One path on the first --clips calibration clips of replication R, with the
            scores of `scores` (checked again).  Seeds [20261215, R, rate index, round, draw].  Output
            <out>/rep<R>_mc_<P>_s<S>/: rounds.json, masks.npz (mask@<budget>), scores_check.json, DONE.
  swap      --rep R --rate P --budget F   Swap search from abs1024 at one budget (pre-registration
            review/PARTIAL_SEARCH_FOLLOWUP_PREREG_2026-11-14.md): protected bits stay a prefix of each group's abs order;
            a move drops the last --move protected bits of one group and protects the next --move of another.  Each
            round proposes the best pair from one-group changes on --K fields (seeds [20261217, R, rate index, budget
            index, round, 0, draw]) and verifies it against the current mask on --K-verify fresh fields ([..., round, 1,
            draw]); accepted if the paired mean gain is positive and above 2 standard errors.  Stops after
            --stop-after rejections in a row or --max-rounds rounds.  <out>/rep<R>_mc_<P>_swap_b<F>/.
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
SEED_SWAP = 20261217


def rebuild_differences(score, stored: dict, counts: dict) -> dict:
    """For each budget fraction, the bits of the stored top-budget mask that the recomputed score would not select
    (0 everywhere = exact rebuild)."""
    from netsurf.mean_variance import top_budget
    return {f"{f:g}": int((stored[f] & ~top_budget(score, counts[f])).sum()) for f in stored}


def stored_abs1024(n, rep):
    from runP_phase12 import unpack
    from runR_kws import OUT as PREP
    from runR_stage1 import bkey
    packed = np.load(PREP / "masks" / f"rep{rep}" / "masks.npz")
    return {f: unpack(packed[f"abs1024@{bkey(f)}"], n) for f in FRACTIONS}


def cmd_scores(a):
    import os
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        raise SystemExit("scores must run on CPU: set CUDA_VISIBLE_DEVICES=\"\" (the stored abs1024 masks were built on CPU)")
    from runQ_calibration import scores
    from runR_kws import CAL_COUNT, KwsModel
    from netsurf.group_search import budget_counts
    t0 = time.time()
    m = KwsModel(a.threads)
    require_reference_tf(m.tf)
    rep = a.rep
    out = a.out / f"rep{rep}"
    out.mkdir(parents=True, exist_ok=False)
    xc, yc, ic = m.clips("ranking", CAL_COUNT, CAL_COUNT * rep)
    assert ic[0] == CAL_COUNT * rep and len(ic) == CAL_COUNT
    log(phase="scores", rep=rep, stage="per-example gradients on the 1,024 calibration clips", done=0, total=1)
    S = scores(m.per_example(np.asarray(xc), np.asarray(yc).astype(np.int64)), m.jumps)["abs"]
    diff = rebuild_differences(S, stored_abs1024(m.n, rep), budget_counts(m.n * 8, FRACTIONS))
    (out / "check.json").write_text(json.dumps(dict(rep=rep, threads=a.threads, differing_bits=diff), indent=1))
    log(stage="abs1024 rebuild", rep=rep, differing_bits=diff, done=1, total=1, seconds=round(time.time() - t0))
    if any(diff.values()):
        raise SystemExit(f"recomputed abs scores do not rebuild the stored abs1024 masks: {diff}")
    np.save(out / "abs.npy", S)


def cmd_search(a):
    from runP_phase12 import pack
    from runR_kws import CAL_COUNT, KwsModel
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

    S = np.load(a.scores / f"rep{rep}" / "abs.npy")
    stored = stored_abs1024(m.n, rep)
    diff = rebuild_differences(S, stored, counts)
    (out / "scores_check.json").write_text(json.dumps(dict(rep=rep, scores=str(a.scores), differing_bits=diff), indent=1))
    log(stage="abs1024 rebuild", differing_bits=diff, seconds=round(time.time() - t0))
    if any(diff.values()):
        raise SystemExit(f"the scores do not rebuild the stored abs1024 masks: {diff}")

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


def cmd_swap(a):
    from runP_phase12 import pack
    from runR_kws import CAL_COUNT, KwsModel
    from runR_stage1 import bkey
    from netsurf.group_search import (apply_move, budget_counts, group_orders, groups, prefix_counts, prefix_mask,
                                      propose_swap, swap_moves)
    t0 = time.time()
    m = KwsModel(a.threads)
    tf = m.tf
    require_reference_tf(tf)
    p, ri, rep, b = a.rate, rate_index(a.rate), a.rep, a.budget
    bi = FRACTIONS.index(b)
    out = a.out / f"rep{rep}_mc_{p:g}_swap_b{b:g}"
    out.mkdir(parents=True, exist_ok=False)
    counts_b = budget_counts(m.n * 8, FRACTIONS)
    S = np.load(a.scores / f"rep{rep}" / "abs.npy")
    stored = stored_abs1024(m.n, rep)
    diff = rebuild_differences(S, stored, counts_b)
    (out / "scores_check.json").write_text(json.dumps(dict(rep=rep, scores=str(a.scores), differing_bits=diff), indent=1))
    if any(diff.values()):
        raise SystemExit(f"the scores do not rebuild the stored abs1024 masks: {diff}")
    gs = groups(m.starts)
    orders = group_orders(gs, S)
    sizes = {g.gid: g.hi - g.lo for g in gs}
    start = prefix_counts(stored[b], gs, orders)
    assert (prefix_mask(m.n, gs, orders, start) == stored[b]).all() and sum(start.values()) == counts_b[b]

    x, y, idx = m.clips("ranking", a.clips, CAL_COUNT * rep)
    x, y = np.asarray(x), np.asarray(y).astype(np.int64)
    J = m.jumps
    passes = 0

    def acc_of(mk, fields):
        nonlocal passes
        r = np.empty(len(fields))
        for d, u in enumerate(fields):
            m.set_displacement(np.where(u & ~mk, J, 0.0).sum(1))
            logits, _ = m.comp.forward(x)
            r[d] = float(np.mean(logits.numpy().argmax(1) == y))
            passes += 1
        return r

    def draw(rr, stage, K):
        return [np.random.default_rng([SEED_SWAP, rep, ri, bi, rr, stage, d]).random((m.n, 8)) < p for d in range(K)]

    log(phase="swap", rep=rep, rate=p, budget=b, move=a.move, K=a.K, K_verify=a.K_verify, clips=len(y), done=0,
        total=a.max_rounds, seconds=round(time.time() - t0))
    counts, rounds, rejections, t_search = dict(start), [], 0, time.time()
    for rr in range(a.max_rounds):
        tr = time.time()
        fields = draw(rr, 0, a.K)
        adds, removes = swap_moves(counts, sizes, a.move)
        base = acc_of(prefix_mask(m.n, gs, orders, counts), fields)
        gain, loss, se = {}, {}, {}
        for g in adds:
            d = acc_of(prefix_mask(m.n, gs, orders, apply_move(counts, sizes, a.move, add=g)), fields) - base
            gain[g], se[("add", g)] = float(d.mean()), float(d.std(ddof=1) / np.sqrt(a.K))
        for g in removes:
            d = base - acc_of(prefix_mask(m.n, gs, orders, apply_move(counts, sizes, a.move, remove=g)), fields)
            loss[g], se[("remove", g)] = float(d.mean()), float(d.std(ddof=1) / np.sqrt(a.K))
        prop = propose_swap(gain, loss)
        rec = dict(index=rr, current=float(base.mean()), adds=len(adds), removes=len(removes))
        if prop is None:
            rec.update(stopped="no swap possible")
            rounds.append(rec)
            break
        o, i, predicted = prop
        new = apply_move(apply_move(counts, sizes, a.move, remove=o), sizes, a.move, add=i)
        vf = draw(rr, 1, a.K_verify)
        dv = acc_of(prefix_mask(m.n, gs, orders, new), vf) - acc_of(prefix_mask(m.n, gs, orders, counts), vf)
        vm, vs = float(dv.mean()), float(dv.std(ddof=1) / np.sqrt(a.K_verify))
        accepted = vm > 0 and vm > 2 * vs
        rec.update(out=o, into=i, predicted=predicted, gain_in=gain[i], se_in=se[("add", i)], loss_out=loss[o],
                   se_out=se[("remove", o)], verify_mean=vm, verify_se=vs, accepted=bool(accepted),
                   seconds=round(time.time() - tr, 1))
        rounds.append(rec)
        if accepted:
            counts, rejections = new, 0
        else:
            rejections += 1
        el = time.time() - t_search
        log(round=rr, done=rr + 1, total=a.max_rounds, out=o, into=i, predicted=round(predicted, 5),
            verify=round(vm, 5), verify_se=round(vs, 5), accepted=bool(accepted), rejections_in_a_row=rejections,
            round_seconds=rec["seconds"], seconds=round(el), eta_minutes=round(el / (rr + 1) * (a.max_rounds - rr - 1) / 60, 1))
        if a.max_round_seconds and rec["seconds"] > a.max_round_seconds:
            (out / "rounds.json").write_text(json.dumps(rounds))
            raise SystemExit(f"round {rr} took {rec['seconds']} s, more than the limit {a.max_round_seconds} s")
        if rejections >= a.stop_after:
            break
    m.set_displacement(np.zeros(m.n))
    final = prefix_mask(m.n, gs, orders, counts)
    assert int(final.sum()) == counts_b[b]
    (out / "rounds.json").write_text(json.dumps(rounds))
    np.savez(out / "masks.npz", **{f"mask@{bkey(b)}": pack(final)})
    moved = {str(g): counts[g] - start[g] for g in counts if counts[g] != start[g]}
    (out / "DONE").write_text(json.dumps(dict(
        rep=rep, rate=p, rate_index=ri, budget=b, budget_index=bi, parts="swap", move=a.move, K=a.K,
        K_verify=a.K_verify, clips=len(y), calibration_first=int(idx[0]), rounds=len(rounds),
        accepted=int(sum(r.get("accepted", False) for r in rounds)), moved_bits_by_group=moved,
        differing_from_abs1024=int((final & ~stored[b]).sum()), passes=passes, seconds=round(time.time() - t0, 1)),
        indent=1))
    log(stage="swap done", rounds=len(rounds), accepted=int(sum(r.get("accepted", False) for r in rounds)),
        passes=passes, seconds=round(time.time() - t0), seconds_per_pass=round((time.time() - t_search) / max(passes, 1), 4))


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
            arm = a.arm or f"partial_s{meta['parts']}"
            masks.append((arm, float(k.split("@")[1]), float(meta["rate"]), unpack(z[k], m.n)))
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
    ap.add_argument("command", choices=("scores", "search", "swap", "evaluate"))
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
    ap.add_argument("--budget", type=float, default=0.125, choices=FRACTIONS)
    ap.add_argument("--move", type=int, default=256)
    ap.add_argument("--K-verify", type=int, default=128)
    ap.add_argument("--stop-after", type=int, default=3, help="swap: stop after this many rejections in a row")
    ap.add_argument("--arm", default="", help="evaluate: arm label (default partial_s<parts>)")
    ap.add_argument("--search", type=Path, default=RUNS / "paper1_partial_search_kws_v1")
    ap.add_argument("--scores", type=Path, default=RUNS / "paper1_partial_search_scores_v1")
    a = ap.parse_args()
    if a.command == "swap" and not a.max_rounds:
        a.max_rounds = 40
    {"scores": cmd_scores, "search": cmd_search, "swap": cmd_swap, "evaluate": cmd_evaluate}[a.command](a)


if __name__ == "__main__":
    main()
