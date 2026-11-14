#!/usr/bin/env python3
"""EXPLORATORY: what the measured DS-CNN top-bit flips predict (plan review/TOPBIT_ANALYSIS_PLAN_2026-11-14.md).

Commands (each write-once into its own output folder, unbuffered progress lines):

  cells     kona.  The Stage 4 masks of replication 0 (abs1024, pos1024, abs64, significance-first, FKeras at the four
            budgets; the analytic mask built for each rate) as one packed file for the Quest background job:
            <out>/cells.npz + cells.json (288 cells = 6 arms x 4 budgets x 12 rates, hash-checked masks).
  masks     kona.  Phase 3 masks from the calibration clips only: the 44,032 top bits ordered by measured single-flip
            damage on the 512 calibration clips ("measured_loss": mean loss increase; "measured_acc": clips turned
            wrong minus clips turned right, then loss), then bit 5 in abs1024 order; floor(f x 176,128) bits per
            budget.  Also Q4: rank agreement between the gradient scores (abs, pos, recomputed on the 1,024 calibration
            clips and checked against the stored abs1024 masks) and the measured damage, network-wide and within
            layers.  <out>/masks.npz, masks.json, q4.json.
  evaluate  CPU, one job per field.  The Phase 3 masks in the Stage 4 protocol: engine check, test window 0, field
            seeds [20261119, 0, f], all 12 rates, TFLite replay.  <out>/field<f>/results.csv.
  analyze   kona.  Q1 and Q2 from the Quest outputs (codes of every top-bit flip; background passes) against the
            Stage 4 TFLite accuracy of the same cells, fields and clips.  <out>/fields.csv, cells.csv, summary.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from site_paths import WT  # noqa: E402

RUNS = WT / "paper1-tecs-revision-v1/workspace/runs"
FLIPS_V1 = RUNS / "paper1_topbit_flips_kws_v1"
STAGE4_MASKS = RUNS / "paper1_runR_stage4_kws_masks_v1"
STAGE4_EVAL = RUNS / "paper1_runR_stage4_kws_eval_v1"
REP, N_EVAL, N_CAL = 0, 800, 512
ARMS = ("abs1024", "pos1024", "abs64", "msb", "fkeras", "ra_analytic")
TOP = (6, 7)


def log(**kw):
    print(json.dumps(dict(t=time.strftime("%H:%M:%S"), **kw)), flush=True)


def write_once(path: Path, text: str):
    if path.exists():
        raise SystemExit(f"{path} exists (write-once)")
    path.write_text(text)


# ---------------------------------------------------------------------------------------------------------- cells
def cmd_cells(a):
    from runP_phase12 import pack, unpack
    from runR_kws import KwsModel, STAGE4_FRACTIONS
    from runR_stage1 import RATES, bkey
    from runR_stage4_kws import load_static
    from netsurf.mean_variance import mask_sha256
    m = KwsModel(a.threads)
    a.out.mkdir(parents=True, exist_ok=False)
    arrays, cells, hashes = {}, [], {}
    for f in STAGE4_FRACTIONS:
        static = load_static(m, REP, f)
        an = np.load(STAGE4_MASKS / f"rep{REP}" / f"b{bkey(f)}" / "analytic.npz")
        for arm in ARMS:
            for p in RATES:
                mk = static[arm] if arm != "ra_analytic" else unpack(an[f"{p:g}"], m.n)
                key = f"{arm}@{bkey(f)}" if arm != "ra_analytic" else f"{arm}@{bkey(f)}@{p:g}"
                if key not in arrays:
                    arrays[key] = pack(mk)
                    hashes[key] = mask_sha256(mk)
                cells.append(dict(key=f"{arm}@{bkey(f)}@{p:g}", arm=arm, budget=f, rate=p, group=f"{arm}@{bkey(f)}",
                                  mask_key=key, bits=int(mk.sum())))
    np.savez(a.out / "cells.npz", **arrays)
    write_once(a.out / "cells.json", json.dumps(dict(rep=REP, n=m.n, cells=cells, masks=hashes), indent=0))
    log(stage="cells written", cells=len(cells), masks=len(arrays))


# ---------------------------------------------------------------------------------------------------------- masks
def load_flips_v1():
    """Measured damage of every top-bit flip on the calibration clips (v1 margins run, locked)."""
    clean = np.load(FLIPS_V1 / "clean.npz")
    c_loss, c_corr = float(clean["loss_cal"]), clean["correct"][N_EVAL:].astype(np.int64)
    W, B, L, D = [], [], [], []
    for f in sorted(FLIPS_V1.glob("layer_*.npz")):
        z = np.load(f)
        W.append(z["weight"]); B.append(z["bit"]); L.append(z["loss_cal"] - c_loss)
        cor = z["correct"][:, N_EVAL:].astype(np.int64)
        D.append(((c_corr[None] == 1) & (cor == 0)).sum(1) - ((c_corr[None] == 0) & (cor == 1)).sum(1))
    return np.concatenate(W), np.concatenate(B), np.concatenate(L), np.concatenate(D)


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x, kind="stable")), np.argsort(np.argsort(y, kind="stable"))
    return float(np.corrcoef(rx, ry)[0, 1])


def cmd_masks(a):
    from runP_phase12 import pack, unpack
    from runQ_calibration import scores
    from runR_kws import CAL_COUNT, OUT as PREP, KwsModel, STAGE4_FRACTIONS
    from runR_stage1 import bkey
    from netsurf.mean_variance import mask_sha256, top_budget
    a.out.mkdir(parents=True, exist_ok=False)
    m = KwsModel(a.threads)
    t0 = time.time()
    x, y, idx = m.clips("ranking", CAL_COUNT, CAL_COUNT * REP)
    S = scores(m.per_example(np.asarray(x), np.asarray(y).astype(np.int64)), m.jumps)
    log(stage="gradient scores", seconds=round(time.time() - t0))
    total = m.n * 8
    stored = np.load(PREP / "masks" / f"rep{REP}" / "masks.npz")
    np.savez(a.out / "scores.npz", **{k: S[k] for k in ("abs", "pos")})
    rebuild = {}                                                # recomputed scores against the stored abs1024 masks
    for f in STAGE4_FRACTIONS:
        k = f"abs1024@{bkey(f)}"
        if k in stored.files:
            mine, ref = top_budget(S["abs"], int(round(f * total))), unpack(stored[k], m.n)
            rebuild[k] = dict(bits=int(ref.sum()), differing=int((mine & ~ref).sum()))
    log(stage="abs1024 rebuild", **{k: v["differing"] for k, v in rebuild.items()})
    W, B, L, D = load_flips_v1()
    order = {"measured_loss": np.lexsort((B, W, -L)), "measured_acc": np.lexsort((B, W, -L, -D))}
    s5 = np.full(m.n * 8, -np.inf)
    s5.reshape(m.n, 8)[:, 5] = S["abs"][:, 5]
    order5 = np.lexsort((np.arange(s5.size), -s5))[: m.n]       # bit 5 by abs1024 score, storage order on ties
    arrays, info = {}, {}
    for name, o in order.items():
        for f in STAGE4_FRACTIONS:
            count = int(np.floor(f * total))
            mk = np.zeros((m.n, 8), dtype=bool)
            k = min(count, len(o))
            mk[W[o[:k]], B[o[:k]]] = True
            rest = count - k
            if rest:
                flat = mk.reshape(-1)
                flat[order5[:rest]] = True
            assert mk.sum() == count
            key = f"{name}@{bkey(f)}"
            arrays[key] = pack(mk)
            info[key] = dict(count=count, sha256=mask_sha256(mk), top_bits=int(mk[:, 6:].sum()))
    np.savez(a.out / "masks.npz", **arrays)
    write_once(a.out / "masks.json", json.dumps(dict(masks=info, calibration_first=idx[0], damage_clips=N_CAL,
                                                     abs1024_rebuild=rebuild), indent=1))
    # Q4
    layer = np.searchsorted(m.starts, W, side="right") - 1
    q4 = {}
    for sname in ("abs", "pos"):
        sc = S[sname][W, B]
        res = dict(all=dict(loss=spearman(sc, L), acc=spearman(sc, D)))
        for b in TOP:
            sel = B == b
            within = [spearman(sc[sel & (layer == li)], L[sel & (layer == li)]) for li in range(len(m.names))]
            res[f"bit{b}"] = dict(network_wide=spearman(sc[sel], L[sel]), within_layer=dict(zip(m.names, within)),
                                  within_layer_median=float(np.median(within)))
        half = len(L) // 2                                       # overlap of the top halves (12.5% budget = half)
        res["top_half_overlap"] = float(len(set(np.argsort(-sc)[:half]) & set(np.argsort(-L)[:half])) / half)
        q4[sname] = res
    layer_means = {n: dict(mean_loss_damage=float(L[layer == i].mean()), mean_abs_score=float(S["abs"][W, B][layer == i].mean()))
                   for i, n in enumerate(m.names)}
    write_once(a.out / "q4.json", json.dumps(dict(scores=q4, layers=layer_means), indent=1))
    log(stage="masks done", seconds=round(time.time() - t0), q4={k: v["all"] for k, v in q4.items()})


# ---------------------------------------------------------------------------------------------------------- evaluate
def cmd_evaluate(a):
    from runP_phase12 import Replayer, unpack
    from runR_kws import OUT as PREP, KwsModel
    from runR_stage1 import RATES
    from runR_stage4_kws import EVAL_SEED, WINDOW, load_static
    m = KwsModel(a.threads)
    t0 = time.time()
    packed = np.load(a.masks / "masks.npz")
    masks = {k: unpack(packed[k], m.n) for k in packed.files}
    out = a.out / f"field{a.field}"
    out.mkdir(parents=True, exist_ok=False)
    rp = Replayer(m)
    ref = json.loads((PREP / "engine_ref.json").read_text())
    xr, yr, ir = m.clips("evaluation", len(ref["evaluation_indices"]), ref["evaluation_indices"][0])
    assert list(ir) == list(ref["evaluation_indices"])
    acc, _, _, _ = rp.evaluate(rp.flips(ref["events"]), load_static(m, REP, 0.125)["msb"], xr, yr)  # as Stage 4
    check = dict(replay_accuracy=acc, saved_accuracy=ref["accuracy"], match=acc == ref["accuracy"])
    (out / "check.json").write_text(json.dumps(check))
    if not check["match"]:
        raise SystemExit(f"engine check failed: {check}")
    xe, ye, ie = m.clips("evaluation", WINDOW, WINDOW * REP)
    assert ie[0] == WINDOW * REP and len(ie) == WINDOW
    u = np.random.default_rng([EVAL_SEED, REP, a.field]).random((m.n, 8))
    rows = []
    for p in (a.rates or RATES):
        fl = u < p
        for k, mk in masks.items():
            acc, ce, prop, app = rp.evaluate(fl, mk, xe, ye)
            arm, b = k.split("@")
            rows.append(dict(rep=REP, field=a.field, ber=p, budget=float(b), arm=arm, accuracy=acc, cross_entropy=ce,
                             proposed=prop, applied=app))
        log(field=a.field, rate_done=p, seconds=round(time.time() - t0))
    with open(out / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (out / "DONE").write_text(json.dumps(dict(seconds=round(time.time() - t0), evaluations=len(rows))))


# ---------------------------------------------------------------------------------------------------------- analyze
_A = None


def _init(path):
    global _A
    _A = np.load(path, mmap_mode="r")


def _fourier_cell(args):
    from netsurf.event_readout import lattice_correct_probability
    rows, p, m0, thr, bmu, bvar = args
    probs, dropped = np.empty(len(m0)), 0.0
    for i in range(len(m0)):
        probs[i], d = lattice_correct_probability(int(m0[i]), _A[i, rows], p, int(thr[i]), float(bmu[i]), float(bvar[i]))
        dropped = max(dropped, d)
    return float(probs.mean()), dropped


def stage4_tflite():
    out = {}
    for f in range(8):
        for r in csv.DictReader(open(STAGE4_EVAL / f"rep{REP}" / f"field{f}" / "results.csv")):
            out[(r["arm"], float(r["budget"]), float(r["ber"]), f)] = float(r["accuracy"])
    return out


def cmd_analyze(a):
    from runP_phase12 import unpack
    from netsurf.event_readout import additive_scores, correct, gaussian_correct_probability, runner_up, tie_threshold
    from runR_stage4_kws import EVAL_SEED
    t0 = time.time()
    a.out.mkdir(parents=True, exist_ok=False)
    spec = json.loads((a.cells / "cells.json").read_text())
    packed = np.load(a.cells / "cells.npz")
    n = spec["n"]
    # event codes of every top-bit flip on the test clips
    clean_z = np.load(a.logits / "clean.npz")
    clean, labels = clean_z["codes"][:N_EVAL].astype(np.int32), clean_z["labels"][:N_EVAL]
    W, B, Q = [], [], []
    for f in sorted(a.logits.glob("layer_*.npz")):
        z = np.load(f)
        W.append(z["weight"]); B.append(z["bit"]); Q.append(z["codes"][:, :N_EVAL])
    W, B, Q = np.concatenate(W), np.concatenate(B), np.concatenate(Q)
    row = np.full((n, 8), -1, dtype=np.int64)
    row[W, B] = np.arange(len(W))
    assert (row[:, 6:] >= 0).all(), "every top bit must have a measured flip"
    log(stage="codes loaded", flips=len(W), seconds=round(time.time() - t0))
    k = runner_up(clean, labels)
    ii = np.arange(N_EVAL)
    m0 = clean[ii, labels] - clean[ii, k]
    thr = tie_threshold(labels, k)
    assert np.array_equal(m0 >= thr, correct(clean, labels))
    A = (Q[:, ii, labels].astype(np.int32) - Q[:, ii, k]) - m0[None]          # margin effect of each flip, per clip
    a_path = a.out / "margin_effects.tmp.npy"
    np.save(a_path, np.ascontiguousarray(A.T.astype(np.int16)))               # clips x flips, for the workers
    del A
    bg_clean = np.load(a.background / "clean.npz")["codes"][:N_EVAL].astype(np.int32)
    assert np.array_equal(bg_clean, clean), "background clean codes differ from the logits run"
    groups = {g.stem: np.load(g) for g in a.background.glob("*@*.npz")}
    tfl = stage4_tflite()
    fields = [np.random.default_rng([EVAL_SEED, REP, f]).random((n, 8))[:, 6:] for f in range(8)]
    field_rows, cell_rows, jobs = [], [], []
    cells = spec["cells"][: a.max_cells or None]
    for ci, c in enumerate(cells):
        g = groups[c["group"]]
        j = int(np.nonzero(g["cells"] == ci)[0][0])
        unprot_top = ~unpack(packed[c["mask_key"]], n)[:, 6:]
        p = c["rate"]
        gauss = g["gauss"][j][:, :N_EVAL].astype(np.int32)
        bm = (gauss[:, ii, labels] - gauss[:, ii, k]) - m0[None]
        bmu, bvar = bm.mean(0), bm.var(0, ddof=1)
        per = dict(E=[], EBreal=[], EBgauss=[], companion=[], tflite=[], top_applied=[])
        for f in range(8):
            applied = fields[f] < p
            applied &= unprot_top
            rows_f = row[:, 6:][applied]
            ev = Q[rows_f]
            low = g["low"][j][f][:N_EVAL].astype(np.int32)
            s_ev = clean + (ev.astype(np.int32) - clean[None]).sum(0)          # = additive_scores before clipping
            clip = lambda s: np.clip(s, -128, 127)  # noqa: E731
            e = correct(clip(s_ev), labels).mean()
            ebr = correct(clip(s_ev + low - clean), labels).mean()
            ebg = np.mean([correct(clip(s_ev + gd - clean), labels).mean() for gd in gauss])
            if ci == 0 and f == 0:                                          # the shortcut equals the tested function
                assert np.array_equal(clip(s_ev + low - clean), additive_scores(clean, ev, low))
            comp = correct(g["full"][j][f][:N_EVAL], labels).mean()
            t = tfl[(c["arm"], c["budget"], p, f)]
            field_rows.append(dict(cell=c["key"], arm=c["arm"], budget=c["budget"], rate=p, field=f, top_applied=len(rows_f),
                                   tflite=t, companion=comp, E=e, EBreal=ebr, EBgauss=ebg))
            for key, v in (("E", e), ("EBreal", ebr), ("EBgauss", ebg), ("companion", comp), ("tflite", t), ("top_applied", len(rows_f))):
                per[key].append(v)
        rows_c = row[:, 6:][unprot_top]
        G = float(gaussian_correct_probability(m0, (np.load(a_path, mmap_mode="r")[:, rows_c]), p, thr, bmu, bvar).mean())
        cell_rows.append(dict(cell=c["key"], arm=c["arm"], budget=c["budget"], rate=p, unprotected_top=int(len(rows_c)),
                              **{f"{k_}_mean": float(np.mean(v)) for k_, v in per.items()}, gaussian=G))
        jobs.append((np.sort(rows_c), p, m0, thr, bmu, bvar))
        if (ci + 1) % 24 == 0:
            log(stage="field-level predictions", cells_done=ci + 1, total=len(cells), seconds=round(time.time() - t0))
    log(stage="lattice readout start", cells=len(jobs), workers=a.workers)
    with Pool(a.workers, initializer=_init, initargs=(str(a_path),)) as pool:
        for ci, (prob, dropped) in enumerate(pool.imap(_fourier_cell, jobs)):
            cell_rows[ci]["fourier"], cell_rows[ci]["fourier_dropped_mass"] = prob, dropped
            if (ci + 1) % 12 == 0:
                log(stage="lattice readout", cells_done=ci + 1, total=len(jobs), seconds=round(time.time() - t0))
    a_path.unlink()
    for name, rows in (("fields.csv", field_rows), ("cells.csv", cell_rows)):
        with open(a.out / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    write_once(a.out / "summary.json", json.dumps(summarize(field_rows, cell_rows), indent=1))
    log(stage="analyze done", seconds=round(time.time() - t0))


def summarize(field_rows, cell_rows):
    def mae(rows, pred, ref):
        d = np.array([r[pred] - r[ref] for r in rows]) * 100
        return dict(mae_points=float(np.abs(d).mean()), mean_points=float(d.mean()), worst_points=float(d[np.argmax(np.abs(d))]))
    rates = sorted({r["rate"] for r in cell_rows})
    out = dict(status="EXPLORATORY", q1_fields={}, q1_cells={}, q2={})
    for pred in ("E", "EBreal", "EBgauss", "companion"):
        out["q1_fields"][pred] = dict(vs_tflite=mae(field_rows, pred, "tflite"),
                                      vs_companion=mae(field_rows, pred, "companion") if pred != "companion" else None,
                                      by_rate={f"{p:g}": mae([r for r in field_rows if r["rate"] == p], pred, "tflite") for p in rates})
    for pred in ("fourier", "gaussian", "EBgauss_mean", "companion_mean"):
        out["q1_cells"][pred] = dict(vs_tflite=mae(cell_rows, pred, "tflite_mean"),
                                     by_rate={f"{p:g}": mae([r for r in cell_rows if r["rate"] == p], pred, "tflite_mean") for p in rates})
    budgets = sorted({r["budget"] for r in cell_rows})
    for pred in ("fourier", "gaussian", "EBgauss_mean"):
        reg = {}
        for b in budgets:
            for p in rates:
                cs = [r for r in cell_rows if r["budget"] == b and r["rate"] == p]
                best = max(r["tflite_mean"] for r in cs)
                chosen = max(cs, key=lambda r: r[pred])
                fixed = next(r for r in cs if r["arm"] == "abs1024")
                reg[f"{b:g}@{p:g}"] = dict(chosen=chosen["arm"], best=max(cs, key=lambda r: r["tflite_mean"])["arm"],
                                           regret_points=100 * (best - chosen["tflite_mean"]),
                                           abs1024_regret_points=100 * (best - fixed["tflite_mean"]))
        r_ = np.array([v["regret_points"] for v in reg.values()])
        out["q2"][pred] = dict(cells=reg, mean_regret=float(r_.mean()), worst_regret=float(r_.max()),
                               picked_best=float(np.mean([v["chosen"] == v["best"] for v in reg.values()])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("cells", "masks", "evaluate", "analyze"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--field", type=int, default=0)
    ap.add_argument("--masks", type=Path, default=RUNS / "paper1_topbit_masks_kws_v1")
    ap.add_argument("--cells", type=Path, default=RUNS / "paper1_topbit_cells_kws_v1")
    ap.add_argument("--logits", type=Path, default=RUNS / "paper1_topbit_logits_kws_v1")
    ap.add_argument("--background", type=Path, default=RUNS / "paper1_topbit_background_kws_v1")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--rates", type=float, nargs="+", default=None, help="evaluate: subset of rates (smoke tests)")
    ap.add_argument("--max-cells", type=int, default=0, help="analyze: first N cells only (smoke tests)")
    a = ap.parse_args()
    {"cells": cmd_cells, "masks": cmd_masks, "evaluate": cmd_evaluate, "analyze": cmd_analyze}[a.command](a)


if __name__ == "__main__":
    main()
