#!/usr/bin/env python3
"""EXPLORATORY pilot, post hoc (written after the Stage 1, Stage 4 and Stage 2 interim outcomes were read).  Not a
confirmatory stage; nothing here is a manuscript result.

Question: is the effect of independent stored-bit faults under a mask reproduced by the first two moments of every
weight's displacement, passed through the network itself?  With unprotected jumps Delta_b and rate p,

    mu_w = p * sum_b Delta_b,      V_w = p (1 - p) * sum_b Delta_b^2      (exact, weights independent)

All passes use the differentiable companion (activation rounding and clipping reproduced, integer multiplier rounding
not), with the weights displaced by a vector; no TFLite fault run is made.

  blur   per (mask, budget, rate): loss and accuracy on the evaluation window of the finished fault runs with
           mean    displacement mu                                   (the drift term of the current formula)
           gauss   mu + sqrt(V) * eps, eps ~ N(0, 1) per weight      (two moments only)
           hybrid  bits 6 and 7 as real flips, bits 0-5 by their two moments
           real    the bit flips of the finished runs' own fields (same seeds), to check the companion path against
                   the measured TFLite outcome field by field
         The same eps is used for every mask of a rate (common random numbers).
  bell   for one budget: per-input margin (true logit minus best other) over many draws of real flips and of the
         two-moment Gaussian, to compare the two distributions (new companion-side draws, evaluation inputs).
  step   single flips: the real loss change of one bit against gbar*Delta and gbar*Delta + F*Delta^2/2, sampled per
         layer and bit position on calibration inputs (the same inputs for both sides, so the difference is the
         approximation error alone, not sampling noise).

Output (write-once per job): <out>/<stage>/<model>_rep<R>/{rows.csv | margins.npz, DONE}.  Default <out>:
workspace/runs/paper1_blur_pilot_v1 in the paper worktree.  Interpreter: the TensorFlow reference environment.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from runP_phase0 import PAPER, Model
from runP_phase12 import unpack
from runR_stage1 import MASKS, NAMES, SEED as STAGE1_SEED, bkey

OUT = PAPER / "workspace/runs/paper1_blur_pilot_v1"
SEED_G, SEED_B, SEED_S = 20261201, 20261203, 20261204


def load(model_name, rep, threads):
    if model_name == "kws":
        from runR_kws import CAL_COUNT, OUT as PREP, KwsModel
        from runR_stage4_kws import EVAL_SEED, WINDOW
        m = KwsModel(threads)
        packed = np.load(PREP / "masks" / f"rep{rep}" / "masks.npz")
        xe, ye, _ = m.clips("evaluation", WINDOW, WINDOW * rep)
        cal = lambda count: m.clips("ranking", count, CAL_COUNT * rep)[:2]  # noqa: E731
        field_seed = lambda fi: [EVAL_SEED, rep, fi]  # noqa: E731
    else:
        m = Model(model_name, threads)
        packed = np.load(MASKS / model_name / f"rep{rep}" / "masks.npz")
        xe, ye, _ = m.images("evaluation", 1000, 1000 * rep)
        cal = lambda count: m.images("ranking", count, 1000 * rep)[:2]  # noqa: E731
        field_seed = lambda fi: [STAGE1_SEED, NAMES.index(model_name), rep, fi]  # noqa: E731
    assert np.all(np.abs(m.jumps[:, 7]) >= np.abs(m.jumps[:, 6])), "bit 7 is expected to be the sign bit"

    def mask(arm, frac):
        if arm == "none":
            return np.zeros((m.n, 8), dtype=bool)
        return unpack(packed[f"{arm}@{bkey(frac)}"], m.n)
    return m, mask, (np.asarray(xe), np.asarray(ye).astype(np.int64)), cal, field_seed


def run(m, disp, x, y, batch=500, margins=False):
    """Mean cross-entropy, accuracy and (optionally) per-input margins with the weights displaced by ``disp``."""
    m.set_displacement(disp)
    losses, correct, marg = [], 0, []
    for s in range(0, len(y), batch):
        logits, _ = m.comp.forward(x[s:s + batch])
        logits = logits.numpy().astype(np.float64)
        yy = y[s:s + batch]
        idx = np.arange(len(yy))
        z = logits - logits.max(1, keepdims=True)
        losses.append(-(z - np.log(np.exp(z).sum(1, keepdims=True)))[idx, yy])
        correct += int((logits.argmax(1) == yy).sum())
        if margins:
            other = logits.copy()
            other[idx, yy] = -np.inf
            marg.append(logits[idx, yy] - other.max(1))
    return float(np.concatenate(losses).mean()), correct / len(y), (np.concatenate(marg) if margins else None)


def moments(J, unprot, p):
    return p * np.where(unprot, J, 0.0).sum(1), p * (1.0 - p) * np.where(unprot, J * J, 0.0).sum(1)


def finish(out, t0, forwards, **extra):
    (out / "DONE").write_text(json.dumps(dict(seconds=round(time.time() - t0, 1), forwards=forwards,
                                              seconds_per_forward=round((time.time() - t0) / max(forwards, 1), 3), **extra)))
    print(json.dumps(dict(done=str(out), seconds=round(time.time() - t0, 1), forwards=forwards)), flush=True)


def blur(a):
    m, mask, (xe, ye), _, fseed = load(a.model, a.rep, a.threads)
    out = a.out / "blur" / f"{a.model}_rep{a.rep}"
    out.mkdir(parents=True, exist_ok=False)
    t0, J, rows, n_fw = time.time(), m.jumps, [], 0
    clean_loss, clean_acc, _ = run(m, np.zeros(m.n), xe, ye)
    U = [np.random.default_rng(fseed(fi)).random((m.n, 8)) for fi in range(min(a.draws, 8))]
    for ri, p in enumerate(a.rates):
        eps = [np.random.default_rng([SEED_G, ri, d]).standard_normal(m.n) for d in range(a.draws)]
        for frac in a.fractions:
            for arm in a.arms:
                unprot = ~mask(arm, frac)
                mu, var = moments(J, unprot, p)
                low = unprot.copy()
                low[:, 6:] = False
                mu_low, var_low = moments(J, low, p)
                todo = [("mean", 0, mu)]
                for d in range(a.draws):
                    todo.append(("gauss", d, mu + np.sqrt(var) * eps[d]))
                    if d < len(U):
                        fl = (U[d] < p) & unprot
                        top = fl.copy()
                        top[:, :6] = False
                        todo.append(("real", d, np.where(fl, J, 0.0).sum(1)))
                        todo.append(("hybrid", d, np.where(top, J, 0.0).sum(1) + mu_low + np.sqrt(var_low) * eps[d]))
                for variant, d, disp in todo:
                    loss, acc, _ = run(m, disp, xe, ye)
                    n_fw += 1
                    rows.append(dict(model=a.model, rep=a.rep, arm=arm, budget=frac, ber=p, variant=variant, draw=d,
                                     loss=loss, accuracy=acc))
        print(json.dumps(dict(rate_done=p, seconds=round(time.time() - t0))), flush=True)
    with open(out / "rows.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    finish(out, t0, n_fw, clean_loss=clean_loss, clean_accuracy=clean_acc, inputs=len(ye))


def bell(a):
    m, mask, (xe, ye), _, _ = load(a.model, a.rep, a.threads)
    out = a.out / "bell" / f"{a.model}_rep{a.rep}"
    out.mkdir(parents=True, exist_ok=False)
    t0, J, n_fw = time.time(), m.jumps, 0
    x, y = xe[:a.inputs], ye[:a.inputs]
    _, _, clean = run(m, np.zeros(m.n), x, y, margins=True)
    store = {"clean": clean}
    for ri, p in enumerate(a.rates):
        for arm in a.arms:
            unprot = ~mask(arm, a.fractions[0])
            mu, var = moments(J, unprot, p)
            for kind in ("real", "gauss"):
                M, L, A = np.zeros((a.draws, len(y))), np.zeros(a.draws), np.zeros(a.draws)
                for d in range(a.draws):
                    if kind == "real":
                        u = np.random.default_rng([SEED_B, 0, ri, d]).random((m.n, 8))
                        disp = np.where((u < p) & unprot, J, 0.0).sum(1)
                    else:
                        disp = mu + np.sqrt(var) * np.random.default_rng([SEED_B, 1, ri, d]).standard_normal(m.n)
                    L[d], A[d], M[d] = run(m, disp, x, y, margins=True)
                    n_fw += 1
                for nm, arr in (("margin", M), ("loss", L), ("accuracy", A)):
                    store[f"{arm}|{p:g}|{kind}|{nm}"] = arr
            print(json.dumps(dict(cell_done=[arm, p], seconds=round(time.time() - t0))), flush=True)
    np.savez_compressed(out / "margins.npz", **store)
    finish(out, t0, n_fw, budget=a.fractions[0], inputs=len(y), draws=a.draws)


def step(a):
    m, _, _, cal, _ = load(a.model, a.rep, a.threads)
    out = a.out / "step" / f"{a.model}_rep{a.rep}"
    out.mkdir(parents=True, exist_ok=False)
    t0, J, rows, n_fw = time.time(), m.jumps, [], 0
    x, y = cal(a.inputs)
    x, y = np.asarray(x), np.asarray(y).astype(np.int64)
    G = m.per_example(x, y)
    gbar, F = G.mean(0), (G * G).mean(0)
    del G
    clean_loss, clean_acc, _ = run(m, np.zeros(m.n), x, y, batch=len(y))
    rng = np.random.default_rng([SEED_S, a.rep])
    for li, name in enumerate(m.names):
        lo, hi = int(m.starts[li]), int(m.starts[li + 1])
        for b in range(8):
            for w in rng.choice(np.arange(lo, hi), size=min(a.per_cell, hi - lo), replace=False):
                d = np.zeros(m.n)
                d[w] = J[w, b]
                loss, acc, _ = run(m, d, x, y, batch=len(y))
                n_fw += 1
                lin = gbar[w] * J[w, b]
                rows.append(dict(model=a.model, rep=a.rep, layer=name, layer_index=li, weight=int(w), bit=b,
                                 jump=J[w, b], gbar=gbar[w], fisher=F[w], real=loss - clean_loss, linear=lin,
                                 quadratic=lin + 0.5 * F[w] * J[w, b] ** 2, accuracy_change=acc - clean_acc))
        print(json.dumps(dict(layer_done=name, seconds=round(time.time() - t0))), flush=True)
    with open(out / "rows.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    finish(out, t0, n_fw, clean_loss=clean_loss, clean_accuracy=clean_acc, inputs=len(y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("blur", "bell", "step"))
    ap.add_argument("--model", required=True, choices=["kws"] + NAMES)
    ap.add_argument("--rep", type=int, required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--rates", type=float, nargs="+", default=[3e-4, 1e-3, 3e-3, 5e-3, 1e-2, 3e-2])
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.125, 0.25])
    ap.add_argument("--arms", nargs="+", default=["abs1024", "pos1024"])
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--inputs", type=int, default=400)
    ap.add_argument("--per-cell", type=int, default=10)
    a = ap.parse_args()
    {"blur": blur, "bell": bell, "step": step}[a.stage](a)


if __name__ == "__main__":
    main()
