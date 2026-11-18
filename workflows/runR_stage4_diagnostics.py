#!/usr/bin/env python3
"""Run R, post-hoc diagnostics of the Stage 4 (DS-CNN) failure.  Exploratory: written after the Stage 4 outcomes were
read; no new fault field is drawn.  Reads only finished Stage 1 (ResNet-8) and Stage 4 (DS-CNN) outputs, the frozen
predictions, the calibration statistics saved by the mask builds, and the R-A5 selection scores.  No Stage 2 outcome
is read.

H1, loss versus accuracy.  The formula predicts expected loss.  The fault runs recorded mean cross-entropy next to
accuracy, so the measured loss difference pos1024 - abs1024 can be compared with the predicted one directly.  If the
measured loss favours the positive mask where predicted while accuracy does not, the failure is in the step from loss
to accuracy; if the measured loss also favours the absolute mask, the loss prediction itself is wrong.

H2, close to a minimum.  The positive mask is built from the mean gradient of 1,024 calibration inputs.  Measures:
(a) how much of the mean gradient is real: sum gbar^2 against its noise sum se^2, and the correlation of gbar
between independent calibration draws; (b) the first-order advantage of pos over abs per unit rate,
gbar . (mu_pos - mu_abs) / p, with the gradient of the mask's own draw (in sample) and with the mean gradient of the
other draws (out of sample); (c) the predicted pos - abs objective on the six other held-out sets of R-A5; (d) the measured loss increase of each
mask against its predicted objective, and the clean loss on calibration against test inputs; (e) across the nine
ResNet-8 checkpoints, rank correlation of (a), (b), (d) with the Stage 1 switch bias.

  python runR_stage4_diagnostics.py      (TensorFlow 2.14.1 reference environment; ~1 min, 1 process)
Output (write-once): workspace/runs/paper1_runR_stage4_diagnostics_v1/{summary.json, cells.csv}.
"""
from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from runP_phase0 import PAPER, sha256_file
from runP_phase12 import unpack

RUNS = PAPER / "workspace/runs"
OUT = RUNS / "paper1_runR_stage4_diagnostics_v1"
MARGIN = 0.05


def tci(d):
    d = np.asarray(d, float)
    se = d.std(ddof=1) / np.sqrt(d.size)
    t = 2.0 if d.size > 40 else 2.4
    return float(d.mean()), float(d.mean() - t * se), float(d.mean() + t * se)


def interp_log(pg, curve, p):
    return float(np.interp(np.log(p), np.log(pg), curve))


def crossing(rates, diffs):
    """Rate where the difference first turns from negative (pos ahead) to positive, log-interpolated; None if never
    negative at the first rate ('below grid') or never positive ('above grid')."""
    r, d = np.asarray(rates, float), np.asarray(diffs, float)
    if d[0] >= 0:
        return "below grid"
    for i in range(1, d.size):
        if d[i] >= 0:
            w = d[i - 1] / (d[i - 1] - d[i])
            return float(np.exp(np.log(r[i - 1]) + w * (np.log(r[i]) - np.log(r[i - 1]))))
    return "above grid"


def load_pairs(models):
    """{(model, budget, rate): {(rep, field): (acc_pos, acc_abs, ce_pos, ce_abs)}} and clean accuracy per model."""
    rows, clean = defaultdict(dict), defaultdict(list)
    for model, path in models.items():
        for fj in sorted(path.glob("rep*/field*/field.json")):
            d = fj.parent
            if not (d / "DONE").exists():
                raise SystemExit(f"{d} not finished")
            clean[model].append(json.loads(fj.read_text())["clean_accuracy"])
            tmp = defaultdict(dict)
            with open(d / "results.csv") as fh:
                for row in csv.DictReader(fh):
                    if row["arm"] in ("pos1024", "abs1024"):
                        k = (model, round(float(row["budget"]), 6), round(float(row["ber"]), 8))
                        tmp[k][row["arm"]] = (float(row["accuracy"]), float(row["cross_entropy"]))
            key = (d.parent.name, d.name)
            for k, v in tmp.items():
                rows[k][key] = (v["pos1024"][0], v["abs1024"][0], v["pos1024"][1], v["abs1024"][1])
    return rows, {m: float(np.mean(v)) for m, v in clean.items()}


def part_h1():
    from runR_stage1 import NAMES
    s1 = RUNS / "paper1_runR_stage1_crossover_v1"
    models = {"kws": RUNS / "paper1_runR_stage4_kws_eval_v1"}
    models.update({n: s1 / n for n in NAMES})
    rows, clean = load_pairs(models)
    pk = json.loads((RUNS / "paper1_runR_kws_prep_v1/predictions_6draws.json").read_text())
    pr = json.loads((RUNS / "paper1_runR_predictions_v1/predictions.json").read_text())
    cells = []
    for (model, b, p), pairs in sorted(rows.items()):
        src, key = (pk, f"{b:.6f}") if model == "kws" else (pr, f"{model}/{b:.6f}")
        pred = src["predictions"][key]
        v = np.array(list(pairs.values()))
        acc = tci(100 * (v[:, 0] - v[:, 1]))
        ce = tci(v[:, 2] - v[:, 3])
        cells.append(dict(model=model, budget=b, ber=p, fields=len(v), p_star=pred["p_star"],
                          predicted_dJ=interp_log(src["pgrid"], pred["curve"], p),
                          dacc=acc[0], dacc_lo=acc[1], dacc_hi=acc[2], dce=ce[0], dce_lo=ce[1], dce_hi=ce[2],
                          usable=bool(max(v[:, 0].mean(), v[:, 1].mean()) >= clean[model] - MARGIN)))
    out = {}
    for model in models:
        mc = [c for c in cells if c["model"] == model and c["usable"]]
        res = lambda c, lo, hi: (c[lo] > 0) or (c[hi] < 0)  # noqa: E731
        pos_pred = [c for c in mc if c["predicted_dJ"] < 0]
        abs_pred = [c for c in mc if c["predicted_dJ"] > 0]
        agree = lambda cs, f, sgn: sum(1 for c in cs if c[f] * sgn > 0 and res(c, f + "_lo", f + "_hi"))  # noqa: E731
        sw = {}
        for b in sorted({c["budget"] for c in mc}):
            cb = sorted([c for c in mc if c["budget"] == b], key=lambda c: c["ber"])
            if len(cb) < 2:
                continue
            sw[f"{b:.6f}"] = dict(predicted=cb[0]["p_star"], loss=crossing([c["ber"] for c in cb], [c["dce"] for c in cb]),
                                  accuracy=crossing([c["ber"] for c in cb], [-c["dacc"] for c in cb]))
        x, y = np.array([c["predicted_dJ"] for c in mc]), np.array([c["dce"] for c in mc])
        out[model] = dict(
            usable_cells=len(mc),
            predicted_pos_ahead=dict(cells=len(pos_pred),
                                     loss_pos_ahead_resolved=agree(pos_pred, "dce", -1),
                                     loss_abs_ahead_resolved=agree(pos_pred, "dce", 1),
                                     accuracy_pos_ahead_resolved=agree(pos_pred, "dacc", 1),
                                     accuracy_abs_ahead_resolved=agree(pos_pred, "dacc", -1)),
            predicted_abs_ahead=dict(cells=len(abs_pred),
                                     loss_abs_ahead_resolved=agree(abs_pred, "dce", 1),
                                     loss_pos_ahead_resolved=agree(abs_pred, "dce", -1),
                                     accuracy_abs_ahead_resolved=agree(abs_pred, "dacc", -1),
                                     accuracy_pos_ahead_resolved=agree(abs_pred, "dacc", 1)),
            loss_slope_measured_on_predicted=float(np.polyfit(x, y, 1)[0]) if len(mc) > 2 else None,
            loss_correlation=float(np.corrcoef(x, y)[0, 1]) if len(mc) > 2 else None,
            switches=sw)
    return cells, out


def grad_stats(gbar, se, g2):
    S, N = float((gbar ** 2).sum()), float((se ** 2).sum())
    return dict(mean_grad_sq=S, noise_sq=N, signal_fraction=(S - N) / S, coherence=(S - N) / float(g2.sum()))


def first_order(jumps, pos, abs_):
    """(mu_pos - mu_abs) / p per weight: bits protected by abs but not pos flip under pos, and vice versa."""
    return (jumps * (abs_.astype(float) - pos.astype(float))).sum(1)


def part_h2_kws():
    from runR_kws import KwsModel, OUT as PREP, STAGE4_FRACTIONS
    from runR_stage1 import bkey
    m = KwsModel(1)
    cals = [np.load(RUNS / f"paper1_runR_stage4_kws_masks_v1/rep{r}/b{bkey(0.125)}/calibration.npz") for r in range(6)]
    return dict(model="kws", **grad_block(m.jumps, cals, lambda r, f: (
        unpack(np.load(PREP / f"masks/rep{r}/masks.npz")[f"pos1024@{bkey(f)}"], m.n),
        unpack(np.load(PREP / f"masks/rep{r}/masks.npz")[f"abs1024@{bkey(f)}"], m.n)), STAGE4_FRACTIONS))


def part_h2_resnet(name):
    from runP_phase0 import Model
    from runR_stage1 import reps_for, bkey
    m = Model(name, 1)
    base = RUNS / "paper1_runR_stage2_masks_v1" / name
    reps = [r for r in reps_for(name) if (base / f"rep{r}/b{bkey(0.125)}/calibration.npz").exists()]
    cals = [np.load(base / f"rep{r}/b{bkey(0.125)}/calibration.npz") for r in reps]
    fr = (0.0625, 0.125, 0.1875, 0.25)

    def masks(i, f):
        st = np.load(base / f"rep{reps[i]}/b{bkey(f)}/static.npz")
        return unpack(st["pos1024"], m.n), unpack(st["abs1024"], m.n)
    return dict(model=name, reps=reps, **grad_block(m.jumps, cals, masks, fr))


def grad_block(jumps, cals, masks, fractions):
    R = len(cals)
    G = np.stack([c["gbar"] for c in cals])
    per_draw = [grad_stats(c["gbar"], c["se"], c["g2"]) for c in cals]
    cor = np.corrcoef(G)[np.triu_indices(R, 1)]
    fo = {}
    for f in fractions:
        ins, outs = [], []
        for i in range(R):
            pos, abs_ = masks(i, f)
            d = first_order(jumps, pos, abs_)
            ins.append(float(G[i] @ d))
            outs.append(float(np.delete(G, i, 0).mean(0) @ d))
        fo[f"{f:.6f}"] = dict(in_sample=float(np.mean(ins)), out_of_sample=float(np.mean(outs)),
                              out_over_in=float(np.mean(outs) / np.mean(ins)),
                              out_of_sample_pos_ahead=int(sum(o < 0 for o in outs)), draws=R)
    keys = ("signal_fraction", "coherence")
    return dict(gradient={k: float(np.median([s[k] for s in per_draw])) for k in keys},
                gbar_correlation_between_draws=float(np.median(cor)),
                first_order_pos_minus_abs_per_unit_rate=fo)


def part_h2_heldout():
    """R-A5 selection scores: predicted J of pos1024 and abs1024 on six other held-out sets per model."""
    out = defaultdict(lambda: defaultdict(list))
    for f in sorted((RUNS / "paper1_runR_v2_picks_v1/jobs").glob("*.json")):
        d = json.loads(f.read_text())
        model = "kws" if d["model"] == "kws" else "resnet8"
        for r in d["rows"]:
            out[model][(r["budget"], r["rate"])].append(r["scores"]["pos1024"] - r["scores"]["abs1024"])
    res = {}
    for model, cells in out.items():
        by_rate = defaultdict(list)
        for (b, p), v in cells.items():
            by_rate[p] += v
        res[model] = {f"{p:g}": dict(median_dJ=float(np.median(v)), share_pos_ahead=float(np.mean(np.array(v) < 0)), n=len(v))
                      for p, v in sorted(by_rate.items())}
    return res


def part_size_and_gap():
    """Size: measured mean loss increase of each mask (test windows, fault fields) against its predicted objective
    J_p (held-out training-side inputs), at rates up to 3e-3 in usable cells.  Gap: clean loss on the calibration
    inputs against the test windows."""
    from runR_stage1 import NAMES
    s1 = RUNS / "paper1_runR_stage1_crossover_v1"
    src = {"kws": (RUNS / "paper1_runR_stage4_kws_eval_v1", sorted((RUNS / "paper1_runR_kws_prep_v1/predict").glob("rep*.json")),
                   sorted(RUNS.glob("paper1_runR_stage4_kws_masks_v1/rep*/b0.125000/build.json")))}
    src.update({n: (s1 / n, sorted((RUNS / "paper1_runR_predictions_v1/jobs").glob(f"{n}_rep*.json")),
                    sorted((RUNS / "paper1_runR_stage2_masks_v1" / n).glob("rep*/b0.125000/build.json"))) for n in NAMES})
    out = {}
    for model, (ev, preds, builds) in src.items():
        J = defaultdict(list)
        for f in preds:
            for r in json.loads(f.read_text())["rows"]:
                J[(r["arm"], round(r["budget"], 6))].append((r["p"], r["J"]))
        meas, clean_ce, clean_acc = defaultdict(list), [], []
        for fj in sorted(ev.glob("rep*/field*/field.json")):
            fd = json.loads(fj.read_text())
            clean_ce.append(fd["clean_cross_entropy"]); clean_acc.append(fd["clean_accuracy"])
            with open(fj.parent / "results.csv") as fh:
                for row in csv.DictReader(fh):
                    if row["arm"] in ("pos1024", "abs1024") and float(row["ber"]) <= 3e-3 + 1e-12:
                        meas[(row["arm"], round(float(row["budget"]), 6), float(row["ber"]))].append(
                            (float(row["cross_entropy"]) - fd["clean_cross_entropy"], float(row["accuracy"])))
        c_acc = float(np.mean(clean_acc))
        ratios = defaultdict(list)
        for (arm, b, p), v in meas.items():
            v = np.array(v)
            if v[:, 1].mean() < c_acc - MARGIN or (arm, b) not in J:
                continue
            pts = np.array(J[(arm, b)])
            pg = np.unique(pts[:, 0])
            curve = [pts[pts[:, 0] == q, 1].mean() for q in pg]
            pred = interp_log(pg, curve, p)
            if pred > 0:
                ratios[arm].append(float(v[:, 0].mean() / pred))
        cal = float(np.mean([json.loads(f.read_text())["clean_calibration_loss"] for f in builds]))
        out[model] = dict(measured_over_predicted_loss_increase={a: float(np.median(r)) for a, r in ratios.items()},
                          cells={a: len(r) for a, r in ratios.items()},
                          clean_loss_calibration=cal, clean_loss_test=float(np.mean(clean_ce)),
                          test_over_calibration=float(np.mean(clean_ce) / cal))
    return out


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main(out=OUT):
    t0 = time.time()
    out.mkdir(parents=False, exist_ok=False)
    cells, h1 = part_h1()
    from runR_stage1 import NAMES
    h2 = [part_h2_kws()] + [part_h2_resnet(n) for n in NAMES]
    held = part_h2_heldout()
    size = part_size_and_gap()
    s1 = json.loads((RUNS / "paper1_runR_stage1_analysis_v1/summary.json").read_text())["per_checkpoint"]
    res = [g for g in h2 if g["model"] != "kws"]
    lr = [s1[g["model"]]["mean_log_ratio"] for g in res]
    across = dict(checkpoints=len(res), note="ResNet-8 only; Spearman with the Stage 1 mean log(measured/predicted switch)",
                  signal_fraction=spearman([g["gradient"]["signal_fraction"] for g in res], lr),
                  gbar_correlation=spearman([g["gbar_correlation_between_draws"] for g in res], lr),
                  out_over_in_at_12p5=spearman([g["first_order_pos_minus_abs_per_unit_rate"]["0.125000"]["out_over_in"] for g in res], lr),
                  test_over_calibration_loss=spearman([size[g["model"]]["test_over_calibration"] for g in res], lr))
    summary = dict(status="post-hoc exploratory diagnostics of the Stage 4 failure; no new fault field; no Stage 2 outcome read",
                   script_sha256=sha256_file(Path(__file__)), H1_loss_vs_accuracy=h1, H2_gradient=h2,
                   H2_predicted_pos_minus_abs_on_RA5_heldout_sets=held,
                   size_and_generalization_gap=size,
                   stage1_mean_log_ratio_measured_over_predicted={k: v["mean_log_ratio"] for k, v in s1.items()},
                   across_resnet_checkpoints=across,
                   seconds=round(time.time() - t0))
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    with open(out / "cells.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cells[0])); w.writeheader(); w.writerows(cells)
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    import sys
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else OUT)
