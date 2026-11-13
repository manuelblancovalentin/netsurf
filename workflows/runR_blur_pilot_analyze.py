#!/usr/bin/env python3
"""EXPLORATORY analysis of the blur pilot (runR_blur_pilot.py), post hoc.  Descriptive; nothing here is a manuscript
result.  Reads the pilot outputs and the finished measured fault runs (Stage 1 ResNet-8, Stage 4 DS-CNN), which use
the same evaluation inputs and, for the ``real`` variant, the same fault fields.

  blur.csv     per (model, rep, arm, budget, rate): measured TFLite accuracy (mean of 8 fields) and the companion's
               accuracy under each variant (mean, gauss, hybrid, real; mean of 8 draws)
  switch.csv   per (model, rep, budget, rate): positive minus absolute, measured and under each variant
  bell.csv     per (model, arm, rate): real flips against the two-moment Gaussian, accuracy over 64 draws and per-input
               margin change pooled over draws and inputs
  step.csv     per (model, bit): single-flip loss change against g*Delta and g*Delta + F*Delta^2/2
  summary.json headline numbers
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from site_paths import WT  # noqa: E402

RUNS = WT / "paper1-tecs-revision-v1/workspace/runs"
PILOT = RUNS / "paper1_blur_pilot_v1"
OUT = RUNS / "paper1_blur_pilot_analysis_v1"
VARIANTS = ("mean", "gauss", "hybrid", "real")


def measured(model, rep):
    if model == "kws":
        base, key = RUNS / "paper1_runR_stage4_kws_eval_v1" / f"rep{rep}", None
    else:
        base = RUNS / "paper1_runR_stage1_crossover_v1" / model / f"rep{rep}"
    acc = defaultdict(list)
    for f in sorted(base.glob("field*/results.csv")):
        for r in csv.DictReader(open(f)):
            if r["arm"] in ("abs1024", "pos1024"):
                acc[(r["arm"], round(float(r["budget"]), 6), float(r["ber"]))].append(float(r["accuracy"]))
    clean = [json.loads((f.parent / "field.json").read_text())["clean_accuracy"] for f in sorted(base.glob("field*/results.csv"))]
    return {k: (float(np.mean(v)), len(v)) for k, v in acc.items()}, float(np.mean(clean))


def blur(summary):
    rows, sw = [], []
    for d in sorted((PILOT / "blur").iterdir()):
        model, rep = d.name.rsplit("_rep", 1)
        meas, clean_meas = measured(model, int(rep))
        done = json.loads((d / "DONE").read_text())
        acc = defaultdict(list)
        for r in csv.DictReader(open(d / "rows.csv")):
            acc[(r["arm"], round(float(r["budget"]), 6), float(r["ber"]), r["variant"])].append(float(r["accuracy"]))
        cells = sorted({k[:3] for k in acc})
        for arm, b, p in cells:
            m, n = meas[(arm, b, p)]
            row = dict(model=model, rep=int(rep), arm=arm, budget=b, ber=p, measured=m, fields=n)
            for v in VARIANTS:
                row[v] = float(np.mean(acc[(arm, b, p, v)]))
                row[f"{v}_sd"] = float(np.std(acc[(arm, b, p, v)], ddof=1)) if len(acc[(arm, b, p, v)]) > 1 else 0.0
            rows.append(row)
        for b, p in sorted({(c[1], c[2]) for c in cells}):
            pos = next(r for r in rows if r["model"] == model and r["rep"] == int(rep) and r["arm"] == "pos1024" and r["budget"] == b and r["ber"] == p)
            ab = next(r for r in rows if r["model"] == model and r["rep"] == int(rep) and r["arm"] == "abs1024" and r["budget"] == b and r["ber"] == p)
            sw.append(dict(model=model, rep=int(rep), budget=b, ber=p, **{k: pos[k] - ab[k] for k in ("measured",) + VARIANTS}))
        summary["clean"][d.name] = dict(measured_tflite=clean_meas, companion=done["clean_accuracy"])
    err = {}
    for model in sorted({r["model"] for r in rows}):
        R = [r for r in rows if r["model"] == model]
        S = [s for s in sw if s["model"] == model]
        err[model] = {}
        for v in VARIANTS:
            e = np.array([100 * (r[v] - r["measured"]) for r in R])
            ds = np.array([100 * (s[v] - s["measured"]) for s in S])
            big = [s for s in S if abs(s["measured"]) >= 0.005]
            agree = sum(np.sign(s[v]) == np.sign(s["measured"]) for s in big)
            err[model][v] = dict(cells=len(R), accuracy_mean_abs_error=float(np.abs(e).mean()), accuracy_mean_signed_error=float(e.mean()),
                                 accuracy_worst_error=float(e[np.argmax(np.abs(e))]),
                                 pos_minus_abs_mean_abs_error=float(np.abs(ds).mean()),
                                 pos_minus_abs_worst_error=float(ds[np.argmax(np.abs(ds))]),
                                 sign_agreement_where_measured_gap_ge_half_point=f"{agree}/{len(big)}")
    summary["blur"] = err
    return rows, sw


def moments(x):
    x = np.asarray(x, dtype=np.float64)
    c = x - x.mean()
    s = c.std()
    return float(x.mean()), float(s), float((c ** 3).mean() / s ** 3) if s > 0 else 0.0, float((c ** 4).mean() / s ** 4 - 3) if s > 0 else 0.0


def ks(a, b):
    a, b = np.sort(a), np.sort(b)
    grid = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, grid, side="right") / len(a) - np.searchsorted(b, grid, side="right") / len(b))))


def bell(summary):
    rows = []
    for d in sorted((PILOT / "bell").iterdir()):
        model = d.name.rsplit("_rep", 1)[0]
        z = np.load(d / "margins.npz")
        clean = z["clean"]
        keys = sorted({k.rsplit("|", 2)[0] for k in z.files if k != "clean"})
        for key in keys:
            arm, p = key.split("|")
            out = dict(model=model, arm=arm, ber=float(p))
            for kind in ("real", "gauss"):
                A, M = z[f"{key}|{kind}|accuracy"], z[f"{key}|{kind}|margin"]
                shift = (M - clean[None, :]).reshape(-1)
                m, s, sk, ku = moments(shift)
                out.update({f"{kind}_acc_mean": float(A.mean()), f"{kind}_acc_sd": float(A.std(ddof=1)),
                            f"{kind}_acc_min": float(A.min()), f"{kind}_shift_mean": m, f"{kind}_shift_sd": s,
                            f"{kind}_shift_skew": sk, f"{kind}_shift_excess_kurtosis": ku,
                            f"{kind}_shift_q001": float(np.quantile(shift, 0.001)), f"{kind}_shift_q01": float(np.quantile(shift, 0.01)),
                            f"{kind}_flipped_per_draw_mean": float(((M < 0) & (clean[None, :] > 0)).sum(1).mean())})
                out[f"{kind}_acc_moments"] = moments(A)
            Mr, Mg = z[f"{key}|real|margin"] - clean[None, :], z[f"{key}|gauss|margin"] - clean[None, :]
            out["ks_shift"] = ks(Mr.reshape(-1), Mg.reshape(-1))
            out["corr_per_input_mean_shift"] = float(np.corrcoef(Mr.mean(0), Mg.mean(0))[0, 1])
            out["corr_per_input_sd_shift"] = float(np.corrcoef(Mr.std(0), Mg.std(0))[0, 1])
            rows.append(out)
    summary["bell"] = [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if not k.endswith("_moments")} for r in rows]
    return rows


def step(summary):
    rows, out = [], {}
    for d in sorted((PILOT / "step").iterdir()):
        model = d.name.rsplit("_rep", 1)[0]
        R = list(csv.DictReader(open(d / "rows.csv")))
        out[model] = {}
        for b in range(8):
            S = [r for r in R if int(r["bit"]) == b]
            real = np.array([float(r["real"]) for r in S])
            lin = np.array([float(r["linear"]) for r in S])
            quad = np.array([float(r["quadratic"]) for r in S])
            jump = np.array([abs(float(r["jump"])) for r in S])
            nz = real != 0

            def rank(a, c):
                ra, rc = np.argsort(np.argsort(a)), np.argsort(np.argsort(c))
                return float(np.corrcoef(ra, rc)[0, 1])
            rec = dict(model=model, bit=b, weights=len(S), median_abs_jump=float(np.median(jump)),
                       median_abs_real=float(np.median(np.abs(real))), median_abs_linear=float(np.median(np.abs(lin))),
                       median_abs_quadratic=float(np.median(np.abs(quad))), zero_real=int((~nz).sum()),
                       sign_agree_linear=float(np.mean(np.sign(real[nz]) == np.sign(lin[nz]))) if nz.any() else float("nan"),
                       sign_agree_quadratic=float(np.mean(np.sign(real[nz]) == np.sign(quad[nz]))) if nz.any() else float("nan"),
                       real_positive=float(np.mean(real > 0)),
                       rank_corr_real_linear=rank(real, lin), rank_corr_real_quadratic=rank(real, quad),
                       median_ratio_real_over_quadratic=float(np.median(real[quad != 0] / quad[quad != 0])),
                       share_of_real_explained_quadratic=float(1 - np.sum((real - quad) ** 2) / np.sum((real - real.mean()) ** 2)))
            rows.append(rec)
            out[model][b] = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in rec.items() if k not in ("model", "bit")}
    summary["step"] = out
    return rows


def write(name, rows):
    with open(OUT / name, "w", newline="") as fh:
        keys = [k for k in rows[0] if not k.endswith("_moments")]
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    summary = dict(status="EXPLORATORY, post hoc", clean={})
    rows, sw = blur(summary)
    write("blur.csv", rows); write("switch.csv", sw)
    write("bell.csv", bell(summary))
    write("step.csv", step(summary))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
