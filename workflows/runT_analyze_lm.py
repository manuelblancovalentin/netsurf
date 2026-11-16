#!/usr/bin/env python3
"""Run T analysis for the W8A8 language models (addendum T2).  Written and committed before any T2 output was read.

Same rules as runT_analyze.py, applied to the language-model layout (<runs>/<target>/draw<D>/ber<I>/results*.csv):
  DistilBERT   accuracy in points; usable within 5 points of clean; resolved at >= 0.5 points with Holm p < 0.05
  Pythia, Qwen mean token NLL in nats (lower is better); the analysis works on minus the NLL, so that "better" keeps
               its sign; usable within 0.5 nats of clean; resolved at >= 0.05 nats with Holm p < 0.05
Unit: one draw (paired across conditions).  Contrasts, tolerated rate and Holm family as in runT_analyze.py.

    runT_analyze_lm.py --runs DIR --out DIR --targets distilbert_sst2_int8 pythia160m_int8 qwen3_0_6b_int8
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import runT_analyze as A

SPEC = {"distilbert_sst2_int8": dict(metric="accuracy", scale=100.0, margin=5.0, min_diff=0.5, draws=8),
        "pythia160m_int8": dict(metric="mean_nll", scale=-1.0, margin=0.5, min_diff=0.05, draws=8),
        "qwen3_0_6b_int8": dict(metric="mean_nll", scale=-1.0, margin=0.5, min_diff=0.05, draws=4)}


def load(base: Path, spec):
    acc, clean, draws = defaultdict(dict), [], set()
    for d in sorted(base.glob("draw*/ber*")):
        if not (d / "DONE").exists():
            raise SystemExit(f"{d} not finished")
        dr = int(d.parent.name[4:])
        draws.add(dr)
        for f in sorted(d.glob("results*.csv")):
            for r in csv.DictReader(open(f)):
                acc[(r["condition"], float(r["ber"]))][(0, dr)] = spec["scale"] * float(r[f"metric_{spec['metric']}"])
        clean.append(spec["scale"] * json.loads((d / "DONE").read_text())["clean"][spec["metric"]])
    if len(draws) != spec["draws"]:
        raise SystemExit(f"{base}: {len(draws)} draws, {spec['draws']} expected")
    return acc, float(np.mean(clean))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--targets", nargs="+", required=True)
    a = ap.parse_args()
    data = {t: load(a.runs / t, SPEC[t]) for t in a.targets}
    a.out.mkdir(parents=True, exist_ok=False)
    summary, crows, trows = {}, [], []
    for t in a.targets:
        spec = SPEC[t]
        A.MARGIN, A.MIN_DIFF = spec["margin"], spec["min_diff"]
        signs = np.random.default_rng(A.SEED).choice((-1.0, 1.0), size=(A.ASSIGNMENTS, spec["draws"]))
        acc, clean = data[t]
        curves, tol, rows = A.analyze(acc, clean, signs)
        unit = "accuracy points" if spec["metric"] == "accuracy" else "minus NLL (nats/token)"
        summary[t] = dict(clean=clean, unit=unit, tolerated={c: tol[c] for c in curves},
                          contrasts_resolved=sum(r["resolved"] for r in rows), contrasts=len(rows))
        crows += [dict(target=t, condition=c, ber=r, value=v, unit=unit) for c in curves for r, v in curves[c].items()]
        trows += [dict(target=t, unit=unit, **r) for r in rows]
        print(f"{t:22s} clean {clean:.3f}  tolerated: abs@12.5% {tol.get('tmr_abs@0.125')}  msb@12.5% {tol.get('tmr_msb@0.125')}  "
              f"secded_32 {tol.get('secded_32')}  abft {tol.get('abft_ideal')}  none {tol.get('unprotected')}")
    for nm, rows in (("curves.csv", crows), ("contrasts.csv", trows)):
        with open(a.out / nm, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (a.out / "summary.json").write_text(json.dumps(dict(
        status="Run T language-model analysis (addendum T2)", script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        targets=summary), indent=1))


if __name__ == "__main__":
    main()
