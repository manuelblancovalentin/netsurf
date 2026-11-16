#!/usr/bin/env python3
"""Run T language-model analysis with 16 fault draws per model (addendum T4).  Written and committed before any of the
added draws exists.

Identical rules to runT_analyze_lm.py (usable margin, resolution threshold, contrasts, Holm family, sign-flip test with
the same seed and number of assignments); the only change is that the draws of several run directories are pooled and
every model must have exactly 16 distinct draws (0..15).  A draw index present in two directories is an error.

    runT_analyze_lm_v2.py --runs DIR [DIR ...] --out DIR --targets distilbert_sst2_int8 pythia160m_int8 qwen3_0_6b_int8
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
from runT_analyze_lm import SPEC as SPEC8

DRAWS = 16
SPEC = {t: dict(s, draws=DRAWS) for t, s in SPEC8.items()}


def load(bases, spec, fkeras=True):
    acc, clean, seen = defaultdict(dict), [], {}
    for base in bases:
        for d in sorted(base.glob("draw*/ber*")):
            if not (d / "DONE").exists() or (fkeras and not (d / "DONE_fkeras").exists()):
                raise SystemExit(f"{d} not finished")
            dr = int(d.parent.name[4:])
            if seen.setdefault(dr, base) != base:
                raise SystemExit(f"draw {dr} in both {seen[dr]} and {base}")
            for f in sorted(d.glob("results*.csv")):
                for r in csv.DictReader(open(f)):
                    acc[(r["condition"], float(r["ber"]))][(0, dr)] = spec["scale"] * float(r[f"metric_{spec['metric']}"])
            clean.append(spec["scale"] * json.loads((d / "DONE").read_text())["clean"][spec["metric"]])
    if sorted(seen) != list(range(spec["draws"])):
        raise SystemExit(f"draws {sorted(seen)}, expected 0..{spec['draws'] - 1}")
    return acc, float(np.mean(clean))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--targets", nargs="+", required=True)
    a = ap.parse_args()
    data = {t: load([r / t for r in a.runs], SPEC[t]) for t in a.targets}
    a.out.mkdir(parents=True, exist_ok=False)
    summary, crows, trows = {}, [], []
    for t in a.targets:
        spec = SPEC[t]
        A.MARGIN, A.MIN_DIFF = spec["margin"], spec["min_diff"]
        signs = np.random.default_rng(A.SEED).choice((-1.0, 1.0), size=(A.ASSIGNMENTS, spec["draws"]))
        acc, clean = data[t]
        curves, tol, rows = A.analyze(acc, clean, signs)
        unit = "accuracy points" if spec["metric"] == "accuracy" else "minus NLL (nats/token)"
        summary[t] = dict(clean=clean, unit=unit, draws=spec["draws"], tolerated={c: tol[c] for c in curves},
                          contrasts_resolved=sum(r["resolved"] for r in rows), contrasts=len(rows))
        crows += [dict(target=t, condition=c, ber=r, value=v, unit=unit) for c in curves for r, v in curves[c].items()]
        trows += [dict(target=t, unit=unit, **r) for r in rows]
        print(f"{t:22s} clean {clean:.3f}  resolved {summary[t]['contrasts_resolved']}/{len(rows)}")
    for nm, rows in (("curves.csv", crows), ("contrasts.csv", trows)):
        with open(a.out / nm, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (a.out / "summary.json").write_text(json.dumps(dict(
        status="Run T language-model analysis, 16 draws (addendum T4)",
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        runs=[str(r) for r in a.runs], targets=summary), indent=1))


if __name__ == "__main__":
    main()
