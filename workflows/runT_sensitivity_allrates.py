#!/usr/bin/env python3
"""Sensitivity of the Run T contrasts to the usability filter (review of draft v22, point 5).

The frozen analyses (runT_analyze.py, runT_analyze_lm_v2.py) test a contrast only at grid rates where the better of the
two conditions is usable, and apply Holm within that data-dependent family.  This script applies the same tests, the
same seed, the same resolution rule and the same Holm correction to the whole predefined family: every fixed contrast at
every grid rate, with no usability filter.  It reads the finished outputs only and writes a new directory.

    runT_sensitivity_allrates.py --tflite DIR --lm DIR DIR --out DIR
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

import runT_analyze as A
import runT_analyze_lm_v2 as L

GEN = ["jet-e1", "jet-e3", "jet-e10", "jet-e30", "jet-e100", "mnist-e1", "mnist-e3", "mnist-e10", "mnist-e30", "vww"]
ENGINE = ["dscnn", "resnet8-pinned", "resnet8-undertrained_e100", "resnet8-pruned_s05", "resnet8-pruned_s07",
          "resnet8-retrained_seed7101", "resnet8-retrained_seed7102", "resnet8-retrained_seed7103",
          "resnet8-retrained_seed7104", "resnet8-retrained_seed7105"]
NO_FILTER = 1e12          # MARGIN so large that no rate is excluded from the contrast family


def run(acc, clean, draws, margin, min_diff):
    signs = np.random.default_rng(A.SEED).choice((-1.0, 1.0), size=(A.ASSIGNMENTS, draws))
    A.MARGIN, A.MIN_DIFF = NO_FILTER, min_diff
    _, _, rows = A.analyze(acc, clean, signs)
    A.MARGIN = margin
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tflite", type=Path, required=True)
    ap.add_argument("--lm", type=Path, nargs=2, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    jobs = [(cid, A.load(a.tflite / cid, ["", "_fkeras"], 24), 24, 5.0, 0.5) for cid in GEN]
    jobs += [(cid, A.load(a.tflite / cid, [""], 24), 24, 5.0, 0.5) for cid in ENGINE]
    jobs += [("toycar", A.load(a.tflite / "toycar", [""], 8), 8, 5.0, 0.5)]
    for t, spec in L.SPEC.items():
        jobs.append((t, L.load([d / t for d in a.lm], spec), spec["draws"], spec["margin"], spec["min_diff"]))
    a.out.mkdir(parents=True, exist_ok=False)
    out, summary = [], {}
    for cid, (acc, clean), draws, margin, min_diff in jobs:
        rows = run(acc, clean, draws, margin, min_diff)
        out += [dict(checkpoint=cid, **r) for r in rows]
        summary[cid] = dict(contrasts=len(rows), confirmed=sum(r["resolved"] for r in rows))
        print(f"{cid:28s} contrasts {len(rows):3d} confirmed {summary[cid]['confirmed']:3d}")
    with open(a.out / "contrasts.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    (a.out / "summary.json").write_text(json.dumps(dict(
        status="Run T sensitivity: every grid rate, no usability filter (review of v22, point 5)",
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), checkpoints=summary), indent=1))


if __name__ == "__main__":
    main()
