#!/usr/bin/env python3
"""Run R amendment R-A5: the selected rate-aware mask (v2, `rate_aware_mask_selected`) for every Stage 2 and Stage 4
cell, from training-side data only.

v2 = the v1 analytic search, unchanged, followed by a held-out choice among {v1 mask built for the rate, abs1024,
pos1024}: the one with the lowest predicted objective J_p on 512 held-out training inputs that the search did not use
(ResNet-8: training images 30000 + 512 r; DS-CNN: validation clips 1024 + 512 r).  The chosen mask is always one of
three masks already evaluated in the frozen Stage 2 / Stage 4 runs, so v2 is evaluated on the same fault fields.

  job --model resnet8 --checkpoint C --rep R | job --model kws --rep R
  summarize   -> picks.json (sha256 recorded in the amendment before any Stage 2 / Stage 4 outcome is read)
Output: workspace/runs/paper1_runR_v2_picks_v1/ (write-once).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from runP_phase0 import PAPER, Model, sha256_file
from runP_phase12 import unpack
from runR_stage1 import NAMES, bkey, reps_for
from runR_v2_dev import RUNS, objective_fn

OUT = PAPER / "workspace/runs/paper1_runR_v2_picks_v1"
CANDIDATES = ("ra_analytic", "abs1024", "pos1024")


def job(model, ckpt, rep):
    target = OUT / "jobs" / f"{model}_{ckpt}_rep{rep}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise SystemExit(f"{target} exists")
    t0 = time.time()
    rows = []
    if model == "resnet8":
        from runR_stage2 import FRACTIONS, RATES
        m = Model(ckpt, 1)
        xs, ys, ixs = m.images("empirical_validation", 512, 5000 + 512 * rep)
        assert ixs[0] == 30000 + 512 * rep
        J = objective_fn(m, xs, ys)
        for f in FRACTIONS:
            base = RUNS / "paper1_runR_stage2_masks_v1" / ckpt / f"rep{rep}" / f"b{bkey(f)}"
            st, an = np.load(base / "static.npz"), np.load(base / "analytic.npz")
            for p in RATES:
                masks = {"ra_analytic": unpack(an[f"{p:g}"], m.n), "abs1024": unpack(st["abs1024"], m.n),
                         "pos1024": unpack(st["pos1024"], m.n)}
                sc = {k: J(v, p) for k, v in masks.items()}
                rows.append(dict(budget=f, rate=p, scores=sc, pick=min(sc, key=sc.get)))
    else:
        from runR_kws import KwsModel, OUT as PREP, STAGE4_FRACTIONS
        from runR_stage1 import RATES
        m = KwsModel(1)
        xs, ys, ixs = m.clips("empirical_validation", 512, 1024 + 512 * rep)
        assert ixs[0] == 1024 + 512 * rep
        J = objective_fn(m, xs, ys)
        for f in STAGE4_FRACTIONS:
            pa = np.load(PREP / "masks" / f"rep{rep}" / "masks.npz")
            an = np.load(RUNS / "paper1_runR_stage4_kws_masks_v1" / f"rep{rep}" / f"b{bkey(f)}" / "analytic.npz")
            for p in RATES:
                masks = {"ra_analytic": unpack(an[f"{p:g}"], m.n), "abs1024": unpack(pa[f"abs1024@{bkey(f)}"], m.n),
                         "pos1024": unpack(pa[f"pos1024@{bkey(f)}"], m.n)}
                sc = {k: J(v, p) for k, v in masks.items()}
                rows.append(dict(budget=f, rate=p, scores=sc, pick=min(sc, key=sc.get)))
    target.write_text(json.dumps(dict(model=model, checkpoint=ckpt, rep=rep, rows=rows, seconds=round(time.time() - t0),
                                      script_sha256=sha256_file(Path(__file__))), indent=0))
    print("done", target.name, round(time.time() - t0), flush=True)


def summarize():
    target = OUT / "picks.json"
    if target.exists():
        raise SystemExit("picks.json exists")
    picks = {}
    for f in sorted((OUT / "jobs").glob("*.json")):
        d = json.loads(f.read_text())
        for r in d["rows"]:
            picks[f"{d['model']}/{d['checkpoint']}/rep{d['rep']}/{r['budget']:.6f}/{r['rate']:g}"] = r["pick"]
    expected = sum(len(reps_for(n)) for n in NAMES) * 4 * 8 + 6 * 4 * 12
    if len(picks) != expected:
        raise SystemExit(f"{len(picks)} picks, expected {expected}")
    target.write_text(json.dumps(dict(status="R-A5 picks, training-side data only; no Stage 2/4 outcome read",
                                      candidates=list(CANDIDATES), picks=picks), indent=0, sort_keys=True))
    from collections import Counter
    print(Counter(v for k, v in picks.items() if k.startswith("resnet8")), Counter(v for k, v in picks.items() if k.startswith("kws")))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("job")
    j.add_argument("--model", choices=("resnet8", "kws"), required=True)
    j.add_argument("--checkpoint", default="kws")
    j.add_argument("--rep", type=int, required=True)
    sub.add_parser("summarize")
    a = ap.parse_args()
    if a.cmd == "job":
        job(a.model, a.checkpoint, a.rep)
    else:
        summarize()
