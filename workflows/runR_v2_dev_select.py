#!/usr/bin/env python3
"""Development, training-side data only: add v1 to the held-out safeguard.  For every row of a finished
`runR_v2_dev.py` job, score the v1 mask on the same selection set, then pick among {v1, v2, abs1024, pos1024} (and,
separately, among {v1, abs1024, pos1024}) by selection-set objective, and report the dev objective of the pick.

  job --file <runR_v2_dev output json>
Output: workspace/runs/paper1_runR_v2_dev_select_v1/<same name>.json (write-once).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from runP_phase0 import PAPER, Model, sha256_file
from runP_phase12 import unpack
from runR_stage1 import bkey
from runR_v2_dev import RATES, RUNS, objective_fn

OUT = PAPER / "workspace/runs/paper1_runR_v2_dev_select_v1"


def job(path: Path):
    d = json.loads(path.read_text())
    target = OUT / path.name
    OUT.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise SystemExit(f"{target} exists")
    b, rep = bkey(d["budget"]), d["rep"]
    if d["model"] == "resnet8":
        m = Model(d["key"], 1)
        xs, ys, ixs = m.images("empirical_validation", 512, 5000 + 512 * rep)
        assert ixs[0] == 30000 + 512 * rep
        v1 = np.load(RUNS / "paper1_runR_stage2_masks_v1" / d["key"] / f"rep{rep}" / f"b{b}" / "analytic.npz")
    else:
        from runR_kws import KwsModel
        m = KwsModel(1)
        xs, ys, _ = m.clips("empirical_validation", 512, 1024 + 512 * rep)
        v1 = np.load(RUNS / "paper1_runR_stage4_kws_masks_v1" / f"rep{rep}" / f"b{b}" / "analytic.npz")
    J_sel = objective_fn(m, xs, ys)
    rows = []
    for r in d["rows"]:
        p = r["rate"]
        sel = dict(r["selection_scores"], v1=J_sel(unpack(v1[f"{p:g}"], m.n), p))
        pick4 = min(sel, key=sel.get)
        pick3 = min(("v1", "abs1024", "pos1024"), key=sel.get)
        rows.append(dict(rate=p, selection=sel, pick_all=pick4, pick_v1_static=pick3,
                         dev_pick_all=r["dev"][pick4], dev_pick_v1_static=r["dev"][pick3], dev=r["dev"]))
    target.write_text(json.dumps(dict(source=path.name, model=d["model"], key=d["key"], budget=d["budget"], rows=rows,
                                      script_sha256=sha256_file(Path(__file__))), indent=0, default=float))
    print("done", path.name, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("job"); j.add_argument("--file", type=Path, required=True)
    a = ap.parse_args()
    job(a.file)
