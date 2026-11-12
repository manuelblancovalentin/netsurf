#!/usr/bin/env python3
"""Run R, amendment R-A3: rebuild `abs1024` and `pos1024` at eight budgets for one (checkpoint, replication).

Same 1,024 calibration images as Run Q (training images from offset 1000 * rep) and Run Q's scoring code.  At the
four budgets of the paper the masks must equal Run Q's bit for bit; otherwise the job writes FAILED and stops.

  job --checkpoint C --rep R [--threads 2]

Output (write-once): workspace/runs/paper1_runR_masks_v1/<checkpoint>/rep<R>/{masks.npz, masks.json, DONE}.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from runP_phase0 import PAPER, ROOT, Model, sha256_file
from runP_phase12 import pack, unpack
from runQ_calibration import OUT as RUNQ, scores
from runR_stage1 import FRACTIONS, NAMES, bkey, reps_for
from netsurf.mean_variance import mask_sha256, top_budget

OUT = PAPER / "workspace/runs/paper1_runR_masks_v1"
OLD = ("0.062500", "0.125000", "0.187500", "0.250000")
ARMS = ("abs", "pos")


def job(name, rep, threads):
    assert rep in reps_for(name)
    out = OUT / name / f"rep{rep}"
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    model = Model(name, threads)
    x, y, idx = model.images("ranking", 1024, 1000 * rep)
    assert idx[0] == 1000 * rep and len(idx) == 1024
    S = scores(model.per_example(x, y), model.jumps)
    total = model.n * 8
    q = np.load(RUNQ / name / f"rep{rep}" / "masks.npz")
    masks, info = {}, {}
    for f in FRACTIONS:
        count = int(round(f * total))
        for a in ARMS:
            k = f"{a}1024@{bkey(f)}"
            masks[k] = top_budget(S[a], count)
            info[k] = dict(count=count, sha256=mask_sha256(masks[k]))
            if bkey(f) in OLD:
                ref = unpack(q[k], model.n)
                info[k]["equals_run_q"] = bool(np.array_equal(ref, masks[k]))
    bad = [k for k, v in info.items() if v.get("equals_run_q") is False]
    if bad:
        (out / "FAILED").write_text(json.dumps(dict(differ_from_run_q=bad, info=info), indent=0))
        raise SystemExit(f"masks differ from Run Q: {bad}")
    np.savez(out / "masks.npz", **{k: pack(v) for k, v in masks.items()})
    (out / "masks.json").write_text(json.dumps(dict(masks=info, seconds=round(time.time() - t0)), indent=0))
    (out / "DONE").write_text(json.dumps(dict(seconds=round(time.time() - t0), script_sha256=sha256_file(Path(__file__)))))
    print(json.dumps(dict(job_done=f"{name}/rep{rep}", seconds=round(time.time() - t0))), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, choices=NAMES)
    ap.add_argument("--rep", type=int, required=True)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    job(a.checkpoint, a.rep, a.threads)
