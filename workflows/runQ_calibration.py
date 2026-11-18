#!/usr/bin/env python3
"""Run Q: calibration size of the gradient rankings on ResNet-8 (plan `RUN_Q_CALIBRATION_SIZE_PLAN_2026-11-11.md`).

The paper builds every data-driven ranking from 64 calibration images.  Run P showed that the absolute-gradient mask
built from 512 nested images (a superset of those 64) is better in many cells.  Run Q asks whether the other gradient
rankings gain in the same way and whether their order changes with calibration size.

  check  --checkpoint C            Rebuild the absolute, positive and global gradient masks of every replication from
                                   the replication's own 64 calibration images and compare them with the saved masks
                                   (identity check, prerequisite).
  job    --checkpoint C --rep R    Build the three rankings from nested calibration sets [1000 R, 1000 R + n),
                                   n in {128, 256, 512, 1024}, at the saved budgets; check that the 512-image
                                   absolute-gradient mask equals Run P's; replay check of the saved absolute-gradient
                                   mask (exact); replay every new mask on every saved schedule at BER 1e-3 .. 3e-2.

Scores (as in `netsurf/estimators_m3.py` and `netsurf/ordering.py`):
  absolute  mean_x |g_i(x) * delta_b|          positive  mean_x max(0, g_i(x) * delta_b)
  global    |mean_x g_i(x) * delta_b|          (gradient of the mean loss)
Ties: flat storage order (`top_budget`).  No asset lock is changed.  Every output directory is new.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from runP_phase0 import BUDGETS, CHECKPOINTS, PAPER, RANKINGS, ROOT, Model, bundle_dir, load_masks, sha256_file
from runP_phase12 import OUT as RUNP_OUT, Replayer, pack, unpack
from netsurf.mean_variance import mask_sha256, top_budget

OUT = PAPER / "workspace/runs/paper1_runQ_calibration_size_v1"
SIZES = (128, 256, 512, 1024)
BERS = (1e-3, 3e-3, 1e-2, 3e-2)
KINDS = ("abs", "pos", "global")


def scores(G, jumps):
    """Per-bit scores of the three gradient rankings from per-example gradients G (images x weights)."""
    J = np.asarray(jumps, dtype=np.float64)
    gabs = np.abs(G).mean(0)
    gpos, gneg = np.maximum(G, 0.0).mean(0), np.maximum(-G, 0.0).mean(0)
    gmean = G.mean(0)
    return {
        "abs": gabs[:, None] * np.abs(J),
        "pos": np.where(J > 0, gpos[:, None] * J, gneg[:, None] * (-J)),
        "global": np.abs(gmean[:, None] * J),
    }


def stage_check(name, threads):
    spec = CHECKPOINTS[name]
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / f"check_{name}.json"
    assert not target.exists(), target
    model = Model(name, threads)
    masks, _ = load_masks(model, bundle_dir(spec["bundle"]), spec["reps"])
    rows = []
    t0 = time.time()
    for rep in spec["reps"]:
        x, y, idx = model.images("ranking", 64, 1000 * rep)
        S = scores(model.per_example(x, y), model.jumps)
        for kind, rk in (("abs", "abs_grad"), ("pos", "pos_grad"), ("global", "global_grad")):
            for b in BUDGETS:
                saved = masks[rep][f"{RANKINGS[rk]}@{b}"]
                rebuilt = top_budget(S[kind], int(saved.sum()))
                rows.append(dict(rep=rep, kind=kind, budget=b, bits=int(saved.sum()),
                                 overlap=float((saved & rebuilt).sum() / saved.sum()), identical=bool((saved == rebuilt).all())))
    t1 = time.time()
    x, y, _ = model.images("ranking", 1024, 1000 * spec["reps"][0])
    G = model.per_example(x, y)
    per_example_1024_seconds = round(time.time() - t1, 1)
    target.write_text(json.dumps(dict(rows=rows, seconds=round(t1 - t0), per_example_1024_seconds=per_example_1024_seconds,
                                      gradient_bytes_1024=int(G.nbytes)), indent=0))
    worst = {k: min(r["overlap"] for r in rows if r["kind"] == k) for k in KINDS}
    print(json.dumps(dict(checkpoint=name, min_overlap=worst, identical=sum(r["identical"] for r in rows), of=len(rows),
                          per_example_1024_seconds=per_example_1024_seconds)), flush=True)


def stage_job(name, rep, threads):
    spec = CHECKPOINTS[name]
    out = OUT / name / f"rep{rep}"
    out.mkdir(parents=True, exist_ok=False)
    model = Model(name, threads)
    saved = np.load(RUNP_OUT / name / "saved_absgrad_masks.npz")
    budget = {b: int(unpack(saved[f"r{rep}@{b}"], model.n).sum()) for b in BUDGETS}
    t0 = time.time()
    x, y, idx = model.images("ranking", max(SIZES), 1000 * rep)
    assert idx[0] == 1000 * rep and len(idx) == max(SIZES)
    G = model.per_example(x, y)
    masks, hashes = {}, {}
    for n in SIZES:
        S = scores(G[:n], model.jumps)
        for kind in KINDS:
            for b in BUDGETS:
                k = f"{kind}{n}@{b}"
                masks[k] = top_budget(S[kind], budget[b])
                hashes[k] = mask_sha256(masks[k])
    del G
    runp = np.load(RUNP_OUT / name / f"rep{rep}" / "masks.npz")
    for b in BUDGETS:  # determinism across runs: the 512-image absolute-gradient mask must equal Run P's
        assert (unpack(runp[f"abs512@{b}"], model.n) == masks[f"abs512@{b}"]).all(), f"abs512@{b} differs from Run P"
    np.savez(out / "masks.npz", **{k: pack(v) for k, v in masks.items()})
    (out / "masks.json").write_text(json.dumps(dict(hashes=hashes, build_seconds=round(time.time() - t0)), indent=0))
    bundle = bundle_dir(spec["bundle"])
    schedules = [s for s in json.loads((bundle / "artifacts/fault_schedules.json").read_text())[f"replication_{rep}"]
                 if any(abs(float(s["logical_ber"]) - p) < 1e-12 for p in BERS)]
    saved_rows = {}
    with open(bundle / "measurements/trials.csv") as f:
        for r in csv.DictReader(f):
            if r["replication"] == f"replication_{rep}" and r["policy"].endswith(f"/{RANKINGS['abs_grad']}@0.125000"):
                saved_rows[r["schedule_sha256"]] = (float(r["accuracy"]), float(r["categorical_crossentropy"]), r["trial"])
    xe, ye, _ = model.images("evaluation", 1000, 1000 * rep)
    rp = Replayer(model)
    check_mask = unpack(saved[f"r{rep}@0.125000"], model.n)
    rows, mismatches = [], 0
    for s in schedules:
        acc, ce, prop, app = rp.evaluate(rp.flips(s["events"]), check_mask, xe, ye)
        sa, sce, trial = saved_rows[s["sha256"]]
        ok = acc == sa and abs(ce - sce) <= 1e-9 * max(1.0, abs(sce))
        mismatches += not ok
        rows.append(dict(replication=rep, schedule_sha256=s["sha256"], ber=s["logical_ber"], trial=int(trial),
                         arm="replay_check_absgrad", budget=0.125, accuracy=acc, cross_entropy=ce, applied=app, match=ok))
    if mismatches:
        (out / "replay_check_FAILED.json").write_text(json.dumps(rows, indent=0))
        raise SystemExit(f"replay check failed on {mismatches} schedules")
    print(json.dumps(dict(replay_check="passed", schedules=len(schedules))), flush=True)
    for s in schedules:
        fl = rp.flips(s["events"])
        trial = int(saved_rows[s["sha256"]][2])
        for k in masks:
            if k.startswith("abs512@"):  # already replayed in Run P on the same schedules
                continue
            arm, b = k.split("@")
            acc, ce, prop, app = rp.evaluate(fl, masks[k], xe, ye)
            rows.append(dict(replication=rep, schedule_sha256=s["sha256"], ber=float(s["logical_ber"]), trial=trial,
                             arm=arm, budget=float(b), accuracy=acc, cross_entropy=ce, applied=app, match=""))
    with open(out / "replay.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (out / "DONE").write_text(json.dumps(dict(seconds=round(time.time() - t0), script_sha256=sha256_file(Path(__file__)))))
    print(json.dumps(dict(job_done=f"{name}/rep{rep}", seconds=round(time.time() - t0))), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("check", "job"))
    ap.add_argument("--checkpoint", required=True, choices=sorted(CHECKPOINTS))
    ap.add_argument("--rep", type=int)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    if a.stage == "check":
        stage_check(a.checkpoint, a.threads)
    else:
        stage_job(a.checkpoint, a.rep, a.threads)


if __name__ == "__main__":
    main()
