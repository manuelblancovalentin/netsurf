#!/usr/bin/env python3
"""Run R, Stage 1 (plan `RUN_R_RATE_AWARE_PLAN_2026-11-12.md`, Section 4): does the formula predict the switch?

One job per (checkpoint, replication, field) (amendment R-A3).  Masks `abs1024` and `pos1024` at eight budgets are
read from `paper1_runR_masks_v1` (rebuilt with Run Q's code, equal to Run Q's at the four old budgets) and checked
against their recorded hashes.  Before any new evaluation the job replays one saved
Run Q schedule with `abs1024` at 12.5% and must reproduce the saved accuracy exactly.  Then, for each of eight fields,
one uniform number per stored kernel bit is drawn from `default_rng([20261115, ckpt, rep, field])` and a bit is
proposed at rate p when its number is below p (fields nested across rates, shared by both masks and all budgets).

  job --checkpoint C --rep R --field F [--out DIR] [--rates ...]

Output (write-once): <out>/<checkpoint>/rep<R>/field<F>/{results.csv, field.json, check.json, DONE}.
No asset lock is changed.  Interpreter: the TensorFlow 2.14.1 reference environment.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from runP_phase0 import BUDGETS, CHECKPOINTS, PAPER, ROOT, Model, bundle_dir, sha256_file
from runP_phase12 import Replayer, unpack
from runQ_calibration import OUT as RUNQ
from netsurf.mean_variance import mask_sha256

MASKS = PAPER / "workspace/runs/paper1_runR_masks_v1"
FRACTIONS = (0.01, 0.03125, 0.0625, 0.125, 0.1875, 0.25, 0.375, 0.5)


def bkey(f):
    return f"{f:.6f}"

OUT = PAPER / "workspace/runs/paper1_runR_stage1_crossover_v1"
RATES = (3e-4, 1e-3, 1.5e-3, 2e-3, 2.5e-3, 3e-3, 4e-3, 5e-3, 7e-3, 1e-2, 1.5e-2, 3e-2)
FIELDS = 8
SEED = 20261115
ARMS = ("abs1024", "pos1024")
NAMES = list(CHECKPOINTS)                       # checkpoint index = position in the order of Run Q
EXCLUDED = {("retrained_seed7102", 4)}          # amendment Q-A1


def reps_for(name):
    return [r for r in CHECKPOINTS[name]["reps"] if (name, r) not in EXCLUDED]


def flips_sha(fl):
    return hashlib.sha256(np.packbits(fl.reshape(-1), bitorder="little").tobytes()).hexdigest()


def job(name, rep, field_index, threads, out_root, rates):
    assert rep in reps_for(name), f"{name} rep{rep} is not in the Stage 1 cohort"
    assert 0 <= field_index < FIELDS
    out = out_root / name / f"rep{rep}" / f"field{field_index}"
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    model = Model(name, threads)
    packed = np.load(MASKS / name / f"rep{rep}" / "masks.npz")
    info = json.loads((MASKS / name / f"rep{rep}" / "masks.json").read_text())["masks"]
    budgets = [bkey(f) for f in FRACTIONS]
    masks = {}
    for a in ARMS:
        for b in budgets:
            mk = unpack(packed[f"{a}@{b}"], model.n)
            if mask_sha256(mk) != info[f"{a}@{b}"]["sha256"] or info[f"{a}@{b}"].get("equals_run_q") is False:
                raise SystemExit(f"mask {a}@{b} does not match its record")
            masks[(a, b)] = mk
    xe, ye, ie = model.images("evaluation", 1000, 1000 * rep)
    assert ie[0] == 1000 * rep and len(ie) == 1000
    rp = Replayer(model)
    # engine check against one saved Run Q outcome
    with open(RUNQ / name / f"rep{rep}" / "replay.csv") as f:
        ref = next(r for r in csv.DictReader(f) if r["arm"] == "abs1024" and abs(float(r["budget"]) - 0.125) < 1e-12)
    spec = CHECKPOINTS[name]
    sched = next(s for s in json.loads((bundle_dir(spec["bundle"]) / "artifacts/fault_schedules.json").read_text())
                 [f"replication_{rep}"] if s["sha256"] == ref["schedule_sha256"])
    acc, ce, _, _ = rp.evaluate(rp.flips(sched["events"]), masks[("abs1024", "0.125000")], xe, ye)
    check = dict(schedule_sha256=ref["schedule_sha256"], ber=sched["logical_ber"], replay_accuracy=acc,
                 saved_accuracy=float(ref["accuracy"]), match=acc == float(ref["accuracy"]))
    (out / "check.json").write_text(json.dumps(check, indent=0))
    if not check["match"]:
        raise SystemExit(f"engine check failed: {check}")
    none = np.zeros((model.n, 8), dtype=bool)
    clean_acc, clean_ce, _, _ = rp.evaluate(none, none, xe, ye)
    ck = NAMES.index(name)
    rows, field_log = [], []
    for fi in (field_index,):
        u = np.random.default_rng([SEED, ck, rep, fi]).random((model.n, 8))
        log = dict(field=fi, seed=[SEED, ck, rep, fi], flips_sha256={})
        for p in rates:
            fl = u < p
            log["flips_sha256"][f"{p:g}"] = flips_sha(fl)
            for b in budgets:
                for a in ARMS:
                    acc, ce, prop, app = rp.evaluate(fl, masks[(a, b)], xe, ye)
                    rows.append(dict(checkpoint=name, ckpt=ck, rep=rep, field=fi, ber=p, budget=float(b), arm=a,
                                     accuracy=acc, cross_entropy=ce, proposed=prop, applied=app))
        field_log.append(log)
        print(json.dumps(dict(job=f"{name}/rep{rep}", field_done=fi, seconds=round(time.time() - t0))), flush=True)
    with open(out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (out / "field.json").write_text(json.dumps(dict(fields=field_log, rates=list(rates), clean_accuracy=clean_acc,
                                                     clean_cross_entropy=clean_ce), indent=0))
    (out / "DONE").write_text(json.dumps(dict(seconds=round(time.time() - t0), evaluations=len(rows),
                                              script_sha256=sha256_file(Path(__file__)),
                                              module_sha256=sha256_file(ROOT / "netsurf/mean_variance.py"))))
    print(json.dumps(dict(job_done=f"{name}/rep{rep}", seconds=round(time.time() - t0))), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage", required=True)
    j = sub.add_parser("job")
    j.add_argument("--checkpoint", required=True, choices=NAMES)
    j.add_argument("--rep", type=int, required=True)
    j.add_argument("--field", type=int, required=True)
    j.add_argument("--threads", type=int, default=1)
    j.add_argument("--out", type=Path, default=OUT)
    j.add_argument("--rates", type=float, nargs="*", default=list(RATES))
    a = ap.parse_args()
    if a.out == OUT and tuple(a.rates) != RATES:
        raise SystemExit("the frozen output directory only takes the frozen rates")
    job(a.checkpoint, a.rep, a.field, a.threads, a.out, tuple(a.rates))


if __name__ == "__main__":
    main()
