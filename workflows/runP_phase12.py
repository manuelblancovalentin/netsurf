#!/usr/bin/env python3
"""Run P, Phases 1 and 2 (plan v2, `RUN_P_MEAN_VARIANCE_PLAN_V2_2026-11-11.md`, with version 1 where unchanged).

Proposed method code name: mean_variance_mask (not qPolarGrad).

  prep   --checkpoint C            Extract, once per checkpoint, the saved absolute-gradient masks of every replication
                                   from the finalized bundle into a packed npz.
  job    --checkpoint C --rep R    Phase 1 for one replication (masks from 512 calibration images: absolute gradient,
                                   squared-effect ranking, range-W mean-variance mask, single-rate mean-variance masks
                                   at the 8 grid BERs, 4 budgets; objective of each on the 512 held-out images), then
                                   Phase 2 for that replication: replay check of the saved absolute-gradient mask on the
                                   saved fault schedules against trials.csv (exact), then every new mask on every saved
                                   schedule of the replication (single-rate masks only at their own BER).
  blockb                           Block B (Run N protocol): pinned ResNet-8, test images 2000-2999, the 16 Run M fault
                                   fields per rate, rates 3e-3, 1e-2, 3e-2, budgets 12.5% and 25%; single-rate masks and
                                   a range-N mask from training images 2000-2511; replay check against Run N's saved
                                   absolute-gradient outcomes.

No asset lock is changed.  Every output directory is new (exist_ok=False).
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from runP_phase0 import (BUDGETS, CHECKPOINTS, HELDOUT_COUNT, HELDOUT_OFFSET, PAPER, RANGE_W, RANKINGS, ROOT, Model,
                         bundle_dir, load_masks, sha256_file)
from netsurf.mean_variance import (bit_second_moments, displacement_moments, greedy_mean_variance_mask,
                                   log_uniform_rates, mask_sha256, top_budget, variance_term)

OUT = PAPER / "workspace/runs/paper1_runP_phase12_v1"
GRID = (1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
RANGE_N = log_uniform_rates(3e-3, 3e-2, 8)
N_CAL = 512
RUN_N = PAPER / "workspace/runs/paper1_runN_method_campaigns_resnet8_v1"
RUN_M = PAPER / "workspace/runs/paper1_runM_fault_aware_resnet8_v1"
B_TARGETS, B_FRACTIONS, B_FIELDS, B_EVAL_SEED, B_REP = (3e-3, 1e-2, 3e-2), ("0.125000", "0.250000"), 16, 20261120, 2


def pack(mask):
    return np.packbits(np.asarray(mask, dtype=bool).reshape(-1), bitorder="little")


def unpack(packed, n):
    return np.unpackbits(packed, bitorder="little")[: n * 8].astype(bool).reshape(n, 8)


def stage_prep(name, threads):
    out = OUT / name
    out.mkdir(parents=True, exist_ok=False)
    model = Model(name, 1)
    spec = CHECKPOINTS[name]
    masks, _ = load_masks(model, bundle_dir(spec["bundle"]), spec["reps"])
    arrays = {f"r{rep}@{b}": pack(masks[rep][f"{RANKINGS['abs_grad']}@{b}"]) for rep in spec["reps"] for b in BUDGETS}
    np.savez(out / "saved_absgrad_masks.npz", **arrays)
    (out / "prep.json").write_text(json.dumps(dict(n=model.n, names=model.names, checkpoint_sha256=model.sha256)))
    print(f"prepared {name}: {len(arrays)} masks", flush=True)


def build_masks(model, x, y, start_masks, rates_single, range_rates):
    """Phase 1 for one calibration set.  Returns {key: mask} and per-mask records (without held-out values)."""
    G = model.per_example(x, y)
    gabs, g2 = np.abs(G).mean(0), (G * G).mean(0)
    del G
    m2 = bit_second_moments(model.jumps, g2)
    model.set_displacement(np.zeros(model.n))
    clean, _ = model.loss_and_grad(x, y)
    term = model.mean_term(x, y, clean)
    abs_score, sq_score = gabs[:, None] * np.abs(model.jumps), m2
    masks, records = {}, []
    for b, budget in start_masks.items():
        a512 = top_budget(abs_score, budget)
        masks[f"abs512@{b}"] = a512
        masks[f"squared@{b}"] = top_budget(sq_score, budget)
        t0 = time.time()
        res = greedy_mean_variance_mask(a512, model.jumps, m2, range_rates, term)
        masks[f"mv_range@{b}"] = res.mask
        records.append(dict(mask=f"mv_range@{b}", calibration_start=res.initial_objective, calibration_end=res.objective,
                            rounds=len(res.trace) - 1, changed_bits=int((res.mask & ~a512).sum()), seconds=round(time.time() - t0)))
        for p in rates_single:
            t0 = time.time()
            res = greedy_mean_variance_mask(a512, model.jumps, m2, [p], term)
            masks[f"mv_single@{b}@{p:g}"] = res.mask
            records.append(dict(mask=f"mv_single@{b}@{p:g}", calibration_start=res.initial_objective,
                                calibration_end=res.objective, rounds=len(res.trace) - 1,
                                changed_bits=int((res.mask & ~a512).sum()), seconds=round(time.time() - t0)))
    model.set_displacement(np.zeros(model.n))
    return masks, records


def heldout_objectives(model, masks, rates_for):
    xh, yh, ih = model.images("empirical_validation", HELDOUT_COUNT, HELDOUT_OFFSET)
    assert ih[0] == 28000 and ih[-1] == 28511
    G = model.per_example(xh, yh)
    m2h = bit_second_moments(model.jumps, (G * G).mean(0))
    del G
    model.set_displacement(np.zeros(model.n))
    clean = model.loss_acc(xh, yh)[0]
    out = {}
    for key, mk in masks.items():
        rates = rates_for(key)
        total = 0.0
        for p in rates:
            mu, _ = displacement_moments(model.jumps, mk, p)
            model.set_displacement(mu)
            total += model.loss_acc(xh, yh)[0] - clean + variance_term(m2h, mk, p)
        out[key] = total / len(rates)
    model.set_displacement(np.zeros(model.n))
    return out


class Replayer:
    def __init__(self, model):
        self.model = model
        self.index = {n: i for i, n in enumerate(model.names)}
        self.u8 = model.codes.view(np.uint8)

    def flips(self, events):
        f = np.zeros((self.model.n, 8), dtype=bool)
        if events:
            pi = np.fromiter((self.index[e[1]] for e in events), dtype=np.int64, count=len(events))
            fi = np.fromiter((e[2] for e in events), dtype=np.int64, count=len(events))
            bi = np.fromiter((e[3] for e in events), dtype=np.int64, count=len(events))
            f[self.model.starts[pi] + fi, bi] = True
        return f

    def evaluate(self, flips, mask, x, y):
        applied = flips & ~mask
        codes = (self.u8 ^ np.packbits(applied, axis=1, bitorder="little")[:, 0]).view(np.int8)
        m = self.model
        values = {n: codes[m.starts[i]: m.starts[i + 1]].reshape(m.shapes[n]) for i, n in enumerate(m.names)}
        with m.adapter.temporary_values(values):
            r = m.adapter.evaluate(x, y)
        return float(r.accuracy), float(r.categorical_crossentropy), int(flips.sum()), int(applied.sum())


def stage_job(name, rep, threads):
    spec = CHECKPOINTS[name]
    out = OUT / name / f"rep{rep}"
    out.mkdir(parents=True, exist_ok=False)
    model = Model(name, threads)
    prep = json.loads((OUT / name / "prep.json").read_text())
    assert prep["n"] == model.n and prep["names"] == model.names and prep["checkpoint_sha256"] == model.sha256
    saved = np.load(OUT / name / "saved_absgrad_masks.npz")
    start = {b: int(unpack(saved[f"r{rep}@{b}"], model.n).sum()) for b in BUDGETS}
    # Phase 1
    t0 = time.time()
    x, y, idx = model.images("ranking", N_CAL, 1000 * rep)
    assert idx[0] == 1000 * rep and len(idx) == N_CAL
    masks, records = build_masks(model, x, y, start, GRID, RANGE_W)
    ho = heldout_objectives(model, masks, lambda k: RANGE_W if (k.startswith("mv_range") or k.split("@")[0] in ("abs512", "squared")) else [float(k.split("@")[2])])
    for r in records:
        r["heldout"] = ho[r["mask"]]
    for b in BUDGETS:
        records.append(dict(mask=f"abs512@{b}", heldout=ho[f"abs512@{b}"]))
        records.append(dict(mask=f"squared@{b}", heldout=ho[f"squared@{b}"]))
    np.savez(out / "masks.npz", **{k: pack(v) for k, v in masks.items()})
    (out / "phase1.json").write_text(json.dumps(dict(records=records, seconds=round(time.time() - t0),
                                                     hashes={k: mask_sha256(v) for k, v in masks.items()}), indent=0))
    print(json.dumps(dict(phase1_done=rep, seconds=round(time.time() - t0))), flush=True)
    # Phase 2: replay check, then new masks
    bundle = bundle_dir(spec["bundle"])
    schedules = json.loads((bundle / "artifacts/fault_schedules.json").read_text())[f"replication_{rep}"]
    saved_rows = {}
    with open(bundle / "measurements/trials.csv") as f:
        for r in csv.DictReader(f):
            if r["replication"] == f"replication_{rep}" and r["policy"].endswith(f"/{RANKINGS['abs_grad']}@0.125000"):
                saved_rows[r["schedule_sha256"]] = (float(r["accuracy"]), float(r["categorical_crossentropy"]), r["trial"])
    xe, ye, ie = model.images("evaluation", 1000, 1000 * rep)
    rp = Replayer(model)
    check_mask = unpack(saved["r%d@0.125000" % rep], model.n)
    rows, mismatches = [], 0
    for s in schedules:
        fl = rp.flips(s["events"])
        acc, ce, prop, app = rp.evaluate(fl, check_mask, xe, ye)
        sa, sce, trial = saved_rows[s["sha256"]]
        ok = acc == sa and abs(ce - sce) <= 1e-9 * max(1.0, abs(sce))
        mismatches += not ok
        rows.append(dict(replication=rep, schedule_sha256=s["sha256"], ber=s["logical_ber"], trial=int(trial),
                         arm="replay_check_absgrad", budget=0.125, accuracy=acc, cross_entropy=ce, proposed=prop,
                         applied=app, saved_accuracy=sa, saved_cross_entropy=sce, match=ok))
    if mismatches:
        (out / "replay_check_FAILED.json").write_text(json.dumps(rows, indent=0))
        raise SystemExit(f"replay check failed on {mismatches} schedules")
    print(json.dumps(dict(replay_check="passed", schedules=len(schedules))), flush=True)
    for s in schedules:
        fl = rp.flips(s["events"])
        p = float(s["logical_ber"])
        trial = int(saved_rows[s["sha256"]][2])
        for b in BUDGETS:
            for arm, key in (("abs512", f"abs512@{b}"), ("squared", f"squared@{b}"), ("mv_range", f"mv_range@{b}"),
                             ("mv_single", f"mv_single@{b}@{p:g}")):
                acc, ce, prop, app = rp.evaluate(fl, masks[key], xe, ye)
                rows.append(dict(replication=rep, schedule_sha256=s["sha256"], ber=p, trial=trial, arm=arm,
                                 budget=float(b), accuracy=acc, cross_entropy=ce, proposed=prop, applied=app,
                                 saved_accuracy="", saved_cross_entropy="", match=""))
    with open(out / "replay.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (out / "DONE").write_text(json.dumps(dict(seconds=round(time.time() - t0), script_sha256=sha256_file(Path(__file__)),
                                              module_sha256=sha256_file(ROOT / "netsurf/mean_variance.py"))))
    print(json.dumps(dict(job_done=f"{name}/rep{rep}", seconds=round(time.time() - t0))), flush=True)


def stage_check(name, rep, count=4):
    """Quick replay check before any mask is built: saved absolute-gradient mask at 12.5% on the first schedules."""
    spec = CHECKPOINTS[name]
    model = Model(name, 1)
    saved = np.load(OUT / name / "saved_absgrad_masks.npz")
    mask = unpack(saved["r%d@0.125000" % rep], model.n)
    bundle = bundle_dir(spec["bundle"])
    schedules = json.loads((bundle / "artifacts/fault_schedules.json").read_text())[f"replication_{rep}"]
    rows = {}
    with open(bundle / "measurements/trials.csv") as f:
        for r in csv.DictReader(f):
            if r["replication"] == f"replication_{rep}" and r["policy"].endswith(f"/{RANKINGS['abs_grad']}@0.125000"):
                rows[r["schedule_sha256"]] = (float(r["accuracy"]), float(r["categorical_crossentropy"]))
    xe, ye, _ = model.images("evaluation", 1000, 1000 * rep)
    rp = Replayer(model)
    picks = [schedules[i] for i in np.linspace(0, len(schedules) - 1, count).astype(int)]
    for s in picks:
        t0 = time.time()
        acc, ce, prop, app = rp.evaluate(rp.flips(s["events"]), mask, xe, ye)
        print(json.dumps(dict(ber=s["logical_ber"], replay=[acc, ce], saved=rows[s["sha256"]],
                              match=acc == rows[s["sha256"]][0] and abs(ce - rows[s["sha256"]][1]) <= 1e-9,
                              applied=app, seconds=round(time.time() - t0, 1))), flush=True)


def stage_blockb(threads):
    out = OUT / "blockb"
    out.mkdir(parents=True, exist_ok=False)
    model = Model("pinned", threads)
    fixed = json.loads((RUN_M / f"fixed_masks_replication_{B_REP}.json").read_text())
    index = {n: i for i, n in enumerate(model.names)}

    def from_list(items):
        m = np.zeros((model.n, 8), dtype=bool)
        for parameter, flat_index, bit in items:
            m[model.starts[index[parameter]] + flat_index, bit] = True
        return m

    engine_abs = {f: from_list(fixed[f]) for f in B_FRACTIONS}
    x, y, idx = model.images("ranking", N_CAL, 1000 * B_REP)
    assert idx[0] == 2000
    G = model.per_example(x, y)
    gabs, g2 = np.abs(G).mean(0), (G * G).mean(0)
    del G
    m2 = bit_second_moments(model.jumps, g2)
    model.set_displacement(np.zeros(model.n))
    clean, _ = model.loss_and_grad(x, y)
    term = model.mean_term(x, y, clean)
    masks = {}
    for f in B_FRACTIONS:
        a512 = top_budget(gabs[:, None] * np.abs(model.jumps), int(engine_abs[f].sum()))
        masks[f"abs512@{f}"] = a512
        masks[f"mv_rangeN@{f}"] = greedy_mean_variance_mask(a512, model.jumps, m2, RANGE_N, term).mask
        for p in B_TARGETS:
            masks[f"mv_single@{f}@{p:g}"] = greedy_mean_variance_mask(a512, model.jumps, m2, [p], term).mask
    model.set_displacement(np.zeros(model.n))
    np.savez(out / "masks.npz", **{k: pack(v) for k, v in masks.items()})
    saved = {}
    for sf in sorted((RUN_N / "eval").glob("shard_*.json")):
        for r in json.loads(sf.read_text()):
            if r["method"] == "abs_grad_fixed":
                saved[(f"{r['fraction']:.6f}", r["test_ber"], r["field"])] = r["accuracy"]
    xe, ye, ie = model.images("evaluation", 1000, 1000 * B_REP)
    rp = Replayer(model)
    rows = []
    for a, p in enumerate(B_TARGETS):
        for i in range(B_FIELDS):
            draws = np.random.default_rng(B_EVAL_SEED + 1000 * a + i).random((model.n, 8)) < p
            for f in B_FRACTIONS:
                acc, ce, prop, app = rp.evaluate(draws, engine_abs[f], xe, ye)
                if acc != saved[(f, p, i)]:
                    raise SystemExit(f"Block B replay check failed at {f} {p} {i}: {acc} vs {saved[(f, p, i)]}")
                for key in (f"abs512@{f}", f"mv_rangeN@{f}", f"mv_single@{f}@{p:g}"):
                    acc, ce, prop, app = rp.evaluate(draws, masks[key], xe, ye)
                    rows.append(dict(test_ber=p, field=i, fraction=float(f), arm=key.split("@")[0], accuracy=acc,
                                     cross_entropy=ce, engine_abs_accuracy=saved[(f, p, i)]))
    with open(out / "blockb.csv", "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (out / "DONE").write_text("ok")
    print("blockb done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prep", "check", "job", "blockb"])
    ap.add_argument("--checkpoint", choices=sorted(CHECKPOINTS))
    ap.add_argument("--rep", type=int)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    if a.stage == "prep":
        stage_prep(a.checkpoint, a.threads)
    elif a.stage == "check":
        stage_check(a.checkpoint, a.rep)
    elif a.stage == "job":
        stage_job(a.checkpoint, a.rep, a.threads)
    else:
        stage_blockb(a.threads)


if __name__ == "__main__":
    main()
