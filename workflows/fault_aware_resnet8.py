#!/usr/bin/env python3
"""Run M (frozen before any outcome): the fault-rate-aware rebuild of Run F on the MLPerf Tiny ResNet-8, through
the authoritative TFLite BUILTIN_REF executable.

Rebuilt mask ("conditional absolute gradient", candidate C1 of Run F, unchanged): start from the engine's
absolute-gradient mask of Run A replication 2 (calibration: CIFAR-10 training images 2000-2063); twice, draw
four fault fields at the target rate p, apply the faults the current mask leaves unprotected, compute the
per-example gradients of the engine's differentiable companion on each faulted model for the same 64
calibration images, and re-rank every kernel bit by mean_x |dL_x/dw| * |exact XOR jump at the fault-free code|;
protect the top B bits.  The signed two-moment candidate of Run F failed there and is not repeated.

Gate: the score on the fault-free model must reproduce the engine's sealed mask (differing bits reported; the
build stops if more than 0.1% of the budget differs).

Targets: p in {3e-3, 1e-2, 3e-2} x B in {12.5%, 25%} (the cells in which the fixed mask loses more than two
points on the Run A data).  Evaluation: test images 2000-2999 (Run A replication 2's window, used before),
16 new fault fields per rate shared by all masks; each rebuilt mask and the fixed mask of the same budget at
all three rates.  Primary contrast: rebuilt minus fixed at the target cell, paired over fields; a gain matters
if it is at least one accuracy point with a 95% interval excluding zero.  Exploratory; one checkpoint.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ENGINE = Path("/projects/netsurf/manuelbv/workspace/implementation-worktrees/paper1-runA-tflite-confirm")
RUN_A = ENGINE / "workspace/runs/paper1_runA_resnet8_confirm_v1"
CHECKPOINT = Path("/data/models/models--mlcommons--tiny/blobs/3c002613d1b2475eb51dd78dfb85a546c8ae658dee71cf6ade43b022fe205415")
DATASET = Path("/data/datasets/cifar10/cifar-10-python.tar.gz")
LOCK = Path("/projects/kona-assets/manuelbv/locks/mlperf-tiny/ic.lock.json")
TARGETS = (3e-3, 1e-2, 3e-2)
FRACTIONS = ("0.125000", "0.250000")
CELLS = tuple((p, f) for p in TARGETS for f in FRACTIONS)
REP, OFFSET, CAL, EVAL = 2, 2000, 64, 1000
T_ITER, K_FIELDS, EVAL_FIELDS = 2, 4, 16
BUILD_SEED, EVAL_SEED = 20261119, 20261120


def extract(out: Path) -> None:
    bundle = next(d for d in RUN_A.iterdir() if (d / "FINALIZED").is_file())
    data = json.loads((bundle / "artifacts/protection_masks.json").read_text())[f"replication_{REP}"]
    masks = {f: data[f"per_example_absolute_global@{f}"] for f in FRACTIONS}
    del data
    (out / f"fixed_masks_replication_{REP}.json").write_text(json.dumps(masks))
    print({k: len(v) for k, v in masks.items()}, flush=True)


def setup():
    sys.path.insert(0, str(ENGINE))
    import tensorflow as tf
    import os
    tf.config.threading.set_intra_op_parallelism_threads(int(os.environ.get("NETSURF_TF_THREADS", "4")))
    from netsurf.adapters.mlperf_tiny import MlperfTinyImageClassificationAdapter
    from netsurf.mlperf_companion import MlperfResnet8DifferentiableCompanion
    adapter = MlperfTinyImageClassificationAdapter(CHECKPOINT, DATASET, asset_lock=LOCK, split="test", injectable_parameters=("kernels",))
    adapter.load()
    return adapter, MlperfResnet8DifferentiableCompanion


def flat(adapter):
    storage = adapter._require_loaded()
    values = storage.values()
    names = list(values)
    sizes = [values[n].size for n in names]
    starts = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    jumps = adapter.bit_perturbations()                                    # parameter -> (*shape, 8) decoded jumps
    absj = np.concatenate([np.abs(jumps[n]).reshape(-1, 8) for n in names]).astype(np.float64)
    codes = np.concatenate([np.ascontiguousarray(values[n]).view(np.uint8).reshape(-1) for n in names])
    return storage, values, names, starts, absj, codes


def mask_from_list(items, names, starts):
    index = {n: i for i, n in enumerate(names)}
    m = np.zeros((int(starts[-1]), 8), dtype=bool)
    for parameter, flat_index, bit in items:
        m[starts[index[parameter]] + flat_index, bit] = True
    return m


def top(score, budget):
    s = score.reshape(-1); n = s.size
    thr = np.partition(s, n - budget)[n - budget]
    sel = s > thr; need = budget - int(sel.sum())
    if need:
        sel[np.flatnonzero(s == thr)[:need]] = True
    return sel.reshape(score.shape)


def to_values(codes_u8, values, names, starts):
    return {n: codes_u8[starts[i]: starts[i + 1]].view(np.int8).reshape(values[n].shape) for i, n in enumerate(names)}


def faulted_codes(rng, codes, protected, p):
    flips = (rng.random(protected.shape) < p) & ~protected
    return codes ^ np.packbits(flips, axis=1, bitorder="little")[:, 0]


def mean_abs_grad(companion_cls, storage, codes_u8, values, names, starts, x, y):
    comp = companion_cls(storage.model_content_from_values(to_values(codes_u8, values, names, starts)))
    g = comp.per_example_gradients(x, y)
    return np.concatenate([np.mean(np.abs(np.asarray(g[n]).reshape(len(y), -1)), axis=0) for n in names]).astype(np.float64)


def build(out: Path, cell: int) -> None:
    p, frac = CELLS[cell]
    dest = out / "build" / f"p{p:g}_f{float(frac):g}"
    dest.mkdir(parents=True, exist_ok=False)
    adapter, Companion = setup()
    storage, values, names, starts, absj, codes = flat(adapter)
    x, y, part = adapter.sample_partition("ranking", CAL, offset=OFFSET)
    assert part.indices[0] == OFFSET
    fixed = mask_from_list(json.loads((out / f"fixed_masks_replication_{REP}.json").read_text())[frac], names, starts)
    budget = int(fixed.sum())
    started = time.time()
    g0 = mean_abs_grad(Companion, storage, codes, values, names, starts, x, y)
    clean_rank = top(g0[:, None] * absj, budget)
    differing = int(np.count_nonzero(clean_rank != fixed)) // 2
    assert differing <= 0.001 * budget, f"gate failed: {differing} bits differ from the engine mask"
    current, trace = fixed.copy(), []
    for t in range(T_ITER):
        acc = np.zeros_like(g0)
        for k in range(K_FIELDS):
            rng = np.random.default_rng(BUILD_SEED + 1000 * cell + 10 * t + k)
            acc += mean_abs_grad(Companion, storage, faulted_codes(rng, codes, current, p), values, names, starts, x, y)
        new = top((acc / K_FIELDS)[:, None] * absj, budget)
        trace.append(dict(iteration=t + 1, changed_from_previous=int(np.count_nonzero(new != current)) // 2,
                          changed_from_fixed=int(np.count_nonzero(new != fixed)) // 2))
        current = new
        print(json.dumps(dict(p=p, fraction=frac, **trace[-1], seconds=round(time.time() - started))), flush=True)
    np.save(dest / "rebuilt_mask.npy", np.packbits(current, axis=1, bitorder="little")[:, 0])
    (dest / "manifest.json").write_text(json.dumps(dict(
        schema="paper1-runM-build/1.0", status="EXPLORATORY", target_ber=p, fraction=float(frac), budget=budget,
        gate_bits_differing_from_engine_mask=differing, trace=trace, iterations=T_ITER, fields_per_iteration=K_FIELDS,
        calibration_indices=[part.indices[0], part.indices[-1]],
        protected_fraction_by_bit_fixed=[float(v) for v in fixed.mean(axis=0)],
        protected_fraction_by_bit_rebuilt=[float(v) for v in current.mean(axis=0)], seconds=time.time() - started), indent=1))
    evaluate(out, cell, adapter, storage, values, names, starts, codes, fixed, current)


def evaluate(out, cell, adapter, storage, values, names, starts, codes, fixed, rebuilt):
    p_t, frac = CELLS[cell]
    x, y, part = adapter.sample_partition("evaluation", EVAL, offset=OFFSET)
    clean = adapter.evaluate(x, y).accuracy
    rows = []
    for a, p in enumerate(TARGETS):
        for i in range(EVAL_FIELDS):
            rng = np.random.default_rng(EVAL_SEED + 1000 * a + i)            # the same field for every cell and mask
            draws = rng.random(fixed.shape) < p
            for name, m in (("fixed", fixed), ("rebuilt", rebuilt)):
                fc = codes ^ np.packbits(draws & ~m, axis=1, bitorder="little")[:, 0]
                with adapter.temporary_values(to_values(fc, values, names, starts)):
                    acc = adapter.evaluate(x, y).accuracy
                rows.append(dict(mask=name, target_ber=p_t, fraction=float(frac), test_ber=p, field=i, accuracy=acc))
        print(json.dumps(dict(cell=cell, test_ber=p, done=True)), flush=True)
    (out / "build" / f"p{p_t:g}_f{float(frac):g}" / "evaluation.json").write_text(json.dumps(dict(
        clean_accuracy=clean, window=[part.indices[0], part.indices[-1]], rows=rows), indent=1))


def analyze(out: Path) -> None:
    from scipy import stats
    res = []
    for p_t, frac in CELLS:
        d = json.loads((out / "build" / f"p{p_t:g}_f{float(frac):g}" / "evaluation.json").read_text())
        for p in TARGETS:
            f = np.array([r["accuracy"] for r in d["rows"] if r["mask"] == "fixed" and r["test_ber"] == p])
            r_ = np.array([r["accuracy"] for r in d["rows"] if r["mask"] == "rebuilt" and r["test_ber"] == p])
            diff = r_ - f; m = float(diff.mean()); h = float(stats.t.ppf(0.975, len(diff) - 1) * diff.std(ddof=1) / math.sqrt(len(diff))) if diff.std() > 0 else 0.0
            res.append(dict(target_ber=p_t, fraction=float(frac), test_ber=p, on_target=p == p_t, clean=d["clean_accuracy"],
                            fixed=float(f.mean()), rebuilt=float(r_.mean()), gain=m, ci=[m - h, m + h],
                            matters=bool(p == p_t and m >= 0.01 and m - h > 0)))
    (Path(str(out) + "_analysis.json")).write_text(json.dumps(dict(schema="paper1-runM-analysis/1.0", status="EXPLORATORY", rows=res), indent=1))
    for r in res:
        print(f"built for {r['target_ber']:<6g} B={r['fraction']:<6g} tested at {r['test_ber']:<6g} fixed {100*r['fixed']:5.1f} rebuilt {100*r['rebuilt']:5.1f} "
              f"gain {100*r['gain']:+6.2f} [{100*r['ci'][0]:+.2f},{100*r['ci'][1]:+.2f}]{'  *' if r['on_target'] else ''}{'  matters' if r['matters'] else ''}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["extract", "build", "analyze"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cell", type=int, default=0)
    a = ap.parse_args()
    if a.stage == "extract":
        a.out.mkdir(parents=True, exist_ok=False); extract(a.out)
    elif a.stage == "build":
        build(a.out, a.cell)
    else:
        analyze(a.out)


if __name__ == "__main__":
    main()
