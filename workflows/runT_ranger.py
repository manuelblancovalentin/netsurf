#!/usr/bin/env python3
"""Ranger at the pooled features, on the fault fields of the main comparison (Run T), ResNet-8 and DS-CNN.

Ranger (Chen et al., DSN 2021) clips activation and pooling outputs to the largest value observed on profiling data.
In the INT8 executables the activation layers already saturate at about that bound (ranger_bounds_check.py); the pooled
features before the classifier do not (observed maximum about a fifth of the representable range).  This script
re-creates, bit for bit, the faulted weights of selected Run T conditions (same masks, seeds and fault fields as
runT_engine.py), runs the TFLite executable with every tensor preserved, and scores each input three ways:
  tflite      the executable's own prediction (must equal the Run T accuracy: checked in the analysis)
  recomputed  the final fully connected layer recomputed from the pooled features (checks the recomputation)
  ranger      the same, with the pooled features clipped to Ranger's profiled bounds
Ranger is applied at the pooled features only; its small extra clipping of three ResNet-8 activation layers (bounds at
80-93% of the INT8 range) is not emulated.

    runT_ranger.py --model pinned|kws --rep R --draw D --bounds DIR [--threads T] --out DIR
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
import zlib
from pathlib import Path

import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runT_engine import build_masks, load  # noqa: E402
from runT_tflite import BERS, SEED, channel_axes, conditions  # noqa: E402

KEEP = ("unprotected", "tmr_abs@0.0625", "tmr_msb@0.0625", "tmr_fkeras@0.0625", "tmr_abs@0.125", "tmr_msb@0.125")
RATES = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)


def fc_logits(pool_codes, zx, sx, w_codes, sw, bias):
    """Dequantized logits of an INT8 fully connected layer: (x - zx) . w_c + b_c, scaled by sx * sw_c."""
    acc = (pool_codes.astype(np.int64) - zx) @ w_codes.astype(np.int64).T + bias.astype(np.int64)
    return acc.astype(np.float64) * (sx * np.asarray(sw, dtype=np.float64))


def ranger_clip(pool_codes, lo, hi):
    return np.clip(pool_codes, lo, hi).astype(np.int8)


def graph(it, pool_name):
    det = {d["index"]: d for d in it.get_tensor_details()}
    fc = [op for op in it._get_ops_details() if op["op_name"] == "FULLY_CONNECTED"]
    if len(fc) != 1:
        raise SystemExit(f"expected one FULLY_CONNECTED op, found {len(fc)}")
    xin, wt, bs = fc[0]["inputs"][:3]
    if det[xin]["name"] != pool_name:
        raise SystemExit(f"FC input is {det[xin]['name']}, bounds are for {pool_name}")
    q = det[xin]["quantization_parameters"]
    wq = det[wt]["quantization_parameters"]
    if np.any(wq["zero_points"] != 0):
        raise SystemExit("asymmetric FC weights are not handled")
    return dict(x=xin, w=wt, b=bs, zx=int(q["zero_points"][0]), sx=float(q["scales"][0]), sw=np.asarray(wq["scales"]))


def evaluate(m, xor, x, y, pool_name, lo, hi, threads):
    import tensorflow as tf
    codes = (m.codes.view(np.uint8) ^ xor).view(np.int8)
    faulted = {n: codes[m.starts[i]: m.starts[i + 1]].reshape(m.shapes[n]) for i, n in enumerate(m.names)}
    with m.adapter.temporary_values(faulted):
        content = m.adapter._model_content
    it = tf.lite.Interpreter(model_content=bytes(content), experimental_delegates=[], num_threads=threads,
                             experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
                             experimental_preserve_all_tensors=True)
    it.allocate_tensors()
    g = graph(it, pool_name)
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    W, B = it.get_tensor(g["w"]), it.get_tensor(g["b"])
    sw = g["sw"] if g["sw"].size == W.shape[0] else np.full(W.shape[0], g["sw"][0])
    pt, pr, pg = [], [], []
    for i in range(len(x)):
        it.set_tensor(inp["index"], np.asarray(x[i:i + 1]).reshape(inp["shape"]))
        it.invoke()
        pt.append(int(np.argmax(it.get_tensor(out["index"])[0])))
        pool = it.get_tensor(g["x"]).reshape(1, -1)
        pr.append(int(np.argmax(fc_logits(pool, g["zx"], g["sx"], W, sw, B)[0])))
        pg.append(int(np.argmax(fc_logits(ranger_clip(pool, lo, hi), g["zx"], g["sx"], W, sw, B)[0])))
    y = np.asarray(y)
    pt, pr, pg = map(np.asarray, (pt, pr, pg))
    return dict(accuracy_tflite=float(np.mean(pt == y)), accuracy_recomputed=float(np.mean(pr == y)),
                accuracy_ranger=float(np.mean(pg == y)), recomputed_agrees=float(np.mean(pr == pt)),
                clipped_inputs=None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["pinned", "kws"], required=True)
    ap.add_argument("--rep", type=int, required=True)
    ap.add_argument("--draw", type=int, required=True)
    ap.add_argument("--bounds", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    t0 = time.time()
    m, saved, (xe, ye), (xc, yc), (xf, yf), name = load(a.model, a.rep, a.threads)
    rows_b = list(csv.DictReader(open(a.bounds / ("resnet8" if a.model == "pinned" else "dscnn") / "bounds.csv")))
    pool = [r for r in rows_b if r["op"] == "RESHAPE"][0]
    lo, hi = int(pool["min_code"]), int(pool["max_code"])
    cid = zlib.crc32(name.encode())
    out = a.out / name / f"rep{a.rep}" / f"draw{a.draw}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "DONE").exists():
        raise SystemExit("write-once: DONE exists")
    masks, _ = build_masks(m, saved, xc, yc, xf, yf, cid, a.rep)
    axes = channel_axes(m)
    rows = []
    for p in RATES:
        bi = BERS.index(p)
        conds, raw = conditions(m, masks, axes, p, lambda purpose, *more: [SEED, cid, a.rep, a.draw, bi, purpose, *more])
        for cname in KEEP:
            xor = conds[cname][0]
            r = evaluate(m, xor, xe, ye, pool["tensor"], lo, hi, a.threads)
            r.pop("clipped_inputs")
            rows.append(dict(checkpoint=name, rep=a.rep, draw=a.draw, ber=p, condition=cname, **r))
            print(json.dumps(dict(id=name, rep=a.rep, draw=a.draw, done=len(rows), total=len(RATES) * len(KEEP),
                                  seconds=round(time.time() - t0))), flush=True)
    with open(out / "results.csv", "x", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (out / "DONE").write_text(json.dumps(dict(seconds=round(time.time() - t0), bounds=dict(tensor=pool["tensor"], lo=lo, hi=hi),
                                              masks_sha256={k: hashlib.sha256(v.tobytes()).hexdigest()[:16] for k, v in masks.items()},
                                              script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())))


if __name__ == "__main__":
    main()
