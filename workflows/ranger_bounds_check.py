#!/usr/bin/env python3
"""Ranger check (review of draft v22): would Ranger's profiled activation bounds clip further than the INT8 executable
already does?

Ranger (Chen et al., DSN 2021) clips the output of activation and following layers to the largest value observed on a
profiling sample of the training data.  In a fully INT8 TFLite model every op output is already saturated to the range
its quantization parameters can represent, and those parameters were calibrated on data when the model was converted.
For every int8 op output of the clean executable this script records the largest and smallest values reached on a
profiling sample of training inputs (Ranger's bounds) and the representable range (what the executable clips to), and
writes their ratio.  No fault is injected.

    ranger_bounds_check.py --model resnet8|dscnn --count N --out DIR
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np


def interpreter_for(content):
    import tensorflow as tf
    it = tf.lite.Interpreter(model_content=bytes(content), experimental_delegates=[], num_threads=4,
                             experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
                             experimental_preserve_all_tensors=True)
    it.allocate_tensors()
    return it


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["resnet8", "dscnn"], required=True)
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    t0 = time.time()
    if a.model == "resnet8":
        from runP_phase0 import Model
        m = Model("pinned", 4)
        x = m.images("ranking", a.count, 0)[0]
    else:
        from runR_kws import KwsModel
        m = KwsModel(4)
        x = m.clips("ranking", a.count, 0)[0]
    content = m.adapter._model_content
    it = interpreter_for(content)
    inp = it.get_input_details()[0]
    ops = it._get_ops_details()
    outs = {}
    for op in ops:
        for t in op["outputs"]:
            d = it._get_tensor_details(t, 0) if hasattr(it, "_get_tensor_details") else None
            outs[t] = op["op_name"]
    det = {d["index"]: d for d in it.get_tensor_details()}
    track = [t for t in outs if det[t]["dtype"] == np.int8 and det[t]["quantization"][0] > 0]
    hi = {t: -999 for t in track}
    lo = {t: 999 for t in track}
    x = np.asarray(x)
    if x.dtype != inp["dtype"]:
        raise SystemExit(f"input dtype {x.dtype}, interpreter expects {inp['dtype']}")
    for i in range(len(x)):
        it.set_tensor(inp["index"], x[i:i + 1].reshape(inp["shape"]))
        it.invoke()
        for t in track:
            v = it.get_tensor(t)
            hi[t] = max(hi[t], int(v.max())); lo[t] = min(lo[t], int(v.min()))
    a.out.mkdir(parents=True, exist_ok=False)
    rows = []
    for t in track:
        scale, zp = det[t]["quantization"]
        rep_hi, obs_hi = scale * (127 - zp), scale * (hi[t] - zp)
        rows.append(dict(tensor=det[t]["name"], op=outs[t], scale=scale, zero_point=zp, max_code=hi[t], min_code=lo[t],
                         representable_max=rep_hi, observed_max=obs_hi,
                         observed_over_representable=obs_hi / rep_hi if rep_hi > 0 else float("nan"),
                         post_relu=bool(zp == -128 and lo[t] == -128)))
    with open(a.out / "bounds.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    relu = [r for r in rows if r["post_relu"]]
    ratios = [r["observed_over_representable"] for r in relu]
    summary = dict(model=a.model, profiling_inputs=int(len(x)), op_outputs=len(rows), post_relu_outputs=len(relu),
                   ratio_min=float(np.min(ratios)) if ratios else None, ratio_median=float(np.median(ratios)) if ratios else None,
                   outputs_reaching_127=sum(r["max_code"] == 127 for r in rows),
                   model_sha256=hashlib.sha256(content).hexdigest(), seconds=round(time.time() - t0, 1),
                   script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    for r in relu:
        print(f"  {r['op']:22s} {r['tensor'][:50]:50s} max code {r['max_code']:4d}  observed/representable {r['observed_over_representable']:.2f}")


if __name__ == "__main__":
    main()
