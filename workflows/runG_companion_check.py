#!/usr/bin/env python3
"""Gradient-trust plan (review/GRADIENT_TRUST_PLAN_2026-11-15.md), validation of the graph-walking companion.

On the two models whose pinned, hand-written companions are already validated (DS-CNN; pinned ResNet-8), the
graph-walking companion built from the same TFLite bytes must give the same kernels, the same forward pass (logits and
encoded outputs) and the same per-example gradients.  Also reports agreement of its encoded outputs with the TFLite
reference interpreter.  CPU only.  Writes <out>/companion_check.json.

    runG_companion_check.py <out> [--count 64] [--grad-count 16]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def compare(m, x, y, gx, gy, label):
    from netsurf.tflite_graph_companion import TFLiteGraphCompanion
    t0 = time.time()
    g = TFLiteGraphCompanion(m.adapter._require_loaded().reference_content)
    if set(g.variables) != set(m.comp.variables):
        raise SystemExit(f"{label}: kernel names differ")
    kern = max(float(np.abs(g.variables[n].numpy() - m.comp.variables[n].numpy()).max()) for n in g.variables)
    m.set_displacement(np.zeros(m.n))
    lp, pp = m.comp.forward(x)
    lg, pg = g.forward(x)
    if hasattr(m.comp, "encoded_outputs"):
        enc_p = m.comp.encoded_outputs(x)
    else:
        sc, zp = g._qparams(g.outputs[0])
        enc_p = np.clip(np.rint(pp.numpy() / sc + zp), -128, 127).astype(np.int8)
    enc_g = g.encoded_outputs(x)
    ref = m.adapter._outputs(x)
    pe_p = m.comp.per_example_gradients(gx, gy)
    pe_g = g.per_example_gradients(gx, gy)
    gmax = max(float(np.abs(pe_g[n] - pe_p[n]).max()) for n in pe_g)
    gscale = max(float(np.abs(pe_p[n]).max()) for n in pe_p)
    return dict(model=label, operators=sorted(set(o["name"] for o in g.ops)), kernels=len(g.variables),
                kernel_max_abs_diff=kern, logits_max_abs_diff=float(np.abs(lg.numpy() - lp.numpy()).max()),
                encoded_outputs_equal_to_pinned=float(np.mean(enc_g == enc_p)),
                argmax_equal_to_pinned=float(np.mean(lg.numpy().argmax(1) == lp.numpy().argmax(1))),
                encoded_outputs_equal_to_tflite=float(np.mean(enc_g == ref)),
                argmax_equal_to_tflite=float(np.mean(enc_g.argmax(1) == ref.argmax(1))),
                per_example_gradient_max_abs_diff=gmax, per_example_gradient_max_abs=gscale,
                inputs=len(y), gradient_inputs=len(gy), seconds=round(time.time() - t0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--count", type=int, default=64)
    ap.add_argument("--grad-count", type=int, default=16)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        raise SystemExit('run on CPU: CUDA_VISIBLE_DEVICES=""')
    a.out.mkdir(parents=True, exist_ok=False)
    rows = []
    from runR_kws import KwsModel
    m = KwsModel(a.threads)
    x, y, _ = m.clips("evaluation", a.count, 0)
    rows.append(compare(m, np.asarray(x), np.asarray(y), np.asarray(x[:a.grad_count]), np.asarray(y[:a.grad_count]), "kws-dscnn"))
    print(json.dumps(rows[-1]), flush=True)
    from runP_phase0 import Model
    r = Model("pinned", a.threads)
    xi, yi, _ = r.images("evaluation", a.count, 0)
    rows.append(compare(r, np.asarray(xi), np.asarray(yi), np.asarray(xi[:a.grad_count]), np.asarray(yi[:a.grad_count]), "resnet8-pinned"))
    print(json.dumps(rows[-1]), flush=True)
    (a.out / "companion_check.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
