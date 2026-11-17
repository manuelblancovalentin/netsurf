#!/usr/bin/env python3
"""Exact companions of event_background_check.py: no fault is drawn.

For each CONV_2D layer of the reference ResNet-8 and each mask of the check, from the stored codes, the mask and the
same clean layer inputs:
  * the number of unprotected top bits (6-7) and low bits (0-5) among one neuron's weights, averaged over the output
    channels; the expected number of flips at rate p is that number times p;
  * the variance fraction of the events, Var(T) / Var(T + L) pooled over neurons, where T and L are the parts of the
    change of a pre-activation due to top and low bits.  T and L are independent, so Var(T + L) = Var(T) + Var(L), and
    both variances are proportional to p(1 - p), so the fraction does not depend on p.
Validation built in: without protection every weight has the fraction (64^2 + 128^2) / sum_j 4^j = 20480/21845, so the
pooled fraction of every layer must equal it; the script stops otherwise.

(The check's `event_share_*` columns are E[T^2] / (E[T^2] + E[L^2]), a ratio of the two parts' second moments.  It is
not a fraction of E[(T + L)^2], which also contains 2 E[T] E[L].)

    event_background_exact.py --out DIR [--inputs 8] [--threads 4]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from netsurf import event_background as E  # noqa: E402
from event_background_check import MASKS, REP, layers_of, real  # noqa: E402

ALL_BITS = (64 ** 2 + 128 ** 2) / sum(4 ** j for j in range(8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--inputs", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    t0 = time.time()
    import tensorflow as tf
    from runT_engine import load
    from runR_stage1 import bkey
    from runP_phase12 import unpack

    m, saved, (xe, _), _, _, cid = load("pinned", REP, a.threads)
    a.out.mkdir(parents=True, exist_ok=False)
    protected = {"unprotected": np.zeros((m.n, 8), bool), "abs@0.125": unpack(saved[f"abs1024@{bkey(0.125)}"], m.n)}
    it = tf.lite.Interpreter(model_content=bytes(m.adapter._model_content), experimental_delegates=[], num_threads=a.threads,
                             experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
                             experimental_preserve_all_tensors=True)
    it.allocate_tensors()
    layers, det = layers_of(it)
    inp = it.get_input_details()[0]
    xs = {L["x"]: [] for L in layers}
    for i in range(a.inputs):
        it.set_tensor(inp["index"], np.asarray(xe[i:i + 1]).reshape(inp["shape"]))
        it.invoke()
        for k in xs:
            xs[k].append(real(it, det, k)[0])
    row = {n: (int(m.starts[i]), int(m.starts[i + 1])) for i, n in enumerate(m.names)}
    out = []
    for li, L in enumerate(layers):
        s, e = row[L["name"]]
        cout, kh, kw, cin = L["shape"]
        codes = m.codes[s:e]
        wq = det[L["w"]]["quantization_parameters"]
        sw = np.asarray(wq["scales"], dtype=np.float64) if len(wq["scales"]) == cout else np.full(cout, float(wq["scales"][0]))
        x2 = np.stack(xs[L["x"]]) ** 2
        strides = [1, L["stride"], L["stride"], 1]
        for mask in MASKS:
            unprot = ~protected[mask][s:e]
            per = unprot.reshape(cout, -1, 8)
            v = {}
            for part, bits in (("low", E.LOW_BITS), ("top", E.TOP_BITS)):
                var = E.moments(codes, unprot, 0.5, bits)[1] * 4.0                # sum of squared jumps; p(1-p) cancels
                f = np.transpose(var.reshape(L["shape"]) * (sw ** 2)[:, None, None, None], (1, 2, 3, 0))
                v[part] = float(tf.nn.conv2d(tf.constant(x2, tf.float64), tf.constant(f, tf.float64), strides, "SAME").numpy().sum())
            frac = v["top"] / (v["top"] + v["low"])
            if mask == "unprotected" and abs(frac - ALL_BITS) > 1e-9:
                raise SystemExit(f"layer {li}: unprotected variance fraction {frac} differs from {ALL_BITS}")
            out.append(dict(layer=li, kernel=L["name"], fan_in=kh * kw * cin, mask=mask,
                            unprotected_top_bits_per_neuron=float(per[..., list(E.TOP_BITS)].sum((1, 2)).mean()),
                            unprotected_low_bits_per_neuron=float(per[..., list(E.LOW_BITS)].sum((1, 2)).mean()),
                            event_variance_fraction=frac))
            print(json.dumps(dict(out[-1], seconds=round(time.time() - t0))), flush=True)
    body = json.dumps(out, indent=1)
    (a.out / "exact.json").write_text(body)
    (a.out / "DONE").write_text(json.dumps(dict(seconds=round(time.time() - t0), checkpoint=cid, rep=REP, inputs=a.inputs,
                                                exact_sha256=hashlib.sha256(body.encode()).hexdigest(),
                                                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())))


if __name__ == "__main__":
    main()
