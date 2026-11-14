#!/usr/bin/env python3
"""EXPLORATORY, post hoc: "exact at the flipped layer, linear downstream" against measured single flips (DS-CNN).

For every single flip measured by the blur pilot (`paper1_blur_pilot_v1/step/kws_rep0`, 80 per bit position, 10 per
layer, real loss change on 512 calibration clips), predict the loss change three ways on the same 512 clips:

  gradient     gbar * Delta                                       (from the pilot rows; one backward pass in total)
  quadratic    gbar * Delta + F * Delta^2 / 2                     (from the pilot rows)
  local        the flipped layer recomputed exactly (its convolution with the flipped kernel, bias, ReLU, rounding to
               INT8 and clipping), then the change of its output carried to the loss linearly:
               dL ~ sum G * d(post), G = dL/d(post) from one backward pass (straight-through rounding downstream).
               For the final dense layer the loss is recomputed exactly (nothing downstream to linearize).

Per flip it also records how the flipped layer's units moved: how many changed their rounded output, crossed zero
(ReLU state change) or hit the INT8 limits.  Reads only locked pilot outputs and the model; writes
<out>/rows.csv and <out>/summary.json (write-once).  Runs on the CPU in minutes.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from site_paths import WT  # noqa: E402

RUNS = WT / "paper1-tecs-revision-v1/workspace/runs"
STEP = RUNS / "paper1_blur_pilot_v1/step/kws_rep0"
OUT = RUNS / "paper1_local_exact_check_v1"
# DS-CNN forward order (MlperfKwsDscnnDifferentiableCompanion.forward): (kernel, bias, output tensor, stride, kind)
LAYERS = [(17, 3, 22, 2, "conv")]
for _k, _b, _pk, _pb, _do, _po in ((5, 4, 18, 6, 23, 24), (8, 7, 19, 9, 25, 26), (11, 10, 20, 12, 27, 28), (14, 13, 21, 15, 29, 30)):
    LAYERS += [(_k, _b, _do, 1, "depthwise"), (_pk, _pb, _po, 1, "conv")]
DENSE = (16, 1, 33)


def rank_corr(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    from runR_kws import CAL_COUNT, KwsModel
    m = KwsModel(a.threads)
    tf, comp = m.tf, m.comp
    a.out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    x, y, _ = m.clips("ranking", 512, CAL_COUNT * 0)                 # the pilot's calibration clips (rep 0)
    x, y = np.asarray(x), np.asarray(y).astype(np.int64)
    labels = tf.convert_to_tensor(y)
    m.set_displacement(np.zeros(m.n))

    def conv(values, kernel, stride, kind):
        k = tf.transpose(kernel, (1, 2, 3, 0))                      # TFLite 1HWO / OHWI -> TensorFlow layout
        if kind == "depthwise":
            return tf.nn.depthwise_conv2d(values, k, strides=(1, stride, stride, 1), padding="SAME")
        return tf.nn.conv2d(values, k, strides=(1, stride, stride, 1), padding="SAME")

    rec = {}
    with tf.GradientTape() as tape:
        v = comp._decode_input(x)
        for (k, b, o, s, kind) in LAYERS:
            pre = conv(v, comp._kernel(k), s, kind) + comp.biases[b]
            post = comp._requantize_ste(tf.nn.relu(pre), o)
            rec[k] = dict(inp=v, pre=pre, post=post, out=o, stride=s, kind=kind, bias=b)
            v = post
        pooled = comp._requantize_ste(tf.reduce_mean(v, axis=(1, 2), keepdims=True), 31)
        pooled = tf.reshape(pooled, (-1, 64))
        logits = comp._requantize_ste(tf.linalg.matmul(pooled, comp._kernel(DENSE[0]), transpose_b=True) + comp.biases[DENSE[1]], DENSE[2])
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    posts = [rec[k]["post"] for k, *_ in LAYERS]
    grads = tape.gradient(loss, posts)
    for (k, *_), g in zip(LAYERS, grads):
        rec[k]["G"] = g
    clean_loss = float(loss.numpy())
    ref_logits, _ = comp.forward(x)
    assert np.array_equal(ref_logits.numpy(), logits.numpy()), "recorded forward differs from the companion forward"
    done = json.loads((STEP / "DONE").read_text())
    print(json.dumps(dict(clean_loss=clean_loss, pilot_clean_loss=done["clean_loss"], seconds=round(time.time() - t0))), flush=True)

    index = {n: i for i, n in enumerate(m.names)}
    kernel_of = {n: int(n.split("_")[2].split("/")[0]) for n in m.names}
    rows = []
    for r in csv.DictReader(open(STEP / "rows.csv")):
        name, w, bit = r["layer"], int(r["weight"]), int(r["bit"])
        k = kernel_of[name]
        local = w - int(m.starts[index[name]])
        delta = float(m.jumps[w, bit])
        dk = np.zeros(m.shapes[name], dtype=np.float32)
        dk.reshape(-1)[local] = delta
        kern = comp._kernel(k) + dk
        out = dict(layer=name, kernel=k, weight=w, bit=bit, jump=delta, real=float(r["real"]), gradient=float(r["linear"]),
                   quadratic=float(r["quadratic"]))
        if k == DENSE[0]:
            new_logits = comp._requantize_ste(tf.linalg.matmul(pooled, kern, transpose_b=True) + comp.biases[DENSE[1]], DENSE[2])
            new_loss = float(tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=new_logits)).numpy())
            out.update(local=new_loss - clean_loss, kind="dense", units_changed=int(np.sum(new_logits.numpy() != logits.numpy())),
                       relu_crossed=0, clipped=0)
        else:
            L = rec[k]
            pre_new = conv(L["inp"], kern, L["stride"], L["kind"]) + comp.biases[L["bias"]]
            post_new = comp._requantize_ste(tf.nn.relu(pre_new), L["out"])
            d_post = (post_new - L["post"]).numpy()
            pre0, pre1 = L["pre"].numpy(), pre_new.numpy()
            scale, zp = comp._quantization(L["out"])
            q1 = np.round(np.maximum(pre1, 0) / scale + zp)
            out.update(local=float(np.sum(L["G"].numpy() * d_post)), kind=L["kind"],
                       units_changed=int(np.count_nonzero(d_post)),
                       relu_crossed=int(np.sum((pre0 > 0) != (pre1 > 0))),
                       clipped=int(np.sum((q1 > 127) | (q1 < -128))))
        rows.append(out)
        if len(rows) % 40 == 0:
            print(json.dumps(dict(flips_done=len(rows), seconds=round(time.time() - t0))), flush=True)
    with open(a.out / "rows.csv", "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows[0])); wtr.writeheader(); wtr.writerows(rows)

    summary = dict(status="EXPLORATORY, post hoc", model="kws_rep0", flips=len(rows), clean_loss=clean_loss,
                   pilot_clean_loss=done["clean_loss"], seconds=round(time.time() - t0, 1), by_bit={}, by_kind={})
    def score(sel):
        real = np.array([r["real"] for r in sel])
        res = {}
        for key in ("gradient", "quadratic", "local"):
            p = np.array([r[key] for r in sel])
            nz = real != 0
            res[key] = dict(rank_corr=rank_corr(real, p), sign_agree=float(np.mean(np.sign(real[nz]) == np.sign(p[nz]))),
                            share_explained=float(1 - np.sum((real - p) ** 2) / np.sum((real - real.mean()) ** 2)),
                            median_abs_ratio=float(np.median(np.abs(p) / np.maximum(np.abs(real), 1e-12))))
        res["units_changed_median"] = float(np.median([r["units_changed"] for r in sel]))
        res["share_with_relu_crossing"] = float(np.mean([r["relu_crossed"] > 0 for r in sel]))
        res["share_with_clipping"] = float(np.mean([r["clipped"] > 0 for r in sel]))
        return res
    for b in range(8):
        summary["by_bit"][b] = score([r for r in rows if r["bit"] == b])
    for kind in ("conv", "depthwise", "dense"):
        sel = [r for r in rows if r["kind"] == kind]
        if sel:
            summary["by_kind"][kind] = score(sel)
    summary["all"] = score(rows)
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    for b, s in summary["by_bit"].items():
        print(f"bit {b}: " + " | ".join(f"{k} rank {s[k]['rank_corr']:+.2f} sign {s[k]['sign_agree']:.2f} R2 {s[k]['share_explained']:+.2f}"
                                       for k in ("gradient", "quadratic", "local"))
              + f" | units changed {s['units_changed_median']:.0f}, relu crossing {s['share_with_relu_crossing']:.2f}, clip {s['share_with_clipping']:.2f}")
    print(json.dumps({k: summary[k] for k in ("all", "by_kind")}, indent=1))


if __name__ == "__main__":
    main()
