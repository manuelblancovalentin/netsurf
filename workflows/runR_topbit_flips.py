#!/usr/bin/env python3
"""EXPLORATORY: measure every top-bit single flip of the DS-CNN, per example (GPU or CPU).

For each weight and each bit in --bits (default 6 and 7), the weight is moved by its exact bit jump and only the
layers from the flipped one onwards are recomputed (the layer inputs are cached once; exact, because earlier layers do
not depend on the flipped weight).  Recorded per flip, on 1,312 clips (the Stage 4 evaluation window of replication 0,
800 clips, then the 512 calibration clips of the blur pilot):
  margin   true logit minus the best other logit, per clip (float32)
  correct  argmax == label, per clip (int8; ties resolved like argmax)
  loss     mean cross-entropy on each of the two sets
plus the clean values once.  These are the per-example single-flip effects a_ib of the bit-level Fourier formulation.

Phases
  check    (1) the cached downstream path reproduces the companion's full forward exactly (clean logits);
           (2) the 800 single flips the blur pilot measured on the CPU (loss change on the 512 calibration clips) are
               recomputed here and compared -- this is the agreement check for a GPU run;
           (3) timing per flip.  Writes <out>/check.json.
  measure  all weights x bits, one output file per layer (write-once, a finished layer is skipped on restart),
           <out>/layer_<k>.npz, then <out>/DONE.  Progress lines every --log-every flips (unbuffered).

  runR_topbit_flips.py check|measure [--out DIR] [--bits 6 7] [--limit N] [--threads T]
Default <out>: workspace/runs/paper1_topbit_flips_kws_v1 in the paper worktree (NETSURF_SITE_WT moves it).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from site_paths import WT  # noqa: E402

RUNS = WT / "paper1-tecs-revision-v1/workspace/runs"
OUT = RUNS / "paper1_topbit_flips_kws_v1"
PILOT_STEP = RUNS / "paper1_blur_pilot_v1/step/kws_rep0/rows.csv"
LAYERS = [(17, 3, 22, 2, "conv")]
for _k, _b, _pk, _pb, _do, _po in ((5, 4, 18, 6, 23, 24), (8, 7, 19, 9, 25, 26), (11, 10, 20, 12, 27, 28), (14, 13, 21, 15, 29, 30)):
    LAYERS += [(_k, _b, _do, 1, "depthwise"), (_pk, _pb, _po, 1, "conv")]
DENSE = (16, 1, 33)


def log(**kw):
    print(json.dumps(dict(t=time.strftime("%H:%M:%S"), **kw)), flush=True)


class Downstream:
    """Cached layer inputs and one compiled 'from layer li onwards' function per layer."""

    def __init__(self, m, x, y):
        self.m, self.tf, self.comp = m, m.tf, m.comp
        tf, comp = self.tf, self.comp
        m.set_displacement(np.zeros(m.n))
        self.labels = tf.convert_to_tensor(y)
        self.cache = []
        v = comp._decode_input(x)
        for (k, b, o, s, kind) in LAYERS:
            self.cache.append(v)
            v = self._layer(v, comp._kernel(k), b, o, s, kind)
        self.pooled = self._pool(v)
        self.order = {k: i for i, (k, *_) in enumerate(LAYERS)}
        self.funcs = {}

    def _conv(self, v, kernel, s, kind):
        tf = self.tf
        k = tf.transpose(kernel, (1, 2, 3, 0))
        if kind == "depthwise":
            return tf.nn.depthwise_conv2d(v, k, strides=(1, s, s, 1), padding="SAME")
        return tf.nn.conv2d(v, k, strides=(1, s, s, 1), padding="SAME")

    def _layer(self, v, kernel, b, o, s, kind):
        return self.comp._requantize_ste(self.tf.nn.relu(self._conv(v, kernel, s, kind) + self.comp.biases[b]), o)

    def _pool(self, v):
        tf = self.tf
        return tf.reshape(self.comp._requantize_ste(tf.reduce_mean(v, axis=(1, 2), keepdims=True), 31), (-1, 64))

    def _logits(self, pooled, kernel):
        return self.comp._requantize_ste(self.tf.linalg.matmul(pooled, kernel, transpose_b=True) + self.comp.biases[DENSE[1]], DENSE[2])

    def func(self, k):
        if k in self.funcs:
            return self.funcs[k]
        tf, comp = self.tf, self.comp
        # Eager, not tf.function: graph mode fuses convolution, bias and ReLU, which changes the float rounding and
        # moves activations across INT8 rounding boundaries (the smoke test caught a mismatch with the companion).
        if k == DENSE[0]:
            def fn(kernel):
                return self._logits(self.pooled, kernel)
        else:
            li = self.order[k]

            def run(kernel):
                v = self.cache[li]
                for j in range(li, len(LAYERS)):
                    kk, b, o, s, kind = LAYERS[j]
                    v = self._layer(v, kernel if j == li else comp._kernel(kk), b, o, s, kind)
                return self._logits(self._pool(v), comp._kernel(DENSE[0]))
            fn = run
        self.funcs[k] = fn
        return fn

    def readout(self, logits, sets):
        z = logits.numpy().astype(np.float64)
        y = self.labels.numpy()
        idx = np.arange(len(y))
        other = z.copy()
        other[idx, y] = -np.inf
        margin = (z[idx, y] - other.max(1)).astype(np.float32)
        correct = (z.argmax(1) == y).astype(np.int8)
        zz = z - z.max(1, keepdims=True)
        ce = -(zz[idx, y] - np.log(np.exp(zz).sum(1)))
        return margin, correct, {name: float(ce[sl].mean()) for name, sl in sets.items()}


def setup(threads):
    from runR_kws import CAL_COUNT, KwsModel
    from runR_stage4_kws import WINDOW
    m = KwsModel(threads)
    xe, ye, _ = m.clips("evaluation", WINDOW, WINDOW * 0)
    xc, yc, _ = m.clips("ranking", 512, CAL_COUNT * 0)
    x = np.concatenate([np.asarray(xe), np.asarray(xc)])
    y = np.concatenate([np.asarray(ye), np.asarray(yc)]).astype(np.int64)
    sets = {"eval": slice(0, len(ye)), "cal": slice(len(ye), len(y))}
    return m, x, y, sets


def flipped_kernel(m, comp, name, w, bit):
    index = {n: i for i, n in enumerate(m.names)}
    dk = np.zeros(m.shapes[name], dtype=np.float32)
    dk.reshape(-1)[w - int(m.starts[index[name]])] = float(m.jumps[w, bit])
    return comp._kernel(int(name.split("_")[2].split("/")[0])) + dk


def check(a):
    m, x, y, sets = setup(a.threads)
    tf = m.tf
    log(phase="check", stage="model loaded", devices=[d.name for d in tf.config.list_logical_devices()])
    D = Downstream(m, x, y)
    ref, _ = m.comp.forward(x)
    path = D.func(LAYERS[0][0])(m.comp._kernel(LAYERS[0][0]))
    same = bool(np.array_equal(ref.numpy(), path.numpy()))
    log(stage="cached path vs full forward", identical=same)
    assert same, "cached downstream path differs from the companion forward"
    _, _, clean = D.readout(ref, sets)
    rows = list(csv.DictReader(open(PILOT_STEP)))[: a.limit or None]
    diffs, t0 = [], time.time()
    for i, r in enumerate(rows):
        name, w, bit = r["layer"], int(r["weight"]), int(r["bit"])
        k = int(name.split("_")[2].split("/")[0])
        _, _, loss = D.readout(D.func(k)(flipped_kernel(m, m.comp, name, w, bit)), sets)
        diffs.append((loss["cal"] - clean["cal"]) - float(r["real"]))
        if (i + 1) % 100 == 0:
            log(stage="pilot flips", done=i + 1, total=len(rows), seconds=round(time.time() - t0, 1))
    sec = (time.time() - t0) / max(len(rows), 1)
    diffs = np.abs(np.array(diffs))
    real_scale = float(np.median([abs(float(r["real"])) for r in rows]))
    res = dict(identical_clean_path=same, pilot_flips=len(rows), clean_loss_cal=clean["cal"],
               max_abs_diff=float(diffs.max()), median_abs_diff=float(np.median(diffs)),
               share_within_1e_6=float(np.mean(diffs < 1e-6)), median_abs_real=real_scale, seconds_per_flip=sec,
               device=os.environ.get("CUDA_VISIBLE_DEVICES", "cpu"))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "check.json").write_text(json.dumps(res, indent=1))
    log(stage="check done", **res)


def measure(a):
    m, x, y, sets = setup(a.threads)
    log(phase="measure", stage="model loaded", devices=[d.name for d in m.tf.config.list_logical_devices()])
    D = Downstream(m, x, y)
    a.out.mkdir(parents=True, exist_ok=True)
    clean_logits, _ = m.comp.forward(x)
    margin0, correct0, loss0 = D.readout(clean_logits, sets)
    if not (a.out / "clean.npz").exists():
        np.savez(a.out / "clean.npz", margin=margin0, correct=correct0, labels=y, **{f"loss_{k}": v for k, v in loss0.items()})
    index = {n: i for i, n in enumerate(m.names)}
    total = sum(int(m.starts[index[n] + 1] - m.starts[index[n]]) for n in m.names) * len(a.bits)
    total = min(total, a.limit) if a.limit else total
    done, t0 = 0, time.time()
    for name in m.names:
        k = int(name.split("_")[2].split("/")[0])
        target = a.out / f"layer_{k}.npz"
        lo, hi = int(m.starts[index[name]]), int(m.starts[index[name] + 1])
        pairs = [(w, b) for w in range(lo, hi) for b in a.bits]
        if a.limit:
            pairs = pairs[: max(0, a.limit - done)]
        if not pairs:
            break
        if target.exists():
            done += len(pairs)
            log(stage="layer skipped (exists)", layer=name)
            continue
        fn = D.func(k)
        M = np.empty((len(pairs), len(y)), dtype=np.float32)
        C = np.empty((len(pairs), len(y)), dtype=np.int8)
        L = {s: np.empty(len(pairs)) for s in sets}
        for i, (w, b) in enumerate(pairs):
            M[i], C[i], loss = D.readout(fn(flipped_kernel(m, m.comp, name, w, b)), sets)
            for s in sets:
                L[s][i] = loss[s]
            done += 1
            if done % a.log_every == 0:
                el = time.time() - t0
                log(done=done, total=total, layer=name, seconds=round(el), ms_per_flip=round(1000 * el / done, 1),
                    eta_minutes=round(el / done * (total - done) / 60, 1))
        tmp = a.out / f"layer_{k}.tmp.npz"
        np.savez(tmp, weight=np.array([p[0] for p in pairs]), bit=np.array([p[1] for p in pairs]),
                 jump=np.array([m.jumps[w, b] for w, b in pairs]), margin=M, correct=C,
                 **{f"loss_{s}": v for s, v in L.items()})
        tmp.rename(target)
        log(stage="layer written", layer=name, flips=len(pairs), done=done, total=total)
    (a.out / "DONE").write_text(json.dumps(dict(flips=done, bits=a.bits, seconds=round(time.time() - t0, 1),
                                                 sets={k: [v.start, v.stop] for k, v in sets.items()})))
    log(stage="measure done", flips=done, seconds=round(time.time() - t0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=("check", "measure"))
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--bits", type=int, nargs="+", default=[6, 7])
    ap.add_argument("--limit", type=int, default=0, help="stop after this many flips (smoke tests)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=1000)
    a = ap.parse_args()
    {"check": check, "measure": measure}[a.phase](a)


if __name__ == "__main__":
    main()
