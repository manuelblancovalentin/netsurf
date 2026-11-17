#!/usr/bin/env python3
"""Low-bit background against top-bit events at the convolution layers of the reference ResNet-8.

Plan: publications/paper1_review/review/EVENT_BACKGROUND_CHECK_PLAN_2026-11-17.md (paper worktree), frozen before
this script was run on the full grid.

For each CONV_2D layer, with the clean inputs of that layer on a few evaluation images held fixed, each unprotected
weight bit of the layer flips independently with probability p (protected bits never flip).  The change of every
pre-activation is linear in the weight displacements, so it splits exactly into the part due to bits 0-5 (background)
and the part due to bits 6-7 (events).  For every layer, mask and rate the script records, over many fault fields:
  * how many top-bit and low-bit flips fall in the fan-in of one neuron;
  * the share of the change's second moment that comes from the events, measured and exact;
  * the size of the event part against the standard deviation of the background, where at least one event occurred;
  * how far each part and their sum are from a normal distribution, after standardizing every neuron by its exact
    mean and variance (excess kurtosis; largest gap of the distribution function);
  * histograms of the standardized sum, and of both parts in units of the standard deviation of the sum.
The reconstruction of each layer is checked against the executable's own output before any fault is drawn.

Output (write-once): <out>/{summary.csv, histograms.npz, layers.json, DONE}; progress lines on stdout.

    event_background_check.py --out DIR [--fields 500] [--inputs 8] [--threads 8] [--layers ...] [--rates ...]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from netsurf import event_background as E  # noqa: E402

SEED = 20261117
RATES = (1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)             # the grid of the main comparison
MASKS = ("unprotected", "abs@0.125")
REP, CHUNK = 1, 50
EDGES = np.linspace(-10, 10, 401)
RATIO_EDGES = np.linspace(-3, 4, 141)                                # log10 of |event part| / background sd
KMAX = 60


def hist(v, edges=EDGES):
    return np.histogram(np.clip(v, edges[0] + 1e-9, edges[-1] - 1e-9), edges)[0]


def layers_of(it):
    det = {d["index"]: d for d in it.get_tensor_details()}
    out = []
    for op in it._get_ops_details():
        if op["op_name"] != "CONV_2D":
            continue
        x, w, b = op["inputs"][:3]
        y = op["outputs"][0]
        hin, hout = int(det[x]["shape"][1]), int(det[y]["shape"][1])
        stride = int(round(hin / hout))
        if -(-hin // stride) != hout:
            raise SystemExit(f"layer {det[w]['name']}: padding is not SAME")
        out.append(dict(x=x, w=w, b=b, y=y, stride=stride, name=f"tflite_tensor_{w:04d}/kernel", shape=tuple(int(s) for s in det[w]["shape"])))
    return out, det


def real(it, det, idx):
    q = det[idx]["quantization_parameters"]
    return (it.get_tensor(idx).astype(np.float64) - float(q["zero_points"][0])) * float(q["scales"][0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fields", type=int, default=500)
    ap.add_argument("--inputs", type=int, default=8)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--layers", nargs="*", type=int, default=None, help="positions in the list of CONV_2D layers")
    ap.add_argument("--rates", nargs="*", type=float, default=list(RATES))
    a = ap.parse_args()
    if a.fields % CHUNK:
        raise SystemExit(f"--fields must be a multiple of {CHUNK}")
    t0 = time.time()
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(a.threads)
    from runT_engine import load
    from runR_stage1 import bkey
    from runP_phase12 import unpack

    m, saved, (xe, _), _, _, cid = load("pinned", REP, a.threads)
    a.out.mkdir(parents=True, exist_ok=True)
    if (a.out / "DONE").exists():
        raise SystemExit("write-once: DONE exists")
    protected = {"unprotected": np.zeros((m.n, 8), bool), "abs@0.125": unpack(saved[f"abs1024@{bkey(0.125)}"], m.n)}
    it = tf.lite.Interpreter(model_content=bytes(m.adapter._model_content), experimental_delegates=[], num_threads=a.threads,
                             experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
                             experimental_preserve_all_tensors=True)
    it.allocate_tensors()
    layers, det = layers_of(it)
    inp = it.get_input_details()[0]
    xs = {L["x"]: [] for L in layers}
    ys = {L["y"]: [] for L in layers}
    for i in range(a.inputs):
        it.set_tensor(inp["index"], np.asarray(xe[i:i + 1]).reshape(inp["shape"]))
        it.invoke()
        for k in xs:
            xs[k].append(real(it, det, k)[0])
        for k in ys:
            ys[k].append(real(it, det, k)[0])
    row = {n: (int(m.starts[i]), int(m.starts[i + 1])) for i, n in enumerate(m.names)}
    which = range(len(layers)) if a.layers is None else a.layers
    rows, hists, record = [], {}, []
    for li in which:
        L = layers[li]
        s, e = row[L["name"]]
        cout, kh, kw, cin = L["shape"]
        codes = m.codes[s:e]
        if not np.array_equal(it.get_tensor(L["w"]).ravel(), codes):
            raise SystemExit(f"{L['name']}: stored codes differ from the executable's tensor")
        wq, yq = det[L["w"]]["quantization_parameters"], det[L["y"]]["quantization_parameters"]
        sw = np.asarray(wq["scales"], dtype=np.float64) if len(wq["scales"]) == cout else np.full(cout, float(wq["scales"][0]))
        x = np.stack(xs[L["x"]])                                                # (N, H, W, Cin), real units
        strides = [1, L["stride"], L["stride"], 1]

        def conv(v, filt, dtype=tf.float64):                                    # filt: (F, Cout, kh, kw, Cin) -> (F, N, Ho, Wo, Cout)
            f = np.transpose(filt, (2, 3, 4, 0, 1)).reshape(kh, kw, cin, -1)
            o = tf.nn.conv2d(tf.constant(v, dtype), tf.constant(f, dtype), strides, "SAME").numpy()
            return np.moveaxis(o.reshape(o.shape[:3] + (filt.shape[0], cout)), 3, 0)

        scale = sw[:, None, None, None]
        # reconstruction check: clean pre-activation against the executable's (clipped, rounded) output
        bq = det[L["b"]]["quantization_parameters"]
        bias = it.get_tensor(L["b"]).astype(np.float64) * np.asarray(bq["scales"], dtype=np.float64)
        z = conv(x, (codes.reshape(L["shape"]).astype(np.float64) * scale)[None])[0] + bias
        yv = np.stack(ys[L["y"]])
        step, zp = float(yq["scales"][0]), float(yq["zero_points"][0])
        lo, hi = (-128 - zp) * step, (127 - zp) * step
        inside = (yv > lo + step / 2) & (yv < hi - step / 2)
        ok = float(np.mean(np.abs(z - yv)[inside] <= 1.01 * step))
        relu = float(np.mean(z[~inside] <= lo + step)) if (~inside).any() else 0.0
        if ok < 0.999:
            raise SystemExit(f"{L['name']}: reconstruction matches the executable on only {ok:.4f} of unsaturated outputs")
        zsd = float(z.std())
        record.append(dict(layer=li, kernel=L["name"], shape=L["shape"], stride=L["stride"], fan_in=kh * kw * cin,
                           reconstruction_within_one_step=ok, unsaturated_share=float(inside.mean()),
                           clipped_low_share=relu, clean_preactivation_sd=zsd, output_step=step))
        print(json.dumps(dict(layer=li, kernel=L["name"], fan_in=kh * kw * cin, reconstruction=ok, seconds=round(time.time() - t0))), flush=True)
        for mi, mask in enumerate(MASKS):
            unprot = ~protected[mask][s:e]
            for p in a.rates:
                pi = RATES.index(p)
                mom = {part: E.moments(codes, unprot, p, bits) for part, bits in (("low", E.LOW_BITS), ("top", E.TOP_BITS))}
                mean, var = {}, {}
                for part in ("low", "top"):
                    mu, v = mom[part]
                    mean[part] = conv(x, (mu.reshape(L["shape"]) * scale)[None])[0]
                    var[part] = conv(x * x, (v.reshape(L["shape"]) * scale * scale)[None])[0]
                mean["sum"], var["sum"] = mean["low"] + mean["top"], var["low"] + var["top"]
                kp = {k: var[k] > 0 for k in var}                          # a part can be fully protected in a fan-in
                keep = kp["sum"]
                sd = {k: np.sqrt(np.where(kp[k], var[k], 1.0)) for k in var}
                acc = {k: np.zeros(5) for k in ("low", "top", "sum")}
                H = {k: np.zeros(len(EDGES) - 1) for k in ("low", "top", "sum", "low_on_sum", "top_on_sum")}
                HR = np.zeros(len(RATIO_EDGES) - 1)
                ev, lw = np.zeros(KMAX + 1), np.zeros(KMAX + 1)
                m2 = dict(low=0.0, top=0.0)
                ratio_n = ratio_big = 0
                for c in range(a.fields // CHUNK):
                    rng = np.random.default_rng([SEED, li, mi, pi, c])
                    flips = (rng.random((CHUNK, e - s, 8)) < p) & unprot[None]
                    dl, dt = E.split_displacement(codes, flips)
                    per = flips.reshape(CHUNK, cout, -1, 8)
                    n_top = per[..., list(E.TOP_BITS)].sum((2, 3))                 # (F, Cout): flips in one neuron's fan-in
                    n_low = per[..., list(E.LOW_BITS)].sum((2, 3))
                    ev += np.bincount(np.minimum(n_top, KMAX).ravel(), minlength=KMAX + 1)
                    lw += np.bincount(np.minimum(n_low, KMAX).ravel(), minlength=KMAX + 1)
                    part = {}
                    for name, d in (("low", dl), ("top", dt)):
                        part[name] = conv(x, d.reshape((CHUNK,) + L["shape"]).astype(np.float32) * scale.astype(np.float32), tf.float32).astype(np.float64)
                        m2[name] += float((part[name] ** 2)[:, keep].sum())
                    part["sum"] = part["low"] + part["top"]
                    for name in ("low", "top", "sum"):
                        zs = ((part[name] - mean[name]) / sd[name])[:, kp[name]]
                        if zs.size:
                            acc[name] += [zs.size, zs.sum(), (zs ** 2).sum(), (zs ** 3).sum(), (zs ** 4).sum()]
                            H[name] += hist(zs)
                    H["low_on_sum"] += hist((part["low"] / sd["sum"])[:, keep])
                    H["top_on_sum"] += hist((part["top"] / sd["sum"])[:, keep])
                    has = np.broadcast_to((n_top > 0)[:, None, None, None, :], part["top"].shape) & kp["low"][None]
                    r = np.abs(part["top"][has]) / np.broadcast_to(sd["low"][None], part["top"].shape)[has]
                    r = r[r > 0]
                    HR += hist(np.log10(r), RATIO_EDGES)
                    ratio_n += r.size
                    ratio_big += int((r > 3).sum())
                cdf = np.cumsum(HR) / max(HR.sum(), 1)
                median_ratio = float(10 ** RATIO_EDGES[1:][np.searchsorted(cdf, 0.5)]) if HR.sum() else float("nan")
                et, el = float((mean["top"] ** 2 + var["top"])[keep].sum()), float((mean["low"] ** 2 + var["low"])[keep].sum())
                exact_share = et / (et + el)
                nan = float("nan")
                k = np.arange(KMAX + 1)
                r_ = dict(layer=li, kernel=L["name"], fan_in=kh * kw * cin, mask=mask, ber=p, fields=a.fields, inputs=a.inputs,
                          neurons_kept_share=float(keep.mean()), neurons_with_events_share=float(kp["top"].mean()),
                          events_mean=float((ev * k).sum() / ev.sum()), events_p0=float(ev[0] / ev.sum()), events_p1=float(ev[1] / ev.sum()),
                          events_p2plus=float(ev[2:].sum() / ev.sum()), lowflips_mean=float((lw * k).sum() / lw.sum()), lowflips_p0=float(lw[0] / lw.sum()),
                          event_share_measured=m2["top"] / (m2["top"] + m2["low"]), event_share_exact=exact_share,
                          sum_sd_over_clean_sd=float(np.sqrt(var["sum"][keep].mean()) / zsd),
                          event_over_background_median=median_ratio, event_over_background_gt3=ratio_big / max(ratio_n, 1),
                          **{f"kurtosis_{n}": E.excess_kurtosis(*acc[n]) if acc[n][0] else nan for n in acc},
                          **{f"ks_{n}": E.ks_to_normal(H[n], EDGES) if acc[n][0] else nan for n in ("low", "top", "sum")})
                rows.append(r_)
                for n, h in H.items():
                    hists[f"L{li}|{mask}|{pi}|{n}"] = h
                hists[f"L{li}|{mask}|{pi}|ratio"] = HR
                hists[f"L{li}|{mask}|{pi}|events"] = ev
                print(json.dumps(dict(layer=li, mask=mask, ber=p, done=len(rows), total=len(list(which)) * len(MASKS) * len(a.rates),
                                      events_mean=round(r_["events_mean"], 3), share=round(r_["event_share_measured"], 4),
                                      ks_sum=round(r_["ks_sum"], 4), seconds=round(time.time() - t0))), flush=True)
    with open(a.out / "summary.csv", "x", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    np.savez_compressed(a.out / "histograms.npz", edges=EDGES, ratio_edges=RATIO_EDGES, **hists)
    (a.out / "layers.json").write_text(json.dumps(record, indent=1))
    (a.out / "DONE").write_text(json.dumps(dict(seconds=round(time.time() - t0), checkpoint=cid, rep=REP, fields=a.fields, inputs=a.inputs,
                                                seed=SEED, rates=list(a.rates), masks=list(MASKS),
                                                summary_sha256=hashlib.sha256((a.out / "summary.csv").read_bytes()).hexdigest(),
                                                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                                module_sha256=hashlib.sha256(Path(E.__file__).read_bytes()).hexdigest())))


if __name__ == "__main__":
    main()
