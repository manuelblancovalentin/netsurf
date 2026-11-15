#!/usr/bin/env python3
"""Gradient-trust plan (review/GRADIENT_TRUST_PLAN_2026-11-15.md): a generic INT8 TFLite model and the Phase 1
diagnostics (no fault runs).

GenModel: any full-integer INT8 TFLite classifier the graph companion supports, flattened in TFLiteModelStorage kernel
order (the same interface as KwsModel: names, starts, codes, jumps, comp, n, set_displacement, per-example
statistics) plus reference-kernel evaluation with flipped stored bits.

Phase 1 for one checkpoint, from R disjoint calibration draws (Section 2 of the plan):
  per draw: gbar, se (= std(ddof=1)/sqrt(N)), g2, and the abs / pos bit scores (runQ_calibration.scores formula);
  D1 = median pairwise correlation of gbar between draws; D2 = median signal fraction (sum gbar^2 - sum se^2) / sum
  gbar^2 (runR_stage4_diagnostics.grad_stats); D3 = out-of-sample over in-sample first-order advantage of pos1024 over
  abs1024 (runR_stage4_diagnostics.grad_block); the histogram of t = gbar / se; D1 at smaller and larger calibration
  sizes (descriptive); the prediction of Section 4.

    runG_gradtrust.py diag-kws <out>      regression check: Phase 1 on the DS-CNN's six Stage 4 calibration draws must
                                          reproduce the Stage 4 diagnostics (D1 0.21, D2 0.23)
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

FRACTIONS = (0.125, 0.25)
CUT_LOW, CUT_HIGH = 0.30, 0.40          # plan Section 4, decided 15 November 2026
T_EDGES = np.linspace(-8, 8, 161)


def log(**kw):
    print(json.dumps(dict(t=time.strftime("%H:%M:%S"), **kw)), flush=True)


def predict(d1: float) -> str:
    if d1 < CUT_LOW:
        return "one regime"
    if d1 > CUT_HIGH:
        return "two regimes"
    return "no prediction"


class GenModel:
    def __init__(self, content: bytes, threads: int = 4, expected_sha256: str | None = None):
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(threads)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        from netsurf.mlperf_companion import affine_bit_perturbations
        from netsurf.tflite_graph_companion import TFLiteGraphCompanion
        from netsurf.tflite_storage import TFLiteModelStorage
        self.tf = tf
        self.storage = TFLiteModelStorage(content, expected_sha256=expected_sha256, eligible_roles=("kernel",))
        values = self.storage.values()
        self.names = list(values)
        self.shapes = {n: values[n].shape for n in self.names}
        self.starts = np.concatenate([[0], np.cumsum([values[n].size for n in self.names])]).astype(np.int64)
        self.codes = np.concatenate([np.asarray(values[n]).reshape(-1) for n in self.names]).astype(np.int8)
        jumps = affine_bit_perturbations(values, self.storage.descriptors)
        self.jumps = np.concatenate([np.asarray(jumps[n]).reshape(-1, 8) for n in self.names]).astype(np.float64)
        self.comp = TFLiteGraphCompanion(content)
        if set(self.comp.variables) != set(self.names):
            raise AssertionError("companion kernels differ from storage kernels")
        self.clean = {n: self.comp.variables[n].numpy().copy() for n in self.names}
        self.n = int(self.starts[-1])
        self.sha256 = self.storage.sha256
        self._interp = None

    def set_displacement(self, mu):
        for i, nm in enumerate(self.names):
            d = mu[self.starts[i]: self.starts[i + 1]].reshape(self.shapes[nm])
            self.comp.variables[nm].assign((self.clean[nm] + d).astype(np.float32))

    def gradient_stats(self, x, y, batch=256):
        """Streaming per-example statistics over (x, y): mean, mean square, mean absolute, mean positive and mean
        negative part of the per-weight gradient, and the standard error of the mean."""
        self.set_displacement(np.zeros(self.n))
        acc = {k: np.zeros(self.n) for k in ("g", "g2", "gabs", "gpos", "gneg")}
        N = len(y)
        for s in range(0, N, batch):
            pe = self.comp.per_example_gradients(x[s:s + batch], y[s:s + batch], batch=batch)
            G = np.concatenate([np.asarray(pe[nm]).reshape(len(y[s:s + batch]), -1) for nm in self.names], axis=1)
            G = G.astype(np.float64)
            acc["g"] += G.sum(0); acc["g2"] += (G * G).sum(0); acc["gabs"] += np.abs(G).sum(0)
            acc["gpos"] += np.maximum(G, 0).sum(0); acc["gneg"] += np.maximum(-G, 0).sum(0)
        st = {k: v / N for k, v in acc.items()}
        var = np.maximum(st["g2"] - st["g"] ** 2, 0.0) * N / (N - 1)
        st["se"] = np.sqrt(var / N)
        st["N"] = N
        return st

    def scores(self, st):
        J = self.jumps
        return dict(abs=st["gabs"][:, None] * np.abs(J),
                    pos=np.where(J > 0, st["gpos"][:, None] * J, st["gneg"][:, None] * (-J)))

    # ---- reference-kernel evaluation with flipped stored bits ----
    def evaluate(self, flips, mask, x, y):
        """Accuracy and mean cross-entropy (from the int8 softmax output) of the TFLite reference interpreter with
        the bits `flips & ~mask` flipped in the stored kernels."""
        tf = self.tf
        applied = flips & ~mask
        u8 = self.codes.view(np.uint8)
        codes = (u8 ^ np.packbits(applied, axis=1, bitorder="little")[:, 0]).view(np.int8)
        values = {nm: codes[self.starts[i]: self.starts[i + 1]].reshape(self.shapes[nm]) for i, nm in enumerate(self.names)}
        content = self.storage.model_content_from_values(values)
        it = tf.lite.Interpreter(model_content=content, experimental_delegates=[], num_threads=1,
                                 experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
        it.allocate_tensors()
        i, o = it.get_input_details()[0], it.get_output_details()[0]
        sc, zp = o["quantization"]
        correct, ce = 0, 0.0
        for s, lab in zip(x, y):
            it.set_tensor(i["index"], s[None])
            it.invoke()
            q = it.get_tensor(o["index"])[0].astype(np.float64)
            p = (q - zp) * sc
            correct += int(np.argmax(q) == lab)
            ce += -np.log(max(p[int(lab)], 1e-12))
        return correct / len(y), ce / len(y), int(flips.sum()), int(applied.sum())


def first_order(jumps, pos, abs_):
    return (jumps * (abs_.astype(float) - pos.astype(float))).sum(1)


def phase1(model: GenModel, draws, sizes=(256, 512)):
    """draws: list of (x, y) disjoint calibration sets of equal size.  Returns (summary, per-draw arrays)."""
    from netsurf.mean_variance import top_budget
    from runR_stage4_diagnostics import grad_stats
    t0 = time.time()
    stats = []
    for r, (x, y) in enumerate(draws):
        stats.append(model.gradient_stats(x, y))
        log(stage="gradients", draw=r, done=r + 1, total=len(draws), seconds=round(time.time() - t0))
    G = np.stack([s["g"] for s in stats])
    R = len(stats)
    cor = np.corrcoef(G)[np.triu_indices(R, 1)]
    per_draw = [grad_stats(s["g"], s["se"], s["g2"]) for s in stats]
    counts = {f: int(np.floor(f * model.n * 8)) for f in FRACTIONS}
    fo = {}
    for f in FRACTIONS:
        ins, outs = [], []
        for i, s in enumerate(stats):
            sc = model.scores(s)
            pos, abs_ = top_budget(sc["pos"], counts[f]), top_budget(sc["abs"], counts[f])
            d = first_order(model.jumps, pos, abs_)
            ins.append(float(G[i] @ d))
            outs.append(float(np.delete(G, i, 0).mean(0) @ d))
        fo[f"{f:g}"] = dict(in_sample=float(np.mean(ins)), out_of_sample=float(np.mean(outs)),
                            out_over_in=float(np.mean(outs) / np.mean(ins)) if np.mean(ins) else None)
    t = np.concatenate([s["g"] / np.where(s["se"] > 0, s["se"], np.inf) for s in stats])
    hist = np.histogram(np.clip(t, T_EDGES[0], T_EDGES[-1]), bins=T_EDGES)[0]
    d1 = float(np.median(cor))
    summary = dict(D1=d1, D1_pairs=[float(c) for c in cor],
                   D2=float(np.median([p["signal_fraction"] for p in per_draw])),
                   coherence=float(np.median([p["coherence"] for p in per_draw])),
                   D3=fo, prediction=predict(d1), draws=R, draw_size=int(stats[0]["N"]), weights=model.n,
                   t_histogram=dict(edges=[float(e) for e in T_EDGES], counts=[int(c) for c in hist],
                                    share_abs_t_above_2=float(np.mean(np.abs(t) > 2)),
                                    share_abs_t_above_2_if_noise=0.0455),
                   seconds=round(time.time() - t0))
    return summary, stats


def size_study(model, draws, sizes):
    """D1 at other calibration sizes (descriptive): the first `n` inputs of every draw, and pairs of draws merged."""
    out = {}
    for n in sizes:
        st = [model.gradient_stats(x[:n], y[:n])["g"] for x, y in draws]
        c = np.corrcoef(np.stack(st))[np.triu_indices(len(st), 1)]
        out[str(n)] = float(np.median(c))
    merged = [(np.concatenate([draws[k][0], draws[k + 1][0]]), np.concatenate([draws[k][1], draws[k + 1][1]]))
              for k in range(0, len(draws) - 1, 2)]
    if len(merged) >= 2:
        st = [model.gradient_stats(x, y)["g"] for x, y in merged]
        c = np.corrcoef(np.stack(st))[np.triu_indices(len(st), 1)]
        out[str(len(merged[0][1]))] = float(np.median(c))
    return out


def cmd_diag_kws(a):
    """Regression check on the DS-CNN: the six Stage 4 calibration draws (training clips 1024 r .. 1024 r + 1023)."""
    from runR_kws import CAL_COUNT, KwsModel
    from runR_stage4_kws import calibration
    k = KwsModel(a.threads)
    m = GenModel(k.adapter._require_loaded().reference_content, a.threads)
    a.out.mkdir(parents=True, exist_ok=False)
    draws = []
    for r in range(6):
        x, y = calibration(k, r)
        draws.append((np.asarray(x), np.asarray(y)))
    summary, stats = phase1(m, draws)
    from runR_partial_search import RUNS
    ref = []
    for r in range(6):
        from runR_stage1 import bkey
        c = np.load(RUNS / f"paper1_runR_stage4_kws_masks_v1/rep{r}/b{bkey(0.125)}/calibration.npz")
        ref.append(dict(draw=r, gbar_max_abs_diff=float(np.abs(c["gbar"] - stats[r]["g"]).max()),
                        gbar_max_abs=float(np.abs(c["gbar"]).max()),
                        se_max_rel_diff=float(np.max(np.abs(c["se"] - stats[r]["se"]) / np.maximum(c["se"], 1e-30)))))
    summary["stage4_reference"] = dict(D1=0.21, D2=0.23, per_draw=ref, calibration_count=CAL_COUNT)
    summary["size_study"] = size_study(m, draws, (256, 512))
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    log(stage="done", D1=round(summary["D1"], 3), D2=round(summary["D2"], 3), prediction=summary["prediction"],
        size_study=summary["size_study"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("diag-kws",))
    ap.add_argument("out", type=Path)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        raise SystemExit('run on CPU: CUDA_VISIBLE_DEVICES=""')
    {"diag-kws": cmd_diag_kws}[a.command](a)


if __name__ == "__main__":
    main()
