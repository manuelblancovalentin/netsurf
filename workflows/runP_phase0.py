#!/usr/bin/env python3
"""Run P, Phase 0 (offline; no fault injection).

Plan: publications/paper1_review/review/RUN_P_MEAN_VARIANCE_DRAFT_PLAN_2026-11-11.md in the paper worktree
(branch paper1-tecs-revision-v1), including amendment A1.  Proposed method code name: mean_variance_mask
(not qPolarGrad).

Stages (one process per checkpoint; the retrained-checkpoint path of the Run K engine is selected by
NETSURF_RETRAINED_IC_SHA256, which this script sets from the checkpoint file before loading):

  stats   For every replication of the finalized bundle: per-example gradients of the clean companion on the
          replication's 64 calibration images; check that the saved absolute-gradient masks are reproduced from
          them; P0a statistics of every saved mask; P0b accuracy and loss of the companion at the mean weights on
          the replication's evaluation window; P0c objective on the calibration images.  Seven rankings, four
          budgets, BER 1e-4 ... 1e-1.
  calib   P0d: replication 1, 12.5%, range W (8 rates log-uniform in [3e-4, 3e-2]); masks from n = 64, 128, 256,
          512 calibration images; objective on 256 held-out images (training images 27000-27255).
  curv    P0e: Hutchinson estimate of the Hessian diagonal against the Fisher diagonal (replication 1).
  analyze Merge with the measured per-cell quality and evaluate the gate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from netsurf.mean_variance import (  # noqa: E402
    bit_second_moments, code_jumps, displacement_moments, greedy_mean_variance_mask, log_uniform_rates,
    mask_sha256, top_budget, variance_term,
)

WT = Path("/projects/netsurf/manuelbv/workspace/implementation-worktrees")
PAPER = WT / "paper1-tecs-revision-v1"
PINNED = Path("/data/models/models--mlcommons--tiny/blobs/3c002613d1b2475eb51dd78dfb85a546c8ae658dee71cf6ade43b022fe205415")
DATASET = Path("/data/datasets/cifar10/cifar-10-python.tar.gz")
LOCK = Path("/projects/kona-assets/manuelbv/locks/mlperf-tiny/ic.lock.json")
RUNJ = PAPER / "workspace/runs/paper1_runJ_resnet8_variants_v1"
RUNK = WT / "paper1-runK-retrained-engine/workspace/runs"
RUNA = WT / "paper1-runA-tflite-confirm/workspace/runs/paper1_runA_resnet8_confirm_v1"
MEASURED = PAPER / "publications/paper1_review/data"
OUT = PAPER / "workspace/runs/paper1_runP_phase0_v1"

CHECKPOINTS = {
    "pinned": dict(file=PINNED, bundle=RUNA, reps=range(1, 10), measured=MEASURED / "tecs_runA/fresh_cell_quality.csv"),
    "undertrained_e100": dict(file=RUNJ / "undertrained_e100/trained_quant.tflite",
                              bundle=RUNK / "paper1_runK_resnet8_undertrained_e100_v2", reps=range(0, 8),
                              measured=MEASURED / "tecs_runK/undertrained_e100/fresh_cell_quality.csv"),
    "pruned_s07": dict(file=RUNJ / "pruned_s0.7/trained_quant.tflite", bundle=RUNK / "paper1_runK_resnet8_pruned_s07_v2",
                       reps=range(0, 8), measured=MEASURED / "tecs_runK/pruned_s07/fresh_cell_quality.csv"),
    "pruned_s05": dict(file=RUNJ / "pruned_s0.5/trained_quant.tflite", bundle=RUNK / "paper1_runK_resnet8_pruned_s05_v2",
                       reps=range(0, 8), measured=MEASURED / "tecs_runK/pruned_s05/fresh_cell_quality.csv"),
}
RANKINGS = {"random": "random_global", "msb": "msb_global", "jump": "numerical_jump_global",
            "global_grad": "gradient_global", "abs_grad": "per_example_absolute_global",
            "pos_grad": "per_example_positive_global", "fkeras": "fkeras_paper"}
BUDGETS = ("0.062500", "0.125000", "0.187500", "0.250000")
BERS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
RANGE_W = log_uniform_rates(3e-4, 3e-2, 8)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def bundle_dir(base: Path) -> Path:
    return next(d for d in sorted(base.iterdir()) if (d / "FINALIZED").is_file())


class Model:
    """Engine adapter + differentiable companion of one checkpoint, flattened in adapter storage order."""

    def __init__(self, name: str, threads: int):
        spec = CHECKPOINTS[name]
        if name != "pinned":
            os.environ["NETSURF_RETRAINED_IC_SHA256"] = sha256_file(spec["file"])
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(threads)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        from netsurf.adapters.mlperf_tiny import MlperfTinyImageClassificationAdapter
        from netsurf.mlperf_companion import MlperfResnet8DifferentiableCompanion
        self.tf = tf
        self.adapter = MlperfTinyImageClassificationAdapter(spec["file"], DATASET, asset_lock=LOCK, split="test",
                                                            injectable_parameters=("kernels",))
        self.adapter.load()
        storage = self.adapter._require_loaded()
        values = storage.values()
        self.names = list(values)
        self.shapes = {n: values[n].shape for n in self.names}
        sizes = [values[n].size for n in self.names]
        self.starts = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        self.codes = np.concatenate([np.asarray(values[n]).reshape(-1) for n in self.names]).astype(np.int8)
        jumps = self.adapter.bit_perturbations()
        self.jumps = np.concatenate([np.asarray(jumps[n]).reshape(-1, 8) for n in self.names]).astype(np.float64)
        # The engine's decoded jumps must carry the two's-complement code jumps (sign and ratio within a word).
        cj = code_jumps(self.codes).astype(np.float64)
        if not np.array_equal(np.sign(self.jumps), np.sign(cj)):
            raise AssertionError("engine jump signs differ from two's-complement code jumps")
        ratio = self.jumps / cj
        if not np.allclose(ratio, ratio[:, :1], rtol=1e-6):
            raise AssertionError("engine jumps are not one scale times the code jumps")
        self.comp = MlperfResnet8DifferentiableCompanion(storage.reference_content)
        if set(self.comp.variables) != set(self.names):
            raise AssertionError("companion kernel names differ from adapter names")
        self.clean = {n: self.comp.variables[n].numpy().copy() for n in self.names}
        self.n = int(self.starts[-1])
        self.sha256 = storage.sha256

    def images(self, role, count, offset):
        x, y, part = self.adapter.sample_partition(role, count, offset=offset)
        return x, np.asarray(y), tuple(int(i) for i in part.indices)

    def set_displacement(self, mu):
        for i, n in enumerate(self.names):
            d = mu[self.starts[i]: self.starts[i + 1]].reshape(self.shapes[n])
            self.comp.variables[n].assign((self.clean[n] + d).astype(np.float32))

    def loss_and_grad(self, x, y):
        tf = self.tf
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(self.comp.losses(x, y))
        grads = tape.gradient(loss, [self.comp.variables[n] for n in self.names])
        return float(loss.numpy()), np.concatenate([g.numpy().reshape(-1) for g in grads]).astype(np.float64)

    def loss_acc(self, x, y, batch=500):
        losses, correct = [], 0
        for s in range(0, len(y), batch):
            logits, _ = self.comp.forward(x[s:s + batch])
            yy = y[s:s + batch]
            losses.append(self.tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.tf.convert_to_tensor(yy, dtype=self.tf.int64), logits=logits).numpy())
            correct += int((np.argmax(logits.numpy(), axis=1) == yy).sum())
        return float(np.concatenate(losses).mean()), correct / len(y)

    def per_example(self, x, y):
        self.set_displacement(np.zeros(self.n))
        g = self.comp.per_example_gradients(x, y)
        return np.concatenate([np.asarray(g[n]).reshape(len(y), -1) for n in self.names], axis=1).astype(np.float64)

    def mean_term(self, x, y, clean_loss):
        def f(mu):
            self.set_displacement(mu)
            loss, grad = self.loss_and_grad(x, y)
            return loss - clean_loss, grad
        return f


def load_masks(model: Model, bundle: Path, reps):
    wanted = {f"{RANKINGS[r]}@{b}" for r in RANKINGS for b in BUDGETS}
    t = time.time()
    data = json.loads((bundle / "artifacts/protection_masks.json").read_text())
    parts = json.loads((bundle / "artifacts/sample_partitions.json").read_text())
    index = {n: i for i, n in enumerate(model.names)}
    out = {}
    for rep in reps:
        key = f"replication_{rep}"
        out[rep] = {}
        for k in wanted:
            m = np.zeros((model.n, 8), dtype=bool)
            items = data[key][k]
            pi = np.fromiter((index[p] for p, _, _ in items), dtype=np.int64, count=len(items))
            fi = np.fromiter((f for _, f, _ in items), dtype=np.int64, count=len(items))
            bi = np.fromiter((b for _, _, b in items), dtype=np.int64, count=len(items))
            m[model.starts[pi] + fi, bi] = True
            out[rep][k] = m
    del data
    print(f"masks loaded in {time.time() - t:.0f}s", flush=True)
    return out, parts


def moments(G):
    return dict(g1=G.mean(0), gabs=np.abs(G).mean(0), g2=(G * G).mean(0),
                gpos=np.maximum(G, 0).mean(0), gneg=np.maximum(-G, 0).mean(0))


def first_order_split(mom, jumps, keep):
    """Summed loss-raising and loss-lowering first-order effects (per unit p) of the bits in `keep`."""
    up = np.where(jumps > 0, jumps * mom["gpos"][:, None], -jumps * mom["gneg"][:, None])
    down = -np.where(jumps > 0, jumps * mom["gneg"][:, None], -jumps * mom["gpos"][:, None])
    return float(np.where(keep, up, 0).sum()), float(np.where(keep, down, 0).sum())


def stage_stats(name: str, threads: int) -> None:
    spec = CHECKPOINTS[name]
    out = OUT / name / "stats"
    out.mkdir(parents=True, exist_ok=False)
    model = Model(name, threads)
    bundle = bundle_dir(spec["bundle"])
    masks, parts = load_masks(model, bundle, spec["reps"])
    cj = code_jumps(model.codes).astype(np.float64)
    rows_a, rows_bc, checks = [], [], []
    for rep in spec["reps"]:
        t0 = time.time()
        part = parts[f"replication_{rep}"]
        xc, yc, ic = model.images("ranking", 64, 1000 * rep)
        xe, ye, ie = model.images("evaluation", 1000, 1000 * rep)
        assert list(ic) == part["ranking"]["indices"] and list(ie) == part["evaluation"]["indices"]
        G = model.per_example(xc, yc)
        mom = moments(G)
        m2 = bit_second_moments(model.jumps, mom["g2"])
        # Reproduction check of the saved absolute-gradient masks from these gradients.
        score = mom["gabs"][:, None] * np.abs(model.jumps)
        for b in BUDGETS:
            saved = masks[rep][f"{RANKINGS['abs_grad']}@{b}"]
            rebuilt = top_budget(score, int(saved.sum()))
            checks.append(dict(replication=rep, budget=float(b), saved_bits=int(saved.sum()),
                               overlap=float((saved & rebuilt).sum() / saved.sum())))
        model.set_displacement(np.zeros(model.n))
        clean_cal, _ = model.loss_and_grad(xc, yc)
        clean_eval_loss, clean_eval_acc = model.loss_acc(xe, ye)
        for r, key in RANKINGS.items():
            for b in BUDGETS:
                mk = masks[rep][f"{key}@{b}"]
                keep = ~mk
                up, down = first_order_split(mom, model.jumps, keep)
                mean_effect = mom["g1"][:, None] * model.jumps
                shift = np.where(keep, cj, 0).sum()
                rows_a.append(dict(checkpoint=name, replication=rep, ranking=r, budget=float(b),
                                   protected_increasing_share=float((mean_effect[mk] > 0).mean()),
                                   unprotected_increasing=up, unprotected_decreasing=down, unprotected_net=up + down,
                                   net_over_increasing=(up + down) / up if up else float("nan"),
                                   code_shift_per_weight_per_p=float(shift / model.n),
                                   code_shift_lopsidedness=float(abs(shift) / np.where(keep, np.abs(cj), 0).sum()),
                                   mask_sha256=mask_sha256(mk)))
                for p in BERS:
                    mu, _ = displacement_moments(model.jumps, mk, p)
                    model.set_displacement(mu)
                    cal_loss = model.loss_acc(xc, yc)[0]
                    ev_loss, ev_acc = model.loss_acc(xe, ye)
                    rows_bc.append(dict(checkpoint=name, replication=rep, ranking=r, budget=float(b), ber=p,
                                        mean_weight_eval_accuracy_drop=clean_eval_acc - ev_acc,
                                        mean_weight_eval_loss_increase=ev_loss - clean_eval_loss,
                                        objective_mean_term=cal_loss - clean_cal,
                                        objective_variance_term=variance_term(m2, mk, p)))
        model.set_displacement(np.zeros(model.n))
        print(json.dumps(dict(replication=rep, seconds=round(time.time() - t0),
                              abs_overlap=[c["overlap"] for c in checks if c["replication"] == rep])), flush=True)
    for fname, rows in (("p0a_lopsidedness.csv", rows_a), ("p0bc_mean_weight.csv", rows_bc),
                        ("abs_mask_reproduction.csv", checks)):
        with open(out / fname, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
    (out / "provenance.json").write_text(json.dumps(dict(
        checkpoint=name, checkpoint_sha256=model.sha256, bundle=str(bundle), script_sha256=sha256_file(Path(__file__)),
        module_sha256=sha256_file(ROOT / "netsurf/mean_variance.py"), bers=BERS, budgets=BUDGETS), indent=1))


def stage_calib(name: str, threads: int) -> None:
    out = OUT / name / "p0d"
    out.mkdir(parents=True, exist_ok=False)
    model = Model(name, threads)
    rep = 1
    masks, _ = load_masks(model, bundle_dir(CHECKPOINTS[name]["bundle"]), [rep])
    start = masks[rep][f"{RANKINGS['abs_grad']}@0.125000"]
    xh, yh, ih = model.images("empirical_validation", 256, 2000)
    assert ih[0] == 27000
    Gh = model.per_example(xh, yh)
    m2h = bit_second_moments(model.jumps, (Gh * Gh).mean(0))
    model.set_displacement(np.zeros(model.n))
    clean_h, _ = model.loss_and_grad(xh, yh)
    heldout_term = model.mean_term(xh, yh, clean_h)

    def heldout(mask):
        total = 0.0
        for p in RANGE_W:
            mu, _ = displacement_moments(model.jumps, mask, p)
            total += heldout_term(mu)[0] + variance_term(m2h, mask, p)
        return total / len(RANGE_W)

    reference = heldout(start)
    rows = []
    for n_cal in (64, 128, 256, 512):
        t0 = time.time()
        xc, yc, _ = model.images("ranking", n_cal, 1000 * rep)
        G = model.per_example(xc, yc)
        m2 = bit_second_moments(model.jumps, (G * G).mean(0))
        model.set_displacement(np.zeros(model.n))
        clean_c, _ = model.loss_and_grad(xc, yc)
        result = greedy_mean_variance_mask(start, model.jumps, m2, RANGE_W, model.mean_term(xc, yc, clean_c))
        value = heldout(result.mask)
        rows.append(dict(checkpoint=name, n_calibration=n_cal, calibration_objective_start=result.initial_objective,
                         calibration_objective_end=result.objective, heldout_objective_abs_grad=reference,
                         heldout_objective_new=value, heldout_reduction=reference - value,
                         rounds=len(result.trace) - 1, changed_bits=int((result.mask & ~start).sum()),
                         mask_sha256=mask_sha256(result.mask), seconds=round(time.time() - t0)))
        np.save(out / f"mask_n{n_cal}.npy", np.packbits(result.mask.reshape(-1), bitorder="little"))
        (out / f"trace_n{n_cal}.json").write_text(json.dumps(result.trace, indent=0))
        print(json.dumps(rows[-1]), flush=True)
    with open(out / "p0d_calibration_size.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def stage_curv(name: str, threads: int, probes: int = 64) -> None:
    out = OUT / name / "p0e"
    out.mkdir(parents=True, exist_ok=False)
    model = Model(name, threads)
    tf = model.tf
    xc, yc, _ = model.images("ranking", 64, 1000)
    G = model.per_example(xc, yc)
    fisher = (G * G).mean(0)
    model.set_displacement(np.zeros(model.n))
    variables = [model.comp.variables[n] for n in model.names]
    rng = np.random.default_rng(20261111)
    acc = np.zeros(model.n)
    for _ in range(probes):
        v = rng.choice((-1.0, 1.0), size=model.n)
        parts = [tf.constant(v[model.starts[i]: model.starts[i + 1]].reshape(model.shapes[n]), dtype=tf.float32)
                 for i, n in enumerate(model.names)]
        with tf.GradientTape() as outer:
            with tf.GradientTape() as inner:
                loss = tf.reduce_mean(model.comp.losses(xc, yc))
            grads = inner.gradient(loss, variables)
            dot = tf.add_n([tf.reduce_sum(g * p) for g, p in zip(grads, parts)])
        hv = outer.gradient(dot, variables)
        hv = np.concatenate([(h.numpy() if h is not None else np.zeros(model.shapes[n])).reshape(-1)
                             for h, n in zip(hv, model.names)])
        acc += v * hv
    hutch = acc / probes
    from scipy import stats
    rho = stats.spearmanr(hutch, fisher).statistic
    top = fisher >= np.quantile(fisher, 0.9)
    rho_top = stats.spearmanr(hutch[top], fisher[top]).statistic
    res = dict(checkpoint=name, probes=probes, spearman_all=float(rho), spearman_top_decile_fisher=float(rho_top),
               negative_hutchinson_share=float((hutch < 0).mean()))
    (out / "p0e_curvature.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["stats", "calib", "curv"])
    ap.add_argument("--checkpoint", choices=sorted(CHECKPOINTS), required=True)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    {"stats": stage_stats, "calib": stage_calib, "curv": stage_curv}[a.stage](a.checkpoint, a.threads)


if __name__ == "__main__":
    main()
