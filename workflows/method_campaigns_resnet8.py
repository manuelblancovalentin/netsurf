#!/usr/bin/env python3
"""Run N (frozen before any outcome): five ways of building a protection mask for the pinned ResNet-8, compared on
the evaluation protocol of Run M.  Design record: review/RUN_N_METHOD_CAMPAIGNS_FROZEN_PLAN_2026-11-11.md.

Common setting.  INT8 kernel bits only (618,880).  Gradients on the 64 calibration images of Run A replication 2
(training images 2000-2063).  Search objectives on 256 validation images (training images 27000-27255, disjoint
from the calibration images and from all test images).  Final evaluation on test images 2000-2999 with the
16 fault fields per rate of Run M (same seeds), rates 3e-3, 1e-2, 3e-2, budgets 12.5% and 25%.

  c1  MC-guided search ("MC search"): start from the engine's absolute-gradient mask; each round draws 4 new fault
      fields under the current mask, averages per-example |gradient| on the faulted models, keeps a running mean of
      the bit scores over rounds, and proposes three masks (all swaps towards the top-B of the running score, the
      best half, the best quarter); each proposal and the current mask are evaluated on the same 12 validation
      fault fields (mean cross-entropy on the 256 validation images) and the best proposal replaces the current
      mask only if its loss is lower.  8 rounds.  Per target cell.
  c2  Relaxed mask ("relaxed"): a protection probability per bit, initialised from the engine's mask (0.88 / 0.12);
      each step samples a fault field, applies the faults scaled by (1 - protection) to the decoded weights of the
      differentiable companion, and moves the probabilities along the batch gradient of the loss (Adam, rate
      0.05) with the budget restored after every step; steps = clip(10/p, 500, 4000); the final mask is the top-B.
  c3  Group search ("group CEM"): 80 groups = 10 kernel tensors x 8 bit positions; the cross-entropy method over
      group inclusion probabilities (24 samples per iteration, 6 elites, smoothing 0.7, 8 iterations), each sample
      filling whole groups in sampled order and the last group partially in absolute-gradient order; objective as
      in c1 on 8 validation fault fields; initial probabilities from the engine mask's group coverage.  Best
      sample kept.
  c45 Thales and HarDNN masks from the exhaustive single-flip table of Run N0 (independent of p):
      Thales: bits ranked by single-flip accuracy loss on the calibration images, ties by cross-entropy increase
      (Thales' A(j) with uniform fault probability; the protection ranking is our adaptation).
      Thales-loss: bits ranked by single-flip cross-entropy increase.
      HarDNN-loss: output filters (one per output channel) ranked by mean |cross-entropy change| over their bits;
      whole filters protected in that order; the last filter filled in significance order.
      HarDNN-gradient: the same with the filter's mean per-example |gradient| (HarDNN's best cheap heuristic).
  evaluate  every mask, plus the engine's absolute-gradient and MSB-first masks and the Run M rebuilt masks, on the
      final protocol.  analyze: paired gains against the absolute-gradient mask.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fault_aware_resnet8 as fa  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "workspace/runs/paper1_runN_method_campaigns_resnet8_v1"
RUN_M = ROOT / "workspace/runs/paper1_runM_fault_aware_resnet8_v1"
RUN_N0 = ROOT / "workspace/runs/paper1_runN0_single_flip_resnet8_v1"
VAL_OFFSET, VAL_COUNT = 2000, 256
SEED = 20261121


def context():
    adapter, Companion = fa.setup()
    storage, values, names, starts, absj, codes = fa.flat(adapter)
    signed = np.concatenate([adapter.bit_perturbations()[n].reshape(-1, 8) for n in names]).astype(np.float64)
    xc, yc, pc = adapter.sample_partition("ranking", fa.CAL, offset=fa.OFFSET)
    xv, yv, pv = adapter.sample_partition("empirical_validation", VAL_COUNT, offset=VAL_OFFSET)
    assert pc.indices[0] == fa.OFFSET and pv.indices[0] == 25000 + VAL_OFFSET
    masks = json.loads((RUN_M / f"fixed_masks_replication_{fa.REP}.json").read_text())
    fixed = {f: fa.mask_from_list(masks[f], names, starts) for f in fa.FRACTIONS}
    return dict(adapter=adapter, Companion=Companion, storage=storage, values=values, names=names, starts=starts,
                absj=absj, signed=signed, codes=codes, xc=xc, yc=yc, xv=xv, yv=yv, fixed=fixed)


def loss_on(ctx, mask, fields):
    """Mean cross-entropy and accuracy on the validation images over the given fault fields (bool arrays)."""
    ce, acc = [], []
    for draws in fields:
        fc = ctx["codes"] ^ np.packbits(draws & ~mask, axis=1, bitorder="little")[:, 0]
        with ctx["adapter"].temporary_values(fa.to_values(fc, ctx["values"], ctx["names"], ctx["starts"])):
            r = ctx["adapter"].evaluate(ctx["xv"], ctx["yv"])
        ce.append(r.categorical_crossentropy); acc.append(r.accuracy)
    return float(np.mean(ce)), float(np.mean(acc))


def swaps(current, score, m):
    """Move m bits: protect the m highest-scoring unprotected bits, release the m lowest-scoring protected bits."""
    s = score.reshape(-1); cur = current.reshape(-1).copy()
    inc = np.flatnonzero(~cur); out = np.flatnonzero(cur)
    m = min(m, inc.size, out.size)
    if m <= 0:
        return current.copy()
    add = inc[np.argpartition(-s[inc], m - 1)[:m]]; rem = out[np.argpartition(s[out], m - 1)[:m]]
    cur[add] = True; cur[rem] = False
    return cur.reshape(current.shape)


def c1(cell: int) -> None:
    p, frac = fa.CELLS[cell]; dest = OUT / "c1" / f"p{p:g}_f{float(frac):g}"; dest.mkdir(parents=True, exist_ok=False)
    ctx = context(); current = ctx["fixed"][frac].copy(); budget = int(current.sum())
    rng_v = np.random.default_rng(SEED + 100 * cell)
    val = [rng_v.random(current.shape) < p for _ in range(12)]
    j_cur, a_cur = loss_on(ctx, current, val)
    g0 = fa.mean_abs_grad(ctx["Companion"], ctx["storage"], ctx["codes"], ctx["values"], ctx["names"], ctx["starts"], ctx["xc"], ctx["yc"])
    total, rounds, trace, started = g0[:, None] * ctx["absj"], 1, [dict(round=0, loss=j_cur, accuracy=a_cur)], time.time()
    for r in range(1, 9):
        acc_g = np.zeros_like(g0)
        for k in range(4):
            rng = np.random.default_rng(SEED + 10000 + 1000 * cell + 10 * r + k)
            acc_g += fa.mean_abs_grad(ctx["Companion"], ctx["storage"], fa.faulted_codes(rng, ctx["codes"], current, p),
                                      ctx["values"], ctx["names"], ctx["starts"], ctx["xc"], ctx["yc"])
        total = total + (acc_g / 4)[:, None] * ctx["absj"]; rounds += 1
        score = total / rounds
        target = fa.top(score, budget); differing = int(np.count_nonzero(target & ~current))
        best = None
        for share in (1.0, 0.5, 0.25):
            cand = swaps(current, score, int(round(share * differing)))
            j, a = loss_on(ctx, cand, val)
            if best is None or j < best[0]:
                best = (j, a, share, cand)
        accepted = best[0] < j_cur
        if accepted:
            current, j_cur, a_cur = best[3], best[0], best[1]
        trace.append(dict(round=r, differing_from_target=differing, best_share=best[2], best_loss=best[0], accepted=bool(accepted),
                          loss=j_cur, accuracy=a_cur, changed_from_fixed=int(np.count_nonzero(current & ~ctx["fixed"][frac])),
                          seconds=round(time.time() - started)))
        print(json.dumps(trace[-1]), flush=True)
    np.savez(dest / "mask.npz", mask=np.packbits(current, axis=1, bitorder="little")[:, 0])
    (dest / "manifest.json").write_text(json.dumps(dict(method="mc_search", target_ber=p, fraction=float(frac), budget=budget, trace=trace), indent=1))


def c2(cell: int) -> None:
    import tensorflow as tf
    p, frac = fa.CELLS[cell]; dest = OUT / "c2" / f"p{p:g}_f{float(frac):g}"; dest.mkdir(parents=True, exist_ok=False)
    ctx = context(); fixed = ctx["fixed"][frac]; budget = int(fixed.sum())
    comp = ctx["Companion"](ctx["storage"].reference_content)
    names, starts = ctx["names"], ctx["starts"]
    shapes = {n: tuple(comp.variables[n].shape) for n in names}
    base = {n: comp.variables[n].numpy().astype(np.float64).reshape(-1) for n in names}
    theta = np.where(fixed, math.log(0.88 / 0.12), math.log(0.12 / 0.88)).astype(np.float64)
    m1 = np.zeros_like(theta); m2 = np.zeros_like(theta); b1, b2, lr, eps = 0.9, 0.999, 0.05, 1e-8
    steps = int(min(4000, max(500, math.ceil(10 / p))))

    def project(th):
        lo, hi = -50.0, 50.0                                   # shift so that sum(sigmoid) == budget
        for _ in range(60):
            mid = (lo + hi) / 2
            if (1 / (1 + np.exp(-(th + mid)))).sum() > budget:
                hi = mid
            else:
                lo = mid
        return th + (lo + hi) / 2
    theta = project(theta)
    rng = np.random.default_rng(SEED + 20000 + 1000 * cell)
    started, trace = time.time(), []
    for t in range(1, steps + 1):
        pi = 1 / (1 + np.exp(-theta))
        z = rng.random(theta.shape) < p
        dz = np.where(z, (1 - pi) * ctx["signed"], 0.0).sum(axis=1)               # per-weight decoded change
        for i, n in enumerate(names):
            comp.variables[n].assign((base[n] + dz[starts[i]: starts[i + 1]]).reshape(shapes[n]).astype(np.float32))
        g = comp.gradients(ctx["xc"], ctx["yc"])
        gw = np.concatenate([np.asarray(g[n], np.float64).reshape(-1) for n in names])
        grad_pi = -np.where(z, ctx["signed"] * gw[:, None], 0.0)                    # dL/dpi_b
        grad = grad_pi * pi * (1 - pi)
        m1 = b1 * m1 + (1 - b1) * grad; m2 = b2 * m2 + (1 - b2) * grad * grad
        theta = theta - lr * (m1 / (1 - b1 ** t)) / (np.sqrt(m2 / (1 - b2 ** t)) + eps)
        theta = project(theta)
        if t % 250 == 0 or t == steps:
            mask = fa.top(theta, budget)
            trace.append(dict(step=t, changed_from_fixed=int(np.count_nonzero(mask & ~fixed)), seconds=round(time.time() - started)))
            print(json.dumps(trace[-1]), flush=True)
    mask = fa.top(theta, budget)
    rng_v = np.random.default_rng(SEED + 100 * cell)
    val = [rng_v.random(mask.shape) < p for _ in range(12)]
    j, a = loss_on(ctx, mask, val); j0, a0 = loss_on(ctx, fixed, val)
    np.savez(dest / "mask.npz", mask=np.packbits(mask, axis=1, bitorder="little")[:, 0])
    (dest / "manifest.json").write_text(json.dumps(dict(method="relaxed", target_ber=p, fraction=float(frac), budget=budget, steps=steps,
                                                         validation=dict(loss=j, accuracy=a, fixed_loss=j0, fixed_accuracy=a0), trace=trace), indent=1))


def c3(cell: int) -> None:
    p, frac = fa.CELLS[cell]; dest = OUT / "c3" / f"p{p:g}_f{float(frac):g}"; dest.mkdir(parents=True, exist_ok=False)
    ctx = context(); fixed = ctx["fixed"][frac]; budget = int(fixed.sum()); starts = ctx["starts"]
    g0 = fa.mean_abs_grad(ctx["Companion"], ctx["storage"], ctx["codes"], ctx["values"], ctx["names"], starts, ctx["xc"], ctx["yc"])
    within = g0[:, None] * ctx["absj"]                                             # order inside a partly filled group
    groups = [(t, b) for t in range(len(ctx["names"])) for b in range(8)]
    size = np.array([starts[t + 1] - starts[t] for t, b in groups])
    q = np.array([fixed[starts[t]: starts[t + 1], b].mean() for t, b in groups]).clip(0.02, 0.98)
    rng = np.random.default_rng(SEED + 30000 + 1000 * cell)
    rng_v = np.random.default_rng(SEED + 100 * cell)
    val = [rng_v.random(fixed.shape) < p for _ in range(8)]

    def realize(include, keys):
        order = sorted(range(len(groups)), key=lambda g: (not include[g], keys[g]))
        mask = np.zeros_like(fixed); left = budget
        for g in order:
            t, b = groups[g]
            if left <= 0:
                break
            if size[g] <= left:
                mask[starts[t]: starts[t + 1], b] = True; left -= size[g]
            else:
                idx = np.argsort(-within[starts[t]: starts[t + 1], b], kind="stable")[:left]
                mask[starts[t] + idx, b] = True; left = 0
        assert int(mask.sum()) == budget
        return mask
    best, trace, started = None, [], time.time()
    for it in range(8):
        samples = []
        for _ in range(24):
            include = rng.random(len(groups)) < q; keys = rng.random(len(groups)) - q
            mask = realize(include, keys)
            j, a = loss_on(ctx, mask, val)
            samples.append((j, a, include, mask))
            if best is None or j < best[0]:
                best = (j, a, mask)
        samples.sort(key=lambda s: s[0])
        elite = np.mean([s[2] for s in samples[:6]], axis=0)
        q = (0.7 * q + 0.3 * elite).clip(0.02, 0.98)
        trace.append(dict(iteration=it + 1, best_loss=best[0], best_accuracy=best[1], elite_loss=float(np.mean([s[0] for s in samples[:6]])),
                          seconds=round(time.time() - started)))
        print(json.dumps(trace[-1]), flush=True)
    np.savez(dest / "mask.npz", mask=np.packbits(best[2], axis=1, bitorder="little")[:, 0])
    (dest / "manifest.json").write_text(json.dumps(dict(method="group_cem", target_ber=p, fraction=float(frac), budget=budget, trace=trace,
                                                         final_group_probabilities=[float(v) for v in q],
                                                         groups=[[ctx["names"][t], b] for t, b in groups]), indent=1))


def c45() -> None:
    dest = OUT / "c45"; dest.mkdir(parents=True, exist_ok=False)
    ctx = context(); starts, names, values = ctx["starts"], ctx["names"], ctx["values"]
    shards = sorted(RUN_N0.glob("shard_*.npz")); assert len(shards) == 32, len(shards)
    acc = np.full(ctx["codes"].shape + (8,), np.nan); ce = np.full_like(acc, np.nan)
    for s in shards:
        d = np.load(s); acc[int(d["lo"]): int(d["hi"])] = d["accuracy"]; ce[int(d["lo"]): int(d["hi"])] = d["cross_entropy"]
        clean_acc, clean_ce = float(d["clean_accuracy"]), float(d["clean_cross_entropy"])
    assert not np.isnan(acc).any()
    g0 = fa.mean_abs_grad(ctx["Companion"], ctx["storage"], ctx["codes"], values, names, starts, ctx["xc"], ctx["yc"])
    filters = []                                                                    # (start code, end code) per output channel
    for i, n in enumerate(names):
        rows = values[n].shape[0]; per = values[n].size // rows
        filters += [(starts[i] + r * per, starts[i] + (r + 1) * per) for r in range(rows)]
    out = {}
    for frac in fa.FRACTIONS:
        budget = int(ctx["fixed"][frac].sum())
        drop = clean_acc - acc; rise = ce - clean_ce
        thales = fa.top(drop * 1e6 + rise, budget)                                  # accuracy loss first, loss increase breaks ties
        thales_loss = fa.top(rise, budget)
        for name, density in (("hardnn_loss", [float(np.abs(ce[a:b] - clean_ce).mean()) for a, b in filters]),
                              ("hardnn_gradient", [float(g0[a:b].mean()) for a, b in filters])):
            mask = np.zeros_like(ctx["fixed"][frac]); left = budget
            for f in np.argsort(-np.asarray(density), kind="stable"):
                a, b = filters[f]; bits = (b - a) * 8
                if bits <= left:
                    mask[a:b] = True; left -= bits
                else:
                    for bit in range(7, -1, -1):
                        take = min(left, b - a); mask[a:a + take, bit] = True; left -= take
                        if left == 0:
                            break
                if left == 0:
                    break
            assert int(mask.sum()) == budget
            out[f"{name}@{frac}"] = mask
        out[f"thales@{frac}"] = thales; out[f"thales_loss@{frac}"] = thales_loss
    np.savez(dest / "masks.npz", **{k: np.packbits(v, axis=1, bitorder="little")[:, 0] for k, v in out.items()})
    (dest / "manifest.json").write_text(json.dumps(dict(clean_accuracy=clean_acc, clean_cross_entropy=clean_ce, filters=len(filters),
                                                         single_flip_bits_with_accuracy_loss=int((clean_acc - acc > 0).sum()),
                                                         masks=sorted(out)), indent=1))
    print(json.dumps(dict(filters=len(filters), masks=sorted(out))), flush=True)


def load_masks(ctx):
    masks = {}
    for f in fa.FRACTIONS:
        masks[("abs_grad_fixed", f, None)] = ctx["fixed"][f]
    bundle = next(d for d in fa.RUN_A.iterdir() if (d / "FINALIZED").is_file())
    for p, f in fa.CELLS:
        d = RUN_M / "build" / f"p{p:g}_f{float(f):g}"
        masks[("rebuild", f, p)] = np.unpackbits(np.load(d / "rebuilt_mask.npy")[:, None], axis=1, bitorder="little").astype(bool)
        for method in ("c1", "c2", "c3"):
            mf = OUT / method / f"p{p:g}_f{float(f):g}" / "mask.npz"
            if mf.is_file():
                masks[({"c1": "mc_search", "c2": "relaxed", "c3": "group_cem"}[method], f, p)] = np.unpackbits(np.load(mf)["mask"][:, None], axis=1, bitorder="little").astype(bool)
    c45f = OUT / "c45" / "masks.npz"
    if c45f.is_file():
        d = np.load(c45f)
        for key in d.files:
            name, f = key.split("@"); masks[(name, f, None)] = np.unpackbits(d[key][:, None], axis=1, bitorder="little").astype(bool)
    return masks, bundle


def evaluate(shard: int, shards: int) -> None:
    ctx = context()
    masks, bundle = load_masks(ctx)
    msb = json.loads((bundle / "artifacts/protection_masks.json").read_text())[f"replication_{fa.REP}"]
    for f in fa.FRACTIONS:
        masks[("msb_first", f, None)] = fa.mask_from_list(msb[f"msb_global@{f}"], ctx["names"], ctx["starts"])
    del msb
    keys = sorted(masks, key=lambda k: (k[0], k[1], k[2] or 0))
    x, y, part = ctx["adapter"].sample_partition("evaluation", fa.EVAL, offset=fa.OFFSET)
    rows = []
    for idx, key in enumerate(keys):
        if idx % shards != shard:
            continue
        for a, p in enumerate(fa.TARGETS):
            for i in range(fa.EVAL_FIELDS):
                draws = np.random.default_rng(fa.EVAL_SEED + 1000 * a + i).random(masks[key].shape) < p      # Run M's fields
                fc = ctx["codes"] ^ np.packbits(draws & ~masks[key], axis=1, bitorder="little")[:, 0]
                with ctx["adapter"].temporary_values(fa.to_values(fc, ctx["values"], ctx["names"], ctx["starts"])):
                    r = ctx["adapter"].evaluate(x, y)
                rows.append(dict(method=key[0], fraction=float(key[1]), target_ber=key[2], test_ber=p, field=i, accuracy=r.accuracy,
                                 cross_entropy=r.categorical_crossentropy))
        print(json.dumps(dict(shard=shard, mask=list(map(str, key)))), flush=True)
    (OUT / "eval").mkdir(exist_ok=True)
    (OUT / "eval" / f"shard_{shard:02d}.json").write_text(json.dumps(rows))


def analyze() -> None:
    from scipy import stats
    rows = [r for f in sorted((OUT / "eval").glob("shard_*.json")) for r in json.loads(f.read_text())]
    acc = {}
    for r in rows:
        acc.setdefault((r["method"], r["fraction"], r["target_ber"], r["test_ber"]), {})[r["field"]] = r["accuracy"]
    res = []
    for (m, f, tp, p), d in sorted(acc.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2] or 0, kv[0][3])):
        base = acc[("abs_grad_fixed", f, None, p)]
        diff = np.array([d[i] - base[i] for i in sorted(d)])
        mu = float(diff.mean()); h = float(stats.t.ppf(0.975, len(diff) - 1) * diff.std(ddof=1) / math.sqrt(len(diff))) if diff.std() > 0 else 0.0
        res.append(dict(method=m, fraction=f, target_ber=tp, test_ber=p, on_target=(tp is None or tp == p), accuracy=float(np.mean(list(d.values()))),
                        gain_vs_abs_grad=mu, ci=[mu - h, mu + h], matters=bool((tp is None or tp == p) and mu >= 0.01 and mu - h > 0)))
    (OUT.parent / (OUT.name + "_analysis.json")).write_text(json.dumps(res, indent=1))
    for r in res:
        if r["on_target"]:
            print(f"{r['method']:16s} B={r['fraction']:<6g} p={r['test_ber']:<6g} acc {100*r['accuracy']:5.1f}  vs abs-grad {100*r['gain_vs_abs_grad']:+6.2f} "
                  f"[{100*r['ci'][0]:+.2f},{100*r['ci'][1]:+.2f}]{'  matters' if r['matters'] else ''}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["c1", "c2", "c3", "c45", "evaluate", "analyze"])
    ap.add_argument("--cell", type=int, default=0); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--shards", type=int, default=1)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"c1": lambda: c1(a.cell), "c2": lambda: c2(a.cell), "c3": lambda: c3(a.cell), "c45": c45,
     "evaluate": lambda: evaluate(a.shard, a.shards), "analyze": analyze}[a.stage]()


if __name__ == "__main__":
    main()
