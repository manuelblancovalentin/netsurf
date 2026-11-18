#!/usr/bin/env python3
"""Run R, Stage 0 checks that need the model (plan Section 3; rules in `RUN_R_STAGE0_RECORD_2026-11-12.md`).

  parity     the new search, with ideal suppression and no correction, must rebuild Run P's
             `mv_single@0.125000@0.001` mask of pinned replication 1 bit for bit (same 512 calibration images,
             same start, 2 TensorFlow threads as in Run P)
  abft       exact ABFT pair outcomes for every output channel of every pinned kernel (32-bit checksum words)
  constants  Stage 2 constants by the rule fixed in the record, on held-out training images 28000-28511 only
             (pinned, replication-1 budget at 12.5%); every search trace is saved

Output: <out>/stage0_checks.json plus traces.  No evaluation image is used.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from runP_phase0 import HELDOUT_COUNT, HELDOUT_OFFSET, ROOT, Model, sha256_file
from runP_phase12 import OUT as RUNP, unpack
from netsurf.mean_variance import bit_second_moments, mask_sha256, top_budget
from netsurf.rate_aware import (IDEAL_LADDER, abft_pair_outcomes, analytic_evaluator, descend, ladder_rates,
                                mc_evaluator, mc_uniforms, value_ranking_start)

KS = (8, 16, 32)
TUNE_RATES = (3e-3, 1e-2)
MC_SEED = (20261118, 0)


def parity(model, out):
    rep, b, p = 1, "0.125000", 1e-3
    t0 = time.time()
    x, y, idx = model.images("ranking", 512, 1000 * rep)
    assert idx[0] == 1000 and len(idx) == 512
    G = model.per_example(x, y)
    gabs, g2 = np.abs(G).mean(0), (G * G).mean(0)
    del G
    m2 = bit_second_moments(model.jumps, g2)
    model.set_displacement(np.zeros(model.n))
    clean, _ = model.loss_and_grad(x, y)
    term = model.mean_term(x, y, clean)
    saved = np.load(RUNP / "pinned" / "saved_absgrad_masks.npz")
    budget = int(unpack(saved[f"r{rep}@{b}"], model.n).sum())
    a512 = top_budget(gabs[:, None] * np.abs(model.jumps), budget)
    rho, step = ladder_rates(IDEAL_LADDER, p)
    res = descend(a512.astype(np.int8), model.jumps, m2, rho, step, analytic_evaluator(model.jumps, m2, rho, term))
    ref = unpack(np.load(RUNP / "pinned" / f"rep{rep}" / "masks.npz")[f"mv_single@{b}@{p:g}"], model.n)
    got = res.levels.astype(bool)
    model.set_displacement(np.zeros(model.n))
    (out / "trace_parity.json").write_text(json.dumps(res.trace, indent=0, default=float))
    return dict(identical=bool(np.array_equal(got, ref)), differing_bits=int((got != ref).sum()),
                new_sha256=mask_sha256(got), run_p_sha256=mask_sha256(ref), rounds=len(res.trace) - 1,
                objective=res.objective, seconds=round(time.time() - t0))


def abft(model):
    t0 = time.time()
    layers = []
    for i, n in enumerate(model.names):
        v = model.codes[model.starts[i]: model.starts[i + 1]].reshape(model.shapes[n][0], -1).astype(np.int64)
        tot = dict(ww_pairs=0, ww_wrong=0, ww_cancel=0, wc_pairs=0, wc_wrong=0)
        for ch in v:
            o = abft_pair_outcomes(ch, 32)
            tot["ww_pairs"] += o["ww"]["pairs"]; tot["ww_wrong"] += o["ww"]["wrong"]; tot["ww_cancel"] += o["ww"]["cancel"]
            tot["wc_pairs"] += o["wc"]["pairs"]; tot["wc_wrong"] += o["wc"]["wrong"]
        layers.append(dict(layer=n, channels=int(v.shape[0]), G=int(v.shape[1]), **tot,
                           ww_wrong_share=tot["ww_wrong"] / tot["ww_pairs"], ww_cancel_share=tot["ww_cancel"] / tot["ww_pairs"],
                           wc_wrong_share=tot["wc_wrong"] / tot["wc_pairs"]))
    agg = {k: sum(l[k] for l in layers) for k in ("ww_pairs", "ww_wrong", "ww_cancel", "wc_pairs", "wc_wrong")}
    return dict(layers=layers, ww_wrong_share=agg["ww_wrong"] / agg["ww_pairs"],
                ww_cancel_share=agg["ww_cancel"] / agg["ww_pairs"], wc_wrong_share=agg["wc_wrong"] / agg["wc_pairs"],
                seconds=round(time.time() - t0))


def constants(model, out):
    rec = dict(rule="see RUN_R_STAGE0_RECORD_2026-11-12.md, Section 2", rates={})
    xh, yh, ih = model.images("empirical_validation", HELDOUT_COUNT, HELDOUT_OFFSET)
    assert ih[0] == 28000 and ih[-1] == 28511
    G = model.per_example(xh, yh)
    gabs, g2, gbar = np.abs(G).mean(0), (G * G).mean(0), G.mean(0)
    se = G.std(0, ddof=1) / math.sqrt(G.shape[0])
    del G
    m2 = bit_second_moments(model.jumps, g2)
    model.set_displacement(np.zeros(model.n))
    clean, _ = model.loss_and_grad(xh, yh)
    term = model.mean_term(xh, yh, clean)
    saved = np.load(RUNP / "pinned" / "saved_absgrad_masks.npz")
    budget = int(unpack(saved["r1@0.125000"], model.n).sum())
    abs_start = top_budget(gabs[:, None] * np.abs(model.jumps), budget).astype(np.int8)
    u32 = mc_uniforms(model.n, max(KS), MC_SEED)
    for p in TUNE_RATES:
        rho, step = ladder_rates(IDEAL_LADDER, p)
        ev = analytic_evaluator(model.jumps, m2, rho, term)
        vstart = value_ranking_start(model.jumps, m2, rho, step, gbar, budget)
        t0 = time.time()
        res = descend(abs_start, model.jumps, m2, rho, step, ev, se=se)
        t_an = time.time() - t0
        res_v = descend(vstart, model.jumps, m2, rho, step, ev, se=se)
        (out / f"trace_analytic_{p:g}.json").write_text(json.dumps(dict(abs_start=res.trace, value_start=res_v.trace),
                                                                    indent=0, default=float))
        losses = []
        for u in u32:
            from netsurf.rate_aware import mc_displacement
            lk, _ = term(mc_displacement(model.jumps, abs_start, rho, u))
            losses.append(float(lk))
        losses = np.array(losses)
        se_k = {K: float(losses[:K].std(ddof=1) / math.sqrt(K)) for K in KS}
        improvement = float(res.initial_objective - res.objective)
        rec["rates"][f"{p:g}"] = dict(J_abs_start=res.initial_objective, J_analytic_final=res.objective,
                                      J_value_start_final=res_v.objective, analytic_improvement=improvement,
                                      analytic_seconds=round(t_an), analytic_rounds=len(res.trace) - 1,
                                      mc_se_at_abs_start=se_k)
    ok = [K for K in KS if all(v["mc_se_at_abs_start"][K] < 0.1 * v["analytic_improvement"] for v in rec["rates"].values())]
    K = ok[0] if ok else max(KS)
    rec["K"] = K
    rho, step = ladder_rates(IDEAL_LADDER, 3e-3)
    t0 = time.time()
    res_mc = descend(abs_start, model.jumps, m2, rho, step, mc_evaluator(model.jumps, m2, rho, term, u32[:K], refresh=1), se=se)
    t_mc = time.time() - t0
    rec["refresh"] = 1 if t_mc < 600 else 4
    rec["mc_build_seconds_refresh1"] = round(t_mc)
    rec["mc_final"] = dict(objective=res_mc.objective, initial=res_mc.initial_objective, rounds=len(res_mc.trace) - 1)
    t0 = time.time()
    res_an = descend(abs_start, model.jumps, m2, rho, step,
                     mc_evaluator(model.jumps, m2, rho, term, u32[:K], refresh=rec["refresh"]), se=se, anneal=True, seed=0)
    rec["anneal_build_seconds"] = round(time.time() - t0)
    rec["anneal_final"] = dict(objective=res_an.objective, initial=res_an.initial_objective, rounds=len(res_an.trace) - 1)
    (out / "trace_mc_0.003.json").write_text(json.dumps(dict(mc=res_mc.trace, anneal=res_an.trace), indent=0, default=float))
    rec["max_rounds"], rec["min_swap"], rec["anneal_factor"] = 40, 16, 0.9
    model.set_displacement(np.zeros(model.n))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", nargs="*", default=["parity", "abft", "constants"])
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    model = Model("pinned", 2)
    result = dict(script_sha256=sha256_file(Path(__file__)), module_sha256=sha256_file(ROOT / "netsurf/rate_aware.py"))
    for step in a.only:
        result[step] = dict(parity=parity, abft=lambda m, o: abft(m), constants=constants)[step](model, a.out)
        print(json.dumps({step: {k: v for k, v in result[step].items() if k != "layers"}}, default=float), flush=True)
        (a.out / "stage0_checks.json").write_text(json.dumps(result, indent=1, default=float))


if __name__ == "__main__":
    main()
