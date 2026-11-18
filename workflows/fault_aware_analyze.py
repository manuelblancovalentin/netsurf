#!/usr/bin/env python3
"""Run F analysis (written before any candidate outcome existed).

Pairs every candidate observation with the finalized outcomes of the fixed rankings on the same saved
fault field.  Primary contrast: candidate built for (p, B), evaluated at (p, B), minus the fixed
absolute-gradient mask at B; 12 contrasts, sign-flip test (100,000 draws), Holm.  A gain "matters" if it is
at least one accuracy point and its 95% interval excludes zero.  Off-target evaluations are secondary.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fault_aware_masks as fa  # noqa: E402

STATIC = ("random", "msb_first", "numerical_bit_jump", "global_first_order", "per_example_absolute", "per_example_positive")
DRAWS, SEED = 100_000, 20261113


def ci(d):
    d = np.asarray(d, float); m = float(d.mean()); sd = float(d.std(ddof=1))
    if sd == 0:
        return m, m, m
    h = stats.t.ppf(0.975, len(d) - 1) * sd / math.sqrt(len(d))
    return m, m - h, m + h


def signflip(d, rng):
    d = np.asarray(d, float)
    if np.allclose(d, 0):
        return 1.0
    obs = abs(d.mean()); count = 0
    for _ in range(DRAWS // 10_000):
        s = rng.integers(0, 2, size=(10_000, len(d)), dtype=np.int8) * 2 - 1
        count += int(np.sum(np.abs((s * d).mean(axis=1)) >= obs - 1e-15))
    return (1 + count) / (DRAWS + 1)


def holm(p):
    p = np.asarray(p, float); idx = np.argsort(p); out = np.empty(len(p)); run = 0.0
    for rank, i in enumerate(idx):
        run = max(run, min(1.0, (len(p) - rank) * p[i])); out[i] = run
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    plan = json.loads((fa.SOURCE_RUN / "plan.json").read_text())
    bits = plan["eligible_bits"]; nf = len(plan["field_seeds"][0])
    rng = np.random.default_rng(SEED)
    clean = json.loads((fa.SOURCE_RUN / "clean/shard_00.json").read_text())["metrics"]

    def saved(arm, index, name):
        r = json.loads((fa.SOURCE_RUN / "fields" / f"arm_{arm:02d}" / f"field_{index:03d}" / f"{name}.json").read_text())["risk"]["metrics"]
        return r["accuracy"], r["mean_nll"]

    def cand(arm, index, name):
        r = json.loads((args.run / "eval" / f"arm_{arm:02d}" / f"field_{index:03d}" / f"{name}.json").read_text())["risk"]["metrics"]
        return r["accuracy"], r["mean_nll"]

    rows, primary = [], []
    for p in fa.TARGET_BERS:
        for frac in fa.FRACTIONS:
            budget = int(round(frac * bits))
            for c in fa.CANDIDATES:
                name = f"{c}_p{p:g}_f{frac:g}"
                for test in fa.TEST_BERS:
                    arm = plan["bers"].index(test)
                    cv = np.array([cand(arm, i, name) for i in range(nf)])
                    st = {m: np.array([saved(arm, i, f"{m}_b{budget}") for i in range(nf)]) for m in STATIC}
                    best = max(STATIC, key=lambda m: st[m][:, 0].mean())
                    d_acc = cv[:, 0] - st["per_example_absolute"][:, 0]; d_nll = cv[:, 1] - st["per_example_absolute"][:, 1]
                    m, lo, hi = ci(d_acc); mn, lon, hin = ci(d_nll)
                    row = dict(candidate=c, target_ber=p, fraction=frac, test_ber=test, on_target=bool(test == p), fields=nf,
                               candidate_accuracy=float(cv[:, 0].mean()), absolute_gradient_accuracy=float(st["per_example_absolute"][:, 0].mean()),
                               best_fixed=best, best_fixed_accuracy=float(st[best][:, 0].mean()), clean_accuracy=clean["accuracy"],
                               accuracy_gain_over_absolute=m, gain_ci_low=lo, gain_ci_high=hi,
                               nll_change_over_absolute=mn, nll_ci_low=lon, nll_ci_high=hin,
                               gain_over_best_fixed=float((cv[:, 0] - st[best][:, 0]).mean()),
                               p_signflip=signflip(d_acc, rng))
                    rows.append(row)
                    if test == p:
                        primary.append(row)
    for r, a in zip(primary, holm([r["p_signflip"] for r in primary])):
        r["p_holm_primary"] = float(a)
        r["gain_matters"] = bool(r["accuracy_gain_over_absolute"] >= 0.01 and r["gain_ci_low"] > 0)
        r["loss_matters"] = bool(r["accuracy_gain_over_absolute"] <= -0.01 and r["gain_ci_high"] < 0)
    summary = dict(schema="paper1-runF-analysis/1.0", status="EXPLORATORY", primary_contrasts=len(primary),
                   gains_that_matter=int(sum(r["gain_matters"] for r in primary)), losses_that_matter=int(sum(r["loss_matters"] for r in primary)),
                   holm_resolved=int(sum(r["p_holm_primary"] < 0.05 for r in primary)), primary=primary, all_rows=rows)
    with open(args.out, "x") as f:
        json.dump(summary, f, indent=1)
    for r in primary:
        print(f"{r['candidate']:26s} p={r['target_ber']:<5g} B={r['fraction']:<6g} cand {100*r['candidate_accuracy']:5.1f}  abs {100*r['absolute_gradient_accuracy']:5.1f}  "
              f"gain {100*r['accuracy_gain_over_absolute']:+6.2f} [{100*r['gain_ci_low']:+.2f}, {100*r['gain_ci_high']:+.2f}]  holm {r['p_holm_primary']:.3g}  best fixed {r['best_fixed']} {100*r['best_fixed_accuracy']:.1f}")


if __name__ == "__main__":
    main()
