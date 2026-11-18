#!/usr/bin/env python3
"""Run R, Stage 4 (DS-CNN) analysis, by the frozen plan `RUN_R_STAGE4_DSCNN_PLAN_2026-11-12.md`.  Write-once.

Committed before any Stage 4 outcome is read.  Refuses to run unless all 48 evaluation jobs are finished and the
predictions file matches its frozen hash.

Switch (pos1024 vs abs1024): a cell (budget, rate) is usable if the better of the two masks is within 0.05 of the
mean clean accuracy of the windows (mean over replications of each window's clean accuracy).  Difference pos - abs,
paired on the field (48 fields); pooled-field sign-flip test, 100,000 assignments, seed 20261121; Holm over usable
cells.  A difference of exactly zero counts against the prediction.  Rules R4a, R4b as in Stage 1.  R4c: for each
replication, the measured switch per budget (last sign change of that replication's mean difference over the usable
rates, log-linear interpolation), log ratio to p*, mean over that replication's budgets with a measured switch;
95% Student-t interval over replications, mapped to a ratio, inside [1/1.5, 1.5].

Method (R4d): a cell is usable if the best arm is within 0.05 of clean.  For every static arm, difference static -
analytic rate-aware mask built for the tested rate, paired on the field; two-sided pooled-field sign-flip test, Holm
over all usable cells x static arms.  A violation is a cell and arm whose mean difference is 0.5 point or more and
whose Holm-adjusted p is below 0.05.

Output: workspace/runs/paper1_runR_stage4_kws_analysis_v1/{summary.json, switch_cells.csv, method_cells.csv,
stage4_switch.png, stage4_method.png}.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

from runP_phase0 import PAPER, sha256_file
from runR_kws import OUT as PREP, STAGE4_FRACTIONS
from runR_stage1 import RATES, bkey
from runR_stage4_kws import EVAL_OUT, FIELDS, REPS, STATIC

PRED = PREP / "predictions_6draws.json"
PRED_SHA = "c59fcfcd234c120db79307599eec31263b11fe1ed5447c68e8a870bf42df0093"
OUTD = PAPER / "workspace/runs/paper1_runR_stage4_kws_analysis_v1"
MARGIN, NEAR, LOC, GAP = 0.05, 1.25, 1.5, 0.5
SIGNS = np.random.default_rng(20261121).choice((-1.0, 1.0), size=(100_000, len(REPS) * FIELDS))


def signflip_p(d):
    d = np.asarray(d, dtype=float)
    if np.allclose(d, 0):
        return 1.0
    obs = abs(d.mean())
    null = np.abs((SIGNS[:, : d.size] * d).mean(axis=1))
    return float((1 + (null >= obs - 1e-15).sum()) / (SIGNS.shape[0] + 1))


def holm(ps):
    ps = np.asarray(ps, dtype=float)
    adj, running = np.empty(ps.size), 0.0
    for rank, i in enumerate(np.argsort(ps)):
        running = max(running, (ps.size - rank) * ps[i])
        adj[i] = min(1.0, running)
    return adj


def tint(x):
    x = np.asarray(x, dtype=float)
    h = stats.t.ppf(0.975, x.size - 1) * x.std(ddof=1) / math.sqrt(x.size) if x.size > 1 else float("nan")
    return float(x.mean() - h), float(x.mean() + h)


def crossing(ps, ds):
    last = None
    for (p0, d0), (p1, d1) in zip(zip(ps, ds), list(zip(ps, ds))[1:]):
        if (d0 > 0) != (d1 > 0):
            last = (p0, d0, p1, d1)
    if last is None:
        return None
    p0, d0, p1, d1 = last
    x0, x1 = math.log(p0), math.log(p1)
    return math.exp(x0 + (x1 - x0) * (-d0) / (d1 - d0))


def load():
    acc, clean, missing = defaultdict(dict), defaultdict(list), []
    for r in REPS:
        for f in range(FIELDS):
            d = EVAL_OUT / f"rep{r}" / f"field{f}"
            if not (d / "DONE").exists():
                missing.append(f"rep{r}/field{f}")
                continue
            clean[r].append(json.loads((d / "field.json").read_text())["clean_accuracy"])
            with open(d / "results.csv") as fh:
                for row in csv.DictReader(fh):
                    p = round(float(row["ber"]), 8)
                    if row["built_for"] and round(float(row["built_for"]), 8) != p:
                        continue
                    acc[(round(float(row["budget"]), 6), p, row["arm"])][(r, f)] = float(row["accuracy"])
    return acc, clean, missing


def main():
    if hashlib.sha256(PRED.read_bytes()).hexdigest() != PRED_SHA:
        raise SystemExit("predictions file does not match its frozen hash")
    pred = json.loads(PRED.read_text())["predictions"]
    acc, clean, missing = load()
    if missing:
        raise SystemExit(f"{len(missing)} Stage 4 jobs are not finished, e.g. {missing[:3]}")
    OUTD.mkdir(exist_ok=False)
    c0 = float(np.mean([np.mean(v) for v in clean.values()]))
    mean = lambda d: float(np.mean(list(d.values())))  # noqa: E731
    # ---------------- switch
    cells = []
    for f in STAGE4_FRACTIONS:
        ps = pred[bkey(f)]["p_star"]
        for p in RATES:
            P, B = acc[(round(f, 6), round(p, 8), "pos1024")], acc[(round(f, 6), round(p, 8), "abs1024")]
            keys = sorted(set(P) & set(B))
            d = np.array([P[k] - B[k] for k in keys]) * 100
            cells.append(dict(budget=f, ber=p, p_star=ps, diff=float(d.mean()), fields=int(d.size), p_raw=signflip_p(d),
                              usable=bool(max(mean(P), mean(B)) >= c0 - MARGIN), near=bool(ps / NEAR <= p <= ps * NEAR),
                              predicted_positive_ahead=bool(p < ps), mean_pos=mean(P), mean_abs=mean(B),
                              by_rep={r: float(np.mean([P[k] - B[k] for k in keys if k[0] == r]) * 100) for r in REPS}))
    use = [c for c in cells if c["usable"]]
    for c, a in zip(use, holm([c["p_raw"] for c in use])):
        c["p_holm"] = float(a)
    for c in cells:
        c.setdefault("p_holm", None)
        c["resolved"] = bool(c["usable"] and c["p_holm"] < 0.05)
        c["agrees"] = bool((c["diff"] > 0) == c["predicted_positive_ahead"] and c["diff"] != 0)
    away = [c for c in use if not c["near"]]
    r4a_viol = [c for c in away if c["resolved"] and not c["agrees"]]
    r4b = sum(c["agrees"] for c in away) / len(away) if away else float("nan")
    pooled, per_rep_logs, missing_rep = {}, defaultdict(list), 0
    for f in STAGE4_FRACTIONS:
        rows = sorted([c for c in use if c["budget"] == f], key=lambda c: c["ber"])
        ps = pred[bkey(f)]["p_star"]
        m = crossing([c["ber"] for c in rows], [c["diff"] for c in rows]) if len(rows) > 1 else None
        pooled[bkey(f)] = dict(predicted=ps, measured=m, ratio=(m / ps) if m else None)
        for r in REPS:
            mr = crossing([c["ber"] for c in rows], [c["by_rep"][r] for c in rows]) if len(rows) > 1 else None
            if mr is None:
                missing_rep += 1
            else:
                per_rep_logs[r].append(math.log(mr / ps))
    rep_means = [float(np.mean(v)) for v in per_rep_logs.values() if v]
    lo, hi = tint(rep_means) if len(rep_means) > 1 else (float("nan"), float("nan"))
    r4c = dict(replications=len(rep_means), missing_replication_budgets=missing_rep,
               ratio=math.exp(float(np.mean(rep_means))) if rep_means else None,
               low=math.exp(lo) if rep_means else None, high=math.exp(hi) if rep_means else None,
               holds=bool(rep_means and math.exp(lo) >= 1 / LOC and math.exp(hi) <= LOC))
    measured = [v for v in pooled.values() if v["measured"]]
    bias = math.exp(np.mean([math.log(v["ratio"]) for v in measured])) if measured else None
    rule_loss = [100 * (max(c["mean_pos"], c["mean_abs"]) - (c["mean_pos"] if c["ber"] < c["p_star"] else c["mean_abs"])) for c in use]
    # ---------------- method
    mcells = []
    for f in STAGE4_FRACTIONS:
        for p in RATES:
            R = acc[(round(f, 6), round(p, 8), "ra_analytic")]
            arms = {a: acc[(round(f, 6), round(p, 8), a)] for a in STATIC}
            best = max([mean(R)] + [mean(v) for v in arms.values()])
            if best < c0 - MARGIN:
                continue
            for a, S in arms.items():
                keys = sorted(set(S) & set(R))
                d = np.array([S[k] - R[k] for k in keys]) * 100
                mcells.append(dict(budget=f, ber=p, arm=a, diff=float(d.mean()), p_raw=signflip_p(d),
                                   mean_arm=mean(S), mean_rate_aware=mean(R)))
    for c, a in zip(mcells, holm([c["p_raw"] for c in mcells])):
        c["p_holm"] = float(a)
        c["violation"] = bool(c["diff"] >= GAP and a < 0.05)
    r4d_viol = [c for c in mcells if c["violation"]]
    largest_gap = {a: max([c["diff"] for c in mcells if c["arm"] == a], default=None) for a in STATIC}
    summary = dict(
        status="Run R Stage 4 (DS-CNN), analysis by the frozen plan", predictions_sha256=PRED_SHA,
        script_sha256=sha256_file(Path(__file__)), clean_accuracy=c0,
        switch=dict(cells=len(cells), usable=len(use), usable_away=len(away),
                    R4a=dict(holds=not r4a_viol, violations=[{k: c[k] for k in ("budget", "ber", "diff", "p_holm")} for c in r4a_viol]),
                    R4b=dict(holds=bool(r4b >= 0.8), share=r4b, agree=sum(c["agrees"] for c in away), of=len(away)),
                    R4c=r4c, confirmed=bool(not r4a_viol and r4b >= 0.8 and r4c["holds"]),
                    bias_geometric_mean_measured_over_predicted=bias, pooled=pooled,
                    switch_rule_loss_points=dict(mean=float(np.mean(rule_loss)) if rule_loss else None,
                                                 max=float(np.max(rule_loss)) if rule_loss else None)),
        method=dict(usable_cell_arm_pairs=len(mcells), R4d=dict(holds=not r4d_viol, violations=[
            {k: c[k] for k in ("budget", "ber", "arm", "diff", "p_holm")} for c in r4d_viol]),
            largest_gap_static_minus_rate_aware=largest_gap))
    (OUTD / "summary.json").write_text(json.dumps(summary, indent=1))
    for name, rows in (("switch_cells.csv", [{k: v for k, v in c.items() if k != "by_rep"} for c in cells]),
                       ("method_cells.csv", mcells)):
        if rows:
            with open(OUTD / name, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(json.dumps({k: summary[k] for k in ("clean_accuracy", "switch", "method")}, indent=1, default=str))
    plots(cells, pooled, per_rep_logs, mcells, pred)


def plots(cells, pooled, per_rep_logs, mcells, pred):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, MUTED, GRID, SURF, BLUE, ORANGE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff", "#2a78d6", "#eb6834"
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8), facecolor=SURF)
    xb = [100 * f for f in STAGE4_FRACTIONS]
    ps = [pred[bkey(f)]["p_star"] for f in STAGE4_FRACTIONS]
    a1.fill_between(xb, [p / LOC for p in ps], [p * LOC for p in ps], color=BLUE, alpha=0.12, lw=0)
    a1.plot(xb, ps, color=BLUE, lw=2, label="Predicted switch (frozen)")
    for x, f in zip(xb, STAGE4_FRACTIONS):
        v = pooled[bkey(f)]
        if v["measured"]:
            a1.plot([x], [v["measured"]], "D", ms=7, mfc=SURF, mec=INK, mew=1.4)
    a1.plot([], [], "D", ms=7, mfc=SURF, mec=INK, label="Measured, all fields")
    a1.set_xscale("log"); a1.set_yscale("log"); a1.set_xticks(xb); a1.set_xticklabels([f"{x:g}" for x in xb])
    a1.set_xlabel("Protected bits (%)"); a1.set_ylabel("Bit-error probability of the switch")
    a1.legend(frameon=False, fontsize=8); a1.grid(True, color=GRID, lw=0.6)
    use = [c for c in cells if c["usable"]]
    for f in STAGE4_FRACTIONS:
        rs = sorted([c for c in use if c["budget"] == f], key=lambda c: c["ber"])
        a2.plot([c["ber"] / c["p_star"] for c in rs], [c["diff"] for c in rs], "o-", ms=3.5, lw=1, alpha=0.8,
                label=f"{100 * f:g}%")
    a2.axhline(0, color=MUTED, lw=0.8); a2.axvline(1, color=MUTED, lw=0.8, ls="--")
    a2.set_xscale("log"); a2.set_yscale("symlog", linthresh=1)
    a2.set_xlabel("Bit-error probability / predicted switch"); a2.set_ylabel("Positive minus absolute (points)")
    a2.legend(frameon=False, fontsize=8, title="Budget"); a2.grid(True, color=GRID, lw=0.6)
    fig.tight_layout(); fig.savefig(OUTD / "stage4_switch.png", dpi=200, facecolor=SURF)
    if mcells:
        fig, ax = plt.subplots(figsize=(6, 3.8), facecolor=SURF)
        for a in STATIC:
            rs = sorted([c for c in mcells if c["arm"] == a], key=lambda c: (c["ber"], c["budget"]))
            ax.plot([c["ber"] for c in rs], [c["diff"] for c in rs], "o", ms=3.5, alpha=0.75, label=a)
        ax.axhline(0, color=MUTED, lw=0.8); ax.axhline(GAP, color=ORANGE, lw=0.8, ls="--")
        ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=1)
        ax.set_xlabel("Bit-error probability"); ax.set_ylabel("Static minus rate-aware (points)")
        ax.legend(frameon=False, fontsize=8); ax.grid(True, color=GRID, lw=0.6)
        fig.tight_layout(); fig.savefig(OUTD / "stage4_method.png", dpi=200, facecolor=SURF)


if __name__ == "__main__":
    main()
