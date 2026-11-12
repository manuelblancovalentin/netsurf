#!/usr/bin/env python3
"""Run R, Stage 1 analysis (plan Section 4, amendments R-A3, analysis note A1).  Write-once.

Reads every finished Stage 1 job (`paper1_runR_stage1_crossover_v1/<ckpt>/rep<r>/field<f>/`) and the frozen
predictions (`paper1_runR_predictions_v1/predictions.json`, sha256 checked).  For every cell (checkpoint, budget,
rate): usable if the mean accuracy of abs1024 or pos1024 is within 0.05 of the mean clean accuracy of the windows;
difference pos1024 - abs1024 paired on the field; pooled-field sign-flip test (100,000 assignments, seed 20261116);
Holm over all usable cells; Student-t interval over replications.  Then rules R1a, R1b, R1c and the secondary
comparison, and two figures.

Output: workspace/runs/paper1_runR_stage1_analysis_v1/{summary.json, cells.csv, stage1_by_group.png,
stage1_measured_vs_predicted.png}.
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
from runR_stage1 import FIELDS, FRACTIONS, NAMES, OUT as RUNS, RATES, bkey, reps_for

PRED = PAPER / "workspace/runs/paper1_runR_predictions_v1/predictions.json"
PRED_SHA = "bf38ffd6651bfa179352c18c8205d13465315e4f0019634642da6e0995f7a037"
OUTD = PAPER / "workspace/runs/paper1_runR_stage1_analysis_v1"
MARGIN, NEAR, LOC = 0.05, 1.25, 1.5
SIGNS = np.random.default_rng(20261116).choice((-1.0, 1.0), size=(100_000, 64))
WELL = ("pinned",) + tuple(f"retrained_seed{s}" for s in range(7101, 7106))


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


def load():
    acc, clean = defaultdict(dict), defaultdict(dict)
    missing = []
    for name in NAMES:
        for rep in reps_for(name):
            for f in range(FIELDS):
                d = RUNS / name / f"rep{rep}" / f"field{f}"
                if not (d / "DONE").exists():
                    missing.append(f"{name}/rep{rep}/field{f}")
                    continue
                clean[name][rep] = json.loads((d / "field.json").read_text())["clean_accuracy"]
                with open(d / "results.csv") as fh:
                    for r in csv.DictReader(fh):
                        acc[(name, round(float(r["budget"]), 6), round(float(r["ber"]), 8), r["arm"])][(rep, f)] = float(r["accuracy"])
    return acc, clean, missing


def main():
    if hashlib.sha256(PRED.read_bytes()).hexdigest() != PRED_SHA:
        raise SystemExit("predictions file does not match its frozen hash")
    pred = json.loads(PRED.read_text())["predictions"]
    acc, clean, missing = load()
    if missing:
        raise SystemExit(f"{len(missing)} Stage 1 jobs are not finished, e.g. {missing[:3]}")
    OUTD.mkdir(exist_ok=False)
    cells = []
    for name in NAMES:
        c0 = float(np.mean(list(clean[name].values())))
        for f in FRACTIONS:
            pr = pred[f"{name}/{bkey(f)}"]
            for p in RATES:
                A = acc[(name, round(f, 6), round(p, 8), "pos1024")]
                B = acc[(name, round(f, 6), round(p, 8), "abs1024")]
                keys = sorted(set(A) & set(B))
                d = np.array([A[k] - B[k] for k in keys]) * 100
                by_rep = defaultdict(list)
                for k, v in zip(keys, d):
                    by_rep[k[0]].append(v)
                lo, hi = tint([np.mean(v) for v in by_rep.values()])
                mean_pos, mean_abs = np.mean(list(A.values())), np.mean(list(B.values()))
                usable = max(mean_pos, mean_abs) >= c0 - MARGIN
                near = pr["p_star"] / NEAR <= p <= pr["p_star"] * NEAR
                cells.append(dict(checkpoint=name, budget=f, ber=p, diff=float(d.mean()), fields=int(d.size),
                                  p_raw=signflip_p(d), rep_low=lo, rep_high=hi, usable=bool(usable), near=bool(near),
                                  p_star=pr["p_star"], p_star_how=pr["how"], clean=c0,
                                  mean_pos=float(mean_pos), mean_abs=float(mean_abs),
                                  predicted_positive_ahead=bool(p < pr["p_star"])))
    use = [c for c in cells if c["usable"]]
    adj = holm([c["p_raw"] for c in use])
    for c, a in zip(use, adj):
        c["p_holm"] = float(a)
    for c in cells:
        c.setdefault("p_holm", None)
        c["resolved"] = bool(c["usable"] and c["p_holm"] < 0.05)
        c["agrees"] = bool((c["diff"] > 0) == c["predicted_positive_ahead"] and c["diff"] != 0)
    away = [c for c in use if not c["near"]]
    r1a_viol = [c for c in away if c["resolved"] and not c["agrees"]]
    r1b_share = sum(c["agrees"] for c in away) / len(away)
    # measured switch per checkpoint and budget
    switches = {}
    for name in NAMES:
        for f in FRACTIONS:
            rows = sorted([c for c in use if c["checkpoint"] == name and c["budget"] == f], key=lambda c: c["ber"])
            last = None
            for a, b in zip(rows, rows[1:]):
                if (a["diff"] > 0) != (b["diff"] > 0):
                    last = (a, b)
            pr = pred[f"{name}/{bkey(f)}"]
            if last is None:
                switches[f"{name}/{bkey(f)}"] = dict(measured=None, predicted=pr["p_star"], how=pr["how"], reason="no sign change")
                continue
            a, b = last
            x0, x1 = math.log(a["ber"]), math.log(b["ber"])
            pm = math.exp(x0 + (x1 - x0) * (-a["diff"]) / (b["diff"] - a["diff"]))
            switches[f"{name}/{bkey(f)}"] = dict(measured=pm, predicted=pr["p_star"], how=pr["how"],
                                                 direction="positive then behind" if a["diff"] > 0 else "behind then ahead")
    per_ckpt = {}
    for name in NAMES:
        logs = [math.log(v["measured"] / v["predicted"]) for k, v in switches.items()
                if k.startswith(name + "/") and v["measured"] is not None and v["how"] == "interpolated"]
        per_ckpt[name] = dict(n_pairs=len(logs), mean_log_ratio=float(np.mean(logs)) if logs else None)
    vals = [v["mean_log_ratio"] for v in per_ckpt.values() if v["mean_log_ratio"] is not None]
    lo, hi = tint(vals)
    r1c = dict(checkpoints=len(vals), ratio=math.exp(float(np.mean(vals))), low=math.exp(lo), high=math.exp(hi),
               holds=bool(math.exp(lo) >= 1 / LOC and math.exp(hi) <= LOC))
    geo = math.exp(np.mean([math.log(v["measured"] / v["predicted"]) for v in switches.values()
                            if v["measured"] is not None and v["how"] == "interpolated"]))
    def gm(name):
        ms = [v["measured"] for k, v in switches.items() if k.startswith(name + "/") and v["measured"]]
        return math.exp(np.mean(np.log(ms))) if ms else None
    under = gm("undertrained_e100")
    secondary = dict(undertrained=under, well_trained={n: gm(n) for n in WELL},
                     above_all=bool(under is not None and all(gm(n) is None or under > gm(n) for n in WELL)))
    summary = dict(
        status="Run R Stage 1, confirmatory analysis by the frozen plan (R-A3, A1)",
        predictions_sha256=PRED_SHA, script_sha256=sha256_file(Path(__file__)),
        cells_total=len(cells), cells_usable=len(use), usable_away_from_switch=len(away),
        R1a=dict(holds=not r1a_viol, violations=[{k: c[k] for k in ("checkpoint", "budget", "ber", "diff", "p_holm", "p_star")} for c in r1a_viol]),
        R1b=dict(holds=r1b_share >= 0.8, share=r1b_share, agree=sum(c["agrees"] for c in away), of=len(away)),
        R1c=r1c, R1_confirmed=bool(not r1a_viol and r1b_share >= 0.8 and r1c["holds"]),
        bias_geometric_mean_measured_over_predicted=geo,
        pairs_without_measured_switch=sum(v["measured"] is None for v in switches.values()),
        pairs_at_grid_end=sum(v["how"] != "interpolated" for v in switches.values()),
        per_checkpoint=per_ckpt, switches=switches, secondary=secondary)
    (OUTD / "summary.json").write_text(json.dumps(summary, indent=1))
    with open(OUTD / "cells.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cells[0])); w.writeheader(); w.writerows(cells)
    print(json.dumps({k: summary[k] for k in ("cells_usable", "usable_away_from_switch", "R1a", "R1b", "R1c",
                                              "R1_confirmed", "bias_geometric_mean_measured_over_predicted",
                                              "pairs_without_measured_switch", "pairs_at_grid_end", "secondary")},
                     indent=1, default=str))
    plots(cells, switches)


def plots(cells, switches):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    BLUE, ORANGE = "#2a78d6", "#eb6834"
    groups = (("Well-trained (pinned and five retrained)", WELL), ("Under-trained", ("undertrained_e100",)),
              ("Pruned (50% and 70%)", ("pruned_s05", "pruned_s07")))
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True, facecolor=SURF)
    for ax, (title, names) in zip(axes, groups):
        ps = [c["p_star"] for c in cells if c["checkpoint"] in names and c["p_star_how"] == "interpolated"]
        ax.axvspan(min(ps), max(ps), color=MUTED, alpha=0.2, lw=0)
        ax.axhline(0, color=MUTED, lw=0.8)
        for c in cells:
            if c["checkpoint"] not in names or not c["usable"]:
                continue
            x = c["ber"] * (1 + 0.05 * (FRACTIONS.index(c["budget"]) - 3.5))
            col = BLUE if c["diff"] > 0 else ORANGE
            ax.plot([x, x], [c["rep_low"], c["rep_high"]], color=col, lw=0.8, alpha=0.5)
            ax.plot([x], [c["diff"]], "o", ms=4.5, mfc=col if c["resolved"] else SURF, mec=col, mew=1.2)
        ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=1)
        ax.set_title(title, loc="left", fontsize=9.5, color=INK); ax.set_xlabel("Bit-error probability")
        ax.grid(True, axis="y", color=GRID, lw=0.6); ax.set_facecolor(SURF)
    axes[0].set_ylabel("Measured accuracy, positive − absolute (points)")
    fig.text(0.01, 0.005, "Usable cells, eight budgets; bars 95% intervals over replications; filled: Holm-resolved. "
             "Grey: range of the frozen predicted switch.", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(OUTD / "stage1_by_group.png", dpi=170, facecolor=SURF)
    fig, ax = plt.subplots(figsize=(4.6, 4.2), facecolor=SURF)
    pts = [(v["predicted"], v["measured"], k.split("/")[0]) for k, v in switches.items() if v["measured"] and v["how"] == "interpolated"]
    lim = (min(min(a, b) for a, b, _ in pts) / 1.5, max(max(a, b) for a, b, _ in pts) * 1.5)
    xs = np.array(lim)
    ax.fill_between(xs, xs / LOC, xs * LOC, color=MUTED, alpha=0.15, lw=0)
    ax.plot(xs, xs, color=MUTED, lw=0.8)
    for a, b, n in pts:
        col = ORANGE if n == "undertrained_e100" else ("#1baf7a" if n.startswith("pruned") else BLUE)
        ax.plot([a], [b], "o", ms=5, mfc=col, mec=SURF, mew=1)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Predicted switch $p^*$"); ax.set_ylabel("Measured switch")
    ax.set_title("One point per checkpoint and budget\nband: factor 1.5", loc="left", fontsize=9, color=INK)
    fig.tight_layout(); fig.savefig(OUTD / "stage1_measured_vs_predicted.png", dpi=170, facecolor=SURF)


if __name__ == "__main__":
    main()
