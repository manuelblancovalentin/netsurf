#!/usr/bin/env python3
"""Exploratory (12 November 2026): compare the predicted crossover of the positive- and absolute-gradient masks
(`runPQ_crossover_theory.py`, no fault outcomes read) with the measured accuracy differences of Run Q.

Measured: pos{n} - abs{n} accuracy on the same saved fault schedules (abs512 from Run P, the others from Run Q),
mean over fields per replication, then mean and Student-t interval over replications (Q-A1 applies: only reps with a
finished Run Q job).  A cell agrees with the prediction when the measured sign matches the predicted one (positive
ahead below the predicted crossover, behind above it).  Nothing here was fixed in advance.

Output: workspace/runs/paper1_runPQ_crossover_theory_v1_analysis/ (summary.json, crossover.png, write-once).
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict

import numpy as np
from scipy import stats

from runP_phase12 import OUT as RUNP
from runP_phase2_analyze import useful_flags
from runQ_calibration import OUT as RUNQ

THEORY = RUNP.parent / "paper1_runPQ_crossover_theory_v1"
OUTD = RUNP.parent / "paper1_runPQ_crossover_theory_v1_analysis"
NAMES = ("pinned", "undertrained_e100", "pruned_s07", "pruned_s05") + tuple(f"retrained_seed{s}" for s in range(7101, 7106))
GROUP = {"pinned": "Pinned", "undertrained_e100": "Under-trained", "pruned_s07": "Pruned", "pruned_s05": "Pruned"}
GROUP.update({f"retrained_seed{s}": "Retrained" for s in range(7101, 7106)})
COLOR = {"Pinned": "#2a78d6", "Under-trained": "#eb6834", "Pruned": "#1baf7a", "Retrained": "#eda100"}
BERS = (1e-3, 3e-3, 1e-2, 3e-2)
BUDGETS = (0.0625, 0.125, 0.1875, 0.25)
SIZES = (512, 1024)


def theory(name):
    d = json.loads((THEORY / f"{name}.json").read_text())
    J = defaultdict(list)
    for r in d["rows"]:
        J[(r["arm"], r["budget"], r["p"])].append(r["drift"] + r["variance"])
    pg = np.array(d["pgrid"])
    return pg, {(n, b): np.array([np.mean(J[(f"pos{n}", b, p)]) - np.mean(J[(f"abs{n}", b, p)]) for p in pg])
                for n in SIZES for b in BUDGETS}


MAX_P, TAG = None, ""   # set by --max-p: sensitivity variant added after seeing the first output (see main)


def crossover(pg, D):
    """Rate above which the positive mask is predicted worse at every larger grid rate (log-linear interpolation)."""
    if MAX_P is not None:
        keep = pg <= MAX_P * (1 + 1e-9)
        pg, D = pg[keep], D[keep]
    neg = np.where(D <= 0)[0]
    if neg.size == 0:
        return float(pg[0]), "below_grid"
    i = int(neg[-1])
    if i == len(pg) - 1:
        return float(pg[-1]), "above_grid"
    x0, x1 = math.log(pg[i]), math.log(pg[i + 1])
    return math.exp(x0 + (x1 - x0) * (-D[i]) / (D[i + 1] - D[i])), "interpolated"


def measured(name):
    acc = defaultdict(dict)
    reps = sorted(int(p.parent.name[3:]) for p in (RUNQ / name).glob("rep*/DONE"))
    for rep in reps:
        for src, keep in ((RUNP / name / f"rep{rep}" / "replay.csv", {"abs512"}),
                          (RUNQ / name / f"rep{rep}" / "replay.csv", {"pos512", "abs1024", "pos1024"})):
            with open(src) as f:
                for r in csv.DictReader(f):
                    if r["arm"] in keep:
                        acc[(round(float(r["ber"]), 6), round(float(r["budget"]), 6), r["arm"])][(rep, r["schedule_sha256"])] = float(r["accuracy"])
    flags = useful_flags(name)
    out = {}
    for n in SIZES:
        for b in BUDGETS:
            for ber in BERS:
                A, C = acc[(round(ber, 6), b, f"pos{n}")], acc[(round(ber, 6), b, f"abs{n}")]
                by_rep = defaultdict(list)
                for k in sorted(set(A) & set(C)):
                    by_rep[k[0]].append(100 * (A[k] - C[k]))
                rm = np.array([np.mean(v) for v in by_rep.values()])
                h = stats.t.ppf(0.975, rm.size - 1) * rm.std(ddof=1) / math.sqrt(rm.size)
                out[(n, b, ber)] = dict(diff=float(rm.mean()), low=float(rm.mean() - h), high=float(rm.mean() + h),
                                        reps=int(rm.size), usable=bool(flags.get((float(f"{ber:.3g}"), round(b, 6)), (False,))[0]))
    return out


def main():
    """Default: the definition written before any output was seen (whole grid, up to 1e-1).
    --max-p 0.03: the same definition on the range of the measured rates only.  This variant was added after the
    first output showed that the predicted difference of the pruned checkpoints turns negative again above 5e-2,
    where no rate was measured and every mask has failed; it is a post hoc sensitivity check and is labelled so."""
    global MAX_P, TAG
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-p", type=float, default=None)
    a = ap.parse_args()
    if a.max_p is not None:
        MAX_P, TAG = a.max_p, f"_maxp{a.max_p:g}"
    OUTD.mkdir(exist_ok=bool(TAG))
    if (OUTD / f"summary{TAG}.json").exists():
        raise SystemExit("output exists (write-once)")
    cells, curves, cross = [], {}, {}
    for name in NAMES:
        pg, D = theory(name)
        meas = measured(name)
        for n in SIZES:
            for b in BUDGETS:
                ps, how = crossover(pg, D[(n, b)])
                cross[f"{name}/{n}/{b}"] = dict(p_star=ps, how=how)
                curves[f"{name}/{n}/{b}"] = [float(v) for v in D[(n, b)]]
                for ber in BERS:
                    m = meas[(n, b, ber)]
                    pred_pos_ahead = ber < ps
                    cells.append(dict(checkpoint=name, size=n, budget=b, ber=ber, p_star=ps, **m,
                                      predicted_positive_ahead=pred_pos_ahead,
                                      agrees=(m["diff"] > 0) == pred_pos_ahead,
                                      resolved=m["low"] > 0 or m["high"] < 0))
    summary = {}
    for n in SIZES:
        for sel, label in ((lambda c: c["usable"], "usable"), (lambda c: c["usable"] and c["resolved"], "usable_resolved"),
                           (lambda c: True, "all")):
            cs = [c for c in cells if c["size"] == n and sel(c)]
            summary[f"{n}/{label}"] = dict(cells=len(cs), agree=sum(c["agrees"] for c in cs),
                                           disagree=[f"{c['checkpoint']} {c['ber']:g} {c['budget']} {c['diff']:+.2f} [{c['low']:+.2f},{c['high']:+.2f}] p*={c['p_star']:.2g}"
                                                     for c in cs if not c["agrees"]])
    (OUTD / f"summary{TAG}.json").write_text(json.dumps(dict(
        status="exploratory, post hoc comparison; the prediction reads no fault outcome", crossover_max_p=MAX_P, pgrid=list(map(float, pg)),
        summary=summary, crossover=cross, cells=cells, curves=curves), indent=1))
    for k, v in summary.items():
        print(k, {kk: vv for kk, vv in v.items() if kk != "disagree"})
        for d in v["disagree"]:
            print("   disagree:", d)
    print("\npredicted crossover p* (12.5% budget):")
    for name in NAMES:
        print(f"  {name:<20}", "  ".join(f"{n}: {cross[f'{name}/{n}/0.125']['p_star']:.2e} ({cross[f'{name}/{n}/0.125']['how']})" for n in SIZES))
    plot(pg, curves, cells, cross)


def plot(pg, curves, cells, cross):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": muted, "axes.labelcolor": ink, "xtick.color": muted,
                         "ytick.color": muted, "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), facecolor="#fcfcfb")
    n, b = 1024, 0.125
    seen = set()
    for name in NAMES:
        g = GROUP[name]
        ax1.plot(pg, curves[f"{name}/{n}/{b}"], color=COLOR[g], lw=2 if g != "Retrained" else 1.2,
                 label=None if g in seen else g, zorder=3 if g != "Retrained" else 2)
        ps = cross[f"{name}/{n}/{b}"]["p_star"]
        ax1.plot([ps], [0], "o", ms=6, mfc=COLOR[g], mec="#fcfcfb", mew=1.5, zorder=4)
        seen.add(g)
    ax1.axhline(0, color=muted, lw=0.8)
    ax1.set_xscale("log"); ax1.set_yscale("symlog", linthresh=0.01)
    ax1.set_xlabel("Bit-error probability p")
    ax1.set_ylabel("Predicted expected loss: positive mask − absolute mask (nats)")
    ax1.set_title("Prediction (12.5% protection, 1,024 images)\nbelow zero: positive mask predicted better", fontsize=9, color=ink, loc="left")
    ax1.grid(True, color=grid, lw=0.6); ax1.legend(frameon=False, fontsize=8, loc="upper left")
    for c in cells:
        if c["size"] != n or not c["usable"]:
            continue
        g = GROUP[c["checkpoint"]]
        x = c["ber"] / c["p_star"]
        ax2.errorbar([x], [c["diff"]], yerr=[[c["diff"] - c["low"]], [c["high"] - c["diff"]]], fmt="none",
                     ecolor=COLOR[g], elinewidth=1, alpha=0.6, zorder=2)
        ax2.plot([x], [c["diff"]], "o", ms=6, mfc=COLOR[g] if c["resolved"] else "#fcfcfb", mec=COLOR[g], mew=1.5, zorder=3)
    ax2.axhline(0, color=muted, lw=0.8); ax2.axvline(1, color=ink, lw=1, ls="--")
    ax2.set_xscale("log"); ax2.set_yscale("symlog", linthresh=1)
    ax2.set_xlabel("Bit-error probability ÷ predicted crossover p*")
    ax2.set_ylabel("Measured accuracy: positive − absolute (points)")
    ax2.set_title("Measurement, all usable cells (4 budgets, 1,024 images)\nfilled: 95% interval over replications excludes zero", fontsize=9, color=ink, loc="left")
    ax2.text(0.03, 0.04, "left of dashed line:\npositive predicted better", transform=ax2.transAxes, color=muted, fontsize=8)
    ax2.text(0.97, 0.96, "right: absolute\npredicted better", transform=ax2.transAxes, color=muted, fontsize=8, ha="right", va="top")
    ax2.grid(True, color=grid, lw=0.6)
    for ax in (ax1, ax2):
        ax.set_facecolor("#fcfcfb")
    fig.tight_layout()
    fig.savefig(OUTD / f"crossover{TAG}.png", dpi=170, facecolor=fig.get_facecolor())


if __name__ == "__main__":
    main()
