"""Interim Stage 1: predicted switch (line) vs measured switch per replication (dot cloud), one panel per checkpoint.
Descriptive only; finished jobs only; not the frozen analysis."""
import json, math, os, sys
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
sys.path.insert(0, "/projects/netsurf/manuelbv/workspace/implementation-worktrees/paper1-runR-rate-aware/workflows")
import runR_stage1_analyze as A
from runR_stage1 import FRACTIONS, RATES, NAMES, bkey

OUT = os.environ.get("OUT_PNG") or "/tmp/claude-1000/-projects-netsurf-manuelbv/eee4f243-fbf3-41d9-8cda-c13310ef18a2/scratchpad/stage1_final_switch_vs_budget.png"
pred = json.loads(A.PRED.read_text())["predictions"]
acc, clean, missing = A.load()
done = 576 - len(missing)
STYLE = {"pinned": ("#2a78d6", "Pinned"), "retrained_seed7101": ("#7c5cd6", "Retrained, seed 7101"), "retrained_seed7102": ("#7c5cd6", "Retrained, seed 7102"), "retrained_seed7103": ("#7c5cd6", "Retrained, seed 7103"), "retrained_seed7104": ("#7c5cd6", "Retrained, seed 7104"), "retrained_seed7105": ("#7c5cd6", "Retrained, seed 7105"),
         "undertrained_e100": ("#eb6834", "Under-trained"), "pruned_s05": ("#1baf7a", "Pruned 50%"),
         "pruned_s07": ("#0e7c86", "Pruned 70%")}
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


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


names = [n for n in STYLE if clean[n]]
plt.rcParams.update({"font.size": 10 if os.environ.get("PAPER") else 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
fig, grid = plt.subplots(2, 5, figsize=(15, 7.6), sharey=True, facecolor=SURF)
axes = grid.ravel()
axes[-1].axis("off")
rng = np.random.default_rng(0)  # horizontal jitter only
for ax, name in zip(axes, names):
    col, lab = STYLE[name]
    c0 = float(np.mean(list(clean[name].values())))
    xb = np.array(FRACTIONS) * 100
    pstar = np.array([pred[f"{name}/{bkey(f)}"]["p_star"] for f in FRACTIONS])
    how = [pred[f"{name}/{bkey(f)}"]["how"] for f in FRACTIONS]
    ax.fill_between(xb, pstar / 1.5, pstar * 1.5, color=col, alpha=0.10, lw=0)
    ax.plot(xb, pstar, color=col, lw=2, label="Predicted $p^*$ (frozen)")
    for x, p, h in zip(xb, pstar, how):
        if h != "interpolated":
            ax.plot([x], [p], marker="^", ms=7, mfc=SURF, mec=col, mew=1.5)
    pooled_x, pooled_y = [], []
    for x, f in zip(xb, FRACTIONS):
        usable_rates, per_rep = [], defaultdict(dict)
        for p in RATES:
            P = acc[(name, round(f, 6), round(p, 8), "pos1024")]; B = acc[(name, round(f, 6), round(p, 8), "abs1024")]
            if max(np.mean(list(P.values())), np.mean(list(B.values()))) < c0 - A.MARGIN:
                continue
            usable_rates.append(p)
            for k in set(P) & set(B):
                per_rep[k[0]].setdefault(p, []).append(P[k] - B[k])
        reps = []
        for r, byp in per_rep.items():
            m = crossing(usable_rates, [np.mean(byp[p]) for p in usable_rates])
            if m:
                reps.append(m)
        if reps:
            ax.scatter(x * np.exp(rng.uniform(-0.06, 0.06, len(reps))), reps, s=22, c=col, alpha=0.35,
                       edgecolors="none", zorder=3)
        allfield = crossing(usable_rates, [np.mean([v for byp in per_rep.values() for v in byp[p]]) for p in usable_rates])
        if allfield:
            pooled_x.append(x); pooled_y.append(allfield)
        else:
            ax.text(x, 1.15e-3, "×", ha="center", va="center", color=MUTED, fontsize=9)
    ax.scatter(pooled_x, pooled_y, s=46, marker="D", c=SURF, edgecolors=INK, linewidths=1.3, zorder=4)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-3, 4e-2)
    ax.xaxis.set_major_locator(FixedLocator([1, 3, 6.25, 12.5, 25, 50]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}")); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(FixedLocator([1e-3, 2e-3, 5e-3, 1e-2, 2e-2]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}")); ax.yaxis.set_minor_formatter(NullFormatter())
    nf = len(acc[(name, FRACTIONS[0], round(RATES[0], 8), "abs1024")])
    ax.set_title(f"{lab}\n{len(clean[name])} replications, {nf} fields", loc="left", fontsize=9.5, color=INK)
    ax.set_xlabel("Protected bits (% of all bits)")
    ax.grid(True, which="major", color=GRID, lw=0.6); ax.set_facecolor(SURF)
for a_ in (grid[0, 0], grid[1, 0]):
    a_.set_ylabel("Bit-error probability where the ranking switches")
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
handles = [Line2D([], [], color=MUTED, lw=2, label="Predicted switch $p^*$ (frozen before the run)"),
           Patch(color=MUTED, alpha=0.18, label="Factor 1.5 around the prediction"),
           Line2D([], [], ls="", marker="o", ms=5, color=MUTED, alpha=0.5, label="Measured switch, one dot per replication"),
           Line2D([], [], ls="", marker="D", ms=6, mfc=SURF, mec=INK, label="Measured switch, all fields pooled"),
           Line2D([], [], ls="", marker="^", ms=6, mfc=SURF, mec=MUTED, label="Prediction at the grid end"),
           Line2D([], [], ls="", marker="x", ms=6, color=MUTED, label="No switch within 5 points of clean")]
axes[-1].set_axis_off(); axes[-1].legend(handles=handles, loc="center left", frameon=False, fontsize=8.5)
None if os.environ.get("PAPER") else fig.suptitle(f"Run R Stage 1, all {done} jobs. Frozen analysis: R1a, R1b, R1c hold",
             x=0.01, ha="left", fontsize=10, color="#0b0b0b")
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(OUT, dpi=170, facecolor=SURF)
print(OUT, done)
