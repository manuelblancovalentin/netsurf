"""Stage 2 interim look (main arms, no MC): gap to the best arm against rate, for v1, v2, abs1024 and pos1024.

Reads only the locked interim analysis (paper1_runR_stage2_interim_main_v1/cells.csv): for every usable cell the paired
mean differences of the static arms against v1 (R2a rows) and against v2 (R2a' rows).  From these the mean of every
arm relative to v1 is recovered exactly (v2 - v1 = (abs1024 - v1) - (abs1024 - v2)); the gap of an arm is the best
arm's mean (six static arms, v1, v2) minus its own.  Unusable cells are absent.  Descriptive.
"""
import csv
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = Path("/projects/netsurf/manuelbv/workspace/implementation-worktrees/paper1-tecs-revision-v1/workspace/runs")
SRC = RUNS / "paper1_runR_stage2_interim_main_v1/cells.csv"
OUT = Path(os.environ.get("OUT_DIR", RUNS / "paper1_runR_stage2_interim_figures_v1"))
STATIC = ("abs64", "abs1024", "pos1024", "squared1024", "fkeras", "msb")
RATES = (1e-4, 3e-4, 1e-3, 2e-3, 3e-3, 5e-3, 1e-2, 3e-2)
BUDGETS = (0.0625, 0.125, 0.1875, 0.25)
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SHOW = {  # arm: (colour, marker, label, horizontal offset in log10 units)
    "abs1024": ("#1baf7a", "^", "Absolute ranking (abs1024)", -0.075),
    "pos1024": ("#eb6834", "v", "Positive ranking (pos1024)", -0.025),
    "ra_analytic": ("#8a8984", "s", "Rate-aware, original (v1)", 0.025),
    "v2": ("#2a78d6", "o", "Rate-aware, selected (v2)", 0.075),
}

diff = defaultdict(dict)  # (ckpt, budget, rate) -> {(ref, arm): mean difference arm - ref, points}
with open(SRC) as fh:
    for r in csv.DictReader(fh):
        diff[(r["checkpoint"], round(float(r["budget"]), 6), round(float(r["ber"]), 8))][(r["ref"], r["arm"])] = float(r["diff"])

gaps = defaultdict(list)  # (arm, budget, rate) -> gaps over checkpoints
for (ck, b, p), d in diff.items():
    m = {a: d[("ra_analytic", a)] for a in STATIC}
    m["ra_analytic"] = 0.0
    m["v2"] = d[("ra_analytic", "abs1024")] - d[("v2", "abs1024")]
    for a in STATIC:  # consistency of the two reference rows
        assert abs((m[a] - m["v2"]) - d[("v2", a)]) < 1e-9
    best = max(m.values())
    for a in SHOW:
        gaps[(a, b, p)].append(best - m[a])

plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "font.family": "DejaVu Sans"})
fig, axes = plt.subplots(1, 4, figsize=(13.5, 4.2), sharey=True, facecolor=SURF)
rng = np.random.default_rng(0)
for ax, b in zip(axes, BUDGETS):
    ax.set_facecolor(SURF)
    ax.axhline(0.5, color=INK, lw=0.9, ls=(0, (4, 3)), zorder=1)
    for a, (col, mk, lab, off) in SHOW.items():
        xs, med = [], []
        for p in RATES:
            g = gaps.get((a, round(b, 6), round(p, 8)))
            if not g:
                continue
            x = p * 10 ** (off + rng.uniform(-0.012, 0.012, len(g)))
            ax.scatter(x, g, s=16, color=col, marker=mk, alpha=0.45, linewidths=0, zorder=3)
            xs.append(p * 10 ** off); med.append(float(np.median(g)))
        ax.plot(xs, med, color=col, lw=2, zorder=4, label=lab)
        ax.scatter(xs, med, s=34, color=col, marker=mk, edgecolors=SURF, linewidths=1.2, zorder=5)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1, linscale=1.2)
    ax.set_ylim(-0.05, 70)
    ax.set_xlim(6e-5, 5e-2)
    ax.set_yticks([0, 0.5, 1, 3, 10, 30])
    ax.set_yticklabels(["0", "0.5", "1", "3", "10", "30"])
    ax.grid(True, which="major", color=GRID, lw=0.7, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(f"Protected bits {100 * b:g}%", color=INK, fontsize=10, loc="left")
    ax.set_xlabel("Bit-flip rate p")
axes[0].set_ylabel("Gap to the best mask (accuracy points)")
axes[-1].text(4.6e-2, 0.53, "0.5 point", color=INK, fontsize=8, ha="right", va="bottom")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.text(0.01, 0.005, "Stage 2 interim (main arms; MC-corrected mask not yet evaluated). Dots: one per ResNet-8 checkpoint "
         "(nine), mean over replications and fields; lines: median over checkpoints.\nBest mask = best of the six static masks, "
         "v1 and v2 in that cell. Usable cells only (best mask within 5 points of clean accuracy).", fontsize=7.5, color=MUTED)
fig.tight_layout(rect=(0, 0.06, 1, 0.93))
OUT.mkdir(parents=True, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"stage2_interim_gap_vs_rate.{ext}", dpi=200, facecolor=SURF)
print(OUT)
