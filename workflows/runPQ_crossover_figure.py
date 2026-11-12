#!/usr/bin/env python3
"""Readable version of the exploratory crossover figure (12 November 2026).  Reads only
`paper1_runPQ_crossover_theory_v1_analysis/summary_maxp0.03.json`; draws one panel per checkpoint group with the
measured accuracy difference against the bit-error probability and the predicted crossover as a band.
Output: workspace/runs/paper1_runPQ_crossover_theory_v1_figures/crossover_by_group.png (write-once).
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from runP_phase12 import OUT as RUNP

SRC = RUNP.parent / "paper1_runPQ_crossover_theory_v1_analysis" / "summary_maxp0.03.json"
OUTD = RUNP.parent / "paper1_runPQ_crossover_theory_v1_figures"
GROUPS = (("Well-trained (pinned and five retrained)", ("pinned",) + tuple(f"retrained_seed{s}" for s in range(7101, 7106))),
          ("Under-trained (100 epochs)", ("undertrained_e100",)),
          ("Pruned (50% and 70%)", ("pruned_s05", "pruned_s07")))
BUDGETS = (0.0625, 0.125, 0.1875, 0.25)
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def main():
    OUTD.mkdir(exist_ok=False)
    d = json.loads(SRC.read_text())
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9), sharey=True, facecolor=SURF)
    for ax, (title, names) in zip(axes, GROUPS):
        ps = [d["crossover"][f"{n}/1024/{b}"]["p_star"] for n in names for b in BUDGETS]
        lo, hi = min(ps), max(ps)
        ax.axvspan(4e-4, lo, color=BLUE, alpha=0.07, lw=0)
        ax.axvspan(hi, 6e-2, color=ORANGE, alpha=0.07, lw=0)
        ax.axvspan(lo, hi, color=MUTED, alpha=0.22, lw=0)
        ax.axhline(0, color=MUTED, lw=0.8)
        for c in d["cells"]:
            if c["size"] != 1024 or not c["usable"] or c["checkpoint"] not in names:
                continue
            x = c["ber"] * (1 + 0.09 * (BUDGETS.index(c["budget"]) - 1.5))
            col = BLUE if c["diff"] > 0 else ORANGE
            ax.plot([x, x], [c["low"], c["high"]], color=col, lw=1, alpha=0.55, zorder=2)
            ax.plot([x], [c["diff"]], "o", ms=5.5, mfc=col if c["resolved"] else SURF, mec=col, mew=1.4, zorder=3)
        ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=1, linscale=1.2)
        ax.set_xlim(4e-4, 6e-2); ax.set_ylim(-70, 4)
        ax.set_yticks([-50, -10, -1, 0, 1, 3]); ax.set_yticklabels(["−50", "−10", "−1", "0", "+1", "+3"])
        ax.set_xticks([1e-3, 3e-3, 1e-2, 3e-2]); ax.set_xticklabels(["0.001", "0.003", "0.01", "0.03"])
        ax.minorticks_off()
        ax.grid(True, axis="y", color=GRID, lw=0.6); ax.set_facecolor(SURF)
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")
        ax.set_xlabel("Bit-error probability")
        ax.text(np.sqrt(lo * hi), 3.2, "predicted\nswitch", ha="center", va="top", fontsize=8, color=INK)
    axes[0].set_ylabel("Measured accuracy,\npositive − absolute gradient (points)")
    axes[0].text(4.6e-4, -40, "formula: positive\ngradient better", fontsize=8, color=BLUE, va="center")
    axes[0].text(5.4e-2, 1.6, "formula: absolute\ngradient better", fontsize=8, color=ORANGE, va="center", ha="right")
    fig.suptitle("The formula predicts where the positive-gradient mask stops beating the absolute-gradient mask "
                 "(ResNet-8, 1,024 calibration images, usable cells, four budgets)", fontsize=9.5, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.005, "Each point is one checkpoint and budget; bars are 95% intervals over replications; filled points exclude zero. "
             "Grey band: range of the predicted crossover over checkpoints and budgets, computed without fault data.",
             fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    fig.savefig(OUTD / "crossover_by_group.png", dpi=170, facecolor=SURF)


if __name__ == "__main__":
    main()
