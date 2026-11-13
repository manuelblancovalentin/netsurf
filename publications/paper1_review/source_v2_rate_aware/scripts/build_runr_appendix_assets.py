#!/usr/bin/env python3
"""Appendix assets for Run R from finalized, read-only outputs:
  figures/runr_search_trace.png  - course of the rate-aware search (Stage 0 traces, pinned replication 1, 12.5%)
  figures/stage1_switch_per_checkpoint.png - copy of the Stage 1 per-checkpoint figure
  generated/pstar_table.tex      - frozen predicted switch rates, nine checkpoints x eight budgets
"""
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parents[1]
RUNS = Path("/projects/netsurf/manuelbv/workspace/implementation-worktrees/paper1-tecs-revision-v1/workspace/runs")
S0 = RUNS / "paper1_runR_stage0_v1/checks"
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"

(HERE / "generated").mkdir(exist_ok=True)
# per-checkpoint figure: scripts/build_stage1_per_checkpoint_figure.py (PAPER=1 OUT_PNG=...)
if False: shutil.copyfile(RUNS / "paper1_runR_stage1_figures_v1/stage1_final_switch_vs_budget.png",
                HERE / "figures/stage1_switch_per_checkpoint.png")

# p* table
pred = json.loads((RUNS / "paper1_runR_predictions_v1/predictions.json").read_text())["predictions"]
names = [("pinned", "Pinned")] + [(f"retrained_seed{s}", f"Seed {s}") for s in range(7101, 7106)] + [
    ("undertrained_e100", "Under-trained"), ("pruned_s05", "Pruned 50\\%"), ("pruned_s07", "Pruned 70\\%")]
fr = ["0.010000", "0.031250", "0.062500", "0.125000", "0.187500", "0.250000", "0.375000", "0.500000"]


def fmt(v):
    return f"{v['p_star'] * 1e3:.1f}" + ("$^\\dagger$" if v["how"] != "interpolated" else "")


rows = [" & ".join([lab] + [fmt(pred[f"{n}/{b}"]) for b in fr]) + r"\\" for n, lab in names]
(HERE / "generated/pstar_table.tex").write_text("\n".join(rows) + "\n")

# search trace figure
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.6), facecolor=SURF)
series = [("Analytic, $p=3\\times10^{-3}$", json.loads((S0 / "trace_analytic_0.003.json").read_text())["abs_start"], BLUE, "-"),
          ("Analytic, $p=10^{-2}$", json.loads((S0 / "trace_analytic_0.01.json").read_text())["abs_start"], ORANGE, "-"),
          ("MC-corrected, $p=3\\times10^{-3}$", json.loads((S0 / "trace_mc_0.003.json").read_text())["mc"], GREEN, "--")]
for lab, t, col, ls in series:
    best, xs, ys = None, [], []
    for r in t:
        if r.get("accepted", True):
            best = r["objective"] if best is None else min(best, r["objective"])
        xs.append(r["round"]); ys.append(best)
    a1.plot(xs, ys, color=col, ls=ls, lw=1.6, label=lab)
    acc = [r for r in t[1:] if r.get("accepted")]
    a2.plot([r["round"] for r in t[1:]], [r["swap"] for r in t[1:]], color=col, ls=ls, lw=1.2)
    a2.plot([r["round"] for r in acc], [r["swap"] for r in acc], "o", ms=3.5, color=col)
a1.set_xlabel("Round"); a1.set_ylabel("Objective (loss increase)")
a1.set_title("Best objective so far", loc="left", fontsize=9, color=INK)
a1.legend(frameon=False, fontsize=7.5)
a2.set_yscale("log"); a2.set_xlabel("Round"); a2.set_ylabel("Bits swapped, $m$")
a2.set_title("Swap size (dots: accepted)", loc="left", fontsize=9, color=INK)
for a in (a1, a2):
    a.grid(True, color=GRID, lw=0.6)
fig.tight_layout()
fig.savefig(HERE / "figures/runr_search_trace.png", dpi=220, facecolor=SURF)
print("written")
