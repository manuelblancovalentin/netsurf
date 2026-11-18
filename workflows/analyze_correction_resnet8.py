#!/usr/bin/env python3
"""Run L2 analysis: accuracy per scheme and fault rate (mean over 8 draws, 95% t interval), paired contrasts of
ranked triplication against the code or sign-bit triplication of equal or larger storage, and the highest fault
rate each scheme keeps within five points of clean.  Writes one JSON and a discussion plot."""
import collections
import glob
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "workspace/runs/paper1_runL2_correction_resnet8_v1"
rows, clean = [], None
for f in sorted(glob.glob(str(RUN / "ber_*_draw_*.json"))):
    d = json.loads(Path(f).read_text()); clean = d["clean_accuracy"]; rows += d["rows"]
acc = collections.defaultdict(dict); cost = {}
for r in rows:
    acc[(r["condition"], r["ber"])][r["draw"]] = r["accuracy"]; cost[r["condition"]] = r["storage_cost"]
bers = sorted({r["ber"] for r in rows}); conds = sorted(cost, key=lambda c: (cost[c], c))


def ci(v):
    v = np.asarray(v, float); m = float(v.mean())
    if len(v) < 2 or v.std(ddof=1) == 0:
        return m, m, m
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / math.sqrt(len(v))
    return m, m - h, m + h


table = {c: {str(b): ci([acc[(c, b)][d] for d in sorted(acc[(c, b)])]) for b in bers} for c in conds}
pairs = [("tmr_abs_6.25", "secded_64"), ("tmr_abs_12.5", "secded_32"), ("tmr_abs_12.5", "tmr_msb"), ("tmr_abs_18.75", "secded_16"), ("tmr_abs_25", "secded_8")]
contrasts = []
for a, b in pairs:
    for p in bers:
        draws = sorted(set(acc[(a, p)]) & set(acc[(b, p)]))
        m, lo, hi = ci([acc[(a, p)][d] - acc[(b, p)][d] for d in draws])
        contrasts.append(dict(ranked=a, other=b, ranked_cost=cost[a], other_cost=cost[b], ber=p, draws=len(draws), diff=m, ci=[lo, hi]))
highest = {}
for c in conds:
    top = None
    for p in bers:
        if clean - table[c][str(p)][0] <= 0.05:
            top = p
        else:
            break
    highest[c] = top
out = dict(schema="paper1-runL2-analysis/1.0", status="EXPLORATORY", clean=clean, draws=len({r["draw"] for r in rows}), cost=cost,
           accuracy=table, contrasts=contrasts, highest_useful_ber=highest)
(RUN.parent / "paper1_runL2_correction_resnet8_v1_analysis.json").write_text(json.dumps(out, indent=1))
print(f"clean {clean}; draws {out['draws']}")
print(f"{'scheme':20s}{'cost':>7s}" + "".join(f"{b:>9g}" for b in bers) + "   highest useful")
for c in conds:
    print(f"{c:20s}{100 * cost[c]:6.1f}%" + "".join(f"{100 * table[c][str(b)][0]:9.1f}" for b in bers) + f"   {highest[c]}")
print("paired differences, ranked TMR minus other, accuracy points [95% CI], at p = 1e-3, 3e-3, 1e-2, 3e-2")
for a, b in pairs:
    sel = [c for c in contrasts if c["ranked"] == a and c["other"] == b and c["ber"] in (1e-3, 3e-3, 1e-2, 3e-2)]
    print(f"  {a} ({100 * cost[a]:.1f}%) vs {b} ({100 * cost[b]:.1f}%): " + "; ".join(f"{100 * c['diff']:+.1f} [{100 * c['ci'][0]:+.1f},{100 * c['ci'][1]:+.1f}]" for c in sel))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
RED, BLUE, DARK, GREY = "#e4572e", "#2a78d6", "#3a3936", "#9a9a9a"
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.edgecolor": "#555555"})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.2, 4.0))
for ax in (a1, a2):
    ax.axhspan(100 * (clean - 0.05), 100 * clean, color="#eef3ea", lw=0, zorder=0); ax.axhline(100 * clean, color="#777777", lw=0.8, ls=":")
    ax.grid(True, color="#e6e6e6", lw=0.5); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
for c, label, col, ls, mk in (("unprotected", "No protection", "#c9c9c9", ":", "x"), ("row_checksum", "ABFT checksum correction, one error per row (~1%)", GREY, "-.", "s"),
                              ("secded_32", "SEC-DED over 32 bits (21.9%)", BLUE, "--", "D"), ("tmr_msb", "TMR on all sign bits, FORTUNE-style (25%)", DARK, "--", "^"),
                              ("tmr_abs_12.5", "TMR on absolute-gradient mask, 12.5% of bits (25%)", RED, "-", "o")):
    m = [100 * table[c][str(b)][0] for b in bers]; lo = [100 * table[c][str(b)][1] for b in bers]; hi = [100 * table[c][str(b)][2] for b in bers]
    a1.fill_between(bers, lo, hi, color=col, alpha=0.15, lw=0); a1.plot(bers, m, color=col, ls=ls, marker=mk, ms=4, lw=2.2 if c == "tmr_abs_12.5" else 1.6, label=label)
a1.set_xscale("log"); a1.set_xlabel("Bit-error probability of every stored bit"); a1.set_ylabel("Accuracy (%)"); a1.set_ylim(0, 92)
a1.set_title("ResNet-8, about one quarter extra storage", loc="left", fontweight="bold"); a1.legend(loc="lower left", frameon=False, fontsize=7.6)
for p, ls in ((3e-3, "-"), (1e-2, "--")):
    e = ["secded_64", "secded_32", "secded_16", "secded_8"]; t = ["tmr_abs_6.25", "tmr_abs_12.5", "tmr_abs_18.75", "tmr_abs_25"]
    a2.plot([100 * cost[c] for c in e], [100 * table[c][str(p)][0] for c in e], color=BLUE, ls=ls, marker="D", ms=4, lw=1.6, label=f"SEC-DED codes, p = {p:g}")
    a2.plot([100 * cost[c] for c in t], [100 * table[c][str(p)][0] for c in t], color=RED, ls=ls, marker="o", ms=4.5, lw=2.2, label=f"TMR on ranked mask, p = {p:g}")
    a2.plot([25], [100 * table["tmr_msb"][str(p)][0]], color=DARK, marker="^", ms=7, ls="none", label=f"TMR on all sign bits, p = {p:g}")
a2.set_xlabel("Extra storage (% of weight bits)"); a2.set_ylabel("Accuracy (%)"); a2.set_ylim(0, 92)
a2.set_title("Accuracy per unit of extra storage", loc="left", fontweight="bold")
a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=False, fontsize=7.2, ncol=2)
fig.text(0.01, -0.13, f"ResNet-8 INT8 (TFLite reference executable), CIFAR-10 test images 1000-1999, {out['draws']} fault draws per point, bands are 95% intervals over draws. Exploratory.",
         fontsize=7.2, color="#555555")
fig.tight_layout()
dest = ROOT / "workspace/figures_for_discussion"; dest.mkdir(parents=True, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(dest / f"correction_mechanisms_resnet8.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
print("saved", dest / "correction_mechanisms_resnet8.png")
