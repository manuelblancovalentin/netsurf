#!/usr/bin/env python3
"""Manuscript figure for the correction-mechanism comparison on ResNet-8 (Run L2), from the tracked
per-draw files in data/tecs_runL2/.  Accuracy per scheme with 95% intervals over the eight fault draws."""
import collections
import glob
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
rows, clean = [], None
for f in sorted(glob.glob(str(ROOT / "data/tecs_runL2/ber_*_draw_*.json"))):
    d = json.loads(Path(f).read_text()); clean = d["clean_accuracy"]; rows += d["rows"]
acc = collections.defaultdict(list); cost = {}
for r in rows:
    acc[(r["condition"], r["ber"])].append(r["accuracy"]); cost[r["condition"]] = r["storage_cost"]
bers = sorted({r["ber"] for r in rows})


def ci(v):
    v = np.asarray(v, float); m = v.mean(); h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / math.sqrt(len(v)) if v.std() > 0 else 0.0
    return 100 * m, 100 * (m - h), 100 * (m + h)


INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
                     "legend.fontsize": 6.8, "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED,
                     "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.1, 2.9))
for ax in (a1, a2):
    ax.axhspan(100 * (clean - 0.05), 100 * clean, color="#eef3ea", lw=0, zorder=0); ax.axhline(100 * clean, color=MUTED, lw=0.5, ls=":")
    ax.grid(True, color=GRID, lw=0.4); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
series = (("unprotected", "No protection", "#8a8983", (0, (4, 2)), "x"),
          ("row_checksum", "ABFT checksum correction (1%)", "#b4b3ad", "-.", "s"),
          ("secded_32", "SEC-DED, 32-bit words (21.9%)", "#2a78d6", "--", "D"),
          ("tmr_msb", "TMR, all sign bits (25%)", "#3a3936", "--", "^"),
          ("tmr_abs_12.5", "TMR, absolute-gradient 12.5% (25%)", "#eb6834", "-", "o"))
for c, label, col, ls, mk in series:
    v = [ci(acc[(c, b)]) for b in bers]
    a1.fill_between(bers, [x[1] for x in v], [x[2] for x in v], color=col, alpha=0.15, lw=0)
    a1.plot(bers, [x[0] for x in v], color=col, ls=ls, marker=mk, ms=3.2, lw=1.6 if c == "tmr_abs_12.5" else 1.2, label=label)
a1.set_xscale("log"); a1.set_xlabel("Bit-error probability of every stored bit $p$"); a1.set_ylabel(r"Accuracy (%)  $\uparrow$"); a1.set_ylim(0, 92)
a1.set_title("About one quarter extra storage", loc="left"); a1.legend(loc="lower left", frameon=False, handlelength=2.2)
for p, ls in ((3e-3, "-"), (1e-2, (0, (3, 1.5)))):
    e = ["secded_64", "secded_32", "secded_16", "secded_8"]; t = ["tmr_abs_6.25", "tmr_abs_12.5", "tmr_abs_18.75", "tmr_abs_25"]
    a2.plot([100 * cost[c] for c in e], [ci(acc[(c, p)])[0] for c in e], color="#2a78d6", ls=ls, marker="D", ms=3.2, lw=1.2, label=f"SEC-DED, $p$={p:g}")
    a2.plot([100 * cost[c] for c in t], [ci(acc[(c, p)])[0] for c in t], color="#eb6834", ls=ls, marker="o", ms=3.4, lw=1.6, label=f"TMR, ranked, $p$={p:g}")
    a2.plot([25], [ci(acc[("tmr_msb", p)])[0]], color="#3a3936", marker="^", ms=5, ls="none", label=f"TMR, sign bits, $p$={p:g}")
a2.set_xlabel("Extra storage (% of weight bits)"); a2.set_ylabel(r"Accuracy (%)  $\uparrow$"); a2.set_ylim(0, 92)
a2.set_title("Accuracy per unit of extra storage", loc="left"); a2.legend(loc="lower right", frameon=False, ncol=2, handlelength=2.2, columnspacing=0.8)
fig.tight_layout(w_pad=1.2)
for ext in ("pdf", "png"):
    fig.savefig(ROOT / f"figures/tecs_correction_mechanisms.{ext}", dpi=220, bbox_inches="tight", facecolor="#ffffff")
print("wrote figures/tecs_correction_mechanisms.pdf")
