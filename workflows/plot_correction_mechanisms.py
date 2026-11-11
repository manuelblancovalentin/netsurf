#!/usr/bin/env python3
"""Discussion figure (exploratory, not a manuscript figure): Run L, DistilBERT.

Left: accuracy against fault rate for schemes of about one quarter extra storage.
Right: accuracy against extra storage at two fault rates, for the error-correcting-code family and for
triplication of the absolute-gradient mask.  Means of two fault draws; no intervals.
"""
import collections
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
rows, clean = [], None
for f in sorted(glob.glob(str(ROOT / "workspace/runs/paper1_runL_correction_mechanisms_v1/ber_*.json"))):
    d = json.loads(Path(f).read_text()); clean = d["clean"]["accuracy"]; rows += d["rows"]
acc = collections.defaultdict(list); cost = {}
for r in rows:
    acc[(r["condition"], r["ber"])].append(r["accuracy"]); cost[r["condition"]] = r["storage_cost"]
mean = lambda c, b: 100 * float(np.mean(acc[(c, b)]))
bers = sorted({r["ber"] for r in rows})
RED, GREY, BLUE, DARK, LIGHT = "#e4572e", "#9a9a9a", "#2a78d6", "#3a3936", "#c9c9c9"
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.edgecolor": "#555555"})
fig, (a, b) = plt.subplots(1, 2, figsize=(11.2, 4.0))
for ax in (a, b):
    ax.axhspan(100 * (clean - 0.05), 100 * clean, color="#eef3ea", lw=0, zorder=0)
    ax.axhline(100 * clean, color="#777777", lw=0.8, ls=":")
    ax.grid(True, color="#e6e6e6", lw=0.5); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
series = [("unprotected", "No protection", LIGHT, ":", "x"),
          ("row_checksum", "Row checksum correction (1%)", GREY, "-.", "s"),
          ("secded_32", "SEC-DED code over 32 bits (21.9%)", BLUE, "--", "D"),
          ("tmr_msb", "TMR on all sign bits, FORTUNE-style (25%)", DARK, "--", "^"),
          ("tmr_abs_12.5", "TMR on absolute-gradient mask, 12.5% of bits (25%)", RED, "-", "o")]
for c, label, col, ls, mk in series:
    a.plot(bers, [mean(c, x) for x in bers], color=col, ls=ls, marker=mk, ms=4, lw=2.2 if c == "tmr_abs_12.5" else 1.6, label=label)
a.set_xscale("log"); a.set_xlabel("Bit-error probability of every stored bit"); a.set_ylabel("Accuracy (%)")
a.set_title("About one quarter extra storage", loc="left", fontweight="bold"); a.set_ylim(45, 94)
a.legend(loc="lower left", frameon=False, fontsize=7.6)
for p, ls in ((0.01, "-"), (0.03, "--")):
    ecc = [c for c in ("secded_64", "secded_32", "secded_16", "secded_8")]
    tmr = [c for c in ("tmr_abs_6.25", "tmr_abs_12.5", "tmr_abs_18.75", "tmr_abs_25")]
    b.plot([100 * cost[c] for c in ecc], [mean(c, p) for c in ecc], color=BLUE, ls=ls, marker="D", ms=4, lw=1.6, label=f"SEC-DED codes, p = {p:g}")
    b.plot([100 * cost[c] for c in tmr], [mean(c, p) for c in tmr], color=RED, ls=ls, marker="o", ms=4.5, lw=2.2, label=f"TMR on ranked mask, p = {p:g}")
    b.plot([100 * cost["tmr_msb"]], [mean("tmr_msb", p)], color=DARK, marker="^", ms=7, ls="none", label=f"TMR on all sign bits, p = {p:g}")
    b.plot([100 * cost["row_checksum"]], [mean("row_checksum", p)], color=GREY, marker="s", ms=6, ls="none", label=f"Row checksum, p = {p:g}")
b.set_xlabel("Extra storage (% of weight bits)"); b.set_ylabel("Accuracy (%)"); b.set_ylim(45, 94)
b.set_title("Accuracy per unit of extra storage", loc="left", fontweight="bold")
b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=False, fontsize=7.2, ncol=2)
fig.text(0.01, -0.13, "DistilBERT W8A8, 512 sentences, mean of two fault draws per point; every stored bit, including copies and check bits, flips with probability p. "
         "Exploratory.", fontsize=7.2, color="#555555")
fig.tight_layout()
out = ROOT / "workspace/figures_for_discussion"; out.mkdir(parents=True, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(out / f"correction_mechanisms_distilbert.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
print("saved", out / "correction_mechanisms_distilbert.png")
