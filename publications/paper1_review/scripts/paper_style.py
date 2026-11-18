"""One visual style for every figure of the paper.

Colours are the Okabe-Ito colour-blind-safe set.  Each protection condition keeps one colour in every figure, and plots
of a paired difference ("gain of the absolute gradient over another method") use the same two background tints
everywhere: BETTER (blue tint) where the absolute gradient does better, WORSE (vermillion tint) where it does worse.
"""
from __future__ import annotations

import matplotlib
from matplotlib.ticker import FuncFormatter

OI = dict(black="#000000", orange="#E69F00", sky="#56B4E9", green="#009E73", yellow="#F0E442", blue="#0072B2",
          vermillion="#D55E00", purple="#CC79A7")

# one colour per condition, used in every figure
ABS = OI["blue"]            # absolute gradient (the ranking this paper recommends)
POS = OI["vermillion"]      # positive (signed) gradient
GLOBAL = OI["green"]        # global (mean) gradient
FK = OI["purple"]           # FKeras as published (tier-first)
FKW = OI["purple"]          # FKeras tier-mixing variant: same hue, dashed line, diamond marker
MSB = "#3a3a3a"             # MSB-first; FORTUNE (all sign bits) is the MSB-first mask at 12.5%
SECDED = OI["orange"]       # SEC-DED codes
ABFT = OI["sky"]            # ABFT checksums
RATE = OI["green"]          # mask chosen for the expected fault rate
NONE = "#9a9a9a"            # no protection
JUMP, RANDOM = "#b07d2b", "#bdbdbd"

INK, INK2, GRID, SURF = "#111111", "#555555", "#e6e6e6", "#ffffff"
BETTER, WORSE = OI["blue"], OI["vermillion"]
TINT = 0.09

RC = {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
      "grid.linewidth": 0.6, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
      "lines.linewidth": 2, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
      "legend.frameon": False, "axes.titlesize": 9, "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
      "pdf.fonttype": 42}


def apply():
    matplotlib.rcParams.update(RC)


def shade(ax, axis="y", better_positive=True, labels=True, label_better="absolute gradient better",
          label_worse="absolute gradient worse", fontsize=6.5):
    """Tint the half of the plot where the absolute gradient does better (blue) and worse (vermillion).

    Call after the limits are final.  axis: the axis that carries the difference."""
    lo, hi = (ax.get_ylim() if axis == "y" else ax.get_xlim())
    pos_col, neg_col = (BETTER, WORSE) if better_positive else (WORSE, BETTER)
    span = ax.axhspan if axis == "y" else ax.axvspan
    if hi > 0:
        span(0, hi, color=pos_col, alpha=TINT, lw=0, zorder=0)
    if lo < 0:
        span(lo, 0, color=neg_col, alpha=TINT, lw=0, zorder=0)
    (ax.axhline if axis == "y" else ax.axvline)(0, color=INK2, lw=0.8, zorder=1)
    if axis == "y":
        ax.set_ylim(lo, hi)
    else:
        ax.set_xlim(lo, hi)
    if labels:
        top, bot = (label_better, label_worse) if better_positive else (label_worse, label_better)
        kw = dict(transform=ax.transAxes, fontsize=fontsize, color=INK2, style="italic", zorder=2)
        if axis == "y":
            if hi > 0:
                ax.text(0.015, 0.97, top, va="top", ha="left", **kw)
            if lo < 0:
                ax.text(0.015, 0.03, bot, va="bottom", ha="left", **kw)
        else:
            if hi > 0:
                ax.text(0.98, 0.005, top, va="bottom", ha="right", **kw)
            if lo < 0:
                ax.text(0.02, 0.005, bot, va="bottom", ha="left", **kw)


plain = FuncFormatter(lambda v, _: f"{v:g}")
BER_TICKS = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
