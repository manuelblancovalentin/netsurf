#!/usr/bin/env python3
"""Schematic of Sections 5.2-5.3 (noise-plus-events model, drift and spread).  Illustrative numbers, no measured data.

  figures/fig_concept_model.{pdf,png}
  (a) one INT8 weight under a mask: bits 0-5 (many small jumps) vs bits 6-7 (rare large jumps), protected bits hatched
  (b) what the faults do to that weight: a Gaussian with an exact mean (drift) and spread, plus rare discrete jumps
  (c) the two terms of Eq. (two-term): a mask that ignores the sign of a flip pays spread only; a mask that keeps the
      helpful flips gains from drift at low rate and pays its square at high rate; the curves cross at p*
  (d-f) one pre-activation under faults at three rates.  Background and event variances both grow like p(1-p), so
      neither overtakes the other; what changes is the number of events reaching the neuron: one (it stands out),
      several, many (their sum is itself bell-shaped).  A sketch, not a measurement.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_style as S  # noqa: E402

S.apply()
ROOT = Path(__file__).resolve().parents[1]
NOISE, EVENT = S.OI["sky"], S.OI["orange"]
NOISE_D = "#2b7fb8"


def word(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    jumps = ["128", "64", "32", "16", "8", "4", "2", "1"]
    protected = {7, 4}
    for i, j in enumerate(range(7, -1, -1)):
        x0 = 0.6 + i * 1.1
        col = EVENT if j >= 6 else NOISE
        ax.add_patch(Rectangle((x0, 3.0), 1.0, 1.0, fc=col, ec="white", lw=1.2, alpha=0.9))
        if j in protected:
            ax.add_patch(Rectangle((x0, 3.0), 1.0, 1.0, fc="none", ec=S.INK, lw=0.0, hatch="////", alpha=0.55))
        ax.text(x0 + 0.5, 4.25, f"b{j}", ha="center", va="bottom", fontsize=7, color=S.INK2)
        ax.text(x0 + 0.5, 2.75, jumps[i], ha="center", va="top", fontsize=6.5, color=S.INK2)
    ax.text(0.55, 2.2, "jump if flipped (code steps)", fontsize=6.3, color=S.INK2, va="top")
    ax.add_patch(Rectangle((0.6, 5.05), 0.5, 0.45, fc="white", ec=S.INK2, lw=0.6, hatch="////"))
    ax.text(1.25, 5.27, "protected by the mask", fontsize=6.5, va="center", color=S.INK2)
    ax.annotate("", xy=(0.6, 1.25), xytext=(2.7, 1.25), arrowprops=dict(arrowstyle="-", color=EVENT, lw=2.4))
    ax.text(1.65, 0.95, "bits 6-7\nrare, large\njumps: events", ha="center", va="top", fontsize=6.3, color=S.INK)
    ax.annotate("", xy=(2.85, 1.25), xytext=(9.3, 1.25), arrowprops=dict(arrowstyle="-", color=NOISE, lw=2.4))
    ax.text(6.07, 0.95, "bits 0-5\nmany small jumps:\nthey add up to noise", ha="center", va="top", fontsize=6.3, color=S.INK)
    ax.set_title("(a) one stored weight under a mask", loc="left", fontsize=8.5)


def one_weight(ax):
    mu, sd = 9.0, 13.0
    x = np.linspace(-150, 90, 1200)
    g = np.exp(-0.5 * ((x - mu) / sd) ** 2)
    ax.fill_between(x, g, color=NOISE, alpha=0.45, lw=0)
    ax.plot(x, g, color=NOISE_D, lw=1.2)
    for pos, h in ((64, 0.32), (-128, 0.22)):
        ax.vlines(pos, 0, h, color=EVENT, lw=2.4)
        ax.plot([pos], [h], "o", color=EVENT, ms=4.5)
    ax.text(64, 0.37, "bit 6\nflips", ha="center", va="bottom", fontsize=6.3, color=S.INK)
    ax.text(-128, 0.27, "sign bit\nflips", ha="center", va="bottom", fontsize=6.3, color=S.INK)
    ax.axvline(0, color=S.INK2, lw=0.6, ls=":")
    ax.annotate("", xy=(mu, 1.12), xytext=(0, 1.12), arrowprops=dict(arrowstyle="-|>", color=S.INK, lw=1.0, mutation_scale=7))
    ax.text(mu + 4, 1.12, "drift (exact mean)", fontsize=6.3, va="center")
    ax.annotate("", xy=(mu - sd * 1.18, 0.6), xytext=(mu + sd * 1.18, 0.6),
                arrowprops=dict(arrowstyle="<|-|>", color=S.INK, lw=0.9, mutation_scale=7))
    ax.text(mu - sd * 1.18 - 3, 0.6, "spread\n(exact variance)", fontsize=6.3, va="center", ha="right")
    ax.set_ylim(0, 1.25); ax.set_yticks([])
    ax.set_xticks([-128, -64, 0, 64])
    ax.set_xlabel("change of the weight (code steps)", fontsize=7)
    ax.spines["left"].set_visible(False); ax.grid(False)
    ax.set_title("(b) what faults do to it", loc="left", fontsize=8.5)


def two_terms(ax):
    p = np.logspace(-4, -1.3, 300)
    a = 1.0                                   # spread cost per unit rate, absolute-gradient mask (no drift)
    a_pos, b, c = 1.05, 0.45, 110.0            # positive mask: similar spread, helpful drift (-b p), its square (c p^2)
    j_abs = a * p
    j_pos = (a_pos - b) * p + c * p ** 2
    ps = (a - a_pos + b) / c
    ax.plot(p, j_abs, color=S.ABS, lw=2)
    ax.plot(p, j_pos, color=S.POS, lw=2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.axvspan(p[0], ps, color=S.WORSE, alpha=S.TINT, lw=0)
    ax.axvspan(ps, p[-1], color=S.BETTER, alpha=S.TINT, lw=0)
    ax.axvline(ps, color=S.INK2, lw=0.8, ls="--")
    kw = dict(transform=ax.transAxes, fontsize=6.2, va="top")
    ax.text(0.04, 0.97, "absolute gradient:\nmostly spread", color=S.ABS, **kw)
    ax.text(0.04, 0.80, "positive gradient:\ndrift helps,\nthen hurts", color=S.POS, **kw)
    ax.text(0.60, 0.36, "switch $p^*$", color=S.INK2, **kw)
    ax.text(0.30, 0.03, "positive gradient\nbetter", transform=ax.transAxes, fontsize=6.0, style="italic", color=S.INK2,
            va="bottom", ha="center")
    ax.text(0.97, 0.03, "absolute gradient\nbetter", transform=ax.transAxes, fontsize=6.0, style="italic", color=S.INK2,
            va="bottom", ha="right")
    ax.set_xlim(p[0], p[-1])
    ax.set_xticks([]); ax.set_yticks([]); ax.minorticks_off()
    ax.set_xlabel("fault rate (log)", fontsize=7); ax.set_ylabel("expected loss increase (log)", fontsize=7)
    ax.grid(False)
    ax.set_title("(c) drift vs spread", loc="left", fontsize=8.5)


def regime(ax, sd, events, title, note, envelope=None):
    x = np.linspace(-11, 11, 1200)
    g = np.exp(-0.5 * (x / sd) ** 2)
    ax.fill_between(x, g, color=NOISE, alpha=0.45, lw=0)
    ax.plot(x, g, color=NOISE_D, lw=1.2)
    if envelope:                               # many events: stems under a bell-shaped envelope (their sum)
        e = 0.62 * np.exp(-0.5 * (x / envelope) ** 2)
        ax.plot(x, e, color=EVENT, lw=1.3, ls="--")
        for pos in events:
            h = 0.62 * np.exp(-0.5 * (pos / envelope) ** 2)
            ax.vlines(pos, 0, h, color=EVENT, lw=1.3, alpha=0.9)
    else:
        for pos in events:
            ax.vlines(pos, 0, 0.55, color=EVENT, lw=2.2)
            ax.plot([pos], [0.55], "o", color=EVENT, ms=4)
    ax.axvline(0, color=S.INK2, lw=0.6, ls=":")
    ax.set_ylim(0, 1.55); ax.set_xlim(-11, 11)
    ax.set_yticks([]); ax.set_xticks([])
    ax.spines["left"].set_visible(False); ax.grid(False)
    ax.set_xlabel("change of one neuron's input", fontsize=7)
    ax.set_title(title, loc="left", fontsize=8.5)
    ax.text(0.5, 0.97, note, transform=ax.transAxes, ha="center", va="top", fontsize=6.5, color=S.INK,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))


def main():
    fig = plt.figure(figsize=(7.2, 5.0))
    gs = fig.add_gridspec(2, 3, height_ratios=(1, 0.95), hspace=0.5, wspace=0.28)
    word(fig.add_subplot(gs[0, 0]))
    one_weight(fig.add_subplot(gs[0, 1]))
    two_terms(fig.add_subplot(gs[0, 2]))
    axs = [fig.add_subplot(gs[1, i]) for i in range(3)]
    regime(axs[0], 0.35, [6.5], "(d) sparse faults", "one top-bit flip reaches the neuron:\nit stands out and decides")
    regime(axs[1], 0.9, [-6.0, 4.5, 7.5], "(e) more faults", "several events:\nstill few enough to count")
    many = np.linspace(-9.5, 9.5, 27) + 0.25 * np.sin(np.arange(27) * 2.1)
    regime(axs[2], 1.9, many, "(f) dense faults", "many events: their sum\nis also close to a bell curve", envelope=4.6)
    arr = FancyArrowPatch((0.12, 0.035), (0.9, 0.035), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=12,
                          color=S.INK2, lw=1.2)
    fig.patches.append(arr)
    fig.text(0.51, 0.005, "fault rate grows: background and events grow together; what changes is how many events reach one neuron",
             ha="center", fontsize=6.8, color=S.INK2)
    h = [Rectangle((0, 0), 1, 1, fc=NOISE, alpha=0.6), matplotlib.lines.Line2D([], [], color=EVENT, lw=2.4, marker="o", ms=4)]
    fig.legend(h, ["Gaussian background from bits 0-5 (exact mean and variance)", "flips of bits 6-7 (events; dashed: their sum)"],
               loc="upper center", ncol=2, fontsize=7, bbox_to_anchor=(0.5, 1.01))
    fig.subplots_adjust(left=0.04, right=0.98, top=0.9, bottom=0.12)
    for ext in ("pdf", "png"):
        fig.savefig(ROOT / "figures" / f"fig_concept_model.{ext}", **({"dpi": 220} if ext == "png" else {}))


if __name__ == "__main__":
    main()
