#!/usr/bin/env python3
"""Analytic residual-error curves of protection schemes (Run R plan, Section 2.1).  No measurement is used.

Writes source_v2_rate_aware/figures/residual_curves.pdf (and .png).  The ABFT curves are the paper's simplified
model (one faulty weight repaired per output channel, checksum words never flip, wrong repairs ignored) and are
drawn dashed because they are an approximation that favours ABFT.
"""
from math import comb
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "source_v2_rate_aware" / "figures"
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"


def nmr(p, n):
    return sum(comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(n // 2 + 1, n + 1))


def main():
    p = np.logspace(-5, np.log10(0.5), 400)
    curves = [
        ("No protection", p, MUTED, "-", 1.2),
        ("3 copies", nmr(p, 3), BLUE, "-", 2),
        ("5 copies", nmr(p, 5), BLUE, (0, (4, 2)), 2),
        ("SEC-DED (13,8)", p * (1 - (1 - p) ** 12), ORANGE, "-", 2),
        ("SEC-DED (72,64)", p * (1 - (1 - p) ** 71), ORANGE, (0, (4, 2)), 2),
        ("ABFT, 144 weights*", p * (1 - (1 - p) ** (8 * 143)), AQUA, "-", 2),
        ("ABFT, 576 weights*", p * (1 - (1 - p) ** (8 * 575)), AQUA, (0, (4, 2)), 2),
    ]
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(5.2, 3.0), facecolor=SURF)
    ax.axvspan(1e-4, 1e-1, color=MUTED, alpha=0.07, lw=0)
    for name, y, col, ls, lw in curves:
        ax.plot(p, y, color=col, ls=ls, lw=lw, label=name)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1e-5, 0.5); ax.set_ylim(1e-14, 1)
    ax.set_xlabel("Bit-error probability $p$ of every stored bit")
    ax.set_ylabel(r"Residual flip probability $\rho_s(p)$")
    ax.grid(True, color=GRID, lw=0.5)
    ax.text(3.2e-3, 2e-14, "band where masks differ", color=MUTED, fontsize=7, ha="center", va="bottom")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "residual_curves.pdf")
    fig.savefig(OUT / "residual_curves.png", dpi=200)


if __name__ == "__main__":
    main()
