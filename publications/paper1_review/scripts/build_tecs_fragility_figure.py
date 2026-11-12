#!/usr/bin/env python3
"""Build the checkpoint-fragility figure from data/tecs_runK/summary_vs_pinned.json (no inference).

Left: largest accuracy regret in decision-useful cells, per ResNet-8 checkpoint, for the rankings that differ most.
Right: accuracy lost at p = 3e-3 and 12.5% protection (unprotected, MSB-first, absolute gradient).
Same palette and markers as build_tecs_figures.py.  Checkpoints: the pinned MLPerf reference, five retrained copies
of its recipe, two pruned and one under-trained checkpoint (Table 5).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_tecs_figures as bf  # noqa: E402

plt = bf.plt
ROOT = Path(__file__).resolve().parents[1]
ORDER = [("pinned (Run A, windows 1-9)", "Pinned"), ("retrained_seed7101", "R1"), ("retrained_seed7102", "R2"),
         ("retrained_seed7103", "R3"), ("retrained_seed7104", "R4"), ("retrained_seed7105", "R5"),
         ("pruned_s05", "Pruned\n50%"), ("pruned_s07", "Pruned\n70%"), ("undertrained_e100", "Under-\ntrained")]
LEFT = ["msb", "fkeras", "pos_grad", "global_grad", "abs_grad"]
RIGHT = ["unprotected", "msb", "abs_grad"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = Path(ap.parse_args().out)
    d = json.loads((ROOT / "data/tecs_runK/summary_vs_pinned.json").read_text())
    xs = list(range(len(ORDER)))
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.1, 2.55), gridspec_kw=dict(width_ratios=[1.1, 1.0]))
    jitter = {m: (i - (len(LEFT) - 1) / 2) * 0.09 for i, m in enumerate(LEFT)}
    for m in LEFT:
        lab, col, mk, ls, z = bf.STYLE[m]
        ys = [100 * d[k]["regret"][m] for k, _ in ORDER]
        a.plot([x + jitter[m] for x in xs], ys, ls="none", marker=mk, ms=4.2 if m == "abs_grad" else 3.6, color=col,
               mec=bf.SURFACE, mew=0.4, zorder=z + 1, label=lab)
    a.set_yscale("symlog", linthresh=1, linscale=0.6)
    a.set_ylim(0, 80)
    yt = [0, 0.5, 1, 2, 5, 10, 20, 50]
    a.set_yticks(yt); a.set_yticklabels([str(t) for t in yt]); a.minorticks_off()
    a.set_ylabel("Largest regret (points)", fontsize=7)
    a.set_title("Largest regret in usable cells", loc="left", pad=14, fontsize=7.6)
    a.axvspan(0.5, 5.5, color="#f3f2ee", zorder=0, lw=0)
    a.text(3.0, 0.06, "retrained copies", ha="center", fontsize=6.0, color=bf.MUTED)
    a.legend(loc="lower left", bbox_to_anchor=(0.0, 0.99), frameon=False, fontsize=6.0, handletextpad=0.1,
             columnspacing=0.8, ncol=5, borderaxespad=0.0)
    width = 0.26
    for i, m in enumerate(RIGHT):
        lab, col, mk, ls, z = bf.STYLE[m]
        ys = [100 * d[k]["drop_125_at"]["0.003"][m] for k, _ in ORDER]
        b.bar([x + (i - 1) * width for x in xs], ys, width=width * 0.92, color=col,
              alpha=0.35 if m == "unprotected" else 0.9, lw=0, label=lab, zorder=2)
    b.axhline(5, color="#008300", lw=0.7, ls=(0, (2, 2)), zorder=3)
    b.set_ylabel("Accuracy lost (points)", fontsize=7)
    b.set_title(r"Accuracy lost at $p=3\times10^{-3}$, 12.5% protection", loc="left", pad=14, fontsize=7.6)
    b.axvspan(0.5, 5.5, color="#f3f2ee", zorder=0, lw=0)
    b.legend(loc="lower left", bbox_to_anchor=(0.0, 0.99), frameon=False, fontsize=6.0, handlelength=1.2, ncol=3, columnspacing=0.8, borderaxespad=0.0)
    for ax in (a, b):
        ax.set_xticks(xs); ax.set_xticklabels([lab for _, lab in ORDER], fontsize=5.9)
        ax.set_xlim(-0.6, len(ORDER) - 0.4)
        ax.grid(True, axis="y", color=bf.GRID, lw=0.4); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout(w_pad=1.2)
    bf.save(fig, out, "tecs_checkpoint_fragility")
    print("wrote", out / "tecs_checkpoint_fragility.pdf")


if __name__ == "__main__":
    main()
