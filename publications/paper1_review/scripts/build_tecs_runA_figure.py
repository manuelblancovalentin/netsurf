#!/usr/bin/env python3
"""Build the confirmation-run figure from the two finalized Run A bundles (no inference).

Row 1: accuracy lost against each replication's clean accuracy at 12.5% protection, new windows only.
Rows 2-3: the within-tier control at 6.25% and 18.75%: FKeras, storage-order MSB-first, and the mean
of four uniformly random fills of the partial tier, with the absolute-gradient mask as a reference.
Bands are 95% Student-t intervals over replications (nine for ResNet-8, three for DS-CNN).
Style and palette are those of build_tecs_figures.py; the random fill reuses the MSB-first colour with
its own marker and line style, so no new colour is introduced.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_tecs_figures as bf  # noqa: E402
import tecs_runA_analysis as ra  # noqa: E402

plt = bf.plt
TITLE = {"resnet8": "ResNet-8, nine new windows", "kws": "DS-CNN, three new windows"}
RT = ("MSB-first, random fill of the tier", "#3a3936", "v", (0, (1, 1.6)), 3)
YT = [0, 1, 2, 3, 4, 5, 10, 20, 40, 80]


def series(data, clean, fresh, bers, keys):
    """Mean and replication-level 95% interval of accuracy lost (points); keys are averaged first."""
    ms, los, his = [], [], []
    for ber in bers:
        per_rep = []
        for r in fresh:
            err = np.mean([np.mean([v[k][0] for v in data[r][ber].values()]) for k in keys])
            per_rep.append(100.0 * (err - (1.0 - clean[r])))
        m, lo, hi = bf.tci(per_rep)
        ms.append(m); los.append(lo); his.append(hi)
    return ms, los, his


def draw(ax, xs, vals, style, label, lw=1.25, ms=3.2, open_marker=False):
    lab, col, mk, ls, z = style
    m, lo, hi = vals
    ax.fill_between(xs, lo, hi, color=col, alpha=0.13, lw=0, zorder=z)
    ax.plot(xs, m, color=col, ls=ls, lw=lw, marker=mk, ms=ms, mfc=bf.SURFACE if open_marker else col,
            mec=col if open_marker else bf.SURFACE, mew=0.8 if open_marker else 0.4, zorder=z + 1, label=label)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(7.1, 6.3), sharex=True, sharey=True)
    for col, model in enumerate(("resnet8", "kws")):
        data, _ = ra.load(model)
        reps = sorted(data); fresh = [r for r in reps if r >= 1]
        bers = sorted(data[reps[0]])
        clean = {r: 1.0 - next(iter(data[r][bers[0]].values()))[("full", 1.0)][0] for r in reps}
        ax = axes[0, col]
        for meth in ("unprotected", "random", "msb", "jump", "global_grad", "abs_grad", "pos_grad"):
            key = ("unprotected", 0.0) if meth == "unprotected" else (meth, 0.125)
            label = bf.STYLE[meth][0] + (" (= FKeras here)" if meth == "msb" else "")
            draw(ax, bers, series(data, clean, fresh, bers, [key]), bf.STYLE[meth], label)
        ax.set_title(f"{TITLE[model]}: rankings at 12.5%", loc="left", pad=3, fontsize=7.6)
        for row, cov in ((1, 0.0625), (2, 0.1875)):
            ax = axes[row, col]
            draw(ax, bers, series(data, clean, fresh, bers, [(c, cov) for c in ra.CONTROLS]), RT, RT[0], open_marker=True)
            draw(ax, bers, series(data, clean, fresh, bers, [("msb", cov)]), bf.STYLE["msb"], "MSB-first, storage-order fill")
            draw(ax, bers, series(data, clean, fresh, bers, [("fkeras", cov)]), bf.STYLE["fkeras"], "FKeras (Hessian order in the tier)", lw=1.5, ms=3.6)
            draw(ax, bers, series(data, clean, fresh, bers, [("abs_grad", cov)]), bf.STYLE["abs_grad"], "Absolute gradient")
            ax.set_title(f"{TITLE[model].split(',')[0]}: partial tier at {100 * cov:g}%", loc="left", pad=3, fontsize=7.6)
    for ax in axes.ravel():
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=5, linscale=1.0)
        ax.set_ylim(-0.6, 95)
        ax.set_yticks(YT); ax.set_yticklabels([str(t) for t in YT]); ax.minorticks_off()
        ax.axhspan(-0.6, 5, color="#eef3ea", zorder=0, lw=0)
        ax.axhline(5, color="#008300", lw=0.7, ls=(0, (2, 2)), zorder=1)
        ax.axhline(0, color=bf.MUTED, lw=0.5, ls=":", zorder=1)
        ax.grid(True, which="major", color=bf.GRID, lw=0.4); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"Accuracy lost (points)  $\downarrow$")
    for ax in axes[2]:
        ax.set_xlabel("Logical bit-error probability $p$")
    # Legends sit in the DS-CNN panels, whose upper-left corner is empty (curves stay low below p = 1e-3).
    axes[0, 1].legend(loc="upper left", ncol=1, frameon=False, fontsize=6.3, handlelength=2.0,
                      handletextpad=0.4, borderaxespad=0.2, labelspacing=0.22)
    axes[1, 1].legend(loc="upper left", ncol=1, frameon=False, fontsize=6.3, handlelength=2.4, handletextpad=0.4,
                      borderaxespad=0.2, labelspacing=0.22)
    fig.tight_layout(w_pad=1.0, h_pad=0.9)
    bf.save(fig, out, "tecs_confirmation_runA")
    print("wrote", out / "tecs_confirmation_runA.pdf")


if __name__ == "__main__":
    main()
