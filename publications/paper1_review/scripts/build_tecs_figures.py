#!/usr/bin/env python3
"""Build the TECS-revision figures from saved policy-field records (no inference).

Inputs : the six finalized main-grid trial files (read-only, via tecs_grid_audit.load)
         and a grid-audit directory produced by tecs_grid_audit.py.
Outputs: PDF + PNG files in --out (new names; nothing existing is overwritten unless
         --force is given for this script's own outputs).

Palette: categorical slots validated with the dataviz validator (light surface);
every series also carries its own marker/line style, so identity is never colour-alone.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tecs_grid_audit as ga  # noqa: E402

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
STYLE = {  # method: (label, colour, marker, linestyle, zorder)
    "unprotected": ("Unprotected", "#8a8983", None, (0, (4, 2)), 2),
    "random": ("Random", "#b4b3ad", "x", "-", 2),
    "msb": ("MSB-first", "#3a3936", "^", "-", 3),
    "jump": ("Numerical jump", "#eda100", "D", "-", 3),
    "global_grad": ("Global gradient", "#1baf7a", "s", "-", 4),
    "abs_grad": ("Absolute gradient", "#2a78d6", "o", "-", 6),
    "pos_grad": ("Positive gradient", "#eb6834", "v", "-", 5),
    "fkeras": ("FKeras", "#e87ba4", "o", (0, (2, 1.2)), 7),
}
ORDER = ["unprotected", "random", "msb", "fkeras", "jump", "global_grad", "abs_grad", "pos_grad"]
TITLE = {"resnet8": "ResNet-8 / CIFAR-10", "kws": "DS-CNN / Speech Commands", "toycar": "ToyCar autoencoder",
         "distilbert": "DistilBERT / SST-2", "pythia": "Pythia-160M / WikiText-2", "qwen": "Qwen3-0.6B / WikiText-2"}
MODELS = ["resnet8", "kws", "toycar", "distilbert", "pythia", "qwen"]
MARGIN = {"tflite_cls": 0.05, "w8a8_cls": 0.05, "tflite_auc": 0.05, "w8a8_lm": 0.5}

plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 6.8,
                     "ytick.labelsize": 6.8, "legend.fontsize": 7, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK, "text.color": INK,
                     "pdf.fonttype": 42, "font.family": "DejaVu Sans"})


def tci(v):
    v = np.asarray(v, float); n = len(v); m = v.mean()
    if n < 2:
        return m, m, m
    h = stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / math.sqrt(n)
    return m, m - h, m + h


def index(model):
    cells = defaultdict(lambda: defaultdict(dict))
    for r in ga.load(model):
        cells[(r["ber"], r["coverage"])][r["method"]][r["field"]] = r
    return cells


def task_value(kind, r, clean_loss):
    """Task metric plotted in Fig. A: accuracy (%), AUC, or delta NLL (nats/token)."""
    if kind in ("tflite_cls", "w8a8_cls"):
        return 100.0 * r["quality"]
    if kind == "tflite_auc":
        return r["quality"]
    return r["loss"] - clean_loss


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=220, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def fig_task_quality(out, coverage=0.125):
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.25))
    for ax, model in zip(axes.ravel(), MODELS):
        spec = ga.STUDIES[model]; kind = spec["kind"]; cells = index(model)
        bers = sorted({k[0] for k in cells})
        full = [r for r in ga.load(model) if r["method"] == "full"]
        clean_loss = float(np.mean([r["loss"] for r in full]))
        clean_q = float(np.nanmean([r["quality"] for r in full])) if kind != "w8a8_lm" else float("nan")
        if kind == "w8a8_lm":
            thr = MARGIN[kind]; ax.axhspan(-0.2, thr, color="#eef3ea", zorder=0, lw=0)
            ax.axhline(thr, color="#008300", lw=0.7, ls=(0, (2, 2)), zorder=1)
            ax.set_yscale("symlog", linthresh=1.0, linscale=0.9); ax.set_ylim(-0.1, 40)
            ax.set_yticks([0, 0.5, 1, 3, 10, 30]); ax.set_yticklabels(["0", "0.5", "1", "3", "10", "30"])
            ax.set_ylabel(r"$\Delta$ NLL (nats/token)  $\downarrow$")
        else:
            scale = 100.0 if kind != "tflite_auc" else 1.0
            thr = scale * (clean_q - MARGIN[kind])
            ax.axhspan(thr, scale * 1.0, color="#eef3ea", zorder=0, lw=0)
            ax.axhline(thr, color="#008300", lw=0.7, ls=(0, (2, 2)), zorder=1)
            ax.axhline(scale * clean_q, color=MUTED, lw=0.5, ls=":", zorder=1)
            if kind == "tflite_auc":
                ax.set_ylim(0.35, 0.92); ax.set_ylabel(r"File ROC-AUC  $\uparrow$")
                ax.axhline(0.5, color=MUTED, lw=0.5, ls=(0, (1, 3)))
            else:
                chance = {"resnet8": 10, "kws": 100 / 12, "distilbert": 50}[model]
                ax.axhline(chance, color=MUTED, lw=0.5, ls=(0, (1, 3)))
                ax.set_ylim(0 if model != "distilbert" else 42, 100 if model != "distilbert" else 95)
                ax.set_ylabel(r"Accuracy (%)  $\uparrow$")
        for meth in ORDER:
            lab, col, mk, ls, z = STYLE[meth]
            xs, ms, los, his = [], [], [], []
            for ber in bers:
                key = (ber, 0.0) if meth == "unprotected" else (ber, coverage)
                fr = cells[key].get(meth)
                if not fr:
                    continue
                m, lo, hi = tci([task_value(kind, r, clean_loss) for r in fr.values()])
                xs.append(ber); ms.append(m); los.append(lo); his.append(hi)
            if not xs:
                continue
            if meth == "fkeras":
                ax.plot(xs, ms, color=col, ls=ls, lw=1.5, marker=mk, ms=4.6, mfc="none", mec=col, mew=1.0,
                        zorder=z + 1, label=lab)
                continue
            ax.fill_between(xs, los, his, color=col, alpha=0.13, lw=0, zorder=z)
            ax.plot(xs, ms, color=col, ls=ls, lw=1.25 if meth != "abs_grad" else 1.6, marker=mk, ms=3.2,
                    mfc=col, mec=SURFACE, mew=0.4, zorder=z + 1, label=lab)
        ax.set_xscale("log"); ax.set_xlim(min(bers) * 0.7, max(bers) * 1.4)
        ax.set_title(TITLE[model], loc="left", pad=14)
        ax.grid(True, which="major", color=GRID, lw=0.4); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        N = spec["eligible_bits"]
        sec = ax.secondary_xaxis("top", functions=(lambda x, N=N: x * N, lambda y, N=N: y / N))
        sec.set_xscale("log"); sec.tick_params(labelsize=5.8, length=2, pad=1, colors=MUTED)
        sec.spines["top"].set_visible(False)
        sec.set_xlabel("expected proposed faults $Np$", fontsize=5.8, labelpad=1.5, color=MUTED)
    for ax in axes[1]:
        ax.set_xlabel("Logical bit-error probability $p$")
    for ax, model in zip(axes.ravel(), MODELS):
        if ga.STUDIES[model]["kind"].startswith("w8a8"):
            ax.text(0.03, 0.04 if ga.STUDIES[model]["kind"] == "w8a8_cls" else 0.96, "FKeras not measured",
                    transform=ax.transAxes, fontsize=5.8, color=MUTED,
                    va="bottom" if ga.STUDIES[model]["kind"] == "w8a8_cls" else "top")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=8, frameon=False, bbox_to_anchor=(0.5, 1.045), handlelength=2.0,
               columnspacing=0.9, handletextpad=0.35, fontsize=6.6)
    fig.tight_layout(w_pad=1.0, h_pad=1.1)
    save(fig, out, "tecs_task_quality_ber_125")


def fig_fkeras_partial(out):
    """FKeras at the two partial-tier budgets, where its mask differs from MSB-first."""
    models = ["resnet8", "kws", "toycar"]
    shown = ["unprotected", "msb", "fkeras", "global_grad", "abs_grad"]
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 3.9), sharex="col")
    for row, coverage in enumerate((0.0625, 0.1875)):
        for ax, model in zip(axes[row], models):
            kind = ga.STUDIES[model]["kind"]; cells = index(model)
            bers = sorted({k[0] for k in cells})
            full = [r for r in ga.load(model) if r["method"] == "full"]
            clean_q = float(np.nanmean([r["quality"] for r in full]))
            scale = 100.0 if kind != "tflite_auc" else 1.0
            thr = scale * (clean_q - MARGIN[kind])
            ax.axhspan(thr, scale * 1.0, color="#eef3ea", zorder=0, lw=0)
            ax.axhline(thr, color="#008300", lw=0.7, ls=(0, (2, 2)), zorder=1)
            ax.axhline(scale * clean_q, color=MUTED, lw=0.5, ls=":", zorder=1)
            for meth in shown:
                lab, col, mk, ls, z = STYLE[meth]
                xs, ms, los, his = [], [], [], []
                for ber in bers:
                    key = (ber, 0.0) if meth == "unprotected" else (ber, coverage)
                    fr = cells[key].get(meth)
                    if not fr:
                        continue
                    m, lo, hi = tci([task_value(kind, r, 0.0) for r in fr.values()])
                    xs.append(ber); ms.append(m); los.append(lo); his.append(hi)
                ax.fill_between(xs, los, his, color=col, alpha=0.13, lw=0, zorder=z)
                if meth == "fkeras":
                    ax.plot(xs, ms, color=col, ls="-", lw=1.5, marker=mk, ms=3.6, mfc=col, mec=SURFACE, mew=0.4,
                            zorder=z + 1, label=lab)
                else:
                    ax.plot(xs, ms, color=col, ls=ls, lw=1.25, marker=mk, ms=3.2, mfc=col, mec=SURFACE, mew=0.4,
                            zorder=z + 1, label=lab if meth != "msb" else "MSB-first (storage-order tie-break)")
            ax.set_xscale("log")
            if kind == "tflite_auc":
                ax.set_ylim(0.35, 0.92); ax.axhline(0.5, color=MUTED, lw=0.5, ls=(0, (1, 3)))
            else:
                ax.set_ylim(0, 100)
            ax.grid(True, which="major", color=GRID, lw=0.4); ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            short = {"resnet8": "ResNet-8", "kws": "DS-CNN", "toycar": "ToyCar"}[model]
            ax.set_title(f"{short}, {100 * coverage:g}% protected", loc="left", pad=3, fontsize=7.6)
            ax.set_ylabel(r"File ROC-AUC  $\uparrow$" if kind == "tflite_auc" else r"Accuracy (%)  $\uparrow$")
    for ax in axes[1]:
        ax.set_xlabel("Logical bit-error probability $p$")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout(w_pad=1.0, h_pad=0.9, rect=(0, 0, 1, 0.94))
    fig.legend(h, l, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.0), handlelength=2.0,
               columnspacing=1.0, handletextpad=0.4, fontsize=6.8)
    save(fig, out, "tecs_fkeras_partial_tiers")


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fig_regret_map(out, audit):
    rows = read_csv(audit / "per_cell_regret.csv")
    risk_for = {"resnet8": "err", "kws": "err", "distilbert": "err", "toycar": "loss", "pythia": "loss", "qwen": "loss"}
    cmap = LinearSegmentedColormap.from_list("blue_seq", ["#f4f8fe", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    letter = {"abs_grad": "A", "global_grad": "G", "pos_grad": "P", "msb": "M", "jump": "J", "random": "R", "fkeras": "F"}
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 3.35))
    im = None
    for ax, model in zip(axes.ravel(), MODELS):
        kind = ga.STUDIES[model]["kind"]; tau = MARGIN[kind]
        sel = [r for r in rows if r["model"] == model and r["risk"] == risk_for[model]]
        bers = sorted({float(r["ber"]) for r in sel}); covs = sorted({float(r["coverage"]) for r in sel})
        Z = np.full((len(covs), len(bers)), np.nan); best = {}; useful = {}
        for r in sel:
            i, j = covs.index(float(r["coverage"])), bers.index(float(r["ber"]))
            if r["method"] == "abs_grad":
                Z[i, j] = float(r["regret_mean"]) / tau
                best[(i, j)] = r["best_method"]; useful[(i, j)] = r["decision_useful_primary"] == "1"
        im = ax.imshow(np.clip(Z, 0, 1), origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=1)
        for (i, j), b in best.items():
            dark = np.clip(Z[i, j], 0, 1) > 0.55
            if not useful[(i, j)]:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="////", ec="#8a8983", lw=0, zorder=3))
            ax.text(j, i, letter[b], ha="center", va="center", fontsize=6.6, zorder=4,
                    color="#ffffff" if dark else INK, fontweight="bold" if b != "abs_grad" else "normal",
                    bbox=dict(boxstyle="round,pad=0.12", fc="#ffffff" if not dark else "#0d366b", ec="none", alpha=0.0 if useful[(i, j)] else 0.85))
        ax.set_xticks(range(len(bers))); ax.set_xticklabels([f"{b:g}".replace("e-05", "e-5") for b in bers], rotation=40, ha="right")
        ax.set_yticks(range(len(covs))); ax.set_yticklabels([f"{100 * c:g}" for c in covs])
        ax.set_xticks(np.arange(-0.5, len(bers)), minor=True); ax.set_yticks(np.arange(-0.5, len(covs)), minor=True)
        ax.grid(which="minor", color=SURFACE, lw=1.2); ax.tick_params(which="both", length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(TITLE[model], loc="left", pad=3)
    for ax in axes[:, 0]:
        ax.set_ylabel("Protected bits (%)")
    for ax in axes[1]:
        ax.set_xlabel("Logical bit-error probability $p$")
    fig.tight_layout(w_pad=0.8, h_pad=0.9, rect=(0, 0, 0.9, 1))
    cax = fig.add_axes([0.915, 0.2, 0.012, 0.62])
    cb = fig.colorbar(im, cax=cax); cb.outline.set_visible(False)
    cb.set_ticks([0, 0.5, 1]); cb.set_ticklabels(["0", "0.5", r"$\geq$1"])
    cb.set_label("Absolute-gradient regret / usefulness margin", fontsize=6.8)
    save(fig, out, "tecs_grid_regret_map")


def fig_protection_vs_unprotected(out, coverage=0.125):
    panels = [("distilbert", "err"), ("pythia", "loss"), ("qwen", "loss")]
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.15))
    for ax, (model, risk) in zip(axes, panels):
        cells = index(model); bers = sorted({k[0] for k in cells})
        for meth in ("msb", "global_grad", "abs_grad", "pos_grad"):
            lab, col, mk, ls, z = STYLE[meth]; xs, ms, los, his = [], [], [], []
            for ber in bers:
                unp = cells[(ber, 0.0)]["unprotected"]; fr = cells[(ber, coverage)][meth]
                d = [(fr[f][risk] - unp[f][risk]) * (100.0 if risk == "err" else 1.0) for f in sorted(unp)]
                m, lo, hi = tci(d); xs.append(ber); ms.append(m); los.append(lo); his.append(hi)
            ax.fill_between(xs, los, his, color=col, alpha=0.15, lw=0, zorder=z)
            ax.plot(xs, ms, color=col, ls=ls, lw=1.25, marker=mk, ms=3.2, mec=SURFACE, mew=0.4, zorder=z + 1, label=lab)
        ax.axhline(0, color=INK, lw=0.7)
        ax.set_xscale("log"); ax.set_title(TITLE[model], loc="left", pad=3)
        if risk == "err":
            ax.set_ylabel(r"$\Delta$ error rate (pp)")
        else:
            ax.set_yscale("symlog", linthresh=1.0, linscale=0.9); ax.set_yticks([-10, -3, -1, 0, 1, 3, 10])
            ax.set_yticklabels(["$-10$", "$-3$", "$-1$", "0", "1", "3", "10"])
            ax.set_ylabel(r"$\Delta$ NLL (nats/token)")
        ax.grid(True, which="major", color=GRID, lw=0.4); ax.set_axisbelow(True)
        ax.set_xlabel("Logical bit-error probability $p$")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        lo_, hi_ = ax.get_ylim()
        ax.text(0.03, 0.97, "worse than unprotected", transform=ax.transAxes, va="top", fontsize=6.2, color=MUTED)
        ax.text(0.03, 0.03, "better than unprotected", transform=ax.transAxes, va="bottom", fontsize=6.2, color=MUTED)
    h, l = axes[0].get_legend_handles_labels()
    fig.tight_layout(w_pad=1.6, rect=(0, 0, 1, 0.92))
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02), handlelength=2.2)
    save(fig, out, "tecs_protection_vs_unprotected_125")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fig_task_quality(out)
    fig_fkeras_partial(out)
    fig_regret_map(out, Path(a.audit))
    fig_protection_vs_unprotected(out)
    print("wrote figures to", out)


if __name__ == "__main__":
    main()
