"""Blur pilot (exploratory), bell-shape check: distribution of the change of each input's margin under real bit flips,
against a normal curve and a Student-t (q-Gaussian) fitted to the same data, and against the two-moment Gaussian
weight noise.  One figure per model; rows = mask (none, abs1024, pos1024 at 12.5%), columns = rate.

Margin = true-class logit minus the best other logit; change = faulty minus clean, per input and draw (64 draws x 400
inputs pooled).  Reads only the locked pilot output paper1_blur_pilot_v1/bell.  Descriptive.
"""
import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

RUNS = Path("/projects/netsurf/manuelbv/workspace/implementation-worktrees/paper1-tecs-revision-v1/workspace/runs")
SRC = RUNS / "paper1_blur_pilot_v1/bell"
OUT = Path(os.environ.get("OUT_DIR", RUNS / "paper1_blur_pilot_figures_v1"))
MODELS = {"kws_rep0": "DS-CNN (replication 0)", "pinned_rep1": "ResNet-8, pinned (replication 1)"}
ARMS = {"none": "No protection", "abs1024": "Absolute mask, 12.5%", "pos1024": "Positive mask, 12.5%"}
RATES = ("0.001", "0.003", "0.01")
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
REAL, GAUSS, NORM, TFIT = "#3d3c39", "#2a78d6", "#eb6834", "#1baf7a"
FIT_N = 8000  # points used for the Student-t fit (fixed subsample; the fit on all 25,600 is slow)

plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "font.family": "DejaVu Sans"})
OUT.mkdir(parents=True, exist_ok=True)
fits = {}
for model, title in MODELS.items():
    z = np.load(SRC / model / "margins.npz")
    clean = z["clean"]
    fig, axes = plt.subplots(3, 3, figsize=(12.5, 9.2), facecolor=SURF)
    for i, arm in enumerate(ARMS):
        for j, p in enumerate(RATES):
            ax = axes[i, j]
            ax.set_facecolor(SURF)
            real = (z[f"{arm}|{p}|real|margin"] - clean[None, :]).reshape(-1)
            gauss = (z[f"{arm}|{p}|gauss|margin"] - clean[None, :]).reshape(-1)
            lo, hi = np.quantile(np.concatenate([real, gauss]), [0.0005, 0.9995])
            pad = 0.1 * (hi - lo)
            # logits are requantized, so margins sit on a grid; bins are whole multiples of the grid step, centred
            # on grid points, so no bin is empty only because it falls between two grid values
            u = np.unique(np.round(np.concatenate([real, gauss]), 6))
            step = float(np.min(np.diff(u)[np.diff(u) > 1e-3]))   # float rounding leaves duplicates ~1e-6 apart
            width = step * max(1, int(np.ceil((hi - lo + 2 * pad) / 80 / step)))
            bins = np.arange(np.floor((lo - pad) / step) * step - step / 2, hi + pad + width, width)
            ax.hist(real, bins=bins, density=True, color=REAL, alpha=0.35, linewidth=0, label="Real bit flips", zorder=2)
            hg, eg = np.histogram(gauss, bins=bins, density=True)
            ax.step(0.5 * (eg[1:] + eg[:-1]), hg, where="mid", color=GAUSS, lw=1.4, label="Two-moment Gaussian weight noise", zorder=3)
            xs = np.linspace(bins[0], bins[-1], 400)
            mu, sd = real.mean(), real.std()
            ax.plot(xs, stats.norm.pdf(xs, mu, sd), color=NORM, lw=1.4, ls=(0, (4, 2)), label="Normal fit to real flips", zorder=4)
            sub = np.random.default_rng(0).choice(real, size=min(FIT_N, real.size), replace=False)
            nu, loc, sc = stats.t.fit(sub)
            q = (nu + 3) / (nu + 1)
            ax.plot(xs, stats.t.pdf(xs, nu, loc, sc), color=TFIT, lw=1.4, label="Student-t (q-Gaussian) fit to real flips", zorder=5)
            c = real - mu
            kurt = float((c ** 4).mean() / sd ** 4 - 3)
            between = float((real.reshape(64, -1).mean(1)).var() / real.var())
            fits[f"{model}|{arm}|{p}"] = dict(normal_mean=float(mu), normal_sd=float(sd), t_nu=float(nu), q=float(q),
                                               excess_kurtosis=kurt, between_draw_share=between)
            ax.set_yscale("log")
            ax.set_ylim(max(1e-5, 0.3 / (real.size * (bins[1] - bins[0]))), None)
            ax.grid(True, which="major", color=GRID, lw=0.6, zorder=0)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.text(0.98, 0.97, f"excess kurtosis {kurt:.1f}\nq = {q:.2f}", transform=ax.transAxes, ha="right", va="top",
                    fontsize=7.5, color=INK)
            if i == 0:
                ax.set_title(f"Bit-flip rate {float(p):g}", color=INK, fontsize=9.5, loc="left")
            if j == 0:
                ax.set_ylabel(f"{ARMS[arm]}\ndensity (log)")
            if i == 2:
                ax.set_xlabel("Change of margin (logit units)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle(title, x=0.01, y=0.955, ha="left", color=INK, fontsize=11)
    fig.text(0.01, 0.005, "Blur pilot, exploratory. 64 draws x 400 evaluation inputs pooled per panel; margin = true-class logit "
             "minus best other. q = (nu+3)/(nu+1) from the Student-t fit (q = 1 is a normal curve).", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"bell_{model}.{ext}", dpi=170, facecolor=SURF)
    plt.close(fig)
(OUT / "bell_fits.json").write_text(json.dumps(fits, indent=1))
print(OUT)
