#!/usr/bin/env python3
"""EXPLORATORY, after the gradient-trust predictions were frozen (7f4fedb): what sets the agreement D1?

Model: two calibration sets estimate the true mean gradient g plus noise of covariance Sigma / n, so the correlation
of the two estimates over weights is about  D1(n) = n s / (n s + 1),  s = |g|^2 / tr(Sigma)  (signal over per-input
spread).  Per checkpoint and draw (n = 1,024): S = sum gbar^2, N = sum se^2, signal = S - N (unbiased for |g|^2),
spread = n N (= tr Sigma), s = signal / spread.  Also, where per-example absolute moments exist (the gradient-trust
checkpoints): tail ratio (mean |g|)^2 / mean g^2 per weight (0.64 for a zero-mean Gaussian, lower for heavy tails or
sparse gradients), and signal concentration (participation ratio of max(gbar^2 - se^2, 0), as a share of weights).
Inputs: gradient-trust Phase 1 (paper1_gradtrust_diag_v1), DS-CNN Stage 4 calibration, ResNet-8 Stage 2 calibration
(1,024 images at offsets 1000 r: consecutive draws share 24 images), kona-assets training logs.  Figures + summary.json.

    runG_gradtrust_explore.py <out>
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runR_partial_search import RUNS  # noqa: E402

LOGS = {"jet": Path("/data/models/local--kona-assets--netsurf-gt-jet-mlp/snapshots/dca215ccb5594cb0f5525b60b96687c3c3c3c9d8/jet-training-log.csv"),
        "mnist": Path("/data/models/local--kona-assets--netsurf-gt-mnist-mlp/snapshots/6280dc693f97a407b0142d640a96f41308ddfc9e/mnist-training-log.csv")}
GT = ["jet-e1", "jet-e3", "jet-e10", "jet-e30", "jet-e100", "mnist-tcad", "mnist-e1", "mnist-e3", "mnist-e10", "mnist-e30", "vww"]
RESNET = ["pinned", "retrained_seed7101", "retrained_seed7102", "retrained_seed7103", "retrained_seed7104",
          "retrained_seed7105", "undertrained_e100", "pruned_s05", "pruned_s07"]
N = 1024
COL = {"jet": "#2a78d6", "mnist": "#eb6834", "vww": "#1baf7a", "kws": "#3d3d3a", "resnet8": "#8a8984"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def block(gs, ses, n=N):
    """Per-draw signal, spread and s; D1 from the draws."""
    rows = []
    for g, se in zip(gs, ses):
        S, Nn = float((g ** 2).sum()), float((se ** 2).sum())
        rows.append(dict(signal=S - Nn, spread=n * Nn, s=(S - Nn) / (n * Nn)))
    G = np.stack(gs)
    d1 = float(np.median(np.corrcoef(G)[np.triu_indices(len(gs), 1)]))
    med = {k: float(np.median([r[k] for r in rows])) for k in rows[0]}
    return dict(D1=d1, **med, s_per_draw=[r["s"] for r in rows], weights=int(G.shape[1]))


def predicted(s, n):
    return n * s / (n * s + 1)


def gt_entry(cid):
    d = RUNS / "paper1_gradtrust_diag_v1" / cid
    z = np.load(d / "stats.npz")
    summ = json.loads((d / "summary.json").read_text())
    R = 6
    gs = [z[f"g@{r}"] for r in range(R)]
    e = block(gs, [z[f"se@{r}"] for r in range(R)])
    tail, conc = [], []
    for r in range(R):
        g2, ga, g, se = z[f"g2@{r}"], z[f"gabs@{r}"], z[f"g@{r}"], z[f"se@{r}"]
        ok = g2 > 0
        tail.append(float(np.median(ga[ok] ** 2 / g2[ok])))
        a = np.maximum(g ** 2 - se ** 2, 0)
        conc.append(float(a.sum() ** 2 / max((a ** 2).sum(), 1e-300) / a.size))
    G = np.stack(gs)
    e.update(tail_ratio=float(np.median(tail)), signal_concentration=float(np.median(conc)),
             pairs=np.corrcoef(G).tolist(), size_study={int(k): v for k, v in summ["size_study"].items()},
             family=cid.split("-")[0], t_hist=summ["t_histogram"], prediction=summ["prediction"])
    e["size_study"][N] = e["D1"]
    return e


def main(out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 2})
    out.mkdir(parents=True, exist_ok=False)
    E = {cid: gt_entry(cid) for cid in GT}
    kws = [np.load(RUNS / f"paper1_runR_stage4_kws_masks_v1/rep{r}/b0.125000/calibration.npz") for r in range(6)]
    E["dscnn"] = block([c["gbar"] for c in kws], [c["se"] for c in kws])
    reg = json.loads((RUNS / "paper1_gradtrust_kws_regression_v1/summary.json").read_text())
    E["dscnn"].update(family="kws", size_study={**{int(k): v for k, v in reg["size_study"].items()}, N: E["dscnn"]["D1"]})
    for nm in RESNET:
        base = RUNS / "paper1_runR_stage2_masks_v1" / nm
        cs = [np.load(p / "b0.125000/calibration.npz") for p in sorted(base.glob("rep*")) if (p / "b0.125000/calibration.npz").exists()]
        E[f"resnet8-{nm}"] = block([c["gbar"] for c in cs], [c["se"] for c in cs])
        E[f"resnet8-{nm}"].update(family="resnet8", size_study={N: E[f"resnet8-{nm}"]["D1"]})
    logs = {f: {int(r["epoch"]) + 1: r for r in csv.DictReader(open(p))} for f, p in LOGS.items()}
    for cid in GT[:-1]:
        fam, ep = cid.split("-")
        if ep.startswith("e"):
            r = logs[fam][int(ep[1:])]
            E[cid].update(epoch=int(ep[1:]), train_loss=float(r["loss"]), val_loss=float(r["val_loss"]),
                          gap=float(r["val_loss"]) - float(r["loss"]))

    # ---- Figure 1: D1 against calibration size, measured and predicted from s at n = 1,024 ----
    fams = [("jet", [c for c in GT if c.startswith("jet")]), ("mnist", [c for c in GT if c.startswith("mnist")]),
            ("vww", ["vww"]), ("kws", ["dscnn"])]
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.4), sharey=True)
    ns = np.geomspace(128, 16384, 60)
    for ax, (fam, ids) in zip(axs, fams):
        for k, cid in enumerate(ids):
            e = E[cid]
            shade = 0.35 + 0.65 * (k + 1) / len(ids)
            col = matplotlib.colors.to_rgb(COL[fam])
            c = tuple(1 - shade * (1 - x) for x in col)
            ax.plot(ns, predicted(e["s"], ns), color=c, lw=1.2, ls="--")
            xs = sorted(e["size_study"])
            ax.plot(xs, [e["size_study"][x] for x in xs], "o-", color=c, ms=5, lw=1.4, label=cid)
        ax.axhspan(0.30, 0.40, color=GRID, alpha=0.6, lw=0)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("calibration inputs per set")
        ax.set_title({"jet": "Jet MLP (by epoch)", "mnist": "MNIST MLP", "vww": "VWW", "kws": "DS-CNN"}[fam], color=INK, loc="left")
        ax.legend(frameon=False, fontsize=7, loc="lower right")
    axs[0].set_ylabel("agreement D1 (median correlation)")
    fig.text(0.01, -0.04, "Points: measured. Dashed: n s / (n s + 1) from the signal-to-spread ratio s at 1,024. "
             "Grey band: the no-prediction zone 0.30-0.40.", color=INK2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig1_agreement_vs_size.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 2: what moves s (signal and spread) ----
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
    ax = axs[0]
    for cid, e in E.items():
        ax.scatter(e["spread"], max(e["signal"], 1e-12), s=36, color=COL[e["family"]], edgecolor="white", lw=1.5, zorder=3,
                   marker="s" if e["family"] == "resnet8" else "o")
        if cid in ("vww", "dscnn", "resnet8-pinned", "jet-e1", "jet-e3", "jet-e100", "mnist-e3", "mnist-e10"):
            ax.annotate(cid.replace("resnet8-", "R8 "), (e["spread"], max(e["signal"], 1e-12)), fontsize=6.5, color=INK2,
                        xytext=(3, 3), textcoords="offset points")
    lim = np.geomspace(10, 1e7, 10)
    for d in (0.3, 0.4):
        sline = d / (1 - d) / N
        ax.plot(lim, sline * lim, color=INK2, lw=0.8, ls=":")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(10, 1e7); ax.set_ylim(3e-3, 1e5)
    ax.set_xlabel("per-input spread  tr(Sigma)"); ax.set_ylabel("signal  |g|^2")
    ax.set_title("Signal against spread (dotted: D1 = 0.30, 0.40 at n = 1,024)", loc="left", color=INK, fontsize=9)
    for k, fam in enumerate(("jet", "mnist")):
        ax = axs[1 + k]
        ids = [c for c in GT if c.startswith(fam) and "epoch" in E[c]]
        ep = [E[c]["epoch"] for c in ids]
        ax.plot(ep, [E[c]["s"] * N for c in ids], "o-", color=COL[fam], label="n s at n = 1,024 (left axis)")
        ax.set_xscale("log"); ax.set_xlabel("training epoch")
        ax.set_ylabel("n s  (signal / (spread / n))")
        ax.set_title(f"{'Jet' if fam == 'jet' else 'MNIST'}: n s by epoch, with losses", loc="left", color=INK, fontsize=9)
        for x, c in zip(ep, ids):
            ax.annotate(f"train {E[c]['train_loss']:.3f}\nval {E[c]['val_loss']:.3f}", (x, E[c]["s"] * N), fontsize=6.5,
                        color=INK2, xytext=(4, -16), textcoords="offset points")
    fig.tight_layout()
    fig.savefig(out / "fig2_signal_spread.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 3: t histograms against the noise-only curve ----
    fig, axs = plt.subplots(2, 6, figsize=(14, 4.6), sharex=True)
    xs = np.linspace(-8, 8, 400)
    for ax, cid in zip(axs.flat, GT):
        h = E[cid]["t_hist"]
        edges, counts = np.array(h["edges"]), np.array(h["counts"], float)
        width = edges[1] - edges[0]
        dens = counts / counts.sum() / width
        ax.bar(edges[:-1], dens, width=width, align="edge", color=COL[E[cid]["family"]], alpha=0.85, lw=0)
        ax.plot(xs, np.exp(-xs ** 2 / 2) / np.sqrt(2 * np.pi), color=INK, lw=1.2)
        ax.set_yscale("log"); ax.set_ylim(1e-5, 1)
        ax.set_title(f"{cid}  D1 {E[cid]['D1']:.2f}", fontsize=8, loc="left", color=INK)
        ax.grid(False)
    axs.flat[-1].axis("off")
    fig.text(0.01, -0.02, "Per-weight t = mean gradient / its standard error (6 draws pooled). Black: what pure noise "
             "gives (standard normal). Mass outside it is real signal.", color=INK2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig3_t_histograms.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 4: jet-e1 oddity: pairwise agreement between draws, and tails ----
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    P = np.array(E["jet-e1"]["pairs"])
    im = axs[0].imshow(P, vmin=0, vmax=1, cmap="Blues")
    for i in range(6):
        for j in range(6):
            axs[0].text(j, i, f"{P[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if P[i, j] > 0.6 else INK)
    axs[0].set_title("jet-e1: agreement between each pair of draws", loc="left", fontsize=9, color=INK)
    axs[0].set_xlabel("draw"); axs[0].set_ylabel("draw"); axs[0].grid(False)
    ids = GT
    axs[1].bar(range(len(ids)), [E[c]["tail_ratio"] for c in ids], color=[COL[E[c]["family"]] for c in ids], width=0.7)
    axs[1].axhline(2 / np.pi, color=INK, lw=1)
    axs[1].text(len(ids) - 0.5, 2 / np.pi + 0.01, "Gaussian 0.64", ha="right", fontsize=7, color=INK2)
    axs[1].set_xticks(range(len(ids)), ids, rotation=60, fontsize=7)
    axs[1].set_ylabel("median (mean|g|)^2 / mean g^2")
    axs[1].set_title("Tail ratio (lower = heavier tails or sparser gradients)", loc="left", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(out / "fig4_jet_e1_and_tails.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    keep = {k: {kk: vv for kk, vv in v.items() if kk not in ("t_hist",)} for k, v in E.items()}
    (out / "summary.json").write_text(json.dumps(dict(status="EXPLORATORY, after the predictions were frozen", entries=keep), indent=1))
    print(f"{'checkpoint':28s} {'weights':>7s} {'D1':>5s} {'pred D1':>7s} {'n s':>7s} {'signal':>9s} {'spread':>9s} {'tail':>5s} {'conc':>6s}")
    for cid, e in E.items():
        print(f"{cid:28s} {e['weights']:7d} {e['D1']:5.2f} {predicted(e['s'], N):7.2f} {e['s'] * N:7.2f} {e['signal']:9.2e} "
              f"{e['spread']:9.2e} {e.get('tail_ratio', float('nan')):5.2f} {e.get('signal_concentration', float('nan')):6.3f}"
              + (f"  epoch {e['epoch']} train {e['train_loss']:.3f} val {e['val_loss']:.3f}" if "epoch" in e else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    main(ap.parse_args().out)
