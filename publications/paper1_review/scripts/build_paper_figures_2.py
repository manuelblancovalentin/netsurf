#!/usr/bin/env python3
"""Remaining paper figures, in the shared style (paper_style.py).

  figures/fig_model_check.{pdf,png}   noise-plus-events model against measured accuracy on 18 held-out checkpoints
                                      (locked analysis paper1_blurconf_analysis_v1, hash-checked by
                                      build_confirmation_figure.py, which copies it to data/tecs_noise_confirmation)
  figures/fig_no_static.{pdf,png}     (a) absolute minus positive gradient against p / predicted switch (fine-rate study,
                                      data/tecs_rate_choice); (b-d) language models: gain of each 12.5% triplicated mask
                                      over no protection (failing-protection runs, paper1_runT_w8a8_v1)
  figures/fig_reversal_all.{...}      absolute minus positive gradient per checkpoint (appendix)
  figures/fig_rate_choice.{...}       gap to the best mask on ResNet-8, fixed rankings and the rate-chosen mask (appendix)
  figures/fig_wrong_rate.{...}        mask built for one rate, used at another (appendix)
  figures/fig_room.{...}              damage left at sparse rates against the low-rate gain (appendix)
  figures/fig_curves_25.{...}, fig_curves_125.{...}   every checkpoint, every scheme, failing protection (appendix)
  data/tecs_paper_figures/figures2.json  plotted numbers
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_story_candidates as C  # noqa: E402
import build_paper_figures as P  # noqa: E402
import paper_style as S  # noqa: E402

S.apply()
FIG = C.ROOT / "figures"
CONF = C.ROOT / "data/tecs_noise_confirmation"
OUT: dict = {}
FAMC = {"resnet8": (S.OI["orange"], "o", "ResNet-8 (8 checkpoints)"), "jet": (S.OI["green"], "s", "jet MLP (5)"),
        "mnist": (S.OI["purple"], "D", "MNIST MLP (4)"), "vww": (S.INK, "^", "Visual Wake Words")}
R8C = ("undertrained_e100", "pruned_s07", "pruned_s05", "retrained_seed7101", "retrained_seed7102", "retrained_seed7103",
       "retrained_seed7104", "retrained_seed7105")
R8L = dict(undertrained_e100="ResNet-8, under-trained", pruned_s07="ResNet-8, pruned 70%", pruned_s05="ResNet-8, pruned 50%",
           **{f"retrained_seed710{i}": f"ResNet-8, retrained {i}" for i in range(1, 6)})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", **({"dpi": 220} if ext == "png" else {}))
    plt.close(fig)


def symlog_y(ax, lin=1.0, ticks=(-50, -20, -10, -5, -2, -1, 0, 1, 2, 5, 10, 20, 50)):
    ax.set_yscale("symlog", linthresh=lin, linscale=0.8)
    lo, hi = ax.get_ylim()
    ax.set_yticks([t for t in ticks if lo <= t <= hi])
    ax.yaxis.set_major_formatter(S.plain)
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())


def fam(c):
    return "resnet8" if c in R8C else c.split("-")[0]


def fit_line(ax, x, y, color=S.INK2, label="least-squares fit"):
    x, y = np.asarray(x, float), np.asarray(y, float)
    b, a = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, a + b * xs, color=color, lw=1.2, ls="-", label=f"{label} (slope {b:.2f})", zorder=4)
    return float(a), float(b)


# ------------------------------------------------------------------ model check (main)
def model_check():
    s = json.loads((CONF / "summary.json").read_text())
    cells = list(csv.DictReader(open(CONF / "cells.csv")))
    pairs = list(csv.DictReader(open(CONF / "pairs.csv")))
    order = list(R8C) + [c for c in s["checkpoints"] if c not in R8C]
    fig, axs = plt.subplots(1, 3, figsize=(12.2, 4.0), gridspec_kw=dict(width_ratios=[1, 1.35, 1]))
    ax = axs[0]
    U = [c for c in cells if c["usable"] == "True"]
    for f, (col, mk, lab) in FAMC.items():
        Q = [c for c in U if fam(c["checkpoint"]) == f]
        ax.scatter([float(c["measured"]) for c in Q], [float(c["hybrid"]) for c in Q], s=10, marker=mk, color=col,
                   alpha=0.35, lw=0, label=lab)
    xm, ym = [float(c["measured"]) for c in U], [float(c["hybrid"]) for c in U]
    lo = min(xm) - 1
    ax.plot([lo, 100], [lo, 100], color=S.INK, lw=0.8, ls="--", label="model = measurement")
    a, b = fit_line(ax, xm, ym, color=S.ABS)
    ax.set_xlabel("measured accuracy (%)"); ax.set_ylabel("predicted by the model (%)")
    ax.set_title(f"(a) {len(U):,} conditions", loc="left")
    ax.legend(fontsize=6.8, loc="upper left", markerscale=1.6)
    OUT["model_check"] = dict(cells=len(U), fit_intercept=round(a, 3), fit_slope=round(b, 4))
    ax = axs[1]
    x = np.arange(len(order))
    for k, (v, col, mk, lab) in enumerate((("mean", S.NONE, "o", "mean shift only"),
                                            ("gauss", S.OI["sky"], "s", "Gaussian noise on all bits"),
                                            ("hybrid", S.ABS, "D", "noise on bits 0-5 + flips of bits 6-7"))):
        ys = [s["checkpoints"][c]["errors_usable"][v]["mean_abs"] for c in order]
        ax.plot(x, ys, mk, color=col, ms=5.5, mec="white", mew=0.6, label=lab, zorder=3)
    ax.axhline(1.0, color=S.OI["vermillion"], lw=1, ls="--")
    ax.text(len(order) - 0.5, 1.03, "limit fixed in advance", ha="right", va="bottom", fontsize=7, color=S.OI["vermillion"])
    ax.set_ylim(0, 2.1); ax.set_ylabel("mean absolute error (points)")
    ax.set_xticks(x, [R8L.get(c, P.NAMES.get(c, c)).replace("ResNet-8, ", "R8 ").replace(", epoch ", " e")
                      .replace("Visual Wake Words", "VWW") for c in order], rotation=60, fontsize=6.8, ha="right")
    ax.set_title("(b) error per checkpoint", loc="left")
    ax.legend(fontsize=6.8, loc="upper right")
    ax = axs[2]
    Q = [q for q in pairs if q["pair"] == "pos1024-abs1024"]
    xs = [-float(q["measured"]) for q in Q]; ys = [-float(q["hybrid"]) for q in Q]       # absolute minus positive
    for f, (col, mk, lab) in FAMC.items():
        R = [q for q in Q if fam(q["checkpoint"]) == f]
        ax.scatter([-float(q["measured"]) for q in R], [-float(q["hybrid"]) for q in R], s=10, marker=mk, color=col,
                   alpha=0.4, lw=0)
    lim = (min(xs + ys) * 1.3, max(xs + ys) * 1.3)
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    for (x0, x1, y0, y1, c) in ((0, lim[1], 0, lim[1], S.BETTER), (lim[0], 0, lim[0], 0, S.WORSE)):
        ax.fill_between([x0, x1], y0, y1, color=c, alpha=S.TINT, lw=0, zorder=0)
    ax.plot(lim, lim, color=S.INK, lw=0.8, ls="--")
    ax.axhline(0, color=S.INK2, lw=0.6); ax.axvline(0, color=S.INK2, lw=0.6)
    for setter, axis in ((ax.set_xscale, ax.xaxis), (ax.set_yscale, ax.yaxis)):
        setter("symlog", linthresh=0.5, linscale=0.7)
        axis.set_major_formatter(S.plain); axis.set_minor_locator(matplotlib.ticker.NullLocator())
    tk = [t for t in (-50, -10, -2, 0, 2, 10, 50) if lim[0] <= t <= lim[1]]
    ax.set_xticks(tk); ax.set_yticks(tk)
    ax.text(0.97, 0.97, "both: absolute\ngradient better", transform=ax.transAxes, ha="right", va="top", fontsize=6.5,
            color=S.INK2, style="italic")
    ax.text(0.03, 0.03, "both: positive\ngradient better", transform=ax.transAxes, ha="left", va="bottom", fontsize=6.5,
            color=S.INK2, style="italic")
    ax.set_xlabel("measured gain of absolute over positive (points)")
    ax.set_ylabel("predicted gain (points)")
    ax.set_title("(c) which mask is better", loc="left")
    agree = sum((a_ > 0) == (b_ > 0) for a_, b_ in zip(xs, ys) if abs(a_) >= 0.5)
    OUT["model_check"]["pairs_sign_agree_ge_0.5"] = [agree, sum(abs(a_) >= 0.5 for a_ in xs)]
    fig.tight_layout()
    save(fig, "fig_model_check")


# ------------------------------------------------------------------ no fixed ranking is safe (main)
def no_static():
    s1 = [r for r in C.rows(C.RC / "stage1_cells.csv") if r["usable"] == "True" and float(r["budget"]) >= 0.0625]
    s4 = [r for r in C.rows(C.RC / "stage4_switch_cells.csv") if r["usable"] == "True"]
    fig = plt.figure(figsize=(12.2, 3.9))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.5, 1, 1, 1], wspace=0.38)
    ax = fig.add_subplot(gs[0])
    x = np.array([float(r["ber"]) / float(r["p_star"]) for r in s1]); y = -np.array([float(r["diff"]) for r in s1])
    ax.scatter(x, y, s=9, color=S.OI["orange"], alpha=0.3, lw=0, label=f"ResNet-8, 9 checkpoints ({len(x)} conditions)")
    edges = np.logspace(np.log10(x.min()), np.log10(x.max()), 14)
    mid, med = [], []
    for lo_, hi_ in zip(edges[:-1], edges[1:]):
        m = (x >= lo_) & (x < hi_)
        if m.sum() >= 8:
            mid.append(np.sqrt(lo_ * hi_)); med.append(np.median(y[m]))
    ax.plot(mid, med, color=S.OI["orange"], lw=2.2, label="ResNet-8, median trend")
    x4 = np.array([float(r["ber"]) / float(r["p_star"]) for r in s4]); y4 = -np.array([float(r["diff"]) for r in s4])
    ax.scatter(x4, y4, s=18, color=S.INK, marker="D", alpha=0.75, lw=0, label=f"DS-CNN ({len(x4)} conditions)")
    ax.axvline(1, color=S.INK2, lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_ylim(min(y.min(), y4.min()) * 1.15, max(y.max(), y4.max()) * 1.15)
    symlog_y(ax)
    S.shade(ax, "y", label_better="absolute gradient better", label_worse="positive gradient better")
    ax.set_xlabel("fault rate / predicted switch rate")
    ax.set_ylabel("absolute minus positive gradient (points)")
    ax.set_title("(a) ResNet-8 and DS-CNN, rate rescaled", loc="left")
    ax.legend(fontsize=6.6, loc="upper left", bbox_to_anchor=(0, 0.93))
    OUT["no_static"] = dict(resnet8_cells=len(x), kws_cells=len(x4))
    for k, cid in enumerate(("distilbert_sst2_int8", "pythia160m_int8", "qwen3_0_6b_int8")):
        ax = fig.add_subplot(gs[k + 1])
        vals, clean, unit, lower = P.per_draw(cid)
        sign = -1 if lower else 1
        bers = sorted({b for (c, b) in vals if c == "unprotected"})
        rec = {}
        for cond, col, ls, lab in (("tmr_abs@0.125", S.ABS, "-", "absolute gradient"), ("tmr_global@0.125", S.GLOBAL, "-", "global gradient"),
                                   ("tmr_msb@0.125", S.MSB, "--", "MSB-first"), ("tmr_pos@0.125", S.POS, "-", "positive gradient")):
            m, lo, hi = [], [], []
            for b in bers:
                A, N = vals[(cond, b)], vals[("unprotected", b)]
                d = sign * np.array([A[u] - N[u] for u in sorted(A)])
                mm, l_, h_ = P.ci(d)
                m.append(mm); lo.append(l_); hi.append(h_)
            ax.fill_between(bers, lo, hi, color=col, alpha=0.15, lw=0)
            ax.plot(bers, m, ls, color=col, lw=1.7, marker="o", ms=2.8, label=lab)
            rec[cond] = [round(v, 3) for v in m]
        ax.set_xscale("log")
        ax.set_xlim(1e-4 * 0.7, 1e-1 * 1.4)
        lo_, hi_ = ax.get_ylim()
        ax.set_ylim(lo_, hi_)
        S.shade(ax, "y", labels=(k == 0), label_better="better than no protection", label_worse="worse than no protection")
        ax.set_xlabel("bit-error probability")
        ax.set_ylabel("accuracy gained (points)" if not lower else "loss reduced (nats/token)")
        ax.set_title(f"({'bcd'[k]}) {P.NAMES[cid]}", loc="left")
        if k == 2:
            ax.legend(fontsize=6.6, loc="upper left")
        OUT["no_static"][cid] = dict(bers=bers, gain_over_none=rec)
    save(fig, "fig_no_static")


# ------------------------------------------------------------------ appendix: reversal per checkpoint
def reversal_all():
    s1 = [r for r in C.rows(C.RC / "stage1_cells.csv") if float(r["budget"]) == 0.125 and r["usable"] == "True"]
    s4 = [r for r in C.rows(C.RC / "stage4_switch_cells.csv") if float(r["budget"]) == 0.125 and r["usable"] == "True"]
    gt = json.loads((C.RC / "gradtrust_summary.json").read_text())["checkpoints"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
    for name in C.R8:
        rr = sorted((float(r["ber"]), -float(r["diff"])) for r in s1 if r["checkpoint"] == name)
        hl = name in ("pinned", "undertrained_e100")
        col = S.OI["orange"] if name != "undertrained_e100" else S.OI["vermillion"]
        axs[0].plot(*zip(*rr), color=col, alpha=0.95 if hl else 0.4, lw=2 if hl else 1.1,
                    label={"pinned": "pinned reference", "undertrained_e100": "under-trained"}.get(name))
        ps = float(next(r["p_star"] for r in s1 if r["checkpoint"] == name))
        axs[0].plot([ps], [0], "|", color=col, ms=11, mew=1.6)
    axs[0].plot([], [], color=S.OI["orange"], alpha=0.4, lw=1.1, label="retrained / pruned")
    axs[0].plot([], [], "|", color=S.INK2, ms=10, label="predicted switch")
    axs[0].set_title("(a) ResNet-8, nine checkpoints", loc="left")
    rr = sorted((float(r["ber"]), -float(r["diff"])) for r in s4)
    axs[1].plot(*zip(*rr), color=S.INK, lw=2.2, label="DS-CNN")
    for cid in C.GT:
        c = sorted((x_["ber"], -x_["diff"]) for x_ in gt[cid]["cells"] if x_["budget"] == 0.125)
        axs[1].plot(*zip(*c), color=FAMC[fam(cid)][0], alpha=0.6, lw=1.2)
    for f in ("jet", "mnist", "vww"):
        axs[1].plot([], [], color=FAMC[f][0], alpha=0.6, lw=1.2, label=FAMC[f][2])
    axs[1].set_title("(b) DS-CNN and ten further checkpoints", loc="left")
    for ax in axs:
        ax.set_xscale("log")
    lo = min(a.get_ylim()[0] for a in axs); hi = max(a.get_ylim()[1] for a in axs)
    for ax in axs:
        ax.set_ylim(lo, hi); symlog_y(ax)
        S.shade(ax, "y", label_better="absolute gradient better", label_worse="positive gradient better")
        ax.set_xlabel("bit-error probability"); ax.legend(fontsize=7, loc="lower left")
    axs[0].set_ylabel("absolute minus positive gradient (points)")
    fig.tight_layout()
    save(fig, "fig_reversal_all")


# ------------------------------------------------------------------ appendix: rate choice, wrong rate, room
def rate_choice():
    acc, cross, clean = C.stage2()
    arms_all = sorted({k[3] for k in acc})
    rates = sorted({k[2] for k in acc})
    show = (("abs1024", S.ABS, "-", "absolute gradient"), ("pos1024", S.POS, "-", "positive gradient"),
            ("msb", S.MSB, "--", "MSB-first"), ("v2", S.RATE, "-", "chosen for the expected rate"))
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    OUT["rate_choice"] = {}
    for ax, b in zip(axs, (0.125, 0.25)):
        for arm, col, ls, lab in show:
            med, lo, hi, xs = [], [], [], []
            for p in rates:
                gaps = []
                for n in C.R8:
                    m = {a: 100 * np.mean(list(acc[(n, b, p, a)].values())) for a in arms_all if acc.get((n, b, p, a))}
                    if max(m.values()) >= 100 * clean[n] - 5:
                        gaps.append(max(m.values()) - m[arm])
                if len(gaps) >= 5:
                    xs.append(p); med.append(np.median(gaps)); lo.append(min(gaps)); hi.append(max(gaps))
            ax.fill_between(xs, lo, hi, color=col, alpha=0.13, lw=0)
            ax.plot(xs, med, ls, color=col, marker="o", ms=3.2, lw=1.8, label=lab)
            OUT["rate_choice"][f"{b}/{arm}"] = dict(rates=xs, median=[round(v, 2) for v in med], max=[round(v, 2) for v in hi])
        ax.axhline(0.5, color=S.INK2, lw=0.8, ls=":")
        ax.set_xscale("log"); ax.set_ylim(0, None)
        ax.set_xlabel("bit-error probability"); ax.set_title(f"({'ab'[int(b == 0.25)]}) {b:.1%} of bits protected", loc="left")
    for ax in axs:
        ax.set_ylim(0, max(a.get_ylim()[1] for a in axs))
        symlog_y(ax, lin=1, ticks=(0, 0.5, 1, 2, 5, 10, 20, 50))
    axs[0].set_ylabel("accuracy behind the best mask (points)")
    axs[0].legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    save(fig, "fig_rate_choice")
    # wrong rate
    fig, axs = plt.subplots(1, 2, figsize=(10.6, 4.4))
    from matplotlib.colors import LinearSegmentedColormap, SymLogNorm
    cmap = LinearSegmentedColormap.from_list("seqv", ["#fdf3ec", "#f2a77a", S.OI["vermillion"], "#6b2600"])
    for ax, n in zip(axs, ("pinned", "undertrained_e100")):
        G = np.full((len(rates), len(rates)), np.nan)
        for i, pa in enumerate(rates):
            for j, pb in enumerate(rates):
                gaps = []
                for b in (0.0625, 0.125, 0.1875, 0.25):
                    m = {a: 100 * np.mean(list(acc[(n, b, pb, a)].values())) for a in arms_all if acc.get((n, b, pb, a))}
                    if max(m.values()) < 100 * clean[n] - 5:
                        continue
                    v = m["ra_analytic"] if pa == pb else (100 * np.mean(list(cross[(n, b, pa, pb)].values()))
                                                         if cross.get((n, b, pa, pb)) else None)
                    if v is not None:
                        gaps.append(max(m.values()) - v)
                if gaps:
                    G[i, j] = np.mean(gaps)
        im = ax.imshow(G, cmap=cmap, norm=SymLogNorm(1, vmin=0, vmax=70), origin="lower")
        for i in range(len(rates)):
            for j in range(len(rates)):
                if np.isfinite(G[i, j]):
                    ax.text(j, i, f"{G[i, j]:.1f}" if G[i, j] < 10 else f"{G[i, j]:.0f}", ha="center", va="center",
                            fontsize=6.8, color="white" if G[i, j] > 8 else S.INK)
        lab = [f"{r:.0e}".replace("e-0", "e-") for r in rates]
        ax.set_xticks(range(len(rates)), lab, rotation=45, fontsize=7); ax.set_yticks(range(len(rates)), lab, fontsize=7)
        ax.set_xlabel("rate at which the mask is used"); ax.set_ylabel("rate the mask was built for")
        ax.grid(False); ax.set_title(f"({'ab'[n != 'pinned']}) ResNet-8, {C.R8_LABEL[n]}", loc="left")
        for sp in ax.spines.values():
            sp.set_visible(False)
    cb = fig.colorbar(im, ax=axs, shrink=0.8, label="accuracy behind the best mask (points)")
    cb.formatter = S.plain; cb.update_ticks()
    save(fig, "fig_wrong_rate")


def room():
    pts = []
    s1 = C.rows(C.RC / "stage1_cells.csv")
    for n in C.R8:
        c = [r for r in s1 if r["checkpoint"] == n and float(r["budget"]) == 0.125]
        at = {round(float(r["ber"]), 6): r for r in c}
        pts.append(("resnet8", 100 * (float(at[0.003]["clean"]) - float(at[0.003]["mean_abs"])),
                    max(float(r["diff"]) for r in c if float(r["ber"]) <= 3e-3 and r["usable"] == "True")))
    s4 = {round(float(r["ber"]), 6): r for r in C.rows(C.RC / "stage4_switch_cells.csv") if float(r["budget"]) == 0.125}
    clean4 = json.loads((C.RC / "stage4_summary.json").read_text())["clean_accuracy"]
    pts.append(("dscnn", 100 * (clean4 - float(s4[0.003]["mean_abs"])),
                max(float(r["diff"]) for p, r in s4.items() if p <= 3e-3 and r["usable"] == "True")))
    s = json.loads((C.RC / "summary.json").read_text())["gradtrust"]["checkpoints"]
    gt = json.loads((C.RC / "gradtrust_summary.json").read_text())["checkpoints"]
    for cid in C.GT:
        pts.append((cid.split("-")[0], s[cid]["clean"] - s[cid]["acc_125"]["abs1024"]["0.003"],
                    max([x_["diff"] for x_ in gt[cid]["cells"] if x_["budget"] == 0.125 and x_["ber"] <= 3e-3] or [np.nan])))
    fig, ax = plt.subplots(figsize=(6.0, 3.9))
    st = dict(FAMC, dscnn=(S.INK2, "X", "DS-CNN"))
    for f, (col, mk, lab) in st.items():
        Q = [p for p in pts if p[0] == f]
        ax.scatter([p[1] for p in Q], [p[2] for p in Q], s=40, marker=mk, color=col, alpha=0.8, lw=0.6, edgecolor="white",
                   label=lab.replace(" (8 checkpoints)", " (9 checkpoints)"), zorder=3)
    lo_, hi_ = ax.get_ylim(); ax.set_ylim(min(lo_, -0.3), hi_)
    S.shade(ax, "y", label_better="positive gradient better", label_worse="absolute gradient better")
    ax.set_xlabel("accuracy lost at p = 3e-3 with the absolute-gradient mask (points)")
    ax.set_ylabel("largest low-rate gain of positive\nover absolute gradient (points)")
    ax.legend(fontsize=7, loc="upper right")
    OUT["room"] = [dict(family=a, lost=round(b, 2), gain=round(c, 2)) for a, b, c in pts]
    fig.tight_layout()
    save(fig, "fig_room")


# ------------------------------------------------------------------ appendix: every checkpoint, every scheme
def curves(tag, conds):
    ids = P.ORDER
    ncol, nrow = 4, int(np.ceil(len(P.ORDER) / 4))
    fig, axs = plt.subplots(nrow, ncol, figsize=(12, 2.35 * nrow + 0.5), squeeze=False)
    for k, cid in enumerate(ids):
        ax = axs[k // ncol, k % ncol]
        vals, clean, unit, lower = P.per_draw(cid)
        band = (clean, clean + 0.5) if lower else (clean - 5, clean)
        ax.axhspan(*band, color=S.GRID, alpha=0.8, lw=0)
        for cond, lab, col, ls, mk in conds:
            xs = sorted(b for (c, b) in vals if c == cond)
            ax.plot(xs, [np.mean(list(vals[(cond, b)].values())) for b in xs], ls, color=col, lw=1.4, marker=mk, ms=2.6,
                    label=lab, alpha=0.95)
        ax.set_xscale("log")
        if lower:
            ax.set_yscale("log"); ax.invert_yaxis()
            ax.set_yticks([3, 5, 10, 20, 30]); ax.yaxis.set_major_formatter(S.plain)
            ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        else:
            ax.set_ylim(max(0, clean - 45), clean + 3)
        ax.set_title(P.NAMES[cid], loc="left", fontsize=8)
        ax.set_xticks(S.BER_TICKS)
        ax.tick_params(labelsize=6.5)
        if k % ncol == 0 or cid in P.LMS:
            ax.set_ylabel("AUC (points)" if cid == "toycar" else "accuracy (%)" if not lower else "loss (nats/token), inverted",
                          fontsize=7)
        if k // ncol == nrow - 1:
            ax.set_xlabel("bit-error probability", fontsize=7.5)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7.5, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    save(fig, f"fig_curves_{tag}")


def main():
    model_check(); no_static(); reversal_all(); rate_choice(); room()
    curves("25", [("tmr_abs@0.125", "triplicated, absolute gradient", S.ABS, "-", "o"),
                  ("tmr_pos@0.125", "triplicated, positive gradient", S.POS, "-", "o"),
                  ("tmr_global@0.125", "triplicated, global gradient", S.GLOBAL, "-", "o"),
                  ("tmr_fkeras_weighted@0.125", "triplicated, FKeras tier-mixing", S.FKW, "--", "D"),
                  ("tmr_msb@0.125", "triplicated sign bits (FORTUNE = FKeras)", S.MSB, "--", "s"),
                  ("secded_32", "SEC-DED code over 32 bits", S.SECDED, "-", "^"),
                  ("abft_ideal", "ABFT checksums", S.ABFT, ":", "v"), ("unprotected", "no protection", S.NONE, ":", "x")])
    curves("125", [("tmr_abs@0.0625", "triplicated, absolute gradient", S.ABS, "-", "o"),
                   ("tmr_fkeras@0.0625", "triplicated, FKeras", S.FK, "-", "D"),
                   ("tmr_msb@0.0625", "triplicated, MSB-first", S.MSB, "--", "s"),
                   ("secded_64", "SEC-DED code over 64 bits", S.SECDED, "-", "^"),
                   ("abft_ideal", "ABFT, fault-free checksums", S.ABFT, ":", "v"),
                   ("abft_faulty", "ABFT, checksums can fail", S.ABFT, "-.", "v"), ("unprotected", "no protection", S.NONE, ":", "x")])
    (C.ROOT / "data/tecs_paper_figures/figures2.json").write_text(json.dumps(OUT, indent=1, default=float) + "\n")
    print(json.dumps({k: v for k, v in OUT.items() if k in ("model_check",)}, indent=1))


if __name__ == "__main__":
    main()
