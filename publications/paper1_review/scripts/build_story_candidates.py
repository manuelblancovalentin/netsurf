#!/usr/bin/env python3
"""Candidate figures for the paper's story (exploratory drafts, not yet in the manuscript).

Reads only locked outputs: data/tecs_audit (main grid), data/tecs_rate_choice (Run R Stages 1/4, gradient trust),
workspace/runs/paper1_runR_stage2_eval_v1 (raw Stage 2 accuracies) and paper1_runR_v2_picks_v1/picks.json.
Writes PNGs and candidates.json (the plotted numbers) to <out>, which must not exist.

    build_story_candidates.py <out>
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT.parents[1] / "workspace/runs"
RC = ROOT / "data/tecs_rate_choice"
PICKS_SHA = "8c1a864f5310903d941d2d2e4eae51c5479c1aed25b0d78fa680eda347618335"
ABS, POS, V2, MSB, INK, INK2, GRID, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#52514e", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "lines.linewidth": 2, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.facecolor": SURF, "legend.frameon": False})
R8 = ["pinned", "retrained_seed7101", "retrained_seed7102", "retrained_seed7103", "retrained_seed7104",
      "retrained_seed7105", "pruned_s05", "pruned_s07", "undertrained_e100"]
R8_LABEL = dict(pinned="pinned", undertrained_e100="under-trained", pruned_s05="pruned 50%", pruned_s07="pruned 70%",
                **{f"retrained_seed710{i}": f"R{i}" for i in range(1, 6)})
GT = ["jet-e1", "jet-e3", "jet-e10", "jet-e30", "jet-e100", "mnist-e1", "mnist-e3", "mnist-e10", "mnist-e30", "vww"]
OUT: dict = {}


def rows(p):
    with open(p) as fh:
        return list(csv.DictReader(fh))


def title(ax, t):
    ax.set_title(t, loc="left", fontsize=9, color=INK)


# ---------------------------------------------------------------- A: blind protection fails at dense rates, everywhere
def fig_blind(out):
    main = defaultdict(dict)
    for r in rows(ROOT / "data/tecs_audit/conditions_with_quality.csv"):
        if r["model"] in ("resnet8", "kws", "distilbert") and float(r["coverage"]) in (0.0, 0.125):
            main[r["model"]][(r["method"], float(r["ber"]))] = (100 * float(r["mean_quality"]), 100 * float(r["clean_quality"]))
    fm = defaultdict(dict)
    for r in rows(RC / "fault_means.csv"):
        if float(r["budget"]) == 0.125:
            fm[r["checkpoint"]][(r["arm"], float(r["ber"]))] = (100 * float(r["mean_accuracy"]), 100 * float(r["clean_accuracy"]))
    panels = [("ResNet-8 (CIFAR-10)", main["resnet8"], "abs_grad", "msb", "unprotected"),
              ("DS-CNN (keywords)", main["kws"], "abs_grad", "msb", "unprotected"),
              ("DistilBERT (SST-2)", main["distilbert"], "abs_grad", "msb", "unprotected"),
              ("Visual Wake Words", fm["vww"], "abs1024", "msb", None),
              ("MNIST MLP, epoch 30", fm["mnist-e30"], "abs1024", "msb", None),
              ("Jet MLP, epoch 30", fm["jet-e30"], "abs1024", "msb", None)]
    fig, axs = plt.subplots(2, 3, figsize=(11, 5.6), sharex=True)
    OUT["A_blind"] = {}
    for ax, (name, d, a, m, u) in zip(axs.flat, panels):
        clean = next(iter(d.values()))[1]
        ax.axhspan(clean - 5, clean, color=GRID, alpha=0.6, lw=0)
        rec = {}
        for arm, col, lab, ls in ((a, ABS, "absolute gradient", "-"), (m, MSB, "MSB-first", "-"), (u, "#a8a7a2", "unprotected", ":")):
            if arm is None:
                continue
            xs = sorted(p for (k, p) in d if k == arm and 1e-4 <= p <= 3e-2)
            ys = [d[(arm, p)][0] for p in xs]
            ax.plot(xs, ys, ls, color=col, marker="o", ms=3.5, label=lab)
            rec[lab] = dict(zip(map(str, xs), [round(y, 2) for y in ys]))
        OUT["A_blind"][name] = rec
        ax.set_xscale("log"); title(ax, name)
        ax.set_ylim(max(0, clean - 45), clean + 2)
    for ax in axs[1]:
        ax.set_xlabel("bit-error probability p")
    for ax in axs[:, 0]:
        ax.set_ylabel("accuracy (%)")
    axs[0, 0].legend(loc="lower left", fontsize=8)
    fig.suptitle("A. At 12.5% protection, protecting by bit significance gives up most of the useful range; "
                 "a gradient ranking keeps it (grey band: within 5 points of clean)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "A_blind_vs_gradient.png", dpi=160)


# ---------------------------------------------------------------- B: magnitude vs direction, the reversal is ResNet-8's
def fig_reversal(out):
    s1 = [r for r in rows(RC / "stage1_cells.csv") if float(r["budget"]) == 0.125 and r["usable"] == "True"]
    s4 = [r for r in rows(RC / "stage4_switch_cells.csv") if float(r["budget"]) == 0.125 and r["usable"] == "True"]
    gt = json.loads((RC / "gradtrust_summary.json").read_text())["checkpoints"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    OUT["B_reversal"] = {"resnet8": {}, "others": {}}
    for i, name in enumerate(R8):
        rr = sorted((float(r["ber"]), float(r["diff"])) for r in s1 if r["checkpoint"] == name)
        col = POS if name == "undertrained_e100" else ABS
        axs[0].plot(*zip(*rr), color=col, alpha=0.9 if name in ("pinned", "undertrained_e100") else 0.45,
                    lw=2 if name in ("pinned", "undertrained_e100") else 1.2, label=R8_LABEL[name] if name in ("pinned", "undertrained_e100") else None)
        ps = float(next(r["p_star"] for r in s1 if r["checkpoint"] == name))
        axs[0].plot([ps], [0], "|", color=col, ms=10, mew=1.5)
        OUT["B_reversal"]["resnet8"][name] = dict(cells=rr, p_star=ps)
    axs[0].plot([], [], color=ABS, alpha=0.45, lw=1.2, label="retrained / pruned")
    axs[0].plot([], [], "|", color=INK2, ms=10, label="predicted switch p*")
    title(axs[0], "ResNet-8, nine checkpoints: a reversal, where predicted")
    rr = sorted((float(r["ber"]), float(r["diff"])) for r in s4)
    axs[1].plot(*zip(*rr), color=INK, lw=2.2, label="DS-CNN")
    OUT["B_reversal"]["others"]["DS-CNN"] = rr
    for cid in GT:
        c = sorted((x["ber"], x["diff"]) for x in gt[cid]["cells"] if x["budget"] == 0.125)
        col = {"jet": ABS, "mnist": POS, "vww": V2}[cid.split("-")[0]]
        axs[1].plot(*zip(*c), color=col, alpha=0.55, lw=1.2)
        OUT["B_reversal"]["others"][cid] = c
    for lab, col in (("jet MLP (5 checkpoints)", ABS), ("MNIST MLP (4 checkpoints)", POS), ("VWW", V2)):
        axs[1].plot([], [], color=col, alpha=0.55, lw=1.2, label=lab)
    title(axs[1], "DS-CNN and ten further checkpoints: no reversal")
    for ax in axs:
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=1)
        ax.set_xlabel("bit-error probability p"); ax.legend(fontsize=8, loc="lower left")
    axs[0].set_ylabel("positive minus absolute gradient (points)")
    fig.suptitle("B. Protecting by a flip's direction beats protecting by its size only on ResNet-8, and only at sparse "
                 "faults (12.5%, usable cells, 1,024-input masks)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "B_reversal_resnet8_only.png", dpi=160)


# ---------------------------------------------------------------- C: collapse on the predicted switch
def fig_collapse(out):
    s1 = [r for r in rows(RC / "stage1_cells.csv") if r["usable"] == "True" and float(r["budget"]) >= 0.0625]
    s4 = [r for r in rows(RC / "stage4_switch_cells.csv") if r["usable"] == "True"]
    fig, ax = plt.subplots(figsize=(6.4, 4))
    x = np.array([float(r["ber"]) / float(r["p_star"]) for r in s1]); y = np.array([float(r["diff"]) for r in s1])
    ax.scatter(x, y, s=9, color=ABS, alpha=0.45, lw=0, label=f"ResNet-8, 9 checkpoints x 6 budgets ({len(x)} cells)")
    x4 = np.array([float(r["ber"]) / float(r["p_star"]) for r in s4]); y4 = np.array([float(r["diff"]) for r in s4])
    ax.scatter(x4, y4, s=16, color=POS, marker="D", lw=0, label=f"DS-CNN, 4 budgets ({len(x4)} cells)")
    ax.axhline(0, color=INK2, lw=0.8); ax.axvline(1, color=INK2, lw=0.8, ls="--")
    ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel("p / predicted switch p*"); ax.set_ylabel("positive minus absolute gradient (points)")
    title(ax, "C. Rescaled by the predicted switch, ResNet-8 changes sign at p* (DS-CNN does not)")
    ax.legend(fontsize=8, loc="lower left")
    OUT["C_collapse"] = dict(resnet8_cells=len(x), resnet8_sign_ok_away=int(np.sum(((x < 1 / 1.25) & (y > 0)) | ((x > 1.25) & (y < 0)))),
                             resnet8_away=int(np.sum((x < 1 / 1.25) | (x > 1.25))), kws_cells=len(x4))
    fig.tight_layout()
    fig.savefig(out / "C_switch_collapse.png", dpi=160)


# ---------------------------------------------------------------- D/E: Stage 2 raw accuracies
def stage2():
    root = RUNS / "paper1_runR_stage2_eval_v1"
    pk = RUNS / "paper1_runR_v2_picks_v1/picks.json"
    assert hashlib.sha256(pk.read_bytes()).hexdigest() == PICKS_SHA
    picks = json.loads(pk.read_text())["picks"]
    acc, cross, clean = defaultdict(dict), defaultdict(dict), defaultdict(dict)
    for part in ("main", "mc"):
        for d in sorted((root / part).glob("*/rep*/field*")):
            if not (d / "DONE").exists():
                raise SystemExit(f"unfinished {d}")
            name, rep, f = d.parts[-3], int(d.parts[-2][3:]), int(d.parts[-1][5:])
            if part == "main":
                clean[name][rep] = json.loads((d / "field.json").read_text())["clean_accuracy"]
            for r in rows(d / "results.csv"):
                b, p, a, v = round(float(r["budget"]), 6), round(float(r["ber"]), 8), r["arm"], float(r["accuracy"])
                if r["built_for"] == "" or round(float(r["built_for"]), 8) == p:
                    acc[(name, b, p, a)][(rep, f)] = v
                elif a == "ra_analytic":
                    cross[(name, b, round(float(r["built_for"]), 8), p)][(rep, f)] = v
    for (name, b, p, a) in [k for k in acc if k[3] == "ra_analytic"]:
        acc[(name, b, p, "v2")] = {(r, f): acc[(name, b, p, picks[f"resnet8/{name}/rep{r}/{b:.6f}/{p:g}"])][(r, f)]
                                    for (r, f) in acc[(name, b, p, "ra_analytic")]}
    return acc, cross, {n: float(np.mean(list(c.values()))) for n, c in clean.items()}


def fig_no_fixed(out, acc, clean):
    arms_all = sorted({k[3] for k in acc})
    rates = sorted({k[2] for k in acc})
    show = (("abs1024", ABS, "absolute gradient (fixed)"), ("pos1024", POS, "positive gradient (fixed)"),
            ("v2", V2, "chosen by expected rate (v2)"), ("msb", MSB, "MSB-first (fixed)"))
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    OUT["D_no_fixed"] = {}
    for ax, b in zip(axs, (0.125, 0.25)):
        for arm, col, lab in show:
            med, lo, hi, xs = [], [], [], []
            for p in rates:
                gaps = []
                for n in R8:
                    m = {a: 100 * np.mean(list(acc[(n, b, p, a)].values())) for a in arms_all if acc.get((n, b, p, a))}
                    if max(m.values()) >= 100 * clean[n] - 5:
                        gaps.append(max(m.values()) - m[arm])
                if len(gaps) >= 5:
                    xs.append(p); med.append(np.median(gaps)); lo.append(min(gaps)); hi.append(max(gaps))
            ax.fill_between(xs, lo, hi, color=col, alpha=0.12, lw=0)
            ax.plot(xs, med, color=col, marker="o", ms=3.5, label=lab)
            OUT["D_no_fixed"][f"{b}/{arm}"] = dict(rates=xs, median=[round(v, 2) for v in med], max=[round(v, 2) for v in hi])
        ax.axhline(0.5, color=INK2, lw=0.8, ls="--")
        ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=1)
        ax.set_xlabel("bit-error probability p"); title(ax, f"{b:.1%} protection")
    axs[0].set_ylabel("accuracy behind the best arm (points)")
    axs[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("D. ResNet-8: each fixed ranking falls behind on one side of the switch; choosing by rate stays close\n"
                 "(median over 9 checkpoints, band = range; rates usable on at least 5 checkpoints; dashed = 0.5 point)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "D_no_fixed_ranking_resnet8.png", dpi=160)


def fig_wrong_rate(out, acc, cross, clean):
    arms_all = sorted({k[3] for k in acc})
    rates = sorted({k[2] for k in acc})
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.3))
    OUT["E_wrong_rate"] = {}
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
        im = ax.imshow(G, cmap="Blues", norm=matplotlib.colors.SymLogNorm(1, vmin=0, vmax=70), origin="lower")
        for i in range(len(rates)):
            for j in range(len(rates)):
                if np.isfinite(G[i, j]):
                    ax.text(j, i, f"{G[i, j]:.1f}" if G[i, j] < 10 else f"{G[i, j]:.0f}", ha="center", va="center",
                            fontsize=7, color="white" if G[i, j] > 8 else INK)
        lab = [f"{r:.0e}".replace("e-0", "e-") for r in rates]
        ax.set_xticks(range(len(rates)), lab, rotation=45, fontsize=7); ax.set_yticks(range(len(rates)), lab, fontsize=7)
        ax.set_xlabel("rate at which the mask is used"); ax.set_ylabel("rate the mask was built for")
        ax.grid(False); title(ax, R8_LABEL[n])
        OUT["E_wrong_rate"][n] = dict(rates=rates, gap=np.where(np.isfinite(G), np.round(G, 2), None).tolist())
    fig.colorbar(im, ax=axs, shrink=0.8, label="points behind the best arm")
    fig.suptitle("E. ResNet-8: building for too high a rate costs little; building for too low a rate costs up to 70 points\n"
                 "(analytic rate-aware mask, mean gap over usable budgets; blank = not usable)", x=0.01, ha="left", fontsize=10)
    fig.savefig(out / "E_built_for_vs_used_at.png", dpi=160, bbox_inches="tight")


# ---------------------------------------------------------------- F: what is different about ResNet-8 (open)
def fig_room(out):
    pts = []
    s1 = rows(RC / "stage1_cells.csv")
    for n in R8:
        c = [r for r in s1 if r["checkpoint"] == n and float(r["budget"]) == 0.125]
        at = {round(float(r["ber"]), 6): r for r in c}
        lost = 100 * (float(at[0.003]["clean"]) - float(at[0.003]["mean_abs"]))
        gain = max(float(r["diff"]) for r in c if float(r["ber"]) <= 3e-3 and r["usable"] == "True")
        pts.append(("ResNet-8", R8_LABEL[n], lost, gain))
    s4 = {round(float(r["ber"]), 6): r for r in rows(RC / "stage4_switch_cells.csv") if float(r["budget"]) == 0.125}
    clean4 = json.loads((RC / "stage4_summary.json").read_text())["clean_accuracy"]
    pts.append(("DS-CNN", "DS-CNN", 100 * (clean4 - float(s4[0.003]["mean_abs"])),
                max(float(r["diff"]) for p, r in s4.items() if p <= 3e-3 and r["usable"] == "True")))
    s = json.loads((RC / "summary.json").read_text())["gradtrust"]["checkpoints"]
    gt = json.loads((RC / "gradtrust_summary.json").read_text())["checkpoints"]
    for cid in GT:
        lost = s[cid]["clean"] - s[cid]["acc_125"]["abs1024"]["0.003"]
        gain = max([x["diff"] for x in gt[cid]["cells"] if x["budget"] == 0.125 and x["ber"] <= 3e-3] or [np.nan])
        pts.append((cid.split("-")[0], cid, lost, gain))
    col = {"ResNet-8": ABS, "DS-CNN": INK, "jet": POS, "mnist": "#e87ba4", "vww": V2}
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for fam in col:
        P = [p for p in pts if p[0] == fam]
        ax.scatter([p[2] for p in P], [p[3] for p in P], s=36, color=col[fam], lw=0.8, edgecolor=SURF,
                   label={"jet": "jet MLP", "mnist": "MNIST MLP", "vww": "VWW"}.get(fam, fam), zorder=3)
    for p in pts:
        if p[0] in ("jet",) or p[1] in ("under-trained", "pinned"):
            ax.annotate(p[1], (p[2], p[3]), fontsize=7, color=INK2, xytext=(4, 3), textcoords="offset points")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel("accuracy lost at p = 3e-3 under the absolute-gradient mask (points)")
    ax.set_ylabel("best low-rate gain of positive over absolute (points)")
    title(ax, "F. Damage left at sparse rates does not explain the reversal (12.5%, p <= 3e-3)")
    ax.legend(fontsize=8, loc="upper right")
    OUT["F_room"] = [dict(family=a, checkpoint=b, lost_at_3e3=round(c, 2), best_low_rate_gain=round(d, 2)) for a, b, c, d in pts]
    fig.tight_layout()
    fig.savefig(out / "F_room_vs_gain.png", dpi=160)


def main(out: Path):
    out.mkdir(parents=True, exist_ok=False)
    fig_blind(out); fig_reversal(out); fig_collapse(out); fig_room(out)
    acc, cross, clean = stage2()
    fig_no_fixed(out, acc, clean); fig_wrong_rate(out, acc, cross, clean)
    (out / "candidates.json").write_text(json.dumps(OUT, indent=1, default=float) + "\n")
    with open(out / "SHA256SUMS", "w") as fh:
        for f in sorted(out.iterdir()):
            if f.name != "SHA256SUMS":
                fh.write(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.name}\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
