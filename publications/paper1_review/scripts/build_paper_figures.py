#!/usr/bin/env python3
"""Paper figures from the failing-protection runs (Run T), all models.

  figures/fig_headline.{pdf,png}   per model: task quality against the share of protected bits (triplicated, copies may
                                   fail) for the absolute gradient, FKeras and MSB-first, at the highest grid BER at
                                   which the absolute-gradient mask at 12.5% keeps the model usable; FORTUNE (all sign
                                   bits) marked; largest gain over MSB-first annotated
  figures/fig_summary.{pdf,png}    every checkpoint: paired difference of the absolute gradient minus MSB-first / FKeras /
                                   FKeras tier-mixing variant at 12.5% of bits, at the same BER rule, 95% intervals
  figures/fig_mechanism_map.{...}  highest usable BER of every protection scheme, all checkpoints (~25% extra storage)
  data/tecs_paper_figures/*.json   every plotted number
Inputs: workspace/runs/paper1_runT_tflite_v1, paper1_runT_w8a8_v1 and the locked Run T analyses (tolerated rates).
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_story_candidates as C  # noqa: E402
import paper_style as S  # noqa: E402

TFL = C.RUNS / "paper1_runT_tflite_v1"
LM = C.RUNS / "paper1_runT_w8a8_v1"
LM_MORE = C.RUNS / "paper1_runT_w8a8_more_v1"          # addendum T4: draws 8-15 (Qwen3 4-15)
LM_ANALYSES = ("paper1_runT_t2_lm_analysis_v1", "paper1_runT_t4_lm_analysis_dbpy_v1", "paper1_runT_t4_lm_analysis_v1")


def sixteen():
    """Language models whose 16-draw analysis exists: their added draws are used everywhere."""
    out = set()
    for a in LM_ANALYSES[1:]:
        f = C.RUNS / a / "summary.json"
        if f.exists():
            out |= set(json.loads(f.read_text())["targets"])
    return out
OUT = C.ROOT / "data/tecs_paper_figures"
ANALYSES = ("paper1_runT_t1a_analysis_v1", "paper1_runT_t1b_fkeras_analysis_v1", "paper1_runT_t1b_engine_analysis_v1",
            "paper1_runT_t3_toycar_analysis_v1")
LMS = {"distilbert_sst2_int8": ("accuracy", 100.0, "accuracy (%)"), "pythia160m_int8": ("mean_nll", 1.0, "loss (nats/token)"),
       "qwen3_0_6b_int8": ("mean_nll", 1.0, "loss (nats/token)")}
NAMES = {"jet-e1": "Jet MLP, epoch 1", "jet-e3": "Jet MLP, epoch 3", "jet-e10": "Jet MLP, epoch 10", "jet-e30": "Jet MLP, epoch 30",
         "jet-e100": "Jet MLP, epoch 100", "mnist-e1": "MNIST MLP, epoch 1", "mnist-e3": "MNIST MLP, epoch 3",
         "mnist-e10": "MNIST MLP, epoch 10", "mnist-e30": "MNIST MLP, epoch 30", "dscnn": "DS-CNN", "vww": "Visual Wake Words",
         "resnet8-pinned": "ResNet-8", "resnet8-retrained_seed7101": "ResNet-8, retrained 1",
         "resnet8-retrained_seed7102": "ResNet-8, retrained 2", "resnet8-retrained_seed7103": "ResNet-8, retrained 3",
         "resnet8-retrained_seed7104": "ResNet-8, retrained 4", "resnet8-retrained_seed7105": "ResNet-8, retrained 5",
         "resnet8-pruned_s05": "ResNet-8, pruned 50%", "resnet8-pruned_s07": "ResNet-8, pruned 70%",
         "resnet8-undertrained_e100": "ResNet-8, under-trained", "toycar": "ToyCar (AUC)",
         "distilbert_sst2_int8": "DistilBERT", "pythia160m_int8": "Pythia-160M", "qwen3_0_6b_int8": "Qwen3-0.6B"}
ORDER = ["jet-e1", "jet-e3", "jet-e10", "jet-e30", "jet-e100", "mnist-e1", "mnist-e3", "mnist-e10", "mnist-e30", "dscnn", "vww",
         "resnet8-pinned", "resnet8-retrained_seed7101", "resnet8-retrained_seed7102", "resnet8-retrained_seed7103",
         "resnet8-retrained_seed7104", "resnet8-retrained_seed7105", "resnet8-pruned_s05", "resnet8-pruned_s07",
         "resnet8-undertrained_e100", "toycar", "distilbert_sst2_int8", "pythia160m_int8", "qwen3_0_6b_int8"]
HEAD = ["jet-e30", "mnist-e30", "vww", "dscnn", "resnet8-pinned", "distilbert_sst2_int8", "pythia160m_int8", "qwen3_0_6b_int8"]
GRID = [1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
SHARES = (0.0625, 0.125, 0.1875, 0.25)
FK, FKW = S.FK, S.FKW
S.apply()


def per_draw(cid):
    """{(condition, ber): {unit: value}}, clean value, unit label, lower_is_better."""
    vals, clean = defaultdict(dict), []
    if cid in LMS:
        metric, scale, unit = LMS[cid]
        dirs = sorted((LM / cid).glob("draw*/ber*")) + (sorted((LM_MORE / cid).glob("draw*/ber*")) if cid in sixteen() else [])
        for d in dirs:
            assert (d / "DONE").exists() and (d / "DONE_fkeras").exists(), d
            dr = int(d.parent.name[4:])
            for f in sorted(d.glob("results*.csv")):
                for r in csv.DictReader(open(f)):
                    vals[(r["condition"], float(r["ber"]))][dr] = scale * float(r[f"metric_{metric}"])
            clean.append(scale * json.loads((d / "DONE").read_text())["clean"][metric])
        return vals, float(np.mean(clean)), unit, metric == "mean_nll"
    for d in sorted((TFL / cid).glob("rep*/draw*")):
        u = (int(d.parent.name[3:]), int(d.name[4:]))
        for f in sorted(d.glob("results*.csv")):
            for r in csv.DictReader(open(f)):
                vals[(r["condition"], float(r["ber"]))][u] = 100 * float(r["accuracy"])
        clean.append(100 * json.loads((d / "DONE").read_text())["clean_accuracy"])
    return vals, float(np.mean(clean)), ("AUC (points)" if cid == "toycar" else "accuracy (%)"), False


def tolerated():
    tol = {}
    for a in ANALYSES + LM_ANALYSES:
        if not (C.RUNS / a / "summary.json").exists():
            continue
        s = json.loads((C.RUNS / a / "summary.json").read_text())
        for cid, v in s.get("checkpoints", s.get("targets", {})).items():
            tol.setdefault(cid, {}).update(v["tolerated"])
    return tol


def ci(d):
    d = np.asarray(d, float)
    m = d.mean()
    h = 1.96 * d.std(ddof=1) / np.sqrt(d.size) if d.size > 1 else 0.0
    return m, m - h, m + h


def headline(tol, data, record):
    fig, axs = plt.subplots(2, 4, figsize=(11.5, 5.4))
    for ax, cid in zip(axs.flat, HEAD):
        vals, clean, unit, lower = data[cid]
        p = tol[cid]["tmr_abs@0.125"]
        x = [100 * s for s in SHARES]
        rec = {"ber": p, "clean": clean}
        for arm, lab, col, ls in (("abs", "absolute gradient", S.ABS, "-"), ("fkeras", "FKeras", S.FK, "-"),
                                  ("msb", "MSB-first", S.MSB, "--")):
            ms = [np.mean(list(vals[(f"tmr_{arm}@{s}", p)].values())) for s in SHARES]
            se = [np.std(list(vals[(f"tmr_{arm}@{s}", p)].values()), ddof=1) / np.sqrt(len(vals[(f"tmr_{arm}@{s}", p)])) for s in SHARES]
            ax.fill_between(x, np.subtract(ms, se), np.add(ms, se), color=col, alpha=0.15, lw=0)
            ax.plot(x, ms, ls, color=col, lw=2, marker="o", ms=3.5, label=lab)
            rec[arm] = ms
        none = np.mean(list(vals[("unprotected", p)].values()))
        allv = rec["abs"] + rec["fkeras"] + rec["msb"] + [clean]
        lo_, hi_ = min(allv), max(allv)
        pad = 0.12 * (hi_ - lo_)
        ylim = (lo_ - pad, hi_ + 2.2 * pad) if not lower else (lo_ - 2.2 * pad, hi_ + pad)
        if ylim[0] <= none <= ylim[1]:
            ax.axhline(none, color=S.NONE, lw=1, ls=":", label="no protection")
        ax.axhline(clean, color=C.INK2, lw=0.8, ls=(0, (1, 3)), label="no faults")
        ax.set_ylim(*ylim)
        fort = rec["msb"][1]
        ax.plot([12.5], [fort], "s", color=S.INK, ms=6, mec="white", mew=0.8, zorder=5, label="FORTUNE (all sign bits)")
        gain = [(a - b) * (-1 if lower else 1) for a, b in zip(rec["abs"], rec["msb"])]
        k = int(np.argmax(gain))
        ax.plot([x[k]], [rec["abs"][k]], "o", color=S.ABS, ms=8, mec="white", mew=1.2, zorder=6,
                label="largest gain over MSB-first")
        lo_k, hi_k = sorted((rec["abs"][k], rec["msb"][k]))
        ax.fill_between([x[k] - 0.9, x[k] + 0.9], lo_k, hi_k, color=S.BETTER, alpha=0.16, lw=0, zorder=1)
        txt = (f"+{gain[k]:.1f} points" if not lower else f"{gain[k]:.2f} nats/token lower loss") + f"\nat {x[k]:g}% of bits"
        ax.text(0.97, 0.05, txt, transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color=S.INK,
                fontweight="bold", bbox=dict(boxstyle="round,pad=0.25", fc=S.BETTER, ec="none", alpha=0.12))
        ax.set_title(NAMES[cid], loc="left")
        ax.set_title(f"p = {p:g}", loc="right", fontsize=7.5, color=C.INK2)
        ax.set_xticks(x, ["6.25", "12.5", "18.75", "25"])
        ax.set_xlim(4.5, 26.5)
        if lower:
            ax.invert_yaxis()
        ax.set_ylabel(unit if not lower else "loss (nats/token), axis inverted")
        rec["largest_gain"], rec["at_share"] = float(gain[k]), x[k]
        rec["none"] = none
        record[cid] = rec
    for ax in axs[1]:
        ax.set_xlabel("protected bits (%), triplicated")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=7, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(C.ROOT / "figures" / f"fig_headline.{ext}", bbox_inches="tight", **({"dpi": 220} if ext == "png" else {}))
    plt.close(fig)


def summary(tol, data, record):
    """Forest plot: TFLite + DistilBERT in accuracy points (top), Pythia/Qwen in nats/token (bottom)."""
    top = [c for c in ORDER if c not in ("toycar", "pythia160m_int8", "qwen3_0_6b_int8")]
    bot = ["pythia160m_int8", "qwen3_0_6b_int8"]
    cols = (("fkeras", 0.0625, "vs FKeras\n(half of the sign bits, 6.25%)", S.FK),
            ("msb", 0.125, "vs FORTUNE = MSB-first = FKeras\n(all sign bits, 12.5%)", S.MSB),
            ("fkeras_weighted", 0.125, "vs FKeras tier-mixing variant\n(12.5%)", S.FKW))
    fig, axs = plt.subplots(2, 3, figsize=(10.5, 7.2), sharey="row", gridspec_kw=dict(height_ratios=(len(top), 2.6)))
    for row, ids, xl, xlab in ((0, top, (-4, 30), "gain of the absolute gradient (accuracy points)"),
                               (1, bot, (-0.3, 1.5), "gain (nats/token)")):
        ys = np.arange(len(ids))[::-1]
        for j, (right, share, lab, col) in enumerate(cols):
            ax = axs[row, j]
            for y, cid in zip(ys, ids):
                vals, clean, unit, lower = data[cid]
                p = tol[cid]["tmr_abs@0.125"]
                A, B = vals[(f"tmr_abs@{share}", p)], vals[(f"tmr_{right}@{share}", p)]
                d = np.array([A[u] - B[u] for u in sorted(A)]) * (-1 if lower else 1)
                m, lo, hi = ci(d)
                ax.plot([lo, hi], [y, y], color=col, lw=1.8, alpha=0.75, solid_capstyle="round")
                ax.plot([m], [y], "o", color=col, ms=4.8, mec="white", mew=0.6, zorder=3)
                record.setdefault(cid, {})[f"abs_minus_{right}@{share}"] = [round(m, 3), round(lo, 3), round(hi, 3)]
            ax.set_xlim(*xl)
            S.shade(ax, axis="x", labels=(j == 0), label_better="better", label_worse="worse")
            ax.set_ylim(-0.7, len(ids) - 0.3)
            if row == 0:
                ax.set_title(lab, loc="left", fontsize=8.5)
                for b in (len(ids) - 5 - 0.5, len(ids) - 9 - 0.5, len(ids) - 11 - 0.5, len(ids) - 20 - 0.5):
                    ax.axhline(b, color=C.INK2, lw=0.4, alpha=0.5)
            ax.set_xlabel(xlab, fontsize=7.5)
            ax.grid(axis="y", visible=False)
        axs[row, 0].set_yticks(ys, [NAMES[c] for c in ids], fontsize=7.5)
    fig.tight_layout(h_pad=1.2)
    for ext in ("pdf", "png"):
        fig.savefig(C.ROOT / "figures" / f"fig_summary.{ext}", bbox_inches="tight", **({"dpi": 220} if ext == "png" else {}))
    plt.close(fig)


def mechanism_map(tol, record):
    """Table-heat map: rows = checkpoints, columns = protection schemes at about 25% extra storage.
    Cell shade = highest grid BER at which the model stays usable (darker = tolerates more faults); outlined = best."""
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle
    cols = [("unprotected", "no\nprotection"), ("abft_ideal", "ABFT\nchecksums"),
            ("tmr_msb@0.125", "triplicated\nsign bits\n(FORTUNE\n= FKeras)"),
            ("tmr_fkeras_weighted@0.125", "triplicated,\nFKeras\ntier-mixing"),
            ("secded_32", "SEC-DED\ncode,\n32 bits"), ("tmr_abs@0.125", "triplicated,\nabsolute\ngradient")]
    lab = ["$10^{-5}$", "$10^{-4}$", "3e-4", "$10^{-3}$", "3e-3", "$10^{-2}$", "3e-2", "$10^{-1}$"]
    txt = ["1e-5", "1e-4", "3e-4", "1e-3", "3e-3", "1e-2", "3e-2", "1e-1"]
    cmap = LinearSegmentedColormap.from_list("seqblue", ["#f2f7fb", "#9cc3e0", S.ABS, "#00305a"])
    n, m = len(ORDER), len(cols)
    fig, ax = plt.subplots(figsize=(6.4, 8.4))
    for i, cid in enumerate(ORDER):
        idx = [GRID.index(tol[cid][c]) if tol[cid].get(c) is not None else -1 for c, _ in cols]
        best = max(idx[1:])
        for j, k in enumerate(idx):
            v = (k + 1) / len(GRID)
            ax.add_patch(Rectangle((j, n - 1 - i), 1, 1, color=cmap(v), lw=0))
            ax.text(j + 0.5, n - 1 - i + 0.5, txt[k] if k >= 0 else "<1e-5", ha="center", va="center", fontsize=6.8,
                    color="white" if v > 0.55 else S.INK)
            if j > 0 and k == best:
                ax.add_patch(Rectangle((j + 0.05, n - 1 - i + 0.07), 0.9, 0.86, fill=False, ec=S.OI["orange"], lw=1.8))
    for b in (5, 9, 11, 20, 21):
        ax.axhline(n - b, color="white", lw=3)
    ax.set_xlim(0, m); ax.set_ylim(0, n)
    ax.set_xticks(np.arange(m) + 0.5, [l for _, l in cols], fontsize=7)
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(n)[::-1] + 0.5, [NAMES[c] for c in ORDER], fontsize=7.5)
    ax.tick_params(length=0); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(C.ROOT / "figures" / f"fig_mechanism_map.{ext}", bbox_inches="tight", **({"dpi": 220} if ext == "png" else {}))
    plt.close(fig)
    record["tolerated"] = {c: tol[c] for c in ORDER}
    record["columns"] = [c for c, _ in cols]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tol = tolerated()
    data = {c: per_draw(c) for c in ORDER}
    rec_h, rec_s, rec_m = {}, {}, {}
    headline(tol, data, rec_h)
    summary(tol, data, rec_s)
    mechanism_map(tol, rec_m)
    for nm, r in (("headline", rec_h), ("summary", rec_s), ("mechanism_map", rec_m)):
        (OUT / f"{nm}.json").write_text(json.dumps(r, indent=1, default=float) + "\n")
    print(json.dumps({c: (round(v["largest_gain"], 2), v["at_share"], v["ber"]) for c, v in rec_h.items()}))


if __name__ == "__main__":
    main()
