"""Compact manuscript alternatives from existing, verified figure data only.

Does not import the original build script (which has write side effects), a model
runtime, or the database. Writes only figures/compact_*.{pdf,png,svg}. Existing
data, figures, scientific sources, and manuscript sections are never modified.
Run with /mnt/raid0/ml/conda_envs/manuelbv/envs/netsurf/bin/python.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "figures"
MODELS = ["resnet8", "kws", "toycar", "distilbert", "pythia", "qwen"]
NAMES = {"resnet8": "ResNet-8", "kws": "DS-CNN", "toycar": "ToyCar",
         "distilbert": "DistilBERT", "pythia": "Pythia-160M", "qwen": "Qwen3-0.6B"}
COLORS = {"random": "#8A8F98", "msb": "#242A33", "jump": "#E69F00",
          "global": "#0072B2", "absolute": "#009E73", "positive": "#D55E00",
          "fkeras": "#A64B8D", "unprotected": "#8996A5"}
LABELS = {"random": "Random", "msb": "MSB-first", "jump": "Numerical jump",
          "global": "Global gradient", "absolute": "Absolute gradient",
          "positive": "Positive gradient", "fkeras": "FKeras (published)",
          "unprotected": "Unprotected"}
ORDER = list(COLORS)
YLAB = {"resnet8": "ΔCE (nats/image)", "kws": "ΔCE (nats/clip)",
        "toycar": "AUC decrease", "distilbert": "Δ capped NLL\n(nats/example)",
        "pythia": "Δ capped NLL\n(nats/token)", "qwen": "Δ capped NLL\n(nats/token)"}
FONT = "DejaVu Sans"


def csv_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


CONDITIONS = csv_rows(DATA / "reliability_conditions.csv")
for row in CONDITIONS:
    for key in ["coverage", "ber", "mean", "low", "high", "clean"]:
        row[key] = float(row[key])
    assert np.isfinite([row[k] for k in ["mean", "low", "high", "clean"]]).all()
    assert row["low"] <= row["mean"] + 1e-11 <= row["high"] + 2e-11
assert len(CONDITIONS) == 1254
PAIRED = csv_rows(DATA / "wide_paired_comparisons.csv")
QWEN = read_json(DATA / "qwen_crossover.json")
KWS = read_json(DATA / "kws_pool_statistics.json")

plt.rcParams.update({
    "font.family": FONT, "font.size": 7, "axes.titlesize": 8,
    "axes.titleweight": "normal", "axes.labelsize": 7,
    "xtick.labelsize": 6.6, "ytick.labelsize": 6.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#A9AFB8", "axes.linewidth": .55,
    "axes.labelcolor": "#303842", "text.color": "#303842",
    "xtick.color": "#4E5865", "ytick.color": "#4E5865",
    "grid.color": "#DDE2E7", "grid.linewidth": .4,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "savefig.facecolor": "white", "figure.facecolor": "white",
})


def style(ax):
    ax.grid(axis="y", alpha=.6, zorder=0)
    ax.tick_params(length=2.2, width=.5, pad=2.3)
    ax.set_axisbelow(True)


def save(fig, stem):
    assert stem.startswith("compact_")
    for extension in ["pdf", "svg", "png"]:
        options = {"dpi": 240} if extension == "png" else {}
        # Fixed canvas size: no large title/footer or variable tight bounding box.
        fig.savefig(OUT / f"{stem}.{extension}", **options)
    plt.close(fig)


def shared_legend(fig, complete_tier=False):
    methods = [m for m in ORDER if not (complete_tier and m == "fkeras")]
    handles = [Line2D([0], [0], color=COLORS[m], lw=.95, alpha=.88,
                      marker="o", ms=2, ls="--" if m in ["fkeras", "unprotected"] else "-")
               for m in methods]
    labels = [LABELS[m] for m in methods]
    if complete_tier:
        labels[methods.index("msb")] = "MSB / FKeras†"
    handles.append(Line2D([0], [0], color="#66717D", lw=.7, ls=":"))
    labels.append("Clean / full")
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.515, .999),
               ncol=4 if complete_tier else 3, frameon=False, fontsize=6.4,
               handlelength=1.5, handletextpad=.45, columnspacing=1.3,
               labelspacing=.35, borderaxespad=0)


def source_series(model, method, coverage=None, ber=None):
    return [r for r in CONDITIONS if r["model"] == model and r["method"] == method
            and (coverage is None or r["coverage"] == coverage or method == "unprotected")
            and (ber is None or r["ber"] == ber)]


def values(rows):
    mean = np.asarray([r["mean"] - r["clean"] for r in rows])
    low = np.asarray([r["low"] - r["clean"] for r in rows])
    high = np.asarray([r["high"] - r["clean"] for r in rows])
    return mean, low, high


RANGES = {}


def ber_grid():
    fig, axes = plt.subplots(2, 3, figsize=(6.2, 3.62))
    for index, model in enumerate(MODELS):
        ax = axes.flat[index]
        bounds = [0.]
        for method in ORDER:
            # At this complete tier the saved FKeras masks/outcomes equal MSB.
            if method == "fkeras":
                continue
            rows = sorted(source_series(model, method, coverage=.125), key=lambda r: r["ber"])
            if not rows:
                continue
            x = [r["ber"] for r in rows]
            mean, low, high = values(rows)
            bounds.extend(low.tolist() + high.tolist())
            ax.fill_between(x, low, high, color=COLORS[method], alpha=.065, lw=0)
            ax.plot(x, mean, color=COLORS[method], lw=.95, alpha=.87,
                    marker="o", ms=1.8, markeredgewidth=0,
                    ls="--" if method == "unprotected" else "-", zorder=3)
        ax.axhline(0, color="#66717D", ls=":", lw=.7, zorder=2)
        ax.set_xscale("log")
        ax.set_xlim(8e-6, .13)
        ax.set_xticks([1e-5, 1e-3, 1e-1], [r"$10^{-5}$", r"$10^{-3}$", r"$10^{-1}$"])
        ax.tick_params(axis="x", which="minor", bottom=False)
        ax.set_title(f"({chr(97 + index)}) {NAMES[model]}", loc="left", pad=4)
        ax.set_ylabel(YLAB[model], labelpad=3)
        if model in ["resnet8", "kws", "distilbert"]:
            ax.set_yscale("asinh", linear_width=.1)
            low = min(bounds) - .01
            high = max(bounds) + max(.01, .04 * (max(bounds) - min(bounds)))
            ticks = [t for t in [-1, -.1, 0, .1, 1, 10, 100] if low <= t <= high]
            ax.set_yticks(ticks, [f"{t:g}" for t in ticks])
            ax.set_ylim(low, high)
            ax.text(1., 1.035, "asinh", transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=5.7, color="#7A838D")
        else:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
        if model == "toycar":
            # The empty interval remains empty; no invented continuation.
            ax.axvspan(.0105, .13, color="#F3F4F5", zorder=-2)
        style(ax)
        RANGES[f"ber125/{model}"] = {"x": list(ax.get_xlim()), "y": list(ax.get_ylim()),
                                      "observed_bers": sorted({r["ber"] for r in source_series(model, "msb", .125)})}
    shared_legend(fig, complete_tier=True)
    fig.supxlabel("Logical bit-error probability", y=.014, fontsize=7.6)
    fig.subplots_adjust(left=.093, right=.985, top=.845, bottom=.132,
                        wspace=.46, hspace=.54)
    save(fig, "compact_reliability_ber_125")


def budget_grid():
    fig, axes = plt.subplots(2, 3, figsize=(6.2, 3.72))
    for index, model in enumerate(MODELS):
        ax = axes.flat[index]
        for method in ORDER[:-1]:
            rows = sorted(source_series(model, method, ber=.01), key=lambda r: r["coverage"])
            if not rows:
                continue
            x = [100 * r["coverage"] for r in rows]
            mean, low, high = values(rows)
            ax.errorbar(x, mean, yerr=[mean - low, high - mean], color=COLORS[method],
                        marker="o", ms=2.2, markeredgewidth=0, lw=.9, alpha=.85,
                        elinewidth=.6, capsize=1.5, capthick=.6,
                        ls="--" if method == "fkeras" else "-", zorder=3)
        unprotected = source_series(model, "unprotected", ber=.01)
        assert len(unprotected) == 1
        r = unprotected[0]
        ax.axhline(r["mean"] - r["clean"], color=COLORS["unprotected"], ls="--", lw=.75)
        ax.axhline(0, color="#66717D", ls=":", lw=.7)
        ax.set_xticks([6.25, 12.5, 18.75, 25], ["6.25", "12.5", "18.75", "25"])
        ax.set_xlim(5.2, 26.1)
        ax.set_title(f"({chr(97 + index)}) {NAMES[model]}", loc="left", pad=4)
        ax.set_ylabel(YLAB[model], labelpad=3)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
        style(ax)
        RANGES[f"budget001/{model}"] = {"x": list(ax.get_xlim()), "y": list(ax.get_ylim()),
                                         "observed_coverages": [.0625, .125, .1875, .25]}
    shared_legend(fig)
    fig.supxlabel("Protected eligible bits (%)", y=.012, fontsize=7.6)
    fig.subplots_adjust(left=.093, right=.985, top=.805, bottom=.128,
                        wspace=.46, hspace=.55)
    save(fig, "compact_reliability_budget_001")


def paired_forest():
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.03))
    pythia = sorted([r for r in PAIRED if r["model"] == "pythia"
                    and float(r["ber"]) == .01 and r["compared_method"] == "global_first_order"
                    and r["reference_method"] == "msb_first"], key=lambda r: float(r["coverage"]))
    assert len(pythia) == 4
    for index, r in enumerate(pythia):
        mean, low, high = [float(r[k]) for k in ["mean_nll_difference_nats", "ci95_low", "ci95_high"]]
        axes[0].errorbar(mean, 3 - index, xerr=[[mean - low], [high - mean]],
                         fmt="o", color=COLORS["global"], ms=4, lw=1.05, capsize=2.3)
    axes[0].set_yticks(range(4), ["25%", "18.75%", "12.5%", "6.25%"])
    axes[0].set_ylabel("Protected bits", labelpad=3)
    axes[0].set_ylim(-.55, 3.55)
    axes[0].set_title("(a) Pythia · BER 1%\nGlobal gradient − MSB", loc="left", fontsize=7.8, pad=6)

    qwen = sorted([r for r in QWEN["conditions_and_contrasts"] if "contrast" in r], key=lambda r: r["ber"])
    assert len(qwen) == 2
    for index, r in enumerate(qwen):
        mean, low, high = [float(r[k]) for k in ["mean_difference_nats", "ci95_low", "ci95_high"]]
        axes[1].errorbar(mean, 1 - index, xerr=[[mean - low], [high - mean]],
                         fmt="o", color=COLORS["absolute"], ms=4, lw=1.05, capsize=2.3)
    axes[1].set_yticks([0, 1], ["7.5%", "5%"])
    axes[1].set_ylabel("Logical BER", labelpad=3)
    axes[1].set_ylim(-.6, 1.6)
    axes[1].set_title("(b) Qwen · 12.5% protection\nAbsolute gradient − MSB", loc="left", fontsize=7.8, pad=6)
    for index, ax in enumerate(axes):
        ax.axvline(0, color="#7B8490", ls="--", lw=.7, zorder=0)
        ax.set_xlim(-11.1, 3.2)
        ax.set_xticks([-10, -5, 0])
        ax.grid(axis="x", lw=.4, alpha=.65)
        ax.tick_params(length=2.3, width=.5, pad=3)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        RANGES[f"paired_forest/{index}"] = {"x": list(ax.get_xlim()), "y": list(ax.get_ylim())}
    fig.supxlabel("Paired capped-NLL difference (nats/token)", y=.015, fontsize=7.5)
    fig.subplots_adjust(left=.105, right=.985, top=.73, bottom=.23, wspace=.50)
    save(fig, "compact_paired_forest")


def beeswarm_offsets(x, radius_x=.00062, step_y=.045):
    """Deterministic dot dodging, not a density estimate or statistical weight."""
    offsets = np.zeros(len(x))
    placed = []
    for index in np.argsort(x, kind="stable"):
        for layer in range(len(x)):
            candidates = [0.] if layer == 0 else [layer * step_y, -layer * step_y]
            chosen = None
            for candidate in candidates:
                if all(abs(x[index] - x[other]) >= radius_x
                       or abs(candidate - offsets[other]) >= step_y * .98 for other in placed):
                    chosen = candidate
                    break
            if chosen is not None:
                break
        offsets[index] = chosen
        placed.append(index)
    return offsets


def kws_pools():
    primary = next(r for r in KWS["comparisons"] if r["role"] == "primary")
    effects = np.asarray(primary["pool_effects"], dtype=float)
    assert len(effects) == 32
    receipt = primary["pool_mean_effect"]
    mean = float(receipt["mean"])
    low, high = map(float, receipt["confidence_interval"])
    assert abs(np.mean(effects) - mean) < 1e-12
    fig, ax = plt.subplots(figsize=(6.2, 1.65))
    offset = beeswarm_offsets(effects)
    colors = np.where(effects < 0, COLORS["absolute"],
                      np.where(effects > 0, COLORS["positive"], "#A3A8B0"))
    ax.scatter(effects, .60 + offset, s=19, c=colors, alpha=.80,
               edgecolors="white", linewidths=.45, zorder=3)
    ax.errorbar(mean, 0, xerr=[[mean - low], [high - mean]], color="#26313D",
                fmt="D", ms=4.6, capsize=3.5, lw=1.15, zorder=4)
    ax.axvline(0, color="#7B8490", ls="--", lw=.75, zorder=1)
    ax.axhline(.29, color="#ECEFF2", lw=.7, zorder=0)
    ax.set_yticks([0, .60], ["Mean and 95% CI", "32 pool effects"])
    ax.set_xticks([-.01, -.005, 0, .005, .01], ["−0.010", "−0.005", "0", "+0.005", "+0.010"])
    ax.set_xlim(min(effects.min(), low) - .0011, max(effects.max(), high) + .0011)
    ax.set_ylim(-.22, .94)
    ax.set_xlabel("Conditional − exact-isolated loss (nats/clip)", labelpad=4)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=5)
    ax.tick_params(axis="x", length=2.5, width=.5, pad=3)
    fig.subplots_adjust(left=.19, right=.985, top=.94, bottom=.32)
    RANGES["kws_pool_distribution"] = {"x": list(ax.get_xlim()), "y": list(ax.get_ylim()),
                                       "pool_count": 32, "effect_min": float(effects.min()),
                                       "effect_max": float(effects.max()), "mean": mean,
                                       "source_ci95": [low, high]}
    save(fig, "compact_kws_pools")


if __name__ == "__main__":
    ber_grid()
    budget_grid()
    paired_forest()
    kws_pools()
    print(json.dumps({
        "input_files": {str(p.relative_to(ROOT)): sha(p) for p in
                        [DATA / "reliability_conditions.csv", DATA / "wide_paired_comparisons.csv",
                         DATA / "qwen_crossover.json", DATA / "kws_pool_statistics.json"]},
        "plot_ranges": RANGES,
        "notes": ["No source data, earlier figure, or manuscript section changed.",
                  "BER125 suppresses duplicate FKeras curve at complete significance tier; dagger applies to TFLite models only.",
                  "All means and original intervals are unchanged; no density estimate or new uncertainty calculation.",
                  "KWS dots are displaced vertically for visibility, never horizontally in effect units."]
    }, indent=2))
