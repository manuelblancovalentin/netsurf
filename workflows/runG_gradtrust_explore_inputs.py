#!/usr/bin/env python3
"""EXPLORATORY (after the gradient-trust predictions were frozen): two checks on the per-input gradients of the jet and
MNIST checkpoints, recomputed from the Phase 1 calibration draws (CPU, reference environment).

  outliers   per draw: the 5 inputs with the largest gradient norm, their share of the squared norm and their
             true-class probability; D1 recomputed without them.
  class mix  per draw: the per-class mean gradients; D1 of the class-balanced mean (equal class weights) against the
             raw mean; the size of the class means relative to the overall mean (how much they cancel).
Writes <out>/inputs_checks.json; --plot-only draws fig5_class_mix.png from it (needs matplotlib).

    runG_gradtrust_explore_inputs.py <out>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runR_partial_search import RUNS  # noqa: E402

IDS = ["jet-e1", "jet-e3", "jet-e10", "jet-e30", "jet-e100", "mnist-e1", "mnist-e3", "mnist-e10", "mnist-e30"]
COL = {"jet": "#2a78d6", "mnist": "#eb6834"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def main(out: Path):
    from runG_gradtrust import card_model, load_card, load_xy
    res = {}
    iu = np.triu_indices(6, 1)
    for cid in IDS:
        card = load_card(RUNS / f"paper1_gradtrust_inputs_v1/{cid}/card.json")
        m = card_model(card, 8)
        raw, bal, wo, ratio, draws = [], [], [], [], []
        for r in range(6):
            x, y = load_xy(card["calibration"][r])
            pe = m.comp.per_example_gradients(x, y, batch=256)
            G = np.concatenate([np.asarray(pe[n]).reshape(len(y), -1) for n in m.names], 1).astype(np.float64)
            norm = np.linalg.norm(G, axis=1)
            top = np.argsort(-norm)[:5]
            K = int(y.max()) + 1
            cm = np.stack([G[y == k].mean(0) for k in range(K)])
            raw.append(G.mean(0)); bal.append(cm.mean(0)); wo.append(np.delete(G, top, 0).mean(0))
            ratio.append(float(np.linalg.norm(cm, axis=1).mean() / max(np.linalg.norm(G.mean(0)), 1e-30)))
            draws.append(dict(class_counts=np.bincount(y, minlength=K).tolist(),
                              top5_share_of_squared_norm=float((norm[top] ** 2).sum() / (norm ** 2).sum())))
        c = lambda L: float(np.median(np.corrcoef(np.stack(L))[iu]))  # noqa: E731
        res[cid] = dict(D1=c(raw), D1_class_balanced=c(bal), D1_without_top5=c(wo),
                        class_mean_over_overall_mean=float(np.median(ratio)), draws=draws)
        print(json.dumps({cid: {k: round(v, 3) for k, v in res[cid].items() if k != "draws"}}), flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "inputs_checks.json").write_text(json.dumps(res, indent=1))


def plot(out: Path):
    """Figure from inputs_checks.json (any environment with matplotlib)."""
    res = json.loads((out / "inputs_checks.json").read_text())
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": GRID, "grid.linewidth": 0.6, "axes.edgecolor": INK2})
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
    xs = np.arange(len(IDS))
    axs[0].bar(xs - 0.2, [res[c]["D1"] for c in IDS], 0.38, color=[COL[c.split("-")[0]] for c in IDS], label="as drawn")
    axs[0].bar(xs + 0.2, [res[c]["D1_class_balanced"] for c in IDS], 0.38, color="white",
               edgecolor=[COL[c.split("-")[0]] for c in IDS], hatch="///", lw=1.2, label="equal class mix")
    axs[0].set_xticks(xs, IDS, rotation=50, fontsize=7.5)
    axs[0].set_ylabel("agreement D1")
    axs[0].set_title("Agreement with the class mix as drawn and equalised", loc="left", fontsize=9, color=INK)
    axs[0].legend(frameon=False, fontsize=8)
    axs[1].plot([int(c.split("-e")[1]) for c in IDS[:5]], [res[c]["class_mean_over_overall_mean"] for c in IDS[:5]], "o-",
                color=COL["jet"], label="jet")
    axs[1].plot([int(c.split("-e")[1]) for c in IDS[5:]], [res[c]["class_mean_over_overall_mean"] for c in IDS[5:]], "o-",
                color=COL["mnist"], label="MNIST")
    axs[1].set_xscale("log"); axs[1].set_xlabel("training epoch")
    axs[1].set_ylabel("class-mean gradient size / overall mean size")
    axs[1].set_title("How much the per-class gradients cancel (higher = more)", loc="left", fontsize=9, color=INK)
    axs[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig5_class_mix.png", dpi=160, bbox_inches="tight")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--plot-only", action="store_true", help="only draw the figure from inputs_checks.json")
    a = ap.parse_args()
    if not a.plot_only:
        main(a.out)
    plot(a.out)
