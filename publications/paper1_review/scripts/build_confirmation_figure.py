#!/usr/bin/env python3
"""Figure and summary of the confirmation of the noise-plus-events model (review/BLUR_CONFIRMATION_PLAN_2026-11-15.md).

Reads the locked analysis workspace/runs/paper1_blurconf_analysis_v1 (summary.json, cells.csv, pairs.csv; hashes in
paper1_blurconf_logs/SHA256SUMS_analysis) and writes figures/tecs_noise_confirmation.{pdf,png} and
data/tecs_noise_confirmation/summary.json (byte copies of the inputs plus the numbers cited in the text).
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_story_candidates as C  # noqa: E402

SRC = C.RUNS / "paper1_blurconf_analysis_v1"
SUMS = C.RUNS / "paper1_blurconf_logs/SHA256SUMS_analysis"
DATA = C.ROOT / "data/tecs_noise_confirmation"
FAM = {"resnet8": ("ResNet-8 (8 checkpoints)", C.ABS), "jet": ("jet MLP (5)", C.POS), "mnist": ("MNIST MLP (4)", "#e87ba4"),
       "vww": ("VWW", C.V2)}
R8 = ("undertrained_e100", "pruned_s07", "pruned_s05", "retrained_seed7101", "retrained_seed7102", "retrained_seed7103",
      "retrained_seed7104", "retrained_seed7105")
LABEL = dict(undertrained_e100="R8 under-tr.", pruned_s07="R8 pruned 70", pruned_s05="R8 pruned 50",
             **{f"retrained_seed710{i}": f"R8 R{i}" for i in range(1, 6)})


def fam(c):
    return "resnet8" if c in R8 else c.split("-")[0]


def main():
    sums = dict(line.split()[::-1] for line in SUMS.read_text().splitlines())
    DATA.mkdir(parents=True, exist_ok=True)
    for f in ("summary.json", "cells.csv", "pairs.csv"):
        assert hashlib.sha256((SRC / f).read_bytes()).hexdigest() == sums[f], f
        shutil.copyfile(SRC / f, DATA / f)
    s = json.loads((DATA / "summary.json").read_text())
    cells = list(csv.DictReader(open(DATA / "cells.csv")))
    pairs = list(csv.DictReader(open(DATA / "pairs.csv")))
    order = list(R8) + [c for c in s["checkpoints"] if c not in R8]
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 4.0), gridspec_kw=dict(width_ratios=[1, 1.35, 1]))
    ax = axs[0]
    U = [c for c in cells if c["usable"] == "True"]
    for f, (lab, col) in FAM.items():
        P = [c for c in U if fam(c["checkpoint"]) == f]
        ax.scatter([float(c["measured"]) for c in P], [float(c["hybrid"]) for c in P], s=7, color=col, alpha=0.6, lw=0, label=lab)
    lo = min(float(c["measured"]) for c in U) - 1
    ax.plot([lo, 100], [lo, 100], color=C.INK2, lw=0.8, ls="--")
    ax.set_xlabel("measured TFLite accuracy (%)"); ax.set_ylabel("noise + events model (%)")
    C.title(ax, f"a. Accuracy, {len(U)} usable cells"); ax.legend(fontsize=7.5, loc="upper left", markerscale=2)
    ax = axs[1]
    x = np.arange(len(order))
    for k, (v, col, lab) in enumerate((("mean", C.MSB, "mean shift only"), ("gauss", C.POS, "Gaussian noise only"),
                                       ("hybrid", C.ABS, "noise + top-bit events"))):
        ax.bar(x + (k - 1) * 0.27, [s["checkpoints"][c]["errors_usable"][v]["mean_abs"] for c in order], 0.27, color=col, label=lab)
    ax.axhline(1.0, color=C.INK2, lw=0.8, ls="--")
    ax.text(-0.5, 1.06, "limit fixed in advance", ha="left", va="bottom", fontsize=7, color=C.INK2)
    ax.set_yscale("log"); ax.set_ylabel("mean absolute error (points)")
    ax.set_xticks(x, [LABEL.get(c, c) for c in order], rotation=60, fontsize=7, ha="right")
    C.title(ax, "b. Error per checkpoint (usable cells)"); ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
    ax = axs[2]
    P = [q for q in pairs if q["pair"] == "pos1024-abs1024"]
    for f, (lab, col) in FAM.items():
        Q = [q for q in P if fam(q["checkpoint"]) == f]
        ax.scatter([float(q["measured"]) for q in Q], [float(q["hybrid"]) for q in Q], s=8, color=col, alpha=0.6, lw=0, label=lab)
    ax.axhline(0, color=C.INK2, lw=0.6); ax.axvline(0, color=C.INK2, lw=0.6)
    ax.set_xscale("symlog", linthresh=0.5); ax.set_yscale("symlog", linthresh=0.5)
    for setter in (ax.set_xticks, ax.set_yticks):
        setter([-10, -1, 0, 1], ["-10", "-1", "0", "1"])
    ax.set_xlabel("measured, positive minus absolute (points)"); ax.set_ylabel("model, positive minus absolute (points)")
    C.title(ax, "c. Which mask is better, cell by cell")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(C.ROOT / "figures" / f"tecs_noise_confirmation.{ext}", **({"dpi": 200} if ext == "png" else {}))
    err = {v: [s["checkpoints"][c]["errors_usable"][v]["mean_abs"] for c in order] for v in ("mean", "gauss", "hybrid", "real")}
    out = dict(verdict=s["verdict"], usable_cells=len(U),
               hybrid_error_range=[round(min(err["hybrid"]), 2), round(max(err["hybrid"]), 2)],
               gauss_error_range=[round(min(err["gauss"]), 2), round(max(err["gauss"]), 2)],
               mean_error_range=[round(min(err["mean"]), 2), round(max(err["mean"]), 2)],
               replay_error_range=[round(min(err["real"]), 2), round(max(err["real"]), 2)],
               hybrid_better_than_gauss=sum(h < g for h, g in zip(err["hybrid"], err["gauss"])),
               hybrid_worst_cell=round(max(abs(s["checkpoints"][c]["errors_usable"]["hybrid"]["worst"]) for c in order), 2),
               regimes={c: s["checkpoints"][c]["regime_measured"] for c in order})
    (DATA / "numbers.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
