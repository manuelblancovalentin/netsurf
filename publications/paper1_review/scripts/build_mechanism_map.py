#!/usr/bin/env python3
"""Mechanism map: the highest usable BER of each protection condition on every TFLite checkpoint (Run T).

Reads the locked Run T analyses (paper1_runT_t1a_analysis_v1, paper1_runT_t1b_fkeras_analysis_v1,
paper1_runT_t1b_engine_analysis_v1, paper1_runT_t3_toycar_analysis_v1; 'tolerated' = the highest grid BER up to which
mean accuracy, or AUC, stays within 5 points of clean at that rate and every lower one) and writes
figures/tecs_mechanism_map.{pdf,png} and data/tecs_runT/tolerated.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_story_candidates as C  # noqa: E402

ANALYSES = ("paper1_runT_t1a_analysis_v1", "paper1_runT_t1b_fkeras_analysis_v1", "paper1_runT_t1b_engine_analysis_v1",
            "paper1_runT_t3_toycar_analysis_v1")
ORDER = [("jet-e1", "jet MLP e1"), ("jet-e3", "jet MLP e3"), ("jet-e10", "jet MLP e10"), ("jet-e30", "jet MLP e30"),
         ("jet-e100", "jet MLP e100"), ("mnist-e1", "MNIST MLP e1"), ("mnist-e3", "MNIST MLP e3"),
         ("mnist-e10", "MNIST MLP e10"), ("mnist-e30", "MNIST MLP e30"), ("dscnn", "DS-CNN"), ("vww", "VWW"),
         ("resnet8-pinned", "ResNet-8 pinned"), ("resnet8-retrained_seed7101", "ResNet-8 R1"),
         ("resnet8-retrained_seed7102", "ResNet-8 R2"), ("resnet8-retrained_seed7103", "ResNet-8 R3"),
         ("resnet8-retrained_seed7104", "ResNet-8 R4"), ("resnet8-retrained_seed7105", "ResNet-8 R5"),
         ("resnet8-pruned_s05", "ResNet-8 pruned 50"), ("resnet8-pruned_s07", "ResNet-8 pruned 70"),
         ("resnet8-undertrained_e100", "ResNet-8 under-trained"), ("toycar", "ToyCar (AUC)")]
ARMS = [("unprotected", "no protection", "#a8a7a2", "x"), ("abft_ideal", "ABFT (1-25%)", C.V2, "v"),
        ("tmr_msb@0.125", "TMR of sign bits: FORTUNE = FKeras", C.MSB, "s"),
        ("tmr_fkeras_weighted@0.125", "TMR, FKeras tier-mixing variant", "#6250d6", "D"),
        ("secded_32", "SEC-DED over 32 bits (21.9%)", C.POS, "^"), ("tmr_abs@0.125", "TMR, absolute gradient", C.ABS, "o")]
GRID = [1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]


def main():
    tol = {}
    for a in ANALYSES:
        s = json.loads((C.RUNS / a / "summary.json").read_text())
        for cid, v in s["checkpoints"].items():
            tol.setdefault(cid, {}).update(v["tolerated"])
    fig, ax = plt.subplots(figsize=(8.6, 7.4))
    ys = np.arange(len(ORDER))[::-1]
    for k, (arm, lab, col, mk) in enumerate(ARMS):
        off = (k - (len(ARMS) - 1) / 2) * 0.12
        xs, yy = [], []
        for y, (cid, _) in zip(ys, ORDER):
            v = tol.get(cid, {}).get(arm)
            xs.append(GRID.index(v) if v is not None else -0.6)
            yy.append(y + off)
        ax.scatter(xs, yy, s=34, marker=mk, color=col, label=lab, zorder=3, edgecolor="white", linewidth=0.5)
    for y in ys:
        ax.axhline(y, color=C.GRID, lw=0.5, zorder=1)
    for y in (len(ORDER) - 5 - 0.5, len(ORDER) - 9 - 0.5, len(ORDER) - 11 - 0.5, 0.5):
        ax.axhline(y, color=C.INK2, lw=0.8)
    ax.set_yticks(ys, [t for _, t in ORDER], fontsize=8)
    ax.set_xticks(range(len(GRID)), ["$10^{-5}$", "$10^{-4}$", "$3\\cdot10^{-4}$", "$10^{-3}$", "$3\\cdot10^{-3}$", "$10^{-2}$",
                                     "$3\\cdot10^{-2}$", "$10^{-1}$"], fontsize=8)
    ax.set_xlim(-0.8, len(GRID) - 0.5)
    ax.set_xlabel("highest bit-error probability at which the model stays within 5 points of clean")
    ax.grid(False)
    ax.legend(fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(C.ROOT / "figures" / f"tecs_mechanism_map.{ext}", bbox_inches="tight", **({"dpi": 200} if ext == "png" else {}))
    (C.ROOT / "data/tecs_runT").mkdir(parents=True, exist_ok=True)
    (C.ROOT / "data/tecs_runT/tolerated.json").write_text(json.dumps({c: tol[c] for c, _ in ORDER}, indent=1) + "\n")


if __name__ == "__main__":
    main()
