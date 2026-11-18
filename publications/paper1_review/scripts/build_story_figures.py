#!/usr/bin/env python3
"""Manuscript figures for the story revision (15 November 2026), as PDF and PNG in figures/.

  tecs_story_blind.pdf         = candidate A    tecs_story_reversal.pdf  = candidate B
  tecs_story_switch.pdf        = candidate C    tecs_story_no_fixed.pdf  = candidate D
  tecs_story_wrong_rate.pdf    = candidate E    tecs_story_room.pdf      = candidate F
  tecs_story_noise_events.pdf  the noise-plus-events model against measured TFLite outcomes (blur pilot, exploratory)

Candidates A-F are drawn by scripts/build_story_candidates.py (same code and inputs; the titles are left to the
captions).  The new figure reads the locked pilot analysis workspace/runs/paper1_blur_pilot_analysis_v1 (blur.csv,
switch.csv, step.csv; summary.json sha256 cf06073c...).

    build_story_figures.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.figure  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_story_candidates as C  # noqa: E402

FIG = C.ROOT / "figures"
PILOT = C.RUNS / "paper1_blur_pilot_analysis_v1"
PILOT_SHA = "cf06073c53e8b77caac25f9c5340ce71739bb5ed3ee15c75648d2322fc23770f"
NAMES = {"A_blind_vs_gradient": "tecs_story_blind", "B_reversal_resnet8_only": "tecs_story_reversal",
         "C_switch_collapse": "tecs_story_switch", "D_no_fixed_ranking_resnet8": "tecs_story_no_fixed",
         "E_built_for_vs_used_at": "tecs_story_wrong_rate", "F_room_vs_gain": "tecs_story_room"}


def patch():
    """Candidate code saves '<X>.png'; here every save becomes figures/<new name>.{pdf,png}, without the suptitle."""
    orig = matplotlib.figure.Figure.savefig

    def save(self, fname, *a, **k):
        stem = NAMES[Path(fname).stem]
        if self._suptitle is not None:
            self._suptitle.set_text("")
        for ext in ("pdf", "png"):
            orig(self, FIG / f"{stem}.{ext}", *a, **{**k, "dpi": 200} if ext == "png" else k)
    matplotlib.figure.Figure.savefig = save


def rows(p):
    with open(p) as fh:
        return list(csv.DictReader(fh))


def fig_noise_events():
    assert hashlib.sha256((PILOT / "summary.json").read_bytes()).hexdigest() == PILOT_SHA
    blur, sw, step = rows(PILOT / "blur.csv"), rows(PILOT / "switch.csv"), rows(PILOT / "step.csv")
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.8))
    lab = {"kws": "DS-CNN", "pinned": "ResNet-8"}
    ax = axs[0]
    for v, col, mk, name in (("mean", C.MSB, "x", "mean shift only"), ("gauss", C.POS, "o", "Gaussian noise only"),
                             ("hybrid", C.ABS, "o", "noise + top-bit events")):
        x = np.array([100 * float(r["measured"]) for r in blur]); y = np.array([100 * float(r[v]) for r in blur])
        err = np.abs(y - x).mean()
        ax.scatter(x, y, s=10 if mk == "o" else 14, marker=mk, color=col, alpha=0.6 if v != "hybrid" else 0.9, lw=0.8 if mk == "x" else 0,
                   label=f"{name} (mean error {err:.2f})", zorder=3 if v == "hybrid" else 2)
    lo = min(float(r["measured"]) for r in blur) * 100 - 2
    ax.plot([lo, 95], [lo, 95], color=C.INK2, lw=0.8, ls="--")
    ax.set_xlabel("measured TFLite accuracy (%)"); ax.set_ylabel("model accuracy (%)")
    C.title(ax, "a. Accuracy, 96 cells, two models"); ax.legend(fontsize=7.5, loc="upper left")
    ax = axs[1]
    for model, col in (("pinned", C.ABS), ("kws", C.MSB)):
        for b, ls in (("0.125", "-"), ("0.25", ":")):
            R = sorted((float(r["ber"]), np.mean([100 * float(q["measured"]) for q in sw if q["model"] == model and q["budget"] == b and q["ber"] == r["ber"]]),
                        np.mean([100 * float(q["hybrid"]) for q in sw if q["model"] == model and q["budget"] == b and q["ber"] == r["ber"]]))
                       for r in sw if r["model"] == model and r["budget"] == b and r["rep"] == sorted({q["rep"] for q in sw if q["model"] == model})[0])
            ax.plot([t[0] for t in R], [t[1] for t in R], ls, color=col, lw=2, label=f"{lab[model]} {float(b):.1%}, measured")
            ax.plot([t[0] for t in R], [t[2] for t in R], ls, color=col, lw=0, marker="o", ms=4, label=f"{lab[model]} {float(b):.1%}, model")
    ax.axhline(0, color=C.INK2, lw=0.8); ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel("bit-error probability p"); ax.set_ylabel("positive minus absolute (points)")
    C.title(ax, "b. The switch: present on ResNet-8, absent on DS-CNN"); ax.legend(fontsize=6.5, loc="lower left", ncol=1)
    ax = axs[2]
    for model, col in (("pinned", C.ABS), ("kws", C.MSB)):
        S = sorted((int(r["bit"]), float(r["rank_corr_real_quadratic"]), float(r["sign_agree_quadratic"])) for r in step if r["model"] == model)
        ax.plot([s[0] for s in S], [s[1] for s in S], "-o", color=col, ms=4, label=f"{lab[model]}: rank correlation")
        ax.plot([s[0] for s in S], [s[2] for s in S], "--s", color=col, ms=3.5, alpha=0.7, label=f"{lab[model]}: sign agreement")
    ax.axhline(0.5, color=C.INK2, lw=0.6, ls=":"); ax.axhline(0, color=C.INK2, lw=0.8)
    ax.set_xlabel("bit position (7 = sign)"); ax.set_ylabel("agreement with the gradient prediction")
    C.title(ax, "c. Single flips: low bits act as noise"); ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"tecs_story_noise_events.{ext}", **({"dpi": 200} if ext == "png" else {}))


def main():
    fig_noise_events()
    patch()
    d = Path("."); C.fig_blind(d); C.fig_reversal(d); C.fig_collapse(d); C.fig_room(d)
    acc, cross, clean = C.stage2()
    C.fig_no_fixed(d, acc, clean); C.fig_wrong_rate(d, acc, cross, clean)
    print("\n".join(sorted(f"{hashlib.sha256(p.read_bytes()).hexdigest()[:16]}  {p.name}" for p in FIG.glob("tecs_story_*"))))


if __name__ == "__main__":
    main()
