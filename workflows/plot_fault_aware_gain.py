#!/usr/bin/env python3
"""Discussion figure (exploratory, not a manuscript figure): accuracy of the fixed absolute-gradient mask and of
the mask rebuilt for the fault rate on DistilBERT, with the paired gains, in the layout of the old Figure 7.

Inputs: the analysis files of Runs F, F2 and F3.  Output: workspace/figures_for_discussion/.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "workspace/runs"
F = json.loads((RUNS / "paper1_runF_fault_aware_masks_v1_analysis.json").read_text())["all_rows"]
F2 = json.loads((RUNS / "paper1_runF2_fault_aware_confirm_v1_analysis.json").read_text())
F3 = json.loads((RUNS / "paper1_runF3_fault_aware_tolerance_v1_analysis.json").read_text())["rows"]
CLEAN = 0.9140625
RED, GREY, TEAL = "#e4572e", "#9a9a9a", "#17a398"


def series(frac):
    """fault rate -> (fixed accuracy, rebuilt accuracy, gain, gain low, gain high); masks built for the rate shown."""
    pts = {}
    for r in F:
        if r["candidate"] == "c1_conditional_absolute" and r["fraction"] == frac and r["on_target"]:
            pts[r["test_ber"]] = (r["absolute_gradient_accuracy"], r["candidate_accuracy"], r["accuracy_gain_over_absolute"], r["gain_ci_low"], r["gain_ci_high"])
    for r in F3:
        if r["fraction"] == frac:
            pts[r["ber"]] = (r["fixed_absolute"], r["candidate"], r["gain"], r["gain_ci"][0], r["gain_ci"][1])
    for r in F2["extension"]:
        if r["fraction"] == frac and r["on_target"]:
            pts[r["target_ber"]] = (r["fixed_absolute"], r["candidate"], *r["gain"])
    return dict(sorted(pts.items()))


def style(ax, xs, clean, title, rotate=0):
    ax.axhspan(100 * (clean - 0.05), 100 * clean, color="#eef3ea", lw=0, zorder=0)
    ax.axhline(100 * clean, color="#777777", lw=0.8, ls=":")
    ax.text(xs[0], 100 * clean + 0.4, f"no faults: {100 * clean:.1f}%", fontsize=7.5, color="#555555", va="bottom")
    ax.set_xscale("log"); ax.set_xticks(xs); ax.set_xticklabels([f"{x:g}" for x in xs], fontsize=7.5, rotation=rotate); ax.minorticks_off()
    ax.set_xlabel("Bit-error probability"); ax.set_ylabel("Accuracy (%)"); ax.set_title(title, loc="left", fontsize=9.5, fontweight="bold")
    ax.grid(True, color="#e6e6e6", lw=0.5); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.edgecolor": "#555555"})
fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9))
for ax, frac, title in ((axes[0], 0.125, "DistilBERT, 12.5% of bits protected"), (axes[1], 0.1875, "DistilBERT, 18.75% of bits protected")):
    s = series(frac); xs = list(s); fixed = [100 * s[x][0] for x in xs]; reb = [100 * s[x][1] for x in xs]
    style(ax, xs, CLEAN, title, rotate=40 if len(xs) > 5 else 0)
    ax.plot(xs, fixed, color=GREY, ls="--", lw=1.8, marker="o", ms=4, label="Fixed absolute-gradient mask (built once on the fault-free model)")
    ax.plot(xs, reb, color=RED, lw=2.4, marker="o", ms=4.5, label="Mask rebuilt for the fault rate")
    best = max(xs, key=lambda x: s[x][2])
    for x in xs:
        if s[x][3] > 0:                                     # annotate gains whose 95% interval excludes zero
            col = TEAL if x == best else RED
            ax.annotate(f"+{100 * s[x][2]:.1f} pp", (x, 100 * s[x][1]), textcoords="offset points", xytext=(4, 12 if x == best else 9),
                        fontsize=9.5 if x == best else 8.5, color=col, fontweight="bold")
            ax.plot([x, x], [100 * s[x][0], 100 * s[x][1]], color=col, lw=1.0, alpha=0.7)
    ax.set_ylim(min(fixed) - 4, 94)
ax = axes[2]
c = F2["confirmation"]; xs = [r["test_ber"] for r in c]
style(ax, xs, c[0]["clean_accuracy"], "Confirmation: 328 unused sentences, new faults, 12.5%")
ax.plot(xs, [100 * r["unprotected"] for r in c], color="#c9c9c9", ls=":", lw=1.6, marker="x", ms=5, label="No protection")
ax.plot(xs, [100 * r["fixed_absolute"] for r in c], color=GREY, ls="--", lw=1.8, marker="o", ms=4, label="Fixed mask")
ax.plot(xs, [100 * r["candidate"] for r in c], color=RED, lw=2.4, marker="o", ms=4.5, label="Mask rebuilt for 0.1")
r = c[-1]
ax.annotate(f"+{100 * r['gain'][0]:.1f} pp", (r["test_ber"], 100 * r["candidate"]), textcoords="offset points", xytext=(-62, -20),
            fontsize=9.5, color=TEAL, fontweight="bold")
ax.plot([r["test_ber"]] * 2, [100 * r["fixed_absolute"], 100 * r["candidate"]], color=TEAL, lw=1.0)
ax.set_ylim(40, 94); ax.legend(loc="lower left", frameon=False, fontsize=7.8)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=8.5, bbox_to_anchor=(0.36, -0.03))
fig.tight_layout(rect=(0, 0.06, 1, 1))
out = ROOT / "workspace/figures_for_discussion"; out.mkdir(parents=True, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(out / f"fault_aware_gain_distilbert.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
print("saved", out / "fault_aware_gain_distilbert.png")
