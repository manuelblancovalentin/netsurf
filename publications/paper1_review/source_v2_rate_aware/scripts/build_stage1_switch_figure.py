"""Interim plot of Stage 1 (finished jobs only; descriptive; not the frozen analysis)."""
import json, math, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/projects/netsurf/manuelbv/workspace/implementation-worktrees/paper1-runR-rate-aware/workflows")
import runR_stage1_analyze as A
from runR_stage1 import FRACTIONS, RATES, NAMES, bkey

OUT = os.environ.get("OUT_PNG") or "/tmp/claude-1000/-projects-netsurf-manuelbv/eee4f243-fbf3-41d9-8cda-c13310ef18a2/scratchpad/stage1_final_switch.png"
pred = json.loads(A.PRED.read_text())["predictions"]
acc, clean, missing = A.load()
done = 576 - len(missing)

STYLE = {  # colour + marker per checkpoint (marker = secondary encoding)
    "pinned": ("#2a78d6", "o", "Pinned"),
    "retrained_seed7101": ("#7c5cd6", "D", "Retrained, seeds 7101-7105"),
    "retrained_seed7102": ("#7c5cd6", "D", "Retrained, seeds 7101-7105"),
    "retrained_seed7103": ("#7c5cd6", "D", "Retrained, seeds 7101-7105"),
    "retrained_seed7104": ("#7c5cd6", "D", "Retrained, seeds 7101-7105"),
    "retrained_seed7105": ("#7c5cd6", "D", "Retrained, seeds 7101-7105"),
    "undertrained_e100": ("#eb6834", "s", "Under-trained"),
    "pruned_s05": ("#1baf7a", "^", "Pruned 50%"),
    "pruned_s07": ("#0e7c86", "v", "Pruned 70%"),
}
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

cells, switches = [], []
for name in NAMES:
    if not clean[name]:
        continue
    c0 = float(np.mean(list(clean[name].values())))
    nrep = len(clean[name])
    for f in FRACTIONS:
        ps = pred[f"{name}/{bkey(f)}"]
        rows = []
        for p in RATES:
            P = acc[(name, round(f, 6), round(p, 8), "pos1024")]; B = acc[(name, round(f, 6), round(p, 8), "abs1024")]
            keys = sorted(set(P) & set(B))
            d = np.array([P[k] - B[k] for k in keys]) * 100
            usable = max(np.mean(list(P.values())), np.mean(list(B.values()))) >= c0 - A.MARGIN
            r = dict(name=name, f=f, p=p, diff=float(d.mean()), usable=bool(usable), pstar=ps["p_star"], how=ps["how"])
            cells.append(r)
            if usable:
                rows.append(r)
        last = None
        for a, b in zip(rows, rows[1:]):
            if (a["diff"] > 0) != (b["diff"] > 0):
                last = (a, b)
        if last and ps["how"] == "interpolated":
            a, b = last
            x0, x1 = math.log(a["p"]), math.log(b["p"])
            switches.append((name, f, ps["p_star"], math.exp(x0 + (x1 - x0) * (-a["diff"]) / (b["diff"] - a["diff"]))))

plt.rcParams.update({"font.size": 10 if os.environ.get("PAPER") else 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8), facecolor=SURF, gridspec_kw=dict(width_ratios=(1, 1.35)))

# Left: measured vs predicted switch
lim = (1.2e-3, 3.2e-2)
xs = np.array(lim)
ax1.fill_between(xs, xs / 1.5, xs * 1.5, color=MUTED, alpha=0.12, lw=0, label="Within a factor 1.5")
ax1.plot(xs, xs, color=MUTED, lw=0.9, ls="--", label="Measured = predicted")
sizes = {f: 18 + 70 * i / (len(FRACTIONS) - 1) for i, f in enumerate(FRACTIONS)}
seen = set()
for name, (col, mk, lab) in STYLE.items():
    pts = [(pp, m, f) for n, f, pp, m in switches if n == name]
    if not pts:
        continue
    nrep = sum(len(clean[n]) for n in STYLE if STYLE[n][2] == lab)
    label = None if lab in seen else f"{lab} ({nrep} reps)"
    seen.add(lab)
    ax1.scatter([a for a, _, _ in pts], [b for _, b, _ in pts], s=[sizes[f] for *_, f in pts], marker=mk,
                c=col, edgecolors=SURF, linewidths=1.0, alpha=0.9, zorder=3,
                label=label)
ax1.set_xscale("log"); ax1.set_yscale("log"); ax1.set_xlim(lim); ax1.set_ylim(lim); ax1.set_aspect("equal")
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
T = [2e-3, 3e-3, 5e-3, 1e-2, 2e-2, 3e-2]
for axis in (ax1.xaxis, ax1.yaxis):
    axis.set_major_locator(FixedLocator(T)); axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())
ax1.set_xlabel("Predicted switch rate $p^*$ (frozen)"); ax1.set_ylabel("Measured switch rate")
ax1.set_title("Measured vs predicted switch\none point per checkpoint and budget (size = budget)", loc="left", fontsize=9.5, color=INK)
ax1.grid(True, which="major", color=GRID, lw=0.6); ax1.set_facecolor(SURF)
ax1.legend(fontsize=7.2, frameon=False, loc="upper left")

# Right: every usable cell, rate normalised by its own prediction
ax2.axvline(1, color=MUTED, lw=0.9, ls="--")
ax2.axvspan(1 / 1.25, 1.25, color=MUTED, alpha=0.12, lw=0)
ax2.text(1.3, 1.6, "grey: within 1.25x of $p^*$,\nexcluded from the\nsign rules", color=MUTED, fontsize=7)
ax2.axhline(0, color=MUTED, lw=0.8)
for name, (col, mk, lab) in STYLE.items():
    for f in FRACTIONS:
        rs = sorted([c for c in cells if c["name"] == name and c["f"] == f and c["usable"]], key=lambda c: c["p"])
        if len(rs) < 2:
            continue
        x = [c["p"] / c["pstar"] for c in rs]; y = [c["diff"] for c in rs]
        ax2.plot(x, y, color=col, lw=0.9, alpha=0.45)
        ax2.scatter(x, y, marker=mk, s=16, c=col, edgecolors=SURF, linewidths=0.6, alpha=0.85, zorder=3)
ax2.set_xscale("log"); ax2.set_yscale("symlog", linthresh=1)
ax2.set_xlabel("Bit-error probability / predicted switch $p^*$")
ax2.set_ylabel("Accuracy, positive − absolute (points)")
ax2.set_title("All usable cells (within 5 points of clean)\nleft of 1: positive predicted ahead; right of 1: absolute predicted ahead",
              loc="left", fontsize=9.5, color=INK)
ax2.text(0.05, 3.2, "positive ahead", color=MUTED, fontsize=8)
ax2.text(0.05, -3.8, "absolute ahead", color=MUTED, fontsize=8)
ax2.grid(True, axis="y", color=GRID, lw=0.6); ax2.set_facecolor(SURF)

None if os.environ.get("PAPER") else fig.suptitle(f"Run R Stage 1, all {done} jobs. Frozen analysis: R1a, R1b, R1c hold",
             x=0.01, ha="left", fontsize=10, color="#0b0b0b")
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(OUT, dpi=170, facecolor=SURF)
print(OUT, done, len(switches))
