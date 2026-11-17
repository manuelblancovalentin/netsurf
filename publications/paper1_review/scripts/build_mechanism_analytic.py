#!/usr/bin/env python3
"""Analytical comparison of protection mechanisms: residual error, extra storage and network size (no measured data).

Formulas: netsurf.protection_analytic (branch paper1-runR-v2-validated, commit ac23bd9 or later), checked against the
bit-level simulators used in the main comparison (tests/test_protection_analytic.py).  Writes
figures/fig_mechanism_analytic.{pdf,png} and data/tecs_mechanism_analytic.json.

  (a) residual error of one protected bit against the bit-error probability p
  (b) residual error against extra storage, at two rates: what each extra bit of storage buys
  (c) expected residual faulty bits in a fully protected network, for a small and a large network
  (d) the least extra storage that keeps fewer than one expected residual fault, against p and network size
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[3] / "paper1-runR-v2-validated"))
import paper_style as S  # noqa: E402
from netsurf import protection_analytic as A  # noqa: E402

S.apply()
ROOT = HERE.parents[0]
P = np.logspace(-5, -1, 200)
MR = {1: "#2171b5", 2: "#08519c", 3: "#08306b", 4: "#041f40"}          # darker blues for 3-, 5-, 7-, 9-MR (ABFT keeps sky)
CODE = {1: "#fdae6b", 2: "#e6550d", 3: "#a63603"}                      # SEC-DED, DEC-TED, 3EC over 64 bits
ABFT = S.ABFT
NETS = {"Jet MLP (34 thousand bits)": 34_048, "ResNet-8 (0.6 million bits)": 618_880, "Qwen3-0.6B (4.8 billion bits)": 4_767_875_072}
OUT = {}


def label_mr(n):
    return {1: "TMR", 2: "5-MR", 3: "7-MR", 4: "9-MR"}[n]


def label_code(t):
    return {1: "SEC-DED", 2: "DEC-TED", 3: "3-error-correcting"}[t]


def panel_a(ax):
    ax.plot(P, P, color=S.NONE, lw=1.6, ls=":", label="no protection")
    for n in (1, 2, 3):
        ax.plot(P, [A.mr_residual(p, n) for p in P], color=MR[n], lw=2, label=f"{label_mr(n)} ({A.mr_cost(n):.0%})")
    for t in (1, 2):
        ax.plot(P, [A.code_residual(p, 64, t) for p in P], color=CODE[t], lw=2, ls="--",
                label=f"{label_code(t)}, 64 bits ({A.code_cost(64, t):.1%})")
    for m, ls in ((64, "-."), (1024, ":")):
        ax.plot(P, [A.abft_residual(p, m) for p in P], color=ABFT, lw=2, ls=ls, label=f"ABFT, {m} weights per channel ({A.abft_cost(m):.1%})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(1e-20, 0.2)
    ax.set_xlabel("bit-error probability p")
    ax.set_ylabel("residual error of one protected bit")
    ax.set_title("(a) error left after the mechanism acts", loc="left")
    ax.legend(fontsize=6.6, loc="lower right")


def panel_b(ax):
    for p, ls, mk in ((1e-3, "-", "o"), (1e-2, "--", "s")):
        xs = [A.mr_cost(n) for n in (1, 2, 3, 4)]
        ys = [A.mr_residual(p, n) for n in (1, 2, 3, 4)]
        ax.plot(xs, ys, ls, color=MR[3], marker=mk, ms=4.5, lw=1.6)
        for n, x, y in zip((1, 2, 3, 4), xs, ys):
            if p == 1e-3:
                ax.annotate(label_mr(n), (x, y), xytext=(4, 2), textcoords="offset points", fontsize=6.3, color=MR[3])
        xs = [A.code_cost(64, t) for t in (1, 2, 3)]
        ys = [A.code_residual(p, 64, t) for t in (1, 2, 3)]
        ax.plot(xs, ys, ls, color=CODE[2], marker=mk, ms=4.5, lw=1.6)
        if p == 1e-3:
            for t, x, y in zip((1, 2, 3), xs, ys):
                ax.annotate(label_code(t).split(",")[0].replace("3-error-correcting", "3EC"), (x, y), xytext=(4, 2),
                            textcoords="offset points", fontsize=6.3, color=CODE[2])
        xs = [A.code_cost(k) for k in (64, 32, 16, 8)]
        ys = [A.code_residual(p, k) for k in (64, 32, 16, 8)]
        ax.plot(xs, ys, ls, color=CODE[1], marker=mk, ms=4, lw=1.2, alpha=0.9)
        ms_ = (4096, 1024, 256, 64, 16)
        xs = [A.abft_cost(m) for m in ms_]
        ys = [A.abft_residual(p, m) for m in ms_]
        ax.plot(xs, ys, ls, color=ABFT, marker=mk, ms=4, lw=1.4)
        ax.axhline(p, color=S.NONE, lw=1, ls=ls, alpha=0.8)
        ax.text(1.1e-3, p * 1.4, f"no protection, p = {p:g}", fontsize=6.3, color=S.INK2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1e-3, 12); ax.set_ylim(1e-14, 0.05)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.1%}" if v < 0.01 else f"{v:.0%}"))
    ax.set_xlabel("extra storage per protected bit")
    ax.set_ylabel("residual error of one protected bit")
    ax.set_title("(b) what each extra bit buys", loc="left")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=MR[3], lw=1.6, marker="o", ms=4), Line2D([], [], color=CODE[2], lw=1.6, marker="o", ms=4),
         Line2D([], [], color=CODE[1], lw=1.2, marker="o", ms=4), Line2D([], [], color=ABFT, lw=1.4, marker="o", ms=4),
         Line2D([], [], color=S.INK2, lw=1.4, ls="-"), Line2D([], [], color=S.INK2, lw=1.4, ls="--")]
    ax.legend(h, ["(2n+1)-MR, n = 1 to 4", "codes over 64 bits, t = 1 to 3", "SEC-DED over 64 to 8 bits",
                  "ABFT, 4096 to 16 weights per channel", "p = 0.001", "p = 0.01"], fontsize=6.4, loc="lower left")


def panel_c(ax):
    for (name, bits), lw in zip(list(NETS.items())[1:], (1.4, 2.4)):
        for res, col, ls, lab in ((lambda p: p, S.NONE, ":", "no protection"), (lambda p: A.mr_residual(p, 1), MR[1], "-", "TMR"),
                                  (lambda p: A.mr_residual(p, 2), MR[2], "-", "5-MR"),
                                  (lambda p: A.code_residual(p, 64, 1), CODE[1], "--", "SEC-DED, 64 bits"),
                                  (lambda p: A.code_residual(p, 64, 2), CODE[2], "--", "DEC-TED, 64 bits")):
            ax.plot(P, [res(p) * bits for p in P], ls, color=col, lw=lw, alpha=0.95 if lw > 2 else 0.6,
                    label=f"{lab}" if lw > 2 else None)
    ax.axhline(1, color=S.INK, lw=0.8)
    ax.text(1.2e-5, 2.2, "one residual fault per network", fontsize=6.5, color=S.INK)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(1e-10, 1e9)
    ax.set_xlabel("bit-error probability p")
    ax.set_ylabel("expected residual faulty bits\n(every bit protected)")
    ax.set_title("(c) small (thin) vs large (thick) network", loc="left")
    ax.legend(fontsize=6.6, loc="lower right")


def panel_d(ax):
    rates = np.logspace(-5, -1, 41)
    styles = {"Jet MLP (34 thousand bits)": ":", "ResNet-8 (0.6 million bits)": "--", "Qwen3-0.6B (4.8 billion bits)": "-"}
    OUT["needed"] = {}
    for name, bits in NETS.items():
        for fam, col in (("mr", MR[3]), ("code", CODE[2])):
            ys = [A.needed(p, bits, fam)[1] for p in rates]
            ys = [np.nan if y is None else y for y in ys]
            ax.step(rates, ys, where="post", color=col, ls=styles[name], lw=1.8)
            OUT["needed"][f"{fam}/{name}"] = dict(rates=rates.tolist(), extra_storage=ys)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("bit-error probability p")
    ax.set_ylabel("extra storage needed for\nfewer than one residual fault")
    ax.set_title("(d) redundancy needed", loc="left")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=MR[3], lw=1.8), Line2D([], [], color=CODE[2], lw=1.8)] + \
        [Line2D([], [], color=S.INK2, lw=1.6, ls=styles[n]) for n in NETS]
    ax.legend(h, ["(2n+1)-MR", "t-error-correcting code, 64 bits"] + list(NETS), fontsize=6.4, loc="upper left")


def main():
    fig, axs = plt.subplots(2, 2, figsize=(7.6, 6.9))
    panel_a(axs[0, 0]); panel_b(axs[0, 1]); panel_c(axs[1, 0]); panel_d(axs[1, 1])
    fig.tight_layout(h_pad=1.6, w_pad=1.4)
    for ext in ("pdf", "png"):
        fig.savefig(ROOT / "figures" / f"fig_mechanism_analytic.{ext}", bbox_inches="tight", **({"dpi": 220} if ext == "png" else {}))
    OUT["costs"] = dict(mr={label_mr(n): A.mr_cost(n) for n in (1, 2, 3, 4)},
                        code64={label_code(t): A.code_cost(64, t) for t in (1, 2, 3)},
                        check_bits64={t: A.bch_check_bits(64, t) for t in (1, 2, 3)})
    (ROOT / "data/tecs_mechanism_analytic.json").write_text(json.dumps(OUT, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
