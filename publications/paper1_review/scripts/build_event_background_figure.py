#!/usr/bin/env python3
"""Measured version of the events-versus-background sketch (Fig. 2 d-f): one layer of the reference ResNet-8.

Plan: review/EVENT_BACKGROUND_CHECK_PLAN_2026-11-17.md.  Reads the analysis (statements.json, summary.csv) and the
histograms of the layer with the largest fan-in; nothing is chosen here: the three rates shown are the lowest, the
highest and the one whose mean number of events per neuron is closest to one, without protection.

  figures/fig_event_background.{pdf,png}, data/tecs_event_background.json
  (a-c) distribution of the change of one neuron's input, split into background (bits 0-5) and events (bits 6-7),
        in units of the standard deviation of their sum
  (d)   mean number of top-bit flips in one neuron's fan-in against the fault rate
  (e)   share of the change's second moment that comes from the events
  (f)   largest gap between each part's distribution and a normal distribution
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
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paper_style as S  # noqa: E402

S.apply()
ROOT = HERE.parent
RUNS = HERE.parents[2] / "workspace" / "runs"
AN, RUN = RUNS / "paper1_event_background_analysis_v1", RUNS / "paper1_event_background_v1"
EXACT = RUNS / "paper1_event_background_exact_v1" / "exact.json"
NOISE, NOISE_D, EVENT = S.OI["sky"], "#2b7fb8", S.OI["orange"]
MASKS = (("unprotected", "no protection", "-"), ("abs@0.125", "12.5% protected", "--"))


def rate(p):
    m, e = f"{p:.0e}".split("e")
    return f"$10^{{{int(e)}}}$" if m == "1" else f"${m}\\times10^{{{int(e)}}}$"


def main():
    st = json.loads((AN / "statements.json").read_text())
    L = st["figure_layer"]
    rows = [r for r in csv.DictReader(open(AN / "summary.csv")) if int(float(r["layer"])) == L]
    by = {m: sorted([r for r in rows if r["mask"] == m], key=lambda r: float(r["ber"])) for m, _, _ in MASKS}
    un = by["unprotected"]
    rates = [float(r["ber"]) for r in un]
    ev = [float(r["events_mean"]) for r in un]
    shown = [0, int(np.argmin(np.abs(np.array(ev) - 1.0))), len(un) - 1]
    # The run recorded flip counts up to 60 per neuron.  The expected count is exactly (unprotected bits among one
    # neuron's weights) x p, with the number of bits counted from the mask (event_background_exact.py); the sampled mean
    # is shown only where the expected count is below 40.
    EX = {r["mask"]: r for r in json.loads(EXACT.read_text()) if r["layer"] == L}
    BITS = dict(events_mean="unprotected_top_bits_per_neuron", lowflips_mean="unprotected_low_bits_per_neuron")

    def expected(rows_, key):
        return [EX[rows_[0]["mask"]][BITS[key]] * q for q in rates]
    Hf = RUN / f"layer{L}" / "histograms.npz"
    H = np.load(Hf)
    edges = H["edges"]
    mid, width = 0.5 * (edges[1:] + edges[:-1]), edges[1] - edges[0]

    fig, axs = plt.subplots(2, 3, figsize=(7.4, 4.9))
    for ax, i, tag in zip(axs[0], shown, "abc"):
        r = un[i]
        for part, col, lab in (("low_on_sum", NOISE_D, "background (bits 0-5)"), ("top_on_sum", EVENT, "events (bits 6-7)")):
            h = H[f"L{L}|unprotected|{i}|{part}"]
            d = h / h.sum() / width
            ax.step(mid, np.where(d > 0, d, np.nan), where="mid", color=col, lw=1.3, label=lab)
            if part == "low_on_sum":
                ax.fill_between(mid, np.where(d > 0, d, 1e-9), 1e-9, step="mid", color=NOISE, alpha=0.35, lw=0)
        ax.set_yscale("log"); ax.set_ylim(1e-6, 4e1); ax.set_xlim(-10, 10)
        ax.set_xlabel("change of one neuron's input\n(standard deviations of the total)", fontsize=7)
        ax.set_title(f"({tag}) $p$ = {rate(float(r['ber']))}", loc="left", fontsize=8.5)
        ax.text(0.97, 0.96, f"events per neuron: {(lambda v: f'{v:.0f}' if v >= 10 else f'{v:.2g}')(expected(un, 'events_mean')[i])}\nnone in {100 * float(r['events_p0']):.3g}% of cases",
                transform=ax.transAxes, ha="right", va="top", fontsize=6.3, color=S.INK,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
        if tag == "a":
            ax.set_ylabel("density (log)", fontsize=7)
            hl = ax.get_legend_handles_labels()
    ax = axs[1, 0]
    for m, lab, ls in MASKS:
        y, ex = [float(r["events_mean"]) for r in by[m]], expected(by[m], "events_mean")
        ax.plot(rates, ex, ls, color=EVENT, lw=1.6, label=f"top bits, {lab}")
        ax.plot([q for q, e_ in zip(rates, ex) if e_ < 40], [v for v, e_ in zip(y, ex) if e_ < 40], "o", color=EVENT, ms=3.5)
    y, ex = [float(r["lowflips_mean"]) for r in un], expected(un, "lowflips_mean")
    ax.plot(rates, ex, "-", color=NOISE_D, lw=1.2, label="low bits, no protection")
    ax.plot([q for q, e_ in zip(rates, ex) if e_ < 40], [v for v, e_ in zip(y, ex) if e_ < 40], "s", color=NOISE_D, ms=3)
    ax.axhline(1, color=S.INK2, lw=0.8, ls=":")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("bit-error probability $p$", fontsize=7); ax.set_ylabel("flips among one neuron's weights (mean)", fontsize=7)
    ax.set_title("(d) how many reach one neuron", loc="left", fontsize=8.5)
    ax.set_ylim(5e-3, 2e3)
    ax.legend(fontsize=5.8, loc="upper left", handlelength=1.8)
    ax = axs[1, 1]
    for m, lab, ls in MASKS:
        y = [float(r["event_share_measured"]) for r in by[m]]
        if max(y) > 0:
            ax.plot(rates, y, ls, color=EVENT, lw=1.8, marker="o", ms=3.5, label=lab)
            ax.axhline(EX[m]["event_variance_fraction"], color=S.INK2, lw=1.0, ls=":")
    ax.set_xscale("log"); ax.set_ylim(0, 1.05)
    ax.set_xlabel("bit-error probability $p$", fontsize=7); ax.set_ylabel("events' second moment\n/ (events' + background's)", fontsize=7)
    ax.set_title("(e) they grow together", loc="left", fontsize=8.5)
    ax.legend(fontsize=6.0, loc="lower left", title="sampled (dotted: exact fraction\nof the variance)", title_fontsize=6.0)
    ax = axs[1, 2]
    for key, col, lab in (("ks_top", EVENT, "events"), ("ks_low", NOISE_D, "background"), ("ks_sum", S.INK, "total")):
        ax.plot(rates, [float(r[key]) for r in un], "-", color=col, lw=1.8, marker="o", ms=3.5, label=lab)
    ax.axhline(st["thresholds"]["ks_limit"], color=S.INK2, lw=0.8, ls=":")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("bit-error probability $p$", fontsize=7); ax.set_ylabel("largest gap from a normal distribution", fontsize=7)
    ax.set_title("(f) distance from a bell curve", loc="left", fontsize=8.5)
    ax.legend(fontsize=6.3, loc="lower left")
    ax.text(1.2e-5, st["thresholds"]["ks_limit"] * 0.86, "limit fixed in advance for $p=10^{-1}$", fontsize=5.8, color=S.INK2, va="top")
    fig.tight_layout(h_pad=1.4, w_pad=1.0, rect=(0, 0, 1, 0.955))
    fig.legend(*hl, loc="upper center", ncol=2, fontsize=7, bbox_to_anchor=(0.5, 1.0), frameon=False)
    for ext in ("pdf", "png"):
        fig.savefig(ROOT / "figures" / f"fig_event_background.{ext}", bbox_inches="tight", **({"dpi": 220} if ext == "png" else {}))
    out = dict(figure_layer=L, layer=[x for x in st["layers"] if x["layer"] == L][0], shown_rates=[rates[i] for i in shown],
               rows={m: by[m] for m, _, _ in MASKS},
               expected_events_mean={m: expected(by[m], "events_mean") for m, _, _ in MASKS},
               exact=json.loads(EXACT.read_text()), statements=st["statements"], thresholds=st["thresholds"],
               layers=st["layers"],
               inputs_sha256=dict(statements=hashlib.sha256((AN / "statements.json").read_bytes()).hexdigest(),
                                  summary=hashlib.sha256((AN / "summary.csv").read_bytes()).hexdigest(),
                                  histograms=hashlib.sha256(Hf.read_bytes()).hexdigest(),
                                  exact=hashlib.sha256(EXACT.read_bytes()).hexdigest()))
    (ROOT / "data" / "tecs_event_background.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
