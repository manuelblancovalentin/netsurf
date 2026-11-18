#!/usr/bin/env python3
"""Run T figures: protection mechanisms under non-ideal protection, one panel per model.

Reads the finished Run T outputs (workspace/runs/paper1_runT_tflite_v1/<id>/rep*/draw*/results*.csv for the TFLite
models; paper1_runT_w8a8_v1/<target>/draw*/ber*/results.csv for the language models), averages over draws, and writes
  figures/tecs_runT_main.{pdf,png}       selected models, two rows: 12.5% and 25% extra storage
  figures/tecs_runT_all.{pdf,png}        every finished TFLite checkpoint, all rankings, 25% extra storage
  data/tecs_runT/curves.csv              the plotted means (and the input file hashes in data/tecs_runT/inputs.json)
A checkpoint is drawn only when all its jobs are finished.
"""
from __future__ import annotations

import csv
import hashlib
import json
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

TFL = C.RUNS / "paper1_runT_tflite_v1"
LM = C.RUNS / "paper1_runT_w8a8_v1"
DATA = C.ROOT / "data/tecs_runT"
MAIN = [("jet-e30", "Jet MLP (epoch 30)"), ("mnist-e30", "MNIST MLP (epoch 30)"), ("vww", "Visual Wake Words"),
        ("dscnn", "DS-CNN"), ("resnet8-pinned", "ResNet-8"), ("toycar", "ToyCar (AUC)"),
        ("distilbert_sst2_int8", "DistilBERT"), ("pythia160m_int8", "Pythia-160M"), ("qwen3_0_6b_int8", "Qwen3-0.6B")]
LMS = {"distilbert_sst2_int8": (64, "accuracy"), "pythia160m_int8": (64, "mean_nll"), "qwen3_0_6b_int8": (32, "mean_nll")}
ROWS = {  # extra storage: (condition, label, colour, style)
    "12.5% extra storage": [("tmr_abs@0.0625", "TMR, absolute gradient", C.ABS, "-"), ("tmr_fkeras@0.0625", "TMR, FKeras", "#6250d6", "-"),
                            ("tmr_msb@0.0625", "TMR, MSB-first", C.MSB, "-"), ("secded_64", "SEC-DED (64 bits)", C.POS, "--"),
                            ("abft_ideal", "ABFT", C.V2, ":"), ("abft_faulty", "ABFT, faulty checksums", C.V2, "-."),
                            ("unprotected", "no protection", "#a8a7a2", ":")],
    "25% extra storage": [("tmr_abs@0.125", "TMR, absolute gradient", C.ABS, "-"), ("tmr_fkeras_weighted@0.125", "TMR, FKeras (tier-mixing variant)", "#6250d6", "--"),
                          ("tmr_msb@0.125", "TMR of sign bits (FORTUNE; = FKeras)", C.MSB, "-"), ("secded_32", "SEC-DED (32 bits, 21.9%)", C.POS, "--"),
                          ("abft_ideal", "ABFT", C.V2, ":"), ("abft_faulty", "ABFT, faulty checksums", C.V2, "-."),
                          ("unprotected", "no protection", "#a8a7a2", ":")],
}


def load_tflite(cid, inputs):
    base = TFL / cid
    dones = sorted(base.glob("rep*/draw*/DONE"))
    want = 8 if cid == "toycar" else 24                      # every job of the checkpoint must be finished
    if len(dones) < want or (cid.startswith(("jet", "mnist", "vww")) and len(list(base.glob("rep*/draw*/DONE_fkeras"))) < 24):
        return None
    acc = defaultdict(list)
    for d in sorted(base.glob("rep*/draw*")):
        for f in sorted(d.glob("results*.csv")):
            inputs[str(f.relative_to(C.RUNS))] = hashlib.sha256(f.read_bytes()).hexdigest()
            for r in csv.DictReader(open(f)):
                acc[(r["condition"], float(r["ber"]))].append(100 * float(r["accuracy"]))
    clean = np.mean([100 * json.loads(d.read_text())["clean_accuracy"] for d in dones])
    return {k: float(np.mean(v)) for k, v in acc.items()}, float(clean), "accuracy (%)"


def load_lm(target, inputs):
    want, metric = LMS[target]
    dones = sorted((LM / target).glob("draw*/ber*/DONE"))
    if len(dones) < want:
        return None
    acc, clean = defaultdict(list), []
    for d in dones:
        f = d.parent / "results.csv"
        inputs[str(f.relative_to(C.RUNS))] = hashlib.sha256(f.read_bytes()).hexdigest()
        for r in csv.DictReader(open(f)):
            v = float(r[f"metric_{metric}"])
            acc[(r["condition"], float(r["ber"]))].append(100 * v if metric == "accuracy" else v)
        c = json.loads(d.read_text())["clean"][metric]
        clean.append(100 * c if metric == "accuracy" else c)
    return {k: float(np.mean(v)) for k, v in acc.items()}, float(np.mean(clean)), ("accuracy (%)" if metric == "accuracy" else "NLL (nats/token)")


def panel(ax, data, clean, unit, conds, legend=False):
    lower_better = unit.startswith("NLL")
    band = (clean, clean + 0.5) if lower_better else (clean - 5, clean)
    ax.axhspan(*band, color=C.GRID, alpha=0.6, lw=0)
    for cond, lab, col, ls in conds:
        xs = sorted(p for (c, p) in data if c == cond)
        if not xs:
            continue
        ax.plot(xs, [data[(cond, p)] for p in xs], ls, color=col, lw=1.6, marker="o", ms=2.5, label=lab)
    ax.set_xscale("log")
    if lower_better:
        ax.set_yscale("log")
    else:
        ax.set_ylim(max(0, clean - 40), clean + 3)


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    inputs, done = {}, {}
    for cid, _ in MAIN:
        r = load_lm(cid, inputs) if cid in LMS else load_tflite(cid, inputs)
        if r:
            done[cid] = r
    sel = [(c, t) for c, t in MAIN if c in done]
    for tag, (row, conds) in zip(("main", "125"), list(ROWS.items())[::-1]):
        ncol = 3
        nrow = int(np.ceil(len(sel) / ncol))
        fig, axs = plt.subplots(nrow, ncol, figsize=(10.5, 3.0 * nrow + 0.6), squeeze=False)
        for k, (cid, title) in enumerate(sel):
            data, clean, unit = done[cid]
            ax = axs[k // ncol, k % ncol]
            panel(ax, data, clean, unit, conds)
            C.title(ax, title)
            ax.set_ylabel(unit, fontsize=8)
            if k // ncol == nrow - 1:
                ax.set_xlabel("bit-error probability p")
        for k in range(len(sel), nrow * ncol):
            axs[k // ncol, k % ncol].axis("off")
        h, l = axs[0, 0].get_legend_handles_labels()
        fig.legend(h, l, fontsize=8, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01))
        fig.tight_layout(rect=(0, 0.07 if nrow > 1 else 0.12, 1, 1))
        name = "tecs_runT_main" if tag == "main" else "tecs_runT_125"
        for ext in ("pdf", "png"):
            fig.savefig(C.ROOT / "figures" / f"{name}.{ext}", bbox_inches="tight", **({"dpi": 200} if ext == "png" else {}))
        plt.close(fig)
    # appendix: every finished TFLite checkpoint, all rankings at 25% extra storage
    ids = sorted(p.name for p in TFL.iterdir() if p.is_dir())
    allc = [(c, load_tflite(c, inputs)) for c in ids]
    allc = [(c, r) for c, r in allc if r]
    ncol = 4
    nrow = int(np.ceil(len(allc) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(12, 2.7 * nrow), squeeze=False)
    rank = [("tmr_abs@0.125", "absolute", C.ABS, "-"), ("tmr_pos@0.125", "positive", C.POS, "-"), ("tmr_global@0.125", "global", "#e87ba4", "-"),
            ("tmr_jump@0.125", "jump", "#eda100", "-"), ("tmr_msb@0.125", "MSB-first (= FKeras)", C.MSB, "-"),
            ("tmr_fkeras_weighted@0.125", "FKeras tier-mixing", "#6250d6", "--"), ("tmr_random@0.125", "random", "#a8a7a2", "--"),
            ("secded_32", "SEC-DED 32", C.POS, ":"), ("abft_ideal", "ABFT", C.V2, ":"), ("unprotected", "none", "#a8a7a2", ":")]
    for k, (cid, (data, clean, unit)) in enumerate(allc):
        ax = axs[k // ncol, k % ncol]
        panel(ax, data, clean, unit, rank)
        C.title(ax, cid)
    for k in range(len(allc), nrow * ncol):
        axs[k // ncol, k % ncol].axis("off")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(C.ROOT / "figures" / f"tecs_runT_all.{ext}", bbox_inches="tight", **({"dpi": 170} if ext == "png" else {}))
    with open(DATA / "curves.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["checkpoint", "condition", "ber", "mean", "clean", "unit"])
        for cid, (data, clean, unit) in list(done.items()) + [(c, r) for c, r in allc if c not in done]:
            for (cond, p), v in sorted(data.items()):
                w.writerow([cid, cond, p, round(v, 4), round(clean, 4), unit])
    (DATA / "inputs.json").write_text(json.dumps(dict(sorted(inputs.items())), indent=0) + "\n")
    print("main panels:", [c for c, _ in sel]); print("appendix panels:", [c for c, _ in allc])


if __name__ == "__main__":
    main()
