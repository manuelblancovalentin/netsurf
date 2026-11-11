#!/usr/bin/env python3
"""EXPLORATORY: compare the mean-field drift prototype with the measured DistilBERT Monte Carlo grid.

Reads  workspace/prototypes/meanfield_drift_v1/meanfield_scores.json  (one forward pass per mask and BER
at the expected stored codes, no fault field sampled) and the tracked full-grid audit table, and writes a
new JSON with the side-by-side values and per-cell rank agreement.  Nothing here is a manuscript result.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
ALIAS = {"msb_first": "msb", "random": "random", "numerical_bit_jump": "jump", "global_first_order": "global_grad",
         "per_example_absolute": "abs_grad", "per_example_positive": "pos_grad", "unprotected": "unprotected"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, default=ROOT / "workspace/prototypes/meanfield_drift_v1/meanfield_scores.json")
    ap.add_argument("--grid", type=Path, default=ROOT / "publications/paper1_review/data/tecs_audit/conditions_with_quality.csv")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    mf = json.loads(args.scores.read_text())
    clean_nll, clean_acc = mf["exact_clean"]["mean_nll"], mf["exact_clean"]["accuracy"]
    measured = {}
    for r in csv.DictReader(open(args.grid)):
        if r["model"] == "distilbert":
            measured[(float(r["ber"]), float(r["coverage"]), r["method"])] = (float(r["mean_loss"]), float(r["mean_quality"]))
    rows = []
    for r in mf["rows"]:
        key = (r["ber"], r["coverage"], ALIAS[r["policy"]])
        if key not in measured:
            continue
        mc_loss, mc_acc = measured[key]
        rows.append(dict(method=key[2], coverage=key[1], ber=key[0],
                         meanfield_nll=r["meanfield_nll"], measured_nll=mc_loss,
                         meanfield_accuracy=r["meanfield_accuracy"], measured_accuracy=mc_acc,
                         measured_excess_nll=mc_loss - clean_nll, meanfield_excess_nll=r["meanfield_nll"] - clean_nll,
                         first_order_drift_per_unit_p=r["first_order_drift_per_unit_p"],
                         diffusion_proxy_per_unit_p=r["diffusion_proxy_per_unit_p"]))
    cells = []
    for cov in sorted({r["coverage"] for r in rows if r["coverage"] > 0}):
        for ber in sorted({r["ber"] for r in rows}):
            sub = [r for r in rows if r["coverage"] == cov and r["ber"] == ber]
            if len(sub) < 6:
                continue
            mc = np.array([r["measured_nll"] for r in sub])
            rho_mf = spearmanr([r["meanfield_nll"] for r in sub], mc).statistic
            rho_diff = spearmanr([r["diffusion_proxy_per_unit_p"] for r in sub], mc).statistic
            cells.append(dict(coverage=cov, ber=ber, measured_spread_nll=float(mc.max() - mc.min()),
                              spearman_meanfield_vs_measured=float(rho_mf), spearman_diffusion_proxy_vs_measured=float(rho_diff),
                              measured_best=sub[int(mc.argmin())]["method"],
                              meanfield_best=sub[int(np.argmin([r["meanfield_nll"] for r in sub]))]["method"]))
    out = dict(schema="paper1-meanfield-drift-compare/1.0", status="EXPLORATORY", clean_nll=clean_nll, clean_accuracy=clean_acc,
               rows=rows, cells=cells)
    with open(args.out, "x") as f:
        json.dump(out, f, indent=1)
    for c in cells:
        print(json.dumps(c))


if __name__ == "__main__":
    main()
