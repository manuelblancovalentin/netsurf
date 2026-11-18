#!/usr/bin/env python3
"""EXPLORATORY: compare the two-moment surrogate with the measured DistilBERT Monte Carlo grid.

Inputs: shard_*.json of drift_diffusion_prototype.py, the drift-only scores of meanfield_drift_v1, and the
tracked full-grid audit table.  Output: one JSON with side-by-side values and agreement summaries.
Nothing here is a manuscript result.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
ALIAS = {"msb_first": "msb", "random": "random", "numerical_bit_jump": "jump", "global_first_order": "global_grad",
         "per_example_absolute": "abs_grad", "per_example_positive": "pos_grad", "unprotected": "unprotected"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, default=ROOT / "workspace/prototypes/drift_diffusion_v1")
    ap.add_argument("--drift-only", type=Path, default=ROOT / "workspace/prototypes/meanfield_drift_v1/meanfield_scores.json")
    ap.add_argument("--grid", type=Path, default=ROOT / "publications/paper1_review/data/tecs_audit/conditions_with_quality.csv")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    measured = {}
    for r in csv.DictReader(open(args.grid)):
        if r["model"] == "distilbert":
            measured[(float(r["ber"]), float(r["coverage"]), r["method"])] = dict(
                nll=float(r["mean_loss"]), acc=float(r["mean_quality"]), min_field_acc=float(r["min_field_quality"]))
    drift = {(r["ber"], r["coverage"], ALIAS[r["policy"]]): dict(acc=r["meanfield_accuracy"], nll=r["meanfield_nll"])
             for r in json.loads(args.drift_only.read_text())["rows"]}
    sur = defaultdict(lambda: defaultdict(list)); shards = sorted(args.scores.glob("shard_*.json")); clean = None
    for path in shards:
        doc = json.loads(path.read_text()); clean = doc["exact_clean"]
        for r in doc["rows"]:
            sur[(r["ber"], r["coverage"], ALIAS[r["policy"]])][r["variant"]].append((r["accuracy"], r["nll"]))
    rows = []
    for key in sorted(sur, key=lambda k: (k[1], k[2], k[0])):
        if key not in measured:
            continue
        full = np.array(sur[key]["full"]); diff = np.array(sur[key]["diffusion"]) if sur[key]["diffusion"] else None
        m = measured[key]; d = drift.get(key)
        rows.append(dict(ber=key[0], coverage=key[1], method=key[2], measured_acc=m["acc"], measured_nll=m["nll"],
                         surrogate_acc=float(full[:, 0].mean()), surrogate_acc_sd=float(full[:, 0].std(ddof=1)) if len(full) > 1 else float("nan"),
                         surrogate_nll=float(full[:, 1].mean()), draws=len(full),
                         diffusion_only_acc=float(diff[:, 0].mean()) if diff is not None else float("nan"),
                         diffusion_only_nll=float(diff[:, 1].mean()) if diff is not None else float("nan"),
                         drift_only_acc=d["acc"] if d else float("nan"), drift_only_nll=d["nll"] if d else float("nan")))
    out = dict(schema="paper1-drift-diffusion-compare/1.0", status="EXPLORATORY", shards=len(shards), exact_clean=clean, rows=rows)

    def err(col):
        e = np.array([abs(r[col] - r["measured_acc"]) for r in rows if r[col] == r[col]])
        return dict(n=int(e.size), mean_abs_error=float(e.mean()), median_abs_error=float(np.median(e)), max_abs_error=float(e.max()),
                    within_2_points=float((e <= 0.02).mean()), within_5_points=float((e <= 0.05).mean()))

    out["accuracy_error"] = {name: err(col) for name, col in (("drift_only", "drift_only_acc"), ("diffusion_only", "diffusion_only_acc"), ("drift_and_diffusion", "surrogate_acc"))}
    cells = []
    for cov in sorted({r["coverage"] for r in rows if r["coverage"] > 0}):
        for ber in sorted({r["ber"] for r in rows}):
            sub = [r for r in rows if r["coverage"] == cov and r["ber"] == ber]
            if len(sub) < 6:
                continue
            mc = np.array([r["measured_acc"] for r in sub])
            cell = dict(coverage=cov, ber=ber, measured_spread_acc=float(mc.max() - mc.min()),
                        measured_best=sub[int(mc.argmax())]["method"])
            for name, col in (("drift_only", "drift_only_acc"), ("diffusion_only", "diffusion_only_acc"), ("drift_and_diffusion", "surrogate_acc")):
                pred = np.array([r[col] for r in sub])
                cell[f"spearman_{name}"] = float(spearmanr(pred, mc).statistic)
                cell[f"best_{name}"] = sub[int(pred.argmax())]["method"]
                cell[f"measured_acc_of_best_{name}"] = float(mc[int(pred.argmax())])
            cells.append(cell)
    out["cells"] = cells
    spread = [c for c in cells if c["measured_spread_acc"] >= 0.02]
    out["cells_with_spread_at_least_2_points"] = dict(
        n=len(spread),
        **{f"mean_spearman_{n}": float(np.mean([c[f"spearman_{n}"] for c in spread])) for n in ("drift_only", "diffusion_only", "drift_and_diffusion")},
        **{f"mean_accuracy_lost_by_choosing_with_{n}": float(np.mean([max(r["measured_acc"] for r in rows if r["coverage"] == c["coverage"] and r["ber"] == c["ber"]) - c[f"measured_acc_of_best_{n}"] for c in spread]))
           for n in ("drift_only", "diffusion_only", "drift_and_diffusion")})
    with open(args.out, "x") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out["accuracy_error"], indent=1)); print(json.dumps(out["cells_with_spread_at_least_2_points"], indent=1))


if __name__ == "__main__":
    main()
