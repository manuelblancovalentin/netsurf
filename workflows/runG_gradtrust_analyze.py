#!/usr/bin/env python3
"""Gradient-trust plan (review/GRADIENT_TRUST_PLAN_2026-11-15.md), Phase 3: the decision analysis, run once.

Inputs: the frozen predictions (Phase 1, predictions.json, sha256 checked) and the Phase 2 fault results
(<faults>/<checkpoint>/rep<r>/field<f>/results.csv with columns checkpoint, rep, field, ber, budget, arm, accuracy,
window and clean.csv with checkpoint, rep, clean_accuracy).

Per checkpoint (Section 5):
  usable cell (budget, rate): the better of pos1024 and abs1024 has mean accuracy within 0.05 of the checkpoint's clean
  accuracy (mean over replications);
  per usable cell: pos1024 minus abs1024 per (replication, field), in inputs of the window; two-sided sign-flip test
  on the pooled differences (exact integer arithmetic, 100,000 assignments, seed 20261123); Holm over the
  checkpoint's usable cells;
  measured "two regimes" if some usable cell has pos1024 ahead by at least 0.2 point with Holm p < 0.05, else "one
  regime".
Decision (Section 6), over test checkpoints with a prediction ("one regime" / "two regimes"), excluding exploratory
ones: supported if at least 80% match and both classes contain a matching checkpoint; not tested both ways if all
predictions are in one class (and at least 80% match); failed below 60%; otherwise inconclusive.

    runG_gradtrust_analyze.py --faults DIR --predictions FILE --predictions-sha HEX --out DIR
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

MARGIN, MIN_GAIN, ALPHA, ASSIGNMENTS, SEED = 0.05, 0.2, 0.05, 100_000, 20261123


def signflip_p(counts, assignments=ASSIGNMENTS, seed=SEED):
    """Two-sided p of |sum of counts| under random sign flips (counts are integers; exact sums in float64)."""
    c = np.asarray(counts, dtype=np.float64)
    if not c.any():
        return 1.0
    signs = np.random.default_rng(seed).choice((-1.0, 1.0), size=(assignments, c.size))
    obs = abs(c.sum())
    return float((np.sum(np.abs(signs @ c) >= obs) + 1) / (assignments + 1))


def holm(ps):
    ps = np.asarray(ps, dtype=float)
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - k) * ps[i]))
        adj[i] = run
    return adj


def measured_regime(rows, clean, window):
    """rows: dicts with rep, field, ber, budget, arm, accuracy for one checkpoint; clean: mean clean accuracy."""
    acc = defaultdict(dict)
    for r in rows:
        acc[(float(r["budget"]), float(r["ber"]), r["arm"])][(int(r["rep"]), int(r["field"]))] = float(r["accuracy"])
    cells = []
    for (b, p) in sorted({(k[0], k[1]) for k in acc}):
        P, A = acc.get((b, p, "pos1024")), acc.get((b, p, "abs1024"))
        if not P or not A or sorted(P) != sorted(A):
            raise ValueError(f"cell {b} {p}: pos1024 and abs1024 fields differ")
        best = max(np.mean(list(P.values())), np.mean(list(A.values())))
        if best < clean - MARGIN:
            continue
        keys = sorted(P)
        n = np.rint(np.array([P[k] - A[k] for k in keys]) * window)
        cells.append(dict(budget=b, ber=p, diff=float(100 * n.mean() / window), fields=len(keys), p_raw=signflip_p(n)))
    for c, h in zip(cells, holm([c["p_raw"] for c in cells]) if cells else []):
        c["p_holm"] = float(h)
        c["pos_ahead"] = bool(h < ALPHA and c["diff"] >= MIN_GAIN)
        c["abs_ahead"] = bool(h < ALPHA and c["diff"] <= -MIN_GAIN)
    regime = "two regimes" if any(c["pos_ahead"] for c in cells) else "one regime"
    return regime, cells


def decide(pairs):
    """pairs: list of (predicted, measured) for test checkpoints with a prediction."""
    if not pairs:
        return dict(outcome="no predictions", matched=0, total=0)
    match = [p == m for p, m in pairs]
    share = float(np.mean(match))
    classes = {p for p, _ in pairs}
    both = all(any(p == c and p == m for p, m in pairs) for c in ("one regime", "two regimes"))
    if share >= 0.8 and both:
        outcome = "supported"
    elif share >= 0.8 and len(classes) == 1:
        outcome = "not tested both ways"
    elif share < 0.6:
        outcome = "failed"
    else:
        outcome = "inconclusive"
    return dict(outcome=outcome, matched=int(sum(match)), total=len(pairs), share=share)


def main(faults: Path, predictions: Path, predictions_sha: str, out: Path):
    raw = predictions.read_bytes()
    if hashlib.sha256(raw).hexdigest() != predictions_sha:
        raise SystemExit("predictions.json differs from its frozen sha256")
    pred = json.loads(raw)
    res = dict(status="gradient-trust Phase 3 (plan Sections 5 and 6)", predictions_sha256=predictions_sha, checkpoints={})
    pairs = []
    for cid, info in sorted(pred["checkpoints"].items()):
        rows = [r for p in sorted((faults / cid).glob("rep*/field*/results.csv")) for r in csv.DictReader(open(p))]
        clean_rows = list(csv.DictReader(open(faults / cid / "clean.csv")))
        clean = float(np.mean([float(r["clean_accuracy"]) for r in clean_rows]))
        window = int(rows[0]["window"])
        regime, cells = measured_regime(rows, clean, window)
        res["checkpoints"][cid] = dict(predicted=info["prediction"], D1=info["D1"], measured=regime, clean=clean,
                                       usable_cells=len(cells), exploratory=bool(info.get("exploratory")), cells=cells)
        if info["prediction"] != "no prediction" and not info.get("exploratory"):
            pairs.append((info["prediction"], regime))
        print(f"{cid:24s} D1 {info['D1']:.3f}  predicted {info['prediction']:14s} measured {regime:12s} "
              f"usable {len(cells)}  pos ahead {sum(c['pos_ahead'] for c in cells)}  abs ahead {sum(c['abs_ahead'] for c in cells)}")
    res["decision"] = decide(pairs)
    print("decision:", json.dumps(res["decision"]))
    out.mkdir(parents=True, exist_ok=False)
    (out / "summary.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--faults", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--predictions-sha", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    main(a.faults, a.predictions, a.predictions_sha, a.out)
