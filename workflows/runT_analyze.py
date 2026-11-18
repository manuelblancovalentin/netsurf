#!/usr/bin/env python3
"""Run T analysis (one checkpoint family at a time).  Protocol: review/RUN_T_T1_PROTOCOL_2026-11-15.md (paper worktree).
Run once per stage, after every job of the stage has ended.  All differences are in accuracy points.

  unit        one fault draw of one replication (24 per checkpoint), paired across conditions
  curve       per (checkpoint, condition, rate): mean accuracy over draws, and accuracy lost against the mean clean
              accuracy of the replications
  tolerated   per (checkpoint, condition): the highest grid rate up to which the mean accuracy stays within 5 points
              of clean at every lower grid rate as well
  contrasts   fixed pairs at (nearly) equal extra storage, per checkpoint and rate, in cells where the better of the
              pair is within 5 points of clean; paired sign-flip test (100,000 assignments, seed 20261123), Holm over
              all contrasts and usable rates of the checkpoint; "resolved" = |difference| >= 0.5 and Holm p < 0.05

    runT_analyze.py --runs DIR --out DIR --ids jet-e1 jet-e3 ... [--draws 24] [--suffix _fkeras ...]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

MARGIN, MIN_DIFF, ALPHA, ASSIGNMENTS, SEED = 5.0, 0.5, 0.05, 100_000, 20261123
# (name, left, right): left minus right
CONTRASTS = (
    ("ranked TMR vs SEC-DED-64, 12.5% storage", "tmr_abs@0.0625", "secded_64"),
    ("ranked TMR (25%) vs SEC-DED-32 (21.9%)", "tmr_abs@0.125", "secded_32"),
    ("ranked TMR vs SEC-DED-16, 37.5% storage", "tmr_abs@0.1875", "secded_16"),
    ("ranked TMR vs sign-bit TMR (FORTUNE), 25% storage", "tmr_abs@0.125", "tmr_msb@0.125"),
    ("abs vs MSB-first ranking, 12.5% storage", "tmr_abs@0.0625", "tmr_msb@0.0625"),
    ("abs vs MSB-first ranking, 37.5% storage", "tmr_abs@0.1875", "tmr_msb@0.1875"),
    ("abs vs MSB-first ranking, 50% storage", "tmr_abs@0.25", "tmr_msb@0.25"),
    ("abs vs FKeras ranking, 12.5% storage", "tmr_abs@0.0625", "tmr_fkeras@0.0625"),
    ("abs vs FKeras ranking, 37.5% storage", "tmr_abs@0.1875", "tmr_fkeras@0.1875"),
    ("abs vs FKeras (weighted) ranking, 25% storage", "tmr_abs@0.125", "tmr_fkeras_weighted@0.125"),
    ("ranked TMR (12.5%) vs ABFT, ideal checksums", "tmr_abs@0.0625", "abft_ideal"),
    ("ABFT: faulty vs ideal checksums", "abft_faulty", "abft_ideal"),
    ("best ranked TMR at 25% vs no protection", "tmr_abs@0.125", "unprotected"),
)


def signflip_p(d, signs):
    d = np.asarray(d, dtype=float)
    if not d.any():
        return 1.0
    null = np.abs(signs[:, : d.size] @ d)
    return float((1 + (null >= abs(d.sum()) - 1e-12).sum()) / (signs.shape[0] + 1))


def holm(ps):
    ps = np.asarray(ps, dtype=float)
    adj, running = np.empty(ps.size), 0.0
    for rank, i in enumerate(np.argsort(ps, kind="stable")):
        running = max(running, (ps.size - rank) * ps[i])
        adj[i] = min(1.0, running)
    return adj


def tolerated(curve, rates, clean):
    """Highest rate r such that the mean accuracy is within MARGIN of clean at r and at every lower grid rate."""
    best = None
    for r in sorted(rates):
        if curve[r] < clean - MARGIN:
            break
        best = r
    return best


def analyze(acc, clean, signs):
    """acc: {(condition, rate): {unit: accuracy in %}}; clean in %.  Returns curves, tolerated rates and contrasts."""
    rates = sorted({k[1] for k in acc})
    conds = sorted({k[0] for k in acc})
    mean = {k: float(np.mean(list(v.values()))) for k, v in acc.items()}
    curves = {c: {r: mean[(c, r)] for r in rates if (c, r) in mean} for c in conds}
    tol = {c: tolerated(curves[c], [r for r in rates if r in curves[c]], clean) for c in conds}
    rows = []
    for name, left, right in CONTRASTS:
        if left not in curves or right not in curves:
            continue
        for r in rates:
            if max(mean[(left, r)], mean[(right, r)]) < clean - MARGIN:
                continue
            L, R = acc[(left, r)], acc[(right, r)]
            if sorted(L) != sorted(R):
                raise SystemExit(f"{left} and {right} have different draws at {r}")
            d = np.array([L[u] - R[u] for u in sorted(L)])
            rows.append(dict(contrast=name, left=left, right=right, ber=r, diff=float(d.mean()), draws=int(d.size),
                             p_raw=signflip_p(d, signs)))
    for row, h in zip(rows, holm([row["p_raw"] for row in rows])):
        row["p_holm"] = float(h)
        row["resolved"] = bool(abs(row["diff"]) >= MIN_DIFF and h < ALPHA)
    return curves, tol, rows


def load(base: Path, suffixes, draws):
    """{(condition, rate): {(rep, draw): accuracy %}} and the mean clean accuracy (%) of one checkpoint."""
    acc, clean, units = defaultdict(dict), {}, set()
    for d in sorted(base.glob("rep*/draw*")):
        rep, dr = int(d.parent.name[3:]), int(d.name[4:])
        for sfx in suffixes:
            if not (d / f"DONE{sfx}").exists():
                raise SystemExit(f"{d}: results{sfx} not finished")
            with open(d / f"results{sfx}.csv") as fh:
                for r in csv.DictReader(fh):
                    acc[(r["condition"], float(r["ber"]))][(rep, dr)] = 100 * float(r["accuracy"])
        clean[rep] = 100 * json.loads((d / "DONE").read_text())["clean_accuracy"]
        units.add((rep, dr))
    if len(units) != draws:
        raise SystemExit(f"{base}: {len(units)} draws found, {draws} expected")
    return acc, float(np.mean(list(clean.values())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ids", nargs="+", required=True)
    ap.add_argument("--draws", type=int, default=24)
    ap.add_argument("--suffix", nargs="*", default=[])
    a = ap.parse_args()
    signs = np.random.default_rng(SEED).choice((-1.0, 1.0), size=(ASSIGNMENTS, a.draws))
    data = {cid: load(a.runs / cid, [""] + list(a.suffix), a.draws) for cid in a.ids}     # fails before writing anything
    a.out.mkdir(parents=True, exist_ok=False)
    summary, crows, trows = {}, [], []
    for cid in a.ids:
        acc, clean = data[cid]
        curves, tol, rows = analyze(acc, clean, signs)
        summary[cid] = dict(clean=clean, tolerated={c: tol[c] for c in curves},
                            contrasts_resolved=sum(r["resolved"] for r in rows), contrasts=len(rows))
        crows += [dict(checkpoint=cid, condition=c, ber=r, accuracy=v, lost=clean - v) for c in curves for r, v in curves[c].items()]
        trows += [dict(checkpoint=cid, **r) for r in rows]
        print(f"{cid:12s} clean {clean:.1f}  tolerated: abs@12.5% {tol.get('tmr_abs@0.125')}  msb@12.5% {tol.get('tmr_msb@0.125')}  "
              f"secded_32 {tol.get('secded_32')}  abft {tol.get('abft_ideal')}  none {tol.get('unprotected')}")
    for nm, rows in (("curves.csv", crows), ("contrasts.csv", trows)):
        with open(a.out / nm, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (a.out / "summary.json").write_text(json.dumps(dict(
        status="Run T analysis (review/RUN_T_T1_PROTOCOL_2026-11-15.md)",
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), checkpoints=summary), indent=1))


if __name__ == "__main__":
    main()
