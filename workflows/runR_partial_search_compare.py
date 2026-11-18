#!/usr/bin/env python3
"""Partial-group search (plan review/PARTIAL_GROUP_SEARCH_PLAN_2026-11-14.md): the search masks against the six Stage 4
masks of the same replications, on the same 8 fault fields and 800-clip test window (TFLite), paired by field.
Written before any partial-search outcome was read.  Reads locked outputs only.  Pure NumPy.

    runR_partial_search_compare.py <out> --reps 0            Phase C1, descriptive (replication 0 already inspected)
    runR_partial_search_compare.py <out> --reps 1 2 3 4 5    Phase C2, the plan's confirmatory analysis

Per replication and cell (rate, budget): mean over the 8 fields of search minus each Stage 4 arm, with the standard
error over fields; the C1 rule (cells behind abs1024 by two standard errors); bits differing from abs1024; each
search's accepted slice order and saturation round.
With two or more replications (C2): per cell, the replication means of (search - abs1024) [mean over fields first],
their mean, standard error over replications, two-sided t test with n-1 degrees of freedom, Holm over the 12 cells
at 0.05; equivalence: the 90% t interval inside +-1 point.  Secondary, descriptive: minus msb, fkeras, ra_analytic.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from site_paths import WT  # noqa: E402

RUNS = WT / "paper1-tecs-revision-v1/workspace/runs"
SEARCH = RUNS / "paper1_partial_search_kws_v1"
NEW = RUNS / "paper1_partial_search_kws_eval_v1"
OLD = RUNS / "paper1_runR_stage4_kws_eval_v1"
STAGE4 = ("abs1024", "pos1024", "abs64", "msb", "fkeras", "ra_analytic")
NAMES = ["dw1", "dw2", "dw3", "dw4", "dense", "conv1", "pw1", "pw2", "pw3", "pw4"]
FIELDS = 8


def t_two_sided(t, df):
    """Two-sided p value of Student's t (regularized incomplete beta by continued fraction; no SciPy needed)."""
    from math import exp, lgamma, log
    x = df / (df + t * t)
    a, b = df / 2.0, 0.5

    # I_x(a, b) via the Lentz continued fraction of the standard form (Numerical Recipes betacf)
    def betai(a, b, x):
        if x <= 0:
            return 0.0
        if x >= 1:
            return 1.0
        bt = exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * log(x) + b * log(1 - x))
        if x < (a + 1) / (a + b + 2):
            return bt * betacf(a, b, x) / a
        return 1.0 - bt * betacf(b, a, 1 - x) / b

    def betacf(a, b, x):
        tiny = 1e-300
        qab, qap, qam = a + b, a + 1, a - 1
        c, d = 1.0, 1.0 - qab * x / qap
        d = 1.0 / (tiny if abs(d) < tiny else d)
        h = d
        for m in range(1, 400):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d; d = 1.0 / (tiny if abs(d) < tiny else d)
            c = 1.0 + aa / c; c = tiny if abs(c) < tiny else c
            h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d; d = 1.0 / (tiny if abs(d) < tiny else d)
            c = 1.0 + aa / c; c = tiny if abs(c) < tiny else c
            de = d * c
            h *= de
            if abs(de - 1.0) < 1e-14:
                break
        return h

    return float(betai(a, b, x))


def t_quantile(q, df):
    """Upper quantile of Student's t by bisection on t_two_sided (q = 0.95 for a 90% interval)."""
    lo, hi = 0.0, 100.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if 1 - t_two_sided(mid, df) / 2 < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - k) * ps[i]))
        adj[i] = run
    return adj


def load(reps):
    acc = {}
    for r in reps:
        for f in range(FIELDS):
            for row in csv.DictReader(open(OLD / f"rep{r}/field{f}/results.csv")):
                acc[(r, row["arm"], float(row["budget"]), float(row["ber"]), f)] = float(row["accuracy"])
            for row in csv.DictReader(open(NEW / f"rep{r}/field{f}/results.csv")):
                acc[(r, row["arm"], float(row["budget"]), float(row["ber"]), f)] = float(row["accuracy"])
    return acc


def main(out: Path, reps):
    acc = load(reps)
    arms = sorted({k[1] for k in acc if k[1].startswith("partial_")})
    cells = sorted({(k[1], k[2], k[3]) for k in acc if k[1].startswith("partial_")})
    res = dict(status=("C2 confirmatory (plan Section 5)" if len(reps) > 1 else "C1 descriptive; replication 0 inspected"),
               reps=list(reps), per_rep={}, searches={})
    for r in reps:
        rows = {}
        print(f"\nreplication {r}: search minus Stage 4 arm, points, mean over {FIELDS} fields (+- standard error)")
        print(f"  {'arm':12s} {'rate':>6s} {'budget':>7s}" + "".join(f"{o:>16s}" for o in STAGE4))
        for arm, b, p in cells:
            me = np.array([acc[(r, arm, b, p, f)] for f in range(FIELDS)])
            row = dict(mean_accuracy=float(me.mean()))
            for o in STAGE4:
                d = 100 * (me - np.array([acc[(r, o, b, p, f)] for f in range(FIELDS)]))
                row[f"minus_{o}"] = dict(mean=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(FIELDS)))
            rows[f"{arm}@{b:g}@{p:g}"] = row
            print(f"  {arm:12s} {p:6g} {b:7g}" + "".join(
                f"{row[f'minus_{o}']['mean']:+8.2f}+-{row[f'minus_{o}']['se']:5.2f}" for o in STAGE4))
        for arm in arms:
            x = [v["minus_abs1024"] for k, v in rows.items() if k.startswith(arm + "@")]
            rule = dict(behind_abs1024_by_2se=int(sum(d["mean"] < -2 * d["se"] for d in x)),
                        ahead_abs1024_by_2se=int(sum(d["mean"] > 2 * d["se"] for d in x)), cells=len(x),
                        mean=float(np.mean([d["mean"] for d in x])), worst=float(min(d["mean"] for d in x)),
                        best=float(max(d["mean"] for d in x)))
            rows[f"{arm} rule"] = rule
            print(f"  {arm}: {json.dumps(rule)}")
        res["per_rep"][str(r)] = rows

    # search records: order, saturation, bits differing from abs1024
    for d in sorted(SEARCH.glob("rep*_mc_*")):
        r = int(d.name.split("_")[0][3:])
        if r not in reps:
            continue
        rounds = json.loads((d / "rounds.json").read_text())
        meta = json.loads((d / "DONE").read_text())
        sat = None
        order = []
        for x in rounds:
            c = next(y for y in x["candidates"] if y["uid"] == x["accepted"])
            order.append(f"{NAMES[c['layer']]}.b{c['bit']}.{c['part']}")
            if sat is None and c["gain"] < c["se"]:
                sat = dict(round=x["index"], protected=x["protected"])
        rec = dict(rounds=len(rounds), saturation=sat, passes=meta.get("passes"), seconds=meta.get("seconds"),
                   differing_from_abs1024=meta.get("differing_from_abs1024"), order=order)
        res["searches"][d.name] = rec
        print(f"\n  {d.name}: {len(rounds)} rounds, saturation {sat}, differing from abs1024 "
              f"{meta.get('differing_from_abs1024')}\n    first 40: {' '.join(order[:40])}")

    if len(reps) > 1:
        conf = {}
        for arm in arms:
            keys = sorted({(b, p) for a, b, p in cells if a == arm}, key=lambda t: (t[1], t[0]))
            for other in ("abs1024", "msb", "fkeras", "ra_analytic"):
                stats = []
                for b, p in keys:
                    m_r = np.array([100 * np.mean([acc[(r, arm, b, p, f)] - acc[(r, other, b, p, f)] for f in range(FIELDS)])
                                    for r in reps])
                    n = len(m_r)
                    mean, se = float(m_r.mean()), float(m_r.std(ddof=1) / np.sqrt(n))
                    t = mean / se if se > 0 else (np.inf if mean else 0.0)
                    pval = t_two_sided(abs(t), n - 1) if np.isfinite(t) else 0.0
                    q = t_quantile(0.95, n - 1)
                    stats.append(dict(rate=p, budget=b, rep_means=[float(v) for v in m_r], mean=mean, se=se, t=float(t),
                                      p=float(pval), ci90=[mean - q * se, mean + q * se],
                                      equivalent_within_1=bool(abs(mean) + q * se < 1.0)))
                adj = holm(np.array([s["p"] for s in stats]))
                for s, a_ in zip(stats, adj):
                    s["p_holm"] = float(a_)
                    s["resolved"] = "ahead" if (a_ < 0.05 and s["mean"] > 0) else "behind" if (a_ < 0.05 and s["mean"] < 0) else "unresolved"
                conf[f"{arm} minus {other}"] = dict(role="primary" if other == "abs1024" else "secondary, descriptive",
                                                    cells=stats)
                print(f"\n  C2 {arm} minus {other} ({conf[f'{arm} minus {other}']['role']}): n = {len(reps)} replications")
                for s in stats:
                    print(f"    rate {s['rate']:<6g} budget {s['budget']:<6g} {s['mean']:+6.2f} +- {s['se']:4.2f}  "
                          f"Holm p {s['p_holm']:.3f}  {s['resolved']:10s} 90% [{s['ci90'][0]:+.2f}, {s['ci90'][1]:+.2f}]"
                          f"{'  within +-1' if s['equivalent_within_1'] else ''}")
        res["confirmatory"] = conf
    out.mkdir(parents=True, exist_ok=False)
    (out / "compare.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--reps", type=int, nargs="+", required=True)
    a = ap.parse_args()
    main(a.out, a.reps)
