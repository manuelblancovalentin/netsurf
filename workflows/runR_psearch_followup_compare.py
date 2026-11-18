#!/usr/bin/env python3
"""Partial-search follow-up, the pre-registered analysis (review/PARTIAL_SEARCH_FOLLOWUP_PREREG_2026-11-14.md).
Written before any follow-up run existed.  Reads locked outputs only.  Pure NumPy.

    runR_psearch_followup_compare.py <out> [--reps 1 2 3 4 5]

Arms partial_k32, partial_k128 and swap at rate 1e-2, four budgets, against the Stage 4 masks of the same
replications on the same 8 fault fields and 800-clip test window, paired by field.
Per replication: mean over fields of (arm - comparator), points.  Across replications: mean, standard error, two-sided
t test (n - 1 degrees of freedom), 90% t interval.
  Primary family: each arm minus abs1024, 3 arms x 4 budgets = 12 tests, Holm at 0.05.
  Selection-noise family: partial_k128 minus partial_k32, 4 tests, Holm at 0.05.
  Descriptive: minus msb, fkeras, ra_analytic; search records.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runR_partial_search_compare import FIELDS, NAMES, OLD, RUNS, holm, t_quantile, t_two_sided  # noqa: E402

RATE = 0.01
BUDGETS = (0.125, 0.1875, 0.25, 0.375)
ARMS = {"partial_k32": "paper1_psearch_d_k32", "partial_k128": "paper1_psearch_d_k128", "swap": "paper1_psearch_d_swap"}


def cell_stats(rep_means):
    m_r = np.asarray(rep_means, dtype=float)
    n = len(m_r)
    mean, se = float(m_r.mean()), float(m_r.std(ddof=1) / np.sqrt(n))
    t = mean / se if se > 0 else (np.inf if mean else 0.0)
    p = t_two_sided(abs(t), n - 1) if np.isfinite(t) else 0.0
    q = t_quantile(0.95, n - 1)
    return dict(rep_means=[float(v) for v in m_r], mean=mean, se=se, t=float(t), p=float(p),
                ci90=[mean - q * se, mean + q * se], equivalent_within_1=bool(abs(mean) + q * se < 1.0))


def family(stats: list) -> list:
    """Holm over the whole family; each entry gains p_holm and resolved (ahead / behind / unresolved)."""
    adj = holm(np.array([s["p"] for s in stats]))
    for s, a in zip(stats, adj):
        s["p_holm"] = float(a)
        s["resolved"] = "ahead" if (a < 0.05 and s["mean"] > 0) else "behind" if (a < 0.05 and s["mean"] < 0) else "unresolved"
    return stats


def load(reps):
    acc = {}
    roots = [OLD] + [RUNS / f"{v}_eval_v1" for v in ARMS.values()]
    for r in reps:
        for f in range(FIELDS):
            for root in roots:
                for row in csv.DictReader(open(root / f"rep{r}/field{f}/results.csv")):
                    acc[(r, row["arm"], float(row["budget"]), float(row["ber"]), f)] = float(row["accuracy"])
    return acc


def rep_means(acc, reps, arm, other, b):
    return [100 * np.mean([acc[(r, arm, b, RATE, f)] - acc[(r, other, b, RATE, f)] for f in range(FIELDS)]) for r in reps]


def records(reps):
    out = {}
    for arm, base in ARMS.items():
        for d in sorted((RUNS / f"{base}_v1").glob("rep*_mc_*")):
            if int(d.name.split("_")[0][3:]) not in reps:
                continue
            rounds = json.loads((d / "rounds.json").read_text())
            meta = json.loads((d / "DONE").read_text())
            rec = dict(arm=arm, rounds=len(rounds), passes=meta.get("passes"), seconds=meta.get("seconds"),
                       differing_from_abs1024=meta.get("differing_from_abs1024"))
            if arm == "swap":
                rec.update(accepted=meta["accepted"], moved_bits=[{"group": f"{NAMES[int(g) // 8]}.b{int(g) % 8}", "bits": v}
                                                                   for g, v in meta["moved_bits_by_group"].items()],
                           verify=[dict(out=x.get("out"), into=x.get("into"), mean=x.get("verify_mean"),
                                        se=x.get("verify_se"), accepted=x.get("accepted")) for x in rounds])
            else:
                sat = None
                for x in rounds:
                    c = next(y for y in x["candidates"] if y["uid"] == x["accepted"])
                    if c["gain"] < c["se"]:
                        sat = dict(round=x["index"], protected=x["protected"])
                        break
                rec["saturation"] = sat
            out[d.name] = rec
    return out


def main(out: Path, reps):
    acc = load(reps)
    res = dict(status="pre-registered follow-up analysis (review/PARTIAL_SEARCH_FOLLOWUP_PREREG_2026-11-14.md)",
               reps=list(reps), rate=RATE)
    prim = family([dict(arm=arm, budget=b, **cell_stats(rep_means(acc, reps, arm, "abs1024", b)))
                   for arm in ARMS for b in BUDGETS])
    noise = family([dict(budget=b, **cell_stats(rep_means(acc, reps, "partial_k128", "partial_k32", b))) for b in BUDGETS])
    desc = {f"{arm} minus {o}": [dict(budget=b, **cell_stats(rep_means(acc, reps, arm, o, b))) for b in BUDGETS]
            for arm in ARMS for o in ("msb", "fkeras", "ra_analytic")}
    res.update(primary_minus_abs1024=prim, selection_noise_k128_minus_k32=noise, descriptive=desc,
               searches=records(reps))
    for title, rows in (("primary: arm minus abs1024 (Holm over 12)", prim),
                        ("selection noise: partial_k128 minus partial_k32 (Holm over 4)", noise)):
        print(f"\n{title}, n = {len(reps)} replications, rate {RATE:g}")
        for s in rows:
            print(f"  {s.get('arm', ''):13s} budget {s['budget']:<6g} {s['mean']:+6.2f} +- {s['se']:4.2f}  "
                  f"Holm p {s['p_holm']:.3f}  {s['resolved']:10s} 90% [{s['ci90'][0]:+.2f}, {s['ci90'][1]:+.2f}]"
                  f"{'  within +-1' if s['equivalent_within_1'] else ''}")
    for k, rows in desc.items():
        print(f"  {k}: " + "  ".join(f"{s['budget']:g}: {s['mean']:+.2f}+-{s['se']:.2f}" for s in rows))
    for k, v in res["searches"].items():
        print(f"  {k}: rounds {v['rounds']}, differing {v['differing_from_abs1024']}"
              + (f", accepted {v['accepted']}" if v["arm"] == "swap" else f", saturation {v['saturation']}"))
    out.mkdir(parents=True, exist_ok=False)
    (out / "compare.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--reps", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    a = ap.parse_args()
    if 0 in a.reps:
        raise SystemExit("replication 0 was inspected before the pre-registration; it is not part of this analysis")
    main(a.out, a.reps)
