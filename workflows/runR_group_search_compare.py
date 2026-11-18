#!/usr/bin/env python3
"""EXPLORATORY, descriptive: group-search Phase A masks against the six Stage 4 masks of replication 0 on the same 8
fault fields and 800 test clips (TFLite), paired by field; plus each search's accepted group order and its saturation
round (first round whose accepted gain is below one paired standard error).  Reads locked outputs only.

    runR_group_search_compare.py <out>
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from site_paths import WT  # noqa: E402

RUNS = WT / "paper1-tecs-revision-v1/workspace/runs"
SEARCH = RUNS / "paper1_group_search_kws_v1"
NEW = RUNS / "paper1_group_search_kws_eval_v1"
OLD = RUNS / "paper1_runR_stage4_kws_eval_v1/rep0"
STAGE4 = ("abs1024", "pos1024", "abs64", "msb", "fkeras", "ra_analytic")
NAMES = ["dw1", "dw2", "dw3", "dw4", "dense", "conv1", "pw1", "pw2", "pw3", "pw4"]   # layer order of the model


def main(out: Path):
    acc = {}
    for f in range(8):
        for r in csv.DictReader(open(OLD / f"field{f}/results.csv")):
            acc[(r["arm"], "", float(r["budget"]), float(r["ber"]), f)] = float(r["accuracy"])
        for r in csv.DictReader(open(NEW / f"field{f}/results.csv")):
            acc[(r["arm"], r["variant"], float(r["budget"]), float(r["ber"]), f)] = float(r["accuracy"])
    new_keys = sorted({k[:4] for k in acc if k[0].startswith("group_")})
    res = dict(status="EXPLORATORY, descriptive; Stage 4 outcomes already read", cells={}, searches={})
    print("paired difference, points (positive = search mask better); mean over 8 fields (+- standard error)")
    for arm, variant, b, p in new_keys:
        me = np.array([acc[(arm, variant, b, p, f)] for f in range(8)])
        means = {a: np.mean([acc[(a, "", b, p, f)] for f in range(8)]) for a in STAGE4}
        best = max(STAGE4[:5], key=means.get)
        row = dict(mean=float(me.mean()), best_fixed=best)
        for o in STAGE4 + ("best_fixed",):
            oa = best if o == "best_fixed" else o
            d = 100 * (me - np.array([acc[(oa, "", b, p, f)] for f in range(8)]))
            row[f"minus_{o}"] = dict(mean=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(8)))
        res["cells"][f"{arm}/{variant}@{b:g}@{p:g}"] = row
    for arm in sorted({k[0] for k in new_keys}):
        for variant in ("full", "whole"):
            print(f"\n  {arm} ({variant})        " + "  ".join(f"{o:>16}" for o in ("abs1024", "msb", "ra_analytic", "best_fixed")))
            for p in sorted({k[3] for k in new_keys}):
                for b in sorted({k[2] for k in new_keys}):
                    c = res["cells"].get(f"{arm}/{variant}@{b:g}@{p:g}")
                    if c:
                        print(f"    rate {p:<6g} budget {b:<6g}" + "  ".join(
                            f"{c[f'minus_{o}']['mean']:+7.2f}+-{c[f'minus_{o}']['se']:4.2f}" for o in ("abs1024", "msb", "ra_analytic", "best_fixed"))
                              + f"   (best fixed: {c['best_fixed']})")
    for d in sorted(SEARCH.glob("*_*")):
        rounds = json.loads((d / "rounds.json").read_text())
        meta = json.loads((d / "DONE").read_text())
        order = []
        sat = None
        for r in rounds:
            c = next(x for x in r["candidates"] if x["gid"] == r["accepted"])
            order.append(dict(layer=NAMES[c["layer"]], bit=c["bit"], gain=c["gain"], se=c["se"], protected=r["protected"]))
            if sat is None and c["gain"] < c["se"]:
                sat = dict(round=r["index"], protected=r["protected"])
        res["searches"][d.name] = dict(order=order, saturation=sat, passes=meta.get("passes"), seconds=meta.get("seconds"))
        print(f"\n  {d.name}: {len(order)} rounds, saturation {sat}; order: " +
              " ".join(f"{o['layer']}.b{o['bit']}" for o in order))
    summ = {}
    for arm in sorted({k[0] for k in new_keys}):
        for variant in ("full", "whole"):
            ds = [v for k, v in res["cells"].items() if k.startswith(f"{arm}/{variant}@")]
            for o in ("abs1024", "best_fixed", "ra_analytic"):
                x = [d[f"minus_{o}"] for d in ds]
                summ[f"{arm}/{variant} minus {o}"] = dict(
                    mean=float(np.mean([d["mean"] for d in x])), worst=float(min(d["mean"] for d in x)),
                    best=float(max(d["mean"] for d in x)), ahead_by_2se=int(sum(d["mean"] > 2 * d["se"] for d in x)),
                    behind_by_2se=int(sum(d["mean"] < -2 * d["se"] for d in x)), cells=len(x))
    res["summary"] = summ
    print("\n" + json.dumps(summ, indent=1))
    out.mkdir(parents=True, exist_ok=False)
    (out / "compare.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
