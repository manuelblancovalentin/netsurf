#!/usr/bin/env python3
"""Summary of runT_ranger.py: Ranger at the pooled features against no Ranger, on the same fault fields.

Per checkpoint, rate and condition: mean accuracy over fields of the recomputed classifier without and with the clip,
the paired difference with a 95% interval, and whether any condition's order changes.  Also checks that every TFLite
accuracy equals the Run T accuracy of the same field.

    runT_ranger_analyze.py --runs DIR --runT DIR --out DIR
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--runT", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    files = sorted(a.runs.glob("*/rep*/draw*/results.csv"))
    rows = [r for f in files for r in csv.DictReader(open(f))]
    mismatch = 0
    for r in rows:
        ref = {(float(x["ber"]), x["condition"]): float(x["accuracy"])
               for x in csv.DictReader(open(a.runT / r["checkpoint"] / f"rep{r['rep']}" / f"draw{r['draw']}" / "results.csv"))}
        mismatch += abs(ref[(float(r["ber"]), r["condition"])] - float(r["accuracy_tflite"])) > 1e-12
    by = defaultdict(list)
    for r in rows:
        by[(r["checkpoint"], float(r["ber"]), r["condition"])].append((100 * float(r["accuracy_recomputed"]), 100 * float(r["accuracy_ranger"])))
    out, order_changes = [], 0
    for (cp, p, c), v in sorted(by.items()):
        v = np.asarray(v); d = v[:, 1] - v[:, 0]
        h = 1.96 * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
        out.append(dict(checkpoint=cp, ber=p, condition=c, fields=len(d), without=v[:, 0].mean(), with_ranger=v[:, 1].mean(),
                        diff=d.mean(), lo=d.mean() - h, hi=d.mean() + h))
    for cp in {o["checkpoint"] for o in out}:
        for p in {o["ber"] for o in out}:
            sel = [o for o in out if o["checkpoint"] == cp and o["ber"] == p]
            a0 = [o["condition"] for o in sorted(sel, key=lambda o: -o["without"])]
            a1 = [o["condition"] for o in sorted(sel, key=lambda o: -o["with_ranger"])]
            order_changes += a0 != a1
    a.out.mkdir(parents=True, exist_ok=False)
    with open(a.out / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    summ = dict(rows=len(rows), tflite_mismatches_vs_runT=int(mismatch),
                max_abs_diff=float(max(abs(o["diff"]) for o in out)),
                max_abs_diff_up_to_3e-3=float(max(abs(o["diff"]) for o in out if o["ber"] <= 3e-3)),
                rate_checkpoint_cells_with_order_change=int(order_changes),
                inputs_sha256=hashlib.sha256(b"".join(f.read_bytes() for f in files)).hexdigest(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (a.out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
