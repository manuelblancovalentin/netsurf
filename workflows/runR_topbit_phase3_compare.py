#!/usr/bin/env python3
"""EXPLORATORY, descriptive: the measured-damage masks (top-bit analysis Phase 3) against the Stage 4 masks of
replication 0, on the same 8 fault fields, 12 rates and 800 test clips (TFLite).  Paired by field.  Prints and writes
<out>/compare.json.  Reads locked outputs only."""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from site_paths import WT  # noqa: E402

RUNS = WT / "paper1-tecs-revision-v1/workspace/runs"
NEW = RUNS / "paper1_topbit_mask_eval_kws_v1"
OLD = RUNS / "paper1_runR_stage4_kws_eval_v1/rep0"
STAGE4 = ("abs1024", "pos1024", "abs64", "msb", "fkeras", "ra_analytic")


def main(out: Path):
    acc = {}
    for f in range(8):
        for path in (NEW / f"field{f}/results.csv", OLD / f"field{f}/results.csv"):
            for r in csv.DictReader(open(path)):
                acc[(r["arm"], float(r["budget"]), float(r["ber"]), f)] = float(r["accuracy"])
    budgets = sorted({k[1] for k in acc})
    rates = sorted({k[2] for k in acc})
    res = dict(status="EXPLORATORY, descriptive; Stage 4 outcomes already read", cells={})
    for arm in ("measured_loss", "measured_acc"):
        for b in budgets:
            for p in rates:
                me = np.array([acc[(arm, b, p, f)] for f in range(8)])
                row = dict(mean=float(me.mean()))
                best_static = max(STAGE4[:5], key=lambda a: np.mean([acc[(a, b, p, f)] for f in range(8)]))
                for other in STAGE4 + ("best_static",):
                    o = best_static if other == "best_static" else other
                    d = 100 * (me - np.array([acc[(o, b, p, f)] for f in range(8)]))
                    row[f"minus_{other}"] = dict(mean=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(8)),
                                                 wins=int((d > 0).sum()), losses=int((d < 0).sum()))
                row["best_static_arm"] = best_static
                res["cells"][f"{arm}@{b:g}@{p:g}"] = row
    print("mean paired difference, points (positive = measured-damage mask better), measured_loss / measured_acc")
    for other in ("abs1024", "pos1024", "msb", "fkeras", "ra_analytic", "best_static"):
        print(f"\n  against {other}")
        print("    rate   " + "  ".join(f"{b:>15g}" for b in budgets))
        for p in rates:
            cells = []
            for b in budgets:
                a1 = res["cells"][f"measured_loss@{b:g}@{p:g}"][f"minus_{other}"]
                a2 = res["cells"][f"measured_acc@{b:g}@{p:g}"][f"minus_{other}"]
                cells.append(f"{a1['mean']:+6.2f}/{a2['mean']:+6.2f}")
            print(f"    {p:<7g}" + "  ".join(f"{c:>15}" for c in cells))
    summ = {}
    for arm in ("measured_loss", "measured_acc"):
        for other in ("abs1024", "best_static", "ra_analytic"):
            ds = [res["cells"][f"{arm}@{b:g}@{p:g}"][f"minus_{other}"] for b in budgets for p in rates]
            summ[f"{arm}_minus_{other}"] = dict(mean=float(np.mean([d["mean"] for d in ds])),
                                                worst=float(min(d["mean"] for d in ds)), best=float(max(d["mean"] for d in ds)),
                                                cells_ahead_by_2se=int(sum(d["mean"] > 2 * d["se"] for d in ds)),
                                                cells_behind_by_2se=int(sum(d["mean"] < -2 * d["se"] for d in ds)), cells=len(ds))
    res["summary"] = summ
    print("\n" + json.dumps(summ, indent=1))
    out.mkdir(parents=True, exist_ok=False)
    (out / "compare.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
