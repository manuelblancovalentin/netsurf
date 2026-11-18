#!/usr/bin/env python3
"""Synthetic check of runR_stage2_analyze.py: fake Stage 2 outputs with planted effects, the real frozen R-A5 choices
(training-side data, not an outcome), and assertions on every rule.  Writes only under the given scratch directory.

  python runR_stage2_analyze_synthetic.py SCRATCH_DIR
Planted: abs1024 1.0 point ahead of ra_analytic at (12.5%, 3e-3) on pinned only (R2a fails there, nowhere else);
pos1024 2 points behind at 1e-2 and abs1024 2 points behind at 1e-4 on every checkpoint (R2b holds); ra_mc 0.4 point
ahead of ra_analytic at 1e-3 on three checkpoints and equal elsewhere (R2c kept); v2 against static arms follows from
the choices.  Field noise is shared by all arms of a field (paired), arm noise 0.05 point.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

import runR_stage2_analyze as A
from runR_stage1 import NAMES, reps_for
from runR_stage2 import CROSS, FIELDS, FRACTIONS, RATES, STATIC


def build(root):
    rng = np.random.default_rng(7)
    mc_gain = set(NAMES[1:4])
    for name in NAMES:
        for rep in reps_for(name):
            for f in range(FIELDS):
                shared = {p: rng.normal(0, 0.003) for p in RATES}
                main, mc = [], []
                for b in FRACTIONS:
                    for p in RATES:
                        base = 0.88 - 2.0 * p + shared[p]
                        val = {a: base + rng.normal(0, 0.0005) for a in STATIC + ("ra_analytic",)}
                        if name == "pinned" and b == 0.125 and p == 3e-3:
                            val["abs1024"] += 0.010
                        if p == 1e-2:
                            val["pos1024"] -= 0.02
                        if p == 1e-4:
                            val["abs1024"] -= 0.02
                        val = {a: round(v, 3) for a, v in val.items()}
                        for a, v in val.items():
                            main.append(dict(checkpoint=name, ckpt=0, rep=rep, field=f, ber=p, budget=b, arm=a,
                                             built_for=p if a == "ra_analytic" else "", accuracy=v, cross_entropy=0,
                                             proposed=0, applied=0))
                        if name in CROSS:
                            for pa in RATES:
                                if pa != p:
                                    main.append(dict(checkpoint=name, ckpt=0, rep=rep, field=f, ber=p, budget=b,
                                                     arm="ra_analytic", built_for=pa, accuracy=base - 0.001,
                                                     cross_entropy=0, proposed=0, applied=0))
                        g = 0.004 if (name in mc_gain and p == 1e-3) else 0.0
                        mc.append(dict(checkpoint=name, ckpt=0, rep=rep, field=f, ber=p, budget=b, arm="ra_mc",
                                       built_for=p, accuracy=round(val["ra_analytic"] + g + rng.normal(0, 0.0002), 3),
                                       cross_entropy=0, proposed=0, applied=0))
                for part, rows in (("main", main), ("mc", mc)):
                    d = root / part / name / f"rep{rep}" / f"field{f}"
                    d.mkdir(parents=True)
                    with open(d / "results.csv", "w", newline="") as fh:
                        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
                    (d / "field.json").write_text(json.dumps(dict(clean_accuracy=0.88)))
                    (d / "DONE").write_text("{}")


if __name__ == "__main__":
    scratch = Path(sys.argv[1])
    build(scratch / "eval")
    si = A.main(scratch / "eval", scratch / "interim", interim=True)
    assert si["R2c"] is None and {(v["checkpoint"], v["arm"]) for v in si["R2a"]["violations"]} == {("pinned", "abs1024")}
    s = A.main(scratch / "eval", scratch / "analysis")
    assert not s["R2a"]["holds"]
    assert {(v["checkpoint"], v["arm"]) for v in s["R2a"]["violations"]} == {("pinned", "abs1024")}, s["R2a"]
    assert s["R2b"]["holds"] and s["R2b"]["abs1024_fails_on"] == 9 and s["R2b"]["pos1024_fails_on"] == 9, s["R2b"]
    assert s["R2c"]["kept"] and s["R2c"]["checkpoints_with_resolved_gain"] == 3, s["R2c"]
    assert all(v["usable_cells"] == 28 for v in s["per_checkpoint"].values())   # 3e-2 is below clean - 0.05 by design
    rows = list(csv.DictReader(open(scratch / "analysis" / "cross_rate.csv")))
    assert len(rows) == 2 * 8 * 7 and sum(r["mean_gap"] == "" for r in rows) == 2 * 7
    assert all(0 < float(r["mean_gap"]) < 0.6 for r in rows if r["mean_gap"])
    print("synthetic checks passed; R2a' violations:", [(v["checkpoint"], v["arm"], v["ber"]) for v in s["R2a_prime"]["violations"]])
