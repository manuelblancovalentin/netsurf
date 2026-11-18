#!/usr/bin/env python3
"""Run F3 (exploratory; frozen after Runs F and F2 were read and before any F3 outcome existed).

Runs F and F2 found that the absolute-gradient mask rebuilt under faults ("conditional absolute gradient")
beats the fixed absolute-gradient mask at p = 0.1 for 12.5% and 18.75% protection on DistilBERT, and not at
p = 0.03.  The study grid has no fault rate between 0.03 and 0.1, so it cannot say how far the tolerated
fault rate moves.  F3 fills that gap with the paper's own usefulness measure.

Design.  Fault rates p in {0.04, 0.05, 0.065, 0.08}; budgets 12.5% and 18.75%; eight cells.  For each cell
the candidate is built for that p with the unchanged Run F builder (two rounds, four fault fields per round,
64 calibration sentences, gates included; new seed base).  Each candidate and the sealed fixed mask of the
same budget are evaluated at that p on 24 new fault fields (seeds unused elsewhere), exact integer
executable, the study's frozen 512-sentence cohort.  Paired gain per cell with a 95% interval.
Reported per budget: the highest of these fault rates at which each mask keeps mean accuracy within 0.05
of clean (the paper's usefulness criterion), together with the existing grid points 0.03 and 0.1.
One model, one checkpoint, a cohort used before.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fault_aware_masks as fa      # noqa: E402
import fault_aware_masks_f2 as f2   # noqa: E402

RATES = (0.04, 0.05, 0.065, 0.08)
FRACS = (0.125, 0.1875)
CELLS = tuple((p, f) for p in RATES for f in FRACS)
FIELDS, FIELD_SEED = 24, 9832000


def patch():
    fa.TARGET_BERS = RATES; fa.FRACTIONS = FRACS; fa.CANDIDATES = (f2.CAND,); fa.SEED = 20261115


def make_fields(out: Path) -> None:
    fa.imports()
    from netsurf.w8a8_dense_ber_pilot import packed_field
    plan = json.loads((fa.SOURCE_RUN / "plan.json").read_text())
    used = {s for arm in plan["field_seeds"] for s in arm}
    for a, p in enumerate(RATES):
        for i in range(FIELDS):
            seed = FIELD_SEED + 1000 * a + i
            assert seed not in used
            d = out / "fields" / f"rate_{a}" / f"field_{i:03d}"
            d.mkdir(parents=True, exist_ok=False)
            fa.write_once(d / "field.json", dict(packed_field(d / "proposed.u8", plan["eligible_bits"], p, seed), ber=p, seed=seed))
    print("fields done", flush=True)


def evaluate(out: Path, shard: int, shards: int) -> None:
    wide, _evaluation, _risk_record, Protection, injected_packed, prepared, Adapter, PackedMask = fa.imports()
    plan, asset = wide.validate(fa.MANIFEST, fa.SOURCE_RUN, verify_masks=False)
    bits = plan["eligible_bits"]
    adapter = Adapter(asset); adapter.load()
    run = _evaluation(adapter, {"target": plan["target"], "final": plan["final_cohort"]})
    clean = json.loads((fa.SOURCE_RUN / "clean/shard_00.json").read_text())
    assert _risk_record(run()) == clean
    masks = {}
    for p, frac in CELLS:
        budget = int(round(frac * bits))
        item = json.loads((fa.cell_dir(out, p, frac) / "manifest.json").read_text())["masks"][f2.CAND]
        cand = f2.protection(Protection, PackedMask, bits, item["path"], item["budget"], item["sha256"])
        fixed = wide._protection(next(r for r in plan["policies"] if r["name"] == f"per_example_absolute_b{budget}"), bits)
        masks[(p, frac)] = (("candidate", cand, prepared(cand)), ("fixed_absolute", fixed, prepared(fixed)))
    for i in range(shard, FIELDS, shards):
        for a, p in enumerate(RATES):
            d = out / "fields" / f"rate_{a}" / f"field_{i:03d}"
            meta = json.loads((d / "field.json").read_text()); proposed = np.memmap(d / "proposed.u8", dtype=np.uint8, mode="r")
            for frac in FRACS:
                for name, prot, pk in masks[(p, frac)]:
                    target = out / "eval" / f"rate_{a}" / f"field_{i:03d}" / f"{name}_f{frac:g}.json"
                    if target.exists():
                        continue
                    with injected_packed(adapter, proposed, meta, prot, pk) as events:
                        risk = run()
                    fa.write_once(target, dict(mask=name, ber=p, fraction=frac, field_index=i, field_sha256=meta["sha256"],
                                               events=dict(events), risk=_risk_record(risk)))
            del proposed
            print(json.dumps(dict(shard=shard, field=i, ber=p)), flush=True)
    assert _risk_record(run()) == clean, "model not restored"
    print("done", flush=True)


def analyze(out: Path) -> None:
    import math
    from scipy import stats
    clean = json.loads((fa.SOURCE_RUN / "clean/shard_00.json").read_text())["metrics"]["accuracy"]
    acc = lambda path: json.loads(Path(path).read_text())["risk"]["metrics"]["accuracy"]
    rows = []
    for a, p in enumerate(RATES):
        for frac in FRACS:
            c = np.array([acc(out / "eval" / f"rate_{a}" / f"field_{i:03d}" / f"candidate_f{frac:g}.json") for i in range(FIELDS)])
            f = np.array([acc(out / "eval" / f"rate_{a}" / f"field_{i:03d}" / f"fixed_absolute_f{frac:g}.json") for i in range(FIELDS)])
            d = c - f; h = stats.t.ppf(0.975, FIELDS - 1) * d.std(ddof=1) / math.sqrt(FIELDS)
            rows.append(dict(ber=p, fraction=frac, clean=clean, fixed_absolute=float(f.mean()), candidate=float(c.mean()),
                             gain=float(d.mean()), gain_ci=[float(d.mean() - h), float(d.mean() + h)],
                             fixed_useful=bool(clean - f.mean() <= 0.05), candidate_useful=bool(clean - c.mean() <= 0.05)))
    with open(Path(str(out) + "_analysis.json"), "x") as fobj:
        json.dump(dict(schema="paper1-runF3-analysis/1.0", status="EXPLORATORY", rows=rows), fobj, indent=1)
    for r in rows:
        print(f"p={r['ber']:<6g} B={r['fraction']:<7g} fixed {100*r['fixed_absolute']:.1f} ({'useful' if r['fixed_useful'] else 'not useful'})  "
              f"rebuilt {100*r['candidate']:.1f} ({'useful' if r['candidate_useful'] else 'not useful'})  gain {100*r['gain']:+.2f} [{100*r['gain_ci'][0]:+.2f},{100*r['gain_ci'][1]:+.2f}]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "fields", "eval", "analyze"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cell", type=int, default=0); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--shards", type=int, default=1)
    args = ap.parse_args()
    if args.cmd == "build":
        patch(); fa.build(args.out, *CELLS[args.cell])
    elif args.cmd == "fields":
        make_fields(args.out)
    elif args.cmd == "eval":
        evaluate(args.out, args.shard, args.shards)
    else:
        analyze(args.out)


if __name__ == "__main__":
    main()
