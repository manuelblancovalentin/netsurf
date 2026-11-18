#!/usr/bin/env python3
"""Run F2 (follow-up to Run F, frozen after Run F was read and before any F2 outcome existed).

Run F result that motivates it: the "conditional absolute gradient" mask built for p = 0.1 at 12.5% gained
9.3 accuracy points over the fixed absolute-gradient mask at p = 0.1 on the study's saved fields and cohort,
and changed nothing in the five target cells where the fixed mask was already within two points of clean.
The second candidate (two-moment benefit) failed in every cell and is dropped.

Part A, confirmation (primary).  The same two masks, unchanged (Run F mask for p = 0.1, 12.5%; sealed fixed
absolute-gradient mask at 12.5%), plus no protection, on inputs and faults that neither the wide-BER study
nor Run F used: the 328 validation sentences after the frozen 512-sentence cohort (source indices 544-871)
and 24 new fault fields per BER at p in {0.01, 0.03, 0.1}.  Whether earlier pilots touched these sentences
was not audited.  Primary contrast: candidate minus fixed mask at p = 0.1, paired over the 24 fields.
Confirmed if the gain is at least one accuracy point and its 95% interval excludes zero.  The contrasts at
0.01 and 0.03 are secondary (cost of using the p = 0.1 mask at lower fault rates).

Part B, extension (exploratory).  Rule fixed here: build the same candidate for every remaining DistilBERT
cell in which the fixed absolute-gradient mask loses more than two points to clean in the main grid:
(p, B) = (0.03, 6.25%), (0.1, 6.25%), (0.1, 18.75%).  Evaluate on the saved fields and frozen cohort at
p in {0.01, 0.03, 0.1}, paired with the finalized fixed-ranking outcomes.  Same builder, same gates, a new
seed base; only the candidate list and the cells differ from Run F.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fault_aware_masks as fa  # noqa: E402

RUN_F = Path(__file__).resolve().parents[1] / "workspace/runs/paper1_runF_fault_aware_masks_v1"
CELLS_B = ((0.03, 0.0625), (0.1, 0.0625), (0.1, 0.1875))
TEST_BERS = (0.01, 0.03, 0.1)
FIELDS = 24
CONFIRM_SEED = 9732000
CAND = "c1_conditional_absolute"


def patch():
    fa.TARGET_BERS = (0.03, 0.1); fa.FRACTIONS = (0.0625, 0.1875); fa.CANDIDATES = (CAND,); fa.SEED = 20261114


def protection(Protection, PackedMask, bits, path, budget, file_sha):
    return Protection("packed", bits, budget, PackedMask(Path(path), bits, budget, file_sha).digest, Path(path), file_sha)


def make_fields(out: Path) -> None:
    fa.imports()
    from netsurf.w8a8_dense_ber_pilot import packed_field
    plan = json.loads((fa.SOURCE_RUN / "plan.json").read_text())
    used = {s for arm in plan["field_seeds"] for s in arm}
    for a, ber in enumerate(TEST_BERS):
        for i in range(FIELDS):
            seed = CONFIRM_SEED + 1000 * a + i
            assert seed not in used
            d = out / "confirm_fields" / f"ber_{a}" / f"field_{i:03d}"
            d.mkdir(parents=True, exist_ok=False)
            meta = packed_field(d / "proposed.u8", plan["eligible_bits"], ber, seed)
            fa.write_once(d / "field.json", dict(meta, ber=ber, seed=seed))
    print("fields done", flush=True)


def eval_confirm(out: Path, shard: int, shards: int) -> None:
    wide, _evaluation, _risk_record, Protection, injected_packed, prepared, Adapter, PackedMask = fa.imports()
    plan, asset = wide.validate(fa.MANIFEST, fa.SOURCE_RUN, verify_masks=False)
    bits = plan["eligible_bits"]; budget = int(round(0.125 * bits))
    adapter = Adapter(asset); adapter.load()
    rows = adapter.sst2_cohorts()["evaluation"][512:]
    assert len(rows) == 328 and [r.source_index for r in rows] == list(range(544, 872))
    assert not set(r.source_index for r in rows) & set(plan["final_cohort"]["source_indices"])
    item = json.loads((fa.cell_dir(RUN_F, 0.1, 0.125) / "manifest.json").read_text())["masks"][CAND]
    ref = next(r for r in plan["policies"] if r["name"] == f"per_example_absolute_b{budget}")
    none = next(r for r in plan["policies"] if r["kind"] == "none")
    masks = [("candidate_p0.1", protection(Protection, PackedMask, bits, item["path"], item["budget"], item["sha256"])),
             ("fixed_absolute", wide._protection(ref, bits)), ("unprotected", wide._protection(none, bits))]
    packed = {n: prepared(p) for n, p in masks}
    clean = _risk_record(adapter.evaluate_sst2(rows))
    if shard == 0:
        fa.write_once(out / "confirm_eval" / "clean.json", clean)
    for i in range(shard, FIELDS, shards):
        for a, ber in enumerate(TEST_BERS):
            d = out / "confirm_fields" / f"ber_{a}" / f"field_{i:03d}"
            meta = json.loads((d / "field.json").read_text()); proposed = np.memmap(d / "proposed.u8", dtype=np.uint8, mode="r")
            for name, prot in masks:
                target = out / "confirm_eval" / f"ber_{a}" / f"field_{i:03d}" / f"{name}.json"
                if target.exists():
                    continue
                with injected_packed(adapter, proposed, meta, prot, packed[name]) as events:
                    risk = adapter.evaluate_sst2(rows)
                fa.write_once(target, dict(mask=name, test_ber=ber, field_index=i, field_sha256=meta["sha256"], events=dict(events),
                                           risk=_risk_record(risk), sentences=len(rows)))
            del proposed
            print(json.dumps(dict(shard=shard, field=i, ber=ber)), flush=True)
    assert _risk_record(adapter.evaluate_sst2(rows)) == clean, "model not restored"
    print("done", flush=True)


def eval_saved(out: Path, shard: int, shards: int) -> None:
    wide, _evaluation, _risk_record, Protection, injected_packed, prepared, Adapter, PackedMask = fa.imports()
    plan, asset = wide.validate(fa.MANIFEST, fa.SOURCE_RUN, verify_masks=False)
    bits = plan["eligible_bits"]
    adapter = Adapter(asset); adapter.load()
    run = _evaluation(adapter, {"target": plan["target"], "final": plan["final_cohort"]})
    assert _risk_record(run()) == json.loads((fa.SOURCE_RUN / "clean/shard_00.json").read_text())
    masks = []
    for p, frac in CELLS_B:
        item = json.loads((fa.cell_dir(out, p, frac) / "manifest.json").read_text())["masks"][CAND]
        prot = protection(Protection, PackedMask, bits, item["path"], item["budget"], item["sha256"])
        masks.append((f"{CAND}_p{p:g}_f{frac:g}", p, frac, prot, prepared(prot)))
    for i in range(shard, len(plan["field_seeds"][0]), shards):
        for ber in TEST_BERS:
            arm = plan["bers"].index(ber); src = fa.SOURCE_RUN / "fields" / f"arm_{arm:02d}" / f"field_{i:03d}"
            meta = json.loads((src / "field.json").read_text()); proposed = np.memmap(src / "proposed.u8", dtype=np.uint8, mode="r")
            if i == shard and ber == TEST_BERS[0]:                      # gate: one finalized observation, bit for bit
                ref = next(r for r in plan["policies"] if r["name"] == f"per_example_absolute_b{int(round(0.125 * bits))}")
                rp = wide._protection(ref, bits)
                with injected_packed(adapter, proposed, meta, rp, prepared(rp)) as events:
                    risk = run()
                saved = json.loads((src / f"{ref['name']}.json").read_text())
                assert _risk_record(risk) == saved["risk"] and dict(events) == saved["events"], "replay gate failed"
            for name, p, frac, prot, pk in masks:
                target = out / "eval" / f"arm_{arm:02d}" / f"field_{i:03d}" / f"{name}.json"
                if target.exists():
                    continue
                with injected_packed(adapter, proposed, meta, prot, pk) as events:
                    risk = run()
                fa.write_once(target, dict(mask=name, target_ber=p, fraction=frac, budget=prot.budget, test_ber=ber, arm=arm, field_index=i,
                                           field_sha256=meta["sha256"], events=dict(events), risk=_risk_record(risk)))
            del proposed
            print(json.dumps(dict(shard=shard, field=i, ber=ber)), flush=True)
    print("done", flush=True)


def analyze(out: Path) -> None:
    from scipy import stats

    def ci(d):
        d = np.asarray(d, float); m = float(d.mean()); sd = float(d.std(ddof=1))
        h = stats.t.ppf(0.975, len(d) - 1) * sd / math.sqrt(len(d)) if sd else 0.0
        return [m, m - h, m + h]

    plan = json.loads((fa.SOURCE_RUN / "plan.json").read_text()); bits = plan["eligible_bits"]
    acc = lambda p: json.loads(Path(p).read_text())["risk"]["metrics"]["accuracy"]
    res = dict(schema="paper1-runF2-analysis/1.0", confirmation=[], extension=[])
    clean = json.loads((out / "confirm_eval/clean.json").read_text())["metrics"]["accuracy"]
    for a, ber in enumerate(TEST_BERS):
        v = {n: np.array([acc(out / "confirm_eval" / f"ber_{a}" / f"field_{i:03d}" / f"{n}.json") for i in range(FIELDS)]) for n in ("candidate_p0.1", "fixed_absolute", "unprotected")}
        g = ci(v["candidate_p0.1"] - v["fixed_absolute"])
        res["confirmation"].append(dict(test_ber=ber, clean_accuracy=clean, candidate=float(v["candidate_p0.1"].mean()), fixed_absolute=float(v["fixed_absolute"].mean()),
                                        unprotected=float(v["unprotected"].mean()), gain=g, primary=bool(ber == 0.1),
                                        confirmed=bool(ber == 0.1 and g[0] >= 0.01 and g[1] > 0)))
    for p, frac in CELLS_B:
        budget = int(round(frac * bits)); name = f"{CAND}_p{p:g}_f{frac:g}"
        for ber in TEST_BERS:
            arm = plan["bers"].index(ber); n = len(plan["field_seeds"][0])
            c = np.array([acc(out / "eval" / f"arm_{arm:02d}" / f"field_{i:03d}" / f"{name}.json") for i in range(n)])
            f = np.array([acc(fa.SOURCE_RUN / "fields" / f"arm_{arm:02d}" / f"field_{i:03d}" / f"per_example_absolute_b{budget}.json") for i in range(n)])
            res["extension"].append(dict(target_ber=p, fraction=frac, test_ber=ber, on_target=bool(ber == p), candidate=float(c.mean()), fixed_absolute=float(f.mean()), gain=ci(c - f)))
    with open(Path(str(out) + "_analysis.json"), "x") as fobj:
        json.dump(res, fobj, indent=1)
    for r in res["confirmation"]:
        print(f"CONFIRM p={r['test_ber']:<5g} clean {100*r['clean_accuracy']:.1f} unprot {100*r['unprotected']:.1f} fixed {100*r['fixed_absolute']:.1f} cand {100*r['candidate']:.1f} gain {100*r['gain'][0]:+.2f} [{100*r['gain'][1]:+.2f},{100*r['gain'][2]:+.2f}] confirmed={r['confirmed']}")
    for r in res["extension"]:
        print(f"EXTEND built p={r['target_ber']:<5g} B={r['fraction']:<7g} test {r['test_ber']:<5g} fixed {100*r['fixed_absolute']:.1f} cand {100*r['candidate']:.1f} gain {100*r['gain'][0]:+.2f} [{100*r['gain'][1]:+.2f},{100*r['gain'][2]:+.2f}]{' *' if r['on_target'] else ''}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "fields", "eval-confirm", "eval-saved", "analyze"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cell", type=int, default=0); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--shards", type=int, default=1)
    args = ap.parse_args()
    if args.cmd == "build":
        patch(); p, frac = CELLS_B[args.cell]; fa.build(args.out, p, frac)
    elif args.cmd == "fields":
        make_fields(args.out)
    elif args.cmd == "eval-confirm":
        eval_confirm(args.out, args.shard, args.shards)
    elif args.cmd == "eval-saved":
        eval_saved(args.out, args.shard, args.shards)
    else:
        analyze(args.out)


if __name__ == "__main__":
    main()
