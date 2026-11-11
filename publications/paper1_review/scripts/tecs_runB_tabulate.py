#!/usr/bin/env python3
"""Offline tabulation of Run B (cap-statistics replay of the language-model grids).

Reads only the write-once observation JSONs of workspace/runs/w8a8_cap_replay_v1/<target>/ and the
tracked full-grid audit table (for the usefulness flags).  Needs no inference.  Writes a new directory.

Per (BER, budget, ranking): number of fields, bitwise replay matches, failed units, the fraction of
finite token losses at or above the 30-nat cap (mean and maximum over fields), and the mean token NLL
with and without the cap.  Per decision-useful cell: whether the best ranking and the regret of the
absolute-gradient mask change when the uncapped mean is used instead of the capped one.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
REPLAY = ROOT / "workspace/runs/w8a8_cap_replay_v1"
GRID = ROOT / "publications/paper1_review/data/tecs_audit/conditions_with_quality.csv"
TARGETS = {"pythia": "pythia160m_int8", "qwen": "qwen3_0_6b_int8"}
ALIAS = {"msb_first": "msb", "random": "random", "numerical_bit_jump": "jump", "global_first_order": "global_grad",
         "per_example_absolute": "abs_grad", "per_example_positive": "pos_grad"}
RANKINGS = ["random", "msb", "jump", "global_grad", "abs_grad", "pos_grad"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--models", nargs="+", default=list(TARGETS))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    useful = {}
    for r in csv.DictReader(open(GRID)):
        useful[(r["model"], float(r["ber"]), float(r["coverage"]), r["method"])] = (int(r["useful_primary"]), float(r["mean_loss"]), int(r["fields"]))
    rows, summary, prov = [], dict(schema="paper1-runB-tabulation/1.0", models={}), {}
    for model in args.models:
        root = REPLAY / TARGETS[model]
        plan = json.loads((root / "plan.json").read_text())
        expected = plan["counts"]["observations"]
        obs = defaultdict(list); listing = hashlib.sha256(); count = mismatch = 0; eligible = None
        for arm in sorted(os.listdir(root / "fields")):
            for field in sorted(os.listdir(root / "fields" / arm)):
                for name in sorted(os.listdir(root / "fields" / arm / field)):
                    if name == "field_done.json":
                        continue
                    o = json.loads((root / "fields" / arm / field / name).read_text())
                    count += 1; mismatch += int(not o["replay_bitwise_match"])
                    listing.update(f"{arm}/{field}/{name}:{o['payload_sha256']}\n".encode())
                    if o["policy"] == "unprotected":
                        key = ("unprotected", 0)
                    else:
                        meth, bits = o["policy"].rsplit("_b", 1); key = (ALIAS[meth], int(bits))
                    obs[(o["ber"], key)].append(o)
        assert count == expected, (model, count, expected)
        budgets = sorted({k[1][1] for k in obs if k[1][1]})
        eligible = budgets[-1] * 4                       # the largest budget is 25% of the eligible bits
        cov_of = {b: round(b / eligible, 4) for b in budgets}; cov_of[0] = 0.0
        cell = {}
        for (ber, (meth, bits)), items in sorted(obs.items()):
            d = [o["diagnostic"] for o in items]; cov = cov_of[bits]
            u = useful.get((model, ber, cov, meth))
            capped = float(np.mean([o["risk"]["metrics"]["mean_nll"] for o in items]))
            uncapped = float(np.mean([x["finite_nll_uncapped_mean"] for x in d]))
            if u is not None:
                assert abs(u[1] - capped) < 1e-9 and u[2] == len(items), (model, ber, cov, meth, u, capped, len(items))
            row = dict(model=model, ber=ber, coverage=cov, method=meth, fields=len(items),
                       bitwise_matches=int(sum(o["replay_bitwise_match"] for o in items)),
                       failed_units=int(sum(o["risk"]["failed_units"] for o in items)),
                       mean_cap_fraction=float(np.mean([x["cap_fraction_of_finite"] for x in d])),
                       max_cap_fraction=float(np.max([x["cap_fraction_of_finite"] for x in d])),
                       fields_with_capped_token=int(sum(x["finite_nll_at_or_above_30_count"] > 0 for x in d)),
                       capped_mean_nll=capped, uncapped_mean_nll=uncapped, uncapped_minus_capped=uncapped - capped,
                       mean_median_nll=float(np.mean([x["finite_nll_uncapped_median"] for x in d])),
                       mean_p95_nll=float(np.mean([x["finite_nll_uncapped_p95"] for x in d])),
                       max_token_nll=float(np.max([x["finite_nll_uncapped_max"] for x in d])),
                       useful_primary=(u[0] if u is not None else ""))
            rows.append(row); cell[(ber, cov, meth)] = row
        # effect of the cap on the comparison in decision-useful cells
        bers = sorted({k[0] for k in cell}); covs = sorted({k[1] for k in cell if k[1] > 0})
        dec = []; changed_best = 0; max_regret_shift = 0.0
        for ber in bers:
            for cov in covs:
                rs = {m: cell[(ber, cov, m)] for m in RANKINGS}
                if not any(r["useful_primary"] == 1 for r in rs.values()):
                    continue
                best_c = min(rs, key=lambda m: rs[m]["capped_mean_nll"]); best_u = min(rs, key=lambda m: rs[m]["uncapped_mean_nll"])
                reg_c = rs["abs_grad"]["capped_mean_nll"] - rs[best_c]["capped_mean_nll"]
                reg_u = rs["abs_grad"]["uncapped_mean_nll"] - rs[best_u]["uncapped_mean_nll"]
                changed_best += int(best_c != best_u); max_regret_shift = max(max_regret_shift, abs(reg_u - reg_c))
                dec.append(dict(ber=ber, coverage=cov, best_capped=best_c, best_uncapped=best_u, abs_regret_capped=reg_c, abs_regret_uncapped=reg_u,
                                max_cap_fraction_useful_masks=max(r["max_cap_fraction"] for r in rs.values() if r["useful_primary"] == 1),
                                max_cap_fraction_any_mask=max(r["max_cap_fraction"] for r in rs.values())))
        mine = [r for r in rows if r["model"] == model]
        use = [r for r in mine if r["useful_primary"] == 1]
        worst = max(mine, key=lambda r: r["mean_cap_fraction"])
        summary["models"][model] = dict(
            observations=count, bitwise_mismatches=mismatch, failed_units=int(sum(r["failed_units"] for r in mine)),
            conditions=len(mine), conditions_with_any_capped_token=int(sum(r["fields_with_capped_token"] > 0 for r in mine)),
            useful_conditions=len(use), useful_conditions_with_any_capped_token=int(sum(r["fields_with_capped_token"] > 0 for r in use)),
            max_field_cap_fraction_in_useful_conditions=max(r["max_cap_fraction"] for r in use),
            max_uncapped_minus_capped_in_useful_conditions=max(r["uncapped_minus_capped"] for r in use),
            max_mean_cap_fraction=dict(value=worst["mean_cap_fraction"], ber=worst["ber"], coverage=worst["coverage"], method=worst["method"],
                                       capped_mean_nll=worst["capped_mean_nll"], uncapped_mean_nll=worst["uncapped_mean_nll"]),
            max_uncapped_minus_capped=max(r["uncapped_minus_capped"] for r in mine), max_token_nll=max(r["max_token_nll"] for r in mine),
            decision_useful_cells=len(dec), decision_useful_cells_where_best_ranking_changes_without_cap=changed_best,
            max_abs_change_of_absgrad_regret_without_cap=max_regret_shift,
            max_cap_fraction_any_mask_in_decision_useful_cells=max(c["max_cap_fraction_any_mask"] for c in dec),
            decision_useful_cells_detail=dec)
        prov[model] = dict(replay_dir=str(root), replay_plan_sha256=hashlib.sha256((root / "plan.json").read_bytes()).hexdigest(),
                           observation_listing_sha256=listing.hexdigest(), observations=count)
    with open(args.out / "cap_statistics.csv", "x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(args.out / "summary.json", "x") as f:
        json.dump(summary, f, indent=1)
    prov["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    prov["outputs"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.out.iterdir())}
    with open(args.out / "provenance.json", "x") as f:
        json.dump(prov, f, indent=1)
    for m, v in summary["models"].items():
        print(m, json.dumps({k: x for k, x in v.items() if k != "decision_useful_cells_detail"}))


if __name__ == "__main__":
    main()
