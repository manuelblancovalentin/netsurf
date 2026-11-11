#!/usr/bin/env python3
"""Summarize Run K analyses next to the pinned checkpoint (Run A): pre-set break criterion, regrets, tolerated BER."""
import collections
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
f = float
sources = {"pinned (Run A, windows 1-9)": ROOT / "publications/paper1_review/data/tecs_runA"}
for name in sys.argv[1:]:
    sources[name] = ROOT / f"workspace/paper1_tecs_revision/runK_{name}_analysis_v1"
RANK = ["abs_grad", "global_grad", "pos_grad", "msb", "fkeras", "jump", "random"]
out = {}
for label, d in sources.items():
    summ = json.loads((d / "summary.json").read_text())["models"]["resnet8"]
    cells = [r for r in csv.DictReader(open(d / "fresh_cell_quality.csv")) if r["model"] == "resnet8"]
    con = [r for r in csv.DictReader(open(d / "absgrad_contrasts.csv")) if r["model"] == "resnet8" and r["risk"] == "err" and r["competitor_set"] == "primary"]
    useful = collections.defaultdict(dict)
    for r in cells:
        if r["decision_useful_primary"] == "1":
            useful[(r["ber"], r["coverage"])][r["method"]] = f(r["mean_accuracy"])
    regret = {m: max(max(v.values()) - v[m] for v in useful.values()) for m in RANK}
    breaks = [dict(ber=r["ber"], coverage=r["coverage"], competitor=r["competitor"], diff=f(r["field_mean_diff"]), p_holm=f(r["p_holm_field"]))
              for r in con if r["decision_useful_primary"] == "1" and f(r["field_mean_diff"]) >= 0.01 and f(r["p_holm_field"]) < 0.05]
    wins = {m: sum(1 for r in con if r["competitor"] == m and r["decision_useful_primary"] == "1" and f(r["p_holm_field"]) < 0.05 and f(r["field_mean_diff"]) < 0) for m in RANK[1:]}
    losses = {m: sum(1 for r in con if r["competitor"] == m and r["decision_useful_primary"] == "1" and f(r["p_holm_field"]) < 0.05 and f(r["field_mean_diff"]) > 0) for m in RANK[1:]}
    best_counts = collections.Counter(max(v, key=v.get) for v in useful.values())
    rule = summ["decision_rule"]["primary/decision_useful_primary"]
    rvb = summ["regret_vs_best_other_ranking"]["decision_useful_primary"]
    clean = sum(summ["clean_accuracy"].values()) / len(summ["clean_accuracy"])
    hu = summ["highest_useful_ber"]
    drop = summ["accuracy_drop_fresh"]
    out[label] = dict(clean=clean, replications=len(summ["fresh_replications"]), useful_cells=len(useful), regret=regret, breaks=breaks,
                      wins_over=wins, losses_to=losses, best_ranking_counts=dict(best_counts),
                      rule_max_rep_upper=rule["max_replication_upper_bound"], rule_cells_ok=rule["cells_replication_upper_below_bound"],
                      vs_best_other_max_rep_upper=rvb["max_rep_upper"], vs_best_other_cells_ok=rvb["cells_rep_upper_below_bound"],
                      highest_useful_125={m: hu[f"0.125/{m}"] for m in ["unprotected", "random", "msb", "fkeras", "global_grad", "abs_grad", "pos_grad"]},
                      highest_useful_25={m: hu[f"0.25/{m}"] for m in ["unprotected", "random", "msb", "fkeras", "global_grad", "abs_grad", "pos_grad"]},
                      drop_125_at={b: {m: drop[f"0.125/{b}/{m}"] for m in ["unprotected", "msb", "abs_grad", "global_grad", "pos_grad"]} for b in ("0.001", "0.003", "0.01")})
print(json.dumps(out, indent=1))
(ROOT / "workspace/paper1_tecs_revision/runK_summary.json").write_text(json.dumps(out, indent=1))
