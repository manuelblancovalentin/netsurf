#!/usr/bin/env python3
"""Run P Phase 2 analysis (plan v2, Sections 4 and 4a; decision rules of plan v1 Section 6).

Pairs every new mask with the saved outcomes of the seven rankings on the same fault schedules (replication,
schedule hash), and evaluates the frozen decision rule for range W against the paper's absolute-gradient mask and,
separately, against the 512-image absolute-gradient mask.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runP_phase0 import CHECKPOINTS, RANKINGS, bundle_dir  # noqa: E402
from runP_phase12 import OUT  # noqa: E402

PRIMARY = ("undertrained_e100", "pruned_s07", "pinned")
SECONDARY = ("pruned_s05",) + tuple(f"retrained_seed{s}" for s in (7101, 7102, 7103, 7104, 7105))
ALIAS = {v: k for k, v in RANKINGS.items()}
NEW = ("abs512", "squared", "mv_range", "mv_single")
RNG = np.random.default_rng(20261113)
SIGNS = None


def signflip_p(d, n=100_000):
    global SIGNS
    d = np.asarray(d, dtype=float)
    if np.allclose(d, 0):
        return 1.0
    if SIGNS is None or SIGNS.shape[1] < d.size:
        SIGNS = RNG.choice((-1.0, 1.0), size=(n, 64))
    obs = abs(d.mean())
    null = np.abs((SIGNS[:, : d.size] * d).mean(axis=1))
    return float((1 + (null >= obs - 1e-15).sum()) / (n + 1))


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(ps) - rank) * ps[i])
        adj[i] = min(1.0, running)
    return adj


def load(name):
    """acc[(ber, budget)][arm][(rep, sha)] = accuracy; saved rankings, unprotected and new arms."""
    acc = defaultdict(lambda: defaultdict(dict))
    reps = [int(p.name[3:]) for p in sorted((OUT / name).glob("rep*")) if (p / "DONE").is_file()]
    bundle = bundle_dir(CHECKPOINTS[name]["bundle"])
    with open(bundle / "measurements/trials.csv") as f:
        for r in csv.DictReader(f):
            rep = int(r["replication"].split("_")[1])
            if rep not in reps:
                continue
            ber_s, pol = r["policy"].split("/", 1)
            ber = float(ber_s.split("=")[1])
            if pol == "no_protection":
                for b in (0.0625, 0.125, 0.1875, 0.25):
                    acc[(ber, b)]["unprotected"][(rep, r["schedule_sha256"])] = float(r["accuracy"])
                continue
            if pol == "full_protection":
                continue
            name_, cov = pol.split("@")
            if name_ not in ALIAS:  # random-fill partial-tier policies of the pinned bundle: not one of the seven
                continue
            acc[(ber, float(cov))][ALIAS[name_]][(rep, r["schedule_sha256"])] = float(r["accuracy"])
    for rep in reps:
        with open(OUT / name / f"rep{rep}" / "replay.csv") as f:
            for r in csv.DictReader(f):
                if r["arm"] in NEW:
                    acc[(float(r["ber"]), float(r["budget"]))][r["arm"]][(rep, r["schedule_sha256"])] = float(r["accuracy"])
    return acc, reps


def useful_flags(name):
    flags = {}
    for r in csv.DictReader(open(CHECKPOINTS[name]["measured"])):
        if r.get("model", "resnet8") != "resnet8":  # the pinned (Run A) table also holds the DS-CNN rows
            continue
        flags[(float(f"{float(r['ber']):.3g}"), round(float(r["coverage"]), 6))] = (
            r["decision_useful_primary"] in ("1", "True"), float(r["mean_clean_accuracy"]))
    return flags


def contrast(acc_cell, arm, ref):
    keys = sorted(set(acc_cell[arm]) & set(acc_cell[ref]))
    d = np.array([acc_cell[arm][k] - acc_cell[ref][k] for k in keys])
    by_rep = defaultdict(list)
    for k, v in zip(keys, d):
        by_rep[k[0]].append(v)
    rm = np.array([np.mean(v) for v in by_rep.values()])
    h = stats.t.ppf(0.975, rm.size - 1) * rm.std(ddof=1) / math.sqrt(rm.size) if rm.size > 1 and rm.std() > 0 else 0.0
    return dict(diff=float(d.mean()), fields=int(d.size), p=signflip_p(d), rep_low=float(rm.mean() - h), rep_high=float(rm.mean() + h))


def phase1_check(name, reps):
    red = []
    for rep in reps:
        rec = {r["mask"]: r for r in json.loads((OUT / name / f"rep{rep}" / "phase1.json").read_text())["records"]}
        for b in ("0.062500", "0.125000", "0.187500", "0.250000"):
            red.append(rec[f"abs512@{b}"]["heldout"] - rec[f"mv_range@{b}"]["heldout"])
    return dict(median_heldout_reduction=float(np.median(red)), passed=bool(np.median(red) > 0))


def analyze(names):
    out, family = {}, {"abs_grad": [], "abs512": []}
    for name in names:
        acc, reps = load(name)
        flags = useful_flags(name)
        cells = []
        for (ber, b), arms in sorted(acc.items()):
            useful, clean = flags.get((float(f"{ber:.3g}"), round(b, 6)), (False, float("nan")))
            row = dict(ber=ber, budget=b, decision_useful=useful, clean=clean,
                       mean={a: float(np.mean(list(v.values()))) for a, v in arms.items()})
            for arm in NEW + ("pos_grad", "global_grad"):
                if arm in arms and arm != "abs_grad":
                    row[f"{arm}_vs_abs"] = contrast(arms, arm, "abs_grad")
            row["mv_range_vs_abs512"] = contrast(arms, "mv_range", "abs512")
            row["mv_single_vs_mv_range"] = contrast(arms, "mv_single", "mv_range")
            row["mv_range_vs_squared"] = contrast(arms, "mv_range", "squared")
            row["new_arm_makes_useful"] = (not useful) and any(
                row["mean"].get(a, -1) >= clean - 0.05 for a in NEW)
            best_old = max(v for a, v in row["mean"].items() if a in ALIAS.values())
            row["regret_new_best_vs_seven"] = {a: best_old - row["mean"][a] for a in NEW if a in row["mean"]}
            if name in PRIMARY:
                family["abs_grad"].append((name, len(cells), row["mv_range_vs_abs"]["p"]))
                family["abs512"].append((name, len(cells), row["mv_range_vs_abs512"]["p"]))
            cells.append(row)
        out[name] = dict(replications=reps, cells=cells, phase1_check=phase1_check(name, reps))
    for ref, fam in family.items():
        if not fam:
            continue
        adj = holm(np.array([p for _, _, p in fam]))
        for (name, i, _), a in zip(fam, adj):
            key = "mv_range_vs_abs" if ref == "abs_grad" else "mv_range_vs_abs512"
            out[name]["cells"][i][key]["p_holm"] = float(a)
    return out


def verdict(res, key):
    eligible = [n for n in PRIMARY if n in res and res[n]["phase1_check"]["passed"]]
    helps = any(c["decision_useful"] and c[key].get("p_holm", 1) < 0.05 and c[key]["diff"] >= 0.01
                for n in eligible if n in ("undertrained_e100", "pruned_s07") for c in res[n]["cells"])
    harmful = any(c["decision_useful"] and c[key].get("p_holm", 1) < 0.05 and c[key]["diff"] <= -0.005
                  for n in eligible for c in res[n]["cells"])
    resolved_half = any(c["decision_useful"] and c[key].get("p_holm", 1) < 0.05 and abs(c[key]["diff"]) >= 0.005
                        for n in eligible for c in res[n]["cells"])
    return dict(eligible=eligible, helps=helps and not harmful, harmful=harmful, no_gain=not resolved_half)


def main():
    names = [n for n in PRIMARY + SECONDARY if (OUT / n).is_dir() and any((OUT / n).glob("rep*/DONE"))]
    res = analyze(names)
    v_abs, v_512 = verdict(res, "mv_range_vs_abs"), verdict(res, "mv_range_vs_abs512")
    summary = dict(verdict_vs_paper_absgrad=v_abs, verdict_vs_absgrad512=v_512,
                   method_helps=v_abs["helps"] and v_512["helps"], checkpoints=res)
    (OUT / "phase2_analysis.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(dict(verdict_vs_paper_absgrad=v_abs, verdict_vs_absgrad512=v_512,
                          method_helps=summary["method_helps"],
                          phase1_checks={n: r["phase1_check"] for n, r in res.items()}), indent=1))
    for n, r in res.items():
        print(f"== {n} (replications {len(r['replications'])})")
        for c in r["cells"]:
            if not c["decision_useful"] and not c["new_arm_makes_useful"]:
                continue
            m = c["mean"]
            k = c["mv_range_vs_abs"]
            print(f"  ber {c['ber']:<7g} B {c['budget']:<6g} clean {100*c['clean']:.1f} abs {100*m['abs_grad']:.1f} "
                  f"abs512 {100*m['abs512']:.1f} sq {100*m['squared']:.1f} mvR {100*m['mv_range']:.1f} mvS {100*m['mv_single']:.1f} "
                  f"pos {100*m['pos_grad']:.1f} | mvR-abs {100*k['diff']:+.2f} p_holm {k.get('p_holm', float('nan')):.3f} "
                  f"rep [{100*k['rep_low']:+.2f},{100*k['rep_high']:+.2f}]{'  (useful only via new arm)' if c['new_arm_makes_useful'] else ''}")


if __name__ == "__main__":
    main()
