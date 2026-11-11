#!/usr/bin/env python3
"""Offline analysis of Run A (TFLite confirmation), following RUN_A_B_FROZEN_PLANS_2026-11-11.md.

Reads only the two finalized Run A bundles (trials.csv, sample_partitions.json, manifest.json,
summary.json).  Writes a new directory; refuses to overwrite.

Definitions fixed in this file before any regret or contrast was computed:

* Unit.  Primary: the replication (one calibration draw, one 1,000-input evaluation window and its
  own fault fields).  A replication's value in a cell is the mean over its fields.  Student-t
  interval over replications.  Secondary: fields pooled over replications, paired Student-t interval
  and a sign-flip Monte Carlo test (100,000 draws); the pooled test ignores that fields of one
  replication share a mask and a window, so it is conditional and reported as secondary.
* Fresh data.  Replication 0 reuses the window of the main grid and is tabulated separately.  All
  confirmation statistics use replications >= 1.
* Clean accuracy per replication = accuracy under full protection (no applied fault).
* Usefulness (same criterion as the manuscript): a mask is useful in a cell when the mean over
  fresh replications of (clean accuracy - accuracy) is at most 0.05 (sensitivity 0.01 and 0.10).
  A cell is decision-useful when at least one of the seven rankings is useful.
* Estimand 1.  d = error(absolute gradient) - error(competitor), paired.  Competitors: the six other
  rankings (primary set, as in the manuscript) and, as the strict set, also the four randomized
  within-tier significance masks.  Regret of the absolute-gradient mask in a cell = max(0, max over
  competitors of mean d).  Decision rule: the largest per-competitor upper 95% bound
  (replication-level) is below 0.01 in every decision-useful cell.
* Estimand 2.  At 6.25% and 18.75%: error(FKeras) - mean error of the four randomized within-tier
  masks, and error(storage-order MSB-first) - the same mean; also FKeras - storage-order MSB-first.
  Holm family: per model and contrast, the 16 (BER, budget) cells.
* Holm for estimand 1: per model, all (cell, competitor) contrasts of the primary set on the grid.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

W = Path("/projects/netsurf/manuelbv/workspace/implementation-worktrees")
RUNS = {"resnet8": W / "paper1-runA-tflite-confirm/workspace/runs/paper1_runA_resnet8_confirm_v1",
        "kws": W / "paper1-runA-tflite-confirm/workspace/runs/paper1_runA_kws_confirm_v1"}
ALIAS = {"random_global": "random", "msb_global": "msb", "numerical_jump_global": "jump",
         "gradient_global": "global_grad", "per_example_absolute_global": "abs_grad",
         "per_example_positive_global": "pos_grad", "fkeras_paper": "fkeras",
         "msb_random_tier_s1": "msb_rt1", "msb_random_tier_s2": "msb_rt2",
         "msb_random_tier_s3": "msb_rt3", "msb_random_tier_s4": "msb_rt4"}
RANKINGS = ["random", "msb", "jump", "global_grad", "abs_grad", "pos_grad", "fkeras"]
CONTROLS = ["msb_rt1", "msb_rt2", "msb_rt3", "msb_rt4"]
MARGINS = {"primary": 0.05, "tight": 0.01, "loose": 0.10}
PARTIAL = (0.0625, 0.1875)
DRAWS = 100_000
SEED = 20261111
BOUND = 0.01


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def t_ci(d):
    d = np.asarray(d, float); n = len(d); m = float(d.mean())
    if n < 2:
        return m, float("nan"), float("nan"), float("nan")
    sd = float(d.std(ddof=1))
    if sd == 0.0:
        return m, m, m, (1.0 if m == 0 else 0.0)
    se = sd / math.sqrt(n)
    h = stats.t.ppf(0.975, n - 1) * se
    return m, m - h, m + h, float(2 * stats.t.sf(abs(m / se), n - 1))


def signflip_p(d, rng) -> float:
    d = np.asarray(d, float)
    if np.allclose(d, 0.0):
        return 1.0
    obs = abs(float(d.mean())); count = 0; chunk = 10_000
    for _ in range(DRAWS // chunk):
        signs = rng.integers(0, 2, size=(chunk, len(d)), dtype=np.int8) * 2 - 1
        count += int(np.sum(np.abs((signs * d).mean(axis=1)) >= obs - 1e-15))
    return (1 + count) / (DRAWS + 1)


def holm(p):
    p = np.asarray(p, float); idx = np.argsort(p); m = len(p); out = np.empty(m); run = 0.0
    for rank, i in enumerate(idx):
        run = max(run, min(1.0, (m - rank) * p[i])); out[i] = run
    return out


def load(model: str):
    root = RUNS[model]
    bundles = [d for d in root.iterdir() if (d / "FINALIZED").is_file()]
    assert len(bundles) == 1, bundles
    run = bundles[0]
    manifest = json.loads((run / "manifest.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    assert manifest["status"] == "complete" and all(summary["checks"].values()), summary["checks"]
    parts = json.loads((run / "artifacts/sample_partitions.json").read_text())
    windows = {}
    for rep, roles in parts.items():
        i = int(rep.split("_")[1]); ev = roles["evaluation"]["indices"]; rk = roles["ranking"]["indices"]
        assert ev == list(range(1000 * i, 1000 * i + 1000)) and rk == list(range(1000 * i, 1000 * i + 64))
        windows[i] = dict(evaluation=[ev[0], ev[-1]], calibration=[rk[0], rk[-1]])
    # data[rep][ber][field][(method, coverage)] = (err, loss)
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    events_bad = 0
    with open(run / "measurements/trials.csv") as f:
        for r in csv.DictReader(f):
            ber_s, pol = r["policy"].split("/", 1)
            ber = float(ber_s.split("=")[1]); rep = int(r["replication"].split("_")[1])
            if pol in ("no_protection", "full_protection"):
                key = ("unprotected" if pol == "no_protection" else "full", 0.0 if pol == "no_protection" else 1.0)
            else:
                name, cov = pol.split("@"); key = (ALIAS[name], float(cov))
            data[rep][ber][int(r["trial"])][key] = (1.0 - float(r["accuracy"]), float(r["categorical_crossentropy"]))
            if int(r["proposed_events"]) != int(r["applied_events"]) + int(r["suppressed_events"]):
                events_bad += 1
            if pol == "full_protection" and int(r["applied_events"]) != 0:
                events_bad += 1
    prov = dict(run_dir=str(run), run_id=manifest["run_id"], source=manifest["source"],
                scientific_payload_sha256=manifest["scientific_payload_sha256"],
                trials_sha256=sha256(run / "measurements/trials.csv"), config_sha256=sha256(run / "config.toml"),
                checks=summary["checks"], windows=windows, event_identity_violations=events_bad)
    return data, prov


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(SEED)
    summary = dict(schema="paper1-runA-analysis/1.1", bound=BOUND, draws=DRAWS, seed=SEED, models={})
    cell_rows, e1_rows, e2_rows, rep0_rows = [], [], [], []
    provenance = {}
    for model in RUNS:
        data, prov = load(model)
        provenance[model] = prov
        reps = sorted(data); fresh = [r for r in reps if r >= 1]
        bers = sorted(data[reps[0]]); covs = sorted({k[1] for k in next(iter(data[reps[0]][bers[0]].values())) if 0 < k[1] < 1})
        clean = {}
        for rep in reps:
            vals = {v[("full", 1.0)][0] for ber in bers for v in data[rep][ber].values()}
            assert len(vals) == 1, (model, rep, vals)
            clean[rep] = 1.0 - vals.pop()
        fields_per = {rep: {ber: len(data[rep][ber]) for ber in bers} for rep in reps}

        def rep_mean(rep, ber, key, which=0):
            return float(np.mean([v[key][which] for v in data[rep][ber].values()]))

        def field_vals(rep_list, ber, key, which=0):
            return np.array([v[key][which] for rep in rep_list for _, v in sorted(data[rep][ber].items())])

        info = dict(replications=len(reps), fresh_replications=fresh, clean_accuracy=clean,
                    fields_per_replication_per_ber=sorted({n for d in fields_per.values() for n in d.values()}),
                    fresh_fields_per_ber={str(b): int(sum(fields_per[r][b] for r in fresh)) for b in bers})
        useful_cells = defaultdict(int); highest = {}
        e1_model = []
        for cov in covs:
            for ber in bers:
                # usefulness on fresh data
                drop = {}
                for m in RANKINGS + CONTROLS:
                    drop[m] = float(np.mean([(1 - clean[r]) * -1 + rep_mean(r, ber, (m, cov)) for r in fresh]))
                drop_unp = float(np.mean([rep_mean(r, ber, ("unprotected", 0.0)) - (1 - clean[r]) for r in fresh]))
                dec = {k: any(drop[m] <= v for m in RANKINGS) for k, v in MARGINS.items()}
                for k in MARGINS:
                    useful_cells[k] += int(dec[k])
                for m in RANKINGS:
                    acc_reps = [1 - rep_mean(r, ber, (m, cov)) for r in fresh]
                    cell_rows.append(dict(model=model, ber=ber, coverage=cov, method=m, fresh_replications=len(fresh),
                                          mean_accuracy=float(np.mean(acc_reps)), min_replication_accuracy=float(np.min(acc_reps)),
                                          mean_clean_accuracy=float(np.mean([clean[r] for r in fresh])),
                                          mean_accuracy_drop=drop[m], useful_primary=int(drop[m] <= MARGINS["primary"]),
                                          mean_loss=float(np.mean([rep_mean(r, ber, (m, cov), 1) for r in fresh])),
                                          decision_useful_primary=int(dec["primary"]), unprotected_accuracy_drop=drop_unp))
                # estimand 1
                for comp in [m for m in RANKINGS if m != "abs_grad"] + CONTROLS:
                    for which, risk in ((0, "err"), (1, "loss")):
                        d_rep = [rep_mean(r, ber, ("abs_grad", cov), which) - rep_mean(r, ber, (comp, cov), which) for r in fresh]
                        d_fld = field_vals(fresh, ber, ("abs_grad", cov), which) - field_vals(fresh, ber, (comp, cov), which)
                        m_r, lo_r, hi_r, p_r = t_ci(d_rep); m_f, lo_f, hi_f, _ = t_ci(d_fld)
                        row = dict(model=model, risk=risk, ber=ber, coverage=cov, competitor=comp,
                                   competitor_set="primary" if comp in RANKINGS else "within_tier_control",
                                   mean_diff=m_r, rep_ci_low=lo_r, rep_ci_high=hi_r, rep_t_p=p_r, n_rep=len(d_rep),
                                   field_mean_diff=m_f, field_ci_low=lo_f, field_ci_high=hi_f, n_field=len(d_fld),
                                   p_signflip_field=signflip_p(d_fld, rng),
                                   decision_useful_primary=int(dec["primary"]), decision_useful_tight=int(dec["tight"]),
                                   decision_useful_loose=int(dec["loose"]))
                        e1_rows.append(row); e1_model.append(row)
                # estimand 2
                if cov in PARTIAL:
                    for which, risk in ((0, "err"), (1, "loss")):
                        ctl_rep = np.mean([[rep_mean(r, ber, (c, cov), which) for r in fresh] for c in CONTROLS], axis=0)
                        ctl_fld = np.mean([field_vals(fresh, ber, (c, cov), which) for c in CONTROLS], axis=0)
                        seed_means = [float(np.mean([rep_mean(r, ber, (c, cov), which) for r in fresh])) for c in CONTROLS]
                        for name, a_rep, a_fld, b_rep, b_fld in (
                            ("fkeras_minus_random_tier", [rep_mean(r, ber, ("fkeras", cov), which) for r in fresh], field_vals(fresh, ber, ("fkeras", cov), which), ctl_rep, ctl_fld),
                            ("msb_storage_minus_random_tier", [rep_mean(r, ber, ("msb", cov), which) for r in fresh], field_vals(fresh, ber, ("msb", cov), which), ctl_rep, ctl_fld),
                            ("fkeras_minus_msb_storage", [rep_mean(r, ber, ("fkeras", cov), which) for r in fresh], field_vals(fresh, ber, ("fkeras", cov), which),
                             np.array([rep_mean(r, ber, ("msb", cov), which) for r in fresh]), field_vals(fresh, ber, ("msb", cov), which))):
                            d_rep = np.asarray(a_rep) - b_rep; d_fld = a_fld - b_fld
                            m_r, lo_r, hi_r, p_r = t_ci(d_rep); m_f, lo_f, hi_f, _ = t_ci(d_fld)
                            e2_rows.append(dict(model=model, risk=risk, contrast=name, ber=ber, coverage=cov,
                                                mean_diff=m_r, rep_ci_low=lo_r, rep_ci_high=hi_r, rep_t_p=p_r, n_rep=len(d_rep),
                                                field_ci_low=lo_f, field_ci_high=hi_f, n_field=len(d_fld),
                                                p_signflip_field=signflip_p(d_fld, rng),
                                                random_tier_seed_sd=float(np.std(seed_means, ddof=1)),
                                                decision_useful_primary=int(dec["primary"])))
                # replication 0, tabulated separately
                for m in RANKINGS:
                    rep0_rows.append(dict(model=model, ber=ber, coverage=cov, method=m, fields=fields_per[0][ber],
                                          accuracy=1 - rep_mean(0, ber, (m, cov)), clean_accuracy=clean[0],
                                          loss=rep_mean(0, ber, (m, cov), 1)))
        # complete tiers: tier-first masks must coincide, so paired differences must be exactly zero
        tier_identity = max(abs(rep_mean(r, ber, ("msb", cov)) - rep_mean(r, ber, (c, cov)))
                            for r in reps for ber in bers for cov in (0.125, 0.25) for c in CONTROLS + ["fkeras"])
        info["complete_tier_max_abs_difference_msb_vs_tier_first_masks"] = float(tier_identity)
        # Holm, estimand 1 (primary competitor set, error rate, whole grid)
        fam = [r for r in e1_model if r["risk"] == "err" and r["competitor_set"] == "primary"]
        for r, a in zip(fam, holm([r["p_signflip_field"] for r in fam])):
            r["p_holm_field"] = float(a)
        fam_l = [r for r in e1_model if r["risk"] == "loss" and r["competitor_set"] == "primary"]
        for r, a in zip(fam_l, holm([r["p_signflip_field"] for r in fam_l])):
            r["p_holm_field"] = float(a)
        for r in e1_model:
            r.setdefault("p_holm_field", float("nan"))
        # Holm, estimand 2
        mine = [r for r in e2_rows if r["model"] == model]
        for risk in ("err", "loss"):
            for name in ("fkeras_minus_random_tier", "msb_storage_minus_random_tier", "fkeras_minus_msb_storage"):
                fam2 = [r for r in mine if r["risk"] == risk and r["contrast"] == name]
                for r, a in zip(fam2, holm([r["p_signflip_field"] for r in fam2])):
                    r["p_holm_field"] = float(a)
        # decision rule
        dec_out = {}
        for cset in ("primary", "strict"):
            for region in ("decision_useful_primary", "decision_useful_tight", "decision_useful_loose"):
                sel = [r for r in e1_model if r["risk"] == "err" and r[region] and (cset == "strict" or r["competitor_set"] == "primary")]
                cells = defaultdict(list)
                for r in sel:
                    cells[(r["ber"], r["coverage"])].append(r)
                per = []
                for (ber, cov), rs in sorted(cells.items()):
                    worst_mean = max(rs, key=lambda r: r["mean_diff"]); worst_hi = max(rs, key=lambda r: r["rep_ci_high"])
                    worst_fhi = max(rs, key=lambda r: r["field_ci_high"])
                    per.append(dict(ber=ber, coverage=cov, regret=max(0.0, worst_mean["mean_diff"]), regret_vs=worst_mean["competitor"],
                                    max_rep_upper=worst_hi["rep_ci_high"], max_rep_upper_vs=worst_hi["competitor"],
                                    max_field_upper=worst_fhi["field_ci_high"], max_field_upper_vs=worst_fhi["competitor"]))
                if not per:
                    continue
                w = max(per, key=lambda c: c["max_rep_upper"]); wr = max(per, key=lambda c: c["regret"]); wf = max(per, key=lambda c: c["max_field_upper"])
                dec_out[f"{cset}/{region}"] = dict(
                    cells=len(per), max_regret=wr["regret"], max_regret_cell=wr,
                    max_replication_upper_bound=w["max_rep_upper"], max_replication_upper_cell=w,
                    cells_replication_upper_below_bound=int(sum(c["max_rep_upper"] < BOUND for c in per)),
                    max_field_upper_bound=wf["max_field_upper"], max_field_upper_cell=wf,
                    cells_field_upper_below_bound=int(sum(c["max_field_upper"] < BOUND for c in per)),
                    confirmed_replication_level=bool(all(c["max_rep_upper"] < BOUND for c in per)),
                    confirmed_field_level=bool(all(c["max_field_upper"] < BOUND for c in per)),
                    cells_failing_replication_level=[c for c in per if not c["max_rep_upper"] < BOUND])
        info["decision_rule"] = dec_out
        # Secondary reading, added after the first run of this script (runA_analysis_v1) showed that the
        # strict rule above is driven, with three replications, by wide intervals against competitors that
        # are far worse than the absolute-gradient mask.  This is the manuscript's regret definition:
        # difference to the best other measured ranking in the cell.  Reported next to, not instead of, the rule.
        rvb = {}
        for region in ("decision_useful_primary", "decision_useful_tight", "decision_useful_loose"):
            cells = defaultdict(list)
            for r in e1_model:
                if r["risk"] == "err" and r["competitor_set"] == "primary" and r[region]:
                    cells[(r["ber"], r["coverage"])].append(r)
            per = []
            for (ber, cov), rs in sorted(cells.items()):
                b = max(rs, key=lambda r: r["mean_diff"])
                per.append(dict(ber=ber, coverage=cov, best_other=b["competitor"], diff=b["mean_diff"],
                                rep_ci=[b["rep_ci_low"], b["rep_ci_high"]], field_ci=[b["field_ci_low"], b["field_ci_high"]]))
            if per:
                rvb[region] = dict(cells=len(per), max_diff=max(c["diff"] for c in per),
                                   max_rep_upper=max(c["rep_ci"][1] for c in per),
                                   cells_rep_upper_below_bound=int(sum(c["rep_ci"][1] < BOUND for c in per)),
                                   max_field_upper=max(c["field_ci"][1] for c in per),
                                   cells_field_upper_below_bound=int(sum(c["field_ci"][1] < BOUND for c in per)),
                                   cells_abs_grad_best=int(sum(c["diff"] <= 0 for c in per)),
                                   cells_failing_rep=[c for c in per if not c["rep_ci"][1] < BOUND], per_cell=per)
        info["regret_vs_best_other_ranking"] = rvb
        # accuracy drop per method and cell on fresh data (for the tolerated-BER statements)
        info["accuracy_drop_fresh"] = {f"{cov}/{ber}/{m}": float(np.mean([
            (rep_mean(r, ber, ("unprotected", 0.0)) if m == "unprotected" else rep_mean(r, ber, (m, cov))) - (1 - clean[r]) for r in fresh]))
            for cov in covs for ber in bers for m in RANKINGS + ["unprotected"]}
        prim = [r for r in fam if r["decision_useful_primary"]]
        info["abs_grad_holm_resolved_in_decision_useful_cells"] = dict(
            contrasts=len(prim), wins=int(sum(r["p_holm_field"] < 0.05 and r["field_mean_diff"] < 0 for r in prim)),
            losses=int(sum(r["p_holm_field"] < 0.05 and r["field_mean_diff"] > 0 for r in prim)),
            loss_list=[dict(ber=r["ber"], coverage=r["coverage"], competitor=r["competitor"], diff=r["field_mean_diff"], p_holm=r["p_holm_field"])
                       for r in prim if r["p_holm_field"] < 0.05 and r["field_mean_diff"] > 0])
        info["decision_useful_cells"] = dict(useful_cells)
        # highest useful BER per method at each budget (largest BER with all lower BERs also useful)
        for cov in covs:
            for m in RANKINGS + ["unprotected"]:
                top = None
                for ber in bers:
                    if m == "unprotected":
                        d = float(np.mean([rep_mean(r, ber, ("unprotected", 0.0)) - (1 - clean[r]) for r in fresh]))
                    else:
                        d = float(np.mean([rep_mean(r, ber, (m, cov)) - (1 - clean[r]) for r in fresh]))
                    if d <= MARGINS["primary"]:
                        top = ber
                    else:
                        break
                highest[f"{cov}/{m}"] = top
        info["highest_useful_ber"] = highest
        # estimand 2 digest
        e2d = {}
        for name in ("fkeras_minus_random_tier", "msb_storage_minus_random_tier", "fkeras_minus_msb_storage"):
            sel = [r for r in mine if r["risk"] == "err" and r["contrast"] == name]
            selu = [r for r in sel if r["decision_useful_primary"]]
            big = max(sel, key=lambda r: abs(r["mean_diff"]))
            e2d[name] = dict(cells=len(sel), holm_resolved_field=int(sum(r["p_holm_field"] < 0.05 for r in sel)),
                             rep_interval_excludes_zero=int(sum(r["rep_ci_low"] > 0 or r["rep_ci_high"] < 0 for r in sel)),
                             useful_cells=len(selu), useful_holm_resolved_field=int(sum(r["p_holm_field"] < 0.05 for r in selu)),
                             useful_rep_interval_excludes_zero=int(sum(r["rep_ci_low"] > 0 or r["rep_ci_high"] < 0 for r in selu)),
                             largest_abs_mean=dict(ber=big["ber"], coverage=big["coverage"], mean=big["mean_diff"], rep_ci=[big["rep_ci_low"], big["rep_ci_high"]], p_holm_field=big["p_holm_field"]),
                             largest_abs_mean_useful=(lambda b: dict(ber=b["ber"], coverage=b["coverage"], mean=b["mean_diff"], rep_ci=[b["rep_ci_low"], b["rep_ci_high"]], p_holm_field=b["p_holm_field"]))(max(selu, key=lambda r: abs(r["mean_diff"]))) if selu else None)
        info["within_tier_control"] = e2d
        summary["models"][model] = info

    def write(name, rows):
        with open(args.out / name, "x", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    write("fresh_cell_quality.csv", cell_rows)
    write("absgrad_contrasts.csv", e1_rows)
    write("within_tier_contrasts.csv", e2_rows)
    write("replication0_cells.csv", rep0_rows)
    with open(args.out / "summary.json", "x") as f:
        json.dump(summary, f, indent=1)
    provenance["script_sha256"] = sha256(Path(__file__))
    provenance["outputs"] = {p.name: sha256(p) for p in sorted(args.out.iterdir())}
    with open(args.out / "provenance.json", "x") as f:
        json.dump(provenance, f, indent=1)
    print(json.dumps({m: {k: v for k, v in i.items() if k in ("decision_useful_cells", "clean_accuracy", "complete_tier_max_abs_difference_msb_vs_tier_first_masks")} for m, i in summary["models"].items()}, indent=1))


if __name__ == "__main__":
    main()
