#!/usr/bin/env python3
"""Numbers cited in the ranking-choice section (Section 7.3 draft and Appendix E), offline.

Sources (all locked, write-once, under workspace/runs/ of this worktree):
  Run R Stage 1   paper1_runR_stage1_analysis_v1/summary.json
  Run R Stage 2   paper1_runR_stage2_analysis_v1/{summary.json,cells.csv,cross_rate.csv}
  Run R Stage 4   paper1_runR_stage4_kws_analysis_v1/{summary.json,switch_cells.csv}
                  paper1_runR_v2_stage4_analysis_v1/summary.json
  Search follow-up paper1_psearch_d_compare_v1/compare.json
  Gradient trust  paper1_gradtrust_analysis_v1/summary.json, paper1_gradtrust_diag_v1/<id>/summary.json,
                  paper1_gradtrust_faults_v1/<id>/{clean.csv,rep*/field*/results.csv}
  DS-CNN cal size paper1_kws_calsize_analysis_v1/summary.json

    tecs_rate_choice_summary.py --refresh <runs dir>   byte-copy the sources into data/tecs_rate_choice/, aggregate
                                                       the raw gradient-trust fault rows into fault_means.csv, and
                                                       record every input hash in provenance.json
    tecs_rate_choice_summary.py                        check the copies against provenance.json and write summary.json

Accuracy differences are in percentage points.
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/tecs_rate_choice"
COPIES = {
    "stage1_summary.json": "paper1_runR_stage1_analysis_v1/summary.json",
    "stage1_cells.csv": "paper1_runR_stage1_analysis_v1/cells.csv",
    "stage2_summary.json": "paper1_runR_stage2_analysis_v1/summary.json",
    "stage2_cells.csv": "paper1_runR_stage2_analysis_v1/cells.csv",
    "stage2_cross_rate.csv": "paper1_runR_stage2_analysis_v1/cross_rate.csv",
    "stage4_summary.json": "paper1_runR_stage4_kws_analysis_v1/summary.json",
    "stage4_switch_cells.csv": "paper1_runR_stage4_kws_analysis_v1/switch_cells.csv",
    "stage4_v2_summary.json": "paper1_runR_v2_stage4_analysis_v1/summary.json",
    "psearch_compare.json": "paper1_psearch_d_compare_v1/compare.json",
    "gradtrust_summary.json": "paper1_gradtrust_analysis_v1/summary.json",
    "kws_calsize_summary.json": "paper1_kws_calsize_analysis_v1/summary.json",
}
IDS = ["jet-e1", "jet-e3", "jet-e10", "jet-e30", "jet-e100", "mnist-e1", "mnist-e3", "mnist-e10", "mnist-e30", "vww",
       "mnist-tcad"]
LOW = 3e-3  # "low rates": at or below this BER


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def pts(x: float) -> float:
    return round(100 * x, 2)


def refresh(runs: Path):
    DATA.mkdir(parents=True, exist_ok=True)
    prov = dict(runs_dir="workspace/runs (paper worktree)", copies={}, diag={}, fault_inputs={})
    for name, src in COPIES.items():
        shutil.copyfile(runs / src, DATA / name)
        prov["copies"][name] = dict(source=src, sha256=sha(DATA / name))
    diag = {}
    for cid in IDS:
        p = runs / f"paper1_gradtrust_diag_v1/{cid}/summary.json"
        prov["diag"][cid] = sha(p)
        d = json.loads(p.read_text())
        diag[cid] = {k: d[k] for k in ("D1", "D2", "weights", "draws", "draw_size", "operators", "size_study", "sha256")
                     if k in d}
    (DATA / "gradtrust_diag.json").write_text(json.dumps(diag, indent=1, sort_keys=True) + "\n")
    prov["copies"]["gradtrust_diag.json"] = dict(source="extract of paper1_gradtrust_diag_v1/<id>/summary.json",
                                                 sha256=sha(DATA / "gradtrust_diag.json"))
    acc = defaultdict(list)
    clean = {}
    for cid in IDS:
        base = runs / f"paper1_gradtrust_faults_v1/{cid}"
        prov["fault_inputs"][f"{cid}/clean.csv"] = sha(base / "clean.csv")
        with open(base / "clean.csv") as fh:
            clean[cid] = [float(r["clean_accuracy"]) for r in csv.DictReader(fh)]
        for f in sorted(base.glob("rep*/field*/results.csv")):
            prov["fault_inputs"][str(f.relative_to(runs / "paper1_gradtrust_faults_v1"))] = sha(f)
            with open(f) as fh:
                for r in csv.DictReader(fh):
                    acc[(cid, float(r["budget"]), float(r["ber"]), r["arm"])].append(float(r["accuracy"]))
    with open(DATA / "fault_means.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["checkpoint", "budget", "ber", "arm", "fields", "mean_accuracy", "clean_accuracy"])
        for (cid, b, p, arm), v in sorted(acc.items()):
            w.writerow([cid, b, p, arm, len(v), sum(v) / len(v), sum(clean[cid]) / len(clean[cid])])
    prov["copies"]["fault_means.csv"] = dict(source="aggregate of paper1_gradtrust_faults_v1 (fault_inputs)",
                                             sha256=sha(DATA / "fault_means.csv"))
    (DATA / "provenance.json").write_text(json.dumps(prov, indent=1, sort_keys=True) + "\n")


def load():
    prov = json.loads((DATA / "provenance.json").read_text())
    for name, rec in prov["copies"].items():
        assert sha(DATA / name) == rec["sha256"], f"{name} does not match provenance.json"
    j = lambda n: json.loads((DATA / n).read_text())  # noqa: E731
    rows = lambda n: list(csv.DictReader(open(DATA / n)))  # noqa: E731
    return j, rows


def main():
    j, rows = load()
    out = {}

    # Stage 1: switch prediction on nine ResNet-8 checkpoints
    s1 = j("stage1_summary.json")
    out["stage1"] = dict(cells=s1["cells_total"], usable=s1["cells_usable"], away=s1["usable_away_from_switch"],
                         sign_agree=s1["R1b"]["agree"], sign_violations=len(s1["R1a"]["violations"]),
                         ratio=round(s1["R1c"]["ratio"], 3), ratio_ci=[round(s1["R1c"]["low"], 3), round(s1["R1c"]["high"], 3)],
                         undertrained_switch=s1["secondary"]["undertrained"],
                         well_trained_switch=[min(s1["secondary"]["well_trained"].values()),
                                              max(s1["secondary"]["well_trained"].values())],
                         pairs_without_switch=s1["pairs_without_measured_switch"])
    away = [r for r in rows("stage1_cells.csv") if r["usable"] == "True" and r["near"] == "False"]
    res = [r for r in away if r["resolved"] == "True"]
    out["stage1"].update(away_resolved=len(res), away_resolved_agree=sum(r["agrees"] == "True" for r in res))

    # Stage 2: rate-aware masks against static rankings on ResNet-8
    s2 = j("stage2_summary.json")
    viol = defaultdict(lambda: defaultdict(int))
    vdiff = defaultdict(list)
    for r in rows("stage2_cells.csv"):
        if r["violation"] == "True":
            viol[r["ref"]][r["ber"]] += 1
            vdiff[r["ref"]].append(float(r["diff"]))
    gaps = {ck: {a: round(v, 1) for a, v in rec["largest_gap"].items()} for ck, rec in s2["per_checkpoint"].items()}
    out["stage2"] = dict(
        R2a=dict(holds=s2["R2a"]["holds"], violations=len(s2["R2a"]["violations"]),
                 checkpoints=len({v["checkpoint"] for v in s2["R2a"]["violations"]}) if s2["R2a"]["violations"]
                 and "checkpoint" in s2["R2a"]["violations"][0] else None,
                 range=[round(min(v["diff"] for v in s2["R2a"]["violations"]), 2),
                        round(max(v["diff"] for v in s2["R2a"]["violations"]), 2)]),
        R2b=s2["R2b"], R2c=s2["R2c"], R2a_prime=dict(holds=s2["R2a_prime"]["holds"],
                                                      violations=len(s2["R2a_prime"]["violations"])),
        beaten_cells={ref: dict(total=sum(c.values()), by_ber=dict(sorted(c.items(), key=lambda t: float(t[0]))),
                                median=round(median(vdiff[ref]), 2), max=round(max(vdiff[ref]), 1))
                      for ref, c in viol.items() if ref in ("abs1024", "pos1024", "v2", "msb")},
        largest_gap=gaps,
        mc_minus_v1_median=None)
    xr = [r for r in rows("stage2_cross_rate.csv") if r["checkpoint"] == "pinned" and float(r["built_for"]) == 1e-4]
    out["stage2"]["pinned_built_for_1e-4"] = {r["tested_at"]: round(float(r["mean_gap"]), 1) for r in xr
                                              if r["mean_gap"]}

    # Stage 4: DS-CNN
    s4 = j("stage4_summary.json")
    v2 = j("stage4_v2_summary.json")
    sw = rows("stage4_switch_cells.csv")
    out["stage4"] = dict(
        pos_minus_abs={f'{float(r["budget"])}@{float(r["ber"])}': round(float(r["diff"]), 2) for r in sw},  # stored in points
        pos_ever_resolved_ahead=any(r["resolved"] == "True" and float(r["diff"]) > 0 for r in sw),
        R4a_holds=s4["switch"]["R4a"]["holds"], R4a_wrong_sign=len(s4["switch"]["R4a"]["violations"]),
        R4d_holds=s4["method"]["R4d"]["holds"], R4d_violations=len(s4["method"]["R4d"]["violations"]),
        R4d_max=round(max(v["diff"] for v in s4["method"]["R4d"]["violations"]), 1),
        R4d_prime=[dict(budget=v["budget"], ber=v["ber"], arm=v["arm"], diff=round(v["diff"], 2))
                   for v in v2["R4d_prime"]["violations"]],
        v2_largest_gap=v2["largest_gap_static_minus_v2"])

    # Search follow-up (DS-CNN, 1e-2, replications 1-5)
    ps = j("psearch_compare.json")
    out["psearch"] = dict(primary=[dict(arm=r["arm"], budget=r["budget"], mean=round(r["mean"], 2), se=round(r["se"], 2),
                                         p_holm=round(r["p_holm"], 3), resolved=r["resolved"])
                                    for r in ps["primary_minus_abs1024"]],
                          resolved_any=any(r["resolved"] != "unresolved" for r in ps["primary_minus_abs1024"]),
                          descriptive=ps["descriptive"])

    # Gradient trust: ten test checkpoints (+ mnist-tcad, exploratory)
    gt = j("gradtrust_summary.json")
    dg = j("gradtrust_diag.json")
    fm = defaultdict(dict)
    for r in rows("fault_means.csv"):
        fm[r["checkpoint"]][(float(r["budget"]), float(r["ber"]), r["arm"])] = float(r["mean_accuracy"])
    gtab = {}
    for cid in IDS:
        c = gt["checkpoints"][cid]
        cells = c["cells"]
        lows = [x["diff"] for x in cells if x["ber"] <= LOW]
        m = fm[cid]
        gap_msb = [100 * (m[(b, p, "abs1024")] - m[(b, p, "msb")]) for (b, p, a) in m if a == "msb"]
        gtab[cid] = dict(
            weights=dg[cid]["weights"], D1=round(c["D1"], 2), clean=pts(c["clean"]), usable=c["usable_cells"],
            exploratory=c["exploratory"], predicted=c["predicted"], measured=c["measured"],
            pos_ahead=sum(x["pos_ahead"] for x in cells), abs_ahead=sum(x["abs_ahead"] for x in cells),
            low_rate_pos_minus_abs=[round(min(lows), 2), round(max(lows), 2)],  # cell diffs are stored in points
            low_rate_mean_pos_minus_abs=round(sum(lows) / len(lows), 2),
            pos_minus_abs_all=[round(min(x["diff"] for x in cells), 2), round(max(x["diff"] for x in cells), 2)],
            acc_125={arm: {p: pts(m[(0.125, p, arm)]) for p in (1e-3, 3e-3, 1e-2, 3e-2)} for arm in ("abs1024", "pos1024", "msb")},
            abs_minus_msb_max=round(max(gap_msb), 1),
            # Table tab:newmodels, one decimal from the raw means (cells are usable-only in the analysis)
            table_row=[f"{100 * c['clean']:.1f}", f"{100 * m[(0.125, 3e-3, 'abs1024')]:.1f}"]
            + [f"{100 * m[(0.125, 3e-2, a)]:.1f}" for a in ("abs1024", "pos1024", "msb")]
            + [str(c["usable_cells"]), f"{min(lows):+.2f} to {max(lows):+.2f}",
               f"{sum(x['pos_ahead'] for x in cells)} / {sum(x['abs_ahead'] for x in cells)}"])
    out["gradtrust"] = dict(decision=gt["decision"], checkpoints=gtab,
                            test_ids=[c for c in IDS if not gtab[c]["exploratory"]])

    # DS-CNN calibration size
    ks = j("kws_calsize_summary.json")
    out["kws_calsize"] = dict(D1={k: round(v, 2) for k, v in ks["D1"].items()},
                              signed_ranking_helps=ks["signed_ranking_helps"],
                              **{k: [dict(budget=r["budget"], ber=r["ber"], diff=round(r["diff"], 2),
                                          p_holm=round(r["p_holm"], 3) if "p_holm" in r else None)
                                     for r in ks[k]]
                                 for k in ("primary_pos8192_minus_abs8192", "pos4096_minus_abs4096",
                                           "pos8192_minus_pos1024", "abs8192_minus_abs1024")})
    (DATA / "summary.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps(out, indent=1, sort_keys=True))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--refresh":
        refresh(Path(sys.argv[2]))
    main()
