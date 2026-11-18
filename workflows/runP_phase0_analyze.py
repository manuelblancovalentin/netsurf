#!/usr/bin/env python3
"""Run P, Phase 0 analysis: merge the offline predictions with the measured per-cell quality and evaluate the gate
fixed in the plan (amendment A1).  Reads only finished Phase 0 outputs and finalized measured tables."""
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
from runP_phase0 import CHECKPOINTS, OUT  # noqa: E402


def key(ranking, coverage, ber):
    return ranking, round(float(coverage), 6), float(f"{float(ber):.3g}")


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def predicted(name):
    rows = read(OUT / name / "stats/p0bc_mean_weight.csv")
    acc = defaultdict(list)
    for r in rows:
        k = key(r["ranking"], r["budget"], r["ber"])
        acc[k].append((float(r["mean_weight_eval_accuracy_drop"]), float(r["objective_mean_term"]),
                       float(r["objective_variance_term"])))
    return {k: dict(mw_drop=float(np.mean([v[0] for v in vs])), mean_term=float(np.mean([v[1] for v in vs])),
                    var_term=float(np.mean([v[2] for v in vs])), objective=float(np.mean([v[1] + v[2] for v in vs])))
            for k, vs in acc.items()}


def measured(name):
    out = {}
    for r in read(CHECKPOINTS[name]["measured"]):
        if r["method"] == "method":
            continue
        out[key(r["method"], r["coverage"], r["ber"])] = dict(drop=float(r["mean_accuracy_drop"]),
                                                             useful=r["decision_useful_primary"] in ("1", "True"))
    return out


def analyze_checkpoint(name):
    pred, meas = predicted(name), measured(name)
    cells = sorted({(k[1], k[2]) for k in pred})
    rankings = sorted({k[0] for k in pred})
    corr = []
    for cov, ber in cells:
        ks = [(r, cov, ber) for r in rankings]
        if not all(k in meas for k in ks):
            continue
        useful = meas[ks[0]]["useful"]
        m = [meas[k]["drop"] for k in ks]
        row = dict(coverage=cov, ber=ber, decision_useful=useful, measured_spread=float(max(m) - min(m)))
        for label in ("objective", "mean_term", "var_term", "mw_drop"):
            p = [pred[k][label] for k in ks]
            rho = stats.spearmanr(p, m).statistic
            row[f"spearman_{label}"] = None if (rho is None or math.isnan(rho)) else float(rho)
        corr.append(row)
    gate_cells = [c for c in corr if c["decision_useful"] and c["ber"] >= 1e-3]
    med = lambda lab: float(np.median([c[f"spearman_{lab}"] for c in gate_cells if c[f"spearman_{lab}"] is not None])) if gate_cells else None  # noqa: E731
    reversal = []
    for cov in (0.125, 0.25):
        for ber in (1e-3, 1e-2):
            kp, ka = key("pos_grad", cov, ber), key("abs_grad", cov, ber)
            if kp in pred and kp in meas:
                pd = pred[ka]["mw_drop"] - pred[kp]["mw_drop"]       # >0: positive gradient more accurate
                md = meas[ka]["drop"] - meas[kp]["drop"]
                reversal.append(dict(coverage=cov, ber=ber, predicted_pos_minus_abs=pd, measured_pos_minus_abs=md,
                                     sign_match=bool(np.sign(pd) == np.sign(md))))
    lops = defaultdict(list)
    for r in read(OUT / name / "stats/p0a_lopsidedness.csv"):
        lops[(r["ranking"], float(r["budget"]))].append(r)
    lopsided = {f"{k[0]}@{k[1]}": {f: float(np.mean([float(x[f]) for x in v])) for f in
                                   ("protected_increasing_share", "unprotected_increasing", "unprotected_decreasing",
                                    "net_over_increasing", "code_shift_per_weight_per_p", "code_shift_lopsidedness")}
                for k, v in sorted(lops.items()) if k[1] in (0.125, 0.25)}
    repro = read(OUT / name / "stats/abs_mask_reproduction.csv")
    return dict(checkpoint=name, cells=corr, gate_cells=len(gate_cells),
                median_spearman={lab: med(lab) for lab in ("objective", "mean_term", "var_term", "mw_drop")},
                reversal=reversal, lopsidedness=lopsided,
                abs_mask_reproduction_min_overlap=min(float(r["overlap"]) for r in repro))


def main():
    res = {}
    for name in CHECKPOINTS:
        if (OUT / name / "stats/p0bc_mean_weight.csv").is_file():
            res[name] = analyze_checkpoint(name)
    calib = {}
    for name in CHECKPOINTS:
        f = OUT / name / "p0d/p0d_calibration_size.csv"
        if f.is_file():
            rows = read(f)
            red = {int(r["n_calibration"]): float(r["heldout_reduction"]) for r in rows}
            chosen = 512 if red.get(512, 0.0) <= 0 else min(n for n, v in red.items() if v >= 0.8 * red[512])
            calib[name] = dict(rows=rows, chosen_n=chosen, transfers=red.get(512, 0.0) > 0)
    curv = {name: json.loads((OUT / name / "p0e/p0e_curvature.json").read_text())
            for name in CHECKPOINTS if (OUT / name / "p0e/p0e_curvature.json").is_file()}
    gate_i = bool(res.get("undertrained_e100") and res["undertrained_e100"]["reversal"]
                  and all(r["sign_match"] for r in res["undertrained_e100"]["reversal"])
                  and len(res["undertrained_e100"]["reversal"]) == 4)
    gate_ii = all(res.get(n, {}).get("median_spearman", {}).get("objective") is not None
                  and res[n]["median_spearman"]["objective"] >= 0.7 for n in ("pinned", "undertrained_e100"))
    summary = dict(gate_i_reversal=gate_i, gate_ii_ranking=gate_ii, gate_passed=gate_i and gate_ii,
                   chosen_calibration_n=max((c["chosen_n"] for c in calib.values()), default=None),
                   checkpoints=res, calibration=calib, curvature=curv)
    (OUT / "phase0_analysis.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(dict(gate_i=gate_i, gate_ii=gate_ii,
                          medians={n: r["median_spearman"] for n, r in res.items()},
                          gate_cells={n: r["gate_cells"] for n, r in res.items()},
                          reversal=res.get("undertrained_e100", {}).get("reversal"),
                          calibration={n: (c["chosen_n"], c["transfers"]) for n, c in calib.items()},
                          curvature=curv), indent=1))


if __name__ == "__main__":
    main()
