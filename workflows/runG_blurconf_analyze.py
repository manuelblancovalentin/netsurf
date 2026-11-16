#!/usr/bin/env python3
"""Confirmatory analysis of the two-moment ("noise plus events") fault model on checkpoints the pilot never used.
Plan and rules: review/BLUR_CONFIRMATION_PLAN_2026-11-15.md (paper worktree).  Run once, after every job has ended.

Targets are the finished TFLite fault runs of the same replications, windows and fields: Stage 1 for the eight
ResNet-8 checkpoints, the gradient-trust fault runs for the jet, MNIST and VWW checkpoints.  All differences are in
accuracy points.

  cell          (replication, arm, budget, rate); the model's value is the mean of its 8 draws, the measured value the
                mean of the 8 TFLite fields
  usable        the better measured arm of abs1024 and pos1024 at that (replication, budget, rate) is within 5 points
                of the replication's measured clean accuracy
  G0  path      real variant (companion replay of the measured fields) against TFLite on usable cells:
                mean |error| <= 0.25 and worst |error| <= 1.0
  C1  accuracy  hybrid against TFLite on usable cells: mean |error| <= 1.0
  C2  order     pairs (pos1024, abs1024) and (msb, abs1024) in usable cells with a measured gap of at least 0.5 point:
                the hybrid gap has the same sign; pooled over each family, at least 85% agree
  C3  regime    "two regimes" if, at some budget and some rate <= 3e-3, pos1024 minus abs1024 pooled over the
                replications (paired by field or draw, usable replications only) has mean >= 0.2 and mean >= 2.5
                standard errors; computed by the same rule on the measured runs and on the hybrid; they must agree
  A checkpoint that fails G0 counts as failing C1 and C3 and is flagged (a path failure, not a failure of the model).

  verdict       CONFIRMED  G0 on >= 15 of 18, C1 on >= 15 of 18, C3 agreement on >= 15 of 18 with >= 6 of the 8
                           ResNet-8 and >= 8 of the 10 others, and C2 in both families
                FAILED     C1 on fewer than 9 of 18, or C3 agreement on fewer than 12 of 18
                PARTIAL    anything else

    runG_blurconf_analyze.py [--conf DIR] [--out DIR]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

R8 = ("undertrained_e100", "pruned_s07", "pruned_s05", "retrained_seed7101", "retrained_seed7102", "retrained_seed7103",
      "retrained_seed7104", "retrained_seed7105")
GEN = ("jet-e1", "jet-e3", "jet-e10", "jet-e30", "jet-e100", "mnist-e1", "mnist-e3", "mnist-e10", "mnist-e30", "vww")
REPS = {**{n: (0, 1, 2, 3) for n in R8}, **{n: (0, 1, 2, 3, 4) for n in GEN}}   # replication 5 was used for timing smokes
VARIANTS = ("mean", "gauss", "hybrid", "real")
MARGIN, G0_MEAN, G0_WORST, C1_MEAN, C2_GAP, C2_SHARE, C3_GAIN, C3_Z, LOW = 0.05, 0.25, 1.0, 1.0, 0.5, 0.85, 0.2, 2.5, 3e-3
PAIRS = (("pos1024", "abs1024"), ("msb", "abs1024"))


def key(arm, b, p):
    return (arm, round(float(b), 6), round(float(p), 8))


def read_model(d: Path):
    """{(arm, budget, rate, variant): [accuracy per draw]} of one job."""
    acc = defaultdict(dict)
    with open(d / "rows.csv") as fh:
        for r in csv.DictReader(fh):
            acc[key(r["arm"], r["budget"], r["ber"]) + (r["variant"],)][int(r["draw"])] = float(r["accuracy"])
    return {k: [v[i] for i in sorted(v)] for k, v in acc.items()}


def read_measured(base: Path, clean_from: str):
    """{(arm, budget, rate): [accuracy per field]} and the mean clean accuracy of one replication of a TFLite run."""
    acc, clean = defaultdict(dict), []
    for f in sorted(base.glob("field*/results.csv")):
        fi = int(f.parent.name[5:])
        with open(f) as fh:
            for r in csv.DictReader(fh):
                acc[key(r["arm"], r["budget"], r["ber"])][fi] = float(r["accuracy"])
        clean.append(json.loads((f.parent / clean_from).read_text())["clean_accuracy"])
    return {k: [v[i] for i in sorted(v)] for k, v in acc.items()}, float(np.mean(clean))


def regime(diffs_by_cell):
    """diffs_by_cell: {(budget, rate): 1-D array of paired pos-minus-abs differences in points}."""
    hits = []
    for (b, p), d in sorted(diffs_by_cell.items()):
        d = np.asarray(d, dtype=float)
        if p > LOW * (1 + 1e-9) or d.size < 2:
            continue
        m, se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(d.size))
        if m >= C3_GAIN and m >= C3_Z * se:
            hits.append(dict(budget=b, ber=p, mean=m, se=se, n=int(d.size)))
    return ("two regimes" if hits else "one regime"), hits


def checkpoint(model_by_rep, meas_by_rep, clean_by_rep):
    """Rules for one checkpoint.  *_by_rep: {rep: dict}; see read_model / read_measured."""
    cells, pairs, dm, dh = [], [], defaultdict(list), defaultdict(list)
    for rep in sorted(model_by_rep):
        mod, meas, clean = model_by_rep[rep], meas_by_rep[rep], clean_by_rep[rep]
        arms = sorted({k[0] for k in mod})
        for (b, p) in sorted({k[1:3] for k in mod}):
            best = max(np.mean(meas[key(a, b, p)]) for a in ("abs1024", "pos1024"))
            usable = bool(best >= clean - MARGIN)
            for arm in arms:
                row = dict(rep=rep, arm=arm, budget=b, ber=p, usable=usable, measured=100 * float(np.mean(meas[key(arm, b, p)])))
                for v in VARIANTS:
                    row[v] = 100 * float(np.mean(mod[key(arm, b, p) + (v,)]))
                cells.append(row)
            if not usable:
                continue
            for a1, a0 in PAIRS:
                if key(a1, b, p) not in meas or key(a1, b, p) + ("hybrid",) not in mod:
                    continue
                gm = 100 * float(np.mean(meas[key(a1, b, p)]) - np.mean(meas[key(a0, b, p)]))
                gh = 100 * float(np.mean(mod[key(a1, b, p) + ("hybrid",)]) - np.mean(mod[key(a0, b, p) + ("hybrid",)]))
                pairs.append(dict(rep=rep, pair=f"{a1}-{a0}", budget=b, ber=p, measured=gm, hybrid=gh,
                                  counted=bool(abs(gm) >= C2_GAP), agree=bool(np.sign(gm) == np.sign(gh))))
            dm[(b, p)] += list(100 * (np.array(meas[key("pos1024", b, p)]) - np.array(meas[key("abs1024", b, p)])))
            dh[(b, p)] += list(100 * (np.array(mod[key("pos1024", b, p) + ("hybrid",)]) - np.array(mod[key("abs1024", b, p) + ("hybrid",)])))
    U = [c for c in cells if c["usable"]]
    err = {v: np.array([c[v] - c["measured"] for c in U]) for v in VARIANTS}
    stat = {v: dict(mean_abs=float(np.abs(e).mean()), mean_signed=float(e.mean()), worst=float(e[np.argmax(np.abs(e))]))
            if e.size else None for v, e in err.items()}
    allerr = {v: float(np.mean([abs(c[v] - c["measured"]) for c in cells])) for v in VARIANTS}
    g0 = bool(U and stat["real"]["mean_abs"] <= G0_MEAN and abs(stat["real"]["worst"]) <= G0_WORST)
    c1 = bool(g0 and stat["hybrid"]["mean_abs"] <= C1_MEAN)
    rm, hm = regime(dm)
    rh, hh = regime(dh)
    counted = [q for q in pairs if q["counted"]]
    return dict(usable_cells=len(U), cells=len(cells), errors_usable=stat, mean_abs_error_all_cells=allerr, G0=g0, C1=c1,
                C2_counted=len(counted), C2_agree=sum(q["agree"] for q in counted),
                regime_measured=rm, regime_hybrid=rh, C3=bool(g0 and rm == rh), regime_hits_measured=hm, regime_hits_hybrid=hh,
                ), cells, pairs


def verdict(per):
    n = len(per)
    g0 = sum(c["G0"] for c in per.values())
    c1 = sum(c["C1"] for c in per.values())
    c3 = sum(c["C3"] for c in per.values())
    fam = {"resnet8": [k for k in per if k in R8], "others": [k for k in per if k in GEN]}
    c3f = {f: sum(per[k]["C3"] for k in ks) for f, ks in fam.items()}
    c2 = {}
    for f, ks in fam.items():
        cnt, ag = sum(per[k]["C2_counted"] for k in ks if per[k]["G0"]), sum(per[k]["C2_agree"] for k in ks if per[k]["G0"])
        c2[f] = dict(counted=cnt, agree=ag, share=(ag / cnt if cnt else None), holds=bool(cnt and ag / cnt >= C2_SHARE))
    confirmed = (n == 18 and g0 >= 15 and c1 >= 15 and c3 >= 15 and c3f["resnet8"] >= 6 and c3f["others"] >= 8
                 and all(v["holds"] for v in c2.values()))
    failed = c1 < 9 or c3 < 12
    return dict(checkpoints=n, G0=g0, C1=c1, C3=c3, C3_by_family=c3f, C2=c2,
                outcome="CONFIRMED" if confirmed else ("FAILED" if failed else "PARTIAL"))


def main():
    from site_paths import WT
    runs = WT / "paper1-tecs-revision-v1/workspace/runs"
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=Path, default=runs / "paper1_blurconf_v1")
    ap.add_argument("--out", type=Path, default=runs / "paper1_blurconf_analysis_v1")
    a = ap.parse_args()
    missing = [f"{n}_rep{r}" for n in R8 + GEN for r in REPS[n] if not (a.conf / "blur" / f"{n}_rep{r}" / "DONE").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} jobs are not finished: {missing[:4]}")
    a.out.mkdir(parents=True, exist_ok=False)
    per, allcells, allpairs = {}, [], []
    for name in R8 + GEN:
        mod, meas, clean = {}, {}, {}
        for r in REPS[name]:
            mod[r] = read_model(a.conf / "blur" / f"{name}_rep{r}")
            if name in R8:
                meas[r], clean[r] = read_measured(runs / "paper1_runR_stage1_crossover_v1" / name / f"rep{r}", "field.json")
            else:
                meas[r], clean[r] = read_measured(runs / "paper1_gradtrust_faults_v1" / name / f"rep{r}", "clean.json")
        per[name], cells, pairs = checkpoint(mod, meas, clean)
        allcells += [dict(checkpoint=name, **c) for c in cells]
        allpairs += [dict(checkpoint=name, **q) for q in pairs]
        c = per[name]
        print(f"{name:20s} G0 {c['G0']!s:5} C1 {c['C1']!s:5} hybrid {c['errors_usable']['hybrid']['mean_abs']:.2f} "
              f"gauss {c['errors_usable']['gauss']['mean_abs']:.2f} real {c['errors_usable']['real']['mean_abs']:.2f}  "
              f"C2 {c['C2_agree']}/{c['C2_counted']}  regime measured {c['regime_measured']}, hybrid {c['regime_hybrid']}")
    v = verdict(per)
    print(json.dumps(v))
    for nm, rows in (("cells.csv", allcells), ("pairs.csv", allpairs)):
        with open(a.out / nm, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (a.out / "summary.json").write_text(json.dumps(dict(
        status="confirmatory analysis (review/BLUR_CONFIRMATION_PLAN_2026-11-15.md)",
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), verdict=v, checkpoints=per), indent=1))


if __name__ == "__main__":
    main()
