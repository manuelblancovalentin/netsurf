#!/usr/bin/env python3
"""Run Q analysis (plan `RUN_Q_CALIBRATION_SIZE_PLAN_2026-11-11.md`, Section 3), written before any Run Q outcome.

Sources, paired by (replication, schedule hash, budget):
  n = 64     saved outcomes of the paper's absolute, positive and global gradient masks (bundle trials.csv)
  abs512     Run P replays (arm abs512)
  others     Run Q replays (arms abs128, pos512, global1024, ...)
Q1 (order):  pos_n - abs_n and global_n - abs_n, n in {512, 1024}, usable cells of all checkpoints, one Holm family.
Q2 (size):   abs_n - abs_64, n in {128, 256, 512, 1024}, usable cells, one Holm family.
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
from runP_phase12 import OUT as RUNP_OUT  # noqa: E402
from runP_phase2_analyze import useful_flags  # noqa: E402
from runQ_calibration import BERS, KINDS, OUT, SIZES  # noqa: E402

NAMES = ("pinned", "undertrained_e100", "pruned_s07", "pruned_s05") + tuple(f"retrained_seed{s}" for s in range(7101, 7106))
SEEDS = NAMES[4:]
SAVED = {RANKINGS["abs_grad"]: "abs64", RANKINGS["pos_grad"]: "pos64", RANKINGS["global_grad"]: "global64"}
SIGNS = np.random.default_rng(20261114).choice((-1.0, 1.0), size=(100_000, 64))


def signflip_p(d):
    d = np.asarray(d, dtype=float)
    if np.allclose(d, 0):
        return 1.0
    obs = abs(d.mean())
    null = np.abs((SIGNS[:, : d.size] * d).mean(axis=1))
    return float((1 + (null >= obs - 1e-15).sum()) / (SIGNS.shape[0] + 1))


def holm(ps):
    ps = np.asarray(ps, dtype=float)
    adj, running = np.empty(ps.size), 0.0
    for rank, i in enumerate(np.argsort(ps)):
        running = max(running, (ps.size - rank) * ps[i])
        adj[i] = min(1.0, running)
    return adj


def load(name):
    """acc[(ber, budget)][arm][(rep, sha)] = accuracy, for replications with a finished Run Q job."""
    reps = sorted(int(p.parent.name[3:]) for p in (OUT / name).glob("rep*/DONE"))
    acc = defaultdict(lambda: defaultdict(dict))
    with open(bundle_dir(CHECKPOINTS[name]["bundle"]) / "measurements/trials.csv") as f:
        for r in csv.DictReader(f):
            rep = int(r["replication"].split("_")[1])
            if rep not in reps:
                continue
            ber_s, pol = r["policy"].split("/", 1)
            ber = float(ber_s.split("=")[1])
            if "@" not in pol or not any(abs(ber - p) < 1e-12 for p in BERS):
                continue
            pname, cov = pol.split("@")
            if pname in SAVED:
                acc[(ber, float(cov))][SAVED[pname]][(rep, r["schedule_sha256"])] = float(r["accuracy"])
    for rep in reps:
        for src, keep in ((RUNP_OUT / name / f"rep{rep}" / "replay.csv", {"abs512"}), (OUT / name / f"rep{rep}" / "replay.csv", None)):
            with open(src) as f:
                for r in csv.DictReader(f):
                    if r["arm"] == "replay_check_absgrad" or (keep is not None and r["arm"] not in keep):
                        continue
                    ber = float(r["ber"])
                    if any(abs(ber - p) < 1e-12 for p in BERS):
                        acc[(ber, float(r["budget"]))][r["arm"]][(rep, r["schedule_sha256"])] = float(r["accuracy"])
    return acc, reps


def contrast(cell, a, b):
    keys = sorted(set(cell[a]) & set(cell[b]))
    if not keys:
        return None
    d = np.array([cell[a][k] - cell[b][k] for k in keys])
    by_rep = defaultdict(list)
    for k, v in zip(keys, d):
        by_rep[k[0]].append(v)
    rm = np.array([np.mean(v) for v in by_rep.values()])
    h = stats.t.ppf(0.975, rm.size - 1) * rm.std(ddof=1) / math.sqrt(rm.size) if rm.size > 1 and rm.std() > 0 else 0.0
    return dict(diff=float(d.mean()), fields=int(d.size), reps=int(rm.size), p=signflip_p(d),
                rep_low=float(rm.mean() - h), rep_high=float(rm.mean() + h))


def main():
    res, q1, q1_ref, q2 = {}, [], [], []
    for name in NAMES:
        if not (OUT / name).is_dir():
            continue
        acc, reps = load(name)
        if not reps:
            continue
        flags = useful_flags(name)
        cells = []
        for (ber, b), arms in sorted(acc.items()):
            useful, clean = flags.get((float(f"{ber:.3g}"), round(b, 6)), (False, float("nan")))
            row = dict(ber=ber, budget=b, decision_useful=useful, clean=clean,
                       mean={a: float(np.mean(list(v.values()))) for a, v in arms.items()}, c={})
            for comp in ("pos", "global"):
                for n in (64, 512, 1024):
                    row["c"][f"{comp}{n}-abs{n}"] = contrast(arms, f"{comp}{n}", f"abs{n}")
            for kind in KINDS:
                for n in SIZES:
                    row["c"][f"{kind}{n}-{kind}64"] = contrast(arms, f"{kind}{n}", f"{kind}64")
            row["c"]["abs1024-abs512"] = contrast(arms, "abs1024", "abs512")
            i = len(cells)
            if useful:
                for comp in ("pos", "global"):
                    q1_ref.append((name, i, f"{comp}64-abs64"))
                    for n in (512, 1024):
                        q1.append((name, i, f"{comp}{n}-abs{n}"))
                for n in SIZES:
                    q2.append((name, i, f"abs{n}-abs64"))
            cells.append(row)
        res[name] = dict(replications=reps, cells=cells)
    for fam, tag in ((q1, "p_holm_q1"), (q1_ref, "p_holm_q1_ref64"), (q2, "p_holm_q2")):
        live = [(n, i, k) for n, i, k in fam if res[n]["cells"][i]["c"][k] is not None]
        for (n, i, k), a in zip(live, holm([res[n]["cells"][i]["c"][k]["p"] for n, i, k in live])):
            res[n]["cells"][i]["c"][k][tag] = float(a)

    def resolved(c, tag, sign=+1, margin=0.005):
        return c is not None and c.get(tag, 1.0) < 0.05 and sign * c["diff"] >= margin

    overtakes = []
    for n, r in res.items():
        for c in r["cells"]:
            if not c["decision_useful"]:
                continue
            for comp in ("pos", "global"):
                if resolved(c["c"][f"{comp}64-abs64"], "p_holm_q1_ref64"):
                    continue
                for size in (512, 1024):
                    if resolved(c["c"][f"{comp}{size}-abs{size}"], "p_holm_q1"):
                        overtakes.append(dict(checkpoint=n, ber=c["ber"], budget=c["budget"], arm=f"{comp}{size}",
                                              diff=c["c"][f"{comp}{size}-abs{size}"]["diff"]))
    size_ckpts = sorted({n for n, r in res.items() for c in r["cells"] if c["decision_useful"] and
                         any(resolved(c["c"][f"abs{s}-abs64"], "p_holm_q2") for s in (512, 1024))})
    pooled = {}
    seed_rows = [res[s]["cells"] for s in SEEDS if s in res]
    if len(seed_rows) == len(SEEDS):
        for idx, c0 in enumerate(seed_rows[0]):
            out = {}
            for k in c0["c"]:
                xs = [rows[idx]["c"][k]["diff"] for rows in seed_rows if rows[idx]["c"].get(k)]
                if len(xs) == len(SEEDS):
                    x = np.array(xs); h = stats.t.ppf(0.975, x.size - 1) * x.std(ddof=1) / math.sqrt(x.size)
                    out[k] = dict(mean=float(x.mean()), low=float(x.mean() - h), high=float(x.mean() + h))
            pooled[f"{c0['ber']:g}@{c0['budget']:g}"] = dict(useful_seeds=sum(rows[idx]["decision_useful"] for rows in seed_rows), contrasts=out)
    summary = dict(q1_default_holds=not overtakes, q1_overtakes=overtakes, q1_family=len(q1),
                   q2_calibration_size_matters=len(size_ckpts) >= 3, q2_checkpoints=size_ckpts, q2_family=len(q2),
                   seeds_pooled=pooled, checkpoints=res)
    (OUT / "analysis.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("q1_default_holds", "q1_overtakes", "q1_family", "q2_calibration_size_matters",
                                              "q2_checkpoints", "q2_family")}, indent=1))
    for n, r in res.items():
        print(f"== {n} (replications {len(r['replications'])})")
        for c in r["cells"]:
            m, cc = c["mean"], c["c"]
            f = lambda k: "   -  " if cc.get(k) is None else f"{100 * cc[k]['diff']:+.2f}{'*' if cc[k].get('p_holm_q2', cc[k].get('p_holm_q1', 1)) < 0.05 else ' '}"
            print(f"  {c['ber']:<6g} {c['budget']:<7g} U={int(c['decision_useful'])} abs64 {100 * m.get('abs64', float('nan')):.1f} | "
                  f"abs n-64: {' '.join(f(f'abs{s}-abs64') for s in SIZES)} | 1024-512 {f('abs1024-abs512')} | "
                  f"pos-abs @64/512/1024: {f('pos64-abs64')} {f('pos512-abs512')} {f('pos1024-abs1024')} | "
                  f"glob-abs: {f('global64-abs64')} {f('global512-abs512')} {f('global1024-abs1024')}")


if __name__ == "__main__":
    main()
