#!/usr/bin/env python3
"""EXPLORATORY diagnostic after Phase C1 of the partial-group search (plan review/PARTIAL_GROUP_SEARCH_PLAN_2026-11-14.md):
does the search mask beat abs1024 on the clips it was searched on, and on the calibration clips it never saw?

For replication R, every finished search path and budget: the search mask, abs1024, and significance-first (msb), under
K fresh real-flip fields at the path's rate (seeds [20261216, R, rate index, f], never used by the search), companion
passes on (a) the 512 search clips (calibration clips 0-511 of R) and (b) the other 512 calibration clips (512-1023).
Paired by field.  If the search wins on (a) but not on (b) or the test window, its losses come from fitting the search
clips or noisy selection; if it loses on (a) too, the greedy allocation itself is worse.

    runR_partial_search_insample.py <out> --rep 0 [--K 64]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from site_paths import WT  # noqa: E402
from runR_group_search import log, rate_index, require_reference_tf  # noqa: E402

RUNS = WT / "paper1-tecs-revision-v1/workspace/runs"
SEED = 20261216


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--search", type=Path, default=RUNS / "paper1_partial_search_kws_v1")
    a = ap.parse_args()
    from runP_phase12 import unpack
    from runR_kws import CAL_COUNT, KwsModel
    from runR_stage4_kws import load_static
    m = KwsModel(a.threads)
    tf = m.tf
    require_reference_tf(tf)
    a.out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    xc, yc, ic = m.clips("ranking", CAL_COUNT, CAL_COUNT * a.rep)
    assert ic[0] == CAL_COUNT * a.rep and len(ic) == CAL_COUNT
    xc, yc = np.asarray(xc), np.asarray(yc).astype(np.int64)
    halves = dict(search=(xc[:512], yc[:512]), unseen=(xc[512:], yc[512:]))
    J = m.jumps
    paths = sorted(d for d in a.search.glob(f"rep{a.rep}_mc_*") if (d / "masks.npz").exists())
    jobs = []
    for d in paths:
        meta = json.loads((d / "DONE").read_text())
        z = np.load(d / "masks.npz")
        for k in z.files:
            b = float(k.split("@")[1])
            st = load_static(m, a.rep, b)
            jobs.append((d.name, float(meta["rate"]), b, dict(search=unpack(z[k], m.n), abs1024=st["abs1024"], msb=st["msb"])))
    total = len(jobs) * a.K
    log(phase="insample", rep=a.rep, paths=len(paths), cells=len(jobs), K=a.K, done=0, total=total)
    rows, done = [], 0
    for name, p, b, masks in jobs:
        ri = rate_index(p)
        acc = {(h, arm): [] for h in halves for arm in masks}
        for f in range(a.K):
            u = np.random.default_rng([SEED, a.rep, ri, f]).random((m.n, 8)) < p
            for arm, mk in masks.items():
                m.set_displacement(np.where(u & ~mk, J, 0.0).sum(1))
                for h, (x, y) in halves.items():
                    logits, _ = m.comp.forward(x)
                    acc[(h, arm)].append(float(np.mean(logits.numpy().argmax(1) == y)))
            done += 1
        row = dict(path=name, rate=p, budget=b)
        for h in halves:
            for other in ("abs1024", "msb"):
                dd = 100 * (np.array(acc[(h, "search")]) - np.array(acc[(h, other)]))
                row[f"{h}: search minus {other}"] = dict(mean=float(dd.mean()), se=float(dd.std(ddof=1) / np.sqrt(a.K)))
            row[f"{h}: search accuracy"] = float(np.mean(acc[(h, "search")]))
        rows.append(row)
        el = time.time() - t0
        log(path=name, budget=b, done=done, total=total, seconds=round(el),
            search_minus_abs1024=dict(search_clips=round(row["search: search minus abs1024"]["mean"], 2),
                                      unseen_clips=round(row["unseen: search minus abs1024"]["mean"], 2)),
            eta_minutes=round(el / done * (total - done) / 60, 1))
    m.set_displacement(np.zeros(m.n))
    (a.out / "insample.json").write_text(json.dumps(dict(status="EXPLORATORY diagnostic after C1", rep=a.rep, K=a.K,
                                                         seed=SEED, rows=rows, seconds=round(time.time() - t0)), indent=1))
    (a.out / "DONE").write_text("ok\n")


if __name__ == "__main__":
    main()
