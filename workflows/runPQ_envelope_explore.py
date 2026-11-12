#!/usr/bin/env python3
"""Exploratory, post hoc (12 November 2026): does the single-rate mean-variance mask (Run P, `mv_single`, built for
the test BER from 512 images) behave like the positive gradient at low BER and like the absolute gradient at high BER?

Reads only finished, write-once replays on the same saved fault schedules:
  Run P  arms abs512, squared, mv_range, mv_single   (workspace/runs/paper1_runP_phase12_v1/<ckpt>/rep*/replay.csv)
  Run Q  arm pos512                                   (workspace/runs/paper1_runQ_calibration_size_v1/<ckpt>/rep*/replay.csv)
Replications with a finished Run Q job only (Q-A1 exclusion applies).  Usable cells from each checkpoint's measured table.
No test or rule was fixed in advance for this comparison; it generates a hypothesis, it does not confirm one.
Output: workspace/runs/paper1_runPQ_envelope_explore_v1/summary.json (write-once).
"""
import csv
import json
from collections import defaultdict

import numpy as np

from runP_phase12 import OUT as RUNP
from runP_phase2_analyze import useful_flags
from runQ_calibration import OUT as RUNQ

NAMES = ("pinned", "undertrained_e100", "pruned_s07", "pruned_s05") + tuple(f"retrained_seed{s}" for s in range(7101, 7106))
BERS = (1e-3, 3e-3, 1e-2, 3e-2)
BUDGETS = (0.0625, 0.125, 0.1875, 0.25)
ARMS_P = {"abs512", "squared", "mv_range", "mv_single"}
CONTRASTS = (("pos512", "abs512"), ("mv_single", "abs512"), ("mv_range", "abs512"), ("squared", "abs512"),
             ("mv_single", "pos512"), ("mv_range", "pos512"))


def main():
    out_dir = RUNP.parent / "paper1_runPQ_envelope_explore_v1"
    out_dir.mkdir(exist_ok=False)
    cells = []
    for name in NAMES:
        flags = useful_flags(name)
        acc = defaultdict(dict)
        for rep in sorted(int(p.parent.name[3:]) for p in (RUNQ / name).glob("rep*/DONE")):
            for src, keep in ((RUNP / name / f"rep{rep}" / "replay.csv", ARMS_P), (RUNQ / name / f"rep{rep}" / "replay.csv", {"pos512"})):
                with open(src) as f:
                    for r in csv.DictReader(f):
                        ber = float(r["ber"])
                        if r["arm"] in keep and any(abs(ber - p) < 1e-12 for p in BERS):
                            acc[(round(ber, 6), round(float(r["budget"]), 6), r["arm"])][(rep, r["schedule_sha256"])] = float(r["accuracy"])
        for ber in BERS:
            for b in BUDGETS:
                if not flags.get((float(f"{ber:.3g}"), round(b, 6)), (False,))[0]:
                    continue
                row = dict(checkpoint=name, ber=ber, budget=b, diff={})
                for a, c in CONTRASTS:
                    A, C = acc[(round(ber, 6), b, a)], acc[(round(ber, 6), b, c)]
                    keys = sorted(set(A) & set(C))
                    row["diff"][f"{a}-{c}"] = round(100 * float(np.mean([A[k] - C[k] for k in keys])), 3)
                m = {a: float(np.mean(list(acc[(round(ber, 6), b, a)].values()))) for a in ("abs512", "pos512", "mv_single")}
                row["diff"]["mv_single-best_of_abs512_pos512"] = round(100 * (m["mv_single"] - max(m["abs512"], m["pos512"])), 3)
                cells.append(row)
    summary = {}
    for ber in BERS:
        for k in [f"{a}-{c}" for a, c in CONTRASTS] + ["mv_single-best_of_abs512_pos512"]:
            v = np.array([c["diff"][k] for c in cells if c["ber"] == ber])
            summary[f"{ber:g}/{k}"] = dict(cells=int(v.size), ahead=int((v > 0).sum()), below_half_point=int((v < -0.5).sum()),
                                           median=round(float(np.median(v)), 2), low=round(float(v.min()), 2), high=round(float(v.max()), 2))
    (out_dir / "summary.json").write_text(json.dumps(dict(status="exploratory, post hoc", summary=summary, cells=cells), indent=1))
    for k, v in summary.items():
        print(f"{k:<44} {v}")


if __name__ == "__main__":
    main()
