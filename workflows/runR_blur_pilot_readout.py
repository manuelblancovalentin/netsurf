#!/usr/bin/env python3
"""EXPLORATORY, post hoc: does accuracy follow from one bell curve per example?

Uses only the locked bell data of the blur pilot (64 draws of real flips and 64 of two-moment Gaussian weight noise,
400 evaluation examples, masks none / abs1024 / pos1024 at 12.5%, rates 1e-3, 3e-3, 1e-2).  For each example i the
faulty margin M_i (true logit minus best other) has a mean mu_i and spread s_i over draws; the bell-curve readout is

    accuracy ~ (1/N) sum_i Phi(mu_i / s_i)

compared with the observed accuracy of the real flips (mean over draws).  Two versions:
  in-sample  mu_i, s_i from the same real-flip draws (tests only the bell shape per example)
  gauss      mu_i, s_i from the Gaussian-noise draws (independent draws; tests the two-moment replacement too)
Output: <out>/readout.csv, <out>/summary.json (write-once).
"""
from __future__ import annotations

import csv
import json
import sys
from math import erf, sqrt
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from site_paths import WT  # noqa: E402

RUNS = WT / "paper1-tecs-revision-v1/workspace/runs"
SRC = RUNS / "paper1_blur_pilot_v1/bell"
OUT = RUNS / "paper1_blur_pilot_readout_v1"
PHI = np.vectorize(lambda z: 0.5 * (1.0 + erf(z / sqrt(2.0))))


def bell_readout(M):
    mu, s = M.mean(0), M.std(0, ddof=1)
    z = np.where(s > 0, mu / np.where(s > 0, s, 1.0), np.where(mu > 0, np.inf, -np.inf))
    return float(PHI(np.clip(z, -40, 40)).mean())


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    rows = []
    for d in sorted(SRC.iterdir()):
        z = np.load(d / "margins.npz")
        clean = z["clean"]
        for key in sorted({k.rsplit("|", 2)[0] for k in z.files if k != "clean"}):
            arm, p = key.split("|")
            real, gauss = z[f"{key}|real|margin"], z[f"{key}|gauss|margin"]
            observed = float((real > 0).mean())
            observed_sd = float((real > 0).mean(1).std(ddof=1) / np.sqrt(real.shape[0]))
            rows.append(dict(model=d.name, arm=arm, ber=float(p), clean=float((clean > 0).mean()), observed=observed,
                             observed_se=observed_sd, bell_in_sample=bell_readout(real), bell_from_gauss=bell_readout(gauss),
                             gauss_observed=float((gauss > 0).mean())))
    with open(OUT / "readout.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    err = {k: np.array([100 * (r[k] - r["observed"]) for r in rows]) for k in ("bell_in_sample", "bell_from_gauss", "gauss_observed")}
    summary = dict(status="EXPLORATORY, post hoc", conditions=len(rows),
                   **{k: dict(mean_abs_points=float(np.abs(v).mean()), worst_points=float(v[np.argmax(np.abs(v))]),
                              mean_signed_points=float(v.mean())) for k, v in err.items()})
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    for r in rows:
        print(f"{r['model']:12s} {r['arm']:8s} {r['ber']:<6g} observed {100*r['observed']:5.1f} ±{100*r['observed_se']:.1f} | "
              f"bell in-sample {100*r['bell_in_sample']:5.1f} | bell from Gaussian draws {100*r['bell_from_gauss']:5.1f} | "
              f"Gaussian draws directly {100*r['gauss_observed']:5.1f}")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
