"""The language-model analysis on fabricated outputs (sign convention for NLL, thresholds, layout)."""
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workflows"))
import runT_analyze_lm as L  # noqa: E402


def write(base, nll_by_cond, draws=4, clean=3.0):
    rng = np.random.default_rng(0)
    for d in range(draws):
        for i, (p, conds) in enumerate(nll_by_cond.items()):
            out = base / f"draw{d}" / f"ber{i}"
            out.mkdir(parents=True)
            with open(out / "results.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=["condition", "ber", "metric_mean_nll", "metric_accuracy"])
                w.writeheader()
                noise = rng.normal(0, 0.002)
                for c, v in conds.items():
                    w.writerow(dict(condition=c, ber=p, metric_mean_nll=v + noise, metric_accuracy=0.9))
            (out / "DONE").write_text(json.dumps(dict(clean=dict(mean_nll=clean, accuracy=0.9))))


def test_nll_lower_is_better_and_thresholds(tmp_path):
    base = tmp_path / "qwen3_0_6b_int8"
    write(base, {1e-4: {"tmr_abs@0.125": 3.0, "tmr_msb@0.125": 3.0, "unprotected": 3.1},
                 1e-3: {"tmr_abs@0.125": 3.1, "tmr_msb@0.125": 3.3, "unprotected": 3.9},
                 1e-2: {"tmr_abs@0.125": 4.0, "tmr_msb@0.125": 6.0, "unprotected": 9.0}})
    acc, clean = L.load(base, L.SPEC["qwen3_0_6b_int8"])
    assert clean == -3.0 and acc[("tmr_abs@0.125", 1e-3)][(0, 0)] < -3.0
    L.A.MARGIN, L.A.MIN_DIFF = 0.5, 0.05
    signs = np.random.default_rng(1).choice((-1.0, 1.0), size=(2000, 4))
    curves, tol, rows = L.A.analyze(acc, clean, signs)
    assert tol["tmr_abs@0.125"] == 1e-3 and tol["unprotected"] == 1e-4
    fortune = {r["ber"]: r for r in rows if r["right"] == "tmr_msb@0.125"}
    assert set(fortune) == {1e-4, 1e-3}                     # 1e-2: neither within 0.5 nats
    assert fortune[1e-3]["diff"] > 0.19                     # abs better: higher minus-NLL
