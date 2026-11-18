"""Synthetic tests of the gradient-trust Phase 3 analysis (review/GRADIENT_TRUST_PLAN_2026-11-15.md), with planted
regimes, written before any Phase 2 fault run."""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workflows"))
from runG_gradtrust import predict  # noqa: E402
from runG_gradtrust_analyze import decide, holm, main, measured_regime, signflip_p  # noqa: E402

RATES = (1e-4, 1e-3, 3e-3, 1e-2, 3e-2)
WINDOW = 1000


def synth(pos_gain_low: float, rng, collapse=0.3):
    """pos1024 ahead by `pos_gain_low` (fraction) at the two lowest rates, behind by 0.02 above; accuracy falls with
    rate; 6 replications x 8 fields; integer counts of a 1,000-input window."""
    rows = []
    for b in (0.125, 0.25):
        for i, p in enumerate(RATES):
            base = 0.9 - collapse * i / len(RATES)
            for r in range(6):
                for f in range(8):
                    a = base + rng.normal(0, 0.004)
                    d = pos_gain_low if i < 2 else -0.02
                    for arm, v in (("abs1024", a), ("pos1024", a + d + rng.normal(0, 0.002))):
                        rows.append(dict(rep=r, field=f, ber=p, budget=b, arm=arm, window=WINDOW,
                                         accuracy=round(float(v) * WINDOW) / WINDOW))
    return rows


def test_planted_regimes_are_found():
    rng = np.random.default_rng(1)
    two, cells = measured_regime(synth(0.006, rng, collapse=0.1), 0.9, WINDOW)   # rate 3e-3 still usable
    assert two == "two regimes" and any(c["pos_ahead"] for c in cells) and any(c["abs_ahead"] for c in cells)
    one, cells = measured_regime(synth(0.0, rng), 0.9, WINDOW)
    assert one == "one regime" and not any(c["pos_ahead"] for c in cells)


def test_small_or_unusable_gains_do_not_count():
    rng = np.random.default_rng(2)
    regime, _ = measured_regime(synth(0.001, rng), 0.9, WINDOW)       # 0.1 point < 0.2 minimum
    assert regime == "one regime"
    regime, cells = measured_regime(synth(0.006, rng, collapse=2.0), 0.99, WINDOW)   # nothing within 5 points
    assert regime == "one regime" and cells == []


def test_signflip_and_holm():
    assert signflip_p(np.zeros(10)) == 1.0
    assert signflip_p(np.full(48, 3.0)) < 1e-4
    assert 0.3 < signflip_p(np.array([1.0, -1.0] * 24)) <= 1.0
    assert np.allclose(holm([0.01, 0.04, 0.03, 0.2]), [0.04, 0.09, 0.09, 0.2])


def test_prediction_cutoffs_and_decision_rule():
    assert [predict(v) for v in (0.21, 0.30, 0.35, 0.40, 0.47)] == ["one regime", "no prediction", "no prediction",
                                                                    "no prediction", "two regimes"]
    O, T = "one regime", "two regimes"
    assert decide([(O, O)] * 4 + [(T, T)] * 4 + [(T, O)])["outcome"] == "supported"
    assert decide([(T, T)] * 5)["outcome"] == "not tested both ways"
    assert decide([(O, T)] * 3 + [(T, T)] * 2)["outcome"] == "failed"
    assert decide([(O, O)] * 3 + [(T, T)] * 3 + [(O, T)] * 2)["outcome"] == "inconclusive"


def test_end_to_end_checks_the_frozen_predictions(tmp_path):
    rng = np.random.default_rng(3)
    faults = tmp_path / "faults"
    for cid, gain in (("a", 0.006), ("b", 0.0)):
        rows = synth(gain, rng)
        for r in range(6):
            for f in range(8):
                d = faults / cid / f"rep{r}" / f"field{f}"
                d.mkdir(parents=True)
                sel = [dict(checkpoint=cid, **x) for x in rows if x["rep"] == r and x["field"] == f]
                with open(d / "results.csv", "w", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=list(sel[0])); w.writeheader(); w.writerows(sel)
        with open(faults / cid / "clean.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["checkpoint", "rep", "clean_accuracy"]); w.writeheader()
            w.writerows([dict(checkpoint=cid, rep=r, clean_accuracy=0.9) for r in range(6)])
    pred = dict(checkpoints=dict(a=dict(D1=0.6, prediction="two regimes"), b=dict(D1=0.2, prediction="one regime")))
    pf = tmp_path / "predictions.json"
    pf.write_text(json.dumps(pred))
    sha = hashlib.sha256(pf.read_bytes()).hexdigest()
    main(faults, pf, sha, tmp_path / "out")
    res = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert res["checkpoints"]["a"]["measured"] == "two regimes" and res["checkpoints"]["b"]["measured"] == "one regime"
    assert res["decision"]["outcome"] == "supported"
