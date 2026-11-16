"""Deterministic checks of the blur-confirmation rules on fabricated data (no model, no TensorFlow)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workflows"))
import runG_blurconf_analyze as A  # noqa: E402

RATES, BUDGETS = (3e-4, 1e-3, 3e-3, 1e-2), (0.125, 0.25)


def fake(rng, gain_low, model_bias=0.0, model_gain=None, real_bias=0.0, arms=("abs1024", "pos1024", "msb")):
    """One replication: abs loses 100*p*3 points... pos = abs + gain at low rates, far behind at 1e-2; msb far behind."""
    meas, mod = {}, {}
    mg = gain_low if model_gain is None else model_gain
    for b in BUDGETS:
        for p in RATES:
            base = 0.90 - 3 * p
            noise = rng.normal(0, 0.002, 8)
            vals = {"abs1024": base + noise, "pos1024": base + noise + (gain_low if p <= 3e-3 else -0.03),
                    "msb": base + noise - 2 * p - 0.008}
            mvals = {"abs1024": base + noise, "pos1024": base + noise + (mg if p <= 3e-3 else -0.03),
                     "msb": base + noise - 2 * p - 0.008}
            for arm in arms:
                meas[A.key(arm, b, p)] = list(vals[arm])
                for v in A.VARIANTS:
                    shift = real_bias if v == "real" else model_bias
                    mod[A.key(arm, b, p) + (v,)] = list(mvals[arm] + shift)
    return mod, meas, 0.90


def run(n_rep=3, **kw):
    rng = np.random.default_rng(7)
    mod, meas, clean = {}, {}, {}
    for r in range(n_rep):
        mod[r], meas[r], clean[r] = fake(rng, **kw)
    return A.checkpoint(mod, meas, clean)[0]


def test_regime_rule():
    assert A.regime({(0.125, 1e-3): np.full(16, 0.5) + np.linspace(-0.1, 0.1, 16)})[0] == "two regimes"
    assert A.regime({(0.125, 1e-3): np.linspace(-1, 1.3, 16)})[0] == "one regime"          # mean 0.15, below 0.2
    assert A.regime({(0.125, 1e-2): np.full(16, 2.0) + np.linspace(-0.1, 0.1, 16)})[0] == "one regime"   # not a low rate
    assert A.regime({(0.125, 1e-3): np.array([5.0] + [0.0] * 15)})[0] == "one regime"      # mean 0.31 but under 2.5 SE


def test_faithful_model_passes_and_reproduces_two_regimes():
    c = run(gain_low=0.006)
    assert c["G0"] and c["C1"] and c["C3"]
    assert c["regime_measured"] == c["regime_hybrid"] == "two regimes"
    assert c["C2_counted"] > 0 and c["C2_agree"] == c["C2_counted"]


def test_one_regime_reproduced():
    c = run(gain_low=0.0)
    assert c["regime_measured"] == c["regime_hybrid"] == "one regime" and c["C3"]


def test_biased_model_fails_c1_only():
    c = run(gain_low=0.006, model_bias=-0.015)
    assert c["G0"] and not c["C1"] and c["C3"]


def test_model_missing_the_gain_fails_c3():
    c = run(gain_low=0.006, model_gain=0.0)
    assert c["G0"] and c["C1"] and not c["C3"]
    assert (c["regime_measured"], c["regime_hybrid"]) == ("two regimes", "one regime")


def test_unfaithful_path_fails_everything():
    c = run(gain_low=0.006, real_bias=0.02)
    assert not c["G0"] and not c["C1"] and not c["C3"]


def test_unusable_cells_are_excluded():
    rng = np.random.default_rng(3)
    mod, meas, clean = fake(rng, gain_low=0.0)
    for k in list(meas):
        if k[2] == round(1e-2, 8):
            meas[k] = [v - 0.2 for v in meas[k]]           # far below clean: not usable, and the model is wrong there
    c = A.checkpoint({0: mod}, {0: meas}, {0: clean})[0]
    assert c["usable_cells"] == c["cells"] - 2 * 3 and c["C1"]


def test_verdict():
    good = dict(G0=True, C1=True, C3=True, C2_counted=10, C2_agree=10)
    per = {n: dict(good) for n in A.R8 + A.GEN}
    assert A.verdict(per)["outcome"] == "CONFIRMED"
    for n in A.R8[:3]:
        per[n] = dict(good, C3=False)                      # only 5 of 8 ResNet-8 regimes reproduced
    assert A.verdict(per)["outcome"] == "PARTIAL"
    per = {n: dict(good, C1=False) for n in A.R8 + A.GEN}
    assert A.verdict(per)["outcome"] == "FAILED"
    per = {n: dict(good) for n in A.R8 + A.GEN}
    for n in A.GEN:
        per[n] = dict(good, C2_agree=8)                    # 80% < 85% in one family
    assert A.verdict(per)["outcome"] == "PARTIAL"
