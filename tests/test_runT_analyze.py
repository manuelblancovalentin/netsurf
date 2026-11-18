"""Deterministic checks of the Run T analysis on fabricated data."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workflows"))
import runT_analyze as A  # noqa: E402

RATES = (1e-4, 1e-3, 1e-2, 1e-1)
UNITS = [(r, d) for r in range(6) for d in range(4)]


def fake(effects):
    """effects: {condition: {rate: accuracy %}}; unit noise is shared across conditions (paired)."""
    rng = np.random.default_rng(1)
    noise = {u: rng.normal(0, 0.3) for u in UNITS}
    return {(c, r): {u: v + noise[u] for u in UNITS} for c, curve in effects.items() for r, v in curve.items()}


def signs():
    return np.random.default_rng(A.SEED).choice((-1.0, 1.0), size=(20_000, len(UNITS)))


def test_tolerated_requires_every_lower_rate():
    assert A.tolerated({1e-4: 90, 1e-3: 89, 1e-2: 80, 1e-1: 10}, RATES, 90) is 1e-3
    assert A.tolerated({1e-4: 84, 1e-3: 89, 1e-2: 89, 1e-1: 89}, RATES, 90) is None       # fails at the lowest rate
    assert A.tolerated({1e-4: 90, 1e-3: 84, 1e-2: 89, 1e-1: 89}, RATES, 90) is 1e-4       # a later recovery does not count
    assert A.tolerated({r: 90 for r in RATES}, RATES, 90) is 1e-1


def test_planted_contrast_is_resolved_and_usability_is_respected():
    acc = fake({"tmr_abs@0.125": {1e-4: 90, 1e-3: 90, 1e-2: 88, 1e-1: 40},
                "tmr_msb@0.125": {1e-4: 90, 1e-3: 89.9, 1e-2: 80, 1e-1: 30},
                "unprotected": {1e-4: 89, 1e-3: 80, 1e-2: 30, 1e-1: 10}})
    curves, tol, rows = A.analyze(acc, 90.0, signs())
    assert tol["tmr_abs@0.125"] == 1e-2 and tol["tmr_msb@0.125"] == 1e-3 and tol["unprotected"] == 1e-4
    fortune = {r["ber"]: r for r in rows if r["right"] == "tmr_msb@0.125"}
    assert set(fortune) == {1e-4, 1e-3, 1e-2}                       # 1e-1: neither arm is usable, not tested
    assert fortune[1e-2]["resolved"] and abs(fortune[1e-2]["diff"] - 8) < 1e-9
    assert not fortune[1e-3]["resolved"]                            # 0.1 point: below the 0.5-point threshold
    assert not fortune[1e-4]["resolved"] and fortune[1e-4]["p_raw"] == 1.0   # identical: exact zero differences
    assert all(r["draws"] == 24 for r in rows)


def test_missing_arms_are_skipped_and_holm_is_monotone():
    acc = fake({"tmr_abs@0.125": {r: 90 for r in RATES}, "unprotected": {r: 85.5 for r in RATES}})
    _, _, rows = A.analyze(acc, 90.0, signs())
    assert {r["contrast"] for r in rows} == {"best ranked TMR at 25% vs no protection"}
    assert all(r["resolved"] for r in rows)
    adj = A.holm([0.01, 0.04, 0.03, 0.5])
    assert np.allclose(adj, [0.04, 0.09, 0.09, 0.5])
