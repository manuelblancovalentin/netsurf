"""Deterministic checks of the Run A (TFLite confirmation) analysis helpers and tracked outputs."""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "publications/paper1_review"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REVIEW / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ra = _load("tecs_runA_analysis")


def test_t_interval_matches_hand_computation():
    mean, low, high, p = ra.t_ci([0.01, 0.03, 0.02])
    # sd = 0.01, se = 0.01/sqrt(3), t(0.975, 2) = 4.302653
    assert abs(mean - 0.02) < 1e-12
    assert abs((high - mean) - 4.302652729911275 * 0.01 / np.sqrt(3)) < 1e-9
    assert abs((mean - low) - (high - mean)) < 1e-12
    assert 0 < p < 1
    assert ra.t_ci([0.0, 0.0, 0.0])[1:] == (0.0, 0.0, 1.0)


def test_holm_is_monotone_and_bounded():
    adjusted = ra.holm([0.01, 0.04, 0.03, 0.5])
    assert np.allclose(adjusted, [0.04, 0.09, 0.09, 0.5])
    assert ra.holm([1e-5] * 200).max() == 200 * 1e-5


def test_signflip_is_seeded_and_floored():
    d = np.full(24, 0.01)
    p1 = ra.signflip_p(d, np.random.default_rng(1))
    p2 = ra.signflip_p(d, np.random.default_rng(1))
    assert p1 == p2 and p1 >= 1 / (ra.DRAWS + 1)
    assert ra.signflip_p(np.zeros(8), np.random.default_rng(1)) == 1.0


def test_tracked_decision_rule_is_recomputable_from_tracked_contrasts():
    summary = json.loads((REVIEW / "data/tecs_runA/summary.json").read_text())
    rows = list(csv.DictReader(open(REVIEW / "data/tecs_runA/absgrad_contrasts.csv")))
    for model, expect_confirmed in (("resnet8", True), ("kws", False)):
        sel = [r for r in rows if r["model"] == model and r["risk"] == "err" and r["competitor_set"] == "primary"
               and r["decision_useful_primary"] == "1"]
        cells = {}
        for r in sel:
            key = (r["ber"], r["coverage"])
            cells[key] = max(cells.get(key, -1.0), float(r["rep_ci_high"]))
        rule = summary["models"][model]["decision_rule"]["primary/decision_useful_primary"]
        assert len(cells) == rule["cells"]
        assert abs(max(cells.values()) - rule["max_replication_upper_bound"]) < 1e-12
        assert all(v < ra.BOUND for v in cells.values()) is expect_confirmed
        assert rule["confirmed_replication_level"] is expect_confirmed
        assert summary["models"][model]["abs_grad_holm_resolved_in_decision_useful_cells"]["losses"] == 0
        assert summary["models"][model]["complete_tier_max_abs_difference_msb_vs_tier_first_masks"] == 0.0
