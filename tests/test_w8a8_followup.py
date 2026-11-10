from __future__ import annotations

import pytest

from netsurf.transformer_risk import language_model_risk_from_stats
from netsurf.w8a8_followup import _risk_diagnostic


def test_finite_cap_is_not_a_nonfinite_failure():
    risk = language_model_risk_from_stats(((0.0, 1.0), (0.0, 31.0), None))
    row = _risk_diagnostic(risk)
    assert row["metrics"]["mean_nll"] == pytest.approx((1 + 30 + 30) / 3)
    assert row["failed_units"] == 1
    assert row["finite_nll_at_or_above_30_count"] == 1
    assert row["finite_nll_uncapped_mean"] == pytest.approx(16.0)
    assert row["cap_or_failure_fraction"] == pytest.approx(2 / 3)


def test_all_finite_tokens_can_exceed_cap_without_failed_units():
    risk = language_model_risk_from_stats(((0.0, 40.0), (0.0, 60.0)))
    row = _risk_diagnostic(risk)
    assert row["failed_units"] == 0
    assert row["finite_nll_at_or_above_30_count"] == 2
    assert row["finite_nll_uncapped_mean"] == 50.0
    assert row["metrics"]["mean_nll"] == 30.0
