"""Statistics of the partial-group search comparison (plan review/PARTIAL_GROUP_SEARCH_PLAN_2026-11-14.md)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workflows"))
from runR_partial_search_compare import holm, t_quantile, t_two_sided  # noqa: E402


@pytest.mark.parametrize("t,df,p", [(2.776445, 4, 0.05), (4.604095, 4, 0.01), (2.131847, 4, 0.10), (0.0, 4, 1.0),
                                    (2.570582, 5, 0.05), (12.7062, 1, 0.05)])
def test_t_two_sided_matches_tables(t, df, p):
    assert t_two_sided(t, df) == pytest.approx(p, rel=1e-4, abs=1e-6)


def test_t_quantile_and_holm():
    assert t_quantile(0.95, 4) == pytest.approx(2.131847, rel=1e-5)
    assert np.allclose(holm(np.array([0.01, 0.04, 0.03, 0.2])), [0.04, 0.09, 0.09, 0.2])
