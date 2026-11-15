"""Statistics of the pre-registered partial-search follow-up (review/PARTIAL_SEARCH_FOLLOWUP_PREREG_2026-11-14.md)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workflows"))
from runR_psearch_followup_compare import cell_stats, family  # noqa: E402


def test_cell_stats_mean_se_t_and_interval():
    s = cell_stats([1.0, 2.0, 3.0, 4.0, 5.0])
    se = np.std([1, 2, 3, 4, 5], ddof=1) / np.sqrt(5)
    assert s["mean"] == 3.0 and s["se"] == pytest.approx(se) and s["t"] == pytest.approx(3.0 / se)
    assert s["ci90"] == pytest.approx([3.0 - 2.131847 * se, 3.0 + 2.131847 * se], rel=1e-5)
    assert not s["equivalent_within_1"]
    assert cell_stats([0.1, -0.1, 0.05, -0.05, 0.0])["equivalent_within_1"]


def test_family_applies_holm_over_the_whole_family_and_signs():
    rows = family([dict(mean=1.0, p=0.01), dict(mean=-1.0, p=0.012), dict(mean=1.0, p=0.04), dict(mean=1.0, p=0.2)])
    assert [r["p_holm"] for r in rows] == pytest.approx([0.04, 0.04, 0.08, 0.2])
    assert [r["resolved"] for r in rows] == ["ahead", "behind", "unresolved", "unresolved"]
