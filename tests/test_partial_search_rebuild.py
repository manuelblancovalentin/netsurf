"""The partial-group search refuses scores that do not rebuild the stored abs1024 masks (plan
review/PARTIAL_GROUP_SEARCH_PLAN_2026-11-14.md, Phase 0)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workflows"))
from runR_partial_search import rebuild_differences  # noqa: E402
from netsurf.mean_variance import top_budget  # noqa: E402


def test_rebuild_check_is_exact_and_catches_a_perturbed_score():
    rng = np.random.default_rng(5)
    s = rng.random((50, 8))
    counts = {0.125: 50, 0.25: 100}
    stored = {f: top_budget(s, c) for f, c in counts.items()}
    assert rebuild_differences(s, stored, counts) == {"0.125": 0, "0.25": 0}
    t = s.copy()
    lo = np.unravel_index(np.argmin(np.where(stored[0.125], np.inf, s)), s.shape)   # lowest-scored unselected bit
    t[lo] = 10.0                                                                    # now outranks everything
    d = rebuild_differences(t, stored, counts)
    assert d["0.125"] == 1 and d["0.25"] == 1
