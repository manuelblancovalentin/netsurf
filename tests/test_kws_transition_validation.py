from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np

from netsurf.cross_pool_analysis import cross_pool_statistics


def test_new_plan_uses_independent_pool_and_final_cohort_geometry():
    path = Path(__file__).parents[1] / "config/experiments/kws_transition_independent_pools_p05_v1.toml"
    plan = tomllib.loads(path.read_text())
    design, samples = plan["design"], plan["samples"]
    assert plan["schema_version"] == "1.1.0"
    assert design["logical_ber"] == 0.05
    assert len(set(design["pool_seeds"])) == 32
    assert len(set(design["final_seeds"])) == 32
    assert samples["final_offset"] + 3048 == 4474
    assert samples["final_count"] == 384
    assert 4474 + 384 <= 4890


def test_new_primary_contrast_remaps_holm_family_without_changing_effects():
    effects = np.zeros((4, 3, 2, 3), dtype=np.float64)
    effects[:, :, :, 0] = -1
    effects[:, :, :, 1] = np.asarray([-2, -1, 1, 2])[:, None, None]
    effects[:, :, :, 2] = 1
    old = cross_pool_statistics(effects, seed=9, replicates=100, primary_index=0)
    new = cross_pool_statistics(effects, seed=9, replicates=100, primary_index=1)
    assert [x["pool_mean_effect"]["mean"] for x in old["comparisons"]] == [
        x["pool_mean_effect"]["mean"] for x in new["comparisons"]]
    assert new["comparisons"][1]["role"] == "primary"
    assert "secondary_holm_p_value" not in new["comparisons"][1]
    assert "secondary_holm_p_value" in new["comparisons"][0]
    assert "secondary_holm_p_value" in new["comparisons"][2]
