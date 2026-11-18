"""Bounded guards for offline recovery of a failed Pythia wide-BER run."""

import json

import pytest

from scripts.recover_pythia_wide_ber_subset import clean_subset, summarize
from netsurf.w8a8_campaign import INTERIOR, METHODS


def _record(value):
    return {"metrics": {"mean_nll": value}, "total_units": 4096}


def _put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_only_three_exact_clean_shards_are_accepted(tmp_path):
    for shard in range(4):
        _put(tmp_path / "clean" / f"shard_{shard:02d}.json",
             _record(4.0 if shard == 0 else 3.0))
        if shard:
            _put(tmp_path / "clean" / f"shard_{shard:02d}_end.json", _record(3.0))
    assert clean_subset(tmp_path, 4) == _record(3.0)

    _put(tmp_path / "clean" / "shard_02_end.json", _record(3.1))
    with pytest.raises(ValueError, match="did not restore"):
        clean_subset(tmp_path, 4)


def test_recovery_rejects_completed_or_nonmatching_source(tmp_path):
    for shard in range(4):
        _put(tmp_path / "clean" / f"shard_{shard:02d}.json", _record(3.0))
        if shard:
            _put(tmp_path / "clean" / f"shard_{shard:02d}_end.json", _record(3.0))
    with pytest.raises(ValueError, match="recorded shard-zero failure"):
        clean_subset(tmp_path, 4)
    (tmp_path / "FINALIZED").write_text("existing run")
    with pytest.raises(ValueError, match="unfinalized"):
        clean_subset(tmp_path, 4)


def test_paired_summary_uses_all_eighteen_accepted_fields():
    bits = 800
    policies = [{"name": "unprotected", "budget": 0, "mask_digest": "none"},
                {"name": "full_protection", "budget": bits, "mask_digest": "full"}]
    policies.extend({"name": f"{method}_b{int(bits * fraction)}",
                     "budget": int(bits * fraction), "mask_digest": "mask"}
                    for method in METHODS for fraction in INTERIOR)
    plan = {"bers": [1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
            "eligible_bits": bits, "target": "pythia160m_int8", "policies": policies}
    rows = []
    for arm, ber in enumerate(plan["bers"]):
        for index in range(24):
            if index % 4 == 0:
                continue
            for policy in policies:
                # Deliberately unequal field baselines: a wrongly unpaired
                # comparison does not cancel this term.
                offset = -.2 if policy["name"].startswith("random_") else 0.0
                rows.append({"arm": arm, "field_index": index, "policy": policy["name"],
                             "mean_nll_nats": 4.0 + index + offset,
                             "failed_units": 0})
    conditions, comparisons = summarize(rows, plan, _record(3.0))
    assert len(conditions) == 208
    assert len(comparisons) == 160
    assert {row["field_count"] for row in conditions + comparisons} == {18}
    random_vs_msb = [row for row in comparisons if row["compared_method"] == "random"]
    assert all(row["mean_nll_difference_nats"] == pytest.approx(-.2) for row in random_vs_msb)
