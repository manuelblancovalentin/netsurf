"""Bounded tests for public PrioriFI scheduling and the NumPy adaptation."""

import pytest

from netsurf.identities import BitIdentity
from netsurf.priorifi import (
    ProbeRecord, candidate_tiers, cost_capped_local_allocation,
    measured_bit_allocation, probe_schedule, significance_tiers,
)
from netsurf.priorifi_numpy_reference import run
from netsurf.priorifi_mlperf_engineering import run as run_bounded_mlperf


def test_tiers_cover_complete_six_bit_words_once():
    bits = tuple(BitIdentity("w", word, bit) for word in range(3) for bit in range(6))
    tiers = significance_tiers(bits, (("w", 2), ("w", 0), ("w", 1)))
    assert [[bit.bit_index for bit in tier] for tier in tiers] == [[bit] * 3 for bit in range(5, -1, -1)]
    assert [bit.flat_index for bit in tiers[0]] == [2, 0, 1]
    assert set(bit for tier in tiers for bit in tier) == set(bits)


def test_feedback_scheduler_seeds_each_tier_and_uses_recent_metrics():
    tiers = tuple(tuple(BitIdentity("w", word, bit) for word in range(4))
                  for bit in (2, 1, 0))
    feedback = {2: 3.0, 1: 2.0, 0: 1.0}
    records = probe_schedule(tiers, lambda bit: feedback[bit.bit_index], last_k=2)
    assert len(records) == 12
    assert [record.tier for record in records[:3]] == [0, 1, 2]
    assert [record.tier for record in records[3:6]] == [0, 0, 0]
    assert len({record.identity for record in records}) == 12
    assert probe_schedule(tiers, lambda bit: feedback[bit.bit_index], last_k=2,
                          max_probes=5) == records[:5]


def test_ties_are_deterministic_and_measured_allocation_is_separate():
    tiers = ((BitIdentity("w", 0, 1), BitIdentity("w", 1, 1)),
             (BitIdentity("w", 0, 0), BitIdentity("w", 1, 0)))
    records = probe_schedule(tiers, lambda bit: 0.0, last_k=1)
    assert [record.tier for record in records] == [0, 1, 0, 1]
    assert measured_bit_allocation(records, 2) == (tiers[0][0], tiers[1][0])
    with pytest.raises(ValueError, match="budget"):
        measured_bit_allocation(records, 5)


def test_population_validation_and_nonfinite_feedback():
    bit = BitIdentity("w", 0, 0)
    with pytest.raises(ValueError, match="duplicate"):
        significance_tiers((bit, bit), (("w", 0),))
    with pytest.raises(ValueError, match="nonfinite"):
        probe_schedule(((bit,),), lambda _: float("nan"), last_k=1)


def test_candidate_tiers_and_cost_capped_global_budget():
    words = (("w", 0), ("w", 1), ("w", 2))
    eligible = tuple(BitIdentity("w", word, bit) for word in range(3) for bit in range(2))
    pool = (BitIdentity("w", 0, 1), BitIdentity("w", 2, 0), BitIdentity("w", 1, 1))
    tiers = candidate_tiers(eligible, pool, words)
    assert tiers == ((pool[0], pool[2]), (pool[1],))
    prior = (pool[0], pool[2], BitIdentity("w", 2, 1),
             BitIdentity("w", 0, 0), BitIdentity("w", 1, 0), pool[1])
    baseline = frozenset(prior[:3])
    observations = (ProbeRecord(pool[0], 0, 0, 0.0),
                    ProbeRecord(pool[2], 0, 1, 1.0),
                    ProbeRecord(pool[1], 1, 2, 3.0))
    result = cost_capped_local_allocation(prior, baseline, observations)
    assert result == frozenset((pool[2], pool[1], BitIdentity("w", 2, 1)))
    assert len(result) == len(baseline)
    assert result - set(pool) == baseline - set(pool)
    assert cost_capped_local_allocation(prior, baseline,
        tuple(ProbeRecord(row.identity, row.tier, row.sequence, 0.0) for row in observations)) == baseline
    assert cost_capped_local_allocation(prior, baseline, ()) == baseline
    with pytest.raises(ValueError, match="prefix"):
        cost_capped_local_allocation(prior, frozenset((prior[0], prior[3], prior[4])), observations)


def test_exact_numpy_engineering_smoke_is_deterministic_and_paired():
    left = run(trials=3)
    right = run(trials=3)
    for receipt in (left, right):
        assert receipt["population_bits"] == 102
        assert len(receipt["probe_records"]) == 102
        assert len({row["bit"] for row in receipt["probe_records"]}) == 102
        assert len(receipt["fault_trials"]) == 12
        assert len(receipt["masks"]["priorifi_derived_measured_prediction_change"]["bits"]) == 17
        partitions = receipt["sample_partitions"]
        assert not (set(partitions["calibration"]["indices"]) &
                    set(partitions["bit_validation"]["indices"]) |
                    set(partitions["calibration"]["indices"]) &
                    set(partitions["final"]["indices"]) |
                    set(partitions["bit_validation"]["indices"]) &
                    set(partitions["final"]["indices"]))
        for trial in range(3):
            rows = [row for row in receipt["fault_trials"] if row["trial"] == trial]
            assert len({row["field_sha256"] for row in rows}) == 1
            assert all(row["proposed"] == row["applied"] + row["suppressed"] for row in rows)
            full = next(row for row in rows if row["method"] == "full_protection")
            assert full["applied"] == 0
            assert full["loss"] == receipt["clean_final_loss"]
    left.pop("runtime_seconds")
    right.pop("runtime_seconds")
    assert left == right


def test_mlperf_pilot_rejects_unbounded_or_empty_candidate_design_before_asset_loading():
    with pytest.raises(ValueError, match="engineering bounds"):
        run_bounded_mlperf(sign_per_status=0)
    with pytest.raises(ValueError, match="engineering bounds"):
        run_bounded_mlperf(max_probes=513, sign_per_status=256, lower_per_tier=64)
    with pytest.raises(ValueError, match="engineering bounds"):
        run_bounded_mlperf(trials=17)
