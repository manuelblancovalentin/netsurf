"""Bounded tests for public PrioriFI scheduling and the NumPy adaptation."""

import pytest

from netsurf.identities import BitIdentity
from netsurf.priorifi import measured_bit_allocation, probe_schedule, significance_tiers
from netsurf.priorifi_numpy_reference import run


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
