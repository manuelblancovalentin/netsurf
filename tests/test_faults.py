import numpy as np

from netsurf.faults import FaultEvent, FaultModel, FaultSchedule, apply_persistent_schedule
from netsurf.identities import BitIdentity
from netsurf.protection import ProtectionMask
from netsurf.quantization import BitCodec, FixedPointFormat


def model(probability=0.5):
    return FaultModel(
        target="weight",
        location="logical",
        persistence="persistent_trial",
        distribution="independent_bernoulli",
        probability=probability,
    )


def test_schedule_replay_is_deterministic():
    universe = tuple(BitIdentity("w", scalar, bit) for scalar in range(3) for bit in range(6))
    first = FaultSchedule.generate(model(), universe, seed=77, inference_count=12)
    second = FaultSchedule.generate(model(), universe, seed=77, inference_count=12)
    assert first == second
    assert first.digest() == second.digest()


def test_zero_error_and_full_protection_boundaries():
    universe = tuple(BitIdentity("w", 0, bit) for bit in range(6))
    zero = FaultSchedule.generate(model(0.0), universe, seed=1, inference_count=1)
    assert zero.events == ()
    all_faults = FaultSchedule.generate(model(1.0), universe, seed=1, inference_count=1)
    full = ProtectionMask("full", frozenset(universe), len(universe), 6.0, {})
    codec = BitCodec(FixedPointFormat.parse("q<6,2,1>"))
    corrupted, report = apply_persistent_schedule(
        {"w": np.array([0.5], dtype=np.float32)}, all_faults, full, codec
    )
    np.testing.assert_array_equal(corrupted["w"], np.array([0.5], dtype=np.float32))
    assert report.applied_events == 0
    assert report.suppressed_events == 6


def test_exact_xor_and_event_accounting():
    codec = BitCodec(FixedPointFormat.parse("q<6,2,1>"))
    events = (FaultEvent(None, BitIdentity("w", 0, 5)), FaultEvent(None, BitIdentity("w", 1, 0)))
    schedule = FaultSchedule(model(1.0), 4, events, universe_size=12, inference_count=2)
    protection = ProtectionMask("none", frozenset(), 12, 0.0, {})
    corrupted, report = apply_persistent_schedule(
        {"w": np.array([0.0, 0.0], dtype=np.float32)}, schedule, protection, codec
    )
    np.testing.assert_array_equal(corrupted["w"], np.array([-4.0, 0.125], dtype=np.float32))
    assert report.proposed_events == report.applied_events == 2
    assert report.suppressed_events == 0


def test_fault_delta_is_added_to_unrounded_checkpoint_value():
    codec = BitCodec(FixedPointFormat.parse("q<6,2,1>"))
    event = FaultEvent(None, BitIdentity("w", 0, 0))
    schedule = FaultSchedule(model(1.0), 8, (event,), universe_size=6, inference_count=1)
    protection = ProtectionMask("none", frozenset(), 6, 0.0, {})
    original = np.array([0.01], dtype=np.float32)
    corrupted, _ = apply_persistent_schedule({"w": original}, schedule, protection, codec)
    # encode(0.01) == 0; flipping its LSB has delta +0.125. The historical
    # checkpoint value is retained rather than silently rounded to zero.
    np.testing.assert_allclose(corrupted["w"], np.array([0.135], dtype=np.float32))
