"""Synthetic KWS search campaign boundaries; no KWS input or model access."""

from types import SimpleNamespace

import pytest

from netsurf.faults import FaultEvent, FaultModel, FaultSchedule
from netsurf.identities import BitIdentity
from netsurf.protection import ProtectionMask
from netsurf.equal_compute_kws_campaign import _fields, _mask, _seeds, _verify_final_result


def test_independent_seed_families_and_deterministic_bernoulli_fields():
    seeds = _seeds()
    values = [*seeds["selection_fields"], *seeds["final_fields"],
              *seeds["isolated_bits"], *seeds["search_random"],
              *seeds["search_greedy"], seeds["analysis"]]
    assert len(values) == len(set(values))
    bits = tuple(BitIdentity("tiny.weight", i, 0) for i in range(1024))
    assert [x.digest() for x in _fields(bits, seeds["selection_fields"][:2], 3)] == [
        x.digest() for x in _fields(bits, seeds["selection_fields"][:2], 3)]
    assert _fields(bits, seeds["selection_fields"][:1], 3)[0].seed != \
        _fields(bits, seeds["final_fields"][:1], 3)[0].seed


def test_full_global_budget_and_field_accounting():
    outside = frozenset(BitIdentity("outside.weight", i, 0) for i in range(21992))
    candidates = tuple(BitIdentity("panel.weight", i, 7) for i in range(24))
    context = SimpleNamespace(outside=outside)
    protected = _mask("chosen", candidates, context)
    assert protected.universe_size == 176128
    assert len(protected.protected) == protected.budget == 22016
    with pytest.raises(ValueError, match="global budget"):
        _mask("short", candidates[:-1], context)

    events = (FaultEvent(None, candidates[0]), FaultEvent(None, BitIdentity("outside.weight", 3, 0)))
    schedule = FaultSchedule(FaultModel("weight", "logical_trainable_parameter_bits",
        "persistent_trial", "independent_bernoulli", .01), 5, events, 176128, 3)
    masks = {"chosen": protected}
    result = {"seed": 5, "schedule_sha256": schedule.digest(),
        "events": [event.key() for event in events],
        "policies": {"chosen": {"schedule_sha256": schedule.digest(),
            "report": {"proposed_events": 2, "applied_events": 0,
                       "suppressed_events": 2},
            "mean_ce": 1.0, "accuracy": .5}}}
    _verify_final_result(result, schedule, masks)
    result["policies"]["chosen"]["report"]["applied_events"] = 1
    with pytest.raises(ValueError, match="accounting"):
        _verify_final_result(result, schedule, masks)
