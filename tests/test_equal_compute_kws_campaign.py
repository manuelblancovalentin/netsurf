"""Synthetic KWS search campaign boundaries; no KWS input or model access."""

from types import SimpleNamespace

import pytest
import numpy as np

from netsurf.faults import FaultEvent, FaultModel, FaultSchedule, InjectionReport
from netsurf.identities import BitIdentity
from netsurf.protection import ProtectionMask
from netsurf.equal_compute_kws_campaign import _fields, _mask, _seeds, _verify_final_result
import netsurf.equal_compute_kws_campaign as campaign


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


def test_synthetic_selection_seal_final_and_offline_aggregation(tmp_path, monkeypatch):
    candidates = tuple(BitIdentity("panel.weight", i, 7) for i in range(72))
    universe = candidates
    original = frozenset(candidates[:24])
    context = SimpleNamespace(outside=frozenset(), original=original)
    data = (np.zeros((2, 1), dtype=np.int8), np.zeros(2, dtype=np.int64))
    class FakeAdapter:
        def evaluate_fault_schedule(self, inputs, labels, schedule, protection):
            applied = [event for event in schedule.events if event.bit not in protection.protected]
            ce = .5 + .0001 * len(applied) + sum(.01 for event in applied
                                                  if event.bit in set(candidates))
            return SimpleNamespace(categorical_crossentropy=ce, accuracy=1.), InjectionReport(
                schedule.digest(), len(schedule.events), len(applied),
                len(schedule.events) - len(applied), len(applied))
    adapter = FakeAdapter()
    seeds = _seeds(); seeds["selection_fields"] = seeds["selection_fields"][:2]
    plan = {"plan_sha256": "a" * 64,
            "initial_mask": [bit.key() for bit in original],
            "candidate_bits": [bit.key() for bit in candidates],
            "selection_window": {"identity_records_sha256": "b" * 64},
            "seeds": seeds}
    monkeypatch.setattr(campaign, "SELECTION_FIELDS", 2)
    monkeypatch.setattr(campaign, "SELECTION_CLIPS", 2)
    monkeypatch.setattr(campaign, "FINAL_CLIPS", 2)
    monkeypatch.setattr(campaign, "QUERIES", 3)
    monkeypatch.setattr(campaign, "_mask", lambda name, local, context: ProtectionMask(
        name, frozenset(local), 72, 24, {}))
    monkeypatch.setattr(campaign, "_static", lambda path: plan)
    monkeypatch.setattr(campaign, "_runtime", lambda plan, final: (
        adapter, universe, candidates, context, data))
    monkeypatch.setattr(campaign, "_source_config", lambda: None)
    monkeypatch.setattr(campaign, "_storage", lambda _: SimpleNamespace(bit_identities=lambda: universe))
    monkeypatch.setattr(campaign, "_context", lambda *args: context)
    plan_path = tmp_path / "plan.json"
    run_root = tmp_path / "run"
    assert campaign.run_selection(plan_path, run_root)["status"] == "SELECTION_COMPLETE"
    assert campaign.seal(plan_path, run_root)["artifacts"] == 8
    assert campaign.run_final(plan_path, run_root)["fields"] == 128
    analysis = campaign.aggregate(plan_path, run_root)
    assert len(analysis["comparisons"]) == 5
    assert analysis["comparisons"][0]["contrast"] == "greedy_minus_random"
    assert len(analysis["risk_rows"]) == 8
    # A partial field or changed sealed query must never silently be restarted.
    (run_root / "final" / "field_000.json").unlink()
    with pytest.raises(ValueError, match="partial final"):
        campaign.run_final(plan_path, run_root)
