"""Model-free exponent-control state and schedule checks."""
import copy
import pytest

from netsurf.identities import BitIdentity
from netsurf.torch_storage import sparse_bernoulli_schedule
from netsurf.transformer_lm_exponent_control import BERS, SEED_BASE, _policies, _verify_field


class Panel:
    bit_count = 96
    bits = tuple(BitIdentity("linear.weight", i, 14) for i in range(96))

    def bit_at(self, ordinal):
        return self.bits[ordinal]


def _fixture():
    bits = Panel.bits
    names = ("random", "significance", "exponent_first", "exact_isolated")
    plan = {"plan_sha256": "a" * 64,
            "masks": {f"{name}_b{budget}": [bit.key() for bit in bits[:budget]]
                      for name in names for budget in (12, 24)}}
    masks = _policies(plan, bits)
    arm, field = 2, 5
    seed = SEED_BASE + arm * 100000 + field
    schedule = sparse_bernoulli_schedule(Panel(), ber=BERS[arm - 1],
                                         seed=seed, inference_count=256, max_events=96)
    identity = {"plan_sha256": plan["plan_sha256"], "arm": arm, "field": field,
                "ber": BERS[arm - 1], "seed": seed,
                "schedule_sha256": schedule.digest(),
                "events": [event.key() for event in schedule.events]}
    clean = {"total_units": 256, "mean_nll": 1.0}
    policies = {}
    for name, mask in masks.items():
        applied = sum(event.bit not in mask.protected for event in schedule.events)
        policies[name] = {"mask_sha256": mask.digest(),
            "expected_applied_faults": (96 - mask.budget) * BERS[arm - 1],
            "report": {"schedule_hash": schedule.digest(),
                       "proposed_events": len(schedule.events),
                       "applied_events": applied,
                       "suppressed_events": len(schedule.events) - applied},
            "risk": clean}
    record = {**identity, "policies": policies}
    intent = {**identity, "policy_names": sorted(masks)}
    return plan, masks, bits, clean, record, intent


def test_pairing_and_partial_field_replay_check():
    plan, masks, bits, clean, record, intent = _fixture()
    _verify_field(record, intent, plan, masks, bits, clean, 2, 5)
    changed = copy.deepcopy(record)
    changed["policies"]["exponent_first_b24"]["report"]["applied_events"] += 1
    with pytest.raises(ValueError, match="accounting"):
        _verify_field(changed, intent, plan, masks, bits, clean, 2, 5)
    changed = copy.deepcopy(record)
    changed["events"] = []
    with pytest.raises(ValueError, match="identity"):
        _verify_field(changed, intent, plan, masks, bits, clean, 2, 5)
    changed = copy.deepcopy(record)
    changed["policies"]["full"]["risk"] = {"total_units": 256, "mean_nll": 2.0}
    with pytest.raises(ValueError, match="full control"):
        _verify_field(changed, intent, plan, masks, bits, clean, 2, 5)


def test_all_sealed_policies_preserve_panel_budget():
    plan, masks, bits, *_ = _fixture()
    assert len(masks) == 10
    for name, mask in masks.items():
        assert len(mask.protected) == mask.budget
        if name.endswith("b12"): assert mask.budget == 12
        if name.endswith("b24"): assert mask.budget == 24
