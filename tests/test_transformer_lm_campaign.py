"""Small synthetic campaign-plan tests; no model asset or inference required."""
import copy
import json
from pathlib import Path

import pytest

from netsurf.identities import BitIdentity
from netsurf.manifests import digest_object
from netsurf.protection import ProtectionMask
from netsurf.torch_storage import expected_faults
from netsurf.transformer_lm_campaign import (_candidate_bits, _schedule, _verify_field,
    validate_plan, BERS, BUDGETS, METHODS, SOURCE_FILES)
from netsurf.transformer_plan import PlanBlocked


class SyntheticStorage:
    def __init__(self, width):
        self.codec=type("Codec",(),{"width":width})()
        self.bit_count=width*128

    def bit_at(self,index):
        return BitIdentity("linear.weight", index//self.codec.width,
                           index%self.codec.width)


def _synthetic_plan():
    bits=[bit.key() for bit in _candidate_bits(SyntheticStorage(16))]
    sha="a"*64
    plan={"schema_version":"transformer-campaign-plan/v1",
        "protocol_variant":"targeted-lm-96/v1","status":"FROZEN",
        "target":"qwen3_0_6b_bf16","source_commit":"b"*40,
        "asset_manifest_sha256":sha,"asset_record_sha256":sha,
        "eligible_bits":96,"eligible_population_sha256":digest_object(bits),
        "candidate_bits":bits,"partitions_sha256":digest_object({
            "selection":["validation",33,161,sha],"final":["test",0,257,sha]}),
        "selection_token_slice":{"start":33,"stop":161,"token_ids_sha256":sha},
        "final_token_slice":{"start":0,"stop":257,"token_ids_sha256":sha},
        "budgets":list(BUDGETS),"ber":list(BERS),"methods":list(METHODS),
        "selection":{"seed":120001,"fields":1,"sample_count":127},
        "final":{"seed":130001,"fields":32,"sample_count":256},
        "analysis":{"seed":140001,"independent_unit":"fault_field",
                    "multiple_comparison_procedure":"paired sign-flip Holm eight",
                    "primary_family":"exact isolated versus significance at BER 4/96, budgets 12 and 24; two-sided Holm across two",
                    "secondary_family":"all eight exact-isolated contrasts, two-sided Holm across eight"},
        "failure_penalty_nats":30.0,"costs":{"total_model_requests":610}}
    plan["plan_sha256"]=digest_object(plan)
    return plan


def test_candidate_sampling_is_deterministic_distinct_and_precision_specific():
    fp=_candidate_bits(SyntheticStorage(32))
    bf=_candidate_bits(SyntheticStorage(16))
    assert fp==_candidate_bits(SyntheticStorage(32))
    assert bf==_candidate_bits(SyntheticStorage(16))
    assert len(fp)==len(set(fp))==len(bf)==len(set(bf))==96
    assert {x.bit_index for x in fp}=={31,30,22}
    assert {x.bit_index for x in bf}=={15,14,6}
    assert len({x.flat_index for x in fp})==32
    assert len({x.flat_index for x in bf})==32


def test_hash_bound_plan_rejects_identity_budget_and_partition_changes():
    plan=_synthetic_plan();validate_plan(plan)
    for key,value in (("target","unapproved"),("budgets",[0,12,96]),
                      ("candidate_bits",plan["candidate_bits"][:-1])):
        changed=copy.deepcopy(plan);changed[key]=value
        with pytest.raises(PlanBlocked):validate_plan(changed)
    changed=copy.deepcopy(plan);changed["final_token_slice"]["token_ids_sha256"]="f"*64
    changed["plan_sha256"]=digest_object({k:v for k,v in changed.items() if k!="plan_sha256"})
    with pytest.raises(PlanBlocked,match="partition"):
        validate_plan(changed)


def test_bernoulli_fields_have_distinct_arms_and_prescribed_exposure():
    storage=SyntheticStorage(16)
    # The scheduler receives a restricted 96-bit universe in the real study.
    storage.bit_count=96
    first=_schedule(storage,1,0,256)
    repeat=_schedule(storage,1,0,256)
    second=_schedule(storage,2,0,256)
    assert first.digest()==repeat.digest() and first.events==repeat.events
    assert first.seed!=second.seed and first.model.probability==1/96
    assert second.model.probability==4/96
    assert all(event.inference_index is None for event in first.events+second.events)
    assert len(SOURCE_FILES)>=10


def test_verified_final_member_rejects_changed_counts_and_intent():
    storage=SyntheticStorage(16);storage.bit_count=96
    schedule=_schedule(storage,2,3,256)
    bits=tuple(storage.bit_at(i) for i in range(96))
    masks={"unprotected":ProtectionMask("unprotected",frozenset(),96,0,{}),
           "full":ProtectionMask("full",frozenset(bits),96,96,{})}
    clean={"mean_nll":1.0,"total_units":256}
    policies={}
    for name,mask in masks.items():
        applied=sum(event.bit not in mask.protected for event in schedule.events)
        policies[name]={"mask_sha256":mask.digest(),
            "expected_faults":expected_faults(96,mask.budget,BERS[2]),
            "report":{"schedule_hash":schedule.digest(),
                      "proposed_events":len(schedule.events),
                      "applied_events":applied,
                      "suppressed_events":len(schedule.events)-applied,
                      "unique_applied_bits":applied},
            "risk":clean}
    item={"field":3,"arm":2,"ber":BERS[2],"seed":schedule.seed,
          "schedule_sha256":schedule.digest(),
          "events":[event.key() for event in schedule.events],"policies":policies}
    intent={k:item[k] for k in ("field","arm","ber","seed","schedule_sha256","events")}
    intent["policy_names"]=sorted(masks)
    _verify_field(item,intent,schedule,masks,clean,arm=2,field=3,ber=BERS[2])
    broken=copy.deepcopy(item)
    broken["policies"]["unprotected"]["report"]["applied_events"]+=1
    with pytest.raises(PlanBlocked,match="accounting"):
        _verify_field(broken,intent,schedule,masks,clean,arm=2,field=3,ber=BERS[2])
    broken_intent=copy.deepcopy(intent);broken_intent["seed"]+=1
    with pytest.raises(PlanBlocked,match="intent"):
        _verify_field(item,broken_intent,schedule,masks,clean,arm=2,field=3,ber=BERS[2])
