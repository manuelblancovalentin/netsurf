"""Fast fake-backend lifecycle tests, with no TensorFlow or external assets."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from netsurf import scaling_study as s
from netsurf.identities import BitIdentity
from netsurf.manifests import sha256_file, digest_object
from netsurf.samples import SamplePartition
from netsurf.faults import InjectionReport


class FakeAdapter:
    def __init__(self):
        self._model_content=b"clean";self.roles=[]
    def load(self):pass
    def values(self):return {"kernel":np.zeros(10,dtype=np.int8)}
    def _set_model_content(self,content):self._model_content=content
    def _outputs(self,x):return np.zeros(len(x))
    def sample_partition(self,role,count,offset=0):
        self.roles.append(role)
        ids=(1,2) if role=="conditional_model_selection" else (3,4)
        return np.array([[i] for i in ids]),np.array([0,1]),SamplePartition(role,"fake",ids,digest_object(ids),"engineering")
    def evaluate(self,x,y):
        self._outputs(x)
        return SimpleNamespace(loss=.5,accuracy=.5)
    def evaluate_fault_schedule(self,x,y,schedule,mask):
        hit=[e.bit for e in schedule.events if e.bit not in mask.protected]
        self._set_model_content(b"faulted")
        try:
            self._outputs(x)
            value=.5+sum((b.flat_index-3)*.001 for b in hit)
            return SimpleNamespace(loss=value,accuracy=.5),InjectionReport(schedule.digest(),len(schedule.events),len(hit),len(schedule.events)-len(hit),len(hit))
        finally:self._set_model_content(b"clean")


def setup(monkeypatch,tmp_path):
    universe=tuple(BitIdentity("kernel",i,0) for i in range(10))
    pool=s.sample_candidates(universe,set(universe[:3]),6,17);pool.update(k=6,id="k6_seed17")
    candidates=tuple(s._identity(r) for r in pool["candidates"])
    protected=frozenset(universe[:3]);outside=protected-set(candidates)
    context=SimpleNamespace(original=protected,original_inside_indices=tuple(pool["original_mask"]),record={"sha":"context"})
    config=SimpleNamespace(risk_metric="categorical_crossentropy")
    cap=SimpleNamespace(target="fake",source_split="fake",reference_inputs=0)
    design={"selection_seeds":[31,32],"final_seeds":[41,42],"exhaustive_max_k":6}
    request={"design":design,"purpose":"engineering","samples":{"selection_count":2,"selection_offset":0,"final_count":2,"final_offset":0},
        "statistics":{"confidence":.95,"bootstrap_seed":1,"bootstrap_replicates":10}}
    plan={"schema_version":s.VERSION,"protocol":s.PROTOCOL,"checkpoint_sha256":"fake",
        "request":request,"pools":[pool],"arms":[{"logical_ber":.2,"expected_applied_flips":1.4}],"partitions":{"selection":[1,2],"final":[3,4]},
        "global_budget":3,"operation_counts":[{"pool":pool["id"],"arm":0,"field_seeds":{
            phase:s.field_seeds(design,pool,0,phase) for phase in ("selection","final")}}]}
    s._seal_directory(tmp_path/"plan",plan,frozen=True)
    path=tmp_path/"plan/plan.json"
    monkeypatch.setattr(s,"_run_context",lambda *args:(plan,config,cap,universe,context,pool,candidates))
    import netsurf.conditional_benchmarks as cb
    adapters=[]
    def create(c):
        a=FakeAdapter();adapters.append(a);return a
    monkeypatch.setattr(cb,"create_conditional_adapter",create)
    monkeypatch.setattr(cb,"audit_exposure",lambda *a,**k:{"exposed_test_indices":[1,2,3,4]})
    # Existing task-metric contract consumes real Evaluation objects; fake scalar
    # boundary is intentionally confined to this fixture.
    import netsurf.conditional_damage as cd
    monkeypatch.setattr(cd,"_risk_value",lambda evaluation,metric:evaluation.loss)
    return path,pool,adapters


def test_selection_seal_final_repeat_and_corruption(monkeypatch,tmp_path):
    path,pool,adapters=setup(monkeypatch,tmp_path)
    a=s.execute(path,pool["id"],0,"selection",tmp_path/"selections/one")
    payload=s.verify_directory(tmp_path/"selections/one")
    assert adapters[-1].roles==["conditional_model_selection"]
    assert "final_fields" not in payload and len(payload["outside_fields"])==2
    assert all(len(m)==pool["budget"] for m in payload["frozen_masks"].values())
    b=s.execute(path,pool["id"],0,"selection",tmp_path/"replay/one")
    assert a["scientific_payload_sha256"]==b["scientific_payload_sha256"]
    with pytest.raises(ValueError):s.execute(path,pool["id"],0,"final",tmp_path/"bad")
    s.seal(path,tmp_path/"selections",tmp_path/"registry")
    final=s.execute(path,pool["id"],0,"final",tmp_path/"finals/one",selection_registry=tmp_path/"registry/plan.json")
    f=s.verify_directory(tmp_path/"finals/one")
    assert adapters[-1].roles==["conditional_final_evaluation"]
    assert f["frozen_masks"]==payload["frozen_masks"] and len(f["field_risks"])==2
    assert all(r["proposed"]==r["applied"]+r["suppressed"] and r["restored"] for r in f["records"])
    from netsurf.scaling_reporting import analyze,plot_bundle
    analyze(path,tmp_path/"registry/plan.json",tmp_path/"finals",tmp_path/"analysis",plots=False)
    plot_bundle(tmp_path/"analysis",tmp_path/"figures")
    assert (tmp_path/"figures/cost_error.png").stat().st_size>100
    with pytest.raises(ValueError):s.seal(path,tmp_path,tmp_path/"duplicated")


def test_field_streams_separate_replication_levels():
    d={"selection_seeds":[1,2],"final_seeds":[3,4]};seen=set()
    for k in (6,12,32,64):
        for poolseed in (101,102):
            for arm in range(3):
                pool={"seed":poolseed,"k":k}
                for phase in ("selection","final"):
                    seeds=s.field_seeds(d,pool,arm,phase)
                    assert seeds==s.field_seeds(d,pool,arm,phase)
                    assert not seen.intersection(seeds);seen.update(seeds)


def test_schema_and_templates():
    root=Path(__file__).resolve().parents[1]
    schema=json.loads((root/"netsurf/schemas/scaling/v1.schema.json").read_text())
    for path in (root/"config/examples").glob("m11_*.toml"):
        raw=s.load_request(path)
        assert set(raw)==set(schema["required"])
