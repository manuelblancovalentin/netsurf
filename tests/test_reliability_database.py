"""Bounded checks for source verification, pairing, and study separation."""

import csv
import json
import sqlite3

import pytest

from netsurf.manifests import digest_object, sha256_file
from netsurf.reliability_database import _mean_ci, _parse_policy, build_database


def _write_csv(path, rows):
    with path.open("w", newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def _fixture(tmp_path, mismatch=False):
    root=tmp_path/"run"; (root/"measurements").mkdir(parents=True); (root/"artifacts").mkdir()
    payload={"scientific_protocol":"test/v1"}; digest=digest_object(payload)
    (root/"scientific_payload.json").write_text(json.dumps(payload))
    (root/"FINALIZED").write_text(f"scientific_payload_sha256={digest}\n")
    manifest={"status":"complete","schema_version":"3.4.0","scientific_payload_sha256":digest,
              "benchmark":{"id":"tiny_int8","checkpoint_sha256":"abc","dataset_sha256":"data","inference_backend":"reference",
                           "sample_partitions":{"replication_0":{"ranking":{"sample_sha256":"cal"},"evaluation":{"sample_sha256":"eval"}}}},
              "execution":{"randomization":{"fault_seed":17}}}
    (root/"manifest.json").write_text(json.dumps(manifest))
    summary={"clean_evaluation":{"accuracy":1.0,"categorical_crossentropy":0.1},"eligible_bits":8}
    (root/"summary.json").write_text(json.dumps(summary))
    trial=[]; repagg=[]; agg=[]
    for rep,base in (("replication_0",1.0),("replication_1",4.0)):
        for trial_idx in range(2):
            schedule=f"schedule-{rep}-{trial_idx}"
            for method,coverage,loss,applied,suppressed in (("no_protection",0.0,base,1,0),("msb_global",0.125,base/2,0,1)):
                trial.append({"replication":rep,"trial":trial_idx,"policy":f"ber=0.1/{method}" if coverage==0 else f"ber=0.1/{method}@0.125000",
                              "schedule_sha256":schedule if not mismatch or method=="no_protection" else "wrong", "categorical_crossentropy":loss,
                              "accuracy":1.0-loss/10,"proposed_events":1,"applied_events":applied,"suppressed_events":suppressed})
                repagg.append({"replication":rep,"logical_ber":0.1,"policy":method,"coverage":coverage,"protected_bits":int(8*coverage),"mask_sha256":f"mask-{rep}-{method}"})
    for method,coverage,mean in (("no_protection",0.0,2.5),("msb_global",0.125,1.25)):
        agg.append({"logical_ber":0.1,"policy":method,"coverage":coverage,"protected_bits":int(8*coverage),"mask_sha256":"combined",
                    "trial_count":4,"ranking_mask_count":2,"loss_mean":mean})
    _write_csv(root/"measurements/trials.csv",trial)
    _write_csv(root/"measurements/protection_results.csv",agg)
    _write_csv(root/"measurements/replication_protection_results.csv",repagg)
    _write_csv(root/"measurements/estimator_runtime.csv",[{"replication":"replication_0","estimator":"msb_first","runtime_seconds":0.1}])
    (root/"artifacts/protection_masks.json").write_text("{}")
    files={str(p.relative_to(root)):sha256_file(p) for p in root.rglob("*") if p.is_file() and p.name!="FINALIZED"}
    (root/"checksums.json").write_text(json.dumps({"files":files,"schema_version":"1.0.0"}))
    src={"schema_version":"1.0.0","studies":[{"id":"tiny","path":str(root),"kind":"whole_kernel","scientific_payload_sha256":digest,"checksums_sha256":sha256_file(root/"checksums.json")}]}
    source_path=tmp_path/"sources.json";source_path.write_text(json.dumps(src))
    return source_path,root


def test_all_replications_are_pooled_and_masks_remain_distinct(tmp_path):
    source,root=_fixture(tmp_path)
    db_path=tmp_path/"out.sqlite";build_database(source,db_path)
    with sqlite3.connect(db_path) as db:
        row=db.execute("SELECT field_count,mean_loss,loss_degradation,paired_unprotected_delta_mean FROM aggregated_conditions WHERE method_id='msb_global'").fetchone()
        assert row==(4,1.25,1.15,-1.25)
        assert db.execute("SELECT count(DISTINCT mask_sha256) FROM protection_masks WHERE method_id='msb_global'").fetchone()[0]==2
        assert db.execute("SELECT count(*) FROM coverage_matrix WHERE status='NOT_MEASURED'").fetchone()[0]>0
        assert db.execute("SELECT count(*) FROM coverage_matrix").fetchone()[0]==72
        assert db.execute("SELECT count(*),min(eligible_bits) FROM reliability_grid").fetchone()==(2,8)
    with pytest.raises(FileExistsError): build_database(source,db_path)
    assert db_path.exists()


def test_mismatched_schedule_or_corrupt_source_fails_closed(tmp_path):
    source,_=_fixture(tmp_path,mismatch=True)
    dest=tmp_path/"bad.sqlite"
    with pytest.raises(AssertionError,match="unpaired schedule"):
        build_database(source,dest)
    assert not dest.exists()
    source,root=_fixture(tmp_path/"other")
    (root/"summary.json").write_text("corrupt")
    with pytest.raises(AssertionError,match="artifact changed"):
        build_database(source,tmp_path/"bad2.sqlite")


def test_ci_and_policy_mapping():
    assert _mean_ci([2.0,2.0,2.0])==(2.0,0.0,0.0,2.0,2.0)
    assert _parse_policy("ber=0.01/first_order_gradient_delta@0.125000")==(.01,"gradient_global",.125)
    assert _parse_policy("msb_global@0.125000",.01)==(.01,"msb_global",.125)
    with pytest.raises(ValueError,match="omitted BER"):
        _parse_policy("msb_global@0.125000")
