import json
from pathlib import Path
import pytest

from workflows.paper1_campaign import verified_stage,write_new,archive,ROOT
from workflows.paper1_evidence_audit import sha
from netsurf.api import load_configuration


def test_stage_reuse_and_corruption_fail_closed(tmp_path):
    stage=tmp_path/'stage';calls=[]
    def produce(p):calls.append(1);write_new(p/'result.json',{'ok':True})
    verified_stage(stage,produce);verified_stage(stage,produce)
    assert calls==[1]
    (stage/'result.json').write_text('corrupt')
    with pytest.raises(ValueError,match='corrupt'):verified_stage(stage,produce)


def test_incomplete_stage_never_reexecuted(tmp_path):
    stage=tmp_path/'stage';stage.mkdir()
    with pytest.raises(RuntimeError,match='incomplete'):verified_stage(stage,lambda p:None)


def test_archive_never_overwrites(tmp_path):
    f=tmp_path/'input';f.write_text('data');z=tmp_path/'test.zip'
    archive(z,[(f,'data')])
    with pytest.raises(FileExistsError):archive(z,[(f,'data')])


def test_frozen_plans_validate_and_counts_include_shared_controls():
    plans=json.loads((ROOT/'config/paper1_campaigns.json').read_text())
    for plan in plans.values():
        p=ROOT/plan['configuration'];assert sha(p)==plan['configuration_sha256']
        c=load_configuration(p)
        assert c.samples.require_disjoint
        assert c.replications.count==4 and c.execution.trials==8
        assert c.protection.coverages==(.0625,.125,.1875,.25)
        assert plan['fault_fields']==4*4*8
        assert plan['mask_evaluations']==plan['fault_fields']*(len(c.policies)*4+2)


def test_target_ledger_preserves_all_precision_targets_and_blockers():
    ledger=json.loads((ROOT/'config/paper1_target_status.json').read_text())
    registry=json.loads((ROOT/'config/benchmark_expansion_registry.json').read_text())
    assert ledger['portfolio_sha256']==sha(ROOT/ledger['portfolio_path'])
    assert {e['id'] for e in ledger['entries']}=={e['id'] for e in registry['entries']}
    assert len(ledger['entries'])==18
    assert len({e['family'] for e in ledger['entries']})==11
    assert ledger['integration_gate']=='INCOMPLETE'
    for e in ledger['entries']:
        if e['status'].startswith('BLOCKED'):
            assert e['blocker'] and e['protection_experiment'] is None
        if e['id']=='streaming_dscnn1d_int8':
            assert e['status']=='VERIFIED_BOUNDED_ONLY'
            assert e['eligible_bits']==368320
