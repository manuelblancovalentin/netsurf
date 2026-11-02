from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import pytest

from netsurf.benchmarks.capabilities import capability_records, capability_record, require_capability, UnsupportedCapability, storage_contract
from netsurf.adapters.mlperf_streaming import event_counts
from netsurf.fkeras_baseline import mask_equivalence


def test_complete_portfolio_and_fail_closed_capabilities():
    records = capability_records()
    assert len(records) == 18 and len({r['family'] for r in records}) == 11
    assert require_capability('kws_dscnn_int8', 'conditional_restricted')
    assert require_capability('resnet8_int8', 'conditional_restricted')  # M98 validated extension.
    assert require_capability('toycar_int8', 'conditional_restricted')  # M99 complete-file extension.
    for name in ('streaming_dscnn1d_int8', 'qwen3_0_6b_int4'):
        with pytest.raises(UnsupportedCapability,match='BLOCKED'):
            require_capability(name,'conditional_restricted')
    assert 'not ROC-AUC' in capability_record('toycar_int8')['surrogate_loss']
    assert 'event' in capability_record('streaming_dscnn1d_int8')['primary_task_metrics']
    for row in records:
        assert row['validation_command'].startswith('/projects/kona-assets/')
        assert row['checkpoint']['revision']
        assert row['dataset'] and row['storage'] and row['resource_plan']
    assert 'BLOCKED' in storage_contract('INT4')['packing']
    assert storage_contract('BF16')['exponent'] == list(range(7,15))
    assert storage_contract('FP32')['mantissa'] == list(range(23))
    with pytest.raises(UnsupportedCapability, match='precision'):
        storage_contract('FP16')


def test_capability_cli_outside_repository_no_framework_import():
    from netsurf.benchmarks.capabilities import ROOT
    import os
    env = {**os.environ, 'PYTHONPATH':str(ROOT)}
    completed = subprocess.run([sys.executable,'-m','netsurf','benchmarks','capabilities','qwen3_0_6b_int4'],
        cwd='/tmp',env=env,check=True,capture_output=True,text=True)
    assert json.loads(completed.stdout)['capabilities']['reference_inference'] == 'BLOCKED'


def test_event_accounting_not_frame_accuracy_and_no_clipping():
    present=np.zeros(120,dtype=np.int8)
    present[20:30]=1
    present[70:80]=1
    detections=np.zeros(120,dtype=bool)
    detections[25:27]=1  # one true event
    detections[45:47]=1 # one false event despite two frames
    assert event_counts(detections,present,stride_samples=1,sample_rate=10) == (1,1,1)
    detections[82]=1    # within half-second tolerance
    assert event_counts(detections,present,stride_samples=1,sample_rate=10) == (2,1,0)
    with pytest.raises(ValueError,match='duration'):
        event_counts(np.ones(121),present,stride_samples=1)


def test_saved_mask_degeneracy_uses_actual_membership():
    rows=mask_equivalence({'a':[('x',0,7)],'b':[('x',1,7)],'c':[('x',0,7)]})
    assert next(r for r in rows if r['right']=='c' and r['left']=='a')['same_mask']
    assert not next(r for r in rows if r['right']=='b')['same_mask']
