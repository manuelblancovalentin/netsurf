from dataclasses import dataclass
import numpy as np
import pytest

from netsurf.adapters.mlperf_streaming import StreamingEvaluation
from netsurf.faults import InjectionReport
from netsurf.identities import BitIdentity
from netsurf.streaming_validation import compare_stream


class TinyStream:
    def fixture(self):return np.zeros((3,1),np.int8),np.array([0,1,0])
    def bit_identities(self):return tuple(BitIdentity('kernel',i,b) for i in range(2) for b in range(8))
    def values(self):return {'kernel':np.array([0,-1],np.int8)}
    def evaluate(self,*args):return StreamingEvaluation(1,0,0,1,'clean')
    def bit_perturbations(self):return {'kernel':np.tile(2.**np.arange(8),(2,1))}
    def dataset_identity_manifest(self):return {'fixture':'synthetic'}
    def representation_manifest(self):return {'int8':True}
    def evaluate_fault_schedule(self,x,y,schedule,mask):
        n=sum(e.bit not in mask.protected for e in schedule.events)
        value=StreamingEvaluation(1,int(n>0),0,1,'faulted' if n else 'clean')
        return value,InjectionReport(schedule.digest(),len(schedule.events),n,len(schedule.events)-n,n)


def test_stream_gate_replay_counts_and_shared_controls():
    a,_=compare_stream(TinyStream(),trials=2)
    b,_=compare_stream(TinyStream(),trials=2)
    assert a==b
    assert a['fault_evaluations']==16 and a['exact_window_inferences_in_comparison']==54
    assert len(a['schedules'])==2
    for trial in range(2):
        rows=[r for r in a['trials'] if r['trial']==trial]
        assert len({r['events']['schedule_hash'] for r in rows})==1
        assert next(r for r in rows if r['policy']=='full_protection')['evaluation']==a['clean']
    assert not any('accuracy' in r['evaluation'] for r in a['trials'])


def test_stream_gate_rejects_unbounded_or_duplicate_conditions():
    with pytest.raises(ValueError):compare_stream(TinyStream(),trials=32)
    with pytest.raises(ValueError):compare_stream(TinyStream(),probability=.1)
    with pytest.raises(ValueError):compare_stream(TinyStream(),coverages=(.125,.125))


def test_stream_gate_fails_on_bad_full_protection():
    class Broken(TinyStream):
        def evaluate_fault_schedule(self,x,y,schedule,mask):
            value,counts=super().evaluate_fault_schedule(x,y,schedule,mask)
            if mask.coverage==1:value=StreamingEvaluation(0,0,1,1,'bad')
            return value,counts
    with pytest.raises(ValueError,match='full protection'):compare_stream(Broken())
