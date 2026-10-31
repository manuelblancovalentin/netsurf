"""Bounded MLPerf tests; never touch unexposed confirmation examples."""
from pathlib import Path
import ast
import hashlib
import json
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from netsurf.benchmarks.capabilities import ROOT, load_capability_adapter
from netsurf.fkeras_estimator import ContinuousKwsHessian
from netsurf.fkeras_baseline import leading_hessian_eigenpairs, repository_weight_hessian_score


def pinned():
    tf=pytest.importorskip('tensorflow')
    if tf.__version__ != '2.14.1':
        pytest.skip('requires pinned TF 2.14.1, not canonical modern TF')
    if not Path('/projects/kona-assets/manuelbv/locks/mlperf-tiny/kws.lock.json').is_file():
        pytest.skip('pinned local assets unavailable')
    return tf


def test_kws_continuous_hessian_finite_difference_mapping_and_state():
    pinned()
    adapter=load_capability_adapter('kws_dscnn_int8')
    x,y,_=adapter.sample_partition('ranking',2)
    operator=ContinuousKwsHessian(adapter,x,y)
    original=operator.theta.copy()
    vector=np.random.default_rng(970777).normal(size=len(original))
    vector/=np.linalg.norm(vector)
    hv=operator(vector)
    np.testing.assert_array_equal(hv,operator(vector))
    # Record the step sweep, including failed coarse-step approximations. This
    # frozen quantized model is close to ReLU boundaries; 1e-7 crosses them.
    errors={}
    for epsilon in (1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10):
        fd=(operator.gradient_at(original+epsilon*vector)-operator.gradient_at(original-epsilon*vector))/(2*epsilon)
        errors[str(epsilon)]=float(np.linalg.norm(fd-hv)/np.linalg.norm(hv))
    assert errors['1e-09'] < 1e-5, errors
    assert errors['1e-10'] < 1e-5, errors
    np.testing.assert_array_equal(operator.theta, original)
    np.testing.assert_array_equal(np.concatenate([v.numpy().ravel() for v in operator.variables]),original)
    vectors,diagnostics=leading_hessian_eigenpairs(operator,len(original),count=2,seed=970201,max_products=300,tolerance=1e-5)
    assert max(diagnostics.normalized_residuals)<1e-5
    assert sum(row['flat_size'] for row in operator.mapping)==22016
    assert len(operator.mapping)==10
    assert adapter.companion_correspondence(x)['companion_is_authoritative'] is False
    print('M97_KWS_HVP',json.dumps({'finite_difference_relative_errors':errors,'eigenvalues':diagnostics.eigenvalues,
        'normalized_residuals':diagnostics.normalized_residuals,'hvp_calls':diagnostics.hvp_calls}))


def _source_ast(repository,relative,name):
    if not (repository/'.git').exists():
        pytest.skip('optional pinned upstream source checkout unavailable')
    revision='4addd0fa08d216e20637637874e084895f289da4'
    source=subprocess.check_output(['git','-C',str(repository),'show',revision+':'+relative],text=True)
    return next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name)


def test_streaming_features_and_events_match_exact_upstream_functions():
    tf=pinned()
    from scipy.io import wavfile
    from netsurf.adapters.mlperf_streaming import streaming_features, event_counts
    import functools
    repository=ROOT/'workspace/references/mlcommons-tiny'
    feature=_source_ast(repository,'benchmark/training/streaming_wakeword/get_dataset.py','get_lfbe_func')
    namespace={'np':np,'tf':tf,'functools':functools}
    exec(compile(ast.Module(body=[feature],type_ignores=[]),'pinned-upstream-feature','exec'),namespace)
    adapter=load_capability_adapter('streaming_dscnn1d_int8')
    rate,wave=wavfile.read(adapter.wave_path)
    waveform=(wave[:,0] if wave.ndim>1 else wave)/2**15
    expected=namespace['get_lfbe_func']({'window_size_samples':1024,'window_stride_samples':512,'sample_rate':16000,'dct_coefficient_count':40})(waveform).numpy()
    np.testing.assert_array_equal(streaming_features(waveform),expected)
    for name in ('debounce_detections','get_true_and_false_detections'):
        node=_source_ast(repository,'benchmark/training/streaming_wakeword/str_ww_util.py',name)
        exec(compile(ast.Module(body=[node],type_ignores=[]),'pinned-upstream-events','exec'),namespace)
    x,labels=adapter.fixture()
    outputs=adapter._outputs(x)
    scale,zero=adapter._interpreter.get_output_details()[0]['quantization']
    detected=((outputs.astype(np.float32)-zero)*scale)[:,0]>.95
    upstream=namespace['get_true_and_false_detections'](detected,labels,SimpleNamespace(sample_rate=16000,window_stride_ms=32))
    expected_counts=tuple(int(np.count_nonzero(a)) for a in upstream)
    assert event_counts(detected,labels)==expected_counts
    # This checkpoint misses one of the three fixture events under the pinned
    # backend. Preserve the observed failure; do not change source thresholds.
    assert expected_counts==(2,0,1)
    from netsurf.faults import FaultEvent, FaultModel, FaultSchedule
    from netsurf.identities import BitIdentity
    from netsurf.protection import ProtectionMask
    bits = adapter.bit_identities()
    assert len(set(bits)) == 368320
    before = adapter.values()
    first = bits[0]
    events = (FaultEvent(None, first), FaultEvent(None, BitIdentity(first.parameter, first.flat_index, 7)))
    schedule = FaultSchedule(FaultModel('weight', 'logical_trainable_parameter_bits',
        'persistent_trial', 'independent_bernoulli', 0.0), 970901, events, len(bits), len(x))
    none = ProtectionMask('none', frozenset(), len(bits), 0, {})
    full = ProtectionMask('full', frozenset(bits), len(bits), len(bits), {})
    content, _ = adapter.storage.apply_schedule(schedule, none)
    after = adapter.storage.values(content)
    original_code = int(before[first.parameter].reshape(-1)[first.flat_index]) & 255
    assert (int(after[first.parameter].reshape(-1)[first.flat_index]) & 255) == original_code ^ 129
    clean = adapter.evaluate(x, labels)
    _, applied = adapter.evaluate_fault_schedule(x, labels, schedule, none)
    protected, suppressed = adapter.evaluate_fault_schedule(x, labels, schedule, full)
    assert protected == adapter.evaluate(x, labels) == clean
    assert (applied.proposed_events, applied.applied_events, applied.suppressed_events,
            suppressed.proposed_events, suppressed.applied_events, suppressed.suppressed_events) == (2, 2, 0, 2, 0, 2)
    assert all(np.array_equal(before[k], adapter.values()[k]) for k in before)
    print('M97_STREAMING',json.dumps({'events':expected_counts,'windows':len(x),
        'input_sha256':hashlib.sha256(x.tobytes()).hexdigest(),'output_sha256':hashlib.sha256(outputs.tobytes()).hexdigest()}))


def test_repository_score_matches_pinned_method_without_importing_qkeras():
    repository=ROOT/'workspace/references/fkeras'
    if not (repository/'.git').exists():
        pytest.skip('optional upstream source checkout unavailable')
    from netsurf.fkeras_baseline import FKERAS_SOURCE_COMMIT
    source=subprocess.check_output(['git','-C',str(repository),'show',FKERAS_SOURCE_COMMIT+':fkeras/metrics/hessian.py'],text=True)
    cls=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='HessianMetrics')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='do_sum_hessian_rank')
    namespace={'np':np}
    exec(compile(ast.Module(body=[method],type_ignores=[]),'pinned-upstream-sum','exec'),namespace)
    class Scalar:
        def __init__(self,x): self.x=x
        def numpy(self): return self.x
    rng=np.random.default_rng(9701)
    q,_=np.linalg.qr(rng.normal(size=(7,3)))
    vectors=q.T; values=np.array([-4.,2.,.3]); theta=rng.normal(size=7)
    expected=namespace['do_sum_hessian_rank'](None,theta,[[v[:2],v[2:]] for v in vectors],[Scalar(x) for x in values],3)
    np.testing.assert_allclose(repository_weight_hessian_score(theta,vectors,values),expected,rtol=1e-14)
