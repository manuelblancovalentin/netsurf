"""Bounded, vector-metric streaming protection gate; not a publication engine.

Reuses the pinned adapter, canonical rankings, Bernoulli schedules and exact
fault evaluator. The complete short stream is one trial. FP and FN are never
collapsed into an invented scalar risk or independent frame observations.
"""
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
import argparse
import json
import resource

import numpy as np

from .asset_lock import sha256_path
from .faults import FaultModel, FaultSchedule
from .manifests import digest_object, git_provenance
from .protection import ProtectionMask, ProtectionPolicy
from .ranking import Ranking, MSBEstimator, RandomEstimator
from .result_store import ResultStore


def compare_stream(adapter, *, trials=2, probability=.001, fault_seed=110201,
                   ranking_seed=110301, coverages=(.0625, .125)):
    if not 1 <= trials <= 4 or not 0 <= probability <= .01:
        raise ValueError('engineering gate is limited to 1–4 fields and BER <= .01')
    if not coverages or len(set(coverages)) != len(coverages) or any(not 0 < c < 1 for c in coverages):
        raise ValueError('unique interior coverages required; controls are shared')
    if len(coverages) > 4:
        raise ValueError('engineering gate allows at most four interior coverages')
    inputs, labels = adapter.fixture()
    bits = tuple(adapter.bit_identities())
    if len(bits) != len(set(bits)):
        raise ValueError('duplicate eligible bits')
    index = {b: i for i, b in enumerate(bits)}
    before = adapter.values()
    clean = adapter.evaluate(inputs, labels)
    delta = adapter.bit_perturbations()
    started = perf_counter()
    rankings = [MSBEstimator().estimate(bits), RandomEstimator(ranking_seed).estimate(bits),
        Ranking('numerical_jump', {b: float(abs(delta[b.parameter].reshape(-1, 8)[b.flat_index,b.bit_index])) for b in bits},
                {'definition':'abs(exact affine encoded XOR jump)'})]
    masks = {'no_protection': ProtectionMask('none',frozenset(),len(bits),0,{}),
             'full_protection': ProtectionMask('full',frozenset(bits),len(bits),len(bits),{})}
    for ranking in rankings:
        for coverage in coverages:
            masks[f'{ranking.estimator}@{coverage:.6f}'] = ProtectionPolicy.global_ranking(ranking,coverage)
    ranking_seconds = perf_counter() - started
    law = FaultModel('weight','logical_trainable_parameter_bits','persistent_trial','independent_bernoulli',probability)
    rows, schedules, timings = [], [], []
    for trial in range(trials):
        schedule = FaultSchedule.generate(law,bits,seed=fault_seed+trial,inference_count=len(inputs))
        schedules.append({'trial':trial,'seed':schedule.seed,'sha256':schedule.digest(),
                          'population_indices':[index[e.bit] for e in schedule.events]})
        for name,mask in masks.items():
            begin = perf_counter()
            value,events = adapter.evaluate_fault_schedule(inputs,labels,schedule,mask)
            timings.append(perf_counter()-begin)
            if events.proposed_events != events.applied_events+events.suppressed_events:
                raise ValueError('event accounting failed')
            if name == 'no_protection' and events.suppressed_events != 0:
                raise ValueError('zero protection suppressed an event')
            if name == 'full_protection' and (value != clean or events.applied_events != 0):
                raise ValueError('full protection did not recover exact clean inference')
            if any(not np.array_equal(v,adapter.values()[k]) for k,v in before.items()):
                raise ValueError('fault evaluator failed restoration')
            rows.append({'trial':trial,'policy':name,'protected_bits':len(mask.protected),
                'coverage':mask.coverage,'mask_sha256':mask.digest(),
                'evaluation':asdict(value),'events':asdict(events),
                'false_positive_difference_from_clean':value.false_positives-clean.false_positives,
                'false_negative_difference_from_clean':value.false_negatives-clean.false_negatives})
    if adapter.evaluate(inputs,labels) != clean:
        raise ValueError('final output digest differs from clean')
    contrasts=[]
    for name in masks:
        if name=='no_protection': continue
        for trial in range(trials):
            a=next(r for r in rows if r['trial']==trial and r['policy']==name)
            b=next(r for r in rows if r['trial']==trial and r['policy']=='no_protection')
            contrasts.append({'trial':trial,'policy':name,'schedule_sha256':a['events']['schedule_hash'],
                **{metric:a['evaluation'][metric]-b['evaluation'][metric] for metric in ['false_positives','false_negatives']}})
    payload={'schema_version':'streaming-protection-engineering/1.0.0',
        'scope':'one previously exposed official short fixture; not population efficacy',
        'partition':adapter.dataset_identity_manifest(),'representation':adapter.representation_manifest(),
        'fault':asdict(law),'ranking_seed':ranking_seed,'eligible_bits':len(bits),
        'population':[b.key() for b in bits],
        'masks':{k:sorted(index[b] for b in v.protected) for k,v in masks.items()},
        'rankings':{r.estimator:{'sha256':r.digest(),'metadata':dict(r.metadata)} for r in rankings},
        'clean':asdict(clean),'trials':rows,'schedules':schedules,'paired_differences':contrasts,
        'restoration_verified':True,'field_count':trials,'window_count_per_evaluation':len(inputs),
        'fault_evaluations':len(rows),'exact_window_inferences_in_comparison':(len(rows)+2)*len(inputs)}
    execution={'ranking_and_mask_seconds':ranking_seconds,'fault_evaluation_seconds':timings,
        'peak_process_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
    return payload,execution


def run(output, **options):
    from .benchmarks.capabilities import load_capability_adapter
    output=Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    started=perf_counter()
    adapter=load_capability_adapter('streaming_dscnn1d_int8')
    loaded=perf_counter()
    payload,execution=compare_stream(adapter,**options)
    execution.update(load_seconds=loaded-started,total_seconds=perf_counter()-started)
    store=ResultStore(output.parent,output.name)
    store.write_json('scientific_payload.json',payload)
    store.write_json('manifest.json',{'schema_version':'streaming-protection-engineering/1.0.0',
        'scientific_payload_sha256':digest_object(payload),'source':git_provenance(Path(__file__).resolve().parents[1]),
        'execution':execution})
    lines=['# Streaming short-fixture protection validation','',
        'Engineering evidence only. FP/FN are event counts for the same complete stream; no accuracy or combined risk is defined.',
        '', '| Trial | Policy | Protected bits | FP | FN | Applied / proposed |',
        '|---|---|---:|---:|---:|---:|']
    for r in payload['trials']:
        lines.append(f"| {r['trial']} | {r['policy']} | {r['protected_bits']} | {r['evaluation']['false_positives']} | {r['evaluation']['false_negatives']} | {r['events']['applied_events']} / {r['events']['proposed_events']} |")
    store.write_text('report.md','\n'.join(lines)+'\n')
    store.write_json('checksums.json',{'schema_version':'1.0.0','files':{p.name:sha256_path(p) for p in output.iterdir()}})
    digest=store.finalize(payload)
    return {'output':str(output),'scientific_payload_sha256':digest,'execution':execution}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--trials',type=int,default=2)
    parser.add_argument('--probability',type=float,default=.001)
    args=parser.parse_args()
    print(json.dumps(run(args.output,trials=args.trials,probability=args.probability),indent=2))


if __name__=='__main__': main()
