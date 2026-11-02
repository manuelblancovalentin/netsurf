"""Foreground, fail-closed lifecycle for the three frozen Paper 1 static studies.

No selection/final optimizer is involved. Calibration-only masks are constructed
by the existing runner before independent fault evaluation. Previously exposed
input cohorts are explicitly exploratory. Never resume an unverified stage.
"""
from pathlib import Path
import argparse
import csv
import fcntl
import json
import os
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from netsurf.manifests import digest_object
from workflows.paper1_evidence_audit import read, sha, audit_static

PINNED='/projects/kona-assets/manuelbv/.venv-mlperf/bin/python'


def write_new(path,value):
    with Path(path).open('x') as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False); f.write('\n'); f.flush();os.fsync(f.fileno())


def code_digest():
    return digest_object({str(p.relative_to(ROOT)):sha(p) for directory in ['netsurf','workflows']
                          for p in sorted((ROOT/directory).rglob('*.py'))})


def verify_run(path):
    path=Path(path).resolve()
    from netsurf.result_bundle import read_result_bundle
    bundle=read_result_bundle(path,verify=True)
    digest=digest_object(read(path/'scientific_payload.json'))
    if digest != bundle.scientific_payload_sha256 or (path/'FINALIZED').read_text().strip()!=f'scientific_payload_sha256={digest}':
        raise ValueError('final scientific payload identity mismatch')
    return bundle


def verified_stage(path, producer):
    """Existing directory without a complete verified receipt is never overwritten."""
    path=Path(path)
    receipt=path/'STAGE_COMPLETE.json'
    if path.exists():
        if not receipt.is_file(): raise RuntimeError(f'incomplete stage; preserve and investigate: {path}')
        expected=read(receipt)
        actual={p.relative_to(path).as_posix():sha(p) for p in sorted(path.rglob('*')) if p.is_file() and p!=receipt}
        if actual != expected['files']: raise ValueError(f'corrupt stage: {path}')
        return
    path.mkdir()
    producer(path)
    files={p.relative_to(path).as_posix():sha(p) for p in sorted(path.rglob('*')) if p.is_file()}
    write_new(receipt,{'schema_version':'1.0.0','files':files})


def archive(zip_path, files):
    with zipfile.ZipFile(zip_path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for source,name in files:
            if source.is_symlink(): raise ValueError('archive refuses symlinks')
            z.write(source,name)


def execute(name, *, preflight_only=False):
    plan=read(ROOT/'config/paper1_campaigns.json')[name]
    config_path=ROOT/plan['configuration']
    if sha(config_path)!=plan['configuration_sha256']: raise ValueError('frozen TOML changed; prepare a new version')
    if Path(sys.executable).absolute()!=Path(PINNED): raise ValueError(f'required interpreter: {PINNED}')
    from netsurf.api import load_configuration,run_experiment
    from netsurf.parallel import ExecutionResources
    from netsurf.adapters.registry import create_adapter
    from netsurf.samples import require_disjoint
    config=load_configuration(config_path)
    checkpoint=Path(config.benchmark.checkpoint)
    lock=Path(config.benchmark.asset_lock)
    if sha(checkpoint)!=plan['checkpoint_sha256'] or sha(lock)!=plan['asset_lock_sha256']:
        raise ValueError('checkpoint/lock differs from frozen plan')
    identity={'schema_version':'paper1-static-lifecycle/1.0.0','plan':plan,'source_sha256':code_digest()}
    destination=ROOT/'workspace/paper1-campaigns'/config.name
    destination.mkdir(parents=True,exist_ok=True)
    with (destination/'execution.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        frozen=destination/'plan.json'
        if frozen.exists():
            if read(frozen)!=identity: raise ValueError('campaign code/plan drift; do not resume under changed code')
        else: write_new(frozen,identity)
        def preflight(where):
            baseline=verify_run(plan['exposure_source'])
            if baseline.scientific_payload_sha256!=plan['exposure_payload_sha256']: raise ValueError('exposure source identity changed')
            old=read(baseline.path/'scientific_payload.json')
            old=(old['replications'][0] if 'replications' in old else old)['benchmark']['partitions']
            adapter=create_adapter(config,None);adapter.load()
            parts=[]
            for role,count in [('ranking',config.samples.ranking_count),('empirical_validation',config.samples.empirical_count),('evaluation',config.samples.evaluation_count)]:
                _,_,part=adapter.sample_partition(role,count)
                prior=old[role]
                if part.source_split!=prior['source_split'] or not set(part.indices)<=set(prior['indices']):
                    raise ValueError(f'{role} not contained in the audited already-exposed source; explicit exposure review required')
                parts.append(part)
            require_disjoint(parts)
            if len(adapter.bit_identities())!=plan['eligible_bits']: raise ValueError('eligible population drift')
            write_new(where/'preflight.json',{'partitions':[p.to_dict() for p in parts],
                'exposure':'previously exposed, exploratory; not untouched confirmation',
                'representation':adapter.representation_manifest(),'planned_fault_fields':plan['fault_fields'],
                'planned_mask_evaluations':plan['mask_evaluations'],'plan_sha256':digest_object(identity)})
        verified_stage(destination/'preflight',preflight)
        print(json.dumps({'campaign':name,'status':'preflight_verified','work':plan['mask_evaluations'],
                          'scope':'exploratory fixed-cohort study'},indent=2),flush=True)
        if preflight_only:return
        receipt=destination/'run.json'
        if receipt.exists():
            record=read(receipt);bundle=verify_run(record['path'])
            if bundle.scientific_payload_sha256!=record['scientific_payload_sha256']:raise ValueError('run receipt mismatch')
        else:
            started=destination/'RUN_STARTED.json'
            if started.exists():raise RuntimeError('unsealed prior attempt: locate and verify its bundle; never launch a duplicate automatically')
            write_new(started,{'plan_sha256':digest_object(identity)})
            result=run_experiment(config,repository_root=ROOT,resources=ExecutionResources(workers=plan['workers'],threads_per_worker=1))
            bundle=verify_run(result)
            write_new(receipt,{'path':str(bundle.path),'scientific_payload_sha256':bundle.scientific_payload_sha256})
        if (bundle.path/'config.toml').read_text()!=config_path.read_text():raise ValueError('run used a different configuration')
        def analysis(where):
            curves,pairs,inventory=audit_static('static_toycar' if name=='toycar' else 'static_'+name,bundle.path,bundle.manifest)
            for filename,rows in [('curves.csv',curves),('paired.csv',pairs)]:
                with (where/filename).open('x',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            write_new(where/'inventory.json',inventory)
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig,axes=plt.subplots(2,2,figsize=(10,7))
            for ax,prob in zip(axes.flat,sorted(plan['probabilities'])):
                for policy in sorted({r['policy'] for r in curves}-{'no_protection','full_protection'}):
                    data=sorted([r for r in curves if r['ber']==prob and r['policy'] in [policy,'no_protection']],key=lambda r:r['coverage'])
                    ax.errorbar([100*r['coverage'] for r in data],[r['mean'] for r in data],
                        yerr=[[r['mean']-r['ci_low'] for r in data],[r['ci_high']-r['mean'] for r in data]],label=policy)
                ax.set_title(f'Logical BER {prob:g}');ax.set_xlabel('Protected eligible bits (%)');ax.set_ylabel(curves[0]['metric'])
            handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,fontsize=7)
            fig.subplots_adjust(bottom=.22,hspace=.4);fig.savefig(where/'budget.png',dpi=180);plt.close(fig)
            (where/'README.txt').write_text('All recorded paired fields; descriptive field-t intervals conditional on fixed inputs/masks. Separate replication counts and within-BER Holm tests remain in the full bundle. Not new-input confirmation.\n')
        verified_stage(destination/'analysis',analysis)
        def exports(where):
            compact_names={'config.toml','resolved_config.json','manifest.json','summary.json','checksums.json','FINALIZED','report.md',
                'measurements/trials.csv','measurements/protection_results.csv','measurements/paired_comparisons.csv','measurements/estimator_runtime.csv'}
            full=[(p,'bundle/'+p.relative_to(bundle.path).as_posix()) for p in sorted(bundle.path.rglob('*')) if p.is_file()]
            compact=[(p,n) for p,n in full if n.removeprefix('bundle/') in compact_names]
            shared=[(p,'analysis/'+p.relative_to(destination/'analysis').as_posix()) for p in sorted((destination/'analysis').rglob('*')) if p.is_file()]
            shared += [(frozen,'plan.json'),(config_path,'frozen_config.toml'),(receipt,'run.json')]
            (where/'COMPACT_SCOPE.txt').write_text('Compact ZIP omits exact score/mask/event/payload files: use full-evidence.zip for complete replay and source checksums.\n')
            archive(where/'compact.zip',compact+shared+[(where/'COMPACT_SCOPE.txt','COMPACT_SCOPE.txt')])
            archive(where/'full-evidence.zip',full+shared)
            commands='\n'.join(f'scp manuelbv@kona:"{where/f}" ~/Downloads/' for f in ['compact.zip','full-evidence.zip'])+'\n'
            (where/'MAC_DOWNLOAD.txt').write_text(commands)
        verified_stage(destination/'exports',exports)
        print((destination/'exports/MAC_DOWNLOAD.txt').read_text())
        print(json.dumps({'status':'complete_verified','run':str(bundle.path),'exports':str(destination/'exports')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('campaign',choices=['resnet8','kws','toycar']);p.add_argument('--preflight-only',action='store_true')
    a=p.parse_args();execute(a.campaign,preflight_only=a.preflight_only)
