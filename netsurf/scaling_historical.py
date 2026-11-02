"""Reaudit ALL frozen M910 production pools using M11 exact/additive definitions."""
import argparse
import json
from pathlib import Path
from time import monotonic

from .interaction_attribution import source_paths,mean_interval
from .interaction_evidence import read_selection
from .cross_pool import load_campaign,verified_receipt,_receipt
from .conditional_scaling import compare_oracle,error_bound
from .scaling_study import VERSION,PROTOCOL,_seal_directory
from .manifests import digest_object,sha256_file


def analyze(sources,output):
    start=monotonic(); campaigns=[];source_checks=[]
    for path in source_paths(sources):
        c=load_campaign(path);sha=sha256_file(path);p=c['design']['logical_ber'];rows=[]
        for pool in c['pools']:
            receipt,payload=verified_receipt(_receipt(path.parent,pool['pool_id'],'selection'),sha,pool['pool_id'],'selection')
            root=Path(receipt['bundle'])
            _,clean,tables,metadata,summary=read_selection(root,payload,pool,c)
            k=len(pool['candidates']);budget=summary['budget']
            a=compare_oracle(list(tables.values()),clean,k,budget,p,summary['original_mask_indices'])
            a.update(pool_id=pool['pool_id'],candidate_sha256=digest_object(pool['candidates']),k=k,budget=budget,
                fields=len(tables),linear_evaluations=len(tables)*(k+1),exact_evaluations=len(tables)*2**k,
                bounded_risk_error_limit=error_bound(k-budget,p,1. if 'toycar' in c['name'] else 27.631021115928547),
                selection_payload_sha256=digest_object(payload))
            rows.append(a)
            source_checks.append({'bundle':str(root),'checksums_sha256':sha256_file(root/'checksums.json'),
                'payload_sha256':digest_object(payload)})
        campaigns.append({'name':c['name'],'ber':p,'pools':rows,'campaign_sha256':sha,
            'outside_max_absolute_error':mean_interval([r['diagnostics']['outside']['max_absolute_error'] for r in rows]),
            'outside_finite_field_regret':mean_interval([r['diagnostics']['outside']['finite_field_selection_regret'] for r in rows]),
            'clean_finite_field_regret':mean_interval([r['diagnostics']['clean']['finite_field_selection_regret'] for r in rows])})
        print(f"Verified {c['name']}: {len(rows)} pools",flush=True)
    for row in source_checks:
        path=Path(row['bundle'])
        if sha256_file(path/'checksums.json')!=row['checksums_sha256'] or digest_object(json.loads((path/'scientific_payload.json').read_text()))!=row['payload_sha256']:
            raise ValueError('historical source changed during reanalysis')
    payload={'schema_version':VERSION,'protocol':PROTOCOL,'checkpoint_sha256':'multiple-historical-checkpoints',
        'new_inferences':0,'campaigns':campaigns,'sources_verified_before_after':source_checks,
        'limitations':['Exploratory reuse of the complete M910 source inventory; no additional evidence of generalization.',
            'KWS pool identities repeat across BER/cohort arms; do not pool as independent candidates.',
            'Finite-field regret uses selection data; independent final effects remain in the M910 report.']}
    lines=['# M11 historical exact-response reaudit','','| Campaign | BER | Pools | Outside max error, pool mean | Outside selection regret | Clean selection regret |',
        '|---|---:|---:|---:|---:|---:|']
    for c in campaigns:
        lines.append(f"| {c['name']} | {c['ber']} | {len(c['pools'])} | {c['outside_max_absolute_error']['mean']:.8g} | {c['outside_finite_field_regret']['mean']:.8g} | {c['clean_finite_field_regret']['mean']:.8g} |")
    lines+=['',*payload['limitations'],'','All pools, masks and outside fields retained. Curvature bounds apply only to these observed fields.']
    return _seal_directory(output,payload,metadata={'offline_seconds':monotonic()-start},report='\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(analyze(args.sources,args.output),indent=2))

if __name__=='__main__':main()
