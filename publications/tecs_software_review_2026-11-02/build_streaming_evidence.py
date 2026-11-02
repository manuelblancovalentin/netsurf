"""Offline engineering plot from two immutable short-stream validation bundles."""
from pathlib import Path
import csv
import hashlib
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    runs=[ROOT/'workspace/runs'/n for n in
          ['paper1_streaming_gate_v1','paper1_streaming_gate_repeat_v1']]
    payloads=[]
    receipts=[]
    for run in runs:
        inventory=json.loads((run/'checksums.json').read_text())
        for name,digest in inventory['files'].items():
            assert sha(run/name)==digest
        payload=json.loads((run/'scientific_payload.json').read_text())
        canonical=json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False)
        digest=hashlib.sha256(canonical.encode()).hexdigest()
        assert json.loads((run/'manifest.json').read_text())['scientific_payload_sha256']==digest
        assert (run/'FINALIZED').read_text().strip()==f'scientific_payload_sha256={digest}'
        payloads.append(payload)
        receipts.append({'path':str(run),'scientific_payload_sha256':digest,
                         'checksums_sha256':sha(run/'checksums.json')})
    assert payloads[0]==payloads[1]
    p=payloads[0]
    out=ROOT/'workspace/streaming-derived-v2'
    out.mkdir(exist_ok=False)
    rows=[{'field':r['trial'],'policy':r['policy'],'coverage':r['coverage'],
           'protected_bits':r['protected_bits'],'false_positives':r['evaluation']['false_positives'],
           'false_negatives':r['evaluation']['false_negatives'],
           **r['events']} for r in p['trials']]
    with (out/'measurements.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    fig,ax=plt.subplots(figsize=(6,3.6))
    for method,label,color in [('msb_first','MSB-first','#222222'),
            ('random','Seeded random','#777777'),('numerical_jump','Numerical jump','#D89000')]:
        selected=[r for r in rows if r['policy'].startswith(method+'@') or r['policy']=='no_protection']
        coverage=sorted({r['coverage'] for r in selected})
        means=[np.mean([r['false_negatives'] for r in selected if r['coverage']==c]) for c in coverage]
        ax.plot(np.array(coverage)*100,means,'o--',label=label,color=color)
        for r in selected:
            ax.plot(r['coverage']*100,r['false_negatives'],'.',color=color,alpha=.5)
    ax.axhline(p['clean']['false_negatives'],color='#0072B2',ls=':',label='Clean / full protection')
    ax.set(xlabel='Protected eligible bits (%)',ylabel='False negatives (complete stream)',
           title='Engineering fixture only: two paired fields, logical BER 0.001',
           xticks=[0,6.25,12.5],yticks=[0,1,2],ylim=(-.15,2.15))
    ax.legend(fontsize=8,loc='upper left');fig.tight_layout();fig.savefig(out/'false_negative_budget.png',dpi=220)
    plt.close(fig)
    (out/'provenance.json').write_text(json.dumps({'sources':receipts,'independent_fields':2,
        'repeated_execution_is_not_extra_replication':True,
        'false_positive_range':[min(r['false_positives'] for r in rows),max(r['false_positives'] for r in rows)],
        'interpretation':'Points: two complete-stream outcomes; lines: paired-field means, no CI. No BER sweep or population claim.',
        'files':{f.name:sha(f) for f in out.iterdir()}},indent=2)+'\n')
    print(out)


if __name__=='__main__':main()
