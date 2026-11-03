"""Offline label/logit verification; requires the existing pinned PyArrow reader.

No model import or execution. The output is a manuscript verification receipt,
not a modification of the archived analysis or its frozen capped-risk protocol.
"""
from pathlib import Path
import collections
import json
import math
import zipfile

import numpy as np
import pyarrow.parquet as pq

from build_distilbert_evidence import HERE, ROOT, digest_bytes, require, bound


def main():
    pin = json.loads((ROOT/'config/paper1_distilbert_source.json').read_text())
    archive = Path(pin['path'])/'complete.zip'
    require(digest_bytes(archive.read_bytes()) == pin['complete_sha256'], 'archive changed')
    with zipfile.ZipFile(archive) as z:
        plan = json.loads(z.read('plan.json')); bound(plan, 'plan_sha256')
        ready = json.loads(z.read('assets/NETSURF_ASSET_READINESS.json'))
        target, = [x for x in ready['targets'] if x['target_id'] == plan['target']]
        file, = [x for x in target['dataset']['files'] if x['path'].startswith('sst2/validation')]
        path = Path(target['dataset']['path'])/file['path']
        require(digest_bytes(path.read_bytes()) == file['sha256'], 'validation dataset changed')
        labels = pq.read_table(path, columns=['label']).column('label').to_pylist()
        labels = np.asarray([labels[i] for i in plan['final_source_indices']])
        counts = collections.defaultdict(lambda: [0, 0, 0])
        for name in z.namelist():
            if not name.startswith('run/final/ber_'): continue
            record = json.loads(z.read(name)); bound(record, plan=plan['plan_sha256'])
            for policy, value in record['policies'].items():
                risk = value['risk']; logits = np.asarray(risk['logits'], dtype=np.float64)
                require(logits.shape == (256, 2) and np.isfinite(logits).all(),
                        'this archived study requires all finite full logits')
                nll = np.asarray([max(row) + math.log(float(np.sum(np.exp(row-max(row))))) - row[y]
                                  for row, y in zip(logits, labels)])
                require(float(np.mean(np.clip(nll, 0, 30))) == risk['mean_nll'], 'saved capped risk differs')
                require(float(np.mean(np.argmax(logits, axis=1) == labels)) == risk['accuracy'],
                        'saved accuracy differs')
                require(risk['failed_units'] == 0 and all(x is None for x in risk['failure_kinds']),
                        'failure counts contradict finite logits')
                count = counts[(record['arm'], policy)]
                count[0] += int(np.count_nonzero(nll > 30)); count[1] += 256; count[2] += 1
    receipt = dict(schema_version='manuscript-logit-verification/v1', source=pin,
        plan_sha256=plan['plan_sha256'], dataset_sha256=file['sha256'],
        final_source_indices=plan['final_source_indices'], final_records_sha256=plan['final_records_sha256'],
        verified_policy_evaluations=sum(x[2] for x in counts.values()),
        verified_example_outcomes=sum(x[1] for x in counts.values()),
        new_model_inferences=0, source_files_written=0,
        counts_interpretation='Repeated examples and shared fault fields: these cells are NOT independent trials',
        clipping=[dict(ber=plan['ber'][arm], policy=policy, capped_outcomes=c[0],
                       example_outcomes=c[1], fields=c[2]) for (arm, policy), c in sorted(counts.items())])
    (HERE/'evidence/distilbert_logit_verification.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ('verified_policy_evaluations','verified_example_outcomes','new_model_inferences')}))


if __name__ == '__main__':
    main()
