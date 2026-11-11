"""Derive compact empirical-validation and construction-cost tables offline."""
import csv,hashlib,json
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data'
REPO=Path('/projects/netsurf/manuelbv');BASE=REPO/'workspace/implementation-worktrees'
SOURCES={'resnet8':REPO/'workspace/runs/m91_resnet8_estimator_validation_stratified320_v1/20261029T014136.049835Z-c23f1d77',
         'kws':REPO/'workspace/runs/m93_kws_estimator_validation_stratified320_v1/20261029T172802.081387Z-f3ebf3fe',
         'toycar':REPO/'workspace/runs/m93_toycar_estimator_validation_stratified320_v1/20261029T181531.665041Z-bbb64f74'}
def load(p):return json.loads(p.read_text())
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def rows(p):
 with p.open(newline='') as f:return list(csv.DictReader(f))
def savecsv(p,r):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(r[0]));w.writeheader();w.writerows(r)
receipts=[];metrics=[];parts={}
for model,p in SOURCES.items():
 assert (p/'FINALIZED').exists()
 manifest=load(p/'manifest.json');outs={r['path']:r['sha256'] for r in manifest['outputs']}
 for name in ['summary.json','measurements/empirical_bits.csv','measurements/ranking_metrics.csv','measurements/estimator_runtime.csv']:
  actual=digest(p/name);assert actual==outs[name];receipts.append({'path':str(p/name),'sha256':actual})
 summary=load(p/'summary.json');parts[model]=manifest['benchmark']
 for r in summary['ranking_metrics_overall']:
  metrics.append({'model':model,'estimator':r['estimator'],'metric':'positive reconstruction-MSE damage' if model=='toycar' else 'positive cross-entropy damage',
                  'candidate_count':r['candidate_count'],'spearman':r['spearman_positive_damage'],
                  'ci95_low':r['spearman_positive_damage_ci_low'],'ci95_high':r['spearman_positive_damage_ci_high'],
                  'runtime_seconds':r['runtime_seconds'],'top10_tie_aware_recall':r['top_10_tie_aware_expected_recall'],
                  'bootstrap_design':r['bootstrap_design']})
 print(model,'summary verified',flush=True)

# Stream scores only for the 320 actual injected identities; no multi-GB frame.
p=SOURCES['toycar'];m=load(p/'manifest.json');outs={r['path']:r['sha256'] for r in m['outputs']}
pred=p/'measurements/estimator_predictions.csv';wanted=[];h=hashlib.sha256()
with pred.open('rb') as f:
 header=f.readline();h.update(header)
 for line in f:
  h.update(line)
  if line.rstrip().endswith(b',True'):wanted.append(line.decode())
assert h.hexdigest()==outs['measurements/estimator_predictions.csv']
receipts.append({'path':str(pred),'sha256':h.hexdigest(),'selection':'all empirical_candidate=True rows, one pass'})
scores=list(csv.DictReader([header.decode()]+wanted));emp=rows(p/'measurements/empirical_bits.csv')
key=lambda r:(r['parameter'],r['flat_index'],r['bit_index'])
damage={key(r):max(0,float(r['primary_metric_damage'])) for r in emp}
auc=[]
for est in sorted({r['estimator'] for r in scores}):
 sub=[r for r in scores if r['estimator']==est and key(r) in damage]
 assert len(sub)==320,(est,len(sub))
 auc.append({'model':'toycar','estimator':est,'candidate_count':320,'metric':'positive complete-file ROC-AUC damage',
             'spearman':float(spearmanr([float(r['score']) for r in sub],[damage[key(r)] for r in sub]).statistic),
             'uncertainty':'descriptive correlation; no new inferential comparison or population weighting'})
savecsv(DATA/'empirical_validation.csv',metrics);savecsv(DATA/'toycar_auc_correlations.csv',auc)
(DATA/'empirical_partitions.json').write_text(json.dumps(parts,indent=2)+'\n')
(DATA/'empirical_provenance.json').write_text(json.dumps(receipts,indent=2)+'\n')

rt=rows(DATA/'estimator_runtime.csv');cost=[]
for model in ['resnet8','kws','toycar']:
 for est in sorted({r['estimator'] for r in rt if r['model']==model}):
  ss=[float(r['runtime_seconds']) for r in rt if r['model']==model and r['estimator']==est]
  cost.append({'model':model,'estimator':est,'replications':len(ss),'mean_seconds':float(np.mean(ss)),
               'min_seconds':min(ss),'max_seconds':max(ss),'note':'shared Jacobian cost; do not add absolute and positive timings' if est.startswith(('s5','s6')) else 'source timing excludes unrelated inference/export'})
savecsv(DATA/'score_construction_costs.csv',cost)
db=BASE/'reliability-db/workspace/m12_current_comparative_v3';pc=rows(db/'paired_contrasts.csv')
fkeras=[r for r in pc if (r['left_method']=='fkeras_paper' or r['right_method']=='fkeras_paper') and r['metric'] in ['categorical_crossentropy','one_minus_file_roc_auc','roc_auc']]
savecsv(DATA/'fkeras_paired_comparisons.csv',fkeras)
print('Saved empirical and FKeras tables; ToyCar AUC:',auc,flush=True)
