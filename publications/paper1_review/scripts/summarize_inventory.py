"""Compact writer handoff and measured cost receipts; no model execution."""
import csv,hashlib,json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];D=ROOT/'data'
B=Path('/projects/netsurf/manuelbv/workspace/implementation-worktrees')
def load(p):return json.loads(p.read_text())
def rows(p):
 with p.open(newline='') as f:return list(csv.DictReader(f))
def savecsv(p,r):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(r[0]));w.writeheader();w.writerows(r)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
meta=load(D/'benchmark_inventory.json');regs={r['model_key']:r for r in rows(B/'reliability-db/workspace/m12_current_comparative_v3/study_registry.csv')}
for r in meta:
 if r['model'] in ['resnet8','kws','toycar']:
  for rep,dd in r['fkeras_diagnostics'].items():
   d=dd['fkeras_paper']
   for key in ['eigenvectors_flat','raw_parameter_score']:d.pop(key,None)
  man=load(Path(r['source'])/'manifest.json');r['measured_cost']=man['execution']['parallel_execution']
 else:
  plan=load(Path(r['source'])/'plan.json');old=Path(plan['source_plan_path'])
  assert digest(old)==plan['source_plan_file_sha256']
  orig=load(old);full=regs[r['model']]['checkpoint_sha256']
  assert full.startswith(orig['model_commit'])
  r['checkpoint_sha256']=full
  r['checkpoint_identity_scope']='full checksum from finalized original source registry, source frozen plan file rehashed and model_commit prefix matched'
  r['loss_cap']='Each finite per-unit NLL is capped at 30 nats; nonfinite units also assigned 30. Failed-unit count does not quantify finite clipping.'
  r['source_plan_file_sha256']=digest(old)
  man=load(Path(r['source'])/'manifest.json');ts=[];verified=[];byber=defaultdict(list)
  for name,h in man['scientific_file_sha256'].items():
   if not name.startswith('fields/') or not name.endswith('.json'):continue
   p=Path(r['source'])/name;j=load(p)
   if 'elapsed_seconds' not in j:continue
   assert digest(p)==h
   if j.get('origin')=='authoritative_faulted_inference':
    ts.append(j['elapsed_seconds']);byber[j['ber']].append(j['elapsed_seconds']);verified.append((name,h))
  r['measured_cost']={'authoritative_faulted_requests':len(ts),'sum_request_seconds':sum(ts),
                       'median_request_seconds':float(np.median(ts)), 'p95_request_seconds':float(np.quantile(ts,.95)),
                       'mean_request_seconds_by_ber':{str(k):float(np.mean(v)) for k,v in byber.items()},
                       'scope':'sum of measured request service times across concurrent shards, not elapsed wall time; reused full-protection controls excluded',
                       'verified_timing_records':len(verified),'verified_record_index_sha256':hashlib.sha256(json.dumps(verified,sort_keys=True).encode()).hexdigest(),
                       'peak_memory':'not recorded in finalized wide-run manifest','ranking_time':'inherited sealed masks; not reconstructed from wall timestamps'}
(D/'benchmark_inventory.json').write_text(json.dumps(meta,indent=2)+'\n')
table=[]
for r in meta:
 d={'model':r['name'],'eligible_bits':r['eligible_bits'],'clean_loss':r['clean_loss'],
    'BERs':';'.join(f'{p:g}' for p in r['bers']),'fields_per_BER':r['fields_per_ber'],'methods':7 if r['model'] in ['resnet8','kws','toycar'] else 6,
    'conditions':r['conditions'],'trial_observations':r['trial_observations'],'cohort':r.get('evaluation_examples',r.get('evaluation_cohort',{}).get('size',4096)),
    'cohort_unit':r.get('evaluation_unit','examples' if r['model']=='distilbert' else 'scored tokens'),
    'source_payload_sha256':r['source_payload_sha256']}
 table.append(d)
savecsv(D/'model_table.csv',table)
lines=['# Empirical evidence handoff','',
'All tables/figures in this directory are derived from finalized source measurements. Every plotted condition mean was reconstructed from saved policy-field observations; no inference was run. Hash receipts record the exact data used. Large original mask/event/score payloads are linked by their source manifests rather than all rehashed in this manuscript build.','',
'## Main current comparisons','',
'| Model | Eligible stored INT8 bits | Final cohort | BERs | Fields / BER | Methods | Conditions | Policy-field observations |',
'|---|---:|---|---:|---:|---:|---:|---:|']
for r in table:lines.append(f"| {r['model']} | {r['eligible_bits']:,} | {r['cohort']} {r['cohort_unit']} | {len(r['BERs'].split(';'))} | {r['fields_per_BER']} | {r['methods']} | {r['conditions']} | {r['trial_observations']} |")
lines+=['','All use 6.25%, 12.5%, 18.75%, and 25% interior protection, plus zero/full controls. ResNet, KWS, DistilBERT, Pythia, and Qwen BERs are 1e-5, 1e-4, 3e-4, .001, .003, .01, .03, and .1. ToyCar BERs are 1e-5, 1e-4, .001, .003, and .01.','',
'The combined presentation has 1,254 conditions and 35,136 labeled policy-field observations over 1,248 separately scheduled fields across six studies. These totals are accounting, not independent cross-model replications. TFLite FKeras/MSB equality at 12.5% and 25% contributes 1,344 duplicate labeled outcomes; repeated full-protection controls are also not independent clean measurements. W8A8 full controls are reused clean references: 4,800 actual faulted requests/model versus 4,992 saved observations/model.','',
'The three TFLite studies use seven methods and four schedule replications with eight fields each. Saved mask hashes verify four random masks and exactly one mask for every other method at each budget. The transformer studies use six methods and one sealed mask per method/budget. Confidence intervals condition on frozen model, cohort and masks; they do not quantify generalization over checkpoints or fresh inputs. For the TFLite random baseline, pooled field intervals combine four fixed sampled masks and do not separately identify population-level mask-construction uncertainty.','',
'## FKeras and computation','',
'Published formula only; absolute value of signed leading-Hessian projection, strictly within descending bit tiers. All three current TFLite studies use two signed largest-magnitude eigenpairs from a float64 continuous decoded-weight companion, ARPACK, tolerance 1e-5, and 25 HVPs per replicate. Activation requantization is removed; fixed decoded bias is excluded from differentiated/faultable population. No full integer-program Hessian is claimed. The same 64 ranking images/clips as gradients are used for ResNet/KWS; ToyCar uses all 3,136 windows from 16 training files, and evaluates protection on complete-file 1-AUC over 64 files/12,544 windows.','',
'Score costs (mean of 4 builds, seconds):','',
'| Model | Global gradient | Shared absolute/positive gradient | Rank-2 FKeras | Full study elapsed h | Coordinator peak GiB |',
'|---|---:|---:|---:|---:|---:|']
cost=rows(D/'score_construction_costs.csv')
for r in meta[:3]:
 x={q['estimator']:float(q['mean_seconds']) for q in cost if q['model']==r['model']};c=r['measured_cost']
 lines.append(f"| {r['name']} | {x['s4_gradient_delta']:.3f} | {x['s5_per_example_absolute']:.3f} | {x['fkeras_paper']:.3f} | {c['coordinator_elapsed_seconds']/3600:.2f} | {c['coordinator_peak_rss_mb']/1024:.2f} |")
num=[]
for r in meta[:3]:
 for rep,z in r['fkeras_diagnostics'].items():
  q=z['fkeras_paper'];e=q['eigenpairs']
  num.append({'model':r['model'],'replication':rep,'implementation':q['companion']['implementation_version'],
              'eigenpairs':len(e['eigenvalues']),'hvp_calls':e['hvp_calls'],
              'maximum_normalized_residual':max(e['normalized_residuals']),
              'orthogonality_error':e['orthogonality_error'],'solver':e['solver'],
              'ordering':e['ordering'],'eigenvalues':json.dumps(e['eigenvalues'])})
savecsv(D/'fkeras_numerical_receipts.csv',num)
lines+=['','Maximum normalized eigenpair residual across the four builds: ' + '; '.join(
 f"{r['name']} {max(q['maximum_normalized_residual'] for q in num if q['model']==r['model']):.3g}"
 for r in meta[:3]) + '. All two leading eigenpairs converged. Exact eigenvalues, orthogonality errors, implementation versions, and HVP counts are retained in data/fkeras_numerical_receipts.csv.']
lines+=['','Absolute/positive share a Jacobian; their timing must not be added. The rank-2 FKeras cost is not a claim about full Hessian analysis. Timestamp fields in older parent manifests reflect an early coordinator phase; use the explicitly saved coordinator elapsed receipt.','',
'Measured wide-run request service times (sum over concurrent shards, not wall time):']
for r in meta[3:]:
 c=r['measured_cost'];lines.append(f"- {r['name']}: median {c['median_request_seconds']:.2f} s, p95 {c['p95_request_seconds']:.2f} s, summed service {c['sum_request_seconds']/3600:.2f} h over {c['authoritative_faulted_requests']} requests. Peak memory and total score-building wall time are not recorded in these finalized wide-run manifests; no invented estimate is supplied.")
lines+=['','## Manuscript-critical findings','',
'At 1% BER and 6.25% protection, FKeras−MSB is +3.0347 [1.8351, 4.2342] nats/image on ResNet but −1.6777 [−2.4252, −.9302] nats/clip on KWS; both original Holm-adjusted p=.01120. This across-model contrast is not pooled in common units. The ResNet 320-field focused comparison independently gives +2.2414 [1.8896, 2.5931], Holm p=.0012. The focused ToyCar 384-field comparison at 18.75% and 1% BER gives FKeras−absolute-gradient risk +.11059 [.09200, .12918], Holm p=.0012. Focused studies are supplemental, never pooled into the main grid.','',
'At 1% BER, Pythia global-gradient−MSB is −4.8228 nats/token at 12.5% but +.7712 at 25%; both original 160-comparison Holm p=.016. The Qwen independent-field exploratory follow-up at 12.5% reverses absolute-gradient−MSB from −9.2968 [−10.0205, −8.5671] at 5% BER to +1.6368 [.6051, 2.6609] at 7.5%. Finite uncapped means reverse too; finite cap frequency nonetheless rises to 7.07% MSB/10.49% absolute at 7.5%.','',
'The new KWS 32-pool prespecified primary conditional−exact-isolated effect is −.0000824 [−.0016935, .0015287] nats/clip, p=.9176. In 27/32 pools masks differ; decision changes did not establish utility improvement. Both conditional and exact-isolated improve slightly over the original absolute-gradient global context. The recovered campaign-level plan controls the primary comparison; generic per-pool report primary labeling is inconsistent and is not used.','',
'These support condition-dependent effectiveness, not causal attribution to multibit interactions, not universal optimizer superiority, and not calibrated radiation reliability.','',
'## Exact-bit score validation','',
'data/empirical_validation.csv contains all six methods against 320 stratified exact candidate-bit damages, source bootstrap CIs and timing. These are sample-limited and are not population-weighted whole-network correlations. ToyCar positive-MSE versus positive-AUC correlations are kept separate in data/toycar_auc_correlations.csv. All 320 score identities were joined to their exact bit-damage measurements after streaming/hash-verifying the original prediction file.','',
'## Missing/limited support','',
'FKeras not measured on the three W8A8 transformers; PrioriFI-derived allocation has small engineering receipts but no finalized matched protection-effectiveness campaign located. Historical qPolarGrad absent from current main grids. Do not fill missing curves or compare a scheduler as if it were a validated protection allocator.','',
'## Figure use','',
'- reliability_ber_125.pdf: six model panels at 12.5%; reliability_ber_250.pdf: the same at 25%. The three classification panels use an explicitly labeled asinh axis (linear near 0) to preserve beneficial and catastrophic tails.','- reliability_budget_001.pdf: all six methods/seven where available, all four interior budgets, common 1% BER.','- paired_reversals_and_kws.pdf: interpretable paired effect signs and the full 32-pool distribution with mean CI.','- reliability_all_budgets_supplement.pdf: all 24 model/budget panels with every measured BER.','- SVG and PNG versions are supplied for review. All curves connect observed cells only. No inferred FKeras or PrioriFI results.']
(ROOT/'EVIDENCE_HANDOFF.md').write_text('\n'.join(lines)+'\n')
print('Compact inventory, model table, costs, evidence handoff saved.')
