"""Finish manuscript provenance and validate copied evidence; no model execution."""
from pathlib import Path
from fractions import Fraction
import csv
import hashlib
import json
import re
import math
import subprocess

BASE = Path(__file__).resolve().parent
E = BASE / 'evidence'
O = BASE / 'overleaf'

def read(path):
    return json.loads(path.read_text())

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()

def write(path,obj):
    path.write_text(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+'\n')

sources=read(E/'sources.json')
# Retain recorded source revisions, including dirty-state fingerprints. Never
# promote an engineering run to a clean committed execution by changing its ID.
provenance=[]
for key,s in sources.items():
    provenance.append({'source_key':key, 'path':s['path'],
        'payload_sha256':s['scientific_payload_sha256'],
        'commit':s.get('source',{}).get('commit'),
        'dirty':s.get('source',{}).get('dirty'),
        'source_fingerprint_sha256':s.get('source',{}).get('source_fingerprint_sha256'),
        'scope':s['scope']})
write(E/'source_revisions.json',provenance)

claims=[
('C01','Sections 3–4; Appendix A','XOR and exact probability-space derivatives','analytical/implementation contract','pool_000_selection;M910','manifest.benchmark; scientific_payload.campaigns[].pools[].candidates','Independent Bernoulli law; fixed inputs/backend; integer inference is not differentiated.'),
('C02','Section 5.1; Figure 1','Isolated ranking need not minimize finite-BER protected risk','analytical','', 'Equation counterexample, exact eight-state enumeration','Illustrative real-valued function, not a measured network.'),
('C03','Sections 4.2 and 7.1; Table 3','S2–S6 and random sample-limited exact-damage association','measured','emp_resnet;emp_kws;emp_toycar','summary.ranking_metrics_overall','320 stratified candidates; ToyCar target MSE; confidence intervals are not superiority tests; random is one seeded score draw.'),
('C04','Section 7.1; Table 4','S5 versus MSB at both interior budgets and BER .01','measured','static_resnet;static_kws;static_toycar','measurements/paired_comparisons.csv: metric=accuracy or roc_auc; BER=.01','Original paired identity and within-BER/metric Holm family; no cross-task pooling.'),
('C05','Section 4.3','FKeras paper and repository score formulas differ','source correspondence','FKeras','summary.fkeras_diagnostics; pinned upstream hessian.py','Continuous surrogate adaptation; not full original-paper reproduction.'),
('C06','Section 7.2; Table 5','Whole-tier FKeras/MSB masks coincide; partial-tier variants differ','measured','FKeras','summary.mask_equivalence','Exact set comparison, not score correlation. Two-field smoke gives no efficacy conclusion.'),
('C07','Section 7.2','Validated KWS eigenpairs and measured score-construction times','engineering','FKeras','summary.fkeras_diagnostics; estimator_runtimes_seconds; manifest.execution.parallel_execution','Float64 continuous companion; negative eigenvalues retained; tiny cohort.'),
('C08','Section 7.3; Table 6; Figure 2','Conditional signed effects and point ordering change','measured','M952','summary.conditional_estimates; rank_comparisons','Ten explicit bits, fixed train/test inputs; shared candidate pairs not independent trials.'),
('C09','Section 7.3; Table 7','Exploratory frozen selected mask improves relative to S5/isolated','measured','M957','summary.replication_analysis.comparisons','256 independent fields, fixed exploratory pool/cohort; not network-wide generalization.'),
('C10','Section 6; Tables 1–2','Three model populations and six completed campaign/BER arms','provenance','M910','scientific_payload.campaigns; source selection/final manifests','Twelve KWS pools recur; arms are not new independently trained models.'),
('C11','Section 7.4; Table 8; Figure 3','Selected-versus-isolated interval spans zero in every arm','measured','M910','aggregate_summary.comparisons[comparison=restricted_selected_vs_exact_isolated]','Marginal pool-t intervals; unresolved superiority, not equivalence.'),
('C12','Section 7.4; Table 8','Selected-versus-S5 means favorable in all KWS arms','measured','M910','aggregate_summary.comparisons[comparison=restricted_selected_vs_original_s5]','Does not separate proxy error from conditioning; preserve declared testing families.'),
('C13','Sections 5.3 and 7.5; Table 10','Clean + outside + within terms close exactly','analytical and measured','M910','scientific_payload.campaigns[].pools[].selection_attribution; max_decomposition_error','Same selection cohort only; no final-outcome attribution available.'),
('C14','Section 7.5; Table 9','Outside conditioning removes most observed mask-order reversals','measured','M910','scientific_payload.campaigns[].pools[].clean_vs_exact_ordering; conditioned_vs_exact_ordering','Diagnostic near-tie threshold 1e-12; pair counts are not independent samples.'),
('C15','Section 7.6; Figure 4','ToyCar nonzero forced-state responses despite small expected contrasts','measured','M910','scientific_payload.campaigns[risk_metric=one_minus_file_roc_auc].pools[].toycar','Complete-file cohort AUC; forced states not uniformly likely Bernoulli events.'),
('C16','Section 6.2','ToyCar prior exposure 200/248; 40 normal and eight anomalous remaining','provenance','ToyCar_exposure','scientific_payload.json; report.md','All files have clean-reference exposure; unrecorded uses cannot be ruled out.'),
('C17','Sections 5.4 and Appendix A','Sharp bounded-risk remainder and finite-field regret inequality','analytical','','Equations bound and samplebound; Appendix proof','No priority claim; bounded nonmonotone risk, frozen pool, independent fields.'),
('C18','Sections 5.3 and 8.2','Forced-clean singleton mean is not full conditional vulnerability','analytical','','Three-way indicator counterexample','Seven-state construction omits candidate-background averaging.'),
('C19','Section 7.7; Table 11','Historical outside-additive finite-field regret near zero','offline measured','M11_reaudit','scientific_payload.campaigns','Same selection tables, not independent final regret or a global optimum.'),
('C20','Section 7.7; Table 12','Bounded K=8 nonzero regret and K=12 ordering reversals','engineering','M11_oracle','scientific_payload.rows[method=outside]','One/two fields and two inputs/files; not large-scale research evidence.'),
('C21','Section 7.7','Bound is loose compared with observed decision effects','analytical comparison','M11_reaudit;M910','Equation bound with stated p,K,r,L; saved error/effect tables','Formal sharpness does not imply a practically informative certificate.'),
('C22','Section 7.8; Table 13','Recorded engine cost and exact state-operation counts','measured and analytical','M910','costs.recorded_historical_execution_costs; Table 2','Summed engine wall times are not concurrent campaign elapsed time.'),
('C23','Sections 2 and 8','No novelty claim for Boolean foundations, feedback analysis or learned protection','literature/source audit','','BIBLIOGRAPHY_VERIFICATION.md','Zenodo companion dataset is not independently verified peer-reviewed manuscript acceptance.'),
('C24','Sections 7–9','No universal superiority, causal redundancy or physical mitigation guarantee','bounded conclusion','M910;M957;M11_reaudit;M11_oracle','All-arm results and explicit limitations','Negative/unresolved evidence retained; no hardware/radiation measurements.'),
]
with (E/'CLAIM_EVIDENCE.csv').open('w') as f:
    w=csv.writer(f)
    w.writerow(['id','manuscript_location','claim','evidence_type','source_keys','artifact_selector','limits','source_payload_sha256'])
    for row in claims:
        keys=row[4].split(';') if row[4] else []
        w.writerow([*row,';'.join(sources[k]['scientific_payload_sha256'] for k in keys)])

products=read(E/'figure_table_provenance.json')['products']
product_by_path={r['artifact']:r for r in products}
display_specs=[
('tab:models','overleaf/sections/06_experimental_setup.tex',['M910'],'Frozen model identity and kernel word counts from source campaign manifests; bits = 8 × kernel words.'),
('tab:design','overleaf/sections/06_experimental_setup.tex',['M910'],'All six completed campaign designs; no selected-pool exclusion.'),
('tab:static-validation','overleaf/tables/static_rank_validation.tex',['emp_resnet','emp_kws','emp_toycar'],'Saved balanced candidate statistics, three-decimal rounding.'),
('tab:static-protection','overleaf/tables/static_comparison.tex',['static_resnet','static_kws','static_toycar'],'All interior budgets at .01; metric-specific effects; CI reversed when sign reversed.'),
('tab:fkeras','overleaf/tables/fkeras_masks.tex',['FKeras'],'Saved exact mask symmetric differences.'),
('tab:conditional-ranks','overleaf/tables/conditional_ranks.tex',['M952'],'All train/test BER rows, no bit filtering.'),
('tab:replication','overleaf/tables/frozen_replication.tex',['M957'],'Saved signed mean, interval, favorable-field count.'),
('tab:heldout','overleaf/tables/heldout.tex',['M910'],'All six arms and two principal contrasts; equal-pool means.'),
('tab:reversals','overleaf/tables/reversals.tex',['M910'],'Counts summed over all pools, diagnostic tolerance retained.'),
('tab:attribution','overleaf/tables/attribution.tex',['M910'],'Equal-pool signed selection attribution; closure checked.'),
('tab:approximation','overleaf/tables/approximation.tex',['M11_reaudit'],'Saved mean errors and same-selection-field regrets.'),
('tab:oracle','overleaf/tables/oracle.tex',['M11_oracle'],'All bounded outside-method oracle rows.'),
('tab:costs','overleaf/tables/costs.tex',['M910'],'Sum source engine wall times across pools, rounded to seconds.'),
('fig:counterexample','overleaf/sections/05_protection_allocation.tex',[],'Analytical polynomial curves from Equation counterexample. No experimental observations.'),
('fig:conditional','overleaf/figures/conditional_damage.pdf',['M952'],'All ten test candidates, two BERs, saved conditional means and marginal field intervals; isolated x axis in 1e-3 nats/clip.'),
('fig:heldout','overleaf/figures/heldout_pool_effects_paper.pdf',['M910'],'Six panels; every pool and all three saved contrasts; marginal paired-field intervals, no aggregate CI.'),
('fig:toycar','overleaf/figures/toycar_resolution.pdf',['M910'],'Byte-identical finalized figure: empty-candidate cohort AUC CDF and changed forced-state fractions.'),
]
aux=(BASE/'build/main.aux').read_text()
displays=[]
for label,path,keys,recipe in display_specs:
    m=re.search(r'\\newlabel\{'+re.escape(label)+r'\}\{\{([^}]+)\}\{([^}]+)\}',aux)
    assert m,label
    displays.append(dict(label=label,number=m.group(1),page=m.group(2),
        artifact=path,sha256=sha(BASE/path),source_keys=keys,
        source_payload_sha256={k:sources[k]['scientific_payload_sha256'] for k in keys},
        recipe=recipe,generated_product=product_by_path.get(path)))
write(E/'DISPLAY_PROVENANCE.json',{'schema_version':'1.0.0','displays':displays})

# Numerical cross-check independent of the saved aggregate construction.
attr=read(Path(sources['M910']['path'])/'scientific_payload.json')
agg=read(Path(sources['M910']['path'])/'aggregate_summary.json')
from scipy.stats import t
recomputed=[]
for c in attr['campaigns']:
    for contrast in ['restricted_selected_vs_original_s5','restricted_selected_vs_exact_isolated','exact_isolated_vs_original_s5']:
        vals=[next(z['mean_effect']['mean'] for z in p['final_comparisons']['comparisons'] if z['comparison']==contrast) for p in c['pools']]
        n=len(vals); mean=sum(vals)/n
        sd=math.sqrt(sum((v-mean)**2 for v in vals)/(n-1))
        ci=[mean-t.ppf(.975,n-1)*sd/math.sqrt(n),mean+t.ppf(.975,n-1)*sd/math.sqrt(n)]
        a=next(x['pool_summary'] for x in agg['comparisons'] if x['campaign']==c['name'] and x['comparison']==contrast)
        assert math.isclose(mean,a['mean'],abs_tol=1e-15)
        assert all(math.isclose(x,y,abs_tol=1e-15) for x,y in zip(ci,a['confidence_interval']))
        recomputed.append(dict(campaign=c['name'],contrast=contrast,count=n,mean=mean,ci=ci))
write(E/'independent_pool_summary_check.json',recomputed)

# Analytic checks only: no executable network or fault engine imported.
def risk(state):
    z=[Fraction(int(bool(state&(1<<i)))) for i in range(3)]
    return Fraction(6,5)*z[0]+z[1]+Fraction(1,10)*z[2]+3*z[1]*z[2]
for p in [Fraction(1,100),Fraction(1,15),Fraction(1,10)]:
    direct=[]
    for protected in range(3):
        total=Fraction(0)
        for a in range(8):
            if a&(1<<protected): continue
            count=a.bit_count()
            total+=risk(a)*p**count*(1-p)**(2-count)
        direct.append(total)
    assert direct==[Fraction(11,10)*p+3*p*p,Fraction(13,10)*p,Fraction(22,10)*p]
# Sharp construction enumerated by occupancy at several exact rational points.
from math import comb
for m in range(2,9):
    for p in [Fraction(1,1000),Fraction(1,100),Fraction(1,2)]:
        exact=sum(Fraction(comb(m,k))*p**k*(1-p)**(m-k) for k in range(m+1) if k!=1)
        additive=1-m*p
        assert exact-additive==m*p*(1-(1-p)**(m-1))

tex='\n'.join(p.read_text() for p in O.rglob('*.tex'))
citations=set()
for match in re.findall(r'\\cite[a-zA-Z]*\{([^}]+)\}',tex):
    citations.update(match.split(','))
bibkeys=set(re.findall(r'@\w+\{([^,]+),',(O/'references.bib').read_text()))
assert citations==bibkeys, (citations-bibkeys,bibkeys-citations)
assert not re.search(r'\\TODO|\bTBD\b|TODO',tex)
log=(BASE/'build/main.log').read_text()
assert 'Output written' in log
assert not re.search(r'undefined|Overfull|Float too large|Fatal error|Emergency stop',log)
original=Path('/home/manuelbv/.codex/attachments/39f4aa2e-3a8b-4973-8f71-8042d510af67/ACM_TECS___NetSuRF__Encoding_Aware_Fault_Susceptibility_and_Budgeted_Protection_for_Quantized_Neural_Networks.zip')
assert sha(original)==sha(BASE/'original_overleaf.zip')=='3d8fba9df4891bd05792b9a8157712c30fc6dd837526dc7c13e9e2a4ed84a5f6'
# Re-read every consumed source after authoring. Does not claim verification of
# intentionally unconsumed multi-GB prediction files.
checked=0
for s in sources.values():
    assert sha(Path(s['path'])/'checksums.json')==s['checksums_sha256']
    for rel,h in s['verified_files'].items():
        assert sha(Path(s['path'])/rel)==h,(s['path'],rel)
        checked+=1
write(E/'FINAL_VALIDATION.json',{
 'source_bundle_count':len(sources),'consumed_source_files_unchanged':checked,
 'canonical_payload_verification':'PASS in prepare_evidence.py',
 'independently_recomputed_pool_contrasts':len(recomputed),
 'analytical_counterexample_checks':'PASS, exact rational arithmetic',
 'sharp_bound_attainment_checks':'PASS, exact rational arithmetic for m=2..8 at three probabilities',
 'display_count':len(displays),'claim_count':len(claims),'cited_bibliography_entries':len(citations),
 'undefined_citations_or_references':0,'overfull_boxes_or_oversized_floats':0,
 'original_overleaf_sha256':sha(original),'new_model_inferences':0,
 'remaining_build_warnings':['ACM reference block suppressed for internal review','CCS concepts not supplied in original author metadata','nonfatal underfull typography'],
 'source_scope':'consumed inventories only; unused multi-GB prediction records not rehashed',
 'manuscript_source':'user-supplied Overleaf ZIP, not older repository manuscript'})
print(f'PASS: {len(claims)} claims, {len(displays)} displays, {len(recomputed)} independently recomputed pool contrasts; {checked} consumed files unchanged.')
