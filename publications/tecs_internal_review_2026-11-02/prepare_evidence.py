"""Read-only evidence verification and manuscript-only tables/figures. No ML imports.

Run with the installed canonical Python. Historical inputs are never opened for
writing. Output is confined to this publication directory, not scientific runs.
"""
from pathlib import Path
import csv
import hashlib
import json
import shutil
import math

HERE = Path(__file__).resolve().parent
ROOT = Path('/projects/netsurf/manuelbv')
W = ROOT / 'workspace/implementation-worktrees'
OUT = HERE / 'evidence'
TEX = HERE / 'overleaf/tables'
FIG = HERE / 'overleaf/figures'
for d in (OUT, TEX, FIG):
    d.mkdir(exist_ok=True, parents=True)
receipts = {}
products = []


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n')


def bundle(tag, p, files=(), all_files=False):
    p = Path(p)
    assert (p / 'FINALIZED').is_file(), p
    m = read(p / 'manifest.json')
    inv = read(p / 'checksums.json')['files']
    required = set(files) | {'manifest.json', 'scientific_payload.json'}
    if all_files:
        required |= set(inv)
    hashes = {}
    for rel in sorted(required):
        assert rel in inv, (p, rel)
        actual = sha(p / rel)
        assert actual == inv[rel], (p, rel, 'checksum mismatch')
        hashes[rel] = actual
    payload = read(p / 'scientific_payload.json')
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    assert digest == m['scientific_payload_sha256'], (p, 'payload mismatch')
    receipts[tag] = dict(path=str(p), scientific_payload_sha256=digest,
                         checksums_sha256=sha(p / 'checksums.json'), verified_files=hashes,
                         source=m.get('source', {}), versions=m.get('versions', {}),
                         benchmark=m.get('benchmark', {}),
                         scope='all inventoried files' if all_files else 'payload, manifest and explicitly consumed files')
    return payload


def artifact(tag, p, files):
    bundle(tag, p, files)
    return p


def table(name, rows, columns, source, recipe):
    (TEX / name).write_text('\n'.join(' & '.join(map(str, r)) + r' \\' for r in rows) + '\n')
    products.append(dict(artifact='overleaf/tables/' + name, sources=source, recipe=recipe))
    with (OUT / name.replace('.tex', '.csv')).open('w') as f:
        c = csv.writer(f); c.writerow(columns); c.writerows(rows)


def num(x):
    if x == 0:
        return '0'
    exp = math.floor(math.log10(abs(x)))
    return f'${x / 10**exp:.3f}\\times10^{{{exp}}}$'


ATTR = W / 'm910-interaction-attribution/workspace/runs/m910_production_attribution_committed'
attr = bundle('M910', ATTR, all_files=True)
sources = read(ATTR / 'source_verification.json')['verified_before_and_after']
for i, s in enumerate(sources):
    for role in ('selection', 'final'):
        p = Path(s[role + '_bundle'])
        assert sha(p / 'checksums.json') == s[role + '_checksums_sha256']
        v = bundle(f'pool_{i:03d}_{role}', p)
        assert receipts[f'pool_{i:03d}_{role}']['scientific_payload_sha256'] == s[role + '_payload_sha256']
    if i % 12 == 0:
        print(f'Verified historical pool {i + 1}/56', flush=True)

names = {'m96_kws_cross_pool_k6': 'KWS A',
         'm97_kws_multiber_ber_0p001': 'KWS B',
         'm97_kws_multiber_ber_0p003': 'KWS B',
         'm97_kws_multiber_ber_0p01': 'KWS B',
         'm98_resnet8_cross_pool_k6_v1': 'ResNet-8',
         'm99_toycar_cross_pool_k6_ber001': 'ToyCar'}
ag = read(ATTR / 'aggregate_summary.json')['comparisons']
rows = []
for c in attr['campaigns']:
    for comparison in ('restricted_selected_vs_original_s5', 'restricted_selected_vs_exact_isolated'):
        r = next(a for a in ag if a['campaign'] == c['name'] and a['comparison'] == comparison)
        s = r['pool_summary']; lo, hi = s['confidence_interval']
        rows.append([names[c['name']], str(c['logical_ber']), 'S5' if comparison.endswith('original_s5') else 'Isolated', s['count'], num(s['mean']), '[' + num(lo) + ', ' + num(hi) + ']'])
table('heldout.tex', rows, ['campaign', 'BER', 'baseline', 'pools', 'mean', '95% CI'], ['M910/aggregate_summary.json'], 'Saved equal-pool means and marginal t intervals; no re-pooling across arms or metrics.')
rows = []
for c in attr['campaigns']:
    a = [p['clean_vs_exact_ordering'] for p in c['pools']]
    b = [p['conditioned_vs_exact_ordering'] for p in c['pools']]
    rows.append([names[c['name']], str(c['logical_ber']), f"{sum(x['strict_reversals'] for x in a)}/{sum(x['strict_comparable_pairs'] for x in a)}", f"{sum(x['strict_reversals'] for x in b)}/{sum(x['strict_comparable_pairs'] for x in b)}"])
table('reversals.tex', rows, ['campaign','BER','clean/exact','outside/exact'], ['M910/scientific_payload.json:campaigns[].pools[]'], 'Sum strict reversals and comparable pairs, diagnostic tolerance 1e-12. Shared pairs are not independent trials.')
rows = []
for c in attr['campaigns']:
    terms = ['clean_term','outside_correction','within_correction','exact_contrast']
    vals = [sum(p['selection_attribution']['restricted_selected_vs_exact_isolated'][t]['mean'] for p in c['pools']) / len(c['pools']) for t in terms]
    assert math.isclose(sum(vals[:3]), vals[3], abs_tol=1e-13)
    rows.append([names[c['name']], str(c['logical_ber'])] + [num(x) for x in vals])
table('attribution.tex', rows, ['campaign','BER','clean','outside','within','exact'], ['M910/scientific_payload.json:campaigns[].pools[].selection_attribution'], 'Equal-pool averages of signed selected-minus-isolated selection components; closure checked.')
for name in ['heldout_pool_effects','selection_attribution','selection_margin_stability','toycar_resolution']:
    shutil.copy2(ATTR / (name + '.pdf'), FIG / (name + '.pdf'))
    products.append(dict(artifact=f'overleaf/figures/{name}.pdf', sources=[f'M910/{name}.pdf'], recipe='Byte-identical finalized offline figure; original source inventory verified.'))
write('campaign_summary.json', {'campaigns': [{k:v for k,v in c.items() if k != 'pools'} for c in attr['campaigns']], 'comparisons':ag})

REA = W / 'm11-conditional-scaling/workspace/runs/m11_historical_reaudit'
rea = bundle('M11_reaudit', REA, all_files=True)
rows = []
for c in rea['campaigns']:
    rows.append([names[c['name']], str(c['ber']), num(c['outside_max_absolute_error']['mean']), num(c['outside_finite_field_regret']['mean']), num(c['clean_finite_field_regret']['mean'])])
table('approximation.tex', rows, ['campaign','BER','mean max error','outside regret','clean regret'], ['M11_reaudit/scientific_payload.json:campaigns'], 'Mean of per-pool max absolute errors and finite-selection-field regrets. Not held-out regret.')
OR = W / 'm11-conditional-scaling/workspace/runs/m11_oracle_analysis_v2'
oracle = bundle('M11_oracle', OR, all_files=True)
rows = []
for r in oracle['rows']:
    if r['method'] != 'outside': continue
    model = {'aeea':'KWS','3c00':'ResNet-8','87cf':'ToyCar'}[r['checkpoint_sha256'][:4]]
    rows.append([model,r['k'],r['fields'],num(r['max_absolute_error']),num(r['finite_field_selection_regret']),r['strict_reversals'],f"{r['measured_total_evaluation_seconds']:.2f}"])
table('oracle.tex', rows, ['model','K','fields','max error','selection regret','reversals','evaluation seconds'], ['M11_oracle/scientific_payload.json:rows[method=outside]'], 'Bounded two-input engineering validation only. Times include controls, not process startup.')

P952 = ROOT / 'workspace/runs/m952_kws_input_stability_v1/20261030T070657.805986Z-d1906079'
v952 = bundle('M952', P952, all_files=True)
s952 = read(P952 / 'summary.json')
table('conditional_ranks.tex', [[r['split'].replace('held_out_evaluation','Test').replace('calibration','Train'),r['logical_ber'],f"{r['strict_rank_reversals']}/{r['strict_comparable_pairs']}",f"{r['spearman_average_rank']:.3f}"] for r in s952['rank_comparisons']], ['split','BER','reversals','Spearman'], ['M952/summary.json:rank_comparisons'], 'Saved point rank associations, not independent-pair statistical tests.')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':9,'pdf.fonttype':42})
fig, axs = plt.subplots(3, 2, figsize=(7.2, 7.1))
contrast_names = ['restricted_selected_vs_original_s5', 'restricted_selected_vs_exact_isolated', 'exact_isolated_vs_original_s5']
contrast_labels = ['Selected − S5', 'Selected − isolated', 'Isolated − S5']
pool_rows = []
for ax, c in zip(axs.flat, attr['campaigns']):
    for j, (contrast, label) in enumerate(zip(contrast_names, contrast_labels)):
        for i, pool in enumerate(c['pools']):
            s = next(x['mean_effect'] for x in pool['final_comparisons']['comparisons'] if x['comparison'] == contrast)
            lo, hi = s['confidence_interval']; mean = s['mean']
            ax.errorbar(i + .22*(j-1), mean, yerr=[[mean-lo],[hi-mean]], fmt=['o','s','^'][j], color=['#0072B2','#D55E00','#009E73'][j], markersize=3, capsize=2, linewidth=.7, label=label if i==0 else None)
            pool_rows.append(dict(campaign=c['name'], ber=c['logical_ber'], pool=pool['pool_id'], contrast=contrast, mean=mean, ci_low=lo, ci_high=hi, fields=s['count'], risk_metric=c['risk_metric']))
    ax.axhline(0, color='gray', linewidth=.6)
    ax.set_title(f"{names[c['name']]}, BER {c['logical_ber']:g}", fontsize=9)
    ax.set_xlabel('Prespecified pool', fontsize=8)
    ax.set_ylabel('Δ AUC risk' if c['risk_metric']=='one_minus_file_roc_auc' else 'Δ CE (nats/input)', fontsize=8)
    ax.set_xticks(range(len(c['pools'])))
    ax.tick_params(labelsize=7)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
handles, labels = axs.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=8, frameon=False)
fig.subplots_adjust(left=.10, right=.98, top=.95, bottom=.10, hspace=.65, wspace=.42)
fig.savefig(FIG/'heldout_pool_effects_paper.pdf'); plt.close(fig)
with (OUT/'heldout_pool_effects.csv').open('w') as f:
    writer=csv.DictWriter(f, fieldnames=list(pool_rows[0])); writer.writeheader(); writer.writerows(pool_rows)
products.append(dict(artifact='overleaf/figures/heldout_pool_effects_paper.pdf', sources=['M910/scientific_payload.json:campaigns[].pools[].final_comparisons.comparisons[].mean_effect'], recipe='All pools and three contrasts, saved marginal paired-field 95% intervals; six panels with independent risk scales. No aggregate pool interval is plotted. Layout-only replot, original figure retained.'))
fig, axs = plt.subplots(1,2,figsize=(7,3.2), layout='constrained')
for ax, ber in zip(axs, [.001,.01]):
    rows = [r for r in s952['conditional_estimates'] if r['split']=='held_out_evaluation' and r['logical_ber']==ber]
    for i,r in enumerate(rows):
        a=r['conditional_damage']; lo,hi=a['confidence_interval']
        x = 1000*r['isolated_damage']
        ax.errorbar(x,a['mean'],yerr=[[a['mean']-lo],[hi-a['mean']]],fmt='o',markersize=3,color='#1d5a85',capsize=2)
        offset = {0:(-20,22),1:(30,-1),8:(-35,-15),9:(8,32),3:(8,-18)}.get(i, (5,3))
        ax.annotate(str(i),(x,a['mean']),xytext=offset,textcoords='offset points',fontsize=7,arrowprops=dict(arrowstyle='-',lw=.4,color='gray'))
    xlim = ax.get_xlim()
    ax.plot(xlim,[v/1000 for v in xlim],':',color='gray',label='Equality')
    ax.set_xlim(xlim)
    ax.set_title(f'Logical BER {ber:g}; 128 fields')
    ax.set_xlabel('Isolated CE change ($10^{-3}$ nats/clip)');ax.set_ylabel('Conditional CE change (nats/clip)')
    ax.axhline(0,color='gray',lw=.5);ax.ticklabel_format(style='sci',axis='y',scilimits=(-2,2))
fig.savefig(FIG/'conditional_damage.pdf');plt.close(fig)
products.append(dict(artifact='overleaf/figures/conditional_damage.pdf',sources=['M952/summary.json:conditional_estimates'],recipe='All ten test candidates at each BER; saved field-level marginal 95% t intervals. Labels index canonical candidate order.'))

P957=W/'m957/workspace/runs/m957_kws_frozen_replication_k8_v1/20261030T184931.331632Z-27c6efd5'
v957=bundle('M957',P957,all_files=True)
rows=[]
for r in read(P957/'summary.json')['replication_analysis']['comparisons']:
    s=r['mean_effect'];lo,hi=s['confidence_interval']
    rows.append([r['comparison'].replace('restricted_selected_vs_original_s5','Selected--S5').replace('restricted_selected_vs_exact_isolated','Selected--isolated').replace('exact_isolated_vs_original_s5','Isolated--S5'),num(s['mean']),'['+num(lo)+', '+num(hi)+']',r['favorable_fields']])
table('frozen_replication.tex',rows,['contrast','mean','95% CI','favorable fields'],['M957/summary.json:replication_analysis.comparisons'],'All 256 fields retained; fixed exploratory pool and masks.')

P97=W/'m97/workspace/runs/m97_kws_fkeras_protection_smoke/20261031T204633.165614Z-ec595393'
v97=bundle('FKeras',P97,files=['summary.json','measurements/trials.csv'])
s97=read(P97/'summary.json')
rows=[]
for coverage in (.0625,.125,.1875,.25):
    pairs=[('fkeras_paper','fkeras_repository'),('msb_first','fkeras_paper'),('msb_first','fkeras_repository')]
    counts=[]
    for a,b in pairs:
        item=next(x for x in s97['mask_equivalence'] if x['left']==f'{a}@{coverage:.6f}' and x['right']==f'{b}@{coverage:.6f}')
        counts.append(item['symmetric_difference_bits'])
    rows.append([f'{coverage*100:g}\\%',item['protected_bits']]+counts)
table('fkeras_masks.tex',rows,['coverage','protected','paper/repo','MSB/paper','MSB/repo'],['FKeras/summary.json:mask_equivalence'],'Exact saved set symmetric differences, not ranks inferred from score correlations.')
diag=s97['fkeras_diagnostics']
def compact_diagnostics(obj):
    if isinstance(obj, dict):
        return {k:compact_diagnostics(v) for k,v in obj.items() if k not in ['eigenvectors_flat','parameter_scores','raw_parameter_scores']}
    if isinstance(obj, list):
        return [compact_diagnostics(v) for v in obj]
    return obj
write('fkeras_numerics.json', compact_diagnostics(diag))
write('fkeras_timings.json', {'score_seconds':s97['estimator_runtimes_seconds'],
    'execution':read(P97/'manifest.json')['execution']['parallel_execution']})

static=[]
for tag,pattern in [('static_resnet','m92_resnet8_static_test1000_v1'),('static_kws','m93_kws_static_ber_sweep1000_v1'),('static_toycar','m93_toycar_static_ber_pilot64_parallel_repair_v1')]:
    paths=[p for p in (ROOT/'workspace/runs'/pattern).iterdir() if (p/'FINALIZED').exists()]
    assert len(paths)==1
    p=paths[0];bundle(tag,p,files=['measurements/paired_comparisons.csv','measurements/protection_results.csv'])
    for r in csv.DictReader((p/'measurements/paired_comparisons.csv').open()):
        if {r['left'],r['right']}=={'per_example_absolute_global','msb_global'}:
            sign=1 if r['left']=='per_example_absolute_global' else -1
            x=dict(source=tag,**r)
            x['s5_minus_msb']=sign*float(r['mean_difference'])
            x['s5_minus_msb_ci']=sorted([sign*float(r['ci_low']),sign*float(r['ci_high'])])
            static.append(x)
    shutil.copyfile(p/'measurements/paired_comparisons.csv',OUT/f'{tag}_all_paired.csv')
# Retain all comparisons, not merely favorable illustrative rows.
write('static_s5_msb.json',static)
rank_rows=[]
for tag,model,pattern in [('emp_resnet','ResNet-8','m91_resnet8_estimator_validation_stratified320_v1'),('emp_kws','KWS','m93_kws_estimator_validation_stratified320_v1'),('emp_toycar','ToyCar MSE','m93_toycar_estimator_validation_stratified320_v1')]:
    paths=[p for p in (ROOT/'workspace/runs'/pattern).iterdir() if (p/'FINALIZED').exists()]
    assert len(paths)==1
    p=paths[0]; bundle(tag,p,files=['summary.json'])
    for r in read(p/'summary.json')['ranking_metrics_overall']:
        if r['estimator'].startswith(('s2_','s3_','s4_','s5_','s6_','random_')):
            label='Random' if r['estimator']=='random_reference' else r['estimator'].split('_')[0].upper()
            rank_rows.append([model,label,f"{r['spearman_positive_damage']:.3f}",f"[{r['spearman_positive_damage_ci_low']:.3f}, {r['spearman_positive_damage_ci_high']:.3f}]",f"{r['runtime_seconds']:.2f}"])
table('static_rank_validation.tex',rank_rows,['model/loss','score','Spearman','sample-design 95% interval','seconds'],['emp_resnet/summary.json','emp_kws/summary.json','emp_toycar/summary.json'],'Saved 320-candidate, balanced-stratum metrics against positive loss damage. ToyCar target is reconstruction MSE, not ROC-AUC. S5/S6 share gradient computation and duplicated runtime must not be summed.')
rows=[]
for tag,model,metric in [('static_resnet','ResNet-8','accuracy'),('static_kws','KWS','accuracy'),('static_toycar','ToyCar','roc_auc')]:
    for r in static:
        if r['source']==tag and r['metric']==metric and float(r['logical_ber'])==.01:
            lo,hi=r['s5_minus_msb_ci']
            rows.append([model,f"{100*float(r['coverage']):g}\\%",r['matched_pairs'],f"{r['s5_minus_msb']:+.5f}",f'[{lo:+.5f}, {hi:+.5f}]',f"{float(r['p_value_holm']):.4f}"])
table('static_comparison.tex',rows,['model','coverage','fields','S5-MSB task metric','95% CI','saved Holm p'],['static_resnet/measurements/paired_comparisons.csv','static_kws/measurements/paired_comparisons.csv','static_toycar/measurements/paired_comparisons.csv'],'All two interior budgets at BER .01; flip sign and reverse endpoints when saved comparison is MSB minus S5. Accuracy and AUC are distinct task metrics. Preserve original within-BER/metric multiplicity families.')
costs=read(ATTR/'costs.json')['recorded_historical_execution_costs']
table('costs.tex',[[names[c['name']],c['logical_ber'],f"{sum(x['selection_runtime_seconds'] for x in costs if x['campaign']==c['name']):.0f}",f"{sum(x['final_runtime_seconds'] for x in costs if x['campaign']==c['name']):.0f}"] for c in attr['campaigns']],['campaign','BER','selection seconds','final seconds'],['M910/costs.json'],'Sum saved engine wall times across pools. These are not concurrent elapsed campaign times or machine-independent speedups.')
write('paper_numeric_inputs.json', {
    'M952_summary':s952,
    'M957_comparisons':read(P957/'summary.json')['replication_analysis']['comparisons'],
    'M11_reaudit_campaigns':rea['campaigns'], 'M11_oracle_rows':oracle['rows'],
    'recorded_costs':costs,
    'all_pool_selection_diagnostics':[{ 'campaign':c['name'], 'ber':c['logical_ber'],
        'pools':[{k:p[k] for k in ('pool_id','clean_selection_risk','background_risk','clean_vs_exact_ordering','conditioned_vs_exact_ordering','selection_attribution','max_decomposition_error')} for p in c['pools']] } for c in attr['campaigns']]})
exposure = W / 'm910-interaction-attribution/workspace/runs/m910_toycar_exposure_v1'
exp_inventory = read(exposure / 'checksums.json')['files']
for rel, expected in exp_inventory.items():
    assert sha(exposure / rel) == expected
exp_digest = hashlib.sha256(json.dumps(read(exposure / 'scientific_payload.json'), sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
assert (exposure / 'FINALIZED').read_text().strip() == 'scientific_payload_sha256=' + exp_digest
receipts['ToyCar_exposure'] = dict(path=str(exposure), scientific_payload_sha256=exp_digest,
    checksums_sha256=sha(exposure / 'checksums.json'), verified_files=exp_inventory,
    scope='all inventoried files; this ledger format has a FINALIZED payload receipt rather than a separate manifest')
write('sources.json',receipts)
write('figure_table_provenance.json',{'schema_version':'1.0.0','sources':'sources.json','products':products,'new_inferences':0})
write('verification_summary.json',{'bundles':len(receipts),'historical_selection_final_bundles':112,'new_model_inferences':0,'source_files_written':0,'all_passed':True,'scope':'All derived attribution and cited conditional bundles; historical campaign payloads/manifests/checksum identities; only cited measurement files of very large static bundles. No claim of rehashing unused multi-GB prediction tables.'})
print(f'PASS: {len(receipts)} bundles verified; {len(products)} manuscript products. No inference or source writes.')
