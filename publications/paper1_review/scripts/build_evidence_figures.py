"""Offline, source-bound manuscript figures. Does not import a model runtime.

Run with /mnt/raid0/ml/conda_envs/manuelbv/envs/netsurf/bin/python.
Only files inside this manuscript directory are written.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('/projects/netsurf/manuelbv/workspace/implementation-worktrees')
DB = BASE / 'reliability-db/workspace/m12_current_comparative_v3'
OUT = ROOT / 'figures'
DATA = ROOT / 'data'
for folder in (OUT, DATA):
    folder.mkdir(parents=True, exist_ok=True)

COLORS = {'random':'#8A8F98','msb':'#242A33','jump':'#E69F00','global':'#0072B2',
          'absolute':'#009E73','positive':'#D55E00','fkeras':'#A64B8D', 'unprotected':'#8996A5'}
LABELS = {'random':'Random','msb':'MSB-first','jump':'Numerical jump','global':'Global gradient',
          'absolute':'Absolute gradient','positive':'Positive gradient','fkeras':'FKeras (published)',
          'unprotected':'Unprotected', 'full':'Clean / full protection'}
ORDER = list(COLORS)
MAP = dict(zip(['random_global','msb_global','numerical_jump_global','gradient_global',
                'per_example_absolute_global','per_example_positive_global','fkeras_paper','no_protection','full_protection'],
               ['random','msb','jump','global','absolute','positive','fkeras','unprotected','full']))
WMAP = {'random':'random','msb_first':'msb','numerical_bit_jump':'jump','global_first_order':'global',
        'per_example_absolute':'absolute','per_example_positive':'positive','unprotected':'unprotected','full_protection':'full'}
NAMES = {'resnet8':'ResNet-8 / CIFAR-10','kws':'DS-CNN / Speech Commands','toycar':'Autoencoder / ToyCar',
         'distilbert':'DistilBERT / SST-2','pythia':'Pythia-160M / WikiText-2','qwen':'Qwen3-0.6B / WikiText-2'}
MODELS = list(NAMES)
UNIT = {'resnet8':'CE increase (nats/image)','kws':'CE increase (nats/clip)',
        'toycar':'AUC decrease','distilbert':'CE increase (nats/example)',
        'pythia':'Capped NLL increase (nats/token)','qwen':'Capped NLL increase (nats/token)'}
WIDE = {'distilbert':BASE/'w8a8-ber-stress/workspace/runs/w8a8_wide_ber_matched_v1/distilbert_sst2_int8',
        'pythia':BASE/'pythia-wide-confirm/workspace/runs/pythia_w8a8_high_ber_clean_gated_v2',
        'qwen':BASE/'w8a8-ber-stress/workspace/runs/w8a8_wide_ber_matched_v1/qwen3_0_6b_int8'}
QFOLLOW = BASE/'w8a8-followup/workspace/runs/qwen_crossover_p05_p075_v1'
KFOLLOW = BASE/'kws-transition-validation/workspace/runs/kws_transition_independent_pools_p05_derived_recovery_v1'
PROVENANCE = []

def readj(path):
    return json.loads(Path(path).read_text())

def readcsv(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def verify(path, expected, role):
    actual=digest(path)
    if actual!=expected: raise ValueError(f'Checksum mismatch: {path}')
    PROVENANCE.append({'path':str(path),'sha256':actual,'role':role,'bytes':Path(path).stat().st_size})

def verify_db():
    assert (DB/'FINALIZED').exists()
    for name, expected in readj(DB/'checksums.json')['files'].items():
        verify(DB/name,expected,'finalized consolidated source')

def verify_members(path, names):
    assert (path/'FINALIZED').exists(),path
    manifest=readj(path/'manifest.json')
    files=manifest.get('scientific_file_sha256') or manifest.get('files_sha256') or {x['path']:x['sha256'] for x in manifest['outputs']}
    for name in names:verify(path/name,files[name],'manifest-bound source used in figures')
    PROVENANCE.append({'path':str(path/'manifest.json'),'sha256':digest(path/'manifest.json'),
                       'role':'source manifest','finalized_marker':(path/'FINALIZED').read_text().strip()})
    return manifest

def writecsv(path, rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

verify_db()
registry=readcsv(DB/'study_registry.csv')
conditions=[]
metadata=[]
runtime_rows=[]
raw_db=readcsv(DB/'current_conditions.csv')
trials_db=readcsv(DB/'current_trials.csv')
for reg in registry[:3]:
    model=reg['model_key'];src=Path(reg['source_path'])
    man=verify_members(src,['scientific_payload.json','summary.json','measurements/trials.csv',
                            'measurements/protection_results.csv','measurements/estimator_runtime.csv',
                            'artifacts/sample_partitions.json','config.toml'])
    summary=readj(src/'summary.json')
    selected=[r for r in raw_db if r['study_id']==reg['study_id']]
    trialgroups=defaultdict(list)
    for r in trials_db:
        if r['study_id']==reg['study_id']:
            trialgroups[(r['method_id'],float(r['coverage']),float(r['ber']))].append(float(r['mean_loss']))
            assert int(r['proposed_faults'])==int(r['applied_faults'])+int(r['suppressed_faults'])
    for r in selected:
        vals=trialgroups[(r['method_id'],float(r['coverage']),float(r['ber']))]
        assert abs(np.mean(vals)-float(r['mean_loss']))<1e-11
        conditions.append(dict(model=model,method=MAP[r['method_id']],coverage=float(r['coverage']),ber=float(r['ber']),
                               mean=float(r['mean_loss']),low=float(r['loss_ci_low']),high=float(r['loss_ci_high']),
                               clean=float(r['clean_loss']),fields=int(r['field_count']),source=reg['study_id']))
    for r in readcsv(src/'measurements/estimator_runtime.csv'):runtime_rows.append({'model':model,**r})
    b=man['benchmark'];parts=b['sample_partitions']
    metadata.append({'model':model,'name':NAMES[model],'source':str(src),'source_payload_sha256':reg['source_payload_sha256'],
                     'checkpoint_sha256':b['checkpoint_sha256'],'eligible_bits':int(reg['eligible_bits']),
                     'ranking_examples':b['ranking_samples'],'evaluation_examples':b['evaluation_samples'],
                     'evaluation_unit':'complete files' if model=='toycar' else 'examples',
                     'clean_loss':selected[0]['clean_loss'],'clean_evaluation':summary['clean_evaluation'],
                     'fields_per_ber':selected[0]['field_count'],'bers':sorted({float(r['ber']) for r in selected}),
                     'conditions':len(selected),'trial_observations':sum(len(v) for v in trialgroups.values()),
                     'ranking_mask_counts':{r['method_id']:r['ranking_mask_count'] for r in selected},
                     'partitions':parts,'backend':'TensorFlow Lite 2.14.1 BUILTIN_REF',
                     'fkeras_diagnostics':summary['fkeras_diagnostics'],'mask_equivalence':summary['mask_equivalence'],
                     'source_revision':man['source']['commit']})

wide_comparisons=[]
for model,src in WIDE.items():
    man=verify_members(src,['plan.json','derived/conditions.csv','derived/trials.csv','derived/paired_comparisons.csv','derived/analysis.json'])
    p=readj(src/'plan.json');a=readj(src/'derived/analysis.json');rows=readcsv(src/'derived/conditions.csv')
    ts=readcsv(src/'derived/trials.csv');groups=defaultdict(list)
    for r in ts:
        assert int(r['proposed'])==int(r['applied'])+int(r['suppressed'])
        groups[(float(r['ber']),r['policy'])].append(float(r['mean_nll_nats']))
    for r in rows:
        assert abs(np.mean(groups[(float(r['ber']),r['policy'])])-float(r['mean_nll_nats']))<1e-11
        policy=r['policy']
        stem=policy.rsplit('_b',1)[0] if '_b' in policy and policy.rsplit('_b',1)[1].isdigit() else policy
        method=WMAP[stem]
        conditions.append(dict(model=model,method=method,coverage=float(r['coverage']),ber=float(r['ber']),
                               mean=float(r['mean_nll_nats']),low=float(r['loss_ci95_low']),high=float(r['loss_ci95_high']),
                               clean=float(r['clean_nll_nats']),fields=int(r['field_count']),source=str(src.name)))
    original=readj(p['source_plan_path'])
    metadata.append({'model':model,'name':NAMES[model],'source':str(src),'source_payload_sha256':man['payload_sha256'],
                     'checkpoint_sha256':original.get('checkpoint_sha256', original.get('model_sha256')),
                     'eligible_bits':p['eligible_bits'],'ranking_cohort':p['selection_cohort'],'evaluation_cohort':p['final_cohort'],
                     'clean_loss':rows[0]['clean_nll_nats'],'clean_evaluation':a['clean_risk'],
                     'fields_per_ber':p['fields_per_ber'],'bers':p['bers'],'conditions':len(rows),'trial_observations':len(ts),
                     'backend':'Pinned PyTorch CPU W8A8 integer-matmul','source_revision':p['source_commit'],
                     'source_plan_path':p['source_plan_path'],'counts':p['counts'],'note':p['exposure_note']})
    for r in readcsv(src/'derived/paired_comparisons.csv'):wide_comparisons.append({'model':model,**r})

writecsv(DATA/'reliability_conditions.csv',conditions)
writecsv(DATA/'wide_paired_comparisons.csv',wide_comparisons)
writecsv(DATA/'estimator_runtime.csv',runtime_rows)
(DATA/'benchmark_inventory.json').write_text(json.dumps(metadata,indent=2)+'\n')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':10,'axes.labelsize':8,
                     'xtick.labelsize':7,'ytick.labelsize':7,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.edgecolor':'#B0B7C0','axes.labelcolor':'#26313D','text.color':'#26313D',
                     'xtick.color':'#465363','ytick.color':'#465363','grid.color':'#DEE3E9','grid.linewidth':.5,
                     'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})

def style(ax):
    ax.grid(axis='y',alpha=.7,zorder=0)
    ax.tick_params(length=2.5,pad=3)

def legend(fig,y):
    handles=[Line2D([0],[0],color=COLORS[m],lw=1.7,marker='o',ms=3,ls='--' if m in ['fkeras','unprotected'] else '-') for m in ORDER]
    handles.append(Line2D([0],[0],color='#465363',lw=1,ls=':'))
    fig.legend(handles,[LABELS[m] for m in ORDER]+['Clean / full protection'],loc='upper center',bbox_to_anchor=(.52,y),ncol=5,frameon=False,fontsize=7.2,columnspacing=1.5,handlelength=2)

def ber_axis(ax,model,cov,letter):
    rs=[r for r in conditions if r['model']==model]
    bounds=[]
    for method in ORDER:
        if method=='fkeras' and cov in [.125,.25]:continue
        xs=sorted([r for r in rs if r['method']==method and (r['coverage']==cov or method=='unprotected')],key=lambda r:r['ber'])
        if not xs:continue
        x=np.array([r['ber'] for r in xs]);y=np.array([r['mean']-r['clean'] for r in xs]);lo=np.array([r['low']-r['clean'] for r in xs]);hi=np.array([r['high']-r['clean'] for r in xs])
        bounds.extend(lo.tolist()+hi.tolist())
        ax.fill_between(x,lo,hi,color=COLORS[method],alpha=.065,lw=0)
        ax.plot(x,y,color=COLORS[method],lw=1.25,marker='o',ms=2.5,ls='--' if method in ['fkeras','unprotected'] else '-',label=LABELS[method],zorder=3)
    ax.axhline(0,color='#465363',ls=':',lw=.9)
    ax.set_xscale('log');ax.set_xlim(8e-6,.13)
    ax.set_xticks([1e-5,1e-3,1e-1],['$10^{-5}$','$10^{-3}$','$10^{-1}$'])
    ax.set_title(f'{letter}  {NAMES[model]}',loc='left',fontweight='semibold',pad=9)
    ax.set_ylabel(UNIT[model]);ax.set_xlabel('Logical BER')
    if model in ['resnet8','kws','distilbert']:
        ax.set_yscale('asinh',linear_width=.1)
        low=min(0,min(bounds))-.01;high=max(bounds)+max(.01,.04*(max(bounds)-min(bounds)))
        ticks=[t for t in [-1,-.1,0,.1,1,10,100] if low<=t<=high]
        ax.set_yticks(ticks,[f'{t:g}' for t in ticks]);ax.set_ylim(low,high)
        ax.text(.02,.97,'asinh scale: linear near zero',transform=ax.transAxes,fontsize=5.7,color='#6C7785',va='top')
    if model=='toycar':
        ax.axvspan(.0105,.13,color='#F0F2F5',zorder=-1)
        ax.text(.98,.48,'No ToyCar\nmeasurements\nabove 1%',ha='right',transform=ax.transAxes,fontsize=6.5,color='#6C7785')
    style(ax)

def save(fig,name):
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(OUT/(name+'.svg'),bbox_inches='tight')
    fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight')
    plt.close(fig)

for cov,slug in [(.125,'125'),(.25,'250')]:
    fig,axs=plt.subplots(3,2,figsize=(7.35,7.0))
    for i,model in enumerate(MODELS):ber_axis(axs.flat[i],model,cov,chr(97+i))
    fig.suptitle(f'How does protection change loss as BER increases?\n{cov*100:g}% of eligible stored bits protected',x=.08,ha='left',fontsize=12,fontweight='semibold',y=.995)
    legend(fig,.945)
    fig.subplots_adjust(top=.81,bottom=.12,hspace=.65,wspace=.36)
    fig.text(.08,.015,'Bands: source 95% field intervals; 32 fields/BER for the first three models, 24 for the transformers.\nFull protection reproduces clean inference. FKeras = MSB at these complete-tier budgets on the three TFLite models.',fontsize=6.5)
    save(fig,'reliability_ber_'+slug)

fig,axs=plt.subplots(2,3,figsize=(7.35,4.9))
for i,model in enumerate(MODELS):
    ax=axs.flat[i];rs=[r for r in conditions if r['model']==model and r['ber']==.01]
    for method in ORDER[:-1]:
        xs=sorted([r for r in rs if r['method']==method],key=lambda r:r['coverage'])
        if not xs:continue
        x=np.array([100*r['coverage'] for r in xs]);y=np.array([r['mean']-r['clean'] for r in xs]);lo=np.array([r['low']-r['clean'] for r in xs]);hi=np.array([r['high']-r['clean'] for r in xs])
        ax.errorbar(x,y,yerr=[y-lo,hi-y],color=COLORS[method],marker='o',ms=3,lw=1.25,elinewidth=.6,capsize=2,ls='--' if method=='fkeras' else '-')
    unp=next(r for r in rs if r['method']=='unprotected')
    ax.axhline(unp['mean']-unp['clean'],color=COLORS['unprotected'],ls='--',lw=1)
    ax.axhline(0,color='#465363',ls=':',lw=.9)
    ax.set_xticks([6.25,12.5,18.75,25],['6.25','12.5','18.75','25'])
    ax.set_xlabel('Protected eligible bits (%)');ax.set_ylabel(UNIT[model])
    ax.set_title(chr(97+i)+'  '+NAMES[model].split(' / ')[0],loc='left',fontweight='semibold')
    style(ax)
fig.suptitle('Does more protection preserve the same method order?\nAll panels: logical BER = 1%',x=.08,ha='left',fontsize=12,fontweight='semibold',y=1)
legend(fig,.93);fig.subplots_adjust(top=.72,bottom=.12,hspace=.62,wspace=.48)
fig.text(.08,.01,'Whiskers: source 95% field intervals. Gray dashed: unprotected. Dotted: clean / full protection.\nFKeras partial-tier allocations are measured on ResNet-8, DS-CNN, and ToyCar; unavailable on the three W8A8 transformers.',fontsize=6.5)
save(fig,'reliability_budget_001')

# Full sweep retained without averaging incompatible model risks.
with PdfPages(OUT/'reliability_all_budgets_supplement.pdf') as pdf:
    for model in MODELS:
        fig,axs=plt.subplots(2,2,figsize=(7.35,5.9))
        for i,cov in enumerate([.0625,.125,.1875,.25]):
            ber_axis(axs.flat[i],model,cov,chr(97+i));axs.flat[i].set_title(f'{cov*100:g}% protected',loc='left',fontweight='semibold')
        fig.suptitle(NAMES[model]+' — every measured BER and interior budget',x=.08,ha='left',fontsize=12,fontweight='semibold')
        legend(fig,.94);fig.subplots_adjust(top=.77,bottom=.12,hspace=.55,wspace=.36)
        fig.text(.08,.02,'Source 95% intervals; fixed checkpoint and evaluation cohort. No interpolation across missing BER cells.\nLines connect measured points for visual guidance. Logical BER is not a calibrated radiation exposure.',fontsize=6.5)
        pdf.savefig(fig,bbox_inches='tight');plt.close(fig)

verify_members(QFOLLOW,['derived/analysis.json','derived/trials.csv','derived/paired.csv'])
qanalysis=readj(QFOLLOW/'derived/analysis.json')
for name,h in readj(KFOLLOW/'checksums.json')['files'].items():
    # Derived analysis uses a plain file->hash map.
    if isinstance(h,str) and len(h)==64 and (KFOLLOW/name).is_file():verify(KFOLLOW/name,h,'finalized recovered KWS aggregate')
kws=readj(KFOLLOW/'scientific_payload.json')
(DATA/'qwen_crossover.json').write_text(json.dumps(qanalysis,indent=2)+'\n')
(DATA/'kws_pool_statistics.json').write_text(json.dumps(kws['statistics'],indent=2)+'\n')

fig,axs=plt.subplots(1,3,figsize=(7.35,3.0),gridspec_kw={'width_ratios':[1,1,1.3]})
p=[r for r in wide_comparisons if r['model']=='pythia' and float(r['ber'])==.01 and r['compared_method']=='global_first_order']
for r in p:
    x=float(r['coverage'])*100;y=float(r['mean_nll_difference_nats']);lo=float(r['ci95_low']);hi=float(r['ci95_high'])
    axs[0].errorbar(x,y,yerr=[[y-lo],[hi-y]],fmt='o',color=COLORS['global'],capsize=3,ms=5,lw=1.2)
axs[0].set_xticks([6.25,12.5,18.75,25],['6.25','12.5','18.75','25']);axs[0].set_xlabel('Protected bits (%)')
axs[0].set_ylabel('Global gradient − MSB\n(nats/token)');axs[0].set_title('a  Pythia: budget reversal\nBER = 1%',loc='left',fontsize=9,fontweight='semibold')
for r in qanalysis['conditions_and_contrasts']:
    if 'contrast' not in r:continue
    y=r['mean_difference_nats'];axs[1].errorbar(r['ber']*100,y,yerr=[[y-r['ci95_low']],[r['ci95_high']-y]],fmt='o',color=COLORS['absolute'],capsize=3,ms=5,lw=1.2)
axs[1].set_xticks([5,7.5]);axs[1].set_xlim(4,8.5);axs[1].set_xlabel('Logical BER (%)');axs[1].set_ylabel('Absolute gradient − MSB\n(nats/token)');axs[1].set_title('b  Qwen: BER reversal\n12.5% protection',loc='left',fontsize=9,fontweight='semibold')
ks=next(r for r in kws['statistics']['comparisons'] if r['role']=='primary');vals=np.asarray(ks['pool_effects'])*1000
rank=np.argsort(vals,kind='stable');jitter=np.zeros(len(vals))
for k,ix in enumerate(rank):jitter[ix]=.17*np.sin(k*2.4)
axs[2].scatter(jitter,vals,s=18,c=np.where(vals<0,COLORS['absolute'],COLORS['positive']),alpha=.8,edgecolor='white',lw=.35)
me=ks['pool_mean_effect'];mean=me['mean']*1000;lo,hi=np.array(me['confidence_interval'])*1000
axs[2].errorbar(.55,mean,yerr=[[mean-lo],[hi-mean]],fmt='D',color='#26313D',ms=5,capsize=4,lw=1.5)
axs[2].set_xlim(-.35,.85);axs[2].set_xticks([0,.55],['32 pools','Mean + 95% CI']);axs[2].set_ylabel('Conditional − exact isolated\n($10^{-3}$ nats/clip)');axs[2].set_title('c  KWS: decisions change,\nheld-out gain unresolved',loc='left',fontsize=9,fontweight='semibold')
for ax in axs:ax.axhline(0,color='#7F8995',ls='--',lw=.9);style(ax)
fig.subplots_adjust(left=.08,right=.99,bottom=.22,top=.8,wspace=.62)
fig.text(.08,.015,'Negative favors the first method. Pythia: 24 paired fields/point; Holm family retained. Qwen: 24 fields/point, exploratory.\nKWS: 32 independent candidate pools, 32 final fields/pool; primary pool-level t interval. Different units are not pooled.',fontsize=6.3)
save(fig,'paired_reversals_and_kws')

(DATA/'figure_provenance.json').write_text(json.dumps({'schema':'paper1-figure-provenance/1.0','source_files':PROVENANCE,
    'figures':{'reliability_ber_125':'All measured methods, six models, 12.5% budget; 95% marginal field intervals.',
               'reliability_ber_250':'All measured methods, six models, 25% budget; 95% marginal field intervals.',
               'reliability_budget_001':'All six models at measured 1% BER and all four interior budgets.',
               'paired_reversals_and_kws':'Illustrative paired reversals plus all 32 KWS pool effects; full grid remains supplied.',
               'reliability_all_budgets_supplement':'All measured BERs and all four interior budgets on one page per model.'},
    'limits':['Large mask/prediction binaries are linked through source manifests, not independently rehashed during this figure build.',
              'Saved trials were used to reconstruct every plotted mean; source uncertainty definitions were retained.',
              'W8A8 wide grids are exploratory; fixed checkpoint, cohort, and mask. No across-checkpoint inference.',
              'Language model token NLL is capped at 30 nats, including finite values; failed-token counts do not measure finite clipping.',
              'KWS/ResNet/ToyCar FKeras is published form. PrioriFI has engineering evidence only and is not shown as measured protection.',
              'No source bundle, database, mask, fault schedule, or model was changed.']},indent=2)+'\n')
print('Generated',len(conditions),'conditions and',len(PROVENANCE),'verified source-file receipts into',ROOT)
