"""Manuscript-only derivations from finalized sources; no ML imports or inference.

Creates a new evidence directory. Existing scientific bundles are read-only.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys
import argparse

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from workflows.paper1_evidence_audit import read, sha, verify, audit_static, estimate, write_json, write_csv

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LOCATORS = Path('/projects/netsurf/manuelbv/publications/tecs_internal_review_2026-11-02/evidence/sources.json')
OUT = HERE / 'evidence'
FIG = HERE / 'overleaf/figures'
TAB = HERE / 'overleaf/tables'
LABELS = ['MSB-first', 'Numerical jump', 'Global first-order', 'Per-example absolute', 'Per-example positive', 'Random']
POLICIES = ['msb_global', 'numerical_jump_global', 'gradient_global', 'per_example_absolute_global', 'per_example_positive_global', 'random_global']
COLORS = ['#222222', '#D89000', '#0072B2', '#009E73', '#CC79A7', '#777777']
MODELS = ['resnet', 'kws', 'toycar']
TITLES = ['ResNet-8 / CIFAR-10', 'DS-CNN / Speech Commands', 'ToyCar autoencoder']
products = []

def table(name, values, sources, recipe):
    (TAB / (name+'.tex')).write_text('\n'.join(' & '.join(map(str,row))+r' \\' for row in values)+'\n')
    products.append(dict(artifact='overleaf/tables/'+name+'.tex', sources=sources, recipe=recipe))

def save(fig, name, sources, recipe):
    for extension in ['pdf','png']:
        fig.savefig(FIG/(name+'.'+extension), dpi=400)
    plt.close(fig)
    products.append(dict(artifact='overleaf/figures/'+name+'.png', vector_alternative='overleaf/figures/'+name+'.pdf', sources=sources, recipe=recipe))

def intervals(ax, x, values, color, label, fmt='o--'):
    y=[r['mean'] for r in values]
    ax.errorbar(x,y,yerr=[[a-max(0,r['ci_low']) for a,r in zip(y,values)],
                         [min(1,r['ci_high'])-a for a,r in zip(y,values)]],
                fmt=fmt,color=color,markersize=3,linewidth=.85,capsize=2,label=label)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh-draft', action='store_true',
                        help='Reverify all sources and replace only this unfinalized manuscript derivation.')
    args = parser.parse_args()
    OUT.mkdir(exist_ok=args.refresh_draft)
    FIG.mkdir(parents=True,exist_ok=True); TAB.mkdir(parents=True,exist_ok=True)
    sources=read(LOCATORS); receipts={}; curves=[]; pairs=[]; inventory={}
    fk_source=read(ROOT/'config/paper1_fkeras_source.json')
    # Check EVERY saved artifact in the newly recovered experiment, not only plots.
    fk_source['verified_files']=read(Path(fk_source['path'])/'checksums.json')['files']
    sources['static_fkeras128']=fk_source
    for tag, source in sources.items():
        receipts[tag], m=verify(tag,source)
        if tag.startswith('static_'):
            c,p,i=audit_static(tag,Path(source['path']),m)
            curves+=c; pairs+=p; inventory[tag]=i
    write_json(OUT/'sources.json',receipts)
    write_json(OUT/'static_inventory.json',inventory)
    write_csv(OUT/'static_curves.csv',curves)
    write_csv(OUT/'paired_comparisons.csv',pairs)
    plt.rcParams.update({'font.size':8.5,'axes.titlesize':9,'pdf.fonttype':42})
    for mode in ['budget','ber']:
        fig,axs=plt.subplots(3,2,figsize=(6.65,7.4))
        for row,(model,title) in enumerate(zip(MODELS,TITLES)):
            data=[r for r in curves if r['benchmark']==model]
            for col,condition in enumerate([.001,.01] if mode=='budget' else [.125,.25]):
                ax=axs[row,col]
                for policy,label,color in zip(POLICIES,LABELS,COLORS):
                    selected=[r for r in data if r['policy']==policy and r['ber' if mode=='budget' else 'coverage']==condition]
                    if mode=='budget': selected += [r for r in data if r['policy']=='no_protection' and r['ber']==condition]
                    selected.sort(key=lambda r:r['coverage' if mode=='budget' else 'ber'])
                    x=[r['coverage']*100 if mode=='budget' else r['ber'] for r in selected]
                    intervals(ax,x,selected,color,label)
                clean=next(r['mean'] for r in data if r['policy']=='full_protection')
                ax.axhline(clean,color='#555555',ls='-.',lw=.8,label='Clean / full protection')
                if mode=='ber':
                    unprotected=sorted([r for r in data if r['policy']=='no_protection'],key=lambda r:r['ber'])
                    intervals(ax,[r['ber'] for r in unprotected],unprotected,'#D55E00','Unprotected','x:')
                    ax.set_xscale('log');ax.set_xlabel('Logical bit-flip probability')
                    ax.set_title(f'{title}\n{100*condition:g}% protected')
                    if model!='toycar':ax.axvspan(.02,.12,color='gray',alpha=.08)
                else:
                    ax.set_xlim(-1,26);ax.set_xticks([0,12.5,25]);ax.set_xlabel('Protected eligible bits (%)')
                    ax.set_title(f'{title}\nLogical BER {condition:g}')
                ax.set_ylabel('File ROC-AUC' if model=='toycar' else 'Accuracy')
                ax.set_ylim(0,1.02);ax.grid(axis='y',alpha=.18)
        h,l=axs[0,0].get_legend_handles_labels()
        fig.legend(h,l,loc='lower center',ncol=3,fontsize=8,frameon=False)
        fig.subplots_adjust(left=.10,right=.98,top=.95,bottom=.18,hspace=.74,wspace=.36)
        save(fig,'protection_'+mode,[f'static_{m}' for m in MODELS],
             'All six policies; raw pooled field means and descriptive 95% t intervals intersected with [0,1]. Fixed cohort, realized masks; guides join observed points only.')

    rank=[]; times=[]; sample_info={}
    for model in MODELS:
        s=read(Path(sources['emp_'+model]['path'])/'summary.json')
        m=read(Path(sources['emp_'+model]['path'])/'manifest.json')
        sample_info[model]=m['benchmark']
        metrics=s['ranking_metrics_overall']
        for prefix,label in zip(['s2_','s3_','s4_','s5_','s6_','random_'],LABELS):
            r=next(x for x in metrics if x['estimator'].startswith(prefix))
            rank.append(dict(model=model,method=label,**r))
        times.append([TITLES[MODELS.index(model)].split(' /')[0]]+[f"{next(r['runtime_seconds'] for r in metrics if r['estimator'].startswith(prefix)):.2f}" for prefix in ['s3_','s4_','s5_']])
    write_json(OUT/'empirical_metrics.json',rank);write_json(OUT/'empirical_partitions.json',sample_info)
    fig,axs=plt.subplots(1,3,figsize=(6.65,2.8),sharey=True)
    for ax,model,title in zip(axs,MODELS,TITLES):
        for i,(label,color) in enumerate(zip(LABELS,COLORS)):
            r=next(x for x in rank if x['model']==model and x['method']==label)
            y=r['spearman_positive_damage'];lo=r['spearman_positive_damage_ci_low'];hi=r['spearman_positive_damage_ci_high']
            ax.errorbar(y,5-i,xerr=[[y-lo],[hi-y]],fmt='o',color=color,capsize=2,markersize=3)
        ax.set_xlim(-.18,.8);ax.axvline(0,color='gray',lw=.6);ax.grid(axis='x',alpha=.2)
        ax.set_title(title.split(' /')[0]+'\n'+('MSE damage' if model=='toycar' else 'Cross-entropy damage'))
        ax.set_xlabel('Spearman correlation')
    axs[0].set_yticks(range(6),LABELS[::-1]);fig.subplots_adjust(left=.24,right=.99,bottom=.20,top=.80,wspace=.15)
    save(fig,'estimator_validity',['emp_'+m for m in MODELS],'Saved balanced-stratum positive-damage Spearman correlations and within-stratum bootstrap intervals; not population-wide bit estimates.')
    table('ranking_costs',times,['emp_'+m for m in MODELS],'Saved construction seconds: numerical jump, global gradient, shared per-example gradients; do not sum shared rows.')

    # Matched illustrative stress contrast: every architecture and both budgets.
    effects=[];tex=[]
    for model in MODELS:
        for coverage in [.125,.25]:
            metric='roc_auc' if model=='toycar' else 'accuracy'
            r=next(x for x in pairs if x['benchmark']==model and float(x['logical_ber'])==.01 and float(x['coverage'])==coverage and x['metric']==metric and {x['left'],x['right']}=={'msb_global','per_example_absolute_global'})
            sign=1 if r['left']=='per_example_absolute_global' else -1
            mean=sign*float(r['mean_difference']);lo,hi=sorted([sign*float(r['ci_low']),sign*float(r['ci_high'])])
            effects.append(dict(model=model,coverage=coverage,mean=mean,ci_low=lo,ci_high=hi,source_row=r))
            scale=1 if model=='toycar' else 100
            tex.append([{'resnet':'ResNet-8 (pp)','kws':'KWS (pp)','toycar':'ToyCar (AUC)'}[model],f'{coverage*100:g}\\%',r['matched_pairs'],f'{mean*scale:+.3f}',f'[{lo*scale:+.3f}, {hi*scale:+.3f}]',f"{float(r['p_value_holm']):.4f}"])
    write_json(OUT/'illustrative_paired_effects.json',effects)
    table('paired_effects',tex,['static_'+m for m in MODELS],'BER .01; per-example absolute minus MSB, reversing saved contrasts if required. Source within-BER/metric Holm values retained.')
    f=read(Path(sources['static_fkeras128']['path'])/'summary.json'); rows=[]
    for cov in [.0625,.125,.1875,.25]:
        counts=[]
        for a,b in [('fkeras_paper','fkeras_repository'),('msb_first','fkeras_paper'),('msb_first','fkeras_repository')]:
            item=next(x for x in f['mask_equivalence'] if x['left']==f'{a}@{cov:.6f}' and x['right']==f'{b}@{cov:.6f}')
            counts.append(item['symmetric_difference_bits'])
        rows.append([f'{cov*100:g}\\%',item['protected_bits']]+counts)
    table('fkeras_masks',rows,['static_fkeras128/summary.json'],'Saved exact mask symmetric differences from the 128-clip exploratory comparison; one ranking realization.')
    write_json(OUT/'fkeras_compact.json',{'mask_equivalence':f['mask_equivalence'],'runtimes':f['estimator_runtimes_seconds'],'clean':f['clean_evaluation']})
    methods=['msb_first','first_order_gradient_delta','per_sample_first_order_magnitude','per_sample_positive_damage','fkeras_paper','fkeras_repository']
    labels=['MSB-first','Global gradient','Absolute gradient','Positive gradient','FKeras paper','FKeras repository']
    colors=['#222222','#0072B2','#009E73','#CC79A7','#D55E00','#6A51A3']
    data=[r for r in curves if r['benchmark']=='fkeras128']
    fig,axs=plt.subplots(2,3,figsize=(6.65,5.1))
    for col,prob in enumerate([.0001,.001,.01]):
        for policy,label,color in zip(methods,labels,colors):
            selected=sorted([r for r in data if r['ber']==prob and r['policy'] in (policy,'no_protection')],key=lambda r:r['coverage'])
            x=[r['coverage']*100 for r in selected]
            intervals(axs[0,col],x,selected,color,label)
            axs[1,col].plot(x,[r['loss_mean'] for r in selected],'o--',color=color,lw=.85,ms=3,label=label)
        axs[0,col].axhline(f['clean_evaluation']['accuracy'],color='gray',ls='-.',lw=.7)
        axs[0,col].set_ylim(0,1.02)
        for ax in axs[:,col]:
            ax.set_xticks([0,6.25,12.5,18.75,25])
            ax.tick_params(axis='x',labelsize=7)
        axs[0,col].set_title(f'Logical BER {prob:g}')
        for ax in axs[:,col]:
            ax.set_xlabel('Protected eligible bits (%)');ax.grid(alpha=.15)
    axs[0,0].set_ylabel('Accuracy');axs[1,0].set_ylabel('Cross-entropy')
    h,l=axs[0,0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=3,fontsize=8,frameon=False)
    fig.subplots_adjust(left=.10,right=.99,bottom=.20,top=.93,hspace=.42,wspace=.33)
    save(fig,'matched_kws128',['static_fkeras128'],'Separate 128-clip cohort; all six configured policies, three BERs, four budgets and shared zero/full controls. Accuracy bars are marginal 95% field-t intervals. CE means; full paired intervals in the CSV supplement. No merging with the 1000-clip study.')

    p=Path(sources['M910']['path']); ag=read(p/'aggregate_summary.json')['comparisons']
    arms=[('m96_kws_cross_pool_k6','KWS: first pool sample, p=.01'),('m97_kws_multiber_ber_0p001','KWS: common pools, p=.001'),('m97_kws_multiber_ber_0p003','KWS: common pools, p=.003'),('m97_kws_multiber_ber_0p01','KWS: common pools, p=.01'),('m98_resnet8_cross_pool_k6_v1','ResNet-8, p=.01'),('m99_toycar_cross_pool_k6_ber001','ToyCar, p=.001')]
    fig,axs=plt.subplots(1,2,figsize=(6.65,3.0),gridspec_kw={'width_ratios':[2.0,1.]})
    selected=[]
    for i,(arm,label) in enumerate(arms):
        ax=axs[0 if i<5 else 1];scale=1e4 if i<5 else 1e6;y=4-i if i<5 else 0
        for j,(comparison,color,method) in enumerate([('restricted_selected_vs_original_s5','#0072B2','Versus per-example absolute'),('restricted_selected_vs_exact_isolated','#D55E00','Versus exact isolated')]):
            r=next(x for x in ag if x['campaign']==arm and x['comparison']==comparison);s=r['pool_summary'];lo,hi=s['confidence_interval'];mean=s['mean']
            ax.errorbar(mean*scale,y+(j-.5)*.18,xerr=[[(mean-lo)*scale],[(hi-mean)*scale]],fmt='o',markersize=3,capsize=2,color=color,label=method if i==0 else None)
            selected.append(dict(label=label,**r))
    axs[0].set_yticks(range(5),[x[1] for x in arms[:5]][::-1]);axs[1].set_yticks([0],['ToyCar\np=.001'])
    for ax in axs:ax.axvline(0,color='gray',lw=.7);ax.grid(axis='x',alpha=.2)
    axs[0].set_xlabel('Risk difference (CE × $10^{-4}$)');axs[1].set_xlabel('Risk difference\n((1−AUC) × $10^{-6}$)', fontsize=8)
    axs[0].set_title('12 pools per KWS arm; 4 ResNet pools', fontsize=8);axs[1].set_title('ToyCar: 4 pools', fontsize=8)
    axs[1].set_ylim(-.4,.4)
    h,l=axs[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=2,fontsize=7.5,frameon=False)
    fig.subplots_adjust(left=.30,right=.97,top=.80,bottom=.30,wspace=.65)
    save(fig,'conditional_limits',['M910/aggregate_summary.json'],'All six arms, equal-pool mean and marginal t intervals, selected minus frozen comparator. Negative favors selected; metrics on separate axes.')
    write_json(OUT/'conditional_effects.json',selected)

    # Supplementary loss trajectories: raw trials, never reinterpret MSE as AUC.
    fig,axs=plt.subplots(3,2,figsize=(6.65,7.4)); lossrows=[]
    for row,(model,title) in enumerate(zip(MODELS,TITLES)):
        with (Path(sources['static_'+model]['path'])/'measurements/trials.csv').open() as stream:trials=list(csv.DictReader(stream))
        key='loss_metric_value' if model=='toycar' else 'categorical_crossentropy'
        for col,cov in enumerate([.125,.25]):
            ax=axs[row,col]
            for policy,label,color in zip(POLICIES+['no_protection'],LABELS+['Unprotected'],COLORS+['#D55E00']):
                probabilities=sorted(set(r['ber'] for r in curves if r['benchmark']==model))
                points=[]
                for prob in probabilities:
                    condition=policy if policy=='no_protection' else f'{policy}@{cov:.6f}'
                    sample=[float(r[key]) for r in trials if r['policy'].split('/',1)[1]==condition and float(r['policy'].split('/',1)[0][4:])==prob]
                    e=estimate(sample);points.append(e['mean']);lossrows.append(dict(model=model,coverage=cov,policy=policy,ber=prob,**e))
                ax.plot(probabilities,points,'o--',color=color,markersize=3,lw=.85,label=label)
            ax.set_xscale('log');ax.set_title(f'{title}\n{cov*100:g}% protected');ax.set_ylabel('Reconstruction MSE' if model=='toycar' else 'Cross-entropy');ax.set_xlabel('Logical bit-flip probability');ax.grid(alpha=.15)
    h,l=axs[0,0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=3,fontsize=8,frameon=False)
    fig.subplots_adjust(left=.10,right=.98,top=.95,bottom=.18,hspace=.74,wspace=.36)
    save(fig,'loss_ber',['static_'+m for m in MODELS],'Raw-trial mean auxiliary loss, all BERs; separate task units. Intervals retained in loss_curves.csv; not shown to avoid overplotting.')
    write_csv(OUT/'loss_curves.csv',lossrows)
    write_json(OUT/'figure_table_provenance.json',{'products':products,'sources':'sources.json','new_inferences':0})
    write_json(OUT/'verification_summary.json',{'records':len(receipts),'consumed_files':sum(len(x['verified_files']) for x in receipts.values()),'static_trial_rows':sum(x['trial_rows'] for x in inventory.values()),'condition_rows':len(curves),'paired_summaries':len(pairs),'new_model_inferences':0,'source_files_written':0})
    print('Verified sources and generated manuscript-only figures/tables:',len(products))

if __name__=='__main__':main()
