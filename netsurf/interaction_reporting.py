"""Publication-oriented figures from immutable offline attribution measurements."""
from pathlib import Path
import numpy as np

from .conditional_config import PAIRED_SWAP_COMPARISONS
from .interaction_attribution import mean_interval
from .result_bundle import _write_bytes, _write_json, _write_csv


def _label(c):
    n = c["name"]
    family = "ToyCar" if "toycar" in n else "ResNet-8" if "resnet" in n else "KWS"
    revision = "M9.6" if n.startswith("m96_") else "M9.7" if n.startswith("m97_") else "M9.8" if n.startswith("m98_") else "M9.9"
    return f"{family} {revision}, BER {c['logical_ber']:g}" + (" (engineering)" if c["purpose"] == "engineering" else "")


def write_report(payload, output, *, plots):
    output=Path(output)
    lines=["# Cross-architecture interaction attribution and decision detectability", "",
           "Offline, exploratory reanalysis of finalized exact TFLite responses. No inference, mask reselection for deployment, or historical result modification.", "",
           "## Exact identity", "",
           "For every outside field and legal fixed-budget mask, the target-minus-baseline risk contrast equals **clean isolated prediction + outside-background correction + within-candidate nonadditive correction**. The shared background intercept cancels. Signed beneficial effects remain. Every field is checked before averaging.", "",
           "The anchored candidate-subcube Möbius expansion retains all candidate interaction orders; it is not a whole-network spectral model. Attribution of selection-set contrasts does not explain held-out contrasts algebraically: the final subcubes were not measured.", "",
           "## Independent final comparisons", "",
           "Negative target-minus-baseline differences favor the target. Equal-pool means and descriptive pool-t 95% intervals condition on the original shared inputs and selected masks. CE and AUC risk are not pooled or placed on a common magnitude scale. No new significance family is declared; original multiplicity protocols remain in each source campaign.", "",
           "| Campaign | Pools | Contrast | Mean | Pool-t 95% interval | Negative / zero / positive pool estimates |", "|---|---:|---|---:|---|---|"]
    aggregate=[]
    for c in payload["campaigns"]:
        pools=c["pools"]
        for ci,name in enumerate(PAIRED_SWAP_COMPARISONS):
            values=[p["final_comparisons"]["comparisons"][ci]["mean_effect"]["mean"] for p in pools]
            e=mean_interval(values)
            signs=[sum(v<0 for v in values),sum(v==0 for v in values),sum(v>0 for v in values)]
            aggregate.append({"campaign":c["name"],"comparison":name,"risk_metric":c["risk_metric"],"pool_summary":e,"sign_counts":signs})
            lines.append(f"| {_label(c)} | {len(pools)} | {name} | {e['mean']:.8g} | {e['confidence_interval']} | {signs} |")
    _write_json(output/"aggregate_summary.json",{"schema_version":"1.0.0","comparisons":aggregate})
    mechanisms=[]
    for c in payload["campaigns"]:
        ps=c["pools"]
        r={"campaign":c["name"],"risk_metric":c["risk_metric"],"pool_count":len(ps),
           "clean_strict_reversals":sum(p["clean_vs_exact_ordering"]["strict_reversals"] for p in ps),
           "clean_strict_pairs":sum(p["clean_vs_exact_ordering"]["strict_comparable_pairs"] for p in ps),
           "conditioned_strict_reversals":sum(p["conditioned_vs_exact_ordering"]["strict_reversals"] for p in ps),
           "conditioned_strict_pairs":sum(p["conditioned_vs_exact_ordering"]["strict_comparable_pairs"] for p in ps)}
        for key in ("clean_term","outside_correction","within_correction","exact_contrast"):
            r[key+"_mean_absolute_pair_mean"] = float(np.mean([p["all_mask_pair_component_magnitudes"][key]["mean_absolute_field_mean_contrast"] for p in ps]))
        mechanisms.append(r)
    _write_csv(output/"mechanism_summary.csv",list(mechanisms[0]),mechanisms)
    lines += ["", "## Outside conditioning versus within-candidate interaction", "",
              "Strict pair counts are descriptive comparisons among the same masks, not independent binomial samples. The magnitude table in mechanism_summary.csv averages absolute field-mean contrasts, avoiding cancellation across pools/pairs.", "",
              "| Campaign | Clean-additive ordering reversals | Background-conditioned additive reversals | Mean absolute outside correction | Mean absolute within-candidate correction |",
              "|---|---:|---:|---:|---:|"]
    for c,r in zip(payload["campaigns"],mechanisms):
        lines.append(f"| {_label(c)} | {r['clean_strict_reversals']}/{r['clean_strict_pairs']} | {r['conditioned_strict_reversals']}/{r['conditioned_strict_pairs']} | {r['outside_correction_mean_absolute_pair_mean']:.8g} | {r['within_correction_mean_absolute_pair_mean']:.8g} |")
    lines += ["", "## All-pool selection diagnostics", "",
              "Selection regrets and winner margins use the same fields that chose the winner: descriptive, optimistically biased, not held-out gains. Near-ties use a disclosed 1e-12 diagnostic threshold without changing saved scores or masks. Raw exact ties are separately recorded.", "",
              "| Campaign / pool | Isolated empirical regret | Clean→exact reversals / strict pairs | Conditioned→exact reversals / strict pairs | Winner gap / paired SE | Near-best masks | Bootstrap selected near-best fraction |", "|---|---:|---|---|---:|---:|---:|"]
    for c in payload["campaigns"]:
        for p in c["pools"]:
            s=p["selection_stability"];a=p["clean_vs_exact_ordering"];b=p["conditioned_vs_exact_ordering"]
            lines.append(f"| {_label(c)} / {p['pool_id']} | {p['isolated_selection_regret']:.8g} | {a['strict_reversals']}/{a['strict_comparable_pairs']} | {b['strict_reversals']}/{b['strict_comparable_pairs']} | {s['gap_to_paired_field_se']} | {s['near_tied_best_count']} | {s['bootstrap']['selected_near_best_fraction']:.3f} |")
    lines += ["", "## ToyCar complete-file audit", "",
              "The sole task risk is 1−file ROC-AUC. Forced subcube states and swap queries are not natural fault-trial samples. Comparison pairs share files and are never independent replications.", ""]
    for c in payload["campaigns"]:
        for p in c["pools"]:
            a=p["toycar"]
            if not a:continue
            s,f=a["selection"],a["final"]
            changed=a["changed_nonempty_candidate_state_count"];total=a["nonempty_candidate_states"]
            lines += [f"### {_label(c)} / {p['pool_id']}", "",
                f"Clean selection AUC **{s['clean_auc']:.8g}** ({s['normal']} normal, {s['anomalous']} anomalous complete files); clean final AUC **{f['clean_auc']:.8g}** ({f['normal']} normal, {f['anomalous']} anomalous).",
                f"Nonempty candidate states changing AUC relative to the same field's empty candidate state: **{changed}/{total} ({changed/total:.2%})**. Selection background AUC range: {min(a['background_auc']):.6g}–{max(a['background_auc']):.6g}.",
                f"Strict pair-reversal AUC lattice: selection {s['strict_order_auc_increment']:.8g}, final {f['strict_order_auc_increment']:.8g}; ties give half-increments. Expected mask contrasts may be much smaller after Bernoulli weighting. This lattice is not a power threshold.",
                "Saved state-level reversal counts, exact AUCs, complete-file bootstrap sensitivity and mask ties are in the per-pool summary. Zero finite-sample contrasts do not imply absent interactions or true equivalence.", ""]
    lines += ["## Detectability and limitations", "",
        "Whole-field t intervals describe fixed-mask Monte Carlo uncertainty, conditional on the fixed cohort. Field bootstrap winner frequencies describe selection instability across all masks jointly; they are not probabilities of true optimality. Pool dispersion reflects the recorded candidate-pool design, with finite-field noise. No masks, path steps, windows or score pairs inflate sample size.", "",
        "Per-pool final_field_sensitivity.json and selection_stability.field_count_sensitivity extrapolate widths as t×observed-SD/√n. These are variance-held-fixed scenarios, not formal power or guarantees against unseen tails. More selection fields cannot by themselves resolve a dataset shift or a mismatch between model-selection and final populations.", "",
        "Classification input resampling assumes exchangeable clips despite source ordering. ToyCar resamples complete files within class, synchronously across paired states, recomputes AUC and the whole paired path; it never averages per-file AUC losses. Both are exploratory sensitivity analyses conditional on observed fields, not new confirmation samples.", "",
        "Clean and final cohorts differ. Without identical-mask response tables on both populations under matched outside fields, field noise and input-population variation cannot be causally separated. Cross-architecture differences additionally confound checkpoint, task, BER and quantization; they do not identify architectural redundancy.", "",
        "## Next experiments (prespecify before outcomes)", "",
        "- **KWS:** prioritize selected versus exact-isolated performance across the prespecified pools, not another selected-versus-S5 demonstration. Increase selection evidence only if the observed winner margins and field instability warrant a new frozen design; retain independent final schedules.",
        "- **ResNet-8:** use the measured selection margin/SE and adverse field tails to compare stronger selection sampling with a lower-BER pilot. Do not assume mean CE near ln(10) establishes an accuracy floor; saved conditional records do not retain every accuracy outcome. Use already-exposed inputs for regime finding, then freeze a fresh final plan.",
        "- **ToyCar:** distinguish further exploratory reuse of already fault-exposed official files, selection-only mechanism-enriched candidate selection, and genuine independent confirmation on additional unexposed labeled recordings. The completed production selection/final cohorts are now exposed. Enrichment changes the pool-design estimand and must not use final labels/outcomes.",
        "- **Equal-compute unguided random swaps:** prespecify legal swaps drawn uniformly from the same source-status candidate universe, include a no-change control, use the same selection-call budget and exact acceptance measurements, freeze a chosen mask before independent final comparison, and charge all proposal/evaluation costs. No held-out performance for these unevaluated masks is claimed here.", "",
        "## Manuscript placement", "",
        "Methods: exact task-risk set function, conditional decomposition and protected-state integration. Results: every-pool attribution and independent held-out contrasts. Statistical methods: field/pool/file units and selection bias. Discussion: detectability, metric alignment, limits of architecture-level mechanism claims. Appendix: source hashes, all signed pairs, Möbius coefficients and cost sensitivity.", ""]
    if plots:
        captions=make_figures(payload,output)
        for filename,caption in captions:
            lines += [f"![{caption}]({filename})", "", caption, ""]
    lines += ["## Reproduction and provenance", "",
              "Every per-pool source path, scientific digest, response-table hash, cohort identity, candidate list and protection-context identity is retained in scientific_payload.json. Full signed field×mask and field×mask-pair CSVs accompany the report. Source verification is repeated at completion. No source bundle is rewritten.", ""]
    _write_bytes(output/"report.md","\n".join(lines).encode())


def make_figures(payload, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False})
    campaigns=payload["campaigns"];n=len(campaigns)
    captions=[]
    def save(fig,name,caption):
        fig.savefig(output/(name+".png"),dpi=200,bbox_inches="tight")
        fig.savefig(output/(name+".pdf"),bbox_inches="tight",metadata={"CreationDate":None,"ModDate":None})
        plt.close(fig);captions.append((name+".png",caption))
    fig,axes=plt.subplots(n,1,figsize=(10,2.5*n),squeeze=False,layout="constrained")
    for ax,c in zip(axes[:,0],campaigns):
        for offset,key,label,color in [(-.2,"clean_term","Clean isolated","#0072B2"),(0,"outside_correction","Outside correction","#D55E00"),(.2,"within_correction","Candidate interaction","#009E73")]:
            y=[p["selection_attribution"][PAIRED_SWAP_COMPARISONS[1]][key]["mean"] for p in c["pools"]]
            ax.bar(np.arange(len(y))+offset,y,width=.2,label=label,color=color)
        y=[p["selection_attribution"][PAIRED_SWAP_COMPARISONS[1]]["exact_contrast"]["mean"] for p in c["pools"]]
        ax.plot(range(len(y)),y,"k_",ms=12,label="Exact sum")
        ax.axhline(0,color="gray",lw=.6);ax.set_title(_label(c));ax.set_ylabel("Δ AUC risk" if "roc_auc" in c["risk_metric"] else "Δ CE (nats/input)")
        ax.set_xlabel("Prespecified pool");ax.ticklabel_format(axis="y",style="sci",scilimits=(-3,3));ax.legend(ncol=4,fontsize=8)
        ax.set_xticks(range(len(c["pools"])))
    save(fig,"selection_attribution","Figure 1. Signed decomposition of selected-minus-isolated model-selection contrasts, all pools. Black marks equal the three signed components. Selection-set quantities are not held-out effects; panels have task-specific scales.")
    fig,axes=plt.subplots(n,1,figsize=(10,2.6*n),squeeze=False,layout="constrained")
    for ax,c in zip(axes[:,0],campaigns):
        for ci,color,label,offset in [(0,"#0072B2","Selected − S5",-.2),(1,"#D55E00","Selected − isolated",0),(2,"#009E73","Isolated − S5",.2)]:
            estimates=[p["final_comparisons"]["comparisons"][ci]["mean_effect"] for p in c["pools"]]
            y=np.array([e["mean"] for e in estimates]);ciarr=np.array([e["confidence_interval"] for e in estimates]).T
            ax.errorbar(np.arange(len(y))+offset,y,yerr=np.maximum(0,np.array([y-ciarr[0],ciarr[1]-y])),fmt="o",ms=3,capsize=2,color=color,label=label)
        ax.axhline(0,color="gray",lw=.6);ax.set_title(_label(c));ax.set_xlabel("Prespecified pool")
        ax.set_ylabel("Δ AUC risk" if "roc_auc" in c["risk_metric"] else "Δ CE (nats/input)");ax.legend(ncol=3,fontsize=8)
        ax.ticklabel_format(axis="y",style="sci",scilimits=(-3,3))
        ax.set_xticks(range(len(c["pools"])))
    save(fig,"heldout_pool_effects","Figure 2. Independently evaluated frozen-mask contrasts with marginal whole-field 95% t intervals. All pools, including adverse and zero estimates, are retained. These are not multiplicity-adjusted significance declarations or input-population intervals.")
    fig,axes=plt.subplots(n,1,figsize=(10,2.2*n),squeeze=False,layout="constrained")
    for ax,c in zip(axes[:,0],campaigns):
        for p in c["pools"]:
            prefix=p["selection_stability"]["prefix"]
            ax.plot([r["fields"] for r in prefix],[r["margin"] for r in prefix],alpha=.65,lw=1,label=p["pool_id"])
        ax.set_yscale("symlog",linthresh=1e-12);ax.set_title(_label(c));ax.set_xlabel("Model-selection outside fields included")
        ax.set_ylabel("Best–runner-up gap\n(symlog, risk units)");ax.legend(ncol=6,fontsize=6)
    save(fig,"selection_margin_stability","Figure 3. Cumulative winner–runner-up gaps for every pool in recorded field order. Symlog retains zeros and tiny gaps. A positive empirical gap is not evidence of a stable or truly optimal decision; winner switches and paired gap uncertainty are reported numerically.")
    toy=[c for c in campaigns if "roc_auc" in c["risk_metric"]]
    if toy:
        fig,axes=plt.subplots(len(toy),2,figsize=(11,3*len(toy)),squeeze=False,layout="constrained")
        for axs,c in zip(axes,toy):
            for p in c["pools"]:
                audit=p["toycar"];vals=np.sort(audit["background_auc"])
                axs[0].step(vals,np.arange(1,len(vals)+1)/len(vals),where="post",label=p["pool_id"])
            first=c["pools"][0]["toycar"]
            axs[0].axvline(first["selection"]["clean_auc"],color="black",ls="--",label="Clean selection")
            axs[0].set(xlabel="Selection outside-field cohort ROC-AUC",ylabel="Empirical CDF",xlim=(0,1),title=_label(c))
            vals=[p["toycar"]["changed_nonempty_candidate_state_count"]/p["toycar"]["nonempty_candidate_states"] for p in c["pools"]]
            axs[1].bar(range(len(vals)),vals,color="#0072B2");axs[1].set(xlabel="Pool",ylabel="Fraction of nonempty forced candidate states\nchanging AUC versus same-field empty state",ylim=(0,1))
            axs[1].set_xticks(range(len(vals)))
            axs[0].legend(fontsize=7)
        save(fig,"toycar_resolution","Figure 4. ToyCar fixed-cohort resolution and degradation: outside-field AUC distribution (left) and fraction of forced candidate states changing the same-field AUC (right). Forced states are not equally probable natural faults. Zero mask effects do not imply zero state interactions; comparison pairs are dependent.")
    return captions
