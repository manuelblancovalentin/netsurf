"""Saved-data summaries, reports, and plots for schema-4 search studies."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .analysis_m3 import spearman
from .statistics import summarize


def summarize_conditional_comparisons(rows: Sequence[Mapping[str, Any]], confidence: float = 0.95):
    """Pool independent whole-path context contributions by BER and strategy."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[(float(row["logical_ber"]), row.get("protection_budget_bits"), str(row["target"]))].append(row)
    result = []
    for (logical_ber, budget, target), values in sorted(grouped.items()):
        conditional_loss = summarize([float(value["loss_difference"]) for value in values], confidence).to_dict()
        conditional_zero_one = summarize([float(value["zero_one_risk_difference"]) for value in values], confidence).to_dict()
        direct_loss = summarize([float(value["direct_loss_difference"]) for value in values], confidence).to_dict()
        direct_zero_one = summarize([float(value["direct_zero_one_risk_difference"]) for value in values], confidence).to_dict()
        result.append({
            "logical_ber": logical_ber, **({"protection_budget_bits": int(budget)} if budget is not None else {}), "target": target,
            "background_context_count": len(values),
            "replication_count": len({str(value["replication"]) for value in values}),
            "input_sample_count": int(values[0]["input_sample_count"]),
            "swap_count_min": min(int(value["swap_count"]) for value in values),
            "swap_count_max": max(int(value["swap_count"]) for value in values),
            "conditional_loss_difference": conditional_loss,
            "conditional_zero_one_risk_difference": conditional_zero_one,
            "direct_loss_difference": direct_loss,
            "direct_zero_one_risk_difference": direct_zero_one,
            "sampling_unit": "one whole-path contribution per independent held-out background context",
        })
    return result


def summarize_search_trials(rows: Sequence[Mapping[str, Any]], confidence: float = 0.95):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row.get("logical_ber"),row.get("protection_budget_bits"),str(row["strategy"]))].append(row)
    result = []
    for (logical_ber,budget,strategy), values in sorted(grouped.items(),key=lambda item:((float("-inf") if item[0][0] is None else float(item[0][0])),-1 if item[0][1] is None else int(item[0][1]),item[0][2])):
        accuracy = summarize([float(value["accuracy"]) for value in values], confidence=confidence, bounds=(0, 1)).to_dict()
        loss = summarize([float(value["categorical_crossentropy"]) for value in values], confidence=confidence).to_dict()
        accuracy["minimum"] = min(float(value["accuracy"]) for value in values)
        accuracy["quantile_05"] = float(np.quantile([float(value["accuracy"]) for value in values], 0.05))
        loss["maximum"] = max(float(value["categorical_crossentropy"]) for value in values)
        result.append({**({"logical_ber":float(logical_ber)} if logical_ber is not None else {}), **({"protection_budget_bits":int(budget)} if budget is not None else {}), "strategy": strategy, "trial_count": len(values), "replication_count": len({str(value["replication"]) for value in values}), "accuracy": accuracy, "categorical_crossentropy": loss})
    return result


def paired_search_comparisons(rows: Sequence[Mapping[str, Any]], confidence: float = 0.95):
    by_condition = defaultdict(lambda:defaultdict(dict))
    for row in rows:
        identity = (str(row["replication"]), int(row["trial"]), str(row["schedule_sha256"]))
        by_condition[(row.get("logical_ber"), row.get("protection_budget_bits"))][str(row["strategy"])][identity] = row
    result = []
    for (logical_ber,budget),by_strategy in sorted(by_condition.items(),key=lambda item:(float("-inf") if item[0][0] is None else float(item[0][0]), -1 if item[0][1] is None else int(item[0][1]))):
        names=sorted(by_strategy); baseline="static" if "static" in names else ("static_gradient" if "static_gradient" in names else None)
        pairs=[(name,baseline) for name in names if name!=baseline] if baseline else list(zip(names[1:],names[:-1]))
        for left, right in pairs:
            shared = sorted(set(by_strategy[left]) & set(by_strategy[right]))
            for metric in ("accuracy", "categorical_crossentropy"):
                differences = [float(by_strategy[left][key][metric]) - float(by_strategy[right][key][metric]) for key in shared]
                if differences:
                    result.append({**({"logical_ber":float(logical_ber)} if logical_ber is not None else {}), **({"protection_budget_bits":int(budget)} if budget is not None else {}), "left": left, "right": right, "metric": metric, "matched_pairs": len(shared), "difference": summarize(differences, confidence=confidence).to_dict(), "pairing_identity": "protection_budget_bits,replication,trial,schedule_sha256"})
    return result


def render_search_report(summary: Mapping[str, Any], resolved: Mapping[str, Any], *, figure_paths=()):
    experiment = resolved["experiment"]
    lines = [
        f"# {experiment['name']}", "", experiment["description"], "",
        "## Objective", "", experiment["objective"], "",
        "## Scientific protocol", "",
        f"- Protocol: `{resolved['scientific_protocol']}`",
        f"- Benchmark: `{resolved['benchmark']['id']}`",
        f"- Quantization: `{resolved['benchmark']['quantization']}`",
        f"- Logical BER condition(s): {', '.join(f'{value:.6g}' for value in resolved['fault'].get('probabilities',[resolved['fault'].get('probability')]))} (not physical SEU rates)",
        f"- Protection budget(s): {', '.join(map(str, resolved['initial_mask'].get('budget_bits_list', [resolved['initial_mask'].get('budget_bits')])))} logical bits",
        f"- Search samples and held-out final samples are {'required to be disjoint' if resolved['samples']['require_disjoint'] else 'not required to be disjoint'}.",
        "- Search decisions use matched three-state conditional loss estimates; held-out schedules are not used to accept proposals.",
        "- Exact inference refers to encoded XOR/decode and supported-network forward equivalence, not exact knowledge of expected risk or global optimization.", "",
        "## Held-out results", "",
        "| BER | Budget bits | Strategy | Trials | Mean accuracy (95% CI) | Accuracy q05 | Minimum accuracy | Mean cross-entropy (95% CI) |",
        "|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for value in summary.get("strategy_results", []):
        accuracy, loss = value["accuracy"], value["categorical_crossentropy"]
        lines.append(f"| {value.get('logical_ber','—')} | {value.get('protection_budget_bits','—')} | {value['strategy']} | {value['trial_count']} | {accuracy['mean']:.6f} [{accuracy['confidence_interval'][0]:.6f}, {accuracy['confidence_interval'][1]:.6f}] | {accuracy['quantile_05']:.6f} | {accuracy['minimum']:.6f} | {loss['mean']:.6f} [{loss['confidence_interval'][0]:.6f}, {loss['confidence_interval'][1]:.6f}] |")
    lines += ["", "Intervals summarize held-out logical fault realizations conditional on the fixed checkpoint, dataset partitions, and each generated mask. Repeated search seeds are not independently trained networks.", "", "## Paired comparisons", "", "| Left − right | Metric | Matched pairs | Mean difference (95% CI) |", "|---|---|---:|---:|"]
    for value in summary.get("paired_comparisons", []):
        difference = value["difference"]
        lines.append(f"| {value['left']} − {value['right']}{' @ BER '+str(value['logical_ber']) if 'logical_ber' in value else ''} | {value['metric']} | {value['matched_pairs']} | {difference['mean']:.6g} [{difference['confidence_interval'][0]:.6g}, {difference['confidence_interval'][1]:.6g}] |")
    conditional = summary.get("conditional_results", [])
    if conditional:
        lines += ["", "## Independent conditional-risk comparison", "",
                  "Primary endpoint: target-mask minus common initial-mask expected cross-entropy. Each independent held-out background contributes one sum over the complete deterministic swap path; path steps are not treated as independent.", "",
                  "| BER | Budget bits | Strategy | Contexts | Replications | Swaps | Conditional Δloss (95% CI) | Direct Bernoulli Δloss (95% CI) | Conditional SE | Direct SE |",
                  "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
        for value in conditional:
            conditional_loss=value["conditional_loss_difference"]; direct_loss=value["direct_loss_difference"]
            swaps=f"{value['swap_count_min']}" if value["swap_count_min"]==value["swap_count_max"] else f"{value['swap_count_min']}–{value['swap_count_max']}"
            lines.append(f"| {value['logical_ber']:.6g} | {value.get('protection_budget_bits','—')} | {value['target']} | {value['background_context_count']} | {value['replication_count']} | {swaps} | {conditional_loss['mean']:.6g} [{conditional_loss['confidence_interval'][0]:.6g}, {conditional_loss['confidence_interval'][1]:.6g}] | {direct_loss['mean']:.6g} [{direct_loss['confidence_interval'][0]:.6g}, {direct_loss['confidence_interval'][1]:.6g}] | {conditional_loss['standard_error']:.4g} | {direct_loss['standard_error']:.4g} |")
        lines += ["", "Intervals are conditional on the fixed held-out input sets and quantify variation over independently generated persistent logical fault fields. The natural and conditional endpoints are reported separately."]
    proposal_quality=summary.get("proposal_quality",[])
    if proposal_quality:
        lines += ["", "## Proposal diagnostics", "", "Proposal probabilities and finite-polar values are ordering diagnostics, not calibrated risk predictions.", "", "| Strategy | Evaluated proposals | Accepted | Improving | Proposal-score Spearman | Top-quartile recall | Polar pairs | Polar-benefit Spearman |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for value in proposal_quality:
            show=lambda item:"—" if item is None else f"{item:.4g}"
            lines.append(f"| {value['strategy']} | {value['proposal_count']} | {value['accepted_count']} | {value['improving_count']} | {show(value['proposal_score_benefit_spearman'])} | {value['top_quartile_recall']:.4g} | {value['polar_measured_count']} | {show(value['polar_benefit_spearman'])} |")
    influence=summary.get("proposal_influence",[])
    if influence:
        lines += ["", "## Polarity-on/off proposal influence", "",
                  "These diagnostics compare the proposal distribution immediately before and after the configured interaction term. A changed distribution does not establish better protection.", "",
                  "| BER | Strategy | Iteration | Changed ranks | Top-k overlap | Total variation | Jensen–Shannon |",
                  "|---:|---|---:|---:|---:|---:|---:|"]
        for value in influence:
            lines.append(f"| {value['logical_ber']:.6g} | {value['strategy']} | {value['iteration']} | {value['changed_fraction']:.3f} | {value['top_k_overlap']:.3f} | {value['total_variation']:.4g} | {value['jensen_shannon_divergence']:.4g} |")
    screening=summary.get("screening_diagnostics",{})
    if screening and screening.get("candidate_rows"):
        coverages=", ".join(f"{100*value:.1f}%" for value in screening.get("aggregate_population_coverages",[])) or "not applicable"
        reconstruction=screening.get("maximum_aggregate_reconstruction_error")
        lines += ["", "## Mask-aware conditional screening", "",
                  "Candidate marginals use common random background fields. Their saved per-context contributions preserve covariance; `C-D` is screening guidance and only the separately recorded three-state verification is an exact swap estimate.", "",
                  f"- Candidate marginal rows: {screening['candidate_rows']}",
                  f"- Paired context-contribution rows: {screening.get('paired_context_rows',0)}",
                  f"- Exact verified factorized/screened decisions: {screening.get('factorized_decisions',0)}",
                  f"- Screening responses: {screening.get('screening_response_count',0)}",
                  f"- Aggregate population coverage values: {coverages}",
                  f"- Maximum dynamic aggregate reconstruction drift: {'—' if reconstruction is None else f'{reconstruction:.3g}'}", ""]
        costs=summary.get("computational_costs",{})
        if costs:
            show=lambda value:"—" if value is None else f"{float(value):.6g}"
            lines += ["### Component cost and break-even diagnostic", "",
                      "These are measured component costs, not an end-to-end speedup claim.", "",
                      "| Factor/aggregate setup (s) | Direct screening (s) | Exact verification (s) | Factor query/update/verification (s) | Responses/s | Verified swaps/s | Estimated break-even iterations |",
                      "|---:|---:|---:|---:|---:|---:|---:|",
                      f"| {show(costs.get('factor_and_aggregate_setup_seconds'))} | {show(costs.get('conditional_screening_seconds'))} | {show(costs.get('exact_verification_seconds'))} | {show(costs.get('factor_query_update_and_verification_seconds'))} | {show(costs.get('candidate_responses_per_second'))} | {show(costs.get('verified_swaps_per_second'))} | {show(costs.get('component_break_even_iterations'))} |", "",
                      "The break-even value is a hypothetical component substitution estimate. The current hybrid still performs exact screening, so it is not an end-to-end speedup measurement.", ""]
    timing=summary.get("timing_audit",[])
    whole=[value for value in timing if value.get("phase")=="whole_process"]
    phases=[value for value in timing if value.get("phase")!="whole_process"]
    if timing:
        lines += ["## Exclusive timing audit", "",
                  "Phase rows are mutually exclusive. Search-result times are inclusive method times and convergence times are cumulative trajectory times; neither may be added to this audit.", "",
                  "| Replication | Exclusive phase sum (s) | Whole elapsed (s) | Uninstrumented (s) |",
                  "|---|---:|---:|---:|"]
        for value in whole:
            lines.append(f"| {value['replication']} | {float(value.get('exclusive_phase_sum_seconds',0)):.6f} | {float(value['exclusive_wall_seconds']):.6f} | {float(value.get('uninstrumented_seconds',0)):.6f} |")
        lines.append("")
    lines += ["", "## Incremental exactness", ""]
    checks = summary.get("incremental_validation", [])
    lines.append(f"Bounded validation cases: {len(checks)}; maximum logit absolute error: {max((float(v['max_logit_absolute_error']) for v in checks), default=0):.3g}; prediction mismatches: {sum(int(v['prediction_mismatches']) for v in checks)}.")
    if summary.get("exact_optimum"):
        lines += ["", "## Restricted exact optimum", "", f"The deliberately small candidate-pool enumeration found risk `{summary['exact_optimum']['risk']:.8g}`. This is an exact optimum only within the configured restricted pool and search-sample risk distribution."]
    if figure_paths:
        lines += ["", "## Figures", ""]
        for path in figure_paths:
            lines += [f"![{Path(path).stem}]({path})", ""]
    lines += ["## Limitations", "", "- Conditional benefits may be positive, zero, or negative; no monotonicity or submodularity is assumed.", "- Polar interactions guide proposals only. Every accepted deterministic improvement is based on exact matched conditional evaluation.", "- Simulated annealing is a noisy heuristic, not an exact posterior sampler or Metropolis–Hastings chain.", "- The incremental evaluator currently supports only the explicit Dense→ReLU→Dense adapters and otherwise must fall back to full inference.", "- Logical bit count is a protection budget, not measured hardware area or power.", "", "## Reproduction", "", f"`netsurf run {resolved['source_configuration']}`", "", f"Scientific payload SHA-256: `{summary['scientific_payload_sha256']}`", ""]
    return "\n".join(lines)


def plot_search_results(summary, output: Path, *, trials=(), convergence=(), validation=(), polar=(), topology=(), search_steps=(), composition=(), search_results=(), proposal_diagnostics=(), screening=(), factorized=(), timing=()):
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        from .plotting import PlottingDependencyError
        raise PlottingDependencyError("Matplotlib is required for search plots") from error
    output.mkdir(parents=True, exist_ok=True)
    generated = []
    if screening:
        strategies=sorted({str(value["strategy"]) for value in screening})
        data=[[float(value["exclusive_wall_seconds"]) for value in screening if str(value["strategy"])==strategy] for strategy in strategies]
        fig,ax=plt.subplots(figsize=(max(7,len(strategies)*1.2),4.5)); ax.boxplot(data,tick_labels=strategies,showmeans=True); ax.set_yscale("log"); ax.set(ylabel="Exclusive seconds per candidate marginal",title="Conditional-screening amortized latency"); ax.tick_params(axis="x",rotation=25); fig.tight_layout(); path=output/"conditional_screening_cost.png"; fig.savefig(path,dpi=160); plt.close(fig); generated.append(path)
        first={(str(value["strategy"]),str(value["identity"])):float(value["weighted_mean"]) for value in screening if int(value["iteration"])==1}
        if len(strategies)>1:
            matrix=np.eye(len(strategies))
            for i,left in enumerate(strategies):
                for j,right in enumerate(strategies):
                    shared=sorted({identity for strategy,identity in first if strategy==left}&{identity for strategy,identity in first if strategy==right})
                    matrix[i,j]=np.nan if len(shared)<2 else spearman([first[(left,key)] for key in shared],[first[(right,key)] for key in shared])
            fig,ax=plt.subplots(figsize=(max(6,len(strategies)*.8),max(5,len(strategies)*.65))); image=ax.imshow(matrix,vmin=-1,vmax=1,cmap="coolwarm"); ax.set_xticks(range(len(strategies)),strategies,rotation=35,ha="right"); ax.set_yticks(range(len(strategies)),strategies); ax.set_title("Candidate marginal rank agreement (iteration 1 overlap)"); fig.colorbar(image,ax=ax,label="Spearman correlation"); fig.tight_layout(); path=output/"candidate_ranking_agreement.png"; fig.savefig(path,dpi=160); plt.close(fig); generated.append(path)
    if factorized:
        comparable=[value for value in factorized if value.get("screening_score") not in (None,"") and value.get("exact_risk_difference") not in (None,"")]
        if comparable:
            fig,ax=plt.subplots(figsize=(6,5)); x=[float(value["screening_score"]) for value in comparable]; y=[float(value["exact_risk_difference"]) for value in comparable]; ax.scatter(x,y,alpha=.7); limits=[min(x+y+[0]),max(x+y+[0])]; ax.plot(limits,limits,"--",color="black"); ax.set(xlabel="Screening estimate C−D (guidance)",ylabel="Exact three-state Δrisk",title="Screening guidance versus exact swap verification"); fig.tight_layout(); path=output/"screening_vs_exact_swap.png"; fig.savefig(path,dpi=160); plt.close(fig); generated.append(path)
        explicit=[]
        for value in factorized:
            for prefix in ("removed","added"):
                if value.get(f"{prefix}_explicit_score") not in (None,""):
                    explicit.append((float(value[f"{prefix}_explicit_score"]),float(value[f"{prefix}_aggregate_score"])))
        if explicit:
            x,y=zip(*explicit); fig,ax=plt.subplots(figsize=(6,5)); ax.scatter(x,y,alpha=.7); limits=[min((*x,*y,0)),max((*x,*y,0))]; ax.plot(limits,limits,"--",color="black"); ax.set(xlabel="Explicit vulnerable-bit Kronecker sum",ylabel="Factorized aggregate score",title="Explicit versus factorized aggregate"); fig.tight_layout(); path=output/"explicit_vs_factorized_aggregate.png"; fig.savefig(path,dpi=160); plt.close(fig); generated.append(path)
        drift=[value for value in factorized if value.get("aggregate_reconstruction_error") not in (None,"")]
        if drift:
            fig,ax=plt.subplots(figsize=(7,4.5))
            for strategy in sorted({str(value["strategy"]) for value in drift}):
                values=sorted((value for value in drift if str(value["strategy"])==strategy),key=lambda value:int(value["iteration"])); ax.plot([int(value["iteration"]) for value in values],[float(value["aggregate_reconstruction_error"]) for value in values],marker="o",label=strategy)
            ax.set_yscale("symlog",linthresh=1e-15); ax.set(xlabel="Search iteration",ylabel="Maximum aggregate reconstruction drift",title="Dynamic Kronecker aggregate consistency"); ax.legend(fontsize=8); fig.tight_layout(); path=output/"aggregate_update_drift.png"; fig.savefig(path,dpi=160); plt.close(fig); generated.append(path)
        score_rows=[value for value in factorized if value.get("removed_aggregate_score") not in (None,"")]
        if score_rows:
            fig,ax=plt.subplots(figsize=(7,4.5))
            for strategy in sorted({str(value["strategy"]) for value in score_rows}):
                values=sorted((value for value in score_rows if str(value["strategy"])==strategy),key=lambda value:int(value["iteration"])); ax.plot([int(value["iteration"]) for value in values],[float(value["removed_aggregate_score"])-float(value["added_aggregate_score"]) for value in values],marker="o",label=strategy)
            ax.axhline(0,color="black",lw=1); ax.set(xlabel="Search iteration",ylabel="Selected removal−addition aggregate score",title="Mask-dependent factorized score trajectory"); ax.legend(fontsize=8); fig.tight_layout(); path=output/"dynamic_aggregate_scores.png"; fig.savefig(path,dpi=160); plt.close(fig); generated.append(path)
        coverage=[value for value in factorized if value.get("aggregate_population_coverage") not in (None,"")]
        if coverage:
            names=sorted({str(value["strategy"]) for value in coverage}); values=[max(float(value["aggregate_population_coverage"]) for value in coverage if str(value["strategy"])==name) for name in names]; fig,ax=plt.subplots(figsize=(max(7,len(names)*1.2),4)); ax.bar(names,values); ax.set_ylim(0,1.05); ax.set(ylabel="Eligible population coverage",title="Factorized interaction aggregate coverage"); ax.tick_params(axis="x",rotation=25); fig.tight_layout(); path=output/"factorized_population_coverage.png"; fig.savefig(path,dpi=160); plt.close(fig); generated.append(path)
    timing_phases=[value for value in timing if value.get("phase")!="whole_process" and value.get("category")!="reporting"]
    if timing_phases:
        grouped=defaultdict(float)
        for value in timing_phases: grouped[str(value["phase"])]+=float(value["exclusive_wall_seconds"])
        fig,ax=plt.subplots(figsize=(max(7,len(grouped)*1.1),4.5)); ax.bar(list(grouped),list(grouped.values())); ax.set(ylabel="Exclusive wall seconds",title="Mutually exclusive engine timing"); ax.tick_params(axis="x",rotation=30); fig.tight_layout(); path=output/"exclusive_phase_timing.png"; fig.savefig(path,dpi=160); plt.close(fig); generated.append(path)
    conditional = summary.get("conditional_results", [])
    if conditional:
        for logical_ber in sorted({float(value["logical_ber"]) for value in conditional}):
            values=[value for value in conditional if float(value["logical_ber"])==logical_ber]
            labels=[str(value["target"]) for value in values]; means=[float(value["conditional_loss_difference"]["mean"]) for value in values]
            errors=[[mean-float(value["conditional_loss_difference"]["confidence_interval"][0]) for mean,value in zip(means,values)],[float(value["conditional_loss_difference"]["confidence_interval"][1])-mean for mean,value in zip(means,values)]]
            fig,ax=plt.subplots(figsize=(max(7,len(values)*1.1),4.5)); ax.errorbar(np.arange(len(values)),means,yerr=errors,fmt="o",capsize=4); ax.axhline(0,color="black",lw=1); ax.set_xticks(np.arange(len(values)),labels,rotation=25); ax.set(ylabel="Conditional Δ cross-entropy vs initial mask",title=f"Held-out conditional mask comparison — BER {logical_ber:g}"); fig.tight_layout(); path=output/f"conditional_risk_ber_{logical_ber:g}.png"; fig.savefig(path,dpi=160); plt.close(fig); generated.append(path)
        labels=[f"{value['target']}\n{value['logical_ber']:g}" for value in conditional]; x=np.arange(len(conditional)); width=.36
        fig,ax=plt.subplots(figsize=(max(8,len(conditional)*1.1),4.5)); ax.bar(x-width/2,[float(value["conditional_loss_difference"]["standard_error"]) for value in conditional],width,label="conditional"); ax.bar(x+width/2,[float(value["direct_loss_difference"]["standard_error"]) for value in conditional],width,label="direct Bernoulli"); ax.set_xticks(x,labels,rotation=25); ax.set(ylabel="Standard error of paired Δ cross-entropy",title="Conditional versus direct Monte Carlo precision"); ax.legend(); fig.tight_layout(); path=output/"conditional_vs_direct_standard_error.png"; fig.savefig(path,dpi=160); plt.close(fig); generated.append(path)
    polar_proposals=[value for value in proposal_diagnostics if value.get("polar_interaction") not in (None,"")]
    if polar_proposals:
        fig,ax=plt.subplots(figsize=(6.5,4.5)); ax.scatter([float(value["polar_interaction"]) for value in polar_proposals],[float(value["exact_search_benefit"]) for value in polar_proposals],alpha=.65); ax.axhline(0,color="black",lw=1); ax.set(xlabel="Finite polar proposal diagnostic",ylabel="Exact search-sample conditional benefit",title="Polar guidance versus verified proposal benefit"); fig.tight_layout(); path=output/"polar_proposal_quality.png"; fig.savefig(path,dpi=160); plt.close(fig); generated.append(path)
    if convergence:
        for xfield, filename, xlabel in (("exact_evaluations", "risk_vs_evaluations.png", "Exact network evaluations"), ("work_proxy_macs", "risk_vs_work.png", "Inference-work proxy (MACs)"), ("wall_seconds", "risk_vs_time.png", "Measured search time (s)")):
            fig, ax = plt.subplots(figsize=(7, 4.5))
            for strategy in sorted({str(v["strategy"]) for v in convergence}):
                rows = sorted((v for v in convergence if str(v["strategy"]) == strategy), key=lambda v: float(v[xfield]))
                ax.plot([float(v[xfield]) for v in rows], [float(v["heldout_loss_mean"]) for v in rows], marker="o", label=strategy)
            ax.set(xlabel=xlabel, ylabel="Held-out mean cross-entropy", title="Post-hoc held-out search trajectory")
            ax.legend(fontsize=8); ax.grid(alpha=.25); fig.tight_layout(); path=output/filename; fig.savefig(path, dpi=160); plt.close(fig); generated.append(path)
    if validation:
        fig, ax = plt.subplots(figsize=(7, 4.5)); names=[str(v["case"]) for v in validation]; x=np.arange(len(names)); width=.35
        ax.bar(x-width/2, [float(v["full_seconds_mean"]) for v in validation], width, label="full")
        ax.bar(x+width/2, [float(v["incremental_seconds_mean"]) for v in validation], width, label="incremental")
        ax.set_xticks(x, names); ax.set_ylabel("Seconds (mean of 3 bounded repeats)"); ax.set_title("Inference timing by earliest affected layer"); ax.legend(); fig.tight_layout(); path=output/"incremental_timing.png"; fig.savefig(path,dpi=160); plt.close(fig); generated.append(path)
    if trials and len({str(v["strategy"]) for v in trials}) > 1:
        static={(str(v["replication"]),int(v["trial"]),str(v["schedule_sha256"])):float(v["accuracy"]) for v in trials if str(v["strategy"])=="static"}
        data=[]; labels=[]
        for strategy in sorted({str(v["strategy"]) for v in trials}-{ "static" }):
            values=[float(v["accuracy"])-static[(str(v["replication"]),int(v["trial"]),str(v["schedule_sha256"]))] for v in trials if str(v["strategy"])==strategy and (str(v["replication"]),int(v["trial"]),str(v["schedule_sha256"])) in static]
            if values: data.append(values); labels.append(strategy)
        if data:
            fig,ax=plt.subplots(figsize=(max(7,len(data)*1.2),4.5)); ax.boxplot(data, tick_labels=labels, showmeans=True); ax.axhline(0,color="black",lw=1); ax.set_ylabel("Paired accuracy difference vs static"); ax.tick_params(axis="x",rotation=25); fig.tight_layout(); path=output/"paired_optimizer_accuracy.png"; fig.savefig(path,dpi=160); plt.close(fig); generated.append(path)
    if polar:
        fig,ax=plt.subplots(figsize=(6,5)); x=[float(v["finite_polar_interaction"]) for v in polar]; y=[float(v["exact_interaction"]) for v in polar]; ax.scatter(x,y,alpha=.65); limits=[min(x+y+[0]),max(x+y+[0])]; ax.plot(limits,limits,"--",color="black"); ax.set(xlabel="Finite additive-logit polar",ylabel="Exact four-state interaction",title="Polar calibration"); fig.tight_layout(); path=output/"polar_calibration.png"; fig.savefig(path,dpi=160); plt.close(fig); generated.append(path)
        errors=[float(v["interaction_prediction_error"]) for v in polar]
        agreements=[bool(v["interaction_sign_agreement"]) for v in polar]
        fig,(error_ax,sign_ax)=plt.subplots(1,2,figsize=(10,4))
        error_ax.hist(errors,bins=min(20,max(5,len(errors)//2))); error_ax.axvline(0,color="black",lw=1)
        error_ax.set(xlabel="Exact − polar interaction",ylabel="Sampled pairs",title="Polar prediction error")
        counts=[sum(agreements),len(agreements)-sum(agreements)]
        sign_ax.bar(["Agreement","Disagreement"],counts)
        sign_ax.set(ylabel="Sampled pairs",title=f"Interaction sign agreement (n={len(agreements)})")
        fig.tight_layout(); path=output/"polar_error_and_sign.png"; fig.savefig(path,dpi=160); plt.close(fig); generated.append(path)
        cancellations=[float(v["cancellation_score"]) for v in polar if v.get("cancellation_score") not in (None,"")]
        if cancellations:
            fig,ax=plt.subplots(figsize=(6,4)); ax.hist(cancellations,bins=min(20,max(5,len(cancellations)//2))); ax.axvline(1,color="black",ls="--"); ax.set(xlabel="Additive-logit cancellation score",ylabel="Sampled pairs",title="Exploratory pair cancellation"); fig.tight_layout(); path=output/"polar_cancellation.png"; fig.savefig(path,dpi=160); plt.close(fig); generated.append(path)
    if topology:
        errors=[abs(float(value["topology_exact_interaction_error"])) for value in topology]
        classes=sorted({str(value["topology_class"]) for value in topology})
        data=[[abs(float(value["topology_exact_interaction_error"])) for value in topology if str(value["topology_class"])==name] for name in classes]
        fig,ax=plt.subplots(figsize=(max(7,len(classes)*1.2),4.5)); ax.boxplot(data,tick_labels=classes,showmeans=True); ax.set_yscale("symlog",linthresh=1e-15); ax.set(ylabel="Absolute interaction reconstruction error",title="Exact topology reconstruction by pair class"); ax.tick_params(axis="x",rotation=25); fig.tight_layout(); path=output/"topology_exact_error.png"; fig.savefig(path,dpi=160); plt.close(fig); generated.append(path)
        topology_times=[float(value["wall_seconds"]) for value in topology if value.get("wall_seconds") not in (None,"")]
        existing_times=[float(value["wall_seconds"]) for value in polar if value.get("wall_seconds") not in (None,"")]
        if topology_times and existing_times:
            fig,ax=plt.subplots(figsize=(6.5,4.5)); ax.boxplot([existing_times,topology_times],tick_labels=["Existing incremental","Topology-aware"],showmeans=True); ax.set_yscale("log"); ax.set(ylabel="Wall time per sampled pair (s)",title="Pair-interaction evaluation latency"); fig.tight_layout(); path=output/"topology_pair_latency.png"; fig.savefig(path,dpi=160); plt.close(fig); generated.append(path)
        for field,title,filename in (("local_quadratic","Local quadratic","local_vs_finite_polar.png"),("kronecker","Conventional Kronecker","kronecker_vs_finite_polar.png"),("stratified_kronecker","Stratified Kronecker","stratified_kron_vs_finite_polar.png")):
            values=[value for value in topology if value.get(field) not in (None,"")]
            if values:
                fig,ax=plt.subplots(figsize=(6,5)); x=[float(value["finite_polar"]) for value in values]; y=[float(value[field]) for value in values]; ax.scatter(x,y,alpha=.65); limits=[min(x+y+[0]),max(x+y+[0])]; ax.plot(limits,limits,"--",color="black"); ax.set(xlabel="Exact finite additive-logit polar",ylabel=title,title=f"{title} approximation"); fig.tight_layout(); path=output/filename; fig.savefig(path,dpi=160); plt.close(fig); generated.append(path)
    layer_rows=[v for v in composition if str(v["dimension"])=="layer"]
    if layer_rows:
        strategies=sorted({str(v["strategy"]) for v in layer_rows}); layers=sorted({str(v["category"]) for v in layer_rows}); x=np.arange(len(strategies)); bottom=np.zeros(len(strategies)); fig,ax=plt.subplots(figsize=(max(7,len(strategies)*1.2),4.5))
        for layer in layers:
            values=[sum(int(v["protected_count"]) for v in layer_rows if str(v["strategy"])==strategy and str(v["category"])==layer) for strategy in strategies]
            ax.bar(x,values,bottom=bottom,label=layer); bottom+=values
        ax.set_xticks(x,strategies,rotation=25); ax.set_ylabel("Protected bits (summed over replications)"); ax.set_title("Best-mask layer composition"); ax.legend(); fig.tight_layout(); path=output/"mask_layer_composition.png"; fig.savefig(path,dpi=160); plt.close(fig); generated.append(path)
    position_rows=[v for v in composition if str(v["dimension"])=="bit_position"]
    if position_rows:
        strategies=sorted({str(v["strategy"]) for v in position_rows}); positions=sorted({str(v["category"]) for v in position_rows},key=int); x=np.arange(len(strategies)); bottom=np.zeros(len(strategies)); fig,ax=plt.subplots(figsize=(max(7,len(strategies)*1.2),4.5))
        for position in positions:
            values=[sum(int(v["protected_count"]) for v in position_rows if str(v["strategy"])==strategy and str(v["category"])==position) for strategy in strategies]
            ax.bar(x,values,bottom=bottom,label=f"bit {position}"); bottom+=values
        ax.set_xticks(x,strategies,rotation=25); ax.set_ylabel("Protected bits (summed over replications)"); ax.set_title("Best-mask bit-position composition"); ax.legend(ncols=min(3,len(positions))); fig.tight_layout(); path=output/"mask_bit_position_composition.png"; fig.savefig(path,dpi=160); plt.close(fig); generated.append(path)
    gaps=[v for v in search_results if v.get("restricted_exact_gap") not in (None,"")]
    if gaps:
        fig,ax=plt.subplots(figsize=(max(6,len(gaps)*.8),4)); ax.bar(range(len(gaps)),[float(v["restricted_exact_gap"]) for v in gaps]); ax.set_xticks(range(len(gaps)),[str(v["strategy"]) for v in gaps],rotation=25); ax.axhline(0,color="black",lw=1); ax.set_ylabel("Exact restricted-pool risk gap"); ax.set_title("Search gap to restricted exact optimum"); fig.tight_layout(); path=output/"restricted_optimization_gap.png"; fig.savefig(path,dpi=160); plt.close(fig); generated.append(path)
    return tuple(generated)
