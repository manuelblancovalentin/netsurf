"""Saved-data summaries, reports, and plots for schema-4 search studies."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .statistics import summarize


def summarize_search_trials(rows: Sequence[Mapping[str, Any]], confidence: float = 0.95):
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["strategy"])].append(row)
    result = []
    for strategy, values in sorted(grouped.items()):
        accuracy = summarize([float(value["accuracy"]) for value in values], confidence=confidence, bounds=(0, 1)).to_dict()
        loss = summarize([float(value["categorical_crossentropy"]) for value in values], confidence=confidence).to_dict()
        accuracy["minimum"] = min(float(value["accuracy"]) for value in values)
        accuracy["quantile_05"] = float(np.quantile([float(value["accuracy"]) for value in values], 0.05))
        loss["maximum"] = max(float(value["categorical_crossentropy"]) for value in values)
        result.append({"strategy": strategy, "trial_count": len(values), "replication_count": len({str(value["replication"]) for value in values}), "accuracy": accuracy, "categorical_crossentropy": loss})
    return result


def paired_search_comparisons(rows: Sequence[Mapping[str, Any]], confidence: float = 0.95):
    by_strategy = defaultdict(dict)
    for row in rows:
        identity = (str(row["replication"]), int(row["trial"]), str(row["schedule_sha256"]))
        by_strategy[str(row["strategy"])][identity] = row
    names = sorted(by_strategy)
    if "static" in names:
        pairs = [(name, "static") for name in names if name != "static"]
    else:
        pairs = list(zip(names[1:], names[:-1]))
    result = []
    for left, right in pairs:
        shared = sorted(set(by_strategy[left]) & set(by_strategy[right]))
        for metric in ("accuracy", "categorical_crossentropy"):
            differences = [float(by_strategy[left][key][metric]) - float(by_strategy[right][key][metric]) for key in shared]
            if differences:
                result.append({"left": left, "right": right, "metric": metric, "matched_pairs": len(shared), "difference": summarize(differences, confidence=confidence).to_dict(), "pairing_identity": "replication,trial,schedule_sha256"})
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
        f"- Logical BER: {resolved['fault']['probability']:.6g} (not a physical SEU rate)",
        f"- Protection budget: {resolved['initial_mask']['budget_bits']} logical bits",
        f"- Search samples and held-out final samples are {'required to be disjoint' if resolved['samples']['require_disjoint'] else 'not required to be disjoint'}.",
        "- Search decisions use matched three-state conditional loss estimates; held-out schedules are not used to accept proposals.",
        "- Exact inference refers to encoded XOR/decode and supported-network forward equivalence, not exact knowledge of expected risk or global optimization.", "",
        "## Held-out results", "",
        "| Strategy | Trials | Mean accuracy (95% CI) | Accuracy q05 | Minimum accuracy | Mean cross-entropy (95% CI) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for value in summary.get("strategy_results", []):
        accuracy, loss = value["accuracy"], value["categorical_crossentropy"]
        lines.append(f"| {value['strategy']} | {value['trial_count']} | {accuracy['mean']:.6f} [{accuracy['confidence_interval'][0]:.6f}, {accuracy['confidence_interval'][1]:.6f}] | {accuracy['quantile_05']:.6f} | {accuracy['minimum']:.6f} | {loss['mean']:.6f} [{loss['confidence_interval'][0]:.6f}, {loss['confidence_interval'][1]:.6f}] |")
    lines += ["", "Intervals summarize held-out logical fault realizations conditional on the fixed checkpoint, dataset partitions, and each generated mask. Repeated search seeds are not independently trained networks.", "", "## Paired comparisons", "", "| Left − right | Metric | Matched pairs | Mean difference (95% CI) |", "|---|---|---:|---:|"]
    for value in summary.get("paired_comparisons", []):
        difference = value["difference"]
        lines.append(f"| {value['left']} − {value['right']} | {value['metric']} | {value['matched_pairs']} | {difference['mean']:.6g} [{difference['confidence_interval'][0]:.6g}, {difference['confidence_interval'][1]:.6g}] |")
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


def plot_search_results(summary, output: Path, *, trials=(), convergence=(), validation=(), polar=(), search_steps=(), composition=(), search_results=()):
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        from .plotting import PlottingDependencyError
        raise PlottingDependencyError("Matplotlib is required for search plots") from error
    output.mkdir(parents=True, exist_ok=True)
    generated = []
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
        cancellations=[float(v["cancellation_score"]) for v in polar if v.get("cancellation_score") not in (None,"")]
        if cancellations:
            fig,ax=plt.subplots(figsize=(6,4)); ax.hist(cancellations,bins=min(20,max(5,len(cancellations)//2))); ax.axvline(1,color="black",ls="--"); ax.set(xlabel="Additive-logit cancellation score",ylabel="Sampled pairs",title="Exploratory pair cancellation"); fig.tight_layout(); path=output/"polar_cancellation.png"; fig.savefig(path,dpi=160); plt.close(fig); generated.append(path)
    layer_rows=[v for v in composition if str(v["dimension"])=="layer"]
    if layer_rows:
        strategies=sorted({str(v["strategy"]) for v in layer_rows}); layers=sorted({str(v["category"]) for v in layer_rows}); x=np.arange(len(strategies)); bottom=np.zeros(len(strategies)); fig,ax=plt.subplots(figsize=(max(7,len(strategies)*1.2),4.5))
        for layer in layers:
            values=[sum(int(v["protected_count"]) for v in layer_rows if str(v["strategy"])==strategy and str(v["category"])==layer) for strategy in strategies]
            ax.bar(x,values,bottom=bottom,label=layer); bottom+=values
        ax.set_xticks(x,strategies,rotation=25); ax.set_ylabel("Protected bits (summed over replications)"); ax.set_title("Best-mask layer composition"); ax.legend(); fig.tight_layout(); path=output/"mask_layer_composition.png"; fig.savefig(path,dpi=160); plt.close(fig); generated.append(path)
    gaps=[v for v in search_results if v.get("restricted_exact_gap") not in (None,"")]
    if gaps:
        fig,ax=plt.subplots(figsize=(max(6,len(gaps)*.8),4)); ax.bar(range(len(gaps)),[float(v["restricted_exact_gap"]) for v in gaps]); ax.set_xticks(range(len(gaps)),[str(v["strategy"]) for v in gaps],rotation=25); ax.axhline(0,color="black",lw=1); ax.set_ylabel("Exact restricted-pool risk gap"); ax.set_title("Search gap to restricted exact optimum"); fig.tight_layout(); path=output/"restricted_optimization_gap.png"; fig.savefig(path,dpi=160); plt.close(fig); generated.append(path)
    return tuple(generated)
