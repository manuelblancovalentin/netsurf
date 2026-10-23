"""Human-readable scientific reports generated from saved result data."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def render_report(summary: Mapping[str, Any], resolved: Mapping[str, Any], *, figure_paths: tuple[str, ...] = ()) -> str:
    experiment, benchmark, fault = resolved["experiment"], resolved["benchmark"], resolved["fault"]
    lines = [
        f"# {experiment['name']}", "", experiment.get("description", ""), "",
        "## Objective", "", experiment.get("objective", "Not specified."), "",
        "## Experimental contract", "",
        f"- Benchmark: `{benchmark['id']}` (`{benchmark['adapter']}`).",
        f"- Dataset split: `{benchmark['split']}`; {benchmark['evaluation_samples']} evaluation samples and {benchmark['ranking_samples']} ranking samples.",
        f"- Quantization: `{benchmark['quantization']}`.",
        f"- Fault model: `{fault['distribution']}` logical `{fault['target']}` faults at probability `{fault['probability']}`, `{fault['persistence']}`.",
        f"- Independent uncertainty unit: fault trial; {summary['trial_count']} trials at confidence `{resolved['execution']['confidence']}`.",
        "- Policy comparisons within a trial use the same fault schedule and are therefore paired.",
        "- Protection is modeled as perfect suppression of selected logical-bit events; this is not a calibrated physical SEU rate or a hardware guarantee.",
        "", "## Baseline", "",
        f"Checkpoint accuracy: `{summary['baseline']['checkpoint']['accuracy']:.6f}`; categorical cross-entropy: `{summary['baseline']['checkpoint']['categorical_crossentropy']:.6f}`.",
        "", "## Measured and derived results", "",
        "Trial rows are measured evaluations. Means and confidence intervals below are derived across independent fault trials.", "",
        "| Policy | Protected bits | Coverage | Accuracy mean | Confidence interval | Cross-entropy mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, result in summary["protection_results"].items():
        accuracy, loss = result["accuracy"], result["categorical_crossentropy"]
        interval = accuracy["confidence_interval"]
        lines.append(f"| {name} | {result['protected_bits']} | {result['coverage']:.6f} | {accuracy['mean']:.6f} | [{interval[0]:.6f}, {interval[1]:.6f}] | {loss['mean']:.6f} |")
    if summary.get("paired_comparisons"):
        lines.extend([
            "", "## Paired protection benefit", "",
            "Differences are computed within each matched fault trial relative to `no_protection`; intervals use fault trials as the independent unit.", "",
            "| Ranker | Accuracy difference | Confidence interval | Cross-entropy difference |",
            "|---|---:|---:|---:|",
        ])
        for name, comparison in summary["paired_comparisons"].items():
            accuracy = comparison["accuracy_difference"]
            loss = comparison["crossentropy_difference"]
            interval = accuracy["confidence_interval"]
            lines.append(
                f"| {name} | {accuracy['mean']:.6f} | [{interval[0]:.6f}, {interval[1]:.6f}] | {loss['mean']:.6f} |"
            )
    if summary.get("estimator_runtimes_seconds"):
        lines.extend(["", "## Ranking computation time", "", "| Estimator | Wall time (seconds) |", "|---|---:|"])
        for name, runtime in summary["estimator_runtimes_seconds"].items():
            lines.append(f"| {name} | {runtime:.6f} |")
        lines.extend(["", "Runtime is execution metadata and is excluded from the stable scientific payload hash."])
    if figure_paths:
        lines.extend(["", "## Figures", ""])
        for path in figure_paths:
            name = Path(path).name
            if name.startswith("loss_"):
                alt, caption = "Loss with trial-based confidence intervals", "Cross-entropy by policy; error bars use independent fault trials."
            elif name.startswith("accuracy_benefit"):
                alt, caption = "Paired accuracy benefit relative to no protection", "Paired accuracy difference relative to no protection; error bars use matched independent fault trials."
            else:
                alt, caption = "Accuracy with trial-based confidence intervals", "Accuracy by policy; error bars use independent fault trials."
            lines.extend([f"![{alt}]({path})", "", caption, ""])
    reproduction = resolved.get("source_configuration", "config.toml")
    lines.extend([
        "## Reproducibility checks", "",
        *[f"- `{name}`: `{value}`." for name, value in summary["checks"].items()], "",
        "## Limitations", "",
        "This report does not establish physical radiation reliability, fabricated-silicon cost, or statistical superiority among policies. Logical protection is ideal event suppression rather than synthesized ECC or TMR.", "",
        "## Reproduction", "", "```bash", f"netsurf run {reproduction}", "```", "",
        "The bundled `config.toml` is the exact submitted configuration; resolved paths and executed provenance are recorded separately in `resolved_config.json` and `manifest.json`.", "",
        f"Scientific payload SHA-256: `{summary['scientific_payload_sha256']}`", "",
    ])
    return "\n".join(lines)


def write_report(summary: Mapping[str, Any], resolved: Mapping[str, Any], output: Path, *, figure_paths: tuple[str, ...] = ()) -> Path:
    output = Path(output)
    output.write_text(render_report(summary, resolved, figure_paths=figure_paths), encoding="utf-8")
    return output


def render_study_report(summary: Mapping[str, Any], resolved: Mapping[str, Any], *, figure_paths: tuple[str, ...] = ()) -> str:
    """Render an M3 report entirely from saved summary and resolved data."""

    experiment = resolved["experiment"]
    benchmark = resolved["benchmark"]
    lines = [
        f"# {experiment['name']}", "", experiment.get("description", ""), "",
        "## Objective", "", experiment.get("objective", "Not specified."), "",
        "## Scientific protocol", "",
        f"- Protocol: `{resolved['scientific_protocol']}`.",
        f"- Benchmark: `{benchmark['id']}` (`{benchmark['adapter']}`).",
        f"- Quantization: `{benchmark['quantization']}`; eligible bits: `{summary['eligible_bits']}`.",
        f"- Ranking/calibration samples: `{resolved['samples']['ranking_count']}`; empirical-validation samples: `{resolved['samples']['empirical_count']}`; final-evaluation samples: `{resolved['samples']['evaluation_count']}`.",
        f"- Sample partitions are required to be disjoint: `{resolved['samples']['require_disjoint']}`.",
        "- Estimator scores, ordering policies, and protection-budget allocation are separate recorded objects.",
        "", "## Clean evaluation", "",
        f"Accuracy: `{summary['clean_evaluation']['accuracy']:.6f}`; categorical cross-entropy: `{summary['clean_evaluation']['categorical_crossentropy']:.6f}`.",
    ]
    if summary.get("empirical_candidate_count", 0):
        lines.extend([
            "", "## Exact isolated-bit reference", "",
            f"Exactly one encoded bit was flipped for each of `{summary['empirical_candidate_count']}` candidates. Signed loss changes were retained.",
            "The reference addresses isolated-fault prediction and damaging-bit identification; it does not prove optimality under multi-fault schedules.",
            "", "### Ranking-quality summary", "",
            "| Estimator | Spearman (positive damage) | Kendall tau-b | Top-10 recall | Zero-score fraction | Wall time (s) |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for row in summary.get("ranking_metrics_overall", []):
            def show(value):
                return "N/A" if value is None else f"{value:.6f}"
            lines.append(
                f"| {row['estimator']} | {show(row['spearman_positive_damage'])} | {show(row['kendall_positive_damage'])} | "
                f"{show(row['top_10_recall'])} | {show(row['zero_score_fraction'])} | {row['runtime_seconds']:.6f} |"
            )
        if summary.get("sample_limited"):
            lines.extend(["", "These metrics are sample-limited to the recorded candidate set and are not whole-network top-k claims. Consult the saved inclusion rule and probabilities for the sampling design."])
    agreement = summary.get("estimator_agreement", [])
    qpolar_agreement = [row for row in agreement if {row["left"], row["right"]} == {"clean_activation_qpolargrad", "uncancelled_activation_drift"}]
    if qpolar_agreement:
        row = qpolar_agreement[0]
        lines.extend([
            "", "## Activation-drift aggregation diagnostic", "",
            f"Clean signed-mean and magnitude-before-mean qPolarGrad scores have Spearman `{row['spearman']}`; `{row['changed_average_rank_count']}` of `{row['candidate_count']}` candidates changed average rank, with maximum shift `{row['maximum_average_rank_shift']}`.",
            f"Exact score equality: `{row['identical_scores']}`; exact deterministic order equality: `{row['identical_order']}`.",
            "Different scores need not imply different orderings; both are recorded explicitly.",
        ])
    if summary.get("protection_results"):
        probabilities = resolved["protection"].get("probabilities") or [resolved["fault"]["probability"]]
        probability_text = ", ".join(str(value) for value in probabilities)
        lines.extend([
            "", "## Matched protection experiment", "",
            f"Logical BER condition(s) `{probability_text}` with persistence `{resolved['fault']['persistence']}` were each evaluated in `{summary.get('pooled_fault_trial_count', summary['trial_count'])}` pooled fault realizations across `{summary.get('replication_count', 1)}` replication(s).",
            "All policies used the same underlying schedule within each trial. Protection means ideal logical event suppression, not ECC or TMR hardware.",
            f"The confidence intervals describe {summary.get('interval_interpretation', 'fault-trial variability conditional on the recorded experiment')}.",
            "", f"| BER | Policy | Coverage | n | Mean accuracy | SD | {100 * resolved['execution']['confidence']:.1f}% CI | Median | 5th percentile | Minimum | Mean loss |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for result in summary["protection_results"]:
            accuracy = result["accuracy"]
            interval = accuracy["confidence_interval"]
            lines.append(
                f"| {result.get('logical_ber', resolved['fault']['probability']):.8g} | {result.get('policy', result.get('condition'))} | {result.get('coverage', 0.0):.6f} | "
                f"{result.get('fault_realization_count', accuracy['count'])} | {accuracy['mean']:.6f} | {accuracy['standard_deviation']:.6f} | "
                f"[{interval[0]:.6f}, {interval[1]:.6f}] | {accuracy.get('median', accuracy['mean']):.6f} | "
                f"{accuracy.get('quantile_05', accuracy['mean']):.6f} | {accuracy.get('minimum', accuracy['mean']):.6f} | "
                f"{result['categorical_crossentropy']['mean']:.6f} |"
            )
        lines.extend([
            "", "### Paired policy comparisons", "",
            "Pairs are matched by replication, logical BER, trial, and shared schedule SHA-256. Holm adjustment is applied within each BER and metric family.", "",
            "| BER | Coverage | Metric | Comparison | Matched pairs | Mean difference | 95% CI | Raw p | Holm p |",
            "|---:|---:|---|---|---:|---:|---:|---:|---:|",
        ])
        for comparison in summary.get("paired_comparisons", []):
            difference = comparison["difference"]
            interval = difference["confidence_interval"]
            lines.append(
                f"| {comparison['logical_ber']:.8g} | {comparison['coverage']:.6f} | {comparison['metric']} | "
                f"{comparison['left']} − {comparison['right']} | {comparison['matched_pairs']} | {difference['mean']:.6f} | "
                f"[{interval[0]:.6f}, {interval[1]:.6f}] | {comparison['p_value_raw']:.6g} | {comparison['p_value_holm']:.6g} |"
            )
        lines.extend([
            "", "Full pooled, replication-level, and per-pair values are stored in `measurements/protection_results.csv`, `measurements/replication_protection_results.csv`, `measurements/paired_comparisons.csv`, `measurements/paired_replication_summaries.csv`, and `measurements/paired_differences.csv`.",
            "No superiority claim is made automatically from point estimates or marginal confidence intervals.",
            "", "### Fault multiplicity", "",
            "Event counts are measured from the saved schedules and protection accounting; brackets show empirical 5th–95th percentiles.", "",
            "| BER | Policy | Coverage | Proposed events | Applied events | Suppressed events | Unique applied bits |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ])
        for result in summary["protection_results"]:
            def event_cell(name):
                value = result[name]
                return f"{value['mean']:.2f} [{value['quantile_05']:.0f}, {value['quantile_95']:.0f}]"
            lines.append(
                f"| {result['logical_ber']:.8g} | {result['policy']} | {result['coverage']:.6f} | "
                f"{event_cell('proposed_events')} | {event_cell('applied_events')} | {event_cell('suppressed_events')} | {event_cell('unique_applied_bits')} |"
            )
        if str(summary.get("benchmark_id", "")).startswith("mnist"):
            near_chance = [result for result in summary["protection_results"] if result["accuracy"]["mean"] <= 0.15]
            if near_chance:
                labels = ", ".join(f"BER {row['logical_ber']:.3g}/{row['condition']}" for row in near_chance)
                lines.extend([
                    "", f"Exploratory saturation flag: `{labels}` have mean accuracy within five percentage points of uniform 10-class chance accuracy (0.10). This descriptive flag is not a hard accuracy floor or a physical reliability threshold.",
                ])
    if summary.get("estimator_runtimes_seconds"):
        lines.extend(["", "## Estimator execution time", "", "| Estimator | Wall time (seconds) |", "|---|---:|"])
        for name, runtime in summary["estimator_runtimes_seconds"].items():
            lines.append(f"| {name} | {runtime:.6f} |")
        lines.extend(["", "Wall-clock time is execution metadata and is excluded from the scientific payload digest."])
    if summary.get("replication_count", 1) > 1:
        lines.extend([
            "", "## Independent replications", "",
            f"This bundle contains `{summary['replication_count']}` explicitly seeded replications. Candidate sampling, Hessian probes, sample offsets, and fault seeds remain recorded as separate factors.",
            "", "| Estimator | Scope | Metric | Mean | Standard deviation | Replications |",
            "|---|---|---|---:|---:|---:|",
        ])
        for row in summary.get("replication_aggregate", []):
            if row["scope"] == "overall" or row["metric"] == "runtime_seconds":
                lines.append(f"| {row['estimator']} | {row['scope']} | {row['metric']} | {row['mean']:.6f} | {row['standard_deviation']:.6f} | {row['replication_count']} |")
        lines.extend([
            "", "Dispersion across candidate samples is not fault-trial uncertainty or Hutchinson probe uncertainty. Per-layer and per-bit-position replication summaries are stored in `measurements/replication_aggregate.csv`.",
        ])
        if summary.get("replication_protection_results"):
            lines.extend([
                "", "### Replication-level protection summaries", "",
                "These rows preserve between-replication variation separately from pooled fault-trial uncertainty. Identical masks across replications do not constitute independent ranking-mask samples.", "",
                "| Replication | BER | Policy | Coverage | Trials | Mean accuracy | SD | Minimum |",
                "|---|---:|---|---:|---:|---:|---:|---:|",
            ])
            for row in summary["replication_protection_results"]:
                lines.append(
                    f"| {row['replication']} | {row['logical_ber']:.8g} | {row['policy']} | {row['coverage']:.6f} | "
                    f"{row['fault_realization_count']} | {row['accuracy']['mean']:.6f} | {row['accuracy']['standard_deviation']:.6f} | {row['accuracy']['minimum']:.6f} |"
                )
    if figure_paths:
        lines.extend(["", "## Figures", ""])
        for path in figure_paths:
            lines.extend([f"![M3 diagnostic figure]({path})", ""])
    lines.extend([
        "", "## Interpretation limits", "",
        "Logical BER is not a radiation-calibrated SEU rate. Isolated-bit susceptibility is not equivalent to multi-fault protection benefit. Detection is not correction, and ideal event suppression is not synthesized hardware.",
        "Recovered `legacy_qpolargrad` is preserved as a historical heuristic; the clean and uncancelled variants are explicitly separate research candidates.",
        "Second-order estimators are local approximations and may be inaccurate for large sign-bit perturbations.",
        "", "## Reproduction", "", "```bash", f"netsurf run {resolved.get('source_configuration', 'config.toml')}", "```", "",
        "The exact submitted configuration, explicit sample indices and hashes, estimator versions, masks, schedules, measurements, and checksums are included in this immutable bundle.", "",
        f"Scientific payload SHA-256: `{summary['scientific_payload_sha256']}`", "",
    ])
    return "\n".join(lines)
