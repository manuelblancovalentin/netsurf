"""Plots derived only from finalized numerical measurements."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .derived_analysis import normalize_trials
from .statistics import summarize


class PlottingDependencyError(RuntimeError):
    """The optional plotting backend is unavailable."""


def plot_protection_results(summary: Mapping[str, Any], output: Path) -> tuple[Path, ...]:
    """Plot policy accuracy and trial-based confidence intervals."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise PlottingDependencyError(
            "plot generation requires matplotlib>=3.8,<4; ask the environment owner "
            "to run: python -m pip install -e '.[plot]'"
        ) from error
    results = summary.get("protection_results", {})
    if not results:
        return ()
    names = list(results)
    means = [results[name]["accuracy"]["mean"] for name in names]
    low = [means[i] - results[name]["accuracy"]["confidence_interval"][0] for i, name in enumerate(names)]
    high = [results[name]["accuracy"]["confidence_interval"][1] - means[i] for i, name in enumerate(names)]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "accuracy_by_policy.png"
    figure, axis = plt.subplots(figsize=(max(6.4, len(names) * 1.7), 4.8))
    axis.errorbar(names, means, yerr=[low, high], fmt="o", capsize=5)
    axis.set_ylabel("Accuracy (fraction)")
    axis.set_xlabel("Protection policy")
    axis.set_title(f"{summary['experiment_name']}: paired fault-trial results")
    axis.set_ylim(0.0, 1.0)
    axis.grid(axis="y", alpha=0.25)
    axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)
    generated = [destination]

    loss_destination = output / "loss_by_policy.png"
    loss_means = [results[name]["categorical_crossentropy"]["mean"] for name in names]
    loss_low = [loss_means[i] - results[name]["categorical_crossentropy"]["confidence_interval"][0] for i, name in enumerate(names)]
    loss_high = [results[name]["categorical_crossentropy"]["confidence_interval"][1] - loss_means[i] for i, name in enumerate(names)]
    figure, axis = plt.subplots(figsize=(max(6.4, len(names) * 1.7), 4.8))
    axis.errorbar(names, loss_means, yerr=[loss_low, loss_high], fmt="o", capsize=5)
    axis.set_ylabel("Categorical cross-entropy (mean per inference)")
    axis.set_xlabel("Protection policy")
    axis.set_title(f"{summary['experiment_name']}: loss by policy")
    axis.grid(axis="y", alpha=0.25)
    axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    figure.savefig(loss_destination, dpi=160)
    plt.close(figure)
    generated.append(loss_destination)

    comparisons = summary.get("paired_comparisons", {})
    if comparisons:
        benefit_destination = output / "accuracy_benefit_vs_unprotected.png"
        comparison_names = list(comparisons)
        benefit = [comparisons[name]["accuracy_difference"]["mean"] for name in comparison_names]
        low = [benefit[i] - comparisons[name]["accuracy_difference"]["confidence_interval"][0] for i, name in enumerate(comparison_names)]
        high = [comparisons[name]["accuracy_difference"]["confidence_interval"][1] - benefit[i] for i, name in enumerate(comparison_names)]
        figure, axis = plt.subplots(figsize=(max(6.4, len(comparison_names) * 1.7), 4.8))
        axis.errorbar(comparison_names, benefit, yerr=[low, high], fmt="o", capsize=5)
        axis.axhline(0.0, color="black", linewidth=1)
        axis.set_ylabel("Accuracy difference vs no protection")
        axis.set_xlabel("Ranking method")
        axis.set_title(f"{summary['experiment_name']}: paired protection benefit")
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", rotation=20)
        figure.tight_layout()
        figure.savefig(benefit_destination, dpi=160)
        plt.close(figure)
        generated.append(benefit_destination)
    return tuple(generated)


def plot_study_results(
    summary: Mapping[str, Any],
    output: Path,
    *,
    empirical_rows: tuple[Mapping[str, Any], ...] = (),
    prediction_rows: tuple[Mapping[str, Any], ...] = (),
    trial_rows: tuple[Mapping[str, Any], ...] = (),
) -> tuple[Path, ...]:
    """Generate M3 diagnostics from serialized measurements only."""

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise PlottingDependencyError(
            "plot generation requires matplotlib>=3.8,<4; ask the environment owner "
            "to run: python -m pip install -e '.[plot]'"
        ) from error
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    metrics = summary.get("ranking_metrics_overall", [])
    if metrics:
        destination = output / "ranking_correlation.png"
        names = [row["estimator"] for row in metrics]
        correlations = [float("nan") if row["spearman_positive_damage"] is None else row["spearman_positive_damage"] for row in metrics]
        figure, axis = plt.subplots(figsize=(max(6.4, len(names) * 1.2), 4.8))
        axis.bar(names, correlations)
        axis.axhline(0.0, color="black", linewidth=1)
        axis.set_ylabel("Spearman correlation")
        axis.set_title("Estimator score versus exact positive loss damage")
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        figure.savefig(destination, dpi=160)
        plt.close(figure)
        generated.append(destination)

        scoped = summary.get("ranking_metrics_by_scope", [])
        for prefix, filename, title in (
            ("layer:", "ranking_correlation_by_tensor.png", "Positive-damage rank correlation by TFLite kernel tensor"),
            ("bit:", "ranking_correlation_by_bit.png", "Positive-damage rank correlation by stored bit position"),
        ):
            selected = [row for row in scoped if str(row.get("scope", "")).startswith(prefix)]
            if not selected:
                continue
            estimators = list(dict.fromkeys(str(row["estimator"]) for row in selected))
            scopes = list(dict.fromkeys(str(row["scope"])[len(prefix):] for row in selected))
            if prefix == "bit:":
                scopes.sort(key=int)
            else:
                scopes.sort()
            lookup = {
                (str(row["estimator"]), str(row["scope"])[len(prefix):]): (
                    float("nan")
                    if row.get("spearman_positive_damage") in {None, "", "None"}
                    else float(row["spearman_positive_damage"])
                )
                for row in selected
            }
            matrix = [[lookup.get((estimator, scope), float("nan")) for scope in scopes] for estimator in estimators]
            destination = output / filename
            figure, axis = plt.subplots(
                figsize=(max(7.0, 0.9 * len(scopes)), max(4.8, 0.65 * len(estimators)))
            )
            image = axis.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm", aspect="auto")
            axis.set_xticks(range(len(scopes)), scopes, rotation=35, ha="right")
            axis.set_yticks(range(len(estimators)), estimators)
            axis.set_xlabel("TFLite kernel tensor" if prefix == "layer:" else "Stored INT8 bit position (7 is sign)")
            axis.set_title(title)
            figure.colorbar(image, ax=axis, label="Spearman correlation")
            figure.tight_layout()
            figure.savefig(destination, dpi=160)
            plt.close(figure)
            generated.append(destination)

        destination = output / "top_k_recall.png"
        figure, axis = plt.subplots(figsize=(max(6.4, len(names) * 1.2), 4.8))
        for row in metrics:
            axis.plot([5, 10, 20], [row["top_5_recall"], row["top_10_recall"], row["top_20_recall"]], marker="o", label=row["estimator"])
        axis.set_xlabel("k within measured candidate set")
        axis.set_ylabel("Top-k damaging-bit recall")
        axis.set_ylim(0.0, 1.0)
        axis.set_title("Sample-limited ranking recall")
        axis.grid(alpha=0.25)
        axis.legend(fontsize="small")
        figure.tight_layout()
        figure.savefig(destination, dpi=160)
        plt.close(figure)
        generated.append(destination)

        destination = output / "estimator_runtime.png"
        runtimes = [row["runtime_seconds"] for row in metrics]
        figure, axis = plt.subplots(figsize=(max(6.4, len(names) * 1.2), 4.8))
        axis.bar(names, runtimes)
        axis.set_ylabel("Wall time (seconds)")
        axis.set_title("Estimator computation time")
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        figure.savefig(destination, dpi=160)
        plt.close(figure)
        generated.append(destination)
    if empirical_rows and prediction_rows:
        actual = {
            (row["parameter"], int(row["flat_index"]), int(row["bit_index"])): float(row["positive_loss_damage"])
            for row in empirical_rows
        }
        estimator_names = sorted({str(row["estimator"]) for row in prediction_rows})
        destination = output / "predicted_vs_measured_damage.png"
        figure, axes = plt.subplots(1, len(estimator_names), figsize=(max(6.4, 4.2 * len(estimator_names)), 4.2), squeeze=False)
        for axis, name in zip(axes[0], estimator_names):
            rows = [row for row in prediction_rows if row["estimator"] == name]
            pairs = [(float(row["score"]), actual[(row["parameter"], int(row["flat_index"]), int(row["bit_index"]))]) for row in rows if (row["parameter"], int(row["flat_index"]), int(row["bit_index"])) in actual]
            if pairs:
                axis.scatter([value[0] for value in pairs], [value[1] for value in pairs], s=10, alpha=0.65)
            axis.set_title(name)
            axis.set_xlabel("Estimator score")
            axis.set_ylabel("Exact positive loss damage")
            axis.grid(alpha=0.2)
        figure.tight_layout()
        figure.savefig(destination, dpi=160)
        plt.close(figure)
        generated.append(destination)
    task_results = summary.get("protection_results", [])
    if task_results and "primary_metric" in task_results[0]:
        names = [row["policy"] for row in task_results]
        for metric_key, name_key, filename in (
            ("primary_metric", "primary_metric_name", "task_primary_metric_by_policy.png"),
            ("loss_metric", "loss_metric_name", "task_loss_metric_by_policy.png"),
        ):
            means = [row[metric_key]["mean"] for row in task_results]
            low = [mean - row[metric_key]["confidence_interval"][0] for mean, row in zip(means, task_results)]
            high = [row[metric_key]["confidence_interval"][1] - mean for mean, row in zip(means, task_results)]
            destination = output / filename
            figure, axis = plt.subplots(figsize=(max(7.0, len(names) * 1.1), 4.8))
            axis.errorbar(names, means, yerr=[low, high], fmt="o", capsize=4)
            axis.set_ylabel(str(task_results[0][name_key]))
            axis.set_title(f"{task_results[0][name_key]} by protection policy")
            axis.tick_params(axis="x", rotation=30)
            axis.grid(axis="y", alpha=0.25)
            figure.tight_layout()
            figure.savefig(destination, dpi=160)
            plt.close(figure)
            generated.append(destination)
        return tuple(generated)
    if summary.get("protection_results") and not trial_rows:
        results = summary["protection_results"]
        destination = output / "accuracy_by_policy_coverage.png"
        names = [row["policy"] for row in results]
        means = [row["accuracy"]["mean"] for row in results]
        low = [means[index] - row["accuracy"]["confidence_interval"][0] for index, row in enumerate(results)]
        high = [row["accuracy"]["confidence_interval"][1] - means[index] for index, row in enumerate(results)]
        figure, axis = plt.subplots(figsize=(max(7.0, len(names) * 1.1), 4.8))
        axis.errorbar(names, means, yerr=[low, high], fmt="o", capsize=4)
        axis.set_ylim(0.0, 1.0)
        axis.set_ylabel("Accuracy (fraction)")
        axis.set_title("Matched logical-fault trials by policy and coverage")
        axis.tick_params(axis="x", rotation=30)
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        figure.savefig(destination, dpi=160)
        plt.close(figure)
        generated.append(destination)
        destination = output / "loss_by_policy_coverage.png"
        losses = [row["categorical_crossentropy"]["mean"] for row in results]
        loss_low = [losses[index] - row["categorical_crossentropy"]["confidence_interval"][0] for index, row in enumerate(results)]
        loss_high = [row["categorical_crossentropy"]["confidence_interval"][1] - losses[index] for index, row in enumerate(results)]
        figure, axis = plt.subplots(figsize=(max(7.0, len(names) * 1.1), 4.8))
        axis.errorbar(names, losses, yerr=[loss_low, loss_high], fmt="o", capsize=4)
        axis.set_ylabel("Categorical cross-entropy (mean per inference)")
        axis.set_title("Matched logical-fault loss by policy and coverage")
        axis.tick_params(axis="x", rotation=30)
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        figure.savefig(destination, dpi=160)
        plt.close(figure)
        generated.append(destination)
    if trial_rows and summary.get("protection_results"):
        pooled = summary["protection_results"]
        default_ber = next((row.get("logical_ber") for row in pooled if row.get("logical_ber") is not None), None)
        rows = normalize_trials(trial_rows, default_ber=default_ber)
        bers = sorted({row["logical_ber"] for row in rows})
        policies = sorted({row["ranking_policy"] for row in rows if row["ranking_policy"] not in {"no_protection", "full_protection"}})
        coverages = sorted({row["coverage"] for row in rows if row["ranking_policy"] not in {"no_protection", "full_protection"}})

        for ber in bers:
            destination = output / f"accuracy_vs_coverage_ber_{ber:.0e}.png"
            figure, axis = plt.subplots(figsize=(7.2, 4.8))
            for policy in policies:
                selected = [row for row in pooled if row["logical_ber"] == ber and row["policy"] == policy]
                selected.sort(key=lambda row: row["coverage"])
                if not selected:
                    continue
                means = [row["accuracy"]["mean"] for row in selected]
                low = [mean - row["accuracy"]["confidence_interval"][0] for mean, row in zip(means, selected)]
                high = [row["accuracy"]["confidence_interval"][1] - mean for mean, row in zip(means, selected)]
                axis.errorbar([row["coverage"] for row in selected], means, yerr=[low, high], marker="o", capsize=3, label=policy)
            for control, style in (("no_protection", "--"), ("full_protection", ":")):
                selected = next((row for row in pooled if row["logical_ber"] == ber and row["policy"] == control), None)
                if selected:
                    axis.axhline(selected["accuracy"]["mean"], linestyle=style, linewidth=1.2, label=f"{control} (n={selected['fault_realization_count']})")
            axis.set(xlabel="Protected eligible bits (fraction)", ylabel="Mean classification accuracy", title=f"Accuracy versus protection coverage — logical BER {ber:.1e}", ylim=(0.0, 1.0))
            axis.grid(alpha=0.25); axis.legend(fontsize="small")
            figure.tight_layout(); figure.savefig(destination, dpi=160); plt.close(figure); generated.append(destination)

        if len(bers) > 1:
            for coverage in coverages:
                destination = output / f"accuracy_vs_ber_coverage_{coverage:.6f}.png"
                figure, axis = plt.subplots(figsize=(7.2, 4.8))
                for policy in policies:
                    selected = [row for row in pooled if row["policy"] == policy and abs(row["coverage"] - coverage) < 1e-9]
                    selected.sort(key=lambda row: row["logical_ber"])
                    if not selected:
                        continue
                    means = [row["accuracy"]["mean"] for row in selected]
                    errors = [[mean - row["accuracy"]["confidence_interval"][0] for mean, row in zip(means, selected)], [row["accuracy"]["confidence_interval"][1] - mean for mean, row in zip(means, selected)]]
                    axis.errorbar([row["logical_ber"] for row in selected], means, yerr=errors, marker="o", capsize=3, label=policy)
                unprotected = sorted((row for row in pooled if row["policy"] == "no_protection"), key=lambda row: row["logical_ber"])
                if unprotected:
                    means = [row["accuracy"]["mean"] for row in unprotected]
                    errors = [[mean - row["accuracy"]["confidence_interval"][0] for mean, row in zip(means, unprotected)], [row["accuracy"]["confidence_interval"][1] - mean for mean, row in zip(means, unprotected)]]
                    axis.errorbar([row["logical_ber"] for row in unprotected], means, yerr=errors, marker="x", linestyle="--", capsize=3, label="no protection")
                clean = summary.get("clean_evaluation", {}).get("accuracy")
                if clean is not None:
                    axis.axhline(clean, color="black", linewidth=1, linestyle=":", label="clean deterministic reference")
                axis.set_xscale("log"); axis.set_ylim(0.0, 1.0)
                axis.set(xlabel="Logical bit-flip probability (BER)", ylabel="Mean classification accuracy", title=f"Accuracy versus logical BER — coverage {coverage:.1%}")
                axis.grid(alpha=0.25); axis.legend(fontsize="small")
                figure.tight_layout(); figure.savefig(destination, dpi=160); plt.close(figure); generated.append(destination)

        for ber in bers:
            selected_conditions = sorted({row["condition"] for row in rows if row["logical_ber"] == ber})
            accuracy_values = [[row["accuracy"] for row in rows if row["logical_ber"] == ber and row["condition"] == condition] for condition in selected_conditions]
            destination = output / f"trial_accuracy_distributions_ber_{ber:.0e}.png"
            figure, axis = plt.subplots(figsize=(max(8.0, len(selected_conditions) * 0.65), 5.2))
            axis.boxplot(accuracy_values, tick_labels=selected_conditions, showfliers=True)
            axis.set_ylim(0.0, 1.0); axis.set_ylabel("Trial classification accuracy")
            axis.set_title(f"Fault-trial accuracy distributions — logical BER {ber:.1e}")
            axis.tick_params(axis="x", rotation=45); axis.grid(axis="y", alpha=0.25)
            figure.tight_layout(); figure.savefig(destination, dpi=160); plt.close(figure); generated.append(destination)

            loss_values = [[row["categorical_crossentropy"] for row in rows if row["logical_ber"] == ber and row["condition"] == condition] for condition in selected_conditions]
            destination = output / f"trial_loss_distributions_ber_{ber:.0e}.png"
            figure, axis = plt.subplots(figsize=(max(8.0, len(selected_conditions) * 0.65), 5.2))
            axis.boxplot(loss_values, tick_labels=selected_conditions, showfliers=True)
            if all(value > 0 for group in loss_values for value in group):
                axis.set_yscale("log")
            axis.set_ylabel("Categorical cross-entropy (log scale when positive)")
            axis.set_title(f"Fault-trial loss distributions — logical BER {ber:.1e}; outliers shown")
            axis.tick_params(axis="x", rotation=45); axis.grid(axis="y", alpha=0.25)
            figure.tight_layout(); figure.savefig(destination, dpi=160); plt.close(figure); generated.append(destination)

        preferred = (("gradient_global", "msb+gradient"), ("gradient_global", "msb+ordered"))
        paired_sets = []
        indexed = {(row["replication"], row["logical_ber"], row["coverage"], row["ranking_policy"], row["trial"], row["schedule_sha256"]): row for row in rows}
        for ber in bers:
            for coverage in coverages:
                for left, right in preferred:
                    left_rows = [row for row in rows if row["logical_ber"] == ber and row["coverage"] == coverage and row["ranking_policy"] == left]
                    values = []
                    for row in left_rows:
                        match = indexed.get((row["replication"], ber, coverage, right, row["trial"], row["schedule_sha256"]))
                        if match is not None:
                            values.append(row["accuracy"] - match["accuracy"])
                    if values:
                        paired_sets.append((f"{left}−{right}\nBER {ber:.0e}, cov {coverage:.1%}", values))
        if paired_sets:
            destination = output / "paired_policy_accuracy_differences.png"
            figure, axis = plt.subplots(figsize=(max(8.0, len(paired_sets) * 0.8), 5.2))
            axis.boxplot([values for _, values in paired_sets], tick_labels=[name for name, _ in paired_sets], showfliers=True)
            estimates = [summarize(values, bounds=(-1.0, 1.0)) for _, values in paired_sets]
            means = [value.mean for value in estimates]
            axis.errorbar(
                range(1, len(estimates) + 1), means,
                yerr=[[mean - value.confidence_interval[0] for mean, value in zip(means, estimates)], [value.confidence_interval[1] - mean for mean, value in zip(means, estimates)]],
                fmt="D", color="black", capsize=3, label="mean and 95% CI",
            )
            axis.axhline(0.0, color="black", linewidth=1)
            axis.set_ylabel("Paired trial accuracy difference"); axis.set_title("Prespecified paired policy differences (matched schedules)")
            axis.tick_params(axis="x", rotation=45); axis.grid(axis="y", alpha=0.25); axis.legend(fontsize="small")
            figure.tight_layout(); figure.savefig(destination, dpi=160); plt.close(figure); generated.append(destination)

    composition = summary.get("mask_composition", [])
    bit_rows = [row for row in composition if row["dimension"] == "bit_position" and row["policy"] not in {"no_protection", "full_protection"}]
    if bit_rows:
        policies = list(dict.fromkeys(row["policy"] for row in bit_rows))
        bits = sorted({row["category"] for row in bit_rows}, key=int, reverse=True)
        values = [[next((row["protected_fraction_of_mask"] for row in bit_rows if row["policy"] == policy and row["category"] == bit), 0.0) for bit in bits] for policy in policies]
        destination = output / "mask_composition_bit_position.png"
        figure, axis = plt.subplots(figsize=(8.0, max(4.8, len(policies) * 0.35)))
        image = axis.imshow(values, aspect="auto", vmin=0.0, vmax=1.0)
        axis.set_xticks(range(len(bits)), [f"b{bit}" for bit in bits])
        axis.set_yticks(range(len(policies)), policies)
        axis.set_xlabel("Encoded bit position")
        axis.set_title("Fraction of each protection mask")
        figure.colorbar(image, ax=axis, label="Fraction")
        figure.tight_layout()
        figure.savefig(destination, dpi=160)
        plt.close(figure)
        generated.append(destination)
    layer_rows = [row for row in composition if row["dimension"] == "layer" and row["policy"] not in {"no_protection", "full_protection"}]
    if layer_rows:
        policies = list(dict.fromkeys(row["policy"] for row in layer_rows))
        layers = sorted({row["category"] for row in layer_rows})
        values = [[next((row["protected_fraction_of_mask"] for row in layer_rows if row["policy"] == policy and row["category"] == layer), 0.0) for layer in layers] for policy in policies]
        destination = output / "mask_composition_layer.png"
        figure, axis = plt.subplots(figsize=(8.0, max(4.8, len(policies) * 0.35)))
        image = axis.imshow(values, aspect="auto", vmin=0.0, vmax=1.0)
        axis.set_xticks(range(len(layers)), layers)
        axis.set_yticks(range(len(policies)), policies)
        axis.set_xlabel("Layer")
        axis.set_title("Fraction of each protection mask")
        figure.colorbar(image, ax=axis, label="Fraction")
        figure.tight_layout()
        figure.savefig(destination, dpi=160)
        plt.close(figure)
        generated.append(destination)
    if trial_rows and not summary.get("protection_results"):
        def condition_and_policy(value: str) -> tuple[str, str]:
            return tuple(value.rsplit("/", 1)) if value.startswith("ber=") else ("", value)

        baselines = {
            (condition_and_policy(str(row["policy"]))[0], int(row["trial"])): float(row["accuracy"])
            for row in trial_rows if condition_and_policy(str(row["policy"]))[1] == "no_protection"
        }
        policies = sorted({
            str(row["policy"]) for row in trial_rows
            if condition_and_policy(str(row["policy"]))[1] not in {"no_protection", "full_protection"}
        })
        if baselines and policies:
            destination = output / "paired_accuracy_differences.png"
            figure, axis = plt.subplots(figsize=(max(7.0, len(policies) * 1.0), 4.8))
            for index, policy in enumerate(policies):
                condition, _ = condition_and_policy(policy)
                differences = [
                    float(row["accuracy"]) - baselines[(condition, int(row["trial"]))]
                    for row in trial_rows if row["policy"] == policy
                ]
                axis.scatter([index] * len(differences), differences, s=18, alpha=0.7)
            axis.axhline(0.0, color="black", linewidth=1)
            axis.set_xticks(range(len(policies)), policies, rotation=30, ha="right")
            axis.set_ylabel("Paired accuracy difference vs no protection")
            axis.set_title("Individual matched fault trials")
            axis.grid(axis="y", alpha=0.25)
            figure.tight_layout()
            figure.savefig(destination, dpi=160)
            plt.close(figure)
            generated.append(destination)
    return tuple(generated)
