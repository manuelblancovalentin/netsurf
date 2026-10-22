"""Plots derived only from finalized numerical measurements."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


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
    if summary.get("protection_results"):
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
    if trial_rows:
        policies = sorted({str(row["policy"]) for row in trial_rows if row["policy"] not in {"no_protection", "full_protection"}})
        baseline = {int(row["trial"]): float(row["accuracy"]) for row in trial_rows if row["policy"] == "no_protection"}
        if baseline and policies:
            destination = output / "paired_accuracy_differences.png"
            figure, axis = plt.subplots(figsize=(max(7.0, len(policies) * 1.0), 4.8))
            for index, policy in enumerate(policies):
                differences = [float(row["accuracy"]) - baseline[int(row["trial"])] for row in trial_rows if row["policy"] == policy]
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
