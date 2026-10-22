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
