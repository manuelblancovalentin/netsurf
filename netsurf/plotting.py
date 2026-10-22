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
    return (destination,)
