"""Small public API shared by command-line and notebook workflows."""

from __future__ import annotations

from pathlib import Path

from .experiment_config import ExperimentConfiguration, load_experiment
from .experiment_engine import ExperimentOutcome, execute_experiment
from .plotting import plot_protection_results
from .reporting import write_report
from .result_bundle import ResultBundle, create_result_bundle, read_result_bundle


def run_experiment(
    configuration: ExperimentConfiguration | str | Path,
    *,
    repository_root: Path | None = None,
) -> Path:
    """Validate, execute, and atomically save an experiment."""
    config = load_experiment(configuration) if isinstance(configuration, (str, Path)) else configuration
    outcome = execute_experiment(config)
    root = repository_root or Path(__file__).resolve().parents[1]
    return create_result_bundle(outcome, repository_root=root)


def regenerate_report(
    bundle: ResultBundle | str | Path,
    output: str | Path,
    *,
    plots: bool = False,
) -> Path:
    """Regenerate presentation artifacts without running scientific code."""
    result = read_result_bundle(bundle) if isinstance(bundle, (str, Path)) else bundle
    destination = Path(output).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    figures: tuple[str, ...] = ()
    if plots:
        generated = plot_protection_results(result.summary, destination / "plots")
        figures = tuple(path.relative_to(destination).as_posix() for path in generated)
    return write_report(result.summary, result.resolved_config, destination / "report.md", figure_paths=figures)


__all__ = [
    "ExperimentConfiguration", "ExperimentOutcome", "ResultBundle", "load_experiment",
    "read_result_bundle", "regenerate_report", "run_experiment",
]
