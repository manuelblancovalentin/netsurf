"""Small public API shared by command-line and notebook workflows."""

from __future__ import annotations

from pathlib import Path
import csv
import tomllib

from .experiment_config import ExperimentConfiguration, load_experiment
from .experiment_engine import ExperimentOutcome, execute_experiment
from .plotting import plot_protection_results, plot_study_results
from .reporting import render_study_report, write_report
from .result_bundle import ResultBundle, create_result_bundle, read_result_bundle
from .study_bundle import create_study_bundle
from .study_config import StudyConfiguration, load_study
from .study_engine import StudyOutcome, execute_study


def load_configuration(path: str | Path):
    """Load the explicitly versioned configuration supported by its schema."""

    source = Path(path).expanduser().resolve()
    with source.open("rb") as stream:
        version = tomllib.load(stream).get("schema_version")
    if version in {"3.0.0", "3.1.0"}:
        return load_study(source)
    return load_experiment(source)


def run_experiment(
    configuration: ExperimentConfiguration | str | Path,
    *,
    repository_root: Path | None = None,
) -> Path:
    """Validate, execute, and atomically save an experiment."""
    root = repository_root or Path(__file__).resolve().parents[1]
    config = load_configuration(configuration) if isinstance(configuration, (str, Path)) else configuration
    if isinstance(config, StudyConfiguration):
        return create_study_bundle(execute_study(config), repository_root=root)
    return create_result_bundle(execute_experiment(config), repository_root=root)


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
    is_study = result.resolved_config.get("schema_version") in {"3.0.0", "3.1.0"}
    if plots and is_study:
        def rows(name):
            path = result.path / "measurements" / name
            if not path.is_file():
                return ()
            with path.open(newline="", encoding="utf-8") as stream:
                return tuple(dict(row) for row in csv.DictReader(stream))
        empirical_rows = rows("empirical_bits.csv")
        prediction_rows = rows("estimator_predictions.csv")
        trial_rows = rows("trials.csv")
        # A bundle-level plot depicts one coherent replication; aggregate
        # dispersion is already reported from replication_aggregate.csv.
        # Never join identical bit identities across different replications.
        if any("replication" in row for row in empirical_rows + prediction_rows + trial_rows):
            names = sorted({row.get("replication", "") for row in empirical_rows + prediction_rows + trial_rows})
            selected = names[0]
            empirical_rows = tuple(row for row in empirical_rows if row.get("replication") == selected)
            prediction_rows = tuple(row for row in prediction_rows if row.get("replication") == selected)
            trial_rows = tuple(row for row in trial_rows if row.get("replication") == selected)
        generated = plot_study_results(
            result.summary, destination / "plots",
            empirical_rows=empirical_rows, prediction_rows=prediction_rows,
            trial_rows=trial_rows,
        )
        figures = tuple(path.relative_to(destination).as_posix() for path in generated)
    elif plots:
        generated = plot_protection_results(result.summary, destination / "plots")
        figures = tuple(path.relative_to(destination).as_posix() for path in generated)
    if is_study:
        output_path = destination / "report.md"
        output_path.write_text(render_study_report(result.summary, result.resolved_config, figure_paths=figures), encoding="utf-8")
        return output_path
    return write_report(result.summary, result.resolved_config, destination / "report.md", figure_paths=figures)


__all__ = [
    "ExperimentConfiguration", "ExperimentOutcome", "StudyConfiguration", "StudyOutcome", "ResultBundle",
    "load_configuration", "load_experiment", "load_study", "execute_study",
    "read_result_bundle", "regenerate_report", "run_experiment",
]
