from __future__ import annotations

from dataclasses import replace
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from netsurf import __main__ as cli
from netsurf.api import run_experiment
from netsurf.progress import LiveProgressReporter, ProgressEvent
from netsurf.study_config import StudyExecution, StudyPolicy, StudyProtection, StudyReplications, load_study
from netsurf.study_engine import execute_study


ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "config/experiments/numpy_qpolar_cancellation.toml"


def _small_protection_study(*, replications: int = 1):
    configuration = load_study(CONTROL)
    estimator = replace(
        configuration.estimators[0], name="msb_first",
        implementation_version="encoded-bit-index/v1", result_name="msb_first",
    )
    return replace(
        configuration,
        name="progress_protection",
        scientific_protocol="multi-coverage-protection/v1",
        study_type="protection",
        estimators=(estimator,),
        policies=(StudyPolicy("msb+ordered", "msb_first", ("bit_significance_desc", "canonical"), None, True),),
        protection=StudyProtection((0.1,), "perfect_event_suppression", (0.01, 0.02)),
        execution=StudyExecution(2, 0.95, 701, 801),
        replications=StudyReplications(replications, 1, 1, 0, 100),
    )


def test_progress_reaches_100_and_does_not_change_scientific_hash():
    configuration = load_study(CONTROL)
    expected = execute_study(configuration)
    events = []
    observed = execute_study(configuration, progress=events.append)
    assert observed.scientific_payload_sha256 == expected.scientific_payload_sha256
    assert events[-1].completed == events[-1].total
    assert events[-1].completed > 0
    empirical = [event for event in events if event.stage == "Empirical bit injection"]
    assert empirical[-1].stage_completed == empirical[-1].stage_total == 103


def test_progress_renderer_failure_cannot_interrupt_scientific_execution():
    configuration = load_study(CONTROL)
    expected = execute_study(configuration)

    def broken_renderer(event):
        raise OSError("closed progress stream")

    observed = execute_study(configuration, progress=broken_renderer)
    assert observed.scientific_payload_sha256 == expected.scientific_payload_sha256


def test_replicated_multi_ber_progress_reports_actual_condition_and_trial():
    events = []
    outcome = execute_study(_small_protection_study(replications=2), progress=events.append)
    assert len(outcome.replications) == 2
    assert events[-1].completed == events[-1].total
    assert any(
        event.replication == 2
        and event.replications == 2
        and event.logical_ber == 0.02
        # Ten of 102 bits are selected: report realized, not nominal, coverage.
        and event.coverage == pytest.approx(10 / 102)
        and event.policy == "msb+ordered@0.100000"
        and event.trial == 2
        and event.trials == 2
        for event in events
    )


def test_redirected_progress_is_plain_text_without_ansi():
    stream = StringIO()
    times = iter((0.0, 0.0, 1.0, 2.0))
    reporter = LiveProgressReporter(
        stream, interactive=False, refresh_seconds=0.0, clock=lambda: next(times)
    )
    reporter(ProgressEvent("example", "Fault evaluation", 0, 2, 0, 2))
    reporter(ProgressEvent("example", "Fault evaluation", 1, 2, 1, 2, trial=1, trials=2))
    reporter(ProgressEvent("example", "Fault evaluation", 2, 2, 2, 2, trial=2, trials=2))
    text = stream.getvalue()
    assert "\x1b" not in text
    assert "100.0%" in text


def test_no_progress_cli_option_suppresses_reporter(monkeypatch, capsys, tmp_path):
    destination = tmp_path / "run"
    captured = {}

    def fake_run(configuration, *, progress=None):
        captured["progress"] = progress
        return destination

    monkeypatch.setattr(cli, "run_experiment", fake_run)
    monkeypatch.setattr(cli, "read_result_bundle", lambda path: SimpleNamespace(scientific_payload_sha256="abc"))
    assert cli.main(["run", "unused.toml", "--no-progress"]) == 0
    assert captured["progress"] is None
    assert "NetSuRF |" not in capsys.readouterr().err


def test_run_api_reports_failure_and_preserves_exception(monkeypatch):
    configuration = load_study(CONTROL)
    events = []
    expected = RuntimeError("deliberate failure")

    def fail(*args, **kwargs):
        raise expected

    monkeypatch.setattr("netsurf.api.execute_study", fail)
    with pytest.raises(RuntimeError) as caught:
        run_experiment(configuration, progress=events.append)
    assert caught.value is expected
    assert events[-1].failed
    assert events[-1].message == "deliberate failure"
