from __future__ import annotations

from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from netsurf import __main__ as cli
from netsurf.progress import ExperimentQueueItem, HierarchicalProgressReporter, ProgressEvent


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value


def _event(name, completed, total, *, stage="Fault evaluation", failed=False, message=None):
    return ProgressEvent(
        name, stage, completed, total, replication=1, replications=2,
        logical_ber=0.01, coverage=0.1, policy="gradient_global",
        trial=completed, trials=total, failed=failed, message=message,
        work_class="Fault evaluation",
        remaining_work=(("Fault evaluation", max(0, total-completed)),),
    )


def test_queue_eta_uses_measured_work_classes_across_pending_experiments():
    clock = FakeClock(); stream = StringIO()
    items = (
        ExperimentQueueItem("first", "first.toml", {"Fault evaluation": 2}),
        ExperimentQueueItem("second", "second.toml", {"Fault evaluation": 2}),
    )
    reporter = HierarchicalProgressReporter(items, stream, interactive=False, refresh_seconds=0, clock=clock)
    reporter.start_experiment(0); reporter.update(0, _event("first", 0, 2))
    clock.value = 2.0; reporter.update(0, _event("first", 1, 2))
    assert reporter._queue_eta_seconds(clock.value) == pytest.approx(6.0)
    reporter.complete_experiment(0)
    reporter.start_experiment(1)
    assert reporter.states[0].status == "completed"
    assert reporter.states[1].status == "running"
    reporter.finish()


def test_plain_queue_output_has_no_ansi_and_retains_failure_status():
    clock = FakeClock(); stream = StringIO()
    reporter = HierarchicalProgressReporter(
        (ExperimentQueueItem("broken", "broken.toml", {"Model loading": 1}),),
        stream, interactive=False, refresh_seconds=0, clock=clock,
    )
    reporter.start_experiment(0)
    reporter.update(0, ProgressEvent("broken", "Model loading", 0, 1, failed=True, message="boom"))
    reporter.finish()
    text = stream.getvalue()
    assert "\x1b" not in text
    assert "Failed broken: boom" in text or "failed" in text.lower()
    assert "0 completed, 1 failed" in text
    assert reporter.states[0].completed == 0


def test_rich_queue_uses_one_live_and_renders_within_narrow_width():
    rich = pytest.importorskip("rich")
    clock = FakeClock(); stream = StringIO()
    reporter = HierarchicalProgressReporter(
        (
            ExperimentQueueItem("a-very-long-first-experiment-name", "one.toml", {"Fault evaluation": 2}),
            ExperimentQueueItem("second", "two.toml", {"Fault evaluation": 1}),
        ),
        stream, interactive=True, width=52, refresh_seconds=0, clock=clock,
    )
    live_identity = id(reporter._live)
    reporter.start_experiment(0)
    clock.value = 1; reporter.update(0, _event("first", 1, 2))
    clock.value = 2; reporter.complete_experiment(0)
    reporter.start_experiment(1)
    reporter.update(1, _event("second", 0, 1, failed=True, message="deliberate"))
    reporter.finish()
    assert id(reporter._live) == live_identity
    assert reporter.states[0].status == "completed"
    assert reporter.states[1].status == "failed"
    assert "NetSuRF experiment queue" in stream.getvalue()
    plain = StringIO()
    console = rich.console.Console(file=plain, width=52, force_terminal=False)
    console.print(reporter._renderable(clock.value))
    assert max(map(len, plain.getvalue().splitlines())) <= 52
    assert "Runtime" in plain.getvalue()
    assert "00:02" in plain.getvalue()


def test_batch_runs_sequentially_continues_after_expected_failure(monkeypatch, capsys, tmp_path):
    configurations = {name: SimpleNamespace(name=name) for name in ("first", "broken", "third")}
    calls = []

    def load(path):
        return configurations[Path(path).stem]

    def run(configuration, *, progress=None):
        calls.append(configuration.name)
        if configuration.name == "broken":
            raise ValueError("deliberate batch failure")
        return tmp_path / configuration.name

    monkeypatch.setattr(cli, "load_configuration", load)
    monkeypatch.setattr(cli, "configuration_operation_plan", lambda config: {"work": 1})
    monkeypatch.setattr(cli, "run_experiment", run)
    monkeypatch.setattr(cli, "read_result_bundle", lambda path: SimpleNamespace(scientific_payload_sha256=f"hash-{path.name}"))
    code = cli.main(["batch", "first.toml", "broken.toml", "third.toml", "--no-progress"])
    captured = capsys.readouterr()
    assert code == 2
    assert calls == ["first", "broken", "third"]
    assert '"completed": 2' in captured.out and '"failed": 1' in captured.out
    assert "2 completed, 1 failed" in captured.err


def test_batch_help_is_exposed():
    parser = cli._parser()
    args = parser.parse_args(["batch", "one.toml", "two.toml", "--no-progress"])
    assert args.command == "batch" and len(args.configurations) == 2
