from io import StringIO

from rich.console import Console

from scripts import watch_pythia_high_ber as watcher
from scripts.watch_w8a8_wide_ber import Snapshot


def test_progress_counts_only_saved_gates_and_evaluations(tmp_path, monkeypatch):
    root = tmp_path / "pythia"
    (root / "fields/arm_00/field_000").mkdir(parents=True)
    (root / "fields/arm_00/field_000/clean_gate.json").write_text("{}")
    (root / "plan.json").write_text("{}")
    (root / "preflight").mkdir()
    (root / "preflight/SEALED.json").write_text("{}")
    monkeypatch.setattr(watcher, "_session_start", lambda: 100.0)
    monkeypatch.setattr(watcher, "inspect_run", lambda *args, **kwargs: Snapshot(
        "pythia", "Running", 2, 192, 30, 4800, 31, 4992, 50.0, 3600.0))
    item = watcher.inspect(root, now=150.0)
    assert item.stage == "Fault evaluation"
    assert item.evaluations == 30
    assert item.fields == 1  # not the two policy-complete fields reported upstream
    assert item.percent == 0.625
    assert "30/4,800" in watcher.plain_line(item)
    assert "1/192" in watcher.plain_line(item)


def test_narrow_rich_progress_is_readable_and_completion_is_not_running():
    item = watcher.Progress("Fault evaluation", 2400, 4800, 96, 192, 3600, 3700, 4)
    output = StringIO()
    console = Console(file=output, width=55, force_terminal=False)
    console.print(watcher.rich_panel(item, 55))
    assert all(len(line) <= 55 for line in output.getvalue().splitlines())
    assert "2,400/4,800" in output.getvalue()
    done = watcher.Progress("Completed", 4800, 4800, 192, 192, None, None, 4)
    assert "Completed" in watcher.plain_line(done)
    assert "ETA" not in watcher.plain_line(done)
