"""Bounded, synthetic checks for the read-only wide-BER status display."""

import json

from scripts.watch_w8a8_wide_ber import Snapshot, inspect_run, _plain_line, _rich_table


def _save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def test_saved_records_not_queue_messages_determine_progress(tmp_path):
    target = "distilbert_sst2_int8"
    run = tmp_path / target
    _save(run / "plan.json", {
        "bers": [1e-4], "fields_per_ber": 2, "execution_shards": 1,
        "policies": [{"name": "unprotected", "kind": "none"},
                     {"name": "full_protection", "kind": "full"}],
        "counts": {"fields": 2, "planned_faulted_model_evaluations": 2},
    })
    first = run / "fields/arm_00/field_000"
    _save(first / "field.json", {"seed": 1})
    _save(first / "unprotected.json", {"elapsed_seconds": 10.0})
    _save(first / "full_protection.json", {"elapsed_seconds": 0.01})
    second = run / "fields/arm_00/field_001"
    _save(second / "field.json", {"seed": 2})
    _save(second / "unprotected.json", {"elapsed_seconds": 12.0})

    row = inspect_run(tmp_path, target, {target: 900.0}, now=1000.0)
    assert (row.fields, row.evaluations, row.observations) == (1, 2, 3)
    assert row.state == "Running"
    assert row.eta is None  # Too few timings for a credible ETA.
    assert "2/2 evals" in _plain_line(row)

    _save(second / "full_protection.json", {"elapsed_seconds": 0.01})
    assert inspect_run(tmp_path, target, {target: 900.0}).state == "Finalizing"
    assert inspect_run(tmp_path, target, {}).state == "Awaiting finalization"
    _save(run / "manifest.json", {})
    (run / "FINALIZED").write_text("done")
    assert inspect_run(tmp_path, target, {}).state == "Finalized"


def test_eta_uses_persisted_inference_durations_and_parallel_shards(tmp_path):
    target = "pythia160m_int8"
    run = tmp_path / target
    _save(run / "plan.json", {
        "bers": [1e-4], "fields_per_ber": 4, "execution_shards": 2,
        "policies": [{"name": f"p{i}", "kind": "packed"} for i in range(5)]
                    + [{"name": "full_protection", "kind": "full"}],
        "counts": {"fields": 4, "planned_faulted_model_evaluations": 20},
    })
    for index in (0, 1):
        directory = run / f"fields/arm_00/field_{index:03d}"
        _save(directory / "field.json", {})
        for p in range(5):
            _save(directory / f"p{p}.json", {"elapsed_seconds": 4.0})
        _save(directory / "full_protection.json", {"elapsed_seconds": 0.01})
    row = inspect_run(tmp_path, target, {target: 900.0}, now=1000.0)
    assert row.evaluations == 10
    assert row.fields == 2
    assert row.eta == 20.0  # Ten remaining evaluations, split across two shards.
    assert row.elapsed == 100.0


def test_missing_and_stopped_states(tmp_path):
    target = "qwen3_0_6b_int8"
    assert inspect_run(tmp_path, target, {}).state == "Pending"
    run = tmp_path / target
    _save(run / "plan.json", {
        "bers": [1e-4], "fields_per_ber": 1, "execution_shards": 1,
        "policies": [{"name": "unprotected", "kind": "none"}],
        "counts": {"fields": 1, "planned_faulted_model_evaluations": 1},
    })
    directory = run / "fields/arm_00/field_000"
    _save(directory / "field.json", {})
    _save(directory / "unprotected.json", {"elapsed_seconds": 4.0})
    assert inspect_run(tmp_path, target, {}).state == "Awaiting finalization"


def test_rich_compact_table_does_not_wrap_on_70_column_terminal():
    from rich.console import Console
    from io import StringIO

    output = StringIO()
    console = Console(file=output, width=70, color_system=None)
    rows = [Snapshot("distilbert_sst2_int8", "Running", 5, 192, 197, 4800,
                     200, 4992, 1200.0, 20000.0)]
    console.print(_rich_table(rows, width=70))
    assert all(len(line) <= 70 for line in output.getvalue().splitlines())
    assert "197/4800" in output.getvalue()
