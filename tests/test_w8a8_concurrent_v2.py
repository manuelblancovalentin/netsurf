"""Deterministic v2 scientific-equivalence, CPU admission and offline report tests."""

import json
from pathlib import Path
import sys

import pytest

from netsurf.w8a8_campaign import _write_bound


SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))
import freeze_distilbert_w8a8_concurrent_v2 as frozen  # noqa: E402
import w8a8_concurrent_gate_v2 as gate  # noqa: E402
import report_distilbert_w8a8_v2 as reporting  # noqa: E402


def _v1():
    """The frozen v1 plan is an ignored run output (workspace/plans/...); skip where this checkout does not have it."""
    path = frozen.V1 / "plan.json"
    if not path.exists():
        pytest.skip(f"local run output not present: {path.name} of {frozen.V1.name}")
    return json.loads(path.read_text())


def test_v2_changes_only_bound_execution_policy():
    original = _v1()
    derived, audit = frozen.derive(original)
    assert frozen.scientific_payload(derived) == frozen.scientific_payload(original)
    assert derived["fields"] == original["fields"]
    assert derived["selection"] == original["selection"]
    assert derived["final"] == original["final"]
    assert derived["methods"] == original["methods"]
    assert derived["budget_fractions"] == original["budget_fractions"]
    assert derived["source_files_sha256"].items() >= original["source_files_sha256"].items()
    assert audit["scientific_fields_equal"] is True
    assert all(x.startswith("source_files_sha256.") or x.startswith("resource_envelope.") or
               x == "plan_sha256" for x in audit["changed_paths"])
    derived["fields"][0]["seeds"][0] += 1
    assert frozen.scientific_payload(derived) != frozen.scientific_payload(original)


def test_resource_gate_identifies_m11_m12_and_selects_distinct_cores(monkeypatch):
    worker = {"args": ["python", "/tmp/m11_run_both_matched_lambda.py", "--worker", "kws"], "uid": 1}
    followup = {"args": ["python", "-m", "workflows.run_reliability_followup", "--plan",
                         "kws_partial_tier_fkeras_matched_v1"], "uid": 1}
    model = {"args": ["python", "-m", "netsurf.w8a8_campaign", "final"], "uid": 2}
    assert gate._role(worker) == "m11_worker"
    assert gate._role(followup) == "m12_kws_seven_method"
    assert gate._role(model) == "other_model"
    monkeypatch.setattr(gate, "_physical_key", lambda cpu: (0, cpu // 2))
    assert gate._select_cores(set(range(2, 18)), {2, 4}, 4) == [6, 8, 10, 12]
    with pytest.raises(ValueError, match="distinct"):
        gate._select_cores({2, 3, 4}, {2}, 4)


def test_full_field_csv_and_ber_plot_are_field_based(tmp_path, monkeypatch):
    plan = _v1()
    run = tmp_path / "run"
    n = plan["eligible_universe"]["bits"]
    budgets = [0, n, *[int(x) for x in plan["interior_budgets_bits"].values()]]
    policy_names = ["unprotected", "full_protection"] + [
        f"{method}_b{budget}" for method in plan["methods"] for budget in budgets[2:]]
    assert len(policy_names) == 26
    first = None
    for arm, spec in enumerate(plan["fields"]):
        for i, seed in enumerate(spec["seeds"]):
            policies = {}
            for name in policy_names:
                budget = 0 if name == "unprotected" else n if name == "full_protection" else \
                    int(name.rpartition("_b")[2])
                policies[name] = {"budget_bits": budget, "mask_sha256": "a"*64,
                                  "expected_applied_faults": spec["lambda"]*(1-budget/n),
                                  "events": {"field_sha256": "b"*64, "proposed": 2,
                                             "applied": 2 if budget == 0 else 0,
                                             "suppressed": 0 if budget == 0 else 2},
                                  "risk": {"metrics": {"mean_nll": .4 + .001*arm + .0001*i,
                                                       "accuracy": .9}, "failed_units": 0,
                                           "total_units": 512, "logit_statistics_sha256": "c"*64}}
            record = {"plan_sha256": plan["plan_sha256"], "ber": spec["ber"],
                      "seed": seed, "field_sha256": "b"*64, "policies": policies}
            path = run / f"final/arm_{arm:02d}/field_{i:03d}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            _write_bound(path, record)
            first = first or path
    rows = reporting.observations(plan, run)
    assert len(rows) == 1872
    grouped = reporting.conditions(plan, rows, .4)
    assert len(grouped) == 78
    assert all(x["independent_fields"] == 24 for x in grouped)
    assert all(x["mean_proposed_faults"] == 2 for x in grouped)
    svg = reporting._svg(grouped, plan)
    assert "per example positive" in svg and "25.00% protected" in svg
    assert svg.endswith("</svg>\n")
    plan_dir = tmp_path / "plan"
    plan_dir.mkdir()
    (plan_dir / "plan.json").write_text(json.dumps(plan))
    (plan_dir / "equivalence_audit.json").write_text("{}\n")
    _write_bound(run / "analysis.json", {"plan_sha256": plan["plan_sha256"],
                                          "field_rows": [{}]*72,
                                          "clean": {"metrics": {"mean_nll": .4}}})
    (run / "scientific_payload.json").write_text("{}\n")
    run.with_name(run.name + "_admission.json").write_text("{}\n")
    run.with_name(run.name + "_stages.json").write_text("{}\n")
    core = run.with_name(run.name + "_export")
    core.mkdir()
    (core / "checksums.json").write_text("{}\n")
    (core / "FINALIZED").write_text("a"*64+"\n")
    monkeypatch.setattr(reporting, "validate", lambda *args: (plan, None))
    monkeypatch.setattr(reporting, "aggregate", lambda *args: {"fields": 72})
    monkeypatch.setattr(reporting, "export", lambda *args: {
        "status": "VERIFIED_FINALIZED_REUSED", "zip_sha256": "b"*64})
    receipt = reporting.report(plan_dir, run)
    assert receipt["conditions"] == 78 and receipt["field_observations"] == 1872
    import zipfile
    output = run.with_name(run.name + "_reporting_v2")
    with zipfile.ZipFile(output / "complete_v2.zip") as archive:
        assert archive.testzip() is None
        assert "run/final/arm_02/field_023.json" in archive.namelist()
        assert "report/condition_level.csv" in archive.namelist()
    with zipfile.ZipFile(output / "compact.zip") as archive:
        assert archive.testzip() is None
        assert "report/loss_degradation_vs_ber.svg" in archive.namelist()
    assert (output / "FINALIZED").read_text().strip() == receipt["reporting_receipt_sha256"]
    item = json.loads(first.read_text())
    item["policies"]["unprotected"]["events"]["applied"] = 3
    first.write_text(json.dumps(item))
    with pytest.raises(ValueError, match="checksum"):
        reporting.observations(plan, run)
