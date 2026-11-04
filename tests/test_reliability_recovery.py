"""Offline recovery must be write-once and independent of model execution."""

import json
import csv
import sqlite3
from pathlib import Path

import pytest

from netsurf.manifests import sha256_file
from workflows import recover_reliability_followup as recovery


def test_staged_publication_is_write_once_and_retains_failed_stage(tmp_path):
    def complete(candidate, final):
        assert not final.exists()
        candidate.mkdir()
        (candidate / "data").write_text("verified")

    final = recovery._publish_staged(tmp_path, "derived", complete)
    assert (final / "data").read_text() == "verified"
    with pytest.raises(FileExistsError, match="write-once"):
        recovery._publish_staged(tmp_path, "derived", complete)
    assert (final / "data").read_text() == "verified"

    def incomplete(candidate, _):
        candidate.mkdir()
        (candidate / "partial").write_text("preserve me")
        raise ValueError("builder failed")

    with pytest.raises(RuntimeError, match="retained for inspection"):
        recovery._publish_staged(tmp_path, "another", incomplete)
    stages = list(tmp_path.glob(".another.stage-*/another/partial"))
    assert len(stages) == 1 and stages[0].read_text() == "preserve me"


def test_recovery_reuses_verified_evidence_and_never_runs_inference(tmp_path, monkeypatch):
    output = tmp_path / "followup"
    output.mkdir()
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "checksums.json").write_text(json.dumps({"files": {"saved": "digest"}}))
    (bundle / "saved").write_text("original evidence")
    source_revision = "frozen-revision"
    scientific_hash = "frozen-scientific-payload"
    preflight = {"model_sha256": "model", "fault_fields": 256}
    (output / "run_receipt.json").write_text(json.dumps({
        "run_directory": str(bundle), "source_git_revision": source_revision,
        "scientific_payload_sha256": scientific_hash, "campaign_preflight": preflight,
    }))
    gate = {"paired_fault_fields": 256}
    (output / "kws_matched_postflight.json").write_text(json.dumps(gate))
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"studies": []}))
    source = {"id": f"{recovery.CAMPAIGN}_followup", "path": str(bundle),
              "kind": "whole_kernel_followup", "scientific_payload_sha256": scientific_hash,
              "checksums_sha256": sha256_file(bundle / "checksums.json")}
    (output / "source_receipt.json").write_text(json.dumps({"studies": [source]}))
    (output / "derived_database").mkdir()
    (output / "derived_database" / "partial.csv").write_text("preserve")
    assets = tmp_path / "assets.json"
    assets.write_text("{}")
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "primary_conditions.csv").write_text("unchanged primary rows")
    (baseline / "checksums.json").write_text(json.dumps({
        "files": {"primary_conditions.csv": sha256_file(baseline / "primary_conditions.csv")}
    }))
    monkeypatch.setattr(recovery, "_verified_bundle", lambda _: (
        bundle, {"source": {"commit": source_revision}}, {"clean_evaluation": {}}, {}
    ))
    monkeypatch.setattr(recovery, "postflight", lambda *_: gate)
    monkeypatch.setattr(recovery, "_audit_saved_fields", lambda *_: {
        "fields": 256, "policy_budget_evaluations": 7680,
        "methods": list(range(7)), "budgets": list(range(4)), "bers": list(range(8)),
    })

    def fake_build(_, __, candidate):
        candidate.mkdir()
        (candidate / "primary_conditions.csv").write_text("unchanged primary rows")
        (candidate / "reliability_v3.sqlite").write_text("derived")

    def fake_export(_, candidate, *, public_destination):
        candidate.mkdir()
        (candidate / "database.zip").write_text("archive")
        (candidate / "export_receipt.json").write_text(json.dumps({"archives": {
            "complete": {"path": str(public_destination / "database.zip"),
                         "sha256": sha256_file(candidate / "database.zip")}
        }}))
        (candidate / "mac_scp_commands.txt").write_text("scp")

    def fake_source_export(_, candidate, *, public_destination):
        (candidate / "source.zip").write_text("source archive")
        receipt = {"archives": {"complete": {
            "path": str(public_destination / "source.zip"),
            "sha256": sha256_file(candidate / "source.zip"),
        }}}
        (candidate / "source_export_receipt.json").write_text(json.dumps(receipt))
        return receipt

    monkeypatch.setattr(recovery, "build_product", fake_build)
    monkeypatch.setattr(recovery, "_verify_product", lambda *_: {
        "primary_conditions": 280, "followup_conditions": 240, "database_sha256": "derived-hash",
    })
    monkeypatch.setattr(recovery, "export", fake_export)
    monkeypatch.setattr(recovery, "export_source_bundle", fake_source_export)
    result = recovery.recover(output, base, assets, baseline)
    assert result["model_inferences_repeated"] == 0
    assert result["partial_artifacts_preserved"] == 1
    assert (output / "derived_database" / "partial.csv").read_text() == "preserve"
    assert (output / recovery.DERIVED_NAME / "reliability_v3.sqlite").exists()
    assert (output / recovery.EXPORT_NAME / "database.zip").exists()
    assert recovery.recover(output, base, assets, baseline)["derived_database_sha256"] == "derived-hash"


def test_matched_kws_figures_use_four_interior_budgets(tmp_path):
    from workflows.build_reliability_database_v3 import matched_kws_plots

    database = tmp_path / "tiny.sqlite"
    methods = (
        "random_global", "msb_global", "numerical_jump_global", "gradient_global",
        "per_example_absolute_global", "per_example_positive_global", "fkeras_paper",
    )
    with sqlite3.connect(database) as db:
        db.execute("CREATE TABLE reliability_grid(study_id TEXT, method_id TEXT, coverage REAL, ber REAL, "
                   "mean_loss REAL, loss_ci_low REAL, loss_ci_high REAL, loss_degradation REAL)")
        study = "kws_partial_tier_fkeras_matched_v1_followup"
        for ber in (1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1):
            for method, budget in (("no_protection", 0.0), ("full_protection", 1.0)) + tuple(
                (method, budget) for method in methods for budget in (0.0625, 0.125, 0.1875, 0.25)
            ):
                damage = 0.0 if method == "full_protection" else ber * 100 * (1-budget)
                db.execute("INSERT INTO reliability_grid VALUES (?,?,?,?,?,?,?,?)",
                           (study, method, budget, ber, 0.4+damage,
                            0.4+damage-0.01, 0.4+damage+0.01, damage))
    output = tmp_path / "plots"
    matched_kws_plots(database, output)
    assert {path.name for path in output.iterdir()} == {
        "kws_loss_vs_ber.png", "kws_loss_vs_budget.png",
    }
    assert all(path.stat().st_size > 1000 for path in output.iterdir())


def test_saved_field_audit_rejects_missing_and_misidentified_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, "BERS", (0.1,))
    monkeypatch.setattr(recovery, "BUDGETS", (0.5,))
    monkeypatch.setattr(recovery, "METHODS", ("msb_global",))
    bundle = tmp_path / "bundle"
    (bundle / "artifacts").mkdir(parents=True)
    (bundle / "measurements").mkdir()
    schedules = {}
    rows = []
    for rep in range(4):
        name = f"replication_{rep}"
        schedules[name] = []
        for trial in range(8):
            digest = f"field-{rep}-{trial}"
            schedules[name].append({"logical_ber": 0.1, "schedule_index": trial,
                                    "sha256": digest, "events": [["kernel", trial, 7]]})
            for policy, applied, suppressed, accuracy, loss in (
                ("no_protection", 1, 0, 0.8, 0.2),
                ("full_protection", 0, 1, 0.9, 0.1),
                ("msb_global@0.500000", 0, 1, 0.9, 0.1),
            ):
                rows.append({"replication": name, "trial": trial,
                             "policy": f"ber=0.1/{policy}", "schedule_sha256": digest,
                             "proposed_events": 1, "applied_events": applied,
                             "suppressed_events": suppressed,
                             "accuracy": accuracy, "categorical_crossentropy": loss})
    (bundle / "artifacts/fault_schedules.json").write_text(json.dumps(schedules))
    trials = bundle / "measurements/trials.csv"
    def save():
        with trials.open("w", newline="") as file:
            writer = csv.DictWriter(file, rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    save()
    clean = {"accuracy": 0.9, "categorical_crossentropy": 0.1}
    assert recovery._audit_saved_fields(bundle, clean)["fields"] == 32
    rows[0]["proposed_events"] = 2
    save()
    with pytest.raises(ValueError, match="proposed fault field"):
        recovery._audit_saved_fields(bundle, clean)
    rows[0]["proposed_events"] = 1
    rows.pop()
    save()
    with pytest.raises(ValueError, match="incomplete"):
        recovery._audit_saved_fields(bundle, clean)
