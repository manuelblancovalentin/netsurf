"""M12 readiness and source-faithful feedback tests; no neural-network execution."""

import json
from pathlib import Path

import pytest

from netsurf.manifests import sha256_file
from netsurf.priorifi_feedback import derived_allocation, feedback_probe_order
from netsurf import reliability_database_v3 as v3


def test_priori_feedback_probes_each_tier_then_updates_from_observations():
    bits = [["a0", "a1", "a2"], ["b0", "b1", "b2"], ["c0"]]
    values = {"a0":.1,"a1":.2,"a2":.3,"b0":.9,"b1":.8,"b2":.7,"c0":.5}
    first = feedback_probe_order(bits, values.__getitem__, last_k=2)
    assert [bit for bit,_ in first] == ["a0","b0","c0","b1","b2","a1","a2"]
    assert first == feedback_probe_order(bits, values.__getitem__, last_k=2)
    assert bits[0] == ["a0","a1","a2"]  # caller's tier lists were not mutated
    assert derived_allocation(first, [b for tier in bits for b in tier], 2) == ("b0","b1")
    with pytest.raises(ValueError, match="complete unique"):
        derived_allocation(first[:-1], [b for tier in bits for b in tier], 2)
    with pytest.raises(ValueError, match="duplicate"):
        feedback_probe_order([["x"],["x"]], lambda _: 0)


def test_asset_gate_requires_exact_manifest_files_and_receipts(tmp_path, monkeypatch):
    target="mini_int8"
    monkeypatch.setattr(v3,"TARGETS",{target:("mini_int8_full",8)})
    model=tmp_path/"model"; model.mkdir()
    data=tmp_path/"data"; data.mkdir()
    (model/"model.safetensors").write_bytes(b"model")
    (model/"quantization_manifest.json").write_bytes(b"quant")
    (data/"eval.parquet").write_bytes(b"data")
    clean=tmp_path/"clean.json";clean.write_text("{}")
    fault=tmp_path/"fault.json";fault.write_text("{}")
    def record(path: Path):
        return {"path":path.name,"sha256":sha256_file(path),"size":path.stat().st_size}
    target_record={"target_id":target,"status":"READY","model":{"path":str(model),"files":[record(model/"model.safetensors"),record(model/"quantization_manifest.json")]},
                   "dataset":{"path":str(data),"files":[record(data/"eval.parquet")]},
                   "eligible_stored_int8_population":{"bits":8},
                   "reference_results":{"smoke_report":str(clean),"smoke_report_sha256":sha256_file(clean)},
                   "fault_mutation_receipt":{"report":str(fault),"report_sha256":sha256_file(fault),"summary":{"passed":True}}}
    manifest=tmp_path/"assets.json";manifest.write_text(json.dumps({"targets":[target_record]}))
    assert v3.verified_asset_records(manifest,expected_sha256=sha256_file(manifest))==[target_record]
    (model/"model.safetensors").write_bytes(b"altered")
    with pytest.raises(ValueError,match="asset file failed"):
        v3.verified_asset_records(manifest,expected_sha256=sha256_file(manifest))


def test_frozen_followup_plan_is_hash_gated(tmp_path):
    from workflows.run_reliability_followup import checked_plan

    source=tmp_path/"source.toml";source.write_text("source")
    root=tmp_path/"product";root.mkdir()
    (root/"plans").mkdir()
    config=root/"plans"/"study.toml";config.write_text("frozen")
    receipt=root/"plans"/"plan_receipt.json"
    plan={"name":"study","status":"VALIDATED_NOT_EXECUTED",
          "configuration":"plans/study.toml","configuration_sha256":sha256_file(config),
          "source_configuration":str(source),"source_configuration_sha256":sha256_file(source)}
    receipt.write_text(json.dumps({"schema_version":"1.0.0","plans":[plan]}))
    assert checked_plan(receipt,"study")[1]==config
    config.write_text("changed")
    with pytest.raises(ValueError,match="frozen plan configuration changed"):
        checked_plan(receipt,"study")


def test_export_verifies_product_and_writes_absolute_download_paths(tmp_path):
    import zipfile
    from workflows.export_reliability_database_v3 import export

    product=tmp_path/"product";product.mkdir()
    database=product/"reliability_v3.sqlite";database.write_bytes(b"database")
    (product/"primary_conditions.csv").write_text("study_id\n")
    files={path.name:sha256_file(path) for path in (database,product/"primary_conditions.csv")}
    (product/"checksums.json").write_text(json.dumps({"schema_version":v3.SCHEMA_VERSION,"files":files}))
    (product/"FINALIZED").write_text(f"reliability_database_sha256={files['reliability_v3.sqlite']}\n")
    destination=tmp_path/"export"
    export(product,destination)
    commands=(destination/"mac_scp_commands.txt").read_text()
    assert f"kona:{destination}/product_compact.zip" in commands
    with zipfile.ZipFile(destination/"product_compact.zip") as archive:
        assert "product/primary_conditions.csv" in archive.namelist()
        assert "product/reliability_v3.sqlite" not in archive.namelist()
    with zipfile.ZipFile(destination/"product_complete.zip") as archive:
        assert "product/reliability_v3.sqlite" in archive.namelist()
    database.write_bytes(b"corrupt")
    with pytest.raises(ValueError,match="artifact changed"):
        export(product,tmp_path/"bad_export")


def test_kws_postflight_requires_paired_fields_and_complete_tier_equivalence(tmp_path):
    import csv
    from workflows.kws_matched_gate import BUDGETS, METHODS, postflight

    bundle = tmp_path / "bundle"
    (bundle / "measurements").mkdir(parents=True)
    cohort = {"ranking": "ranking-hash", "empirical_validation": "validation-hash",
              "evaluation": "evaluation-hash"}
    benchmark = {
        "checkpoint_sha256": "checkpoint-hash",
        "dataset_identity": {"asset_lock_digest": "lock-digest"},
        "sample_partitions": {f"replication_{rep}": {
            role: {"sample_sha256": digest} for role, digest in cohort.items()
        } for rep in range(4)},
    }
    (bundle / "manifest.json").write_text(json.dumps({"benchmark": benchmark}))
    rows = [
        {"replication": f"replication_{rep}", "left": f"msb_global@{budget:.6f}",
         "right": f"fkeras_paper@{budget:.6f}", "protected_bits": round(176128 * budget),
         "same_mask": True, "symmetric_difference_bits": 0}
        for rep in range(4) for budget in (0.125, 0.25)
    ]
    (bundle / "summary.json").write_text(json.dumps({"eligible_bits": 176128, "mask_equivalence": rows}))
    policies = ("no_protection", "full_protection") + tuple(
        f"{method}@{budget:.6f}" for method in METHODS for budget in BUDGETS
    )
    fields = ("replication", "trial", "policy", "schedule_sha256", "accuracy",
              "categorical_crossentropy", "proposed_events", "applied_events", "suppressed_events")
    trials = [{
        "replication": f"replication_{rep}", "trial": "0", "policy": f"ber=0.001/{policy}",
        "schedule_sha256": f"field-{rep}", "accuracy": "0.9",
        "categorical_crossentropy": "0.3", "proposed_events": "3",
        "applied_events": "3", "suppressed_events": "0",
    } for rep in range(4) for policy in policies]
    trial_path = bundle / "measurements/trials.csv"
    def write_trials():
        with trial_path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(trials)
    write_trials()
    receipt = {"model_sha256": "checkpoint-hash", "asset_lock_digest": "lock-digest",
               "cohort_sha256": cohort, "fault_fields": 4}
    result = postflight(bundle, receipt)
    assert result["paired_fault_fields"] == 4
    assert result["policy_budget_evaluations"] == 120
    assert len(result["complete_tier_equivalence"]) == 8
    trials[policies.index("fkeras_paper@0.125000")]["accuracy"] = "0.8"
    write_trials()
    with pytest.raises(ValueError, match="identical masks produced different accuracy"):
        postflight(bundle, receipt)
    trials[policies.index("fkeras_paper@0.125000")]["accuracy"] = "0.9"
    trials[policies.index("gradient_global@0.062500")]["schedule_sha256"] = "unpaired"
    write_trials()
    with pytest.raises(ValueError, match="unmatched policies or fault schedules"):
        postflight(bundle, receipt)
    rows[0]["same_mask"] = False
    (bundle / "summary.json").write_text(json.dumps({"eligible_bits": 176128, "mask_equivalence": rows}))
    with pytest.raises(ValueError, match="complete-tier mask mismatch"):
        postflight(bundle, receipt)


def test_followup_source_export_preserves_complete_and_labeled_compact_archive(tmp_path, monkeypatch):
    import zipfile
    from workflows import export_reliability_followup as exporter

    bundle = tmp_path / "source"
    (bundle / "measurements").mkdir(parents=True)
    (bundle / "measurements/trials.csv").write_text("trial\n0\n")
    (bundle / "measurements/estimator_predictions.csv").write_text("bit,score\n0,1\n")
    (bundle / "FINALIZED").write_text("finalized\n")
    (bundle / "checksums.json").write_text("{}\n")
    checks = {"files": {
        "measurements/trials.csv": sha256_file(bundle / "measurements/trials.csv"),
        "measurements/estimator_predictions.csv": sha256_file(bundle / "measurements/estimator_predictions.csv"),
    }}
    source = {"scientific_payload_sha256": "scientific-hash", "checksums_sha256": "inventory-hash"}
    monkeypatch.setattr(exporter, "_verified_bundle", lambda _: (bundle, {}, {}, checks))
    destination = tmp_path / "export"
    destination.mkdir()
    (destination / "mac_scp_commands.txt").write_text("# database archives\n")
    receipt = exporter.export_source_bundle(source, destination)
    assert exporter.export_source_bundle(source, destination) == receipt
    with zipfile.ZipFile(receipt["archives"]["compact"]["path"]) as archive:
        names = archive.namelist()
        assert "source_bundle/measurements/trials.csv" in names
        assert "source_bundle/measurements/estimator_predictions.csv" not in names
        assert "source_bundle/COMPACT_ARCHIVE_NOTICE.txt" in names
    with zipfile.ZipFile(receipt["archives"]["complete"]["path"]) as archive:
        assert "source_bundle/measurements/estimator_predictions.csv" in archive.namelist()
    assert (destination / "mac_scp_commands.txt").read_text().count("scp kona:") == 2
    (destination / "source_compact.zip").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="source archive changed"):
        exporter.export_source_bundle(source, destination)
