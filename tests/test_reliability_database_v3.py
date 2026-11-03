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
