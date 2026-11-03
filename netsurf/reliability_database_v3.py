"""M12 derived asset-readiness layer for the immutable reliability database.

Asset readiness is deliberately distinct from a validated NetSuRF adapter and
from measured protection effectiveness. No model is loaded by this module.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .manifests import sha256_file
from .reliability_database import build_database

SCHEMA_VERSION = "1.2.0"
ASSET_MANIFEST_SHA256 = "7128399521cef0d3e08f5d70bcb4f5b360fe2c85406c8ea8ac495e85be77b1b0"
ASSET_COMMIT = "1aea52b18b28af9e6078316f7948d92282393f83"
TARGETS = {
    "distilbert_sst2_int8": ("distilbert_sst2_int8_full", 344469504),
    "pythia160m_int8": ("pythia160m_int8_full", 988545024),
    "qwen3_0_6b_int8": ("qwen3_0_6b_int8_full", 4767875072),
}


def verified_asset_records(manifest_path: Path, *, expected_sha256: str = ASSET_MANIFEST_SHA256) -> list[dict]:
    """Verify the pinned manifest and every declared model/data/receipt byte.

    This is a provenance gate, not a model-execution or NetSuRF-adapter gate.
    """
    manifest_path = Path(manifest_path)
    if sha256_file(manifest_path) != expected_sha256:
        raise ValueError("Kona asset-readiness manifest SHA-256 differs from pinned M12 receipt")
    document = json.loads(manifest_path.read_text())
    indexed = {row["target_id"]: row for row in document["targets"]}
    verified = []
    for target_id, (_, expected_bits) in TARGETS.items():
        row = indexed[target_id]
        if row["status"] != "READY" or row["eligible_stored_int8_population"]["bits"] != expected_bits:
            raise ValueError(f"asset status or eligible-bit universe changed: {target_id}")
        model_root = Path(row["model"]["path"])
        # The WikiText asset records give a registry ID rather than an
        # absolute root; use its pinned local dataset contract explicitly.
        dataset_root = Path(row["dataset"].get("path") or {
            "wikitext-2": "/data/datasets/wikitext2",
        }[row["dataset"]["registry_id"]])
        for root, files in ((model_root, row["model"]["files"]),
                            (dataset_root, row["dataset"]["files"])):
            for item in files:
                path = root / item["path"]
                if not path.is_file() or path.stat().st_size != item["size"] or sha256_file(path) != item["sha256"]:
                    raise ValueError(f"asset file failed SHA-256/size gate: {path}")
        for key, location, digest in (
            ("clean", row["reference_results"]["smoke_report"], row["reference_results"]["smoke_report_sha256"]),
            ("fault", row["fault_mutation_receipt"]["report"], row["fault_mutation_receipt"]["report_sha256"]),
        ):
            if sha256_file(Path(location)) != digest:
                raise ValueError(f"{target_id} {key} receipt SHA-256 differs")
        if not row["fault_mutation_receipt"]["summary"]["passed"]:
            raise ValueError(f"{target_id} asset fault receipt did not pass")
        verified.append(row)
    return verified


def build_database_v3(sources: Path, destination: Path, asset_manifest: Path) -> None:
    """Create a new SQLite product; leave v1/v2 products and sources untouched."""
    records = verified_asset_records(asset_manifest)
    build_database(sources, destination)
    try:
        with sqlite3.connect(destination) as db:
            db.execute("UPDATE metadata SET value=? WHERE key='schema_version'", (SCHEMA_VERSION,))
            db.execute("INSERT INTO metadata VALUES (?,?)", ("asset_manifest_sha256", ASSET_MANIFEST_SHA256))
            db.execute("INSERT INTO metadata VALUES (?,?)", ("asset_repo_commit", ASSET_COMMIT))
            db.execute("UPDATE methods SET mathematical_definition=?, implementation_status=? WHERE method_id='priorifi_derived'", (
                "Source-pinned feedback fault-prioritization order; separately adapted full-calibration change-fraction allocation",
                "FEEDBACK_ONLY_ALLOCATION_UNVALIDATED"))
            db.execute("""CREATE TABLE asset_readiness (
            target_id TEXT PRIMARY KEY, model_id TEXT NOT NULL, asset_status TEXT NOT NULL,
            adapter_status TEXT NOT NULL, experiment_status TEXT NOT NULL,
            checkpoint_sha256 TEXT NOT NULL, quantization_manifest_sha256 TEXT NOT NULL,
            eligible_bits INTEGER NOT NULL, backend_contract TEXT NOT NULL,
            clean_metric_json TEXT NOT NULL, asset_record_json TEXT NOT NULL)""")
            for record in records:
                target_id = record["target_id"]
                model_id, bits = TARGETS[target_id]
                model_files = {Path(f["path"]).name: f["sha256"] for f in record["model"]["files"]}
                db.execute("INSERT INTO asset_readiness VALUES (?,?,?,?,?,?,?,?,?,?,?)", (
                    target_id, model_id, "READY", "NOT_VALIDATED", "NOT_MEASURED",
                    model_files["model.safetensors"], model_files["quantization_manifest.json"], bits,
                    record["model"]["artifact"]["kind"],
                    json.dumps(record["reference_results"]["metrics"], sort_keys=True),
                    json.dumps(record, sort_keys=True),
                ))
                db.execute("""UPDATE planned_model_methods SET status='NOT_IMPLEMENTED',
                reason='READY hash-verified W8A8 asset; no validated NetSuRF full-universe adapter or mask'
                WHERE target_id=? AND status='ASSET_BLOCKED'""", (model_id,))
            db.execute("""CREATE VIEW primary_condition_context AS
                SELECT p.*, m.eligible_bits,
                       (m.eligible_bits-p.protected_bits)*p.ber AS expected_applied_faults,
                       m.calibration_cohort_json, m.evaluation_cohort_json,
                       m.dataset_identity, m.backend
                FROM primary_conditions p JOIN studies s USING(study_id)
                JOIN models m ON s.model_id=m.model_id""")
            db.commit()
    except Exception:
        Path(destination).unlink(missing_ok=True)
        raise
