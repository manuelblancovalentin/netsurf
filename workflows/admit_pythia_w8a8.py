"""Admit a finalized Pythia W8A8 handoff as a separate, offline database study.

This module never imports a model adapter or launches inference. The existing
three-model INT8 comparative view and historical primary view remain unchanged.
The compact handoff is checked against the locally available complete source
archive, including every source-member checksum and every saved field result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import shutil
import sqlite3
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

from netsurf.manifests import sha256_file

SCHEMA = "m12-pythia-w8a8-admission/1.0.0"
STUDY_ID = "pythia160m_w8a8_full_eligible_v3"
METHODS = {
    "unprotected": "no_protection",
    "random": "random_global",
    "msb_first": "msb_global",
    "numerical_bit_jump": "numerical_jump_global",
    "global_first_order": "gradient_global",
    "per_example_absolute": "per_example_absolute_global",
    "per_example_positive": "per_example_positive_global",
    "full_protection": "full_protection",
}


def _csv(archive: zipfile.ZipFile, name: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(archive.read(name).decode("utf-8"))))


def _same(actual: float | int, expected: str, label: str) -> None:
    if float(actual) != float(expected):
        raise ValueError(f"{label}: source record differs from compact observation")


def verify_handoff(compact: Path, complete: Path) -> dict:
    """Verify source/archive linkage, all members, and field-to-summary identities."""
    with zipfile.ZipFile(compact) as small:
        if small.testzip() is not None:
            raise ValueError("compact ZIP CRC failed")
        checks = json.loads(small.read("evidence_checksums.json"))
        if set(small.namelist()) != set(checks) | {"evidence_checksums.json"}:
            raise ValueError("compact ZIP member inventory differs")
        for name, expected in checks.items():
            if hashlib.sha256(small.read(name)).hexdigest() != expected:
                raise ValueError(f"compact member checksum differs: {name}")
        provenance = json.loads(small.read("handoff/provenance.json"))
        audit = json.loads(small.read("handoff/audit.json"))
        plan = json.loads(small.read("plan/plan.json"))
        ready = json.loads(small.read("plan/target_ready_record.json"))
        admission = json.loads(small.read("run/admission.json"))
        payload = json.loads(small.read("run/scientific_payload.json"))
        conditions = _csv(small, "handoff/condition_level.csv")
        fields = _csv(small, "handoff/field_observations.csv")
        masks = _csv(small, "handoff/mask_summary.csv")
        pairs = _csv(small, "handoff/paired_comparisons_original.csv")
        source_checks = json.loads(small.read("original_complete/checksums.json"))
        if (ready["status"] != "READY" or ready["target_id"] != "pythia160m_int8" or
                admission["target"] != "pythia160m_int8" or
                payload["payload_sha256"] != provenance["scientific_payload_sha256"] or
                admission["plan_sha256"] != provenance["plan_sha256"] or
                admission["ready_target_record_sha256"] != provenance["target_ready_record_sha256"] or
                admission["quantization_manifest_sha256"] != provenance["quantization_manifest_sha256"] or
                admission["checkpoint_sha256"] != next(
                    f["sha256"] for f in ready["model"]["files"] if f["path"] == "model.safetensors") or
                hashlib.sha256(small.read("run/analysis.json")).hexdigest() != provenance["analysis_file_sha256"] or
                small.read("original_complete/FINALIZED").decode().strip() !=
                provenance["original_complete_archive_finalized_digest"]):
            raise ValueError("Pythia source, plan or target identity differs")
        if hashlib.sha256(small.read("plan/plan.json")).hexdigest() != provenance["plan_file_sha256"]:
            raise ValueError("frozen plan file checksum differs")
        if sha256_file(complete) != provenance["original_complete_archive_sha256"]:
            raise ValueError("complete source archive SHA-256 differs")
        if complete.stat().st_size != provenance["original_complete_archive_bytes"]:
            raise ValueError("complete source archive byte count differs")

        # The complete ZIP is independently checked, not merely trusted from
        # the compact handoff's digest. Streaming prevents large mask loads.
        with zipfile.ZipFile(complete) as source:
            if set(source.namelist()) != set(source_checks) | {"evidence_checksums.json"}:
                raise ValueError("complete source member inventory differs")
            for name, expected in source_checks.items():
                digest = hashlib.sha256()
                with source.open(name) as member:
                    for chunk in iter(lambda: member.read(1 << 20), b""):
                        digest.update(chunk)
                if digest.hexdigest() != expected:
                    raise ValueError(f"complete source member checksum differs: {name}")

            population = int(provenance["eligible_population_bits"])
            if (population != int(plan["eligible_universe"]["bits"]) or
                    provenance["eligible_population_sha256"] != plan["eligible_universe"]["sha256"] or
                    len(conditions) != 78 or len(fields) != 1872 or len(masks) != 26 or len(pairs) != 60):
                raise ValueError("Pythia W8A8 study geometry differs")
            expected_methods = set(METHODS)
            if set(audit["ranking_methods"]) != expected_methods - {"unprotected", "full_protection"}:
                raise ValueError("ranking method inventory differs")
            if audit["independent_fields"] != 72 or audit["observations"] != 1872:
                raise ValueError("independent field counts differ")
            mask_map = {item["policy_id"]: item for item in masks}
            if len(mask_map) != len(masks):
                raise ValueError("duplicate mask identities")
            groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
            field_groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
            for row in fields:
                method = row["method"]
                if method not in METHODS:
                    raise ValueError(f"unknown method: {method}")
                if (row["plan_sha256"] != provenance["plan_sha256"] or
                        int(row["eligible_population_bits"]) != population or
                        row["eligible_population_sha256"] != provenance["eligible_population_sha256"]):
                    raise ValueError("field population or plan identity differs")
                budget = int(row["protected_bits"])
                coverage = float(row["coverage"])
                if budget != math.floor(population * coverage + 1e-9):
                    raise ValueError("protected-bit cardinality differs")
                if int(row["proposed_faults"]) != int(row["applied_faults"]) + int(row["suppressed_faults"]):
                    raise ValueError("proposed != applied + suppressed")
                if method == "unprotected" and int(row["suppressed_faults"]) != 0:
                    raise ValueError("unprotected control suppressed a fault")
                if method == "full_protection" and (
                    int(row["applied_faults"]) != 0 or
                    float(row["mean_nll"]) != audit["clean_nll"] or
                    int(row["failed_tokens"]) != 0
                ):
                    raise ValueError("full protection differs from clean")
                mask = mask_map.get(row["policy_id"])
                if mask is None or (mask["mask_sha256"] != row["mask_sha256"] or
                                    int(mask["protected_bits"]) != budget or
                                    mask["method"] != method):
                    raise ValueError("field mask differs from frozen mask summary")
                groups[(row["ber"], method, row["protected_bits"])].append(row)
                field_groups[(row["ber"], row["field_index"], row["field_sha256"])].append(row)
            if len(field_groups) != 72 or len(groups) != 78:
                raise ValueError("field or condition grouping differs")
            planned_fields = {float(item["ber"]): item for item in plan["fields"]}
            for (ber, index, digest), rows in field_groups.items():
                if len(rows) != 26 or len({r["policy_id"] for r in rows}) != 26:
                    raise ValueError("paired policy field is incomplete")
                if len({r["seed"] for r in rows}) != 1 or len({r["proposed_faults"] for r in rows}) != 1:
                    raise ValueError("paired proposed field differs across policies")
                frozen = planned_fields.get(float(ber))
                if (frozen is None or int(index) not in range(len(frozen["seeds"])) or
                        int(rows[0]["seed"]) != frozen["seeds"][int(index)] or
                        int(rows[0]["expected_proposed_faults"]) != frozen["lambda"]):
                    raise ValueError("field seed or planned exposure differs")
                source_paths = {r["field_record"] for r in rows}
                if len(source_paths) != 1:
                    raise ValueError("paired source record identity differs")
                record = json.loads(source.read("run/" + source_paths.pop()))
                if (float(ber) != record["ber"] or int(index) != record["field"] or
                        digest != record["field_sha256"] or
                        int(rows[0]["seed"]) != record["seed"] or
                        len(record["policies"]) != 26):
                    raise ValueError("compact field identity differs from original record")
                for row in rows:
                    saved = record["policies"][row["policy_id"]]
                    risk = saved["risk"]
                    if saved["mask_sha256"] != row["mask_sha256"]:
                        raise ValueError("saved mask differs")
                    for key, column in (("proposed", "proposed_faults"),
                                        ("applied", "applied_faults"),
                                        ("suppressed", "suppressed_faults")):
                        _same(saved["events"][key], row[column], key)
                    for key, column in (("mean_nll", "mean_nll"),
                                        ("perplexity", "perplexity"),
                                        ("failure_fraction", "failure_fraction")):
                        _same(risk["metrics"][key], row[column], key)
                    _same(risk["failed_units"], row["failed_tokens"], "failed_tokens")
                    _same(risk["total_units"], row["scored_tokens"], "scored_tokens")
            if {float(k[0]) for k in field_groups} != {float(v["ber"]) for v in plan["fields"]}:
                raise ValueError("BER grid differs from plan")
            condition_keys = {(c["ber"], c["method"], c["protected_bits"]) for c in conditions}
            if condition_keys != set(groups) or len(condition_keys) != len(conditions):
                raise ValueError("condition CSV duplicates or omits a field group")
            for condition in conditions:
                key = (condition["ber"], condition["method"], condition["protected_bits"])
                values = groups.get(key, [])
                if len(values) != 24 or int(condition["independent_fields"]) != 24:
                    raise ValueError("condition lacks 24 fields")
                mean = sum(float(row["mean_nll"]) for row in values) / 24
                if not math.isclose(mean, float(condition["mean_nll"]), abs_tol=1e-11):
                    raise ValueError("condition mean NLL differs from field rows")
                if not math.isclose(mean - audit["clean_nll"], float(condition["mean_loss_degradation"]), abs_tol=1e-11):
                    raise ValueError("condition degradation differs from clean")
                if not math.isclose(sum(float(r["failure_fraction"]) for r in values) / 24,
                                    float(condition["failed_fraction"]), abs_tol=1e-11):
                    raise ValueError("condition failure fraction differs")
        return {"provenance": provenance, "audit": audit, "plan": plan, "ready": ready,
                "admission": admission, "conditions": conditions, "fields": fields,
                "masks": masks, "pairs": pairs}


def build(base: Path, compact: Path, complete: Path, output: Path) -> dict:
    base, compact, complete, output = (p.resolve() for p in (base, compact, complete, output))
    if output.exists():
        raise FileExistsError(f"write-once output already exists: {output}")
    parent_checks = json.loads((base / "checksums.json").read_text())
    if (base / "FINALIZED").read_text().strip() != (
        f"database_sha256={parent_checks['files']['current_comparative.sqlite']}"):
        raise ValueError("parent comparative database is not finalized")
    for name, digest in parent_checks["files"].items():
        if sha256_file(base / name) != digest:
            raise ValueError(f"parent comparative artifact changed: {name}")
    evidence = verify_handoff(compact, complete)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".m12-w8a8-stage-", dir=output.parent))
    candidate = stage / output.name
    candidate.mkdir()
    try:
        database = candidate / "reliability_current_w8a8.sqlite"
        with sqlite3.connect(f"file:{base / 'current_comparative.sqlite'}?mode=ro", uri=True) as prior, sqlite3.connect(database) as db:
            prior.backup(db)
        with sqlite3.connect(database) as db:
            if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("parent SQLite integrity check failed")
            if (db.execute("SELECT count(*) FROM current_comparative_conditions").fetchone()[0] != 552 or
                    db.execute("SELECT count(*) FROM primary_conditions").fetchone()[0] != 280):
                raise ValueError("parent comparison counts differ")
            db.executescript("""
                CREATE TABLE w8a8_language_model_studies (
                    study_id TEXT PRIMARY KEY, target_id TEXT NOT NULL,
                    checkpoint_sha256 TEXT NOT NULL, plan_sha256 TEXT NOT NULL,
                    ready_record_sha256 TEXT NOT NULL, quantization_manifest_sha256 TEXT NOT NULL,
                    eligible_bits INTEGER NOT NULL, eligible_population_sha256 TEXT NOT NULL,
                    execution_backend TEXT NOT NULL, source_complete_sha256 TEXT NOT NULL,
                    source_compact_sha256 TEXT NOT NULL, scientific_payload_sha256 TEXT NOT NULL,
                    clean_nll REAL NOT NULL, clean_perplexity REAL NOT NULL,
                    selection_cohort_json TEXT NOT NULL, final_cohort_json TEXT NOT NULL,
                    source_complete_path TEXT NOT NULL, status TEXT NOT NULL);
                CREATE TABLE w8a8_language_model_masks (
                    study_id TEXT NOT NULL REFERENCES w8a8_language_model_studies(study_id),
                    method_id TEXT NOT NULL REFERENCES methods(method_id),
                    coverage REAL NOT NULL, protected_bits INTEGER NOT NULL,
                    mask_sha256 TEXT NOT NULL, source_policy_id TEXT NOT NULL,
                    packed_payload_sha256 TEXT NOT NULL,
                    PRIMARY KEY(study_id,method_id,coverage));
                CREATE TABLE w8a8_language_model_trials (
                    study_id TEXT NOT NULL REFERENCES w8a8_language_model_studies(study_id),
                    method_id TEXT NOT NULL REFERENCES methods(method_id),
                    coverage REAL NOT NULL, protected_bits INTEGER NOT NULL,
                    ber REAL NOT NULL, expected_proposed_faults INTEGER NOT NULL,
                    field_index INTEGER NOT NULL, seed INTEGER NOT NULL,
                    field_sha256 TEXT NOT NULL, mask_sha256 TEXT NOT NULL,
                    proposed_faults INTEGER NOT NULL, applied_faults INTEGER NOT NULL,
                    suppressed_faults INTEGER NOT NULL, mean_nll REAL NOT NULL,
                    perplexity REAL NOT NULL, failed_tokens INTEGER NOT NULL,
                    scored_tokens INTEGER NOT NULL, failure_fraction REAL NOT NULL,
                    source_field_record TEXT NOT NULL,
                    PRIMARY KEY(study_id,method_id,coverage,ber,field_index));
                CREATE TABLE w8a8_language_model_conditions (
                    study_id TEXT NOT NULL REFERENCES w8a8_language_model_studies(study_id),
                    method_id TEXT NOT NULL REFERENCES methods(method_id),
                    coverage REAL NOT NULL, protected_bits INTEGER NOT NULL,
                    ber REAL NOT NULL, expected_proposed_faults INTEGER NOT NULL,
                    field_count INTEGER NOT NULL, mean_nll REAL NOT NULL,
                    mean_loss_degradation REAL NOT NULL,
                    ci95_degradation_low REAL NOT NULL, ci95_degradation_high REAL NOT NULL,
                    penalized_perplexity REAL NOT NULL, failed_fraction REAL NOT NULL,
                    mean_proposed_faults REAL NOT NULL, mean_applied_faults REAL NOT NULL,
                    mean_suppressed_faults REAL NOT NULL,
                    PRIMARY KEY(study_id,method_id,coverage,ber));
                CREATE VIEW current_w8a8_language_model_conditions AS
                    SELECT * FROM w8a8_language_model_conditions WHERE study_id='pythia160m_w8a8_full_eligible_v3';
            """)
            source = evidence["provenance"]
            audit = evidence["audit"]
            plan = evidence["plan"]
            ready = evidence["ready"]
            db.execute("INSERT INTO w8a8_language_model_studies VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                STUDY_ID, "pythia160m_int8", evidence["admission"]["checkpoint_sha256"],
                source["plan_sha256"], source["target_ready_record_sha256"],
                source["quantization_manifest_sha256"], source["eligible_population_bits"],
                source["eligible_population_sha256"], ready["reference_results"]["computation_dtype"],
                source["original_complete_archive_sha256"], sha256_file(compact),
                source["scientific_payload_sha256"], audit["clean_nll"], audit["clean_perplexity"],
                json.dumps(plan["selection"], sort_keys=True), json.dumps(plan["final"], sort_keys=True),
                str(complete), "FINALIZED_MEASURED"))
            for item in evidence["masks"]:
                db.execute("INSERT INTO w8a8_language_model_masks VALUES (?,?,?,?,?,?,?)", (
                    STUDY_ID, METHODS[item["method"]], float(item["coverage_fraction"]),
                    int(item["protected_bits"]), item["mask_sha256"], item["policy_id"],
                    item["packed_payload_sha256_in_original_complete_archive"]))
            for item in evidence["fields"]:
                db.execute("INSERT INTO w8a8_language_model_trials VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                    STUDY_ID, METHODS[item["method"]], float(item["coverage"]),
                    int(item["protected_bits"]), float(item["ber"]), int(item["expected_proposed_faults"]),
                    int(item["field_index"]), int(item["seed"]), item["field_sha256"], item["mask_sha256"],
                    int(item["proposed_faults"]), int(item["applied_faults"]), int(item["suppressed_faults"]),
                    float(item["mean_nll"]), float(item["perplexity"]), int(item["failed_tokens"]),
                    int(item["scored_tokens"]), float(item["failure_fraction"]), item["field_record"]))
            for item in evidence["conditions"]:
                db.execute("INSERT INTO w8a8_language_model_conditions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                    STUDY_ID, METHODS[item["method"]], float(item["coverage"]), int(item["protected_bits"]),
                    float(item["ber"]), int(item["expected_proposed_faults"]), int(item["independent_fields"]),
                    float(item["mean_nll"]), float(item["mean_loss_degradation"]),
                    float(item["ci95_degradation_low"]), float(item["ci95_degradation_high"]),
                    float(item["penalized_perplexity"]), float(item["failed_fraction"]),
                    float(item["mean_proposed_faults"]), float(item["mean_applied_faults"]),
                    float(item["mean_suppressed_faults"])))
            db.execute("INSERT INTO metadata VALUES (?,?)", ("w8a8_admission_schema", SCHEMA))
            if (db.execute("SELECT count(*) FROM w8a8_language_model_trials").fetchone()[0] != 1872 or
                    db.execute("SELECT count(*) FROM w8a8_language_model_conditions").fetchone()[0] != 78 or
                    db.execute("SELECT count(*) FROM current_comparative_conditions").fetchone()[0] != 552 or
                    db.execute("SELECT count(*) FROM primary_conditions").fetchone()[0] != 280 or
                    db.execute("PRAGMA integrity_check").fetchone()[0] != "ok"):
                raise ValueError("admitted SQLite counts or integrity differ")
            db.commit()
        with zipfile.ZipFile(compact) as small:
            for source_name, target_name in (
                ("handoff/condition_level.csv", "pythia_w8a8_conditions.csv"),
                ("handoff/field_observations.csv", "pythia_w8a8_trials.csv"),
                ("handoff/mask_summary.csv", "pythia_w8a8_masks.csv"),
                ("handoff/paired_comparisons_original.csv", "pythia_w8a8_original_paired_comparisons.csv"),
                ("handoff/loss_degradation_vs_ber.svg", "pythia_w8a8_loss_vs_ber.svg"),
                ("handoff/loss_degradation_vs_budget.svg", "pythia_w8a8_loss_vs_budget.svg"),
            ):
                (candidate / target_name).write_bytes(small.read(source_name))
        shutil.copyfile(compact, candidate / "pythia_source_compact.zip")
        (candidate / "SOURCE_PROVENANCE.json").write_text(json.dumps({
            "schema": SCHEMA, "study_id": STUDY_ID,
            "source_compact_sha256": sha256_file(compact),
            "source_complete_sha256": source["original_complete_archive_sha256"],
            "source_complete_path": str(complete),
            "scientific_payload_sha256": source["scientific_payload_sha256"],
            "parent_database_sha256": sha256_file(base / "current_comparative.sqlite"),
            "source_members_verified": 287, "model_inferences_repeated": 0,
        }, sort_keys=True, indent=2) + "\n")
        (candidate / "README.md").write_text(
            "# Current reliability database with separate Pythia W8A8 study\n\n"
            "The `current_comparative_conditions` view remains the original 552-condition "
            "ResNet-8/KWS/ToyCar TFLite INT8 comparison; `primary_conditions` remains the historical "
            "280-condition view. Neither view contains Pythia. The new "
            "`current_w8a8_language_model_conditions` view contains 78 separate Pythia conditions "
            "and `w8a8_language_model_trials` contains 1,872 observations from 72 paired fields.\n\n"
            "Pythia uses derived INT8 stored Linear qweights and per-token dynamic INT8 Linear inputs "
            "with exact INT32 accumulation; embeddings, normalization, attention products and other "
            "operations remain floating point. The fault universe is only 988,545,024 stored qweight "
            "bits, not all model parameters. Task loss is token NLL in nats; perplexity and failed-token "
            "fraction are separate. The final cohort contains 4,096 scored WikiText-2 test tokens. "
            "It is not comparable in units or backend to classification cross-entropy or ToyCar AUC.\n\n"
            "The included compact source is linked by SHA-256 to the omitted complete 3.37-GB archive; "
            "all 287 listed complete-archive members were verified during admission. The original "
            "60 Holm-adjusted paired comparisons are copied unchanged. No tested BER produced positive "
            "mean unprotected loss degradation; none of those 60 comparisons passes Holm at 0.05. "
            "Do not infer ranking superiority or a whole-model degradation transition from this grid.\n")
        files = {str(path.relative_to(candidate)): sha256_file(path)
                 for path in sorted(candidate.iterdir()) if path.is_file()}
        (candidate / "checksums.json").write_text(json.dumps({"schema": SCHEMA, "files": files}, sort_keys=True, indent=2) + "\n")
        (candidate / "FINALIZED").write_text(f"database_sha256={files[database.name]}\n")
        if output.exists():
            raise FileExistsError(output)
        os.rename(candidate, output)
        stage.rmdir()
    except Exception as error:
        raise RuntimeError(f"Pythia admission staging retained for inspection: {stage}") from error
    archive = output.with_suffix(".zip")
    if archive.exists():
        raise FileExistsError(archive)
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as package:
        for path in sorted(output.iterdir()):
            if path.is_file():
                package.write(path, arcname=str(Path(output.name) / path.name))
    with zipfile.ZipFile(archive) as package:
        if package.testzip() is not None:
            raise ValueError("Pythia admission export ZIP CRC failed")
    return {"product": str(output), "archive": str(archive),
            "database_sha256": sha256_file(output / "reliability_current_w8a8.sqlite"),
            "archive_sha256": sha256_file(archive), "pythia_conditions": 78,
            "pythia_field_observations": 1872,
            "unchanged_tflite_current_conditions": 552,
            "unchanged_historical_primary_conditions": 280,
            "model_inferences_repeated": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("workspace/m12_current_comparative_v2"))
    parser.add_argument("--compact", type=Path, required=True)
    parser.add_argument("--complete", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("workspace/m12_current_comparative_w8a8_v1"))
    args = parser.parse_args()
    print(json.dumps(build(args.base, args.compact, args.complete, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
