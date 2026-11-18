"""Write-once, offline union of finalized TFLite and W8A8 reliability studies.

The historical primary and v2 comparative views are retained in the copied
database.  No model inference, mask construction, or fault scheduling is used.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import statistics
import tempfile
import zipfile

from netsurf.manifests import sha256_file
from netsurf.reliability_database import _insert_study, _verified_bundle

ROOT = Path(__file__).resolve().parent.parent
WORKTREES = ROOT.parent
BASE = ROOT / "workspace/m12_current_comparative_v2"
TRANSFORMER = WORKTREES / "transformer-integration"
FKERAS = WORKTREES / "m12-fkeras-multibench"
OUTPUT = ROOT / "workspace/m12_current_comparative_v3"
TFLITE = (
    ("resnet8", "m12_resnet8_fkeras_matched_grid_v1", 240, 7680),
    ("kws", "kws_partial_tier_fkeras_matched_v1_followup", 240, 7680),
    ("toycar", "m12_toycar_fkeras_matched_grid_v1", 150, 4800),
)
FOCUSED = (
    ("m12_resnet8_fkeras_paired_precision_v1", 16, 5120),
    ("m12_toycar_fkeras_paired_precision_v1", 16, 6144),
)
W8A8 = (
    ("distilbert", "distilbert_sst2_w8a8_reliability_v2",
     "distilbert_sst2_w8a8_reliability_v2_run1", "_reporting_v2", 344469504),
    ("pythia", "pythia160m_w8a8_reliability_v3",
     "pythia160m_w8a8_reliability_v3_run1", "_reporting_v3", 988545024),
    ("qwen", "qwen3_0_6b_w8a8_reliability_v3",
     "qwen3_0_6b_w8a8_reliability_v3_run1", "_reporting_v3", 4767875072),
)
ALIASES = {
    "unprotected": "no_protection",
    "random": "random_global", "msb_first": "msb_global",
    "numerical_bit_jump": "numerical_jump_global",
    "global_first_order": "gradient_global",
    "per_example_absolute": "per_example_absolute_global",
    "per_example_positive": "per_example_positive_global",
}
BUDGETS = (0.0, 0.0625, 0.125, 0.1875, 0.25, 1.0)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _tflite_source(name: str) -> dict:
    roots = [p.parent for p in (FKERAS / "workspace/runs" / name).glob("*/FINALIZED")]
    if len(roots) != 1:
        raise ValueError(f"expected one finalized {name} bundle, found {len(roots)}")
    root = roots[0]
    return {"id": name, "path": str(root), "kind": "whole_kernel_fkeras_followup",
            "scientific_payload_sha256": (root / "FINALIZED").read_text().strip().split("=", 1)[1],
            "checksums_sha256": sha256_file(root / "checksums.json")}


def _verify_transformer(spec: tuple) -> tuple[dict, dict, dict, Path, Path, dict]:
    key, plan_name, run_name, reporting_suffix, expected_bits = spec
    plan_path = TRANSFORMER / "workspace/plans" / plan_name / "plan.json"
    run = TRANSFORMER / "workspace/runs" / run_name
    report = run.with_name(run.name + reporting_suffix)
    plan = json.loads(plan_path.read_text())
    receipt_path = report / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if plan["eligible_universe"]["bits"] != expected_bits or \
       receipt["plan_sha256"] != plan["plan_sha256"] or \
       (report / "FINALIZED").read_text().strip() != sha256_file(receipt_path) or \
       receipt["condition_count"] != 78 or receipt["field_observation_count"] != 1872:
        raise ValueError(f"{key} finalized plan/report identity differs")
    complete = run.with_name(run.name + "_export") / "complete.zip"
    compact = report / "compact.zip"
    # DistilBERT has a core scientific archive and a separate reporting archive.
    # The receipt's full_zip_sha256 identifies the latter, not the core archive.
    expected_complete = receipt.get("complete_zip_sha256", receipt.get("core_zip_sha256"))
    if sha256_file(complete) != expected_complete or \
       sha256_file(compact) != receipt["compact_zip_sha256"]:
        raise ValueError(f"{key} complete or compact ZIP checksum differs")
    if "core_zip_sha256" in receipt and \
       sha256_file(report / "complete_v2.zip") != receipt["full_zip_sha256"]:
        raise ValueError(f"{key} reporting ZIP checksum differs")
    member_hashes = receipt.get("member_sha256", receipt.get("compact_member_sha256"))
    with zipfile.ZipFile(compact) as archive:
        if archive.testzip() is not None:
            raise ValueError(f"{key} compact ZIP CRC differs")
        for name, digest in member_hashes.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError(f"{key} compact member differs: {name}")
    payload = json.loads((run / "scientific_payload.json").read_text())
    if payload["plan_sha256"] != plan["plan_sha256"] or \
       sha256_file(run / "analysis.json") != payload["analysis_sha256"]:
        raise ValueError(f"{key} scientific payload differs")
    for name, digest in payload["final_member_hashes"].items():
        if sha256_file(run / name) != digest:
            raise ValueError(f"{key} sealed final member differs: {name}")
    analysis = json.loads((run / "analysis.json").read_text())
    if analysis["independent_field_count"] != 72 or len(analysis["comparisons"]) != 60:
        raise ValueError(f"{key} field analysis incomplete")
    admission = json.loads(run.with_name(run.name + "_admission.json").read_text())
    admission_plan_sha = admission.get("plan_sha256", admission.get("v2_plan_sha256"))
    if admission_plan_sha != plan["plan_sha256"]:
        raise ValueError(f"{key} admission plan differs")
    return plan, receipt, payload, run, report, admission


def _schema(db: sqlite3.Connection) -> None:
    db.executescript("""
    CREATE TABLE current_study_registry_v3(
      model_key TEXT PRIMARY KEY, study_id TEXT NOT NULL UNIQUE,
      study_kind TEXT NOT NULL, source_path TEXT NOT NULL,
      source_payload_sha256 TEXT NOT NULL, checkpoint_sha256 TEXT NOT NULL,
      eligible_bits INTEGER NOT NULL, task_loss_name TEXT NOT NULL,
      condition_count INTEGER NOT NULL, trial_count INTEGER NOT NULL);
    CREATE TABLE current_conditions_v3(
      study_id TEXT NOT NULL, benchmark_id TEXT NOT NULL,
      checkpoint_sha256 TEXT NOT NULL, precision TEXT NOT NULL,
      backend TEXT NOT NULL, eligible_bits INTEGER NOT NULL,
      task_loss_name TEXT NOT NULL, quality_name TEXT NOT NULL,
      method_id TEXT NOT NULL, method_name TEXT NOT NULL,
      coverage REAL NOT NULL, protected_bits INTEGER NOT NULL,
      ber REAL NOT NULL, expected_proposed_faults REAL NOT NULL,
      field_count INTEGER NOT NULL, replication_count INTEGER NOT NULL,
      ranking_mask_count INTEGER NOT NULL, clean_loss REAL NOT NULL,
      mean_loss REAL NOT NULL, loss_sd REAL NOT NULL, loss_se REAL NOT NULL,
      loss_ci_low REAL NOT NULL, loss_ci_high REAL NOT NULL,
      loss_degradation REAL NOT NULL, mean_quality REAL,
      failure_fraction REAL, ci_basis TEXT NOT NULL,
      status TEXT NOT NULL,
      PRIMARY KEY(study_id,method_id,coverage,ber));
    CREATE TABLE transformer_trials_v3(
      study_id TEXT NOT NULL, method_id TEXT NOT NULL, coverage REAL NOT NULL,
      ber REAL NOT NULL, field_index INTEGER NOT NULL, seed INTEGER NOT NULL,
      field_sha256 TEXT NOT NULL, mask_sha256 TEXT NOT NULL,
      proposed_faults INTEGER NOT NULL, applied_faults INTEGER NOT NULL,
      suppressed_faults INTEGER NOT NULL, mean_loss REAL NOT NULL,
      quality REAL, failed_units INTEGER NOT NULL, total_units INTEGER NOT NULL,
      source_field_record TEXT NOT NULL,
      PRIMARY KEY(study_id,method_id,coverage,ber,field_index));
    CREATE TABLE paired_contrasts_v3(
      study_id TEXT NOT NULL, study_role TEXT NOT NULL, ber REAL NOT NULL,
      coverage REAL NOT NULL, left_method TEXT NOT NULL,
      right_method TEXT NOT NULL, metric TEXT NOT NULL,
      matched_pairs INTEGER NOT NULL, mean_loss_difference REAL NOT NULL,
      ci95_loss_low REAL NOT NULL, ci95_loss_high REAL NOT NULL,
      p_raw REAL NOT NULL, p_holm REAL NOT NULL,
      multiplicity_family TEXT NOT NULL, ci_basis TEXT NOT NULL);
    CREATE TABLE focused_conditions_v3(
      study_id TEXT NOT NULL, method_id TEXT NOT NULL, coverage REAL NOT NULL,
      ber REAL NOT NULL, field_count INTEGER NOT NULL, mean_task_loss REAL NOT NULL,
      mean_quality REAL, ci95_loss_low REAL NOT NULL,
      ci95_loss_high REAL NOT NULL, PRIMARY KEY(study_id,method_id,coverage,ber));
    CREATE TABLE current_method_coverage_v3(
      study_id TEXT NOT NULL, method_id TEXT NOT NULL,
      status TEXT NOT NULL, measured_conditions INTEGER NOT NULL,
      reason TEXT NOT NULL, PRIMARY KEY(study_id,method_id));
    CREATE TABLE current_coverage_matrix_v3(
      study_id TEXT NOT NULL, method_id TEXT NOT NULL,
      coverage REAL NOT NULL, ber REAL NOT NULL, status TEXT NOT NULL,
      PRIMARY KEY(study_id,method_id,coverage,ber));
    CREATE VIEW current_comparative_conditions_v3 AS
      SELECT * FROM current_conditions_v3;
    DROP VIEW current_comparative_conditions;
    CREATE VIEW current_comparative_conditions AS
      SELECT * FROM current_comparative_conditions_v3;
    CREATE VIEW current_trial_observations_v3 AS
      SELECT t.study_id,t.method_id,t.coverage,t.ber,t.replication,t.trial,
             t.schedule_sha256 AS field_sha256,t.proposed_events AS proposed_faults,
             t.applied_events AS applied_faults,t.suppressed_events AS suppressed_faults,
             t.task_loss AS mean_loss,t.task_quality AS quality,
             t.source_relative_path AS source_field_record
      FROM trial_observations t JOIN current_study_registry_v3 s USING(study_id)
      WHERE s.study_kind='whole_kernel_tflite_int8'
      UNION ALL
      SELECT study_id,method_id,coverage,ber,'field_only',field_index,
             field_sha256,proposed_faults,applied_faults,suppressed_faults,
             mean_loss,quality,source_field_record
      FROM transformer_trials_v3;
    """)


def _tflite(db: sqlite3.Connection, model_key: str, sid: str,
            expected_conditions: int, expected_trials: int, source: dict | None) -> dict:
    if source is not None:
        _verified_bundle(source)
        _insert_study(db, source)
    row = db.execute("""SELECT s.source_path,s.source_payload_sha256,m.checkpoint_sha256,
       m.eligible_bits,m.task_loss_name,m.clean_task_loss FROM studies s
       JOIN models m USING(model_id) WHERE s.study_id=?""", (sid,)).fetchone()
    if row is None:
        raise ValueError(f"missing TFLite study {sid}")
    path, payload, checkpoint, bits, metric, clean = row
    conditions = [dict(r) for r in db.execute("SELECT * FROM reliability_grid WHERE study_id=?", (sid,))]
    trial_count = db.execute("SELECT count(*) FROM trial_observations WHERE study_id=?", (sid,)).fetchone()[0]
    if len(conditions) != expected_conditions or trial_count != expected_trials:
        raise ValueError(f"TFLite study {sid} count differs")
    db.execute("INSERT INTO current_study_registry_v3 VALUES (?,?,?,?,?,?,?,?,?,?)",
               (model_key, sid, "whole_kernel_tflite_int8", path, payload,
                checkpoint, bits, metric, len(conditions), trial_count))
    for r in conditions:
        db.execute("INSERT INTO current_conditions_v3 VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                   (sid, r["benchmark_id"], checkpoint, r["precision"], r["backend"],
                    bits, metric, r["quality_name"], r["method_id"], r["method_name"],
                    r["coverage"], r["protected_bits"], r["ber"], bits*r["ber"],
                    r["field_count"], r["replication_count"], r["ranking_mask_count"],
                    clean, r["mean_loss"], r["loss_sd"], r["loss_se"],
                    r["loss_ci_low"], r["loss_ci_high"], r["loss_degradation"],
                    r["mean_quality"], r["failure_fraction"],
                    "95% t interval over fault fields; fixed checkpoint and cohort",
                    "FINALIZED_MEASURED"))
    _tflite_pairs(db, sid, Path(path), "current")
    return {"study_id": sid, "source_path": path, "scientific_payload_sha256": payload,
            "conditions": len(conditions), "trials": trial_count}


def _tflite_pairs(db: sqlite3.Connection, sid: str, root: Path, role: str) -> None:
    for r in _rows(root / "measurements/paired_comparisons.csv"):
        if r["metric"] not in ("categorical_crossentropy", "roc_auc"):
            continue
        sign = -1 if r["metric"] == "roc_auc" else 1
        lo, hi = float(r["ci_low"]), float(r["ci_high"])
        db.execute("INSERT INTO paired_contrasts_v3 VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                   (sid, role, float(r["logical_ber"]), float(r["coverage"]),
                    r["left"], r["right"], r["metric"], int(r["matched_pairs"]),
                    sign*float(r["mean_difference"]), min(sign*lo, sign*hi),
                    max(sign*lo, sign*hi), float(r["p_value_raw"]),
                    float(r["p_value_holm"]), r["multiplicity_family"],
                    "source 95% paired interval; Holm applies to p-value, not interval"))


def _focused(db: sqlite3.Connection, name: str, conditions: int, trials: int) -> dict:
    source = _tflite_source(name)
    root, manifest, summary, _ = _verified_bundle(source)
    rows = _rows(root / "measurements/protection_results.csv")
    if len(rows) != conditions or len(_rows(root / "measurements/trials.csv")) != trials:
        raise ValueError(f"focused study {name} is incomplete")
    for r in rows:
        toycar = "toycar" in name
        quality = float(r["primary_metric_mean"]) if toycar else float(r["accuracy_mean"])
        loss = 1-quality if toycar else float(r["loss_mean"])
        low = 1-float(r["primary_metric_ci_high"]) if toycar else float(r["loss_ci_low"])
        high = 1-float(r["primary_metric_ci_low"]) if toycar else float(r["loss_ci_high"])
        db.execute("INSERT INTO focused_conditions_v3 VALUES (?,?,?,?,?,?,?,?,?)",
                   (name, r["policy"], float(r["coverage"]), float(r["logical_ber"]),
                    int(r["trial_count"]), loss, quality, low, high))
    _tflite_pairs(db, name, root, "focused_paired_precision")
    return {"study_id": name, "source_path": str(root),
            "scientific_payload_sha256": source["scientific_payload_sha256"],
            "conditions": conditions, "trials": trials,
            "task_loss_name": "one_minus_file_roc_auc" if "toycar" in name else "categorical_crossentropy"}


def _transformer(db: sqlite3.Connection, spec: tuple) -> dict:
    key, _, run_name, _, bits = spec
    plan, receipt, payload, run, report, admission = _verify_transformer(spec)
    sid = run_name
    observations = _rows(report / "field_observations.csv")
    conditions = _rows(report / "condition_level.csv")
    if len(observations) != 1872 or len(conditions) != 78:
        raise ValueError(f"{key} report rows incomplete")
    clean = json.loads((run / "final/clean.json").read_text())["risk"]["metrics"]
    clean_loss = float(clean.get("mean_nll", clean.get("mean_cross_entropy")))
    field_keys: dict[tuple[float, int], tuple[int, str]] = {}
    groups: dict[tuple[float, str, float], list[float]] = {}
    masks: dict[tuple[str, float], set[str]] = {}
    for r in observations:
        method = ALIASES.get(r["method"], r["method"])
        ber, coverage = float(r["ber"]), float(r["coverage"])
        field_index = int(r["field_index"])
        proposed, applied, suppressed = (int(r[k]) for k in
                                          ("proposed_faults", "applied_faults", "suppressed_faults"))
        if proposed != applied+suppressed or \
           (method == "full_protection" and applied != 0):
            raise ValueError(f"{key} fault accounting differs")
        identity = (int(r["seed"]), r["field_sha256"])
        prior = field_keys.setdefault((ber, field_index), identity)
        if identity != prior:
            raise ValueError(f"{key} policy fields are not paired")
        loss = float(r.get("mean_nll", r.get("mean_cross_entropy")))
        if method == "full_protection" and not math.isclose(loss, clean_loss, abs_tol=1e-12):
            raise ValueError(f"{key} full protection differs from clean")
        quality = float(r["accuracy"]) if key == "distilbert" else None
        failed = int(r.get("failed_tokens", r.get("failed_examples")))
        total = int(r.get("scored_tokens", r.get("evaluated_examples")))
        db.execute("INSERT INTO transformer_trials_v3 VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                   (sid, method, coverage, ber, field_index, identity[0], identity[1],
                    r["mask_sha256"], proposed, applied, suppressed, loss, quality,
                    failed, total, r["field_record"]))
        groups.setdefault((ber, method, coverage), []).append(loss)
        masks.setdefault((method, coverage), set()).add(r["mask_sha256"])
    if len(field_keys) != 72 or any(len(values) != 24 for values in groups.values()) or \
       len(groups) != 78 or any(len(digests) != 1 for digests in masks.values()):
        raise ValueError(f"{key} paired field, condition, or mask count differs")
    metric = "categorical_crossentropy" if key == "distilbert" else "token_nll"
    quality_name = "accuracy" if key == "distilbert" else "none"
    db.execute("INSERT INTO current_study_registry_v3 VALUES (?,?,?,?,?,?,?,?,?,?)",
               (key, sid, "whole_eligible_w8a8", str(run), payload["payload_sha256"],
                admission["checkpoint_sha256"], bits, metric, 78, 1872))
    for r in conditions:
        method = ALIASES.get(r["method"], r["method"])
        ber, coverage = float(r["ber"]), float(r["coverage"])
        values = groups[(ber, method, coverage)]
        mean = statistics.mean(values)
        source_mean = float(r.get("mean_nll", r.get("mean_cross_entropy")))
        if not math.isclose(mean, source_mean, abs_tol=1e-10):
            raise ValueError(f"{key} condition mean differs from saved fields")
        sd = statistics.stdev(values)
        low = clean_loss + float(r["ci95_degradation_low"])
        high = clean_loss + float(r["ci95_degradation_high"])
        db.execute("INSERT INTO current_conditions_v3 VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                   (sid, plan["target"], admission["checkpoint_sha256"], "INT8 W8A8",
                    "pinned PyTorch CPU integer-linear W8A8", bits, metric, quality_name,
                    method, db.execute("SELECT display_name FROM methods WHERE method_id=?",
                                       (method,)).fetchone()[0],
                    coverage, int(r["protected_bits"]), ber,
                    float(r["expected_proposed_faults"]), int(r["independent_fields"]),
                    1, len(masks[(method, coverage)]), clean_loss, mean, sd,
                    sd/math.sqrt(len(values)), low, high,
                    float(r["mean_loss_degradation"]),
                    float(r["mean_accuracy"]) if key == "distilbert" else None,
                    float(r["failed_fraction"]),
                    "source 95% percentile bootstrap over paired fault fields; fixed checkpoint and cohort",
                    "FINALIZED_MEASURED"))
    analysis = json.loads((run / "analysis.json").read_text())
    for r in analysis["comparisons"]:
        ber = float(r["ber"])
        budget = int(r["budget_bits"])
        coverage = budget/bits
        left = ALIASES[r["method_minus_random"]]
        lo, hi = (float(x) for x in r["ci95"])
        db.execute("INSERT INTO paired_contrasts_v3 VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                   (sid, "current", ber, coverage, left, "random_global", metric, 24,
                    float(r["mean_nll_difference"]), lo, hi, float(r["p_signflip"]),
                    float(r["p_holm_60"]), "60 method-minus-random contrasts across 3 BERs and 4 budgets",
                    "source 95% paired bootstrap interval; Holm applies to p-value, not interval"))
    return {"study_id": sid, "source_path": str(run),
            "plan_sha256": plan["plan_sha256"],
            "scientific_payload_sha256": payload["payload_sha256"],
            "complete_zip_sha256": receipt.get("complete_zip_sha256", receipt.get("core_zip_sha256")),
            "compact_zip_sha256": receipt["compact_zip_sha256"],
            "conditions": 78, "trials": 1872, "task_loss_name": metric}


def _coverage(db: sqlite3.Connection) -> None:
    methods = [row[0] for row in db.execute("SELECT method_id FROM methods WHERE method_id!='fkeras_repository'")]
    for sid, kind in db.execute("SELECT study_id,study_kind FROM current_study_registry_v3"):
        bers = [row[0] for row in db.execute("SELECT DISTINCT ber FROM current_conditions_v3 WHERE study_id=?", (sid,))]
        for method in methods:
            count = db.execute("SELECT count(*) FROM current_conditions_v3 WHERE study_id=? AND method_id=?",
                               (sid, method)).fetchone()[0]
            if count:
                status, reason = "FINALIZED_MEASURED", "finalized matched whole-population study"
            elif method == "exponent_first":
                status, reason = "INCOMPATIBLE", "INT8 qweights have no exponent field"
            elif method == "priorifi_derived":
                status, reason = "NOT_IMPLEMENTED", "no validated fixed-budget PrioriFI-derived allocation"
            elif method == "fkeras_paper" and kind == "whole_eligible_w8a8":
                status, reason = "NOT_IMPLEMENTED", "no validated transformer loss-HVP/eigenpair and bit mapping"
            elif method == "empirical_isolated":
                status, reason = "NOT_MEASURED", "sampled bit effects do not define a full-universe mask"
            else:
                status, reason = "NOT_MEASURED", "no finalized matched condition"
            db.execute("INSERT INTO current_method_coverage_v3 VALUES (?,?,?,?,?)",
                       (sid, method, status, count, reason))
            for ber in bers:
                for coverage in BUDGETS:
                    measured = db.execute("""SELECT 1 FROM current_conditions_v3
                        WHERE study_id=? AND method_id=? AND coverage=? AND ber=?""",
                        (sid, method, coverage, ber)).fetchone() is not None
                    if measured:
                        cell = "FINALIZED_MEASURED"
                    elif method == "no_protection" and coverage != 0 or \
                         method == "full_protection" and coverage != 1 or \
                         method not in ("no_protection", "full_protection") and coverage in (0, 1):
                        cell = "INCOMPATIBLE"
                    else:
                        cell = status
                    db.execute("INSERT INTO current_coverage_matrix_v3 VALUES (?,?,?,?,?)",
                               (sid, method, coverage, ber, cell))


def _export(db: sqlite3.Connection, directory: Path, table: str, filename: str) -> int:
    cur = db.execute(f"SELECT * FROM {table}")
    with (directory / filename).open("x", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([column[0] for column in cur.description])
        rows = 0
        for row in cur:
            writer.writerow(row)
            rows += 1
    return rows


def build(output: Path = OUTPUT) -> dict:
    if output.exists():
        raise FileExistsError(f"write-once comparative product exists: {output}")
    checks = json.loads((BASE / "checksums.json").read_text())
    old_db = BASE / "current_comparative.sqlite"
    if sha256_file(old_db) != checks["files"][old_db.name] or \
       (BASE / "FINALIZED").read_text().strip() != "database_sha256="+sha256_file(old_db):
        raise ValueError("v2 database source is not sealed")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".m12-v3-", dir=output.parent) as temp:
        stage = Path(temp) / output.name
        stage.mkdir()
        database = stage / "current_comparative.sqlite"
        with sqlite3.connect(f"file:{old_db}?mode=ro", uri=True) as source, \
             sqlite3.connect(database) as target:
            source.backup(target)
        provenance = {"schema": "m12-current-comparative/3.0.0",
                      "base_v2_database_sha256": sha256_file(old_db),
                      "main_studies": [], "focused_studies": []}
        with sqlite3.connect(database) as db:
            db.row_factory = sqlite3.Row
            if db.execute("SELECT count(*) FROM primary_conditions").fetchone()[0] != 280 or \
               db.execute("SELECT count(*) FROM current_comparative_conditions_v2").fetchone()[0] != 552:
                raise ValueError("historical primary or v2 comparative view changed")
            _schema(db)
            for key, sid, conditions, trials in TFLITE:
                source = _tflite_source(sid) if key != "kws" else None
                provenance["main_studies"].append(_tflite(db, key, sid, conditions, trials, source))
            for name, conditions, trials in FOCUSED:
                provenance["focused_studies"].append(_focused(db, name, conditions, trials))
            for spec in W8A8:
                provenance["main_studies"].append(_transformer(db, spec))
            _coverage(db)
            counts = {name: db.execute(f"SELECT count(*) FROM {name}").fetchone()[0] for name in
                      ("current_conditions_v3", "current_trial_observations_v3",
                       "paired_contrasts_v3", "focused_conditions_v3",
                       "current_method_coverage_v3", "current_coverage_matrix_v3")}
            if counts["current_conditions_v3"] != 864 or \
               counts["current_trial_observations_v3"] != 25776 or \
               counts["focused_conditions_v3"] != 32 or \
               counts["current_method_coverage_v3"] != 72:
                raise ValueError(f"current grid counts differ: {counts}")
            if db.execute("SELECT count(*) FROM primary_conditions").fetchone()[0] != 280 or \
               db.execute("SELECT count(*) FROM current_comparative_conditions_v2").fetchone()[0] != 552 or \
               db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("historical view or SQLite integrity changed")
            db.execute("INSERT INTO metadata VALUES (?,?)",
                       ("current_comparative_schema_v3", "m12-current-comparative/3.0.0"))
            db.commit()
            provenance["counts"] = counts
            exports = {}
            for table, filename in (("current_conditions_v3", "current_conditions.csv"),
                                    ("current_trial_observations_v3", "current_trials.csv"),
                                    ("paired_contrasts_v3", "paired_contrasts.csv"),
                                    ("focused_conditions_v3", "focused_conditions.csv"),
                                    ("current_method_coverage_v3", "method_coverage.csv"),
                                    ("current_coverage_matrix_v3", "coverage_matrix.csv"),
                                    ("current_study_registry_v3", "study_registry.csv")):
                exports[filename] = _export(db, stage, table, filename)
            provenance["csv_rows"] = exports
        (stage / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True)+"\n")
        readme = ("# Current comparative reliability database v3\n\n"
                  "One selected finalized whole-eligible-population study per benchmark: "
                  "ResNet-8, KWS, ToyCar, DistilBERT W8A8, Pythia W8A8 and Qwen W8A8. "
                  "The first three use exact TFLite INT8 kernel XOR; the latter three "
                  "use pinned PyTorch CPU W8A8 stored-qweight XOR. Task losses have different "
                  "units and are never averaged across benchmarks.\n\n"
                  "`current_comparative_conditions` and `current_comparative_conditions_v3` "
                  "contain 864 measured conditions and `current_trial_observations_v3` "
                  "contains 25,776 policy-field observations. Historical "
                  "`primary_conditions` (280) and `current_comparative_conditions_v2` "
                  "(552) remain intact in this copied database; the original v2 artifact "
                  "has not been edited. The 320-field ResNet-8 and 384-field ToyCar "
                  "focused comparisons are separate supplemental tables, not pooled.\n\n"
                  "Condition intervals are source t intervals (TFLite) or source field-bootstrap "
                  "intervals (W8A8), conditional on fixed checkpoints, cohorts and masks. "
                  "Paired contrast p-values use each source's original within-BER Holm family "
                  "(TFLite) or 60-comparison family (W8A8); their displayed 95% intervals "
                  "are not multiplicity-adjusted. ToyCar source AUC contrasts are negated "
                  "in `paired_contrasts_v3` so negative always means lower task risk. "
                  "No cross-model p-values or mixed-unit means are computed.\n\n"
                  "The 12.5% and 25% published-form FKeras masks are identical to MSB on "
                  "homogeneous INT8; duplicate rows are retained as measured identities, "
                  "not independent policies. FKeras remains unimplemented on W8A8 transformers; "
                  "PrioriFI-derived allocation remains unmeasured everywhere.\n")
        (stage / "README.md").write_text(readme)
        files = {p.name: sha256_file(p) for p in stage.iterdir() if p.is_file()}
        (stage / "checksums.json").write_text(json.dumps({"schema": "m12-current-comparative-checksums/v3",
                                                      "files": files}, indent=2, sort_keys=True)+"\n")
        (stage / "FINALIZED").write_text("database_sha256="+files[database.name]+"\n")
        os.rename(stage, output)
    return {"output": str(output), "database_sha256": files[database.name], **provenance["counts"]}


if __name__ == "__main__":
    print(json.dumps(build(), sort_keys=True))
