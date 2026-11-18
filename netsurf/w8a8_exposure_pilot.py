"""Frozen, unprotected full-qweight DistilBERT fault-exposure pilot.

No ranking is constructed here.  Each field is one independent whole-model
Bernoulli draw held fixed across all 128 labeled SST-2 examples.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
import zipfile

import numpy as np

from .manifests import canonical_json_bytes, digest_object, sha256_file
from .transformer_data import read_sst2_parquet, sst2_records_digest
from .w8a8_assets import INTERPRETER, load_ready_w8a8
from .w8a8_faults import BoundaryMask, W8A8FaultAdapter, W8A8Universe, bernoulli_field


ROOT = Path(__file__).resolve().parent.parent
TARGET = "distilbert_sst2_int8"
LAMBDAS = (1024, 4096, 16384, 65536)
FIELDS_PER_ARM = 16
MAX_EVENTS = 131072
SEED_BASE = 9701001
PILOT_INDICES = tuple(range(544, 672))
RESERVED_FINAL_INDICES = tuple(range(672, 872))
SOURCE_FILES = (
    "netsurf/manifests.py", "netsurf/transformer_data.py", "netsurf/transformer_risk.py",
    "netsurf/w8a8_assets.py", "netsurf/w8a8_faults.py",
    "netsurf/w8a8_exposure_pilot.py", "scripts/run_distilbert_w8a8_exposure_pilot_v2.sh",
    "scripts/w8a8_exposure_pilot_gate_v2.py",
)


def _write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical_json_bytes(obj) + b"\n")


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _write_bound(path: Path, obj: dict) -> None:
    _write(path, {**obj, "payload_sha256": digest_object(obj)})


def _read_bound(path: Path) -> dict:
    obj = _read(path)
    if digest_object({k: v for k, v in obj.items() if k != "payload_sha256"}) != obj.get("payload_sha256"):
        raise ValueError(f"immutable pilot member changed: {path}")
    return obj


def _rows(asset):
    path = asset.dataset_root / "sst2/validation-00000-of-00001.parquet"
    rows = read_sst2_parquet(path, "validation")
    if len(rows) != 872 or tuple(r.source_index for r in rows) != tuple(range(872)):
        raise ValueError("pinned SST-2 validation source indices changed")
    return rows


def _cohort(rows, indices, role):
    chosen = [rows[i] for i in indices]
    return {"role": role, "split": "validation", "source_indices": list(indices),
            "ordered_labels_and_text_sha256": sst2_records_digest(chosen), "size": len(chosen),
            "prior_exposure": "Kona clean inference on all 872; prior FP32 fault outcomes on subsets. No virgin-input claim."}


def _expected_plan(manifest: Path) -> tuple[dict, object]:
    asset = load_ready_w8a8(manifest, TARGET)
    universe = W8A8Universe(asset)
    if universe.bits != 344469504 or asset.modules != 38:
        raise ValueError("complete declared DistilBERT W8A8 qweight universe changed")
    rows = _rows(asset)
    arms = []
    for arm_index, lam in enumerate(LAMBDAS):
        ber = lam / universe.bits
        fields = []
        for i in range(FIELDS_PER_ARM):
            seed = SEED_BASE + 100000 * arm_index + i
            field = bernoulli_field(universe.bits, ber, seed, max_events=MAX_EVENTS)
            fields.append({"index": i, "seed": seed, "sha256": field.sha256,
                           "proposed": len(field.ordinals)})
        arms.append({"expected_proposed": lam, "ber": ber, "fields": fields,
                     "actual_proposed_total": sum(f["proposed"] for f in fields)})
    seeds = [f["seed"] for arm in arms for f in arm["fields"]]
    if len(set(seeds)) != len(seeds) or any(seed in range(1203001, 1203100) for seed in seeds):
        raise ValueError("pilot seed collision")
    plan = {
        "schema": "netsurf-w8a8-distilbert-exposure-pilot/v1",
        "status": "FROZEN_NOT_RUN", "target": TARGET,
        "target_record_sha256": asset.record_sha256,
        "ready_manifest_sha256_at_freeze": asset.manifest_sha256,
        "model_commit": asset.record["model"]["commit"],
        "model_files": {item["path"]: item["sha256"] for item in asset.record["model"]["files"]},
        "quantization_manifest_sha256": sha256_file(asset.model_root / "quantization_manifest.json"),
        "eligible_universe": {"representation": "stored signed INT8 qweight codes only",
                              "bits": universe.bits, "codes": universe.codes,
                              "modules": asset.modules, "sha256": universe.digest,
                              "outside": asset.record["eligible_stored_int8_population"]["outside_population"]},
        "pilot": _cohort(rows, PILOT_INDICES, "pilot_only"),
        "reserved_final": _cohort(rows, RESERVED_FINAL_INDICES, "reserved_no_pilot_inference"),
        "exposure_audit": {"prior_w8a8_final_indices": [32, 543],
                           "prior_fp32_final_or_selection_overlap": True,
                           "pilot_final_disjoint": True,
                           "historical_artifacts_unchanged": True},
        "fault_law": "independent Bernoulli over every eligible stored INT8 qweight bit; geometric-gap exact sampler",
        "arms": arms, "max_events_per_field": MAX_EVENTS,
        "policies": ["unprotected", "full_protection_control"],
        "full_protection_control": "first field of each BER arm only, same proposed schedule, all events suppressed",
        "clean_controls": "same 128 pilot clips before and after all fault trials",
        "ranking_methods": [], "protection_budget_bits": [0, universe.bits],
        "primary_metric": "mean cross-entropy in nats over 128 labeled SST-2 examples",
        "secondary_metrics": ["accuracy", "failed_examples", "failure_fraction", "applied_fault_count"],
        "numerical_failure_policy": "nonfinite logits or runtime-failed batch: 30 nats per affected example and incorrect prediction; retain failure kind",
        "statistics": {"unit": "independent whole-model fault field within each BER; clips fixed and shared",
                       "summary": "field mean and SD, t-interval 95% for CE degradation, median, one-field-per-tail trimmed mean, largest field and leave-largest-out",
                       "status": "exploratory exposure calibration; no ranking-method hypothesis test"},
        "regime_rule": {"first_informative": "smallest prespecified lambda with mean unprotected CE increase >=0.02 nats and failed-example fraction <=0.10",
                        "bracket": "report that arm and immediately lower prespecified arm; if first arm qualifies, report first two; if none qualifies, unresolved",
                        "no_method_outcome_selection": True,
                        "later_final": "new plan, six methods, fresh independent fields and reserved indices 672..871; no final run from this pilot"},
        "counts": {"ber_arms": 4, "independent_unprotected_fields": 64,
                   "full_protection_control_requests": 4, "clean_requests": 2,
                   "model_requests": 70, "clips_per_request": 128,
                   "clip_inference_requests": 8960,
                   "expected_unique_proposed_faults": FIELDS_PER_ARM * sum(LAMBDAS),
                   "actual_unique_proposed_faults": sum(a["actual_proposed_total"] for a in arms)},
        "resources": {"cpu_threads": 4, "max_rss_gib": 8, "min_free_disk_gib": 5,
                      "estimated_wall_minutes": [10, 60],
                      "estimate_status": "conservative extrapolation from exposed-input high-lambda probe; not a full-cohort benchmark",
                      "engineering_probe": {"validation_indices": list(range(8)), "seed": 9700001,
                          "lambda": 65536, "ber": 65536/universe.bits,
                          "schedule_sha256": "484960d34e5f472b25d6b4bca00f48e9b9605473471b5606f2a108b01812d28d",
                          "proposed_applied": 65875, "suppressed": 0,
                          "load_seconds": 7.949, "clean_8_seconds": 0.188,
                          "fault_8_seconds_including_mutation_and_restoration": 3.433,
                          "restored_8_seconds": 0.199, "peak_rss_bytes": 1904881664,
                          "exact_clean_output_restored": True,
                          "engineering_only": True}},
        "source_files_sha256": {name: sha256_file(ROOT / name) for name in SOURCE_FILES},
        "m12_handoff": "distinct exploratory pilot; central M12 database owns ingestion; no effectiveness claim",
    }
    plan["plan_sha256"] = digest_object(plan)
    return plan, asset


def freeze(manifest: Path, plan_dir: Path) -> dict:
    if plan_dir.exists():
        raise ValueError("write-once frozen pilot destination exists")
    plan, asset = _expected_plan(manifest)
    plan_dir.mkdir(parents=True, exist_ok=False)
    _write(plan_dir / "plan.json", plan)
    _write(plan_dir / "target_ready_record.json", asset.record)
    _write(plan_dir / "quantization_manifest.json", asset.quantization)
    (plan_dir / "FROZEN").write_text(sha256_file(plan_dir / "plan.json") + "\n")
    return {"plan_sha256": plan["plan_sha256"], "plan_file_sha256": sha256_file(plan_dir / "plan.json"),
            "counts": plan["counts"]}


def validate(manifest: Path, plan_dir: Path) -> tuple[dict, object]:
    path = plan_dir / "plan.json"
    plan = _read(path)
    if (sha256_file(path) != (plan_dir / "FROZEN").read_text().strip() or
        digest_object({k: v for k, v in plan.items() if k != "plan_sha256"}) != plan.get("plan_sha256")):
        raise ValueError("frozen pilot plan checksum changed")
    expected, asset = _expected_plan(manifest)
    # The global readiness manifest may acquire unrelated target records.
    # The full relevant target record and every scientific plan field must agree.
    expected["ready_manifest_sha256_at_freeze"] = plan["ready_manifest_sha256_at_freeze"]
    expected["plan_sha256"] = digest_object({k: v for k, v in expected.items() if k != "plan_sha256"})
    if (plan != expected or _read(plan_dir / "target_ready_record.json") != asset.record or
        _read(plan_dir / "quantization_manifest.json") != asset.quantization):
        raise ValueError("pilot target, source, schedule, population or input identity changed")
    return plan, asset


def _risk_summary(risk) -> dict:
    return {"cross_entropy_nats": risk.mean_nll, "accuracy": risk.accuracy,
            "failed_examples": risk.failed_units, "total_examples": risk.total_units,
            "failure_fraction": risk.failure_fraction,
            "failure_kinds": {kind: risk.failure_kinds.count(kind) for kind in
                              ("nonfinite_logits", "inference_failure")}}


def _field(plan, arm_index: int, field_index: int):
    arm = plan["arms"][arm_index]
    spec = arm["fields"][field_index]
    field = bernoulli_field(plan["eligible_universe"]["bits"], arm["ber"], spec["seed"],
                            max_events=plan["max_events_per_field"])
    if field.sha256 != spec["sha256"] or len(field.ordinals) != spec["proposed"]:
        raise ValueError("frozen exact Bernoulli schedule changed")
    return field


def _ordinal_bytes(field) -> bytes:
    """Exact sorted full-universe bit ordinals, little-endian uint32."""
    return np.asarray(field.ordinals, dtype="<u4").tobytes()


def run(manifest: Path, plan_dir: Path, run_dir: Path) -> dict:
    plan, asset = validate(manifest, plan_dir)
    if run_dir.exists() or run_dir.with_name(run_dir.name + "_export").exists():
        raise ValueError("write-once pilot run or export destination exists")
    adapter = W8A8FaultAdapter(asset)
    adapter.load()
    rows = _rows(asset)
    cohort = [rows[i] for i in PILOT_INDICES]
    unprotected = BoundaryMask(adapter.universe.bits, False)
    protected = BoundaryMask(adapter.universe.bits, True)
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_bound(run_dir / "identity.json", {"plan_sha256": plan["plan_sha256"],
                                                "target_record_sha256": asset.record_sha256,
                                                "pilot_cohort_sha256": plan["pilot"]["ordered_labels_and_text_sha256"]})
    started = time.monotonic()
    clean_start = adapter.evaluate_sst2(cohort)
    _write_bound(run_dir / "clean_start.json", {"risk": _risk_summary(clean_start),
                                                 "logits_sha256": digest_object(clean_start.logits)})
    for arm_index, arm in enumerate(plan["arms"]):
        for spec in arm["fields"]:
            i = spec["index"]
            name = f"lambda_{arm['expected_proposed']}/field_{i:02d}"
            field = _field(plan, arm_index, i)
            raw = _ordinal_bytes(field)
            schedule_path = run_dir / f"{name}.ordinals.u32le"
            schedule_path.parent.mkdir(parents=True, exist_ok=True)
            with schedule_path.open("xb") as output:
                output.write(raw)
            _write_bound(run_dir / f"{name}.intent.json", {"plan_sha256": plan["plan_sha256"],
                "field_sha256": field.sha256, "seed": field.seed, "ber": field.ber,
                "ordinals_sha256": digest_object(field.ordinals),
                "ordinals_u32le_sha256": hashlib.sha256(raw).hexdigest(),
                "proposed": len(field.ordinals)})
            begin = time.monotonic()
            with adapter.injected(field, unprotected) as receipt:
                result = adapter.evaluate_sst2(cohort)
            if receipt["applied"] != spec["proposed"] or receipt["suppressed"] != 0:
                raise ValueError("unprotected event accounting changed")
            _write_bound(run_dir / f"{name}.json", {"plan_sha256": plan["plan_sha256"],
                "field_index": i, "expected_proposed": arm["expected_proposed"], "ber": arm["ber"],
                "field_sha256": field.sha256, "seed": field.seed, "proposed": receipt["proposed"],
                "applied": receipt["applied"], "suppressed": receipt["suppressed"],
                "mask_sha256": receipt["mask_sha256"], "restored": True,
                "touched_module_hashes_before": receipt["touched_module_hashes_before"],
                "risk": _risk_summary(result), "logits_sha256": digest_object(result.logits),
                "seconds": round(time.monotonic() - begin, 3)})
            print(json.dumps({"lambda": arm["expected_proposed"], "field": i + 1,
                              "of": FIELDS_PER_ARM, "applied": receipt["applied"],
                              "ce": result.mean_nll, "failed": result.failed_units}), flush=True)
        field = _field(plan, arm_index, 0)
        with adapter.injected(field, protected) as receipt:
            control = adapter.evaluate_sst2(cohort)
        if (receipt["applied"] != 0 or receipt["suppressed"] != len(field.ordinals) or
            control.logits != clean_start.logits):
            raise ValueError("full protection failed exact clean-output control")
        _write_bound(run_dir / f"lambda_{arm['expected_proposed']}/full_control.json",
                     {"field_sha256": field.sha256, "proposed": receipt["proposed"],
                      "applied": 0, "suppressed": receipt["suppressed"],
                      "mask_sha256": receipt["mask_sha256"], "risk": _risk_summary(control),
                      "logits_sha256": digest_object(control.logits), "restored": True})
    clean_end = adapter.evaluate_sst2(cohort)
    if clean_end.logits != clean_start.logits or adapter.baseline_hashes is None:
        raise ValueError("post-fault clean output changed")
    from kona_assets import int8_torch as execution
    if any(execution.tensor_sha256(adapter.modules[name].qweight) != value
           for name, value in adapter.baseline_hashes.items()):
        raise ValueError("post-fault INT8 bytes differ from original")
    _write_bound(run_dir / "clean_end.json", {"risk": _risk_summary(clean_end),
                                               "logits_sha256": digest_object(clean_end.logits),
                                               "all_qweights_restored": True,
                                               "seconds_total": round(time.monotonic() - started, 3)})
    return {"status": "RUN_COMPLETE_PENDING_OFFLINE_AGGREGATION", "run_dir": str(run_dir)}


def aggregate(manifest: Path, plan_dir: Path, run_dir: Path) -> dict:
    plan, _ = validate(manifest, plan_dir)
    identity = _read_bound(run_dir / "identity.json")
    if identity["plan_sha256"] != plan["plan_sha256"]:
        raise ValueError("run binds another scientific plan")
    clean = _read_bound(run_dir / "clean_start.json")
    end = _read_bound(run_dir / "clean_end.json")
    if end["logits_sha256"] != clean["logits_sha256"] or not end["all_qweights_restored"]:
        raise ValueError("clean or storage restoration failed")
    rows = []
    for arm_index, arm in enumerate(plan["arms"]):
        field0 = _field(plan, arm_index, 0)
        control = _read_bound(run_dir / f"lambda_{arm['expected_proposed']}/full_control.json")
        if (control["field_sha256"] != field0.sha256 or control["applied"] != 0 or
            control["suppressed"] != len(field0.ordinals) or control["logits_sha256"] != clean["logits_sha256"]):
            raise ValueError("full protection control changed")
        for spec in arm["fields"]:
            name = f"lambda_{arm['expected_proposed']}/field_{spec['index']:02d}"
            intent = _read_bound(run_dir / f"{name}.intent.json")
            result = _read_bound(run_dir / f"{name}.json")
            field = _field(plan, arm_index, spec["index"])
            raw = (run_dir / f"{name}.ordinals.u32le").read_bytes()
            if (raw != _ordinal_bytes(field) or
                intent["ordinals_u32le_sha256"] != hashlib.sha256(raw).hexdigest() or
                intent["field_sha256"] != field.sha256 or intent["ordinals_sha256"] != digest_object(field.ordinals) or
                result["field_sha256"] != field.sha256 or result["plan_sha256"] != plan["plan_sha256"] or
                result["proposed"] != len(field.ordinals) or result["applied"] != len(field.ordinals) or
                result["suppressed"] != 0 or not result["restored"] or
                result["mask_sha256"] != BoundaryMask(plan["eligible_universe"]["bits"], False).digest):
                raise ValueError("fault schedule or accounting changed")
            risk = result["risk"]
            if risk["total_examples"] != 128 or risk["failed_examples"] != sum(risk["failure_kinds"].values()):
                raise ValueError("SST-2 failure accounting changed")
            rows.append({"lambda": arm["expected_proposed"], "ber": arm["ber"],
                         "field_index": spec["index"], "seed": spec["seed"],
                         "schedule_sha256": field.sha256, "proposed": result["proposed"],
                         "applied": result["applied"], "suppressed": 0,
                         "cross_entropy_nats": risk["cross_entropy_nats"],
                         "delta_ce_nats": risk["cross_entropy_nats"]-clean["risk"]["cross_entropy_nats"],
                         "accuracy": risk["accuracy"], "failed_examples": risk["failed_examples"],
                         "failure_fraction": risk["failure_fraction"], "seconds": result["seconds"]})
    summaries = []
    for arm in plan["arms"]:
        group = [r for r in rows if r["lambda"] == arm["expected_proposed"]]
        effects = [r["delta_ce_nats"] for r in group]
        mean = statistics.mean(effects)
        sd = statistics.stdev(effects)
        # Fixed 16-field t interval: t_0.975,15 = 2.13145.
        half = 2.13145 * sd / math.sqrt(len(effects))
        sorted_effects = sorted(effects)
        summaries.append({"lambda": arm["expected_proposed"], "ber": arm["ber"],
            "fields": len(group), "mean_applied": statistics.mean(r["applied"] for r in group),
            "mean_ce_nats": statistics.mean(r["cross_entropy_nats"] for r in group),
            "mean_delta_ce_nats": mean, "sd_delta_ce_nats": sd,
            "delta_ce_t95": [mean-half, mean+half], "median_delta_ce_nats": statistics.median(effects),
            "trimmed_mean_delta_ce_nats": statistics.mean(sorted_effects[1:-1]),
            "largest_abs_delta_ce_nats": max(effects, key=abs),
            "leave_largest_abs_out_mean": statistics.mean([x for x in effects if x != max(effects, key=abs)])
               if effects.count(max(effects, key=abs)) == 1 else
               statistics.mean((sorted(effects, key=abs))[:-1]),
            "mean_accuracy": statistics.mean(r["accuracy"] for r in group),
            "failed_examples": sum(r["failed_examples"] for r in group),
            "mean_seconds_per_field": statistics.mean(r["seconds"] for r in group)})
    informative = next((i for i, x in enumerate(summaries) if
                        x["mean_delta_ce_nats"] >= 0.02 and
                        x["failed_examples"]/(16*128) <= 0.10), None)
    bracket = [] if informative is None else ([summaries[0]["lambda"], summaries[1]["lambda"]]
        if informative == 0 else [summaries[informative-1]["lambda"], summaries[informative]["lambda"]])
    report = {"schema": "netsurf-w8a8-exposure-pilot-analysis/v1",
              "status": "FINALIZED_EXPLORATORY_CALIBRATION", "plan_sha256": plan["plan_sha256"],
              "clean": clean["risk"], "summary": summaries,
              "regime_decision": {"first_informative_index": informative, "bracket_lambda": bracket,
                                  "unresolved": informative is None},
              "total_seconds": end["seconds_total"], "field_count": len(rows),
              "reserved_final_used": False}
    report_path = run_dir / "analysis.json"
    if report_path.exists():
        if _read_bound(report_path) != {**report, "payload_sha256": digest_object(report)}:
            raise ValueError("existing immutable analysis differs")
    else:
        _write_bound(report_path, report)
    csv_path = run_dir / "field_observations.csv"
    if not csv_path.exists():
        with csv_path.open("x", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    else:
        with csv_path.open(newline="") as source:
            if list(csv.DictReader(source)) != [{k: str(v) for k, v in row.items()} for row in rows]:
                raise ValueError("existing field CSV differs")
    return {"status": report["status"], "field_count": len(rows),
            "regime_decision": report["regime_decision"], "analysis_sha256": sha256_file(report_path)}


def export(manifest: Path, plan_dir: Path, run_dir: Path, host: str = "kona") -> dict:
    summary = aggregate(manifest, plan_dir, run_dir)
    destination = run_dir.with_name(run_dir.name + "_export")
    if destination.exists():
        raise ValueError("write-once export destination exists")
    destination.mkdir(parents=True, exist_ok=False)
    files = sorted(p for p in run_dir.rglob("*") if p.is_file())
    admission = run_dir.with_name(run_dir.name + "_admission.json")
    if not admission.is_file():
        raise ValueError("measured resource-admission receipt missing")
    admission_record = _read_bound(admission)
    if admission_record.get("plan_sha256") != _read(plan_dir / "plan.json")["plan_sha256"] or not admission_record.get("admitted"):
        raise ValueError("resource-admission receipt does not bind this pilot")
    members = {str(p.relative_to(run_dir)): sha256_file(p) for p in files}
    _write(destination / "members_sha256.json", {"schema": "netsurf-evidence-members/v1",
                                                  "plan_sha256": _read(plan_dir / "plan.json")["plan_sha256"],
                                                  "members": members})
    for kind in ("compact", "complete"):
        selected = files if kind == "complete" else [p for p in files if p.name in
            {"analysis.json", "field_observations.csv", "identity.json", "clean_start.json", "clean_end.json"}]
        archive_path = destination / f"{kind}.zip"
        with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
            archive.write(plan_dir / "plan.json", "plan/plan.json")
            archive.write(plan_dir / "FROZEN", "plan/FROZEN")
            archive.write(plan_dir / "target_ready_record.json", "plan/target_ready_record.json")
            archive.write(plan_dir / "quantization_manifest.json", "plan/quantization_manifest.json")
            archive.write(admission, "admission.json")
            archive.write(destination / "members_sha256.json", "members_sha256.json")
            for p in selected:
                archive.write(p, "run/" + str(p.relative_to(run_dir)))
        with zipfile.ZipFile(archive_path) as archive:
            if archive.testzip() is not None:
                raise ValueError("evidence ZIP CRC verification failed")
            for name in archive.namelist():
                if name.startswith("run/"):
                    if hashlib.sha256(archive.read(name)).hexdigest() != members[name[4:]]:
                        raise ValueError("evidence ZIP member changed during export")
        (destination / f"{kind}.zip.sha256").write_text(sha256_file(archive_path) + "  " + archive_path.name + "\n")
    commands = "\n".join(f"scp {host}:{destination / (kind + '.zip')} ~/Downloads/distilbert_w8a8_exposure_pilot_v2_{kind}.zip"
                         for kind in ("compact", "complete")) + "\n"
    (destination / "mac_scp_commands.txt").write_text(commands)
    _write(destination / "export_receipt.json", {"status": "VERIFIED", "summary": summary,
        "archives": {kind: sha256_file(destination / f"{kind}.zip") for kind in ("compact", "complete")},
        "members_sha256": sha256_file(destination / "members_sha256.json"), "created_utc": datetime.now(timezone.utc).isoformat()})
    return {"status": "EXPORTED", "archives": {kind: str(destination / f"{kind}.zip") for kind in ("compact", "complete")},
            "scp_commands": commands.strip()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("freeze", "validate", "run", "aggregate", "export"))
    parser.add_argument("manifest", type=Path)
    parser.add_argument("plan_dir", type=Path)
    parser.add_argument("run_dir", type=Path, nargs="?")
    args = parser.parse_args()
    if args.phase in ("run", "aggregate", "export") and args.run_dir is None:
        parser.error("run_dir required")
    if args.phase == "freeze":
        result = freeze(args.manifest, args.plan_dir)
    elif args.phase == "validate":
        plan, _ = validate(args.manifest, args.plan_dir)
        result = {"status": "VALIDATED", "plan_sha256": plan["plan_sha256"],
                  "counts": plan["counts"]}
    elif args.phase == "run":
        result = run(args.manifest, args.plan_dir, args.run_dir)
    elif args.phase == "aggregate":
        result = aggregate(args.manifest, args.plan_dir, args.run_dir)
    else:
        result = export(args.manifest, args.plan_dir, args.run_dir)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
