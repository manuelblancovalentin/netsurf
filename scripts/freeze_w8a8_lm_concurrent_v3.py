"""Freeze M12 co-execution plans while preserving every v1 scientific field."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from netsurf.manifests import canonical_json_bytes, digest_object, sha256_file


ROOT = Path(__file__).resolve().parent.parent
STEMS = {"pythia": "pythia160m_w8a8_reliability",
         "qwen": "qwen3_0_6b_w8a8_reliability"}
NEW_SOURCE = ("scripts/freeze_w8a8_lm_concurrent_v3.py",
              "scripts/admit_w8a8_lm_v3.py",
              "scripts/w8a8_phase_entry_v2.py",
              "scripts/w8a8_concurrent_gate_v2.py",
              "scripts/freeze_distilbert_w8a8_concurrent_v2.py")


def locations(model: str) -> tuple[Path, Path]:
    stem = STEMS[model]
    return ROOT / "workspace/plans" / (stem + "_v1"), ROOT / "workspace/plans" / (stem + "_v3")


def scientific_payload(plan: dict) -> dict:
    return {k: v for k, v in plan.items()
            if k not in ("plan_sha256", "resource_envelope", "source_files_sha256")}


def admission(model: str) -> dict:
    ram, disk = (12, 10) if model == "pythia" else (64, 40)
    return {"cpu_threads": 4, "distinct_physical_cores": 4,
            "max_total_cpu_busy_fraction": 0.5,
            "max_load1_per_physical_core": 0.7, "max_runnable_tasks": 40,
            "max_m11_worker_cpu_cores": 1.5, "max_m12_kws_cpu_cores": 4.0,
            "max_other_major_model_cpu_cores": 4.0,
            "max_other_major_model_rss_gib": 16,
            "max_new_process_rss_gib": ram,
            "min_memory_available_gib": 2*ram,
            "min_disk_free_gib": disk,
            "monitor_interval_seconds": 2, "contention_sample_seconds": 1,
            "torch_interop_threads": 1}


def derive(model: str, v1: dict) -> tuple[dict, dict]:
    expected_target = "pythia160m_int8" if model == "pythia" else "qwen3_0_6b_int8"
    if v1.get("target") != expected_target or \
       v1.get("schema") != "netsurf-w8a8-whole-qweight-campaign/v1":
        raise ValueError("wrong frozen v1 LM plan")
    v1_dir, _ = locations(model)
    v2 = json.loads(json.dumps(v1))
    for name in NEW_SOURCE:
        v2["source_files_sha256"][name] = sha256_file(ROOT / name)
    v2["resource_envelope"]["contention_gate"] = (
        "measured M12 model co-execution up to four total model CPU cores; "
        "Pythia and Qwen campaigns remain sequential; new campaign pinned to four physical cores")
    v2["resource_envelope"]["admission"] = admission(model)
    v2["resource_envelope"]["parent_plan_file_sha256"] = sha256_file(v1_dir / "plan.json")
    v2["plan_sha256"] = digest_object({k: v for k, v in v2.items() if k != "plan_sha256"})
    if scientific_payload(v1) != scientific_payload(v2) or any(
        v2["source_files_sha256"].get(k) != value
        for k, value in v1["source_files_sha256"].items()) or any(
        v2["resource_envelope"].get(k) != value for k, value in
        v1["resource_envelope"].items() if k != "contention_gate"):
        raise ValueError("v2 plan changed scientific fields, original source or resource limits")
    audit = {"schema": "netsurf-w8a8-lm-execution-equivalence/v1",
             "v1_plan_file_sha256": sha256_file(v1_dir / "plan.json"),
             "v1_plan_sha256": v1["plan_sha256"], "v2_plan_sha256": v2["plan_sha256"],
             "scientific_payload_sha256": digest_object(scientific_payload(v1)),
             "scientific_fields_equal": True,
             "unchanged_source_bindings": v1["source_files_sha256"],
             "added_source_bindings": {name: v2["source_files_sha256"][name] for name in NEW_SOURCE},
             "changed_paths": ["plan_sha256", "resource_envelope.contention_gate",
                               "resource_envelope.admission",
                               "resource_envelope.parent_plan_file_sha256",
                               *["source_files_sha256." + name for name in NEW_SOURCE]],
             "v1_contention_gate": v1["resource_envelope"]["contention_gate"],
             "v2_contention_gate": v2["resource_envelope"]["contention_gate"],
             "v2_admission": admission(model)}
    return v2, audit


def verify(model: str) -> dict:
    v1_dir, v2_dir = locations(model)
    for folder in (v1_dir, v2_dir):
        if sha256_file(folder / "plan.json") != (folder / "FROZEN").read_text().strip():
            raise ValueError("frozen plan checksum changed")
    v1 = json.loads((v1_dir / "plan.json").read_text())
    v2, audit = derive(model, v1)
    if json.loads((v2_dir / "plan.json").read_text()) != v2 or \
       json.loads((v2_dir / "equivalence_audit.json").read_text()) != audit:
        raise ValueError("scientific-equivalence audit or v2 plan changed")
    for name in ("target_ready_record.json", "quantization_manifest.json"):
        if sha256_file(v1_dir / name) != sha256_file(v2_dir / name):
            raise ValueError("frozen target asset snapshot changed")
    return audit


def freeze(model: str) -> dict:
    v1_dir, v2_dir = locations(model)
    if v2_dir.exists():
        raise ValueError("write-once v2 plan already exists")
    if sha256_file(v1_dir / "plan.json") != (v1_dir / "FROZEN").read_text().strip():
        raise ValueError("v1 plan checksum changed")
    v2, audit = derive(model, json.loads((v1_dir / "plan.json").read_text()))
    v2_dir.mkdir(parents=True, exist_ok=False)
    for name in ("target_ready_record.json", "quantization_manifest.json"):
        shutil.copyfile(v1_dir / name, v2_dir / name)
    (v2_dir / "plan.json").write_bytes(canonical_json_bytes(v2)+b"\n")
    (v2_dir / "equivalence_audit.json").write_bytes(canonical_json_bytes(audit)+b"\n")
    (v2_dir / "FROZEN").write_text(sha256_file(v2_dir / "plan.json")+"\n")
    verify(model)
    return {"model": model, "plan_sha256": v2["plan_sha256"],
            "plan_file_sha256": sha256_file(v2_dir / "plan.json"),
            "audit_sha256": sha256_file(v2_dir / "equivalence_audit.json")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("freeze", "verify"))
    parser.add_argument("model", choices=tuple(STEMS))
    args = parser.parse_args()
    print(json.dumps(freeze(args.model) if args.phase == "freeze" else verify(args.model),
                     sort_keys=True))
