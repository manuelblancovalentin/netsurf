"""Derive the concurrent DistilBERT execution plan without changing v1 science."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from netsurf.manifests import canonical_json_bytes, digest_object, sha256_file


ROOT = Path(__file__).resolve().parent.parent
V1 = ROOT / "workspace/plans/distilbert_sst2_w8a8_reliability_v1"
V2 = ROOT / "workspace/plans/distilbert_sst2_w8a8_reliability_v2"
NEW_SOURCE = (
    "scripts/freeze_distilbert_w8a8_concurrent_v2.py",
    "scripts/w8a8_concurrent_gate_v2.py",
    "scripts/w8a8_phase_entry_v2.py",
    "scripts/run_distilbert_w8a8_concurrent_v2.sh",
    "scripts/report_distilbert_w8a8_v2.py",
)
ADMISSION = {
    "cpu_threads": 4,
    "distinct_physical_cores": 4,
    "max_total_cpu_busy_fraction": 0.5,
    "max_load1_per_physical_core": 0.7,
    "max_runnable_tasks": 40,
    "max_m11_worker_cpu_cores": 1.5,
    "max_m12_kws_cpu_cores": 4.0,
    "max_other_major_model_cpu_cores": 4.0,
    "min_memory_available_gib": 16,
    "max_new_process_rss_gib": 8,
    "min_disk_free_gib": 5,
    "monitor_interval_seconds": 2,
    "contention_sample_seconds": 1,
    "torch_interop_threads": 1,
}


def scientific_payload(plan: dict) -> dict:
    return {k: v for k, v in plan.items()
            if k not in ("plan_sha256", "resource_envelope", "source_files_sha256")}


def derive(v1: dict, source_root: Path = ROOT) -> tuple[dict, dict]:
    if v1.get("target") != "distilbert_sst2_int8" or \
       v1.get("schema") != "netsurf-w8a8-whole-qweight-campaign/v1":
        raise ValueError("v1 DistilBERT W8A8 plan required")
    v2 = json.loads(json.dumps(v1))
    for name in NEW_SOURCE:
        v2["source_files_sha256"][name] = sha256_file(source_root / name)
    v2["resource_envelope"]["contention_gate"] = (
        "measured admission permits M11 and M12 KWS under frozen thresholds; "
        "new process pinned to four distinct physical cores")
    v2["resource_envelope"]["admission"] = ADMISSION
    v2["resource_envelope"]["parent_plan_file_sha256"] = sha256_file(V1 / "plan.json")
    v2["plan_sha256"] = digest_object({k: v for k, v in v2.items() if k != "plan_sha256"})
    if scientific_payload(v1) != scientific_payload(v2):
        raise ValueError("scientific v1 fields changed in concurrent plan")
    if any(v2["source_files_sha256"].get(k) != value
           for k, value in v1["source_files_sha256"].items()):
        raise ValueError("v1 source binding changed in concurrent plan")
    if any(v2["resource_envelope"].get(k) != value for k, value in
           v1["resource_envelope"].items() if k != "contention_gate"):
        raise ValueError("v1 resource limit changed in concurrent plan")
    audit = {
        "schema": "netsurf-w8a8-execution-equivalence/v1",
        "v1_plan_file_sha256": sha256_file(V1 / "plan.json"),
        "v1_plan_sha256": v1["plan_sha256"],
        "v2_plan_sha256": v2["plan_sha256"],
        "scientific_payload_sha256": digest_object(scientific_payload(v1)),
        "scientific_fields_equal": True,
        "unchanged_source_bindings": v1["source_files_sha256"],
        "added_source_bindings": {k: v2["source_files_sha256"][k] for k in NEW_SOURCE},
        "changed_paths": ["plan_sha256", "resource_envelope.contention_gate",
                          "resource_envelope.admission",
                          "resource_envelope.parent_plan_file_sha256",
                          *["source_files_sha256." + name for name in NEW_SOURCE]],
        "v1_contention_gate": v1["resource_envelope"]["contention_gate"],
        "v2_contention_gate": v2["resource_envelope"]["contention_gate"],
        "v2_admission": ADMISSION,
    }
    return v2, audit


def verify(v1_dir: Path = V1, v2_dir: Path = V2) -> dict:
    v1_path, v2_path = v1_dir / "plan.json", v2_dir / "plan.json"
    if sha256_file(v1_path) != (v1_dir / "FROZEN").read_text().strip() or \
       sha256_file(v2_path) != (v2_dir / "FROZEN").read_text().strip():
        raise ValueError("frozen plan checksum changed")
    v1, actual = json.loads(v1_path.read_text()), json.loads(v2_path.read_text())
    expected, audit = derive(v1)
    if actual != expected or json.loads((v2_dir / "equivalence_audit.json").read_text()) != audit:
        raise ValueError("v2 plan or machine-readable scientific-equivalence audit changed")
    for name in ("target_ready_record.json", "quantization_manifest.json"):
        if sha256_file(v1_dir / name) != sha256_file(v2_dir / name):
            raise ValueError(f"frozen {name} changed between v1 and v2")
    return audit


def freeze(v1_dir: Path = V1, v2_dir: Path = V2) -> dict:
    if v2_dir.exists():
        raise ValueError("v2 write-once plan destination already exists")
    v1_path = v1_dir / "plan.json"
    if sha256_file(v1_path) != (v1_dir / "FROZEN").read_text().strip():
        raise ValueError("v1 plan checksum changed")
    v1 = json.loads(v1_path.read_text())
    v2, audit = derive(v1)
    v2_dir.mkdir(parents=True, exist_ok=False)
    for name, obj in (("plan.json", v2), ("equivalence_audit.json", audit)):
        (v2_dir / name).write_bytes(canonical_json_bytes(obj) + b"\n")
    for name in ("target_ready_record.json", "quantization_manifest.json"):
        shutil.copyfile(v1_dir / name, v2_dir / name)
    (v2_dir / "FROZEN").write_text(sha256_file(v2_dir / "plan.json") + "\n")
    verify(v1_dir, v2_dir)
    return {"v1_plan_sha256": v1["plan_sha256"], "v2_plan_sha256": v2["plan_sha256"],
            "v2_plan_file_sha256": sha256_file(v2_dir / "plan.json"),
            "scientific_payload_sha256": audit["scientific_payload_sha256"],
            "changed_paths": audit["changed_paths"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("freeze", "verify"))
    args = parser.parse_args()
    print(json.dumps(freeze() if args.phase == "freeze" else verify(), sort_keys=True))
