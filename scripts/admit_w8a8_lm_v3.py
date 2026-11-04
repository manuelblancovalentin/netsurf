"""Measured admission and foreground execution of the unchanged frozen LM v1 plans.

The scientific runner and plan remain hash-bound. This operational wrapper only
restricts its own CPU affinity and monitors its own process group and resources.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from netsurf.manifests import canonical_json_bytes, sha256_file
from netsurf.w8a8_campaign import validate
import w8a8_concurrent_gate_v2 as measured
from freeze_w8a8_lm_concurrent_v3 import admission, verify


READY = Path("/projects/kona-assets/manuelbv/NETSURF_ASSET_READINESS.json")
PYTHON = Path("/projects/kona-assets/manuelbv/envs/torch-2.9.1-cpu-py311/bin/python")
TARGETS = {
    "pythia": ("pythia160m_int8", "pythia160m_w8a8_reliability_v1", 12, 10, 8),
    "qwen": ("qwen3_0_6b_int8", "qwen3_0_6b_w8a8_reliability_v1", 64, 40, 24),
}
GIB = 1024**3


def paths(model: str) -> tuple[Path, Path, Path]:
    _, stem, _, _, _ = TARGETS[model]
    stem = stem.removesuffix("_v1") + "_v3"
    return (ROOT / "workspace/plans" / stem,
            ROOT / "workspace/runs" / (stem + "_run1"),
            ROOT / "scripts/w8a8_phase_entry_v2.py")


def _group_memory(pgid: int) -> tuple[int, list[int]]:
    rss, pids = 0, []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            parts = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            if int(parts[2]) != pgid:
                continue
            status = (entry / "status").read_text()
            for line in status.splitlines():
                if line.startswith("VmRSS:"):
                    rss += int(line.split()[1]) * 1024
                    break
            pids.append(int(entry.name))
        except (FileNotFoundError, PermissionError, ValueError, IndexError):
            continue
    return rss, pids


def check(model: str) -> tuple[dict, dict]:
    if Path(sys.executable).resolve() != PYTHON.resolve():
        raise ValueError("pinned Kona interpreter required")
    target, _, ram, disk, hours = TARGETS[model]
    plan_dir, run, runner = paths(model)
    audit = verify(model)
    plan, asset = validate(READY, plan_dir)
    if plan["target"] != target or plan["resource_envelope"]["ram_gib"] != ram or \
       plan["resource_envelope"]["free_disk_gib"] != disk or \
       plan["resource_envelope"]["foreground_hours"] != hours or \
       plan["resource_envelope"]["cpu_threads"] != 4 or \
       plan["resource_envelope"]["admission"] != admission(model) or \
       plan["resource_envelope"]["parent_plan_file_sha256"] != audit["v1_plan_file_sha256"]:
        raise ValueError("frozen model identity or execution limits changed")
    for suffix in ("", "_export", "_reporting_v1", "_admission.json", "_runtime_receipt.json"):
        if run.with_name(run.name + suffix).exists():
            raise ValueError("write-once campaign output already exists: " + str(run.with_name(run.name+suffix)))
    policy = admission(model)
    measured.ADMISSION = policy
    measured.RUN = run
    resources = measured.measure(policy["contention_sample_seconds"])
    major = resources["processes"].get("other_model", [])
    if any(item["rss_bytes"] > policy["max_other_major_model_rss_gib"]*GIB for item in major):
        resources["failures"].append("other model memory exceeds frozen co-execution limit")
    # The v2 policy permits one measured M12 model job, but never overlaps two
    # full eligible-qweight W8A8 campaigns or an unclassified major model.
    if any("w8a8_campaign" in item["command"] or "admit_w8a8_lm" in item["command"]
           for item in major):
        resources["failures"].append("another W8A8 campaign is active")
    occupied = {item["last_cpu"] for role in ("m11_worker", "m12_kws_seven_method", "other_model")
                for item in resources["processes"].get(role, [])}
    resources["selected_cpus"] = measured._select_cores(set(os.sched_getaffinity(0)), occupied, 4)
    resources["selected_physical_cores"] = [list(measured._physical_key(cpu))
                                            for cpu in resources["selected_cpus"]]
    resources["proposed_new_affinity"] = ",".join(map(str, resources["selected_cpus"]))
    resources["admitted"] = not resources["failures"]
    checkpoint = next(file["sha256"] for file in asset.record["model"]["files"]
                      if file["path"] == "model.safetensors")
    identity = {"schema": "netsurf-w8a8-lm-operational-admission/v1",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(), "target": target,
                "v1_plan_sha256": audit["v1_plan_sha256"],
                "plan_sha256": plan["plan_sha256"],
                "plan_file_sha256": sha256_file(plan_dir / "plan.json"),
                "equivalence_audit_sha256": sha256_file(plan_dir / "equivalence_audit.json"),
                "ready_target_record_sha256": asset.record_sha256,
                "checkpoint_sha256": checkpoint,
                "quantization_manifest_sha256": plan["quantization_manifest_sha256"],
                "eligible_bits": plan["eligible_universe"]["bits"],
                "methods": plan["methods"], "budgets": plan["interior_budgets_bits"],
                "fields": [{"ber": field["ber"], "lambda": field["lambda"],
                            "count": len(field["seeds"]), "first_seed": field["seeds"][0],
                            "last_seed": field["seeds"][-1]} for field in plan["fields"]],
                "selection": plan["selection"], "final": plan["final"],
                "counts": plan["counts"], "metric": plan["primary_metric"],
                "nonfinite_penalty_nats": plan["nonfinite_penalty_nats"],
                "interpreter_sha256": sha256_file(PYTHON),
                "runner_sha256": sha256_file(runner),
                "wrapper_sha256": sha256_file(Path(__file__)),
                "measurement_source_sha256": sha256_file(Path(measured.__file__)),
                "resources": resources}
    return identity, resources


def run(model: str) -> None:
    identity, resources = check(model)
    print(json.dumps({"admitted": resources["admitted"], "failures": resources["failures"],
                      "plan_sha256": identity["plan_sha256"],
                      "checkpoint_sha256": identity["checkpoint_sha256"],
                      "interpreter_sha256": identity["interpreter_sha256"],
                      "eligible_bits": identity["eligible_bits"],
                      "selected_cpus": resources["selected_cpus"],
                      "cpu_busy": resources["total_cpu_busy_fraction"],
                      "load1": resources["load1"],
                      "memory_available_gib": round(resources["memory_available_bytes"]/GIB, 2),
                      "disk_available_gib": round(resources["disk_available_bytes"]/GIB, 2)},
                     sort_keys=True), flush=True)
    if not resources["admitted"]:
        raise RuntimeError("resource admission failed; no scientific run started")
    cpus = set(resources["selected_cpus"])
    os.sched_setaffinity(0, cpus)
    if os.sched_getaffinity(0) != cpus:
        raise RuntimeError("requested CPU affinity was not enforced")
    plan_dir, run_dir, runner = paths(model)
    path = run_dir.with_name(run_dir.name + "_admission.json")
    identity["actual_new_affinity"] = sorted(cpus)
    with path.open("xb") as file:
        file.write(canonical_json_bytes(identity) + b"\n")
    print(json.dumps({"status": "ADMITTED", "receipt": str(path),
                      "receipt_sha256": sha256_file(path)}), flush=True)
    env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
               OPENBLAS_NUM_THREADS="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               PYTHONPATH=str(ROOT))
    started = time.monotonic()
    stages = []
    peak_rss = 0
    for phase in ("validate", "selection", "seal", "final", "aggregate", "export"):
        arguments = [str(PYTHON), str(runner), phase, str(READY), str(plan_dir)]
        if phase != "validate":
            arguments.append(str(run_dir))
        stage_start = time.monotonic()
        child = subprocess.Popen(arguments, cwd=ROOT, env=env, start_new_session=True)
        stage_peak = 0
        try:
            while child.poll() is None:
                rss, pids = _group_memory(child.pid)
                stage_peak = max(stage_peak, rss)
                peak_rss = max(peak_rss, rss)
                if rss > TARGETS[model][2]*GIB:
                    os.killpg(child.pid, signal.SIGTERM)
                    raise RuntimeError("campaign process group exceeded frozen RAM limit")
                for pid in pids:
                    try:
                        if os.sched_getaffinity(pid) != cpus:
                            os.killpg(child.pid, signal.SIGTERM)
                            raise RuntimeError("campaign child escaped admitted CPU affinity")
                    except ProcessLookupError:
                        continue
                time.sleep(2)
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait()
            raise
        stage = {"phase": phase, "exit_code": child.returncode,
                 "seconds": round(time.monotonic()-stage_start, 3),
                 "peak_group_rss_bytes": stage_peak, "affinity": sorted(cpus)}
        stages.append(stage)
        print(json.dumps({"stage_complete": stage}, sort_keys=True), flush=True)
        if child.returncode:
            raise RuntimeError("scientific stage failed; immutable partial state retained")
    runtime = {"schema": "netsurf-w8a8-lm-runtime/v1", "target": TARGETS[model][0],
               "admission_sha256": sha256_file(path),
               "seconds": round(time.monotonic()-started, 3), "peak_group_rss_bytes": peak_rss,
               "affinity": sorted(cpus), "stages": stages, "exit_code": 0}
    receipt = run_dir.with_name(run_dir.name + "_runtime_receipt.json")
    with receipt.open("xb") as file:
        file.write(canonical_json_bytes(runtime) + b"\n")
    print(json.dumps({"status": "COMPLETE",
                      "runtime_receipt": str(receipt), "peak_rss_bytes": peak_rss}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("preflight", "run"))
    parser.add_argument("model", choices=tuple(TARGETS))
    args = parser.parse_args()
    if args.phase == "preflight":
        identity, resources = check(args.model)
        print(json.dumps({"identity": identity, "admitted": resources["admitted"]}, sort_keys=True))
        if not resources["admitted"]:
            raise SystemExit(3)
    else:
        run(args.model)
