"""Measured, foreground admission and execution for frozen DistilBERT W8A8 v2.

This script never changes the affinity or priority of an existing job.  Only
its own campaign children inherit the selected four physical CPU cores.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from netsurf.manifests import canonical_json_bytes, sha256_file
from netsurf.w8a8_campaign import validate
from freeze_distilbert_w8a8_concurrent_v2 import ADMISSION, ROOT, V1, V2, verify


READY = Path("/projects/kona-assets/manuelbv/NETSURF_ASSET_READINESS.json")
PYTHON = Path("/projects/kona-assets/manuelbv/envs/torch-2.9.1-cpu-py311/bin/python")
RUN = ROOT / "workspace/runs/distilbert_sst2_w8a8_reliability_v2_run1"
GIB = 1024 ** 3


def _cpu_ticks() -> tuple[int, int]:
    values = [int(x) for x in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
    return sum(values), values[3] + values[4]


def _read_proc(pid: int) -> dict | None:
    base = Path("/proc") / str(pid)
    try:
        raw = (base / "stat").read_text().rsplit(")", 1)[1].split()
        status = (base / "status").read_text()
        args = [x.decode("utf-8", "replace") for x in (base / "cmdline").read_bytes().split(b"\0") if x]
        fields = dict(line.split(":", 1) for line in status.splitlines() if ":" in line)
        uid = int(fields["Uid"].split()[0])
        return {"pid": pid, "uid": uid, "ticks": int(raw[11]) + int(raw[12]),
                "processor": int(raw[36]), "rss_bytes": int(fields.get("VmRSS", "0 kB").split()[0])*1024,
                "threads": int(fields["Threads"].strip()),
                "affinity": fields["Cpus_allowed_list"].strip(), "args": args}
    except (FileNotFoundError, ProcessLookupError, PermissionError, IndexError, KeyError, ValueError):
        return None


def _processes() -> dict[int, dict]:
    result = {}
    for entry in Path("/proc").iterdir():
        if entry.name.isdecimal():
            item = _read_proc(int(entry.name))
            if item is not None:
                result[item["pid"]] = item
    return result


def _role(item: dict) -> str | None:
    args = item["args"]
    if not args:
        return None
    joined = " ".join(args)
    if any(Path(a).name == "m11_run_both_matched_lambda.py" for a in args) and "--worker" in args:
        return "m11_worker"
    if (("workflows.run_reliability_followup" in args and
         "kws_partial_tier_fkeras_matched_v1" in joined) or
        "kws_partial_tier_fkeras_matched_v1.toml" in joined or
        "m12_kws_seven_method_followup" in joined):
        return "m12_kws_seven_method"
    if any("python" in Path(a).name for a in args[:1]) and \
       any(word in joined.lower() for word in ("netsurf", "transformer", "w8a8", "qwen", "pythia", "distilbert")):
        return "other_model"
    return None


def _physical_key(cpu: int) -> tuple[int, int]:
    base = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
    return (int((base / "physical_package_id").read_text()),
            int((base / "core_id").read_text()))


def _select_cores(allowed: set[int], occupied: set[int], count: int) -> list[int]:
    excluded = {_physical_key(x) for x in occupied}
    candidates: dict[tuple[int, int], list[int]] = defaultdict(list)
    for cpu in sorted(allowed):
        key = _physical_key(cpu)
        if key not in excluded:
            candidates[key].append(cpu)
    by_socket: dict[int, list[int]] = defaultdict(list)
    for (socket, _), siblings in candidates.items():
        by_socket[socket].append(min(siblings))
    for socket in sorted(by_socket):
        choices = sorted(c for c in by_socket[socket] if c > 1)
        if len(choices) >= count:
            return choices[:count]
    raise ValueError("four distinct free physical cores on one socket unavailable")


def _memory_available() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise ValueError("MemAvailable unavailable")


def measure(sample_seconds: float = 1.0) -> dict:
    before, start = _processes(), _cpu_ticks()
    start_time = time.monotonic()
    time.sleep(sample_seconds)
    after, end = _processes(), _cpu_ticks()
    elapsed = time.monotonic() - start_time
    busy = (end[0]-start[0]-(end[1]-start[1])) / (end[0]-start[0])
    load = Path("/proc/loadavg").read_text().split()
    allowed = set(os.sched_getaffinity(0))
    physical = {_physical_key(cpu) for cpu in allowed}
    roles: dict[str, list[dict]] = defaultdict(list)
    for pid, item in after.items():
        role = _role(item)
        if role is None:
            continue
        old = before.get(pid)
        cores = max(0., (item["ticks"]-old["ticks"]) / os.sysconf("SC_CLK_TCK") / elapsed) if old else 0.
        if role == "other_model" and cores < .5 and item["rss_bytes"] < 2*GIB:
            continue
        roles[role].append({"pid": pid, "uid": item["uid"], "cpu_cores": round(cores, 3),
                            "rss_bytes": item["rss_bytes"], "threads": item["threads"],
                            "last_cpu": item["processor"], "affinity": item["affinity"],
                            "command": " ".join(Path(a).name if "/" in a else a for a in item["args"][:5])})
    occupied = {p["last_cpu"] for role in ("m11_worker", "m12_kws_seven_method")
                for p in roles.get(role, [])}
    selected = _select_cores(allowed, occupied, ADMISSION["distinct_physical_cores"])
    disk_free = shutil.disk_usage(RUN.parent).free
    data = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "sample_seconds": elapsed, "logical_cpus_allowed": len(allowed),
            "physical_cores_allowed": len(physical), "total_cpu_busy_fraction": round(busy, 4),
            "load1": float(load[0]), "load5": float(load[1]), "load15": float(load[2]),
            "runnable_tasks": int(load[3].split("/")[0]),
            "memory_available_bytes": _memory_available(), "disk_available_bytes": disk_free,
            "processes": {k: sorted(v, key=lambda p: p["pid"]) for k, v in roles.items()},
            "selected_cpus": selected,
            "selected_physical_cores": [list(_physical_key(c)) for c in selected],
            "proposed_new_affinity": ",".join(map(str, selected)),
            "admission_thresholds": ADMISSION}
    failures = []
    if busy > ADMISSION["max_total_cpu_busy_fraction"]:
        failures.append("total CPU busy above frozen threshold")
    if data["load1"] > ADMISSION["max_load1_per_physical_core"] * len(physical):
        failures.append("one-minute runnable load above frozen threshold")
    if data["runnable_tasks"] > ADMISSION["max_runnable_tasks"]:
        failures.append("instantaneous runnable tasks above frozen threshold")
    if data["memory_available_bytes"] < ADMISSION["min_memory_available_gib"]*GIB:
        failures.append("available RAM below frozen reserve")
    if disk_free < ADMISSION["min_disk_free_gib"]*GIB:
        failures.append("available output disk below frozen reserve")
    if len(roles.get("m11_worker", [])) > 2 or any(
        p["cpu_cores"] > ADMISSION["max_m11_worker_cpu_cores"] for p in roles.get("m11_worker", [])):
        failures.append("M11 worker count or consumption exceeds frozen allowance")
    if sum(p["cpu_cores"] for p in roles.get("m12_kws_seven_method", [])) > ADMISSION["max_m12_kws_cpu_cores"]:
        failures.append("M12 KWS consumption exceeds frozen allowance")
    if sum(p["cpu_cores"] for p in roles.get("other_model", [])) > ADMISSION["max_other_major_model_cpu_cores"]:
        failures.append("other model experiments exceed frozen allowance")
    data["admitted"] = not failures
    data["failures"] = failures
    return data


def preflight() -> tuple[dict, dict]:
    if Path(sys.executable).resolve() != PYTHON.resolve():
        raise ValueError("pinned Kona PyTorch interpreter required")
    audit = verify(V1, V2)
    plan, asset = validate(READY, V2)
    if plan["resource_envelope"]["admission"] != ADMISSION:
        raise ValueError("resource thresholds differ from frozen plan")
    if RUN.exists() or RUN.with_name(RUN.name + "_export").exists() or \
       RUN.with_name(RUN.name + "_reporting_v2").exists() or \
       RUN.with_name(RUN.name + "_admission.json").exists():
        raise ValueError("write-once campaign output already exists")
    return plan, {"schema": "netsurf-w8a8-distilbert-concurrent-admission/v2",
                  "v1_plan_sha256": audit["v1_plan_sha256"],
                  "v2_plan_sha256": plan["plan_sha256"],
                  "v2_plan_file_sha256": sha256_file(V2 / "plan.json"),
                  "checkpoint_sha256": next(f["sha256"] for f in asset.record["model"]["files"]
                                            if f["path"] == "model.safetensors"),
                  "quantization_manifest_sha256": plan["quantization_manifest_sha256"],
                  "ready_target_record_sha256": plan["target_record_sha256"],
                  "eligible_bits": plan["eligible_universe"]["bits"],
                  "methods": plan["methods"], "budgets": plan["interior_budgets_bits"],
                  "fields": plan["fields"], "selection": plan["selection"],
                  "final": plan["final"], "counts": plan["counts"],
                  "primary_metric": plan["primary_metric"],
                  "nonfinite_penalty_nats": plan["nonfinite_penalty_nats"],
                  "source_files_sha256": plan["source_files_sha256"],
                  "interpreter_sha256": sha256_file(PYTHON),
                  "resource_envelope": plan["resource_envelope"]}


def _run_stage(name: str, args: list[str], receipt: dict) -> dict:
    env = os.environ.copy()
    env.update({"PYTHONPATH": str(ROOT), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false"})
    start = time.monotonic()
    process = subprocess.Popen([str(PYTHON), *args], cwd=ROOT, env=env,
                               start_new_session=True)
    expected_affinity = set(receipt["resources"]["selected_cpus"])
    if os.sched_getaffinity(process.pid) != expected_affinity:
        os.killpg(process.pid, signal.SIGTERM)
        raise RuntimeError("new DistilBERT phase did not inherit frozen CPU affinity")
    peak_rss, peak_threads = 0, 0
    try:
        while process.poll() is None:
            item = _read_proc(process.pid)
            if item is not None:
                peak_rss = max(peak_rss, item["rss_bytes"])
                peak_threads = max(peak_threads, item["threads"])
                if item["rss_bytes"] > ADMISSION["max_new_process_rss_gib"]*GIB:
                    os.killpg(process.pid, signal.SIGTERM)
                    raise RuntimeError("new DistilBERT process exceeded frozen 8 GiB RSS cap")
            time.sleep(ADMISSION["monitor_interval_seconds"])
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait()
        raise
    result = {"stage": name, "exit_code": process.returncode,
              "seconds": round(time.monotonic()-start, 3),
              "peak_rss_bytes": peak_rss, "peak_threads": peak_threads,
              "affinity": sorted(expected_affinity)}
    if process.returncode:
        raise RuntimeError(f"{name} failed with exit code {process.returncode}; immutable partial output retained")
    return result


def execute() -> None:
    plan, identity = preflight()
    resources = measure(ADMISSION["contention_sample_seconds"])
    receipt = {**identity, "resources": resources, "run_dir": str(RUN)}
    print(json.dumps({"preflight": {"admitted": resources["admitted"],
         "failures": resources["failures"], "m11": resources["processes"].get("m11_worker", []),
         "m12_kws": resources["processes"].get("m12_kws_seven_method", []),
         "other_model": resources["processes"].get("other_model", []),
         "cpu_busy": resources["total_cpu_busy_fraction"], "load1": resources["load1"],
         "memory_available_gib": round(resources["memory_available_bytes"]/GIB, 2),
         "disk_available_gib": round(resources["disk_available_bytes"]/GIB, 2),
         "selected_cpus": resources["selected_cpus"]}}, sort_keys=True), flush=True)
    if not resources["admitted"]:
        raise RuntimeError("measured resource admission failed; campaign not started")
    live_worker_cpus = {item["processor"] for pid in
                        [p["pid"] for role in ("m11_worker", "m12_kws_seven_method")
                         for p in resources["processes"].get(role, [])]
                        if (item := _read_proc(pid)) is not None}
    if {_physical_key(cpu) for cpu in live_worker_cpus} & \
       {_physical_key(cpu) for cpu in resources["selected_cpus"]}:
        raise RuntimeError("an M11 or M12 worker moved onto a selected physical core before launch")
    os.sched_setaffinity(0, set(resources["selected_cpus"]))
    if os.sched_getaffinity(0) != set(resources["selected_cpus"]):
        raise RuntimeError("new process CPU affinity could not be enforced")
    receipt["actual_new_affinity"] = sorted(os.sched_getaffinity(0))
    path = RUN.with_name(RUN.name + "_admission.json")
    with path.open("xb") as file:
        file.write(canonical_json_bytes(receipt) + b"\n")
    print(json.dumps({"status": "ADMITTED", "receipt": str(path),
                      "receipt_sha256": sha256_file(path),
                      "plan_sha256": plan["plan_sha256"]}), flush=True)
    stages = []
    for phase in ("validate", "selection", "seal", "final", "aggregate", "export"):
        args = [str(ROOT / "scripts/w8a8_phase_entry_v2.py"), phase, str(READY), str(V2)]
        if phase != "validate":
            args.append(str(RUN))
        stages.append(_run_stage(phase, args, receipt))
        print(json.dumps({"stage_complete": stages[-1]}, sort_keys=True), flush=True)
    stage_path = RUN.with_name(RUN.name + "_stages.json")
    with stage_path.open("xb") as file:
        file.write(canonical_json_bytes({"schema": "netsurf-w8a8-stage-runtime/v1",
                                         "admission_sha256": sha256_file(path),
                                         "stages": stages}) + b"\n")
    stages.append(_run_stage("report", [str(ROOT / "scripts/report_distilbert_w8a8_v2.py"),
                                        str(V2), str(RUN)], receipt))
    print(json.dumps({"stage_complete": stages[-1]}, sort_keys=True), flush=True)
    output = RUN.with_name(RUN.name + "_reporting_v2")
    runtime_path = RUN.with_name(RUN.name + "_runtime_receipt.json")
    with runtime_path.open("xb") as file:
        file.write(canonical_json_bytes({
        "schema": "netsurf-w8a8-runtime-receipt/v1", "admission_sha256": sha256_file(path),
        "stages": stages, "elapsed_stage_seconds": sum(x["seconds"] for x in stages)}) + b"\n")
    print((output / "mac_scp_commands.txt").read_text(), end="", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("preflight", "run"))
    args = parser.parse_args()
    if args.phase == "preflight":
        _, identity = preflight()
        resources = measure(ADMISSION["contention_sample_seconds"])
        print(json.dumps({"identity": {"v1_plan_sha256": identity["v1_plan_sha256"],
             "v2_plan_sha256": identity["v2_plan_sha256"],
             "v2_plan_file_sha256": identity["v2_plan_file_sha256"],
             "checkpoint_sha256": identity["checkpoint_sha256"],
             "quantization_manifest_sha256": identity["quantization_manifest_sha256"],
             "ready_target_record_sha256": identity["ready_target_record_sha256"],
             "eligible_bits": identity["eligible_bits"], "methods": identity["methods"],
             "budgets": identity["budgets"], "fields": [
                 {"ber": f["ber"], "lambda": f["lambda"], "count": len(f["seeds"]),
                  "first_seed": f["seeds"][0], "last_seed": f["seeds"][-1]}
                 for f in identity["fields"]], "selection": {"split": identity["selection"]["split"],
                  "size": identity["selection"]["size"],
                  "ordered_labels_and_text_sha256": identity["selection"]["ordered_labels_and_text_sha256"]},
             "final": {"split": identity["final"]["split"], "size": identity["final"]["size"],
                       "first_index": identity["final"]["source_indices"][0],
                       "last_index": identity["final"]["source_indices"][-1],
                       "ordered_labels_and_text_sha256": identity["final"]["ordered_labels_and_text_sha256"]},
             "primary_metric": identity["primary_metric"],
             "nonfinite_penalty_nats": identity["nonfinite_penalty_nats"],
             "counts": identity["counts"]}, "resources": resources}, sort_keys=True))
        if not resources["admitted"]:
            raise SystemExit(3)
    else:
        execute()


if __name__ == "__main__":
    main()
