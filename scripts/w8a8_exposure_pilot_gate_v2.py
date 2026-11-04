"""Measured admission for the bounded DistilBERT exposure pilot only."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time

from netsurf.manifests import canonical_json_bytes, digest_object
from netsurf.w8a8_exposure_pilot import validate


GIB = 1024**3


def _ticks():
    values = [int(x) for x in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
    return sum(values), values[3] + values[4]


def _proc(pid):
    base = Path("/proc") / str(pid)
    try:
        raw = (base / "stat").read_text().rsplit(")", 1)[1].split()
        status = dict(line.split(":", 1) for line in (base / "status").read_text().splitlines() if ":" in line)
        args = [x.decode("utf-8", "replace") for x in (base / "cmdline").read_bytes().split(b"\0") if x]
        affinity = status["Cpus_allowed_list"].strip()
        return {"pid": pid, "ticks": int(raw[11]) + int(raw[12]),
                "last_cpu": int(raw[36]), "rss_bytes": int(status.get("VmRSS", "0 kB").split()[0])*1024,
                "threads": int(status["Threads"].strip()), "affinity": affinity, "args": args}
    except (FileNotFoundError, ProcessLookupError, PermissionError, IndexError, KeyError, ValueError):
        return None


def _processes():
    return {item["pid"]: item for entry in Path("/proc").iterdir() if entry.name.isdecimal()
            if (item := _proc(int(entry.name))) is not None}


def _physical(cpu):
    root = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
    return int((root / "physical_package_id").read_text()), int((root / "core_id").read_text())


def _parse_affinity(value):
    result = set()
    for item in value.split(","):
        if "-" in item:
            low, high = map(int, item.split("-"))
            result.update(range(low, high + 1))
        else:
            result.add(int(item))
    return result


def _occupied_physical(item, physical=_physical):
    affinity = _parse_affinity(item["affinity"])
    return {physical(cpu) for cpu in
            (affinity if len(affinity) <= 16 else {item["last_cpu"]})}


def measure(run_dir: Path) -> dict:
    before, first = _processes(), _ticks()
    start = time.monotonic()
    time.sleep(1.0)
    after, second = _processes(), _ticks()
    elapsed = time.monotonic() - start
    busy = (second[0]-first[0]-(second[1]-first[1]))/(second[0]-first[0])
    load = Path("/proc/loadavg").read_text().split()
    allowed = sorted(os.sched_getaffinity(0))
    physical = {_physical(cpu) for cpu in allowed}
    active = []
    occupied = set()
    for pid, item in after.items():
        prior = before.get(pid)
        cores = max(0., (item["ticks"]-prior["ticks"])/os.sysconf("SC_CLK_TCK")/elapsed) if prior else 0.
        command = " ".join(item["args"])
        if cores < .25 or not any(x in command.lower() for x in
                ("w8a8", "pythia", "qwen", "distilbert", "m11", "m12", "reliability_followup")):
            continue
        role = ("m11" if "m11" in command.lower() else "m12" if "m12" in command.lower() or
                "reliability_followup" in command.lower() else "other_model")
        active.append({"pid": pid, "role": role, "cpu_cores": round(cores, 3),
                       "rss_bytes": item["rss_bytes"], "threads": item["threads"],
                       "last_cpu": item["last_cpu"], "affinity": item["affinity"]})
        # Pinned jobs occupy their whole assigned physical set. A broadly
        # scheduled job has no exclusive set, so avoid its sampled CPU only.
        occupied.update(_occupied_physical(item))
    candidates = defaultdict(list)
    for cpu in allowed:
        if cpu <= 1:
            continue
        key = _physical(cpu)
        if key not in occupied:
            candidates[key[0]].append((key, cpu))
    selected = []
    for socket, entries in sorted(candidates.items()):
        seen = set()
        unique = []
        for key, cpu in entries:
            if key not in seen:
                unique.append(cpu); seen.add(key)
        if len(unique) >= 4:
            selected = unique[:4]
            break
    available_kib = next(int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()
                         if line.startswith("MemAvailable:"))
    disk = shutil.disk_usage(run_dir.parent).free
    failures = []
    if busy > .60 or float(load[0]) > min(32, .8*len(physical)) or int(load[3].split("/")[0]) > 32:
        failures.append("host CPU/load above pilot admission threshold")
    if available_kib*1024 < 16*GIB:
        failures.append("less than 16 GiB RAM available for 8 GiB pilot cap and reserve")
    if disk < 5*GIB:
        failures.append("less than 5 GiB output disk available")
    if len(selected) != 4:
        failures.append("four distinct physical CPUs unavailable")
    if sum(x["cpu_cores"] for x in active) > 16:
        failures.append("major model experiments exceed 16 CPU cores")
    return {"schema": "netsurf-w8a8-pilot-resource-admission/v1",
            "sampled_utc": datetime.now(timezone.utc).isoformat(), "sample_seconds": elapsed,
            "cpu_busy_fraction": round(busy, 4), "load1": float(load[0]),
            "runnable_tasks": int(load[3].split("/")[0]), "physical_cores_allowed": len(physical),
            "active_model_jobs": active, "selected_cpus": selected,
            "selected_physical_cores": [list(_physical(cpu)) for cpu in selected],
            "memory_available_bytes": available_kib*1024, "disk_available_bytes": disk,
            "limits": {"pilot_threads": 4, "pilot_max_rss_gib_planned": 8,
                       "min_available_ram_gib": 16, "min_free_disk_gib": 5,
                       "max_host_cpu_busy_fraction": .60, "max_load1": min(32, .8*len(physical)),
                       "max_runnable_tasks": 32, "max_major_model_cpu_cores": 16},
            "admitted": not failures, "failures": failures}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("plan_dir", type=Path)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    plan, _ = validate(args.manifest, args.plan_dir)
    receipt = args.run_dir.with_name(args.run_dir.name + "_admission.json")
    if args.run_dir.exists() or receipt.exists() or args.run_dir.with_name(args.run_dir.name + "_export").exists():
        raise ValueError("write-once pilot output or admission receipt already exists")
    measured = measure(args.run_dir)
    measured["plan_sha256"] = plan["plan_sha256"]
    measured["run_dir"] = str(args.run_dir)
    print(json.dumps(measured, sort_keys=True), file=__import__("sys").stderr, flush=True)
    if not measured["admitted"]:
        raise RuntimeError("measured resource admission rejected pilot before scientific execution")
    with receipt.open("xb") as stream:
        stream.write(canonical_json_bytes({**measured, "payload_sha256": digest_object(measured)}) + b"\n")
    print(",".join(map(str, measured["selected_cpus"])))


if __name__ == "__main__":
    main()
