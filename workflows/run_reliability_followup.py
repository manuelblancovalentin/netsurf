"""Foreground, hash-gated execution of one frozen reliability follow-up plan.

Dry-run is the default. --execute is reserved for the researcher after resource
approval. Completed source bundles are reused only after full hash verification.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from netsurf.manifests import sha256_file
from netsurf.reliability_database import _verified_bundle
from netsurf.reliability_database_v3 import verified_asset_records
from .build_reliability_database_v3 import build_product
from .export_reliability_database_v3 import export
from .export_reliability_followup import export_source_bundle
from .kws_matched_gate import CAMPAIGN, postflight as kws_postflight, preflight as kws_preflight

PINNED_PYTHON = Path("/projects/kona-assets/manuelbv/.venv-mlperf/bin/python")


def live_resources() -> dict:
    """A read-only, point-in-time Kona receipt; not a runtime guarantee."""
    memory = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, _, value = line.partition(":")
        if name in {"MemTotal", "MemAvailable", "SwapFree"}:
            memory[name] = round(int(value.split()[0]) / (1024 ** 2), 2)
    jobs = []
    process_table = subprocess.run(
        ["ps", "-eo", "pid=,%cpu=,rss=,args="], check=True,
        capture_output=True, text=True,
    ).stdout
    for line in process_table.splitlines():
        parts = line.strip().split(maxsplit=3)
        if len(parts) == 4 and "m11_run_both_matched_lambda.py --worker" in parts[3]:
            jobs.append({"pid": int(parts[0]), "cpu_percent": float(parts[1]),
                         "rss_mib": round(int(parts[2]) / 1024, 2),
                         "worker": parts[3].rsplit("--worker", 1)[-1].strip()})
    return {"logical_cpus": os.cpu_count(), "load_average": tuple(round(v, 2) for v in os.getloadavg()),
            "memory_total_gib": memory["MemTotal"],
            "memory_available_gib": memory["MemAvailable"],
            "swap_free_gib": memory["SwapFree"], "active_m11_workers": jobs}


def source_revision() -> tuple[str, bool]:
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
        check=True, capture_output=True, text=True,
    ).stdout.strip())
    return revision, dirty


def checked_plan(receipt_path: Path, name: str) -> tuple[dict, Path]:
    receipt_path=receipt_path.resolve()
    receipt=json.loads(receipt_path.read_text())
    if receipt["schema_version"]!="1.0.0":
        raise ValueError("unsupported frozen plan receipt")
    plans=[plan for plan in receipt["plans"] if plan["name"]==name]
    if len(plans)!=1:
        raise ValueError("plan name must identify one frozen plan")
    plan=plans[0]
    config=receipt_path.parent.parent/plan["configuration"]
    if sha256_file(config)!=plan["configuration_sha256"]:
        raise ValueError("frozen plan configuration changed")
    if sha256_file(Path(plan["source_configuration"]))!=plan["source_configuration_sha256"]:
        raise ValueError("source configuration changed")
    if plan["status"]!="VALIDATED_NOT_EXECUTED":
        raise ValueError("plan is not in validated/unexecuted state")
    return plan,config


def execute(receipt_path: Path, name: str, sources: Path, assets: Path,
            output: Path, *, run: bool = False) -> dict:
    workspace = Path(__file__).resolve().parents[1]/"workspace"
    if not output.resolve().is_relative_to(workspace):
        raise ValueError(f"follow-up output must remain under {workspace}")
    plan,config=checked_plan(receipt_path,name)
    revision, dirty = source_revision()
    if run and dirty:
        raise ValueError("tracked source is dirty; commit and verify the exact source revision before execution")
    verified_asset_records(assets)
    if not PINNED_PYTHON.is_file():
        raise FileNotFoundError(f"pinned MLPerf interpreter unavailable: {PINNED_PYTHON}")
    subprocess.run([str(PINNED_PYTHON),"-m","netsurf","validate",str(config)],check=True)
    campaign_preflight = kws_preflight(config,plan,sources) if name == CAMPAIGN else None
    free_gib=shutil.disk_usage(output.parent).free/(1024**3)
    estimate=plan["rough_output_gib_from_source_ratio"]
    if free_gib < 2*estimate:
        raise ValueError(f"less than twice estimated output disk free: {free_gib:.1f} GiB")
    summary={"plan":name,"config":str(config),"config_sha256":plan["configuration_sha256"],
             "policy_budget_evaluations":plan["policy_budget_evaluations"],
             "rough_elapsed_hours":plan["rough_elapsed_hours_from_source_ratio"],
             "rough_output_gib":estimate,"disk_free_gib":round(free_gib,2),
             "host_resources":live_resources(),
             "source_git_revision":revision,"tracked_source_dirty":dirty,
             "campaign_preflight":campaign_preflight,
             "warning":"The 64-clip KWS FKeras eigensolve passed a bounded preflight; contention, repeated setup and campaign ranking are not captured by the rough time/disk ratio."}
    if not run:
        return {**summary,"status":"PREFLIGHT_ONLY_NOT_EXECUTED"}
    print(json.dumps({"preflight_status":"PASSED", "source_git_revision":revision,
                      "host_resources":summary["host_resources"],
                      "planned_policy_budget_evaluations":plan["policy_budget_evaluations"]},
                     sort_keys=True), file=sys.stderr, flush=True)
    if not output.exists():
        output.mkdir(parents=True)
    receipt_file=output/"run_receipt.json"
    if receipt_file.exists():
        record=json.loads(receipt_file.read_text())
        identity_fields = (
            "model_sha256", "asset_lock_digest", "cohort_sha256",
            "sample_index_sha256", "source_scientific_payload_sha256",
        )
        previous = record.get("campaign_preflight")
        if record.get("source_git_revision") != revision or (
            previous is None and campaign_preflight is not None
        ) or (
            previous is not None and campaign_preflight is not None and
            any(previous.get(key) != campaign_preflight.get(key) for key in identity_fields)
        ):
            raise ValueError("resume source revision or campaign preflight differs from the sealed run receipt")
    else:
        # Stderr is inherited so Rich/plain progress stays live in tmux;
        # stdout contains the final machine-readable CLI result.
        result=subprocess.run([str(PINNED_PYTHON),"-m","netsurf","run",str(config)],
                              check=True,stdout=subprocess.PIPE,text=True)
        record=json.loads(result.stdout)
        record.update({"source_git_revision":revision,"campaign_preflight":campaign_preflight})
        receipt_file.write_text(json.dumps(record,sort_keys=True,indent=2)+"\n")
    bundle=Path(record["run_directory"])
    source={"id":f"{name}_followup","path":str(bundle),"kind":"whole_kernel_followup",
            "scientific_payload_sha256":record["scientific_payload_sha256"],
            "checksums_sha256":sha256_file(bundle/"checksums.json")}
    _verified_bundle(source)
    if campaign_preflight is not None:
        gate = kws_postflight(bundle, campaign_preflight)
        gate_path = output/"kws_matched_postflight.json"
        if gate_path.exists():
            if json.loads(gate_path.read_text()) != gate:
                raise ValueError("existing KWS postflight receipt differs")
        else:
            gate_path.write_text(json.dumps(gate,sort_keys=True,indent=2)+"\n")
    source_receipt=json.loads(sources.read_text())
    if any(item["id"]==source["id"] for item in source_receipt["studies"]):
        raise ValueError("follow-up source ID already exists; do not silently replace")
    source_receipt["studies"].append(source)
    new_sources=output/"source_receipt.json"
    if not new_sources.exists():
        new_sources.write_text(json.dumps(source_receipt,sort_keys=True,indent=2)+"\n")
    elif json.loads(new_sources.read_text())!=source_receipt:
        raise ValueError("resume source receipt differs")
    derived=output/"derived_database"
    if not derived.exists():
        build_product(new_sources,assets,derived)
    else:
        checks=json.loads((derived/"checksums.json").read_text())
        if checks["sources_sha256"]!=sha256_file(new_sources):
            raise ValueError("existing derived database references different source receipt")
        for path,digest in checks["files"].items():
            if sha256_file(derived/path)!=digest:
                raise ValueError(f"derived database changed: {path}")
    archive=output/"export"
    if not archive.exists():
        export(derived,archive)
    source_export = export_source_bundle(source, archive)
    return {**summary,"status":"FINALIZED_DERIVED","run_directory":str(bundle),
            "scientific_payload_sha256":source["scientific_payload_sha256"],
            "database":str(derived/"reliability_v3.sqlite"),"export":str(archive),
            "source_archives":source_export["archives"],
            "mac_scp_commands":str(archive/"mac_scp_commands.txt"),
            "kws_postflight":str(output/"kws_matched_postflight.json") if campaign_preflight else None}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt",type=Path,required=True)
    parser.add_argument("--plan",required=True)
    parser.add_argument("--sources",type=Path,default=Path("config/reliability_sources_v1.json"))
    parser.add_argument("--assets",type=Path,default=Path("/projects/kona-assets/manuelbv/NETSURF_ASSET_READINESS.json"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--execute",action="store_true",help="run the large frozen campaign in the foreground")
    args=parser.parse_args()
    print(json.dumps(execute(args.receipt,args.plan,args.sources,args.assets,args.output,run=args.execute),indent=2,sort_keys=True))


if __name__=="__main__":
    main()
