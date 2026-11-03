"""Foreground, hash-gated execution of one frozen reliability follow-up plan.

Dry-run is the default. --execute is reserved for the researcher after resource
approval. Completed source bundles are reused only after full hash verification.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from netsurf.manifests import sha256_file
from netsurf.reliability_database import _verified_bundle
from netsurf.reliability_database_v3 import verified_asset_records
from .build_reliability_database_v3 import build_product
from .export_reliability_database_v3 import export

PINNED_PYTHON = Path("/projects/kona-assets/manuelbv/.venv-mlperf/bin/python")


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
    verified_asset_records(assets)
    if not PINNED_PYTHON.is_file():
        raise FileNotFoundError(f"pinned MLPerf interpreter unavailable: {PINNED_PYTHON}")
    subprocess.run([str(PINNED_PYTHON),"-m","netsurf","validate",str(config)],check=True)
    free_gib=shutil.disk_usage(output.parent).free/(1024**3)
    estimate=plan["rough_output_gib_from_source_ratio"]
    if free_gib < 2*estimate:
        raise ValueError(f"less than twice estimated output disk free: {free_gib:.1f} GiB")
    summary={"plan":name,"config":str(config),"config_sha256":plan["configuration_sha256"],
             "policy_budget_evaluations":plan["policy_budget_evaluations"],
             "rough_elapsed_hours":plan["rough_elapsed_hours_from_source_ratio"],
             "rough_output_gib":estimate,"disk_free_gib":round(free_gib,2),
             "warning":"The 64-clip KWS FKeras eigensolve passed a bounded preflight; contention, repeated setup and campaign ranking are not captured by the rough time/disk ratio."}
    if not run:
        return {**summary,"status":"PREFLIGHT_ONLY_NOT_EXECUTED"}
    if not output.exists():
        output.mkdir(parents=True)
    receipt_file=output/"run_receipt.json"
    if receipt_file.exists():
        record=json.loads(receipt_file.read_text())
    else:
        # Stderr is inherited so Rich/plain progress stays live in tmux;
        # stdout contains the final machine-readable CLI result.
        result=subprocess.run([str(PINNED_PYTHON),"-m","netsurf","run",str(config)],
                              check=True,stdout=subprocess.PIPE,text=True)
        record=json.loads(result.stdout)
        receipt_file.write_text(json.dumps(record,sort_keys=True,indent=2)+"\n")
    bundle=Path(record["run_directory"])
    source={"id":f"{name}_followup","path":str(bundle),"kind":"whole_kernel_followup",
            "scientific_payload_sha256":record["scientific_payload_sha256"],
            "checksums_sha256":sha256_file(bundle/"checksums.json")}
    _verified_bundle(source)
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
    return {**summary,"status":"FINALIZED_DERIVED","run_directory":str(bundle),
            "scientific_payload_sha256":source["scientific_payload_sha256"],
            "database":str(derived/"reliability_v3.sqlite"),"export":str(archive)}


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
