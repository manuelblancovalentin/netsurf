"""Recover a completed reliability campaign using saved evidence only.

This module has no model-loading, ranking, schedule-generation, or experiment
execution path. Run it with the analysis interpreter, never the MLPerf runtime.
Existing incomplete derived outputs are preserved; new outputs are staged and
published with write-once renames after checksum and SQLite verification.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import tempfile
from pathlib import Path

from netsurf.manifests import sha256_file
from netsurf.reliability_database import _verified_bundle
from .build_reliability_database_v3 import build_product
from .export_reliability_database_v3 import export
from .export_reliability_followup import export_source_bundle
from .kws_matched_gate import BERS, BUDGETS, CAMPAIGN, METHODS, postflight

DERIVED_NAME = "derived_database_recovered_v2"
EXPORT_NAME = "export_recovered_v2"


def _inventory(directory: Path) -> dict[str, str]:
    return {str(path.relative_to(directory)): sha256_file(path)
            for path in sorted(directory.rglob("*")) if path.is_file()}


def _audit_saved_fields(bundle: Path, clean: dict) -> dict:
    """Check exact frozen grid, saved schedules, controls, and event counts."""
    schedules = json.loads((bundle / "artifacts/fault_schedules.json").read_text())
    expected_reps = {f"replication_{n}" for n in range(4)}
    if set(schedules) != expected_reps:
        raise ValueError("saved fault schedules do not have four replications")
    fields = {}
    for rep, entries in schedules.items():
        if len(entries) != len(BERS) * 8:
            raise ValueError(f"saved fault schedule count differs for {rep}")
        for ber_position, ber in enumerate(BERS):
            for trial in range(8):
                schedule = entries[8 * ber_position + trial]
                if schedule["logical_ber"] != ber or schedule["schedule_index"] != 8 * ber_position + trial:
                    raise ValueError(f"saved BER/trial schedule geometry differs for {rep}")
                fields[(rep, ber, trial)] = schedule
    expected_policies = {"no_protection", "full_protection"} | {
        f"{method}@{coverage:.6f}" for method in METHODS for coverage in BUDGETS
    }
    observed = {}
    with (bundle / "measurements/trials.csv").open(newline="") as file:
        for row in csv.DictReader(file):
            ber_text, policy = row["policy"].split("/", 1)
            key = (row["replication"], float(ber_text.removeprefix("ber=")), int(row["trial"]))
            if key not in fields or policy not in expected_policies:
                raise ValueError(f"unexpected saved condition: {key}, {policy}")
            if policy in observed.setdefault(key, {}):
                raise ValueError(f"duplicate saved policy: {key}, {policy}")
            schedule = fields[key]
            proposed = len(schedule["events"])
            if row["schedule_sha256"] != schedule["sha256"] or int(row["proposed_events"]) != proposed:
                raise ValueError(f"saved trial does not match proposed fault field: {key}, {policy}")
            if policy == "no_protection" and (int(row["applied_events"]) != proposed or int(row["suppressed_events"]) != 0):
                raise ValueError(f"unprotected control suppressed faults: {key}")
            if policy == "full_protection":
                if int(row["applied_events"]) != 0 or int(row["suppressed_events"]) != proposed:
                    raise ValueError(f"fully protected control applied faults: {key}")
                if (float(row["accuracy"]) != clean["accuracy"] or
                        float(row["categorical_crossentropy"]) != clean["categorical_crossentropy"]):
                    raise ValueError(f"fully protected control differs from clean inference: {key}")
            observed[key][policy] = row
    if len(observed) != len(expected_reps) * len(BERS) * 8 or any(
            set(policies) != expected_policies for policies in observed.values()):
        raise ValueError("saved trial grid is incomplete")
    return {"fields": len(observed), "policy_budget_evaluations": sum(map(len, observed.values())),
            "bers": list(BERS), "budgets": list(BUDGETS), "methods": list(METHODS)}


def _verify_product(product: Path, sources: Path, assets: Path) -> dict:
    checks = json.loads((product / "checksums.json").read_text())
    if checks["sources_sha256"] != sha256_file(sources) or checks["asset_manifest_sha256"] != sha256_file(assets):
        raise ValueError("derived product source or asset identity differs")
    for relative, digest in checks["files"].items():
        if sha256_file(product / relative) != digest:
            raise ValueError(f"derived product artifact changed: {relative}")
    if _inventory(product).keys() != checks["files"].keys() | {"checksums.json", "FINALIZED"}:
        raise ValueError("derived product artifact inventory differs")
    if (product / "FINALIZED").read_text().strip() != f"reliability_database_sha256={checks['files']['reliability_v3.sqlite']}":
        raise ValueError("derived product finalization marker differs")
    with sqlite3.connect(f"file:{product / 'reliability_v3.sqlite'}?mode=ro", uri=True) as database:
        if database.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("derived SQLite integrity check failed")
        primary_count = database.execute("SELECT count(*) FROM primary_conditions").fetchone()[0]
        followup_count = database.execute(
            "SELECT count(*) FROM reliability_grid WHERE study_id=?", (f"{CAMPAIGN}_followup",)
        ).fetchone()[0]
    if primary_count != 280 or followup_count != 240:
        raise ValueError(f"derived study counts differ: primary={primary_count}, followup={followup_count}")
    return {"primary_conditions": primary_count, "followup_conditions": followup_count,
            "database_sha256": checks["files"]["reliability_v3.sqlite"]}


def _publish_staged(output: Path, name: str, builder) -> Path:
    final = output / name
    if final.exists():
        raise FileExistsError(f"write-once destination already exists: {final}")
    staging = Path(tempfile.mkdtemp(prefix=f".{name}.stage-", dir=output))
    candidate = staging / name
    try:
        builder(candidate, final)
        if final.exists():
            raise FileExistsError(f"write-once destination appeared during staging: {final}")
        os.rename(candidate, final)
        staging.rmdir()
    except Exception as error:
        raise RuntimeError(f"recovery stage retained for inspection: {staging}") from error
    return final


def _verify_exports(destination: Path) -> dict:
    database_receipt = json.loads((destination / "export_receipt.json").read_text())
    source_receipt = json.loads((destination / "source_export_receipt.json").read_text())
    for receipt in (database_receipt, source_receipt):
        for archive in receipt["archives"].values():
            path = Path(archive["path"])
            if path.parent != destination or sha256_file(path) != archive["sha256"]:
                raise ValueError(f"published archive identity differs: {path}")
    return {"database_archives": database_receipt["archives"],
            "source_archives": source_receipt["archives"]}


def recover(output: Path, base_sources: Path, assets: Path, baseline: Path) -> dict:
    output, base_sources, assets, baseline = (
        path.resolve() for path in (output, base_sources, assets, baseline)
    )
    if not output.is_dir():
        raise FileNotFoundError(f"completed campaign receipt directory is missing: {output}")
    run = json.loads((output / "run_receipt.json").read_text())
    preflight = run["campaign_preflight"]
    bundle = Path(run["run_directory"]).resolve()
    source = {"id": f"{CAMPAIGN}_followup", "path": str(bundle),
              "kind": "whole_kernel_followup",
              "scientific_payload_sha256": run["scientific_payload_sha256"],
              "checksums_sha256": sha256_file(bundle / "checksums.json")}
    _, manifest, summary, _ = _verified_bundle(source)
    if manifest["source"]["commit"] != run["source_git_revision"]:
        raise ValueError("run receipt and finalized source revision differ")
    if postflight(bundle, preflight) != json.loads((output / "kws_matched_postflight.json").read_text()):
        raise ValueError("saved KWS postflight receipt differs from verified measurements")
    field_audit = _audit_saved_fields(bundle, summary["clean_evaluation"])
    pristine = json.loads(base_sources.read_text())
    if any(item["id"] == source["id"] for item in pristine["studies"]):
        raise ValueError("follow-up study already occurs in historical source receipt")
    pristine["studies"].append(source)
    sources = output / "source_receipt.json"
    if json.loads(sources.read_text()) != pristine:
        raise ValueError("saved follow-up source receipt differs from verified source")
    baseline_checks = json.loads((baseline / "checksums.json").read_text())
    baseline_primary = baseline / "primary_conditions.csv"
    if sha256_file(baseline_primary) != baseline_checks["files"]["primary_conditions.csv"]:
        raise ValueError("historical primary table failed checksum verification")
    partial = output / "derived_database"
    partial_before = _inventory(partial) if partial.exists() else {}
    derived = output / DERIVED_NAME
    if not derived.exists():
        def build(candidate: Path, _: Path) -> None:
            build_product(sources, assets, candidate)
            _verify_product(candidate, sources, assets)
            if sha256_file(candidate / "primary_conditions.csv") != sha256_file(baseline_primary):
                raise ValueError("historical 280 primary conditions changed")
        _publish_staged(output, DERIVED_NAME, build)
    counts = _verify_product(derived, sources, assets)
    if sha256_file(derived / "primary_conditions.csv") != sha256_file(baseline_primary):
        raise ValueError("historical 280 primary conditions changed")
    archive = output / EXPORT_NAME
    if not archive.exists():
        def build_export(candidate: Path, final: Path) -> None:
            export(derived, candidate, public_destination=final)
            export_source_bundle(source, candidate, public_destination=final)
        _publish_staged(output, EXPORT_NAME, build_export)
    archives = _verify_exports(archive)
    if partial.exists() and _inventory(partial) != partial_before:
        raise ValueError("failed-stage partial artifacts changed during recovery")
    audit = {
        "schema_version": "1.0.0",
        "source_bundle": str(bundle),
        "source_scientific_payload_sha256": source["scientific_payload_sha256"],
        "source_checksums_sha256": source["checksums_sha256"],
        "failed_stage_directory": str(partial),
        "failed_stage_artifacts_sha256": partial_before,
        "historical_primary_conditions_sha256": sha256_file(baseline_primary),
        "recovered_database_sha256": counts["database_sha256"],
        "recovered_archives": archives,
        "field_geometry": field_audit,
        "model_inferences_repeated": 0,
    }
    audit_path = output / "recovery_audit_v2.json"
    if audit_path.exists():
        if json.loads(audit_path.read_text()) != audit:
            raise ValueError("existing recovery audit differs from verified evidence")
    else:
        # Hard-link creation is atomic and fails rather than replacing an
        # existing audit if two recovery processes finish concurrently.
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output,
                                         prefix=".recovery-audit-", delete=False) as file:
            file.write(json.dumps(audit, indent=2, sort_keys=True) + "\n")
            temporary = Path(file.name)
        try:
            os.link(temporary, audit_path)
        finally:
            temporary.unlink()
    return {"status": "RECOVERED_OFFLINE", "source_bundle": str(bundle),
            "scientific_payload_sha256": source["scientific_payload_sha256"],
            "source_revision": run["source_git_revision"],
            "source_artifact_count": len(json.loads((bundle / "checksums.json").read_text())["files"]),
            "paired_fault_fields": field_audit["fields"],
            "policy_budget_evaluations": field_audit["policy_budget_evaluations"],
            "method_count": len(field_audit["methods"]),
            "interior_budget_count": len(field_audit["budgets"]),
            "ber_count": len(field_audit["bers"]),
            "partial_artifacts_preserved": len(partial_before),
            "historical_primary_sha256": sha256_file(baseline_primary),
            "derived_database": str(derived / "reliability_v3.sqlite"),
            "derived_database_sha256": counts["database_sha256"],
            "primary_conditions": counts["primary_conditions"],
            "followup_conditions": counts["followup_conditions"],
            "export": str(archive), "archives": archives,
            "mac_scp_commands": str(archive / "mac_scp_commands.txt"),
            "recovery_audit": str(audit_path),
            "model_inferences_repeated": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sources", type=Path, default=Path("config/reliability_sources_v1.json"))
    parser.add_argument("--assets", type=Path, default=Path("/projects/kona-assets/manuelbv/NETSURF_ASSET_READINESS.json"))
    parser.add_argument("--baseline", type=Path, default=Path("workspace/reliability_database_v3_m12_release"))
    args = parser.parse_args()
    print(json.dumps(recover(args.output, args.sources, args.assets, args.baseline), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
