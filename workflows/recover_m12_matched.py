"""Offline, write-once recovery of a finalized M12 matched INT8 campaign.

This module imports no benchmark adapter or experiment runner. It cannot launch
inference, construct ranking scores, or regenerate a fault schedule. Run it with
the separate analysis interpreter after the exact source bundle is finalized.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sqlite3
import tomllib
from pathlib import Path

from netsurf.manifests import sha256_file
from netsurf.reliability_database import _verified_bundle
from .build_reliability_database_v3 import build_product
from .export_reliability_database_v3 import export
from .export_reliability_followup import export_source_bundle
from .recover_reliability_followup import _publish_staged, _verify_exports


def _plan(receipt: Path, name: str) -> tuple[dict, Path, dict]:
    document = json.loads(receipt.read_text())
    matches = [item for item in document["plans"] if item["name"] == name]
    if document["schema_version"] != "1.0.0" or len(matches) != 1:
        raise ValueError("plan is absent or ambiguous in the frozen receipt")
    plan = matches[0]
    if name not in {"resnet8_partial_tier_matched_v1", "toycar_partial_tier_matched_v1"}:
        raise ValueError("this recovery path is restricted to the two M12 six-method plans")
    config = receipt.parent.parent / plan["configuration"]
    if sha256_file(config) != plan["configuration_sha256"]:
        raise ValueError("frozen plan TOML changed")
    if sha256_file(Path(plan["source_configuration"])) != plan["source_configuration_sha256"]:
        raise ValueError("historical source configuration changed")
    document = tomllib.loads(config.read_text())
    if plan["status"] != "VALIDATED_NOT_EXECUTED" or document["experiment"]["name"] != name:
        raise ValueError("frozen plan status or experiment identity differs")
    return plan, config, document


def _mask_sha256(identities: list) -> str:
    keys = [f"{name}[{int(word)}]#b{int(bit)}" for name, word, bit in sorted(identities)]
    return hashlib.sha256(json.dumps(keys, separators=(",", ":")).encode()).hexdigest()


def audit_saved(bundle: Path, plan: dict, config: dict, summary: dict) -> dict:
    """Verify full frozen geometry, bit masks, pairing, event counts and controls."""

    reps = {f"replication_{i}" for i in range(config["replications"]["count"])}
    bers = tuple(float(v) for v in config["protection"]["probabilities"])
    budgets = tuple(float(v) for v in config["protection"]["coverages"])
    methods = tuple(item["name"] for item in config["policies"])
    trials = int(config["execution"]["trials"])
    expected_policies = {"no_protection", "full_protection"} | {
        f"{method}@{coverage:.6f}" for method in methods for coverage in budgets
    }
    if len(methods) != 6 or len(budgets) != 4 or len(bers) != plan["bers"]:
        raise ValueError("saved plan does not have the frozen six-method geometry")
    schedules = json.loads((bundle / "artifacts/fault_schedules.json").read_text())
    if set(schedules) != reps:
        raise ValueError("fault-schedule replications differ from the frozen plan")
    field_map = {}
    for rep, entries in schedules.items():
        if len(entries) != len(bers) * trials:
            raise ValueError(f"fault-field count differs for {rep}")
        for ber_index, ber in enumerate(bers):
            for trial in range(trials):
                schedule = entries[ber_index * trials + trial]
                if (float(schedule["logical_ber"]) != ber or
                        int(schedule["schedule_index"]) != ber_index * trials + trial):
                    raise ValueError(f"fault schedule order differs for {rep}, {ber}, {trial}")
                field_map[(rep, ber, trial)] = schedule
    clean = summary["clean_evaluation"]
    task = "roc_auc" if "roc_auc" in clean else "categorical_crossentropy"
    if task == "roc_auc":
        clean_values = (float(clean["roc_auc"]), float(clean["reconstruction_mse"]))
    else:
        clean_values = (float(clean["accuracy"]), float(clean["categorical_crossentropy"]))
    observed = {}
    with (bundle / "measurements/trials.csv").open(newline="") as file:
        for row in csv.DictReader(file):
            ber_text, policy = row["policy"].split("/", 1)
            key = (row["replication"], float(ber_text.removeprefix("ber=")), int(row["trial"]))
            if key not in field_map or policy not in expected_policies:
                raise ValueError(f"unexpected trial identity: {key}, {policy}")
            policies = observed.setdefault(key, {})
            if policy in policies:
                raise ValueError(f"duplicate trial identity: {key}, {policy}")
            schedule = field_map[key]
            proposed = len(schedule["events"])
            applied = int(row["applied_events"])
            suppressed = int(row["suppressed_events"])
            if (row["schedule_sha256"] != schedule["sha256"] or
                    int(row["proposed_events"]) != proposed or
                    proposed != applied + suppressed):
                raise ValueError(f"schedule or event accounting differs: {key}, {policy}")
            if policy == "no_protection" and (applied != proposed or suppressed != 0):
                raise ValueError(f"no-protection control suppressed events: {key}")
            if policy == "full_protection":
                if applied != 0 or suppressed != proposed:
                    raise ValueError(f"full-protection control applied events: {key}")
                values = ((float(row["primary_metric_value"]), float(row["loss_metric_value"]))
                          if task == "roc_auc" else
                          (float(row["accuracy"]), float(row["categorical_crossentropy"])))
                if values != clean_values:
                    raise ValueError(f"fully protected result differs from clean: {key}")
            if task == "roc_auc" and (row["primary_metric_name"] != "roc_auc" or
                                       row["loss_metric_name"] != "reconstruction_mse"):
                raise ValueError("ToyCar metric was mislabeled")
            policies[policy] = row
    if set(observed) != set(field_map) or any(set(rows) != expected_policies for rows in observed.values()):
        raise ValueError("saved trial grid is incomplete")
    if len(observed) != len(bers) * plan["independent_fault_fields_per_ber"]:
        raise ValueError("independent proposed-field count differs from frozen receipt")
    if sum(map(len, observed.values())) != plan["policy_budget_evaluations"]:
        raise ValueError("policy/budget evaluation count differs from frozen receipt")

    # The final mask artifact is large, but its actual stored identities—not a
    # summary count—are the authority for cardinality and digest verification.
    masks = json.loads((bundle / "artifacts/protection_masks.json").read_text())
    if set(masks) != reps:
        raise ValueError("mask replications differ")
    eligible = int(summary["eligible_bits"])
    expected_mask_keys = expected_policies
    for rep, rep_masks in masks.items():
        if set(rep_masks) != expected_mask_keys:
            raise ValueError(f"mask policy set differs for {rep}")
        eligible_identities = {tuple(bit) for bit in rep_masks["full_protection"]}
        if len(eligible_identities) != eligible:
            raise ValueError(f"full mask does not enumerate the eligible bit population: {rep}")
        for policy, identities in rep_masks.items():
            requested = 0.0 if policy == "no_protection" else 1.0 if policy == "full_protection" else float(policy.rsplit("@", 1)[1])
            cardinality = min(eligible, max(0, math.floor(requested * eligible + 1e-12)))
            selected = {tuple(bit) for bit in identities}
            if len(identities) != cardinality or len(selected) != cardinality:
                raise ValueError(f"mask cardinality or uniqueness differs: {rep}, {policy}")
            if not selected.issubset(eligible_identities):
                raise ValueError(f"mask contains a bit outside the eligible population: {rep}, {policy}")
            if any(len(bit) != 3 or not 0 <= int(bit[2]) < 8 or int(bit[1]) < 0 for bit in identities):
                raise ValueError(f"invalid INT8 bit identity: {rep}, {policy}")
            digest = _mask_sha256(identities)
            rep_masks[policy] = (cardinality, digest)
    with (bundle / "measurements/replication_protection_results.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != len(reps) * len(bers) * len(expected_policies):
        raise ValueError("replication-level mask report is incomplete")
    for row in rows:
        rep, policy = row["replication"], row["condition"]
        if rep not in masks or policy not in masks[rep]:
            raise ValueError("unexpected replication mask report")
        count, digest = masks[rep][policy]
        if int(row["protected_bits"]) != count or row["mask_sha256"] != digest:
            raise ValueError(f"stored mask differs from report: {rep}, {policy}")
    return {"paired_fields": len(field_map), "saved_evaluations": sum(map(len, observed.values())),
            "bers": list(bers), "budgets": list(budgets), "methods": list(methods),
            "eligible_bits": eligible, "task_metric": task}


def recover(receipt: Path, name: str, bundle: Path, output: Path,
            base_sources: Path, assets: Path, baseline: Path) -> dict:
    receipt, bundle, output, base_sources, assets, baseline = (
        path.resolve() for path in (receipt, bundle, output, base_sources, assets, baseline)
    )
    plan, config_path, config = _plan(receipt, name)
    if sha256_file(bundle / "config.toml") != plan["configuration_sha256"]:
        raise ValueError("finalized source does not use the frozen TOML")
    checks_path = bundle / "checksums.json"
    source = {"id": f"{name}_followup", "path": str(bundle), "kind": "whole_kernel_followup",
              "scientific_payload_sha256": json.loads((bundle / "manifest.json").read_text())["scientific_payload_sha256"],
              "checksums_sha256": sha256_file(checks_path)}
    _, manifest, summary, _ = _verified_bundle(source)
    if manifest["status"] != "complete" or manifest["source"]["dirty"]:
        raise ValueError("source was not finalized from a clean tracked revision")
    audit = audit_saved(bundle, plan, config, summary)
    pristine = json.loads(base_sources.read_text())
    if source["id"] in {item["id"] for item in pristine["studies"]}:
        raise ValueError("new source ID already exists")
    pristine["studies"].append(source)
    output.mkdir(parents=True, exist_ok=True)
    sources = output / "source_receipt.json"
    if sources.exists():
        if json.loads(sources.read_text()) != pristine:
            raise ValueError("saved source receipt differs")
    else:
        with sources.open("x") as file:
            json.dump(pristine, file, indent=2, sort_keys=True)
            file.write("\n")
    baseline_checks = json.loads((baseline / "checksums.json").read_text())
    historical = baseline / "primary_conditions.csv"
    if sha256_file(historical) != baseline_checks["files"]["primary_conditions.csv"]:
        raise ValueError("historical 280-row primary table changed")
    derived = output / "derived_database_v1"
    if not derived.exists():
        def build(candidate: Path, _: Path) -> None:
            build_product(sources, assets, candidate)
            if sha256_file(candidate / "primary_conditions.csv") != sha256_file(historical):
                raise ValueError("derived product changed the historical primary table")
        _publish_staged(output, derived.name, build)
    if sha256_file(derived / "primary_conditions.csv") != sha256_file(historical):
        raise ValueError("historical primary table differs")
    with sqlite3.connect(f"file:{derived / 'reliability_v3.sqlite'}?mode=ro", uri=True) as db:
        if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("derived SQLite integrity check failed")
        counts = db.execute("SELECT count(*) FROM reliability_grid WHERE study_id=?", (source["id"],)).fetchone()[0]
        primary = db.execute("SELECT count(*) FROM primary_conditions").fetchone()[0]
    expected_conditions = len(audit["bers"]) * (len(audit["methods"]) * len(audit["budgets"]) + 2)
    if counts != expected_conditions or primary != 280:
        raise ValueError("derived condition counts differ from frozen geometry")
    archive = output / "export_v1"
    if not archive.exists():
        def build_export(candidate: Path, final: Path) -> None:
            export(derived, candidate, public_destination=final)
            export_source_bundle(source, candidate, public_destination=final)
        _publish_staged(output, archive.name, build_export)
    exports = _verify_exports(archive)
    result = {"status": "FINALIZED_OFFLINE", "source_bundle": str(bundle),
              "scientific_payload_sha256": source["scientific_payload_sha256"],
              "source_checksums_sha256": source["checksums_sha256"],
              "source_revision": manifest["source"]["commit"], "field_audit": audit,
              "condition_count": counts, "historical_primary_conditions": primary,
              "database": str(derived / "reliability_v3.sqlite"),
              "database_sha256": sha256_file(derived / "reliability_v3.sqlite"),
              "exports": exports, "scp_commands": str(archive / "mac_scp_commands.txt"),
              "model_inferences_repeated": 0}
    record = output / "recovery_audit_v1.json"
    if record.exists():
        if json.loads(record.read_text()) != result:
            raise ValueError("existing recovery receipt differs")
    else:
        with record.open("x") as file:
            json.dump(result, file, indent=2, sort_keys=True)
            file.write("\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-sources", type=Path, default=Path("workspace/m12_kws_seven_method_followup/source_receipt.json"))
    parser.add_argument("--assets", type=Path, default=Path("/projects/kona-assets/manuelbv/NETSURF_ASSET_READINESS.json"))
    parser.add_argument("--baseline", type=Path, default=Path("workspace/m12_kws_seven_method_followup/derived_database_recovered_v2"))
    args = parser.parse_args()
    print(json.dumps(recover(args.receipt, args.plan, args.run_directory, args.output,
                             args.base_sources, args.assets, args.baseline), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
