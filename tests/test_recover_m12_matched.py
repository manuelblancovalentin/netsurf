"""Bounded, synthetic checks for the offline-only M12 recovery gate."""

from __future__ import annotations

import csv
import json

import pytest

from workflows.recover_m12_matched import _mask_sha256, audit_saved


def _fixture(tmp_path):
    bundle = tmp_path
    (bundle / "artifacts").mkdir()
    (bundle / "measurements").mkdir()
    methods = ("random_global", "msb_global", "numerical_jump_global",
               "gradient_global", "per_example_absolute_global", "per_example_positive_global")
    budgets = (0.0625, 0.125, 0.1875, 0.25)
    bers = (0.001, 0.01)
    bits = [["kernel", word, bit] for word in range(2) for bit in range(8)]
    policies = {"no_protection": [], "full_protection": bits}
    for method in methods:
        for budget in budgets:
            policies[f"{method}@{budget:.6f}"] = bits[:int(16 * budget)]
    (bundle / "artifacts/protection_masks.json").write_text(json.dumps({"replication_0": policies}))
    schedules = [{"logical_ber": ber, "schedule_index": i, "sha256": f"schedule-{i}",
                  "events": [[None, "kernel", 0, 7]]} for i, ber in enumerate(bers)]
    (bundle / "artifacts/fault_schedules.json").write_text(json.dumps({"replication_0": schedules}))
    with (bundle / "measurements/trials.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("replication", "trial", "policy", "schedule_sha256",
                                                 "proposed_events", "applied_events", "suppressed_events",
                                                 "accuracy", "categorical_crossentropy"))
        writer.writeheader()
        for i, ber in enumerate(bers):
            for policy in policies:
                full = policy == "full_protection"
                writer.writerow({"replication": "replication_0", "trial": 0,
                                 "policy": f"ber={ber:g}/{policy}", "schedule_sha256": f"schedule-{i}",
                                 "proposed_events": 1, "applied_events": 0 if full else 1,
                                 "suppressed_events": 1 if full else 0,
                                 "accuracy": 0.9 if full else 0.8,
                                 "categorical_crossentropy": 0.2 if full else 0.3})
    with (bundle / "measurements/replication_protection_results.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("replication", "logical_ber", "condition",
                                                 "protected_bits", "mask_sha256"))
        writer.writeheader()
        for ber in bers:
            for policy, selected in policies.items():
                writer.writerow({"replication": "replication_0", "logical_ber": ber,
                                 "condition": policy, "protected_bits": len(selected),
                                 "mask_sha256": _mask_sha256(selected)})
    plan = {"bers": 2, "independent_fault_fields_per_ber": 1,
            "policy_budget_evaluations": 2 * 26}
    config = {"replications": {"count": 1}, "protection": {"probabilities": bers,
              "coverages": budgets}, "execution": {"trials": 1},
              "policies": [{"name": method} for method in methods]}
    return bundle, plan, config, {"clean_evaluation": {"accuracy": 0.9,
                                 "categorical_crossentropy": 0.2}, "eligible_bits": 16}


def test_offline_audit_accepts_complete_paired_grid(tmp_path):
    bundle, plan, config, summary = _fixture(tmp_path)
    audit = audit_saved(bundle, plan, config, summary)
    assert audit["paired_fields"] == 2
    assert audit["saved_evaluations"] == 52
    assert audit["eligible_bits"] == 16


def test_offline_audit_rejects_wrong_mask_and_unpaired_field(tmp_path):
    bundle, plan, config, summary = _fixture(tmp_path)
    masks_path = bundle / "artifacts/protection_masks.json"
    masks = json.loads(masks_path.read_text())
    masks["replication_0"]["msb_global@0.125000"][0] = ["kernel", 0, 1]
    masks_path.write_text(json.dumps(masks))
    with pytest.raises(ValueError, match="cardinality or uniqueness"):
        audit_saved(bundle, plan, config, summary)


def test_offline_audit_rejects_bits_outside_eligible_population(tmp_path):
    bundle, plan, config, summary = _fixture(tmp_path)
    masks_path = bundle / "artifacts/protection_masks.json"
    masks = json.loads(masks_path.read_text())
    masks["replication_0"]["msb_global@0.125000"][0] = ["other-kernel", 0, 0]
    masks_path.write_text(json.dumps(masks))
    with pytest.raises(ValueError, match="outside the eligible population"):
        audit_saved(bundle, plan, config, summary)


def test_offline_audit_rejects_missing_policy_trial(tmp_path):
    bundle, plan, config, summary = _fixture(tmp_path)
    trial_path = bundle / "measurements/trials.csv"
    rows = trial_path.read_text().splitlines()
    trial_path.write_text("\n".join(rows[:-1]) + "\n")
    with pytest.raises(ValueError, match="incomplete"):
        audit_saved(bundle, plan, config, summary)


def test_offline_audit_preserves_complete_file_auc_semantics(tmp_path):
    bundle, plan, config, _ = _fixture(tmp_path)
    trial_path = bundle / "measurements/trials.csv"
    with trial_path.open(newline="") as file:
        old = list(csv.DictReader(file))
    fields = ("replication", "trial", "policy", "schedule_sha256", "proposed_events",
              "applied_events", "suppressed_events", "primary_metric_name",
              "primary_metric_value", "loss_metric_name", "loss_metric_value")
    with trial_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in old:
            writer.writerow({key: row[key] for key in fields[:7]} | {
                "primary_metric_name": "roc_auc", "primary_metric_value": row["accuracy"],
                "loss_metric_name": "reconstruction_mse",
                "loss_metric_value": row["categorical_crossentropy"],
            })
    summary = {"clean_evaluation": {"roc_auc": 0.9, "reconstruction_mse": 0.2},
               "eligible_bits": 16}
    assert audit_saved(bundle, plan, config, summary)["task_metric"] == "roc_auc"
    rows = trial_path.read_text().replace("roc_auc", "accuracy", 1)
    trial_path.write_text(rows)
    with pytest.raises(ValueError, match="mislabeled"):
        audit_saved(bundle, plan, config, summary)
