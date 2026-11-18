"""Read-only preflight and postflight gates for the frozen KWS matched grid.

This module neither constructs rankings nor evaluates faulted models.  The
schema-3 engine does those operations; this gate checks that the resulting
immutable evidence implements the prespecified matched comparison.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from netsurf.adapters.registry import create_adapter
from netsurf.fkeras_estimator import fkeras_family
from netsurf.asset_lock import load_benchmark_asset_lock
from netsurf.manifests import sha256_file
from netsurf.reliability_database import _verified_bundle
from netsurf.study_config import load_study


CAMPAIGN = "kws_partial_tier_fkeras_matched_v1"
SOURCE_STUDY = "kws_speechcommands_int8_full"
METHODS = (
    "random_global", "msb_global", "numerical_jump_global",
    "gradient_global", "per_example_absolute_global",
    "per_example_positive_global", "fkeras_paper",
)
BUDGETS = (0.0625, 0.125, 0.1875, 0.25)
BERS = (1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
COMPLETE_TIERS = (0.125, 0.25)
ELIGIBLE_BITS = 176128


def preflight(config_path: Path, plan: dict, sources_path: Path) -> dict:
    """Validate the frozen comparison geometry and its source/asset identity."""
    if plan["name"] != CAMPAIGN or plan["study_id"] != SOURCE_STUDY:
        raise ValueError("not the frozen KWS matched campaign")
    config = load_study(config_path)
    if (config.schema_version, config.scientific_protocol, config.name) != (
        "3.5.0", "multi-coverage-protection/v1", CAMPAIGN,
    ):
        raise ValueError("KWS schema/protocol/name changed")
    benchmark = config.benchmark
    if (benchmark.adapter, benchmark.id, benchmark.inference_backend,
            tuple(benchmark.injectable_parameters)) != (
        "mlperf_tiny_kws_int8_v1", "mlperf_tiny_kws_dscnn_int8_v1",
        "tensorflow_lite_builtin_ref_v1", ("kernels",),
    ):
        raise ValueError("KWS model/backend/eligible storage contract changed")
    if (benchmark.split, config.samples.ranking_count,
            config.samples.evaluation_count, config.samples.require_disjoint) != (
        "official_subset_1000", 64, 1000, True,
    ):
        raise ValueError("KWS calibration/evaluation cohort contract changed")
    if tuple(policy.name for policy in config.policies) != METHODS:
        raise ValueError("the seven frozen KWS methods changed")
    if {item.name for item in config.estimators} != {
        "random", "msb_first", "numerical_bit_jump",
        "first_order_gradient_delta", "per_sample_first_order_magnitude",
        "per_sample_positive_damage", "fkeras_paper",
    }:
        raise ValueError("published-form FKeras must be the only Hessian variant")
    fkeras = next(item for item in config.estimators if item.name == "fkeras_paper")
    if (fkeras.implementation_version != "fkeras-paper-continuous-kws/v1"
            or fkeras.hessian != {
                "eigenpairs": 2, "max_iterations": 100, "max_products": 300,
                "tolerance": 1e-5, "ordering": "largest_magnitude_signed",
            }):
        raise ValueError("bounded published-form FKeras definition changed")
    policy = next(item for item in config.policies if item.name == "fkeras_paper")
    if policy.priority != ("bit_significance_desc", "estimator_score_desc", "stable_identity"):
        raise ValueError("FKeras must rank by complete significance tiers first")
    if config.protection.coverages != BUDGETS or config.protection.probabilities != BERS:
        raise ValueError("KWS protection-budget or BER grid changed")
    if (config.execution.trials, config.replications.count,
            config.replications.sample_offset_stride,
            config.replications.fault_seed_stride,
            config.replications.estimator_seed_stride) != (8, 4, 0, 100003, 1009):
        raise ValueError("KWS trial/replication/seed geometry changed")
    if (config.fault.distribution, config.fault.persistence,
            config.protection.mitigation_model) != (
        "independent_bernoulli", "persistent_trial", "perfect_event_suppression",
    ):
        raise ValueError("KWS fault or mitigation contract changed")
    expected_evaluations = len(BERS) * config.execution.trials * config.replications.count * (2 + len(METHODS) * len(BUDGETS))
    if (plan["bers"], plan["interior_budgets"], plan["methods"],
            plan["independent_fault_fields_per_ber"],
            plan["policy_budget_evaluations"]) != (
        len(BERS), len(BUDGETS), len(METHODS), 32, expected_evaluations,
    ):
        raise ValueError("frozen operation-count receipt changed")
    sources = json.loads(sources_path.read_text())
    matches = [item for item in sources["studies"] if item["id"] == SOURCE_STUDY]
    if len(matches) != 1 or matches[0]["scientific_payload_sha256"] != plan["source_scientific_payload_sha256"]:
        raise ValueError("primary KWS source study identity changed")
    source_root, manifest, source_summary, _ = _verified_bundle(matches[0])
    source_benchmark = manifest["benchmark"]
    checkpoint_sha = sha256_file(benchmark.checkpoint)
    lock = load_benchmark_asset_lock(str(benchmark.asset_lock), "mlperf-tiny/kws")
    lock_sha = sha256_file(benchmark.asset_lock)
    if checkpoint_sha != source_benchmark["checkpoint_sha256"]:
        raise ValueError("frozen KWS checkpoint hash differs from verified source")
    dataset_identity = source_benchmark["dataset_identity"]
    reference = lock.benchmark["reference_results"]
    if (reference["model_sha256"] != checkpoint_sha
            or reference["sample_index_sha256"] != dataset_identity["sample_index_sha256"]):
        raise ValueError("KWS checkpoint or evaluation-index identity differs from asset lock")
    partitions = source_benchmark["sample_partitions"]
    if len(partitions) != 4 or len({json.dumps(roles, sort_keys=True) for roles in partitions.values()}) != 1:
        raise ValueError("primary KWS cohort varies across the four source replications")
    cohort = next(iter(partitions.values()))
    # The current Kona lock has a newer canonical digest than the historical
    # bundle's lock.  Never equate those two lock identities.  Instead verify
    # the exact checkpoint/index above and rederive every selected cohort from
    # the pinned reference adapter; fail if even one label-bound hash differs.
    adapter = create_adapter(config, None)
    adapter.load()
    inputs = {}
    ranking_data = None
    evaluation_data = None
    for role, count in (("ranking", 64), ("empirical_validation", 1), ("evaluation", 1000)):
        encoded, labels, selected = adapter.sample_partition(role, count)
        if selected.sample_sha256 != cohort[role]["sample_sha256"]:
            raise ValueError(f"KWS {role} cohort hash differs from finalized source")
        inputs[role] = {
            "sample_sha256": selected.sample_sha256,
            "encoded_input_sha256": hashlib.sha256(encoded.tobytes()).hexdigest(),
            "labels_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
        }
        if role == "ranking":
            ranking_data = (encoded, labels)
        elif role == "evaluation":
            evaluation_data = (encoded, labels)
    clean = adapter.evaluate(*evaluation_data)
    source_clean = source_summary["clean_evaluation"]
    if (clean.sample_count != source_clean["sample_count"]
            or clean.accuracy != source_clean["accuracy"]
            or abs(clean.categorical_crossentropy - source_clean["categorical_crossentropy"]) > 1e-12):
        raise ValueError("KWS clean reference inference differs from finalized source")
    output = fkeras_family(
        adapter, *ranking_data, tuple(adapter.bit_identities()),
        seed=fkeras.seed, options=dict(fkeras.hessian), requested=("fkeras_paper",),
    )["fkeras_paper"]
    eigensolve = output.metadata["eigenpairs"]
    if output.implementation_version != fkeras.implementation_version or len(output.scores) != ELIGIBLE_BITS:
        raise ValueError("bounded published-form FKeras scores do not cover the KWS population")
    return {
        "model_sha256": checkpoint_sha,
        "asset_lock_sha256": lock_sha,
        "asset_lock_digest": lock.digest,
        "source_asset_lock_digest": dataset_identity["asset_lock_digest"],
        "asset_lock_identity_equal_to_source": lock.digest == dataset_identity["asset_lock_digest"],
        "asset_lock_lineage_warning": (
            None if lock.digest == dataset_identity["asset_lock_digest"] else
            "Current self-validating lock has a different canonical digest from the historical source; "
            "the exact model, reference index, label-bound cohorts and current encoded inputs are verified separately."
        ),
        "sample_index_sha256": reference["sample_index_sha256"],
        "source_bundle": str(source_root),
        "source_scientific_payload_sha256": matches[0]["scientific_payload_sha256"],
        "cohort_sha256": {role: value["sample_sha256"] for role, value in cohort.items()},
        "current_encoded_cohort_receipt": inputs,
        "clean_reference": {
            "accuracy": clean.accuracy,
            "categorical_crossentropy": clean.categorical_crossentropy,
            "sample_count": clean.sample_count,
        },
        "eligible_bits": ELIGIBLE_BITS,
        "fault_fields": len(BERS) * config.execution.trials * config.replications.count,
        "policy_budget_evaluations": expected_evaluations,
        "estimated_clip_inferences": expected_evaluations * config.samples.evaluation_count,
        "bounded_hessian_preflight": {
            "definition": output.implementation_version,
            "eigenpairs": len(eigensolve["eigenvalues"]),
            "converged": eigensolve["converged"],
            "hvp_calls": eigensolve["hvp_calls"],
            "normalized_residuals": eigensolve["normalized_residuals"],
            "orthogonality_error": eigensolve["orthogonality_error"],
            "symmetry_error": eigensolve["symmetry_error"],
            "solver": eigensolve["solver"],
        },
    }


def postflight(bundle: Path, preflight_receipt: dict) -> dict:
    """Reject unmatched fields or false independent FKeras/MSB claims before ingest."""
    summary = json.loads((bundle / "summary.json").read_text())
    manifest = json.loads((bundle / "manifest.json").read_text())
    benchmark = manifest["benchmark"]
    if benchmark["checkpoint_sha256"] != preflight_receipt["model_sha256"]:
        raise ValueError("executed KWS checkpoint differs from preflight")
    if benchmark["dataset_identity"]["asset_lock_digest"] != preflight_receipt["asset_lock_digest"]:
        raise ValueError("executed KWS asset lock differs from preflight")
    for roles in benchmark["sample_partitions"].values():
        if {role: value["sample_sha256"] for role, value in roles.items()} != preflight_receipt["cohort_sha256"]:
            raise ValueError("executed KWS cohort differs from preflight")
    if summary["eligible_bits"] != ELIGIBLE_BITS:
        raise ValueError("KWS eligible bit population changed")
    equivalence = summary.get("mask_equivalence", [])
    duplicate_masks = []
    for replication in range(4):
        for budget in COMPLETE_TIERS:
            left = f"msb_global@{budget:.6f}"
            right = f"fkeras_paper@{budget:.6f}"
            matches = [row for row in equivalence if row["replication"] == f"replication_{replication}"
                       and {row["left"], row["right"]} == {left, right}]
            if len(matches) != 1 or not matches[0]["same_mask"] or matches[0]["symmetric_difference_bits"] != 0:
                raise ValueError(f"FKeras/MSB complete-tier mask mismatch: {replication}, {budget}")
            duplicate_masks.append({"replication": f"replication_{replication}",
                                    "coverage": budget, "methods": ["msb_global", "fkeras_paper"],
                                    "protected_bits": matches[0]["protected_bits"],
                                    "interpretation": "identical allocation, not independent policies"})
    by_field: dict[tuple[str, float, int], dict[str, dict]] = {}
    with (bundle / "measurements/trials.csv").open(newline="") as file:
        for row in csv.DictReader(file):
            ber_text, policy = row["policy"].split("/", 1)
            key = (row["replication"], float(ber_text.removeprefix("ber=")), int(row["trial"]))
            if policy in by_field.setdefault(key, {}):
                raise ValueError(f"duplicate policy for fault field {key}: {policy}")
            by_field[key][policy] = row
            if int(row["proposed_events"]) != int(row["applied_events"]) + int(row["suppressed_events"]):
                raise ValueError(f"event accounting fails for {key}, {policy}")
    expected_policies = {"no_protection", "full_protection"} | {
        f"{method}@{budget:.6f}" for method in METHODS for budget in BUDGETS
    }
    if len(by_field) != preflight_receipt["fault_fields"]:
        raise ValueError("wrong number of independent proposed fault fields")
    for key, policies in by_field.items():
        if set(policies) != expected_policies or len({row["schedule_sha256"] for row in policies.values()}) != 1:
            raise ValueError(f"unmatched policies or fault schedules for {key}")
        for budget in COMPLETE_TIERS:
            first = policies[f"msb_global@{budget:.6f}"]
            second = policies[f"fkeras_paper@{budget:.6f}"]
            for field in ("accuracy", "categorical_crossentropy", "applied_events", "suppressed_events"):
                if first[field] != second[field]:
                    raise ValueError(f"identical masks produced different {field} for {key}")
    return {"schema_version": "1.0.0", "paired_fault_fields": len(by_field),
            "policy_budget_evaluations": sum(len(rows) for rows in by_field.values()),
            "complete_tier_equivalence": duplicate_masks,
            "warning": "FKeras and MSB at 12.5% and 25% are duplicate protected sets; do not count them as independent allocation policies."}
