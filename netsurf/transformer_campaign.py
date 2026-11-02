"""Foreground, write-once DistilBERT targeted-bit BER/budget campaign.

The first 32 classifier scalar words are the complete declared 1,024-bit
fault-law support. This does not estimate full-model reliability. No final
field runs until all three method selections are sealed under one plan.
"""

from __future__ import annotations

from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import statistics
import subprocess

import numpy as np

from .adapters.transformers_pinned import PinnedTransformerAdapter, read_sst2_cohorts
from .faults import FaultEvent, FaultModel, FaultSchedule
from .identities import BitIdentity
from .manifests import canonical_json_bytes, digest_object, sha256_file
from .protection import ProtectionMask
from .torch_storage import TorchStoredLinearWeights, expected_faults, sparse_bernoulli_schedule
from .transformer_assets import load_ready_asset
from .transformer_baselines import exact_isolated_scores, random_scores, select_equal_budget, significance_scores
from .transformer_data import Sst2Row, sst2_records_digest
from .transformer_plan import PlanBlocked, require_final_gate, seal_selection, validate_frozen_plan
from .transformer_smoke import read_smoke_receipt


METHODS = ("random", "significance", "exact_isolated")


def targeted_classifier_bits() -> tuple[BitIdentity, ...]:
    return tuple(BitIdentity("classifier.weight", word, bit)
                 for word in range(32) for bit in range(32))


SOURCE_FILES = (
    "netsurf/transformer_campaign.py", "netsurf/transformer_campaign_cli.py",
    "netsurf/transformer_plan.py", "netsurf/transformer_assets.py",
    "netsurf/transformer_data.py", "netsurf/transformer_risk.py",
    "netsurf/transformer_baselines.py", "netsurf/torch_storage.py",
    "netsurf/storage_bit_codec.py", "netsurf/adapters/transformers_pinned.py",
    "netsurf/faults.py", "netsurf/protection.py", "netsurf/manifests.py",
    "scripts/run_distilbert_targeted_campaign_once.sh",
)


def validate_targeted_plan(plan: dict) -> str:
    checksum = validate_frozen_plan(plan)
    bits = targeted_classifier_bits()
    if (plan["target"] != "distilbert_sst2_fp32" or plan["eligible_bits"] != 1024 or
        plan.get("candidate_bit_keys") != [bit.key() for bit in bits] or
        plan["eligible_population_sha256"] != digest_object(plan["candidate_bit_keys"]) or
        plan["budgets"] != [0, 128, 256, 1024] or
        plan["ber"] != [0.0, 1/1024, 4/1024, 16/1024] or
        plan["methods"] != list(METHODS) or
        plan["selection"]["sample_count"] != 32 or plan["final"]["sample_count"] != 256 or
        plan["final"]["fields"] != 32 or plan.get("max_events_per_field") != 128 or
        set(plan.get("source_files_sha256", {})) != set(SOURCE_FILES)):
        raise PlanBlocked("targeted classifier scientific design differs from frozen protocol")
    costs = plan["costs"]
    if (costs["selection_model_requests"] != 1025 or costs["selection_clip_inferences"] != 32800 or
        costs["final_model_requests"] != 769 or costs["final_clip_inferences"] != 196864 or
        costs["total_model_requests"] != 1794 or costs["total_clip_inferences"] != 229664 or
        costs["final_fields"] != 96 or costs["policies_per_field"] != 8):
        raise PlanBlocked("campaign request/clip costs differ from complete enumeration")
    if (plan["analysis"]["seed"] in {plan["selection"]["seed"], plan["final"]["seed"]} or
        plan["analysis"]["bootstrap_replicates"] != 10000 or
        plan["analysis"]["signflip_replicates"] != 10000 or
        plan["analysis"]["multiple_comparison_procedure"] !=
        "Holm family of 12 paired sign-flip comparisons"):
        raise PlanBlocked("analysis family or seeds differ from frozen protocol")
    if len(set(plan["selection_source_indices"])) != 32 or len(set(plan["final_source_indices"])) != 256:
        raise PlanBlocked("selection/final cohort identities changed")
    return checksum


def verify_campaign_source(plan: dict) -> None:
    validate_targeted_plan(plan)
    root = Path(__file__).resolve().parents[1]
    for relative, expected in plan["source_files_sha256"].items():
        source = (root / relative).resolve()
        if not source.is_relative_to(root) or sha256_file(source) != expected:
            raise PlanBlocked(f"campaign source changed: {relative}")


def freeze_distilbert_plan(readiness_path: Path, portfolio_path: Path,
                           engineering_receipt: Path, output: Path,
                           snapshot: Path) -> dict:
    """Freeze one prospective study from READY identities and passed smoke.

    Only the engineering gate is inspected. No final risk or fault outcome is
    loaded; fresh selection and final seeds are fixed here.
    """
    if Path(output).exists() or Path(snapshot).exists():
        raise FileExistsError("plan or READY snapshot already exists; freeze is write-once")
    asset = load_ready_asset(readiness_path, "distilbert_sst2_fp32", portfolio_path)
    receipt = read_smoke_receipt(engineering_receipt)
    if receipt["asset_manifest_sha256"] != asset.manifest_sha256 or \
       receipt["asset_record_sha256"] != asset.record_sha256 or \
       receipt["source"]["dirty"] or receipt["schema_version"] != "transformer-smoke/v2" or \
       receipt["trials"]["full_control"]["risk"] != receipt["clean"]:
        raise PlanBlocked("committed-source real-model engineering gate is absent")
    cohorts = read_sst2_cohorts(asset)
    all_indices = {name: [row.source_index for row in rows] for name, rows in cohorts.items()}
    all_hashes = {name: _cohort_hash(rows,
        "train" if name in ("calibration", "selection") else "validation")
        for name, rows in cohorts.items()}
    selection = cohorts["selection"][:32]
    evaluation = cohorts["evaluation"][:256]
    selected_indices = [row.source_index for row in selection]
    final_indices = [row.source_index for row in evaluation]
    selection_hash = _cohort_hash(selection, "train")
    final_hash = _cohort_hash(evaluation, "validation")
    partition_binding = {"cohort_source_indices": all_indices,
        "cohort_records_sha256": all_hashes, "selection": selected_indices,
        "final": final_indices, "selection_records_sha256": selection_hash,
        "final_records_sha256": final_hash}
    source_root = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source_root, text=True).strip()
    source_hashes = {relative: sha256_file(source_root / relative) for relative in SOURCE_FILES}
    n = 1024
    nonzero = [1 / n, 4 / n, 16 / n]
    fields = 32
    policies_per_field = 2 + 2 * len(METHODS)
    selection_requests = 1 + n
    final_requests = 1 + len(nonzero) * fields * policies_per_field
    plan = {"schema_version": "transformer-campaign-plan/v1", "status": "FROZEN",
        "target": "distilbert_sst2_fp32", "scope": "first 32 classifier.weight FP32 words only",
        "source_commit": commit, "source_files_sha256": source_hashes,
        "portfolio_sha256": sha256_file(portfolio_path),
        "asset_manifest_sha256": asset.manifest_sha256,
        "asset_record_sha256": asset.record_sha256,
        "clean_reference_report_sha256": asset.record["clean_report_sha256"],
        "engineering_receipt_manifest_sha256": sha256_file(Path(engineering_receipt) / "manifest.json"),
        "eligible_population_sha256": digest_object([bit.key() for bit in targeted_classifier_bits()]),
        "eligible_bits": n, "candidate_bit_keys": [bit.key() for bit in targeted_classifier_bits()],
        "cohort_source_indices": all_indices, "cohort_records_sha256": all_hashes,
        "selection_source_indices": selected_indices, "final_source_indices": final_indices,
        "selection_records_sha256": selection_hash, "final_records_sha256": final_hash,
        "partitions_sha256": digest_object(partition_binding),
        "budgets": [0, 128, 256, n], "ber": [0.0, *nonzero],
        "methods": list(METHODS),
        "selection": {"seed": 16001, "fields": 1, "sample_count": len(selection),
                      "exact_isolated_requests": n, "random_and_significance": "no model calls"},
        "final": {"seed": 26001, "fields": fields, "sample_count": len(evaluation),
                  "schedule_seed_rule": "26001 + 100000*BER_arm + field_index",
                  "shared_field_across_policies": True},
        "max_events_per_field": 128,
        "failure_penalty_nats": 30.0,
        "analysis": {"independent_unit": "fault_field", "seed": 36001,
                     "multiple_comparison_procedure": "Holm family of 12 paired sign-flip comparisons",
                     "hypothesis_family": "exact isolated versus random and significance at each of two intermediate budgets and three nonzero BERs",
                     "bootstrap_replicates": 10000, "signflip_replicates": 10000,
                     "same_fixed_evaluation_clips_across_arms": True},
        "costs": {"selection_model_requests": selection_requests,
                  "selection_clip_inferences": selection_requests * len(selection),
                  "final_model_requests": final_requests,
                  "final_clip_inferences": final_requests * len(evaluation),
                  "total_model_requests": selection_requests + final_requests,
                  "total_clip_inferences": selection_requests * len(selection) + final_requests * len(evaluation),
                  "final_fields": len(nonzero) * fields,
                  "policies_per_field": policies_per_field,
                  "estimated_wall_hours_from_engineering_throughput": round(
                      (selection_requests * len(selection) + final_requests * len(evaluation))
                      / len(receipt["engineering_source_indices"]) *
                      (receipt["runtime_seconds"] / 11) / 3600, 2),
                  "measured_engineering_peak_rss_mb": receipt["peak_rss_mb"]},
        "exposure_note": "Kona clean reference saw all 872 validation rows; NetSuRF fault outcomes saw only validation rows 0-31 before this freeze. Train selection rows were in model fine-tuning data; final fault-outcome rows 32-287 are reserved and untouched."}
    plan["plan_sha256"] = digest_object(plan)
    validate_targeted_plan(plan)
    Path(snapshot).parent.mkdir(parents=True, exist_ok=True)
    with Path(snapshot).open("xb") as stream:
        stream.write(Path(readiness_path).read_bytes()); stream.flush(); os.fsync(stream.fileno())
    if sha256_file(snapshot) != asset.manifest_sha256:
        raise PlanBlocked("READY snapshot failed byte verification")
    _write_json(Path(output), plan)
    return plan


def _cohort_hash(rows, split: str) -> str:
    return sst2_records_digest(tuple(Sst2Row(split, r.source_index, r.text, r.label) for r in rows))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(canonical_json_bytes(payload).decode("utf-8") + "\n")
        stream.flush(); os.fsync(stream.fileno())


def _read_bound(path: Path, plan_hash: str) -> dict:
    data = json.loads(path.read_text())
    checksum = data.get("payload_sha256")
    if data.get("plan_sha256") != plan_hash or checksum != digest_object(
        {k: v for k, v in data.items() if k != "payload_sha256"}):
        raise PlanBlocked(f"corrupt or wrong-plan artifact: {path}")
    return data


def _bound(payload: dict) -> dict:
    return {**payload, "payload_sha256": digest_object(payload)}


def _prepare(plan_path: Path, readiness_path: Path, portfolio_path: Path):
    plan = json.loads(Path(plan_path).read_text())
    verify_campaign_source(plan)
    if sha256_file(Path(readiness_path)) != plan["asset_manifest_sha256"]:
        raise PlanBlocked("READY manifest differs from frozen campaign plan")
    if sha256_file(Path(portfolio_path)) != plan["portfolio_sha256"]:
        raise PlanBlocked("source portfolio differs from frozen campaign plan")
    asset = load_ready_asset(readiness_path, "distilbert_sst2_fp32", portfolio_path)
    if asset.record_sha256 != plan["asset_record_sha256"]:
        raise PlanBlocked("READY target record differs from frozen campaign plan")
    bits = targeted_classifier_bits()
    if plan["eligible_bits"] != len(bits) or plan["eligible_population_sha256"] != digest_object(
        [bit.key() for bit in bits]):
        raise PlanBlocked("targeted classifier bit identities differ from plan")
    cohorts = read_sst2_cohorts(asset)
    all_cohort_hashes = {name: _cohort_hash(rows,
        "train" if name in ("calibration", "selection") else "validation")
        for name, rows in cohorts.items()}
    all_cohort_indices = {name: [row.source_index for row in rows]
                          for name, rows in cohorts.items()}
    if all_cohort_hashes != plan["cohort_records_sha256"] or \
       all_cohort_indices != plan["cohort_source_indices"]:
        raise PlanBlocked("Kona cohort identities differ from frozen plan")
    selection = cohorts["selection"][:plan["selection"]["sample_count"]]
    evaluation = cohorts["evaluation"][:plan["final"]["sample_count"]]
    if [row.source_index for row in selection] != plan["selection_source_indices"] or \
       [row.source_index for row in evaluation] != plan["final_source_indices"] or \
       _cohort_hash(selection, "train") != plan["selection_records_sha256"] or \
       _cohort_hash(evaluation, "validation") != plan["final_records_sha256"]:
        raise PlanBlocked("fault-exposure input identities differ from plan")
    if plan["partitions_sha256"] != digest_object({
        "cohort_source_indices": all_cohort_indices,
        "cohort_records_sha256": all_cohort_hashes,
        "selection": plan["selection_source_indices"], "final": plan["final_source_indices"],
        "selection_records_sha256": plan["selection_records_sha256"],
        "final_records_sha256": plan["final_records_sha256"]}):
        raise PlanBlocked("partition binding differs from plan")
    adapter = PinnedTransformerAdapter(asset)
    adapter.load()
    adapter.storage = TorchStoredLinearWeights(adapter.model, "FP32", eligible_bits=bits)
    if adapter.storage.bit_count != 1024:
        raise PlanBlocked("runtime targeted bit count changed")
    return plan, adapter, bits, selection, evaluation


def _mask(name: str, bits: frozenset[BitIdentity], n: int, budget: int) -> ProtectionMask:
    return ProtectionMask(name, bits, n, budget, {})


def _selection_artifact(plan: dict, method: str, scores: dict[BitIdentity, float],
                        bits: tuple[BitIdentity, ...]) -> dict:
    masks = {str(budget): sorted(bit.key() for bit in select_equal_budget(scores, budget))
             for budget in plan["budgets"]}
    return _bound({"schema_version": "transformer-selection-member/v1",
        "plan_sha256": plan["plan_sha256"], "method": method,
        "signed_scores": [scores[bit] for bit in bits], "masks": masks,
        "bit_order_sha256": plan["eligible_population_sha256"]})


def run_selection(plan_path: Path, readiness_path: Path, portfolio_path: Path,
                  run_root: Path) -> dict:
    plan, adapter, bits, selection, _ = _prepare(plan_path, readiness_path, portfolio_path)
    root = Path(run_root) / "selection"
    root.mkdir(parents=True, exist_ok=True)
    paths = {method: root / f"{method}.json" for method in METHODS}
    if tuple(plan["methods"]) != METHODS:
        raise PlanBlocked("method family differs from campaign implementation")
    if (root / "seal.json").exists():
        require_final_gate(plan_path, root / "seal.json")
        return {"status": "SEALED", "seal": str(root / "seal.json")}
    for method in METHODS:
        if paths[method].exists():
            item = _read_bound(paths[method], plan["plan_sha256"])
            if item.get("method") != method or len(item.get("signed_scores", [])) != len(bits):
                raise PlanBlocked(f"invalid completed selection member: {method}")
            continue
        if method == "random":
            scores = random_scores(bits, plan["selection"]["seed"])
        elif method == "significance":
            scores = significance_scores(bits)
        else:
            clean = adapter.evaluate_sst2(selection)
            law = FaultModel("weight", "cpu_classifier_first_32_words", "persistent_trial",
                             "forced_engineering", 0.0)
            empty = _mask("unprotected", frozenset(), len(bits), 0)

            def one(bit: BitIdentity) -> float:
                ordinal = bits.index(bit)
                schedule = FaultSchedule(law, plan["selection"]["seed"] + 1 + ordinal,
                    (FaultEvent(None, bit),), len(bits), len(selection))
                with adapter.storage.injected(schedule, empty):
                    return adapter.evaluate_sst2(selection).mean_nll

            scores = exact_isolated_scores(bits, clean.mean_nll, one)
        artifact = _selection_artifact(plan, method, scores, bits)
        _write_json(paths[method], artifact)
    seal_selection(plan_path, paths, root / "seal.json")
    return {"status": "SEALED", "seal": str(root / "seal.json"),
            "methods": list(METHODS)}


def _policy_masks(plan: dict, root: Path, bits: tuple[BitIdentity, ...]) -> dict[str, ProtectionMask]:
    key_to_bit = {bit.key(): bit for bit in bits}
    n = len(bits)
    policies = {"unprotected": _mask("unprotected", frozenset(), n, 0),
                "full": _mask("full", frozenset(bits), n, n)}
    for method in METHODS:
        item = _read_bound(root / "selection" / f"{method}.json", plan["plan_sha256"])
        if item["method"] != method or item["bit_order_sha256"] != plan["eligible_population_sha256"]:
            raise PlanBlocked("selection member method or bit population changed")
        if len(item["signed_scores"]) != n or any(not math.isfinite(value) for value in item["signed_scores"]):
            raise PlanBlocked("selection member scores missing or nonfinite")
        scores = dict(zip(bits, item["signed_scores"]))
        for budget in plan["budgets"][1:-1]:
            keys = item["masks"][str(budget)]
            if len(keys) != budget or len(set(keys)) != budget or any(key not in key_to_bit for key in keys):
                raise PlanBlocked("selected mask has wrong budget or coordinates")
            if set(keys) != {bit.key() for bit in select_equal_budget(scores, budget)}:
                raise PlanBlocked("selected mask differs from frozen ranking")
            name = f"{method}_b{budget}"
            policies[name] = _mask(name, frozenset(key_to_bit[key] for key in keys), n, budget)
    return policies


def _member_path(root: Path, arm: int, field: int) -> Path:
    return root / "final" / f"ber_{arm:02d}" / f"field_{field:03d}.json"


def run_final(plan_path: Path, readiness_path: Path, portfolio_path: Path,
              run_root: Path) -> dict:
    plan = require_final_gate(plan_path, Path(run_root) / "selection" / "seal.json")
    plan, adapter, bits, _, evaluation = _prepare(plan_path, readiness_path, portfolio_path)
    root = Path(run_root)
    policies = _policy_masks(plan, root, bits)
    clean_path = root / "final" / "clean.json"
    if clean_path.exists():
        clean = _read_bound(clean_path, plan["plan_sha256"])
    else:
        risk = adapter.evaluate_sst2(evaluation)
        clean = _bound({"schema_version": "transformer-final-clean/v1",
            "plan_sha256": plan["plan_sha256"], "risk": asdict(risk)})
        _write_json(clean_path, clean)
    completed = 0
    for arm, ber in enumerate(plan["ber"][1:], start=1):
        for field in range(plan["final"]["fields"]):
            path = _member_path(root, arm, field)
            seed = plan["final"]["seed"] + arm * 100000 + field
            schedule = sparse_bernoulli_schedule(adapter.storage, ber=ber, seed=seed,
                inference_count=len(evaluation), max_events=plan["max_events_per_field"])
            if path.exists():
                prior = _read_bound(path, plan["plan_sha256"])
                if prior.get("seed") != seed or prior.get("schedule_sha256") != schedule.digest() or \
                   prior.get("ber") != ber or prior.get("schedule_events") != [event.key() for event in schedule.events] or \
                   set(prior.get("policies", {})) != set(policies):
                    raise PlanBlocked(f"final member cannot be verified for reuse: {path}")
                completed += 1
                continue
            results = {}
            for name, mask in policies.items():
                with adapter.storage.injected(schedule, mask) as report:
                    risk = adapter.evaluate_sst2(evaluation)
                if name == "full" and asdict(risk) != clean["risk"]:
                    raise PlanBlocked("full logical protection did not restore clean risk")
                results[name] = {"risk": asdict(risk), "report": asdict(report),
                                 "mask_sha256": mask.digest(),
                                 "expected_faults": expected_faults(len(bits), mask.budget, ber)}
            payload = _bound({"schema_version": "transformer-final-field/v1",
                "plan_sha256": plan["plan_sha256"], "arm": arm, "ber": ber,
                "field": field, "seed": seed, "schedule_sha256": schedule.digest(),
                "schedule_events": [event.key() for event in schedule.events],
                "policies": results})
            _write_json(path, payload)
            completed += 1
            print(f"BER arm {arm}/{len(plan['ber'])-1}, field {field+1}/{plan['final']['fields']} sealed", flush=True)
    return {"status": "FINAL_MEMBERS_COMPLETE", "members": completed,
            "expected_members": (len(plan["ber"]) - 1) * plan["final"]["fields"]}


def aggregate(plan_path: Path, run_root: Path) -> dict:
    plan = require_final_gate(plan_path, Path(run_root) / "selection" / "seal.json")
    verify_campaign_source(plan)
    root = Path(run_root)
    bits = targeted_classifier_bits()
    policies = _policy_masks(plan, root, bits)
    class OfflineUniverse:
        bit_count = len(bits)
        def bit_at(self, ordinal: int) -> BitIdentity:
            return bits[ordinal]
    universe = OfflineUniverse()
    clean = _read_bound(root / "final" / "clean.json", plan["plan_sha256"])
    if clean["risk"]["total_units"] != plan["final"]["sample_count"]:
        raise PlanBlocked("clean final cohort count changed")
    rows, tests = [], []
    for arm, ber in enumerate(plan["ber"][1:], start=1):
        fields = [_read_bound(_member_path(root, arm, field), plan["plan_sha256"])
                  for field in range(plan["final"]["fields"])]
        if any(item["arm"] != arm or item["ber"] != ber or item["field"] != i
               for i, item in enumerate(fields)):
            raise PlanBlocked("final field identities changed")
        for index, item in enumerate(fields):
            seed = plan["final"]["seed"] + arm * 100000 + index
            schedule = sparse_bernoulli_schedule(universe, ber=ber, seed=seed,
                inference_count=plan["final"]["sample_count"],
                max_events=plan["max_events_per_field"])
            if (item["seed"] != seed or item["schedule_sha256"] != schedule.digest() or
                item["schedule_events"] != [event.key() for event in schedule.events] or
                set(item["policies"]) != set(policies)):
                raise PlanBlocked("final event schedule or policy family differs from plan")
            for name, mask in policies.items():
                entry = item["policies"][name]
                applied = sum(event.bit not in mask.protected for event in schedule.events)
                if (entry["mask_sha256"] != mask.digest() or
                    entry["report"]["schedule_hash"] != schedule.digest() or
                    entry["report"]["proposed_events"] != len(schedule.events) or
                    entry["report"]["applied_events"] != applied or
                    entry["report"]["suppressed_events"] != len(schedule.events) - applied or
                    entry["expected_faults"] != expected_faults(len(bits), mask.budget, ber) or
                    entry["risk"]["total_units"] != plan["final"]["sample_count"]):
                    raise PlanBlocked("final member accounting or risk cohort differs from plan")
        for control, budget in (("unprotected", 0), ("full", plan["eligible_bits"])):
            values = [item["policies"][control]["risk"] for item in fields]
            rows.append({"ber": ber, "budget": budget, "method": control,
                         "expected_applied_faults": expected_faults(plan["eligible_bits"], budget, ber)["applied"],
                         "mean_nll": statistics.mean(value["mean_nll"] for value in values),
                         "accuracy": statistics.mean(value["accuracy"] for value in values),
                         "failed_clip_fraction": statistics.mean(value["failed_units"] / value["total_units"]
                                                                 for value in values)})
        for budget in plan["budgets"][1:-1]:
            for method in METHODS:
                name = f"{method}_b{budget}"
                values = [item["policies"][name]["risk"] for item in fields]
                rows.append({"ber": ber, "budget": budget, "method": method,
                             "expected_applied_faults": expected_faults(plan["eligible_bits"], budget, ber)["applied"],
                             "mean_nll": statistics.mean(value["mean_nll"] for value in values),
                             "accuracy": statistics.mean(value["accuracy"] for value in values),
                             "failed_clip_fraction": statistics.mean(value["failed_units"] /
                                 value["total_units"] for value in values)})
            for baseline in ("random", "significance"):
                a = f"{baseline}_b{budget}"; b = f"exact_isolated_b{budget}"
                differences = np.array([item["policies"][a]["risk"]["mean_nll"] -
                                        item["policies"][b]["risk"]["mean_nll"] for item in fields])
                rng = np.random.default_rng(plan["analysis"]["seed"] + len(tests))
                draws = rng.integers(0, len(differences), size=(10000, len(differences)))
                means = differences[draws].mean(axis=1)
                # Paired sign-flip randomization under the stated symmetric-null assumption.
                signs = rng.choice(np.array([-1.0, 1.0]), size=(10000, len(differences)))
                p_value = (1 + int(np.count_nonzero(np.abs((signs * differences).mean(axis=1)) >=
                                                      abs(float(differences.mean()))))) / 10001
                tests.append({"ber": ber, "budget": budget, "comparison": f"exact_isolated_vs_{baseline}",
                    "mean_baseline_minus_isolated": float(differences.mean()),
                    "ci95_percentile_bootstrap": [float(x) for x in np.quantile(means, [0.025, 0.975])],
                    "favorable_fields": int(np.count_nonzero(differences > 0)),
                    "adverse_fields": int(np.count_nonzero(differences < 0)),
                    "median_field_difference": float(np.median(differences)),
                    "max_absolute_field_difference": float(np.max(np.abs(differences))),
                    "p_unadjusted_signflip": p_value})
    order = sorted(range(len(tests)), key=lambda i: tests[i]["p_unadjusted_signflip"])
    cumulative = 0.0
    for rank, index in enumerate(order):
        cumulative = max(cumulative, min(1.0, (len(tests) - rank) * tests[index]["p_unadjusted_signflip"]))
        tests[index]["p_holm_family12"] = cumulative
    report = _bound({"schema_version": "transformer-targeted-analysis/v1",
        "plan_sha256": plan["plan_sha256"], "clean": clean["risk"], "rows": rows,
        "comparisons": tests, "scope": "targeted classifier 1,024-bit support; fault fields are independent units",
        "uncertainty": "percentile bootstrap over 32 fields; sign-flip assumes symmetric null; fixed inputs do not estimate corpus-sampling uncertainty"})
    output = root / "analysis.json"
    if output.exists():
        if _read_bound(output, plan["plan_sha256"]) != report:
            raise PlanBlocked("existing analysis differs from verified member aggregation")
    else:
        _write_json(output, report)
    return report
