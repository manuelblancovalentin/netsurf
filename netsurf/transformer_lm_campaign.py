"""Frozen, targeted 96-bit LM protection campaign on pinned READY assets.

This is a deliberately narrow linear-weight storage population, not a
full-model BER experiment. No inference occurs on import or during freeze.
"""
from __future__ import annotations

from dataclasses import asdict
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import time
import zipfile

import numpy as np

from .adapters.transformers_pinned import PinnedTransformerAdapter
from .faults import FaultEvent, FaultModel, FaultSchedule
from .identities import BitIdentity
from .manifests import canonical_json_bytes, digest_object, sha256_file
from .protection import ProtectionMask
from .torch_storage import TorchStoredLinearWeights, expected_faults, sparse_bernoulli_schedule
from .transformer_assets import load_ready_asset
from .transformer_baselines import (exact_isolated_scores, random_scores,
                                     select_equal_budget, significance_scores)
from .transformer_plan import PlanBlocked, require_final_gate, seal_selection, validate_frozen_plan

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = ("netsurf/transformer_lm_campaign.py", "netsurf/transformer_lm_campaign_cli.py",
    "scripts/run_transformer_lm_campaign_once.sh", "netsurf/adapters/transformers_pinned.py",
    "netsurf/transformer_assets.py", "netsurf/transformer_data.py", "netsurf/transformer_risk.py",
    "netsurf/transformer_baselines.py", "netsurf/transformer_plan.py", "netsurf/torch_storage.py",
    "netsurf/storage_bit_codec.py", "netsurf/faults.py", "netsurf/protection.py", "netsurf/manifests.py")
TARGETS = {"pythia160m_fp32stored_fp32compute", "pythia160m_bf16stored_fp32compute",
           "qwen3_0_6b_bf16"}
METHODS = ("random", "significance", "exact_isolated")
BERS = (0.0, 1 / 96, 4 / 96)
BUDGETS = (0, 12, 24, 96)
SELECTION_SEED = 120001
FINAL_SEED = 130001
ANALYSIS_SEED = 140001
CANDIDATE_SEED = 991001


def _write(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(canonical_json_bytes(document).decode("utf-8") + "\n")
        stream.flush(); os.fsync(stream.fileno())


def _bound(document: dict) -> dict:
    return {**document, "payload_sha256": digest_object(document)}


def _read_bound(path: Path, plan_sha256: str) -> dict:
    item = json.loads(path.read_text())
    if (item.get("plan_sha256") != plan_sha256 or
        item.get("payload_sha256") != digest_object({k:v for k,v in item.items() if k != "payload_sha256"})):
        raise PlanBlocked(f"corrupt or wrong-plan artifact: {path}")
    return item


def _candidate_bits(storage: TorchStoredLinearWeights) -> tuple[BitIdentity, ...]:
    width = storage.codec.width
    if width not in (16, 32) or storage.bit_count % width:
        raise PlanBlocked("unsupported pinned LM storage width")
    scalar_count = storage.bit_count // width
    if scalar_count < 32:
        raise PlanBlocked("too few verified linear-weight words")
    # Uniform without replacement over actual linear-weight words. Three bit
    # classes are fixed before any task outcome: sign, top exponent, mantissa.
    word_ordinals = sorted(random.Random(CANDIDATE_SEED).sample(range(scalar_count), 32))
    positions = (width - 1, width - 2, width - 10)
    bits = tuple(storage.bit_at(word * width + position)
                 for word in word_ordinals for position in positions)
    if len(bits) != 96 or len(set(bits)) != 96:
        raise PlanBlocked("candidate word/bit sampling did not yield 96 identities")
    return bits


def _token_sha(ids) -> str:
    return hashlib.sha256(np.asarray(ids, dtype="<i4").tobytes()).hexdigest()


def _source_binding() -> tuple[str, dict[str, str]]:
    subprocess.run(["git", "ls-files", "--error-unmatch", *SOURCE_FILES],
                   cwd=ROOT, capture_output=True, check=True)
    subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *SOURCE_FILES],
                   cwd=ROOT, check=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                     text=True).strip()
    return commit, {name: sha256_file(ROOT / name) for name in SOURCE_FILES}


def freeze(manifest_path: Path, registry_path: Path, target: str,
           plan_path: Path, snapshot_path: Path) -> dict:
    """Freeze candidate and input identities; no faulted or clean inference."""
    if target not in TARGETS or Path(plan_path).exists() or Path(snapshot_path).exists():
        raise PlanBlocked("unsupported target or existing immutable plan/snapshot")
    commit, source_hashes = _source_binding()
    asset = load_ready_asset(manifest_path, target, registry_path)
    adapter = PinnedTransformerAdapter(asset)
    adapter.load()
    full_bits = adapter.storage.bit_count
    candidates = _candidate_bits(adapter.storage)
    descriptors_sha = digest_object([asdict(item) for item in adapter.storage.descriptors])
    validation = adapter.token_sequence(split="validation")
    validation_identity = dict(adapter.last_token_sequence_identity)
    test = adapter.token_sequence(split="test")
    test_identity = dict(adapter.last_token_sequence_identity)
    selection_ids = validation[33:161]
    final_ids = test[:257]
    if len(selection_ids) != 128 or len(final_ids) != 257 or len(set(candidates)) != 96:
        raise PlanBlocked("pinned token cohorts or candidate support changed")
    if asset.precision == "BF16" and asset.computation_precision == "FP32":
        contract = "encoded BF16 source XOR, then verified widening into a distinct FP32 inference copy"
    else:
        contract = "encoded stored-word XOR in the inference-consumed CPU parameter"
    plan = {"schema_version": "transformer-campaign-plan/v1", "protocol_variant": "targeted-lm-96/v1",
        "status": "FROZEN", "target": target, "source_commit": commit,
        "source_files_sha256": source_hashes, "asset_manifest_sha256": sha256_file(manifest_path),
        "asset_record_sha256": asset.record_sha256, "registry_sha256": sha256_file(registry_path),
        "stored_precision": asset.precision, "computation_precision": asset.computation_precision,
        "storage_to_computation_contract": contract, "checkpoint_files": list(asset.model_files),
        "dataset_files": list(asset.dataset_files), "full_linear_weight_eligible_bits": full_bits,
        "full_linear_weight_descriptors_sha256": descriptors_sha,
        "candidate_sampling": "Python Random(991001).sample(range(full_verified_linear_weight_scalar_count),32), sorted; sign/top exponent/high significand of each word",
        "candidate_bits": [bit.key() for bit in candidates],
        "eligible_bits": 96, "eligible_population_sha256": digest_object([bit.key() for bit in candidates]),
        "budgets": list(BUDGETS), "ber": list(BERS), "methods": list(METHODS),
        "failure_penalty_nats": 30.0,
        "validation_token_identity": validation_identity, "test_token_identity": test_identity,
        "selection_token_slice": {"split": "validation", "start": 33, "stop": 161,
                                  "token_ids_sha256": _token_sha(selection_ids)},
        "final_token_slice": {"split": "test", "start": 0, "stop": 257,
                              "token_ids_sha256": _token_sha(final_ids)},
        "partitions_sha256": digest_object({"selection": ["validation", 33, 161, _token_sha(selection_ids)],
            "final": ["test", 0, 257, _token_sha(final_ids)]}),
        "selection": {"seed": SELECTION_SEED, "fields": 1, "sample_count": 127,
                      "exact_isolated_requests": 96, "random_significance_model_requests": 0},
        "final": {"seed": FINAL_SEED, "fields": 32, "sample_count": 256,
                  "shared_fault_field_across_policies": True},
        "analysis": {"seed": ANALYSIS_SEED, "independent_unit": "fault_field",
            "multiple_comparison_procedure": "two-sided paired sign-flip, Holm family of eight",
            "primary_family": "exact isolated versus significance at BER 4/96, budgets 12 and 24; two-sided Holm across two",
            "secondary_family": "all eight exact-isolated contrasts, two-sided Holm across eight",
            "bootstrap_replicates": 10000, "signflip_replicates": 10000,
            "input_reuse": "fixed test token cohort across all fields and BER arms; fields are stochastic units"},
        "costs": {"selection_model_requests": 97, "final_model_requests": 513,
            "total_model_requests": 610, "selection_scored_token_requests": 97 * 127,
            "final_scored_token_requests": 513 * 256,
            "total_scored_token_requests": 97 * 127 + 513 * 256,
            "selection_input_token_positions": 97 * 128,
            "final_input_token_positions": 513 * 257,
            "estimated_runtime_note": "Use measured 10-request smoke only as a lower-bound planning signal; benchmark 257-token calls before execution."},
        "scope": "96 presampled bits from 32 verified nn.Linear.weight words; no full-model BER claim"}
    plan["plan_sha256"] = digest_object(plan)
    validate_plan(plan)
    snapshot_path = Path(snapshot_path)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    with snapshot_path.open("xb") as stream:
        stream.write(Path(manifest_path).read_bytes()); stream.flush(); os.fsync(stream.fileno())
    if sha256_file(snapshot_path) != plan["asset_manifest_sha256"]:
        raise PlanBlocked("READY snapshot changed during freeze")
    _write(Path(plan_path), plan)
    return plan


def validate_plan(plan: dict) -> None:
    validate_frozen_plan(plan)
    if (plan.get("protocol_variant") != "targeted-lm-96/v1" or
        plan.get("target") not in TARGETS or plan.get("eligible_bits") != 96 or
        plan.get("budgets") != list(BUDGETS) or plan.get("ber") != list(BERS) or
        plan.get("methods") != list(METHODS) or
        plan.get("selection", {}).get("seed") != SELECTION_SEED or
        plan.get("final", {}).get("seed") != FINAL_SEED or
        plan.get("analysis", {}).get("seed") != ANALYSIS_SEED or
        plan.get("analysis", {}).get("primary_family") !=
            "exact isolated versus significance at BER 4/96, budgets 12 and 24; two-sided Holm across two" or
        plan.get("analysis", {}).get("secondary_family") !=
            "all eight exact-isolated contrasts, two-sided Holm across eight" or
        plan.get("selection_token_slice", {}).get("start") != 33 or
        plan.get("selection_token_slice", {}).get("stop") != 161 or
        plan.get("final_token_slice", {}).get("start") != 0 or
        plan.get("final_token_slice", {}).get("stop") != 257 or
        plan.get("costs", {}).get("total_model_requests") != 610):
        raise PlanBlocked("targeted LM protocol or exact costs changed")
    if len(plan.get("candidate_bits", ())) != 96 or len(set(plan["candidate_bits"])) != 96 or \
       digest_object(plan["candidate_bits"]) != plan["eligible_population_sha256"]:
        raise PlanBlocked("candidate population changed")
    if plan["partitions_sha256"] != digest_object({
        "selection": ["validation", 33, 161, plan["selection_token_slice"]["token_ids_sha256"]],
        "final": ["test", 0, 257, plan["final_token_slice"]["token_ids_sha256"]]}):
        raise PlanBlocked("input partition binding changed")


def _prepare(plan_path: Path, snapshot_path: Path, registry_path: Path):
    plan = json.loads(Path(plan_path).read_text())
    validate_plan(plan)
    if any(sha256_file(ROOT / name) != value for name, value in plan["source_files_sha256"].items()):
        raise PlanBlocked("scientific source differs from frozen LM plan")
    if sha256_file(snapshot_path) != plan["asset_manifest_sha256"] or \
       sha256_file(registry_path) != plan["registry_sha256"]:
        raise PlanBlocked("frozen READY snapshot or runtime registry changed")
    asset = load_ready_asset(snapshot_path, plan["target"], registry_path)
    if asset.record_sha256 != plan["asset_record_sha256"] or \
       asset.precision != plan["stored_precision"] or \
       asset.computation_precision != plan["computation_precision"]:
        raise PlanBlocked("target scientific runtime identity changed")
    adapter = PinnedTransformerAdapter(asset)
    adapter.load()
    if adapter.storage.bit_count != plan["full_linear_weight_eligible_bits"] or \
       digest_object([asdict(item) for item in adapter.storage.descriptors]) != plan["full_linear_weight_descriptors_sha256"]:
        raise PlanBlocked("full verified linear-weight universe changed")
    bits = _candidate_bits(adapter.storage)
    if [bit.key() for bit in bits] != plan["candidate_bits"]:
        raise PlanBlocked("presampled candidate bit identities changed")
    validation = adapter.token_sequence(split="validation")
    if adapter.last_token_sequence_identity != plan["validation_token_identity"]:
        raise PlanBlocked("validation token sequence changed")
    test = adapter.token_sequence(split="test")
    if adapter.last_token_sequence_identity != plan["test_token_identity"]:
        raise PlanBlocked("test token sequence changed")
    selection_ids, final_ids = validation[33:161], test[:257]
    if _token_sha(selection_ids) != plan["selection_token_slice"]["token_ids_sha256"] or \
       _token_sha(final_ids) != plan["final_token_slice"]["token_ids_sha256"]:
        raise PlanBlocked("fault-exposure token cohort changed")
    source_model = adapter.storage_model or adapter.model
    adapter.storage = TorchStoredLinearWeights(source_model, asset.precision, eligible_bits=bits)
    return plan, adapter, bits, selection_ids, final_ids


def _mask(name: str, selected, budget: int) -> ProtectionMask:
    return ProtectionMask(name, frozenset(selected), 96, budget, {})


def _selection_member(plan, method, bits, scores, member_hashes=()):
    masks = {str(b): sorted(bit.key() for bit in select_equal_budget(scores,b)) for b in BUDGETS}
    return _bound({"schema_version": "targeted-lm-selection-member/v1",
        "plan_sha256": plan["plan_sha256"], "method": method,
        "bit_order_sha256": plan["eligible_population_sha256"],
        "signed_scores": [scores[bit] for bit in bits], "masks": masks,
        "isolated_member_file_sha256": list(member_hashes)})


def run_selection(plan_path: Path, snapshot_path: Path, registry_path: Path,
                  run_root: Path) -> dict:
    plan, adapter, bits, selection_ids, _ = _prepare(plan_path, snapshot_path, registry_path)
    root = Path(run_root) / "selection"
    root.mkdir(parents=True, exist_ok=True)
    if (root / "seal.json").exists():
        require_final_gate(plan_path, root / "seal.json")
        return {"status": "SEALED_VERIFIED", "seal": str(root / "seal.json")}
    clean_path = root / "clean.json"
    if clean_path.exists():
        clean = _read_bound(clean_path, plan["plan_sha256"])
    else:
        clean = _bound({"schema_version": "targeted-lm-selection-clean/v1",
            "plan_sha256": plan["plan_sha256"],
            "risk": asdict(adapter.evaluate_language_model(selection_ids, context=1024, stride=512))})
        _write(clean_path, clean)
    law = FaultModel("weight", "targeted_verified_linear_weight_words",
                     "persistent_trial", "forced_engineering", 0.0)
    empty = _mask("unprotected", (), 0)
    member_hashes = []
    isolated = {}
    for ordinal, bit in enumerate(bits):
        path = root / "isolated" / f"bit_{ordinal:03d}.json"
        schedule = FaultSchedule(law, SELECTION_SEED + ordinal + 1,
                                 (FaultEvent(None, bit),), 96, len(selection_ids) - 1)
        if path.exists():
            item = _read_bound(path, plan["plan_sha256"])
            if item.get("bit") != bit.key() or item.get("schedule_sha256") != schedule.digest():
                raise PlanBlocked("isolated selection member changed")
        else:
            with adapter.injected(schedule, empty) as report:
                risk = adapter.evaluate_language_model(selection_ids, context=1024, stride=512)
            item = _bound({"schema_version": "targeted-lm-isolated-bit/v1",
                "plan_sha256": plan["plan_sha256"], "ordinal": ordinal,
                "bit": bit.key(), "schedule_sha256": schedule.digest(),
                "event": schedule.events[0].key(), "report": asdict(report), "risk": asdict(risk)})
            _write(path, item)
            print(f"selection isolated bit {ordinal+1}/96 sealed", flush=True)
        isolated[bit] = item["risk"]["mean_nll"] - clean["risk"]["mean_nll"]
        member_hashes.append(sha256_file(path))
    score_maps = {"random": random_scores(bits, SELECTION_SEED),
                  "significance": significance_scores(bits),
                  "exact_isolated": isolated}
    artifact_paths = {}
    for method in METHODS:
        path = root / f"{method}.json"
        expected = _selection_member(plan, method, bits, score_maps[method],
                                     member_hashes if method == "exact_isolated" else ())
        if path.exists():
            if _read_bound(path, plan["plan_sha256"]) != expected:
                raise PlanBlocked("existing selection member changed")
        else:
            _write(path, expected)
        artifact_paths[method] = path
    seal_selection(plan_path, artifact_paths, root / "seal.json")
    return {"status": "SEALED", "isolated_members": 96,
            "seal": str(root / "seal.json")}


def _policies(plan: dict, run_root: Path, bits) -> dict[str, ProtectionMask]:
    mapping = {bit.key(): bit for bit in bits}
    policies = {"unprotected": _mask("unprotected", (), 0),
                "full": _mask("full", bits, 96)}
    for method in METHODS:
        item = _read_bound(run_root / "selection" / f"{method}.json", plan["plan_sha256"])
        if item["method"] != method or item["bit_order_sha256"] != plan["eligible_population_sha256"]:
            raise PlanBlocked("selection method or bit population changed")
        scores = dict(zip(bits, item["signed_scores"]))
        if len(scores) != 96 or any(not math.isfinite(x) for x in scores.values()):
            raise PlanBlocked("selection scores invalid")
        if method == "exact_isolated":
            member_hashes = item.get("isolated_member_file_sha256")
            if (not isinstance(member_hashes, list) or len(member_hashes) != 96 or
                any(sha256_file(run_root / "selection" / "isolated" /
                    f"bit_{ordinal:03d}.json") != expected
                    for ordinal, expected in enumerate(member_hashes))):
                raise PlanBlocked("isolated selection evidence differs from sealed scores")
        for budget in (12, 24):
            names = item["masks"][str(budget)]
            if (len(names) != budget or set(names) != {bit.key() for bit in select_equal_budget(scores,budget)}):
                raise PlanBlocked("selection mask differs from sealed ranking")
            name = f"{method}_b{budget}"
            policies[name] = _mask(name, (mapping[key] for key in names), budget)
    return policies


def _field_path(root: Path, arm: int, field: int) -> Path:
    return root / "final" / f"ber_{arm:02d}" / f"field_{field:03d}.json"


def _schedule(storage, arm: int, field: int, n: int):
    return sparse_bernoulli_schedule(storage, ber=BERS[arm],
        seed=FINAL_SEED + arm * 100000 + field, inference_count=n, max_events=96)


def _verify_field(item: dict, intent: dict, schedule: FaultSchedule,
                  policies: dict[str, ProtectionMask], clean_risk: dict,
                  *, arm: int, field: int, ber: float) -> None:
    events = [event.key() for event in schedule.events]
    if (item.get("field") != field or item.get("arm") != arm or item.get("ber") != ber or
        item.get("seed") != schedule.seed or item.get("schedule_sha256") != schedule.digest() or
        item.get("events") != events or set(item.get("policies", {})) != set(policies) or
        intent.get("field") != field or intent.get("arm") != arm or intent.get("ber") != ber or
        intent.get("seed") != schedule.seed or intent.get("schedule_sha256") != schedule.digest() or
        intent.get("events") != events or set(intent.get("policy_names", ())) != set(policies)):
        raise PlanBlocked("final field identity, intent or schedule changed")
    for name, mask in policies.items():
        entry = item["policies"][name]
        report = entry["report"]
        applied = sum(event.bit not in mask.protected for event in schedule.events)
        if (entry["mask_sha256"] != mask.digest() or
            entry["expected_faults"] != expected_faults(96, mask.budget, ber) or
            report["schedule_hash"] != schedule.digest() or
            report["proposed_events"] != len(schedule.events) or
            report["applied_events"] != applied or
            report["suppressed_events"] != len(schedule.events) - applied or
            report["unique_applied_bits"] != applied or
            entry["risk"]["total_units"] != 256):
            raise PlanBlocked("final policy accounting or token count changed")
        if name == "full" and entry["risk"] != clean_risk:
            raise PlanBlocked("full protection changed from clean risk")


def run_final(plan_path: Path, snapshot_path: Path, registry_path: Path,
              run_root: Path) -> dict:
    require_final_gate(plan_path, Path(run_root) / "selection" / "seal.json")
    plan, adapter, bits, _, final_ids = _prepare(plan_path, snapshot_path, registry_path)
    root = Path(run_root)
    policies = _policies(plan, root, bits)
    clean_path = root / "final" / "clean.json"
    if clean_path.exists():
        clean = _read_bound(clean_path, plan["plan_sha256"])
    else:
        clean = _bound({"schema_version": "targeted-lm-final-clean/v1",
            "plan_sha256": plan["plan_sha256"],
            "risk": asdict(adapter.evaluate_language_model(final_ids, context=1024, stride=512))})
        _write(clean_path, clean)
    completed = 0
    for arm in (1, 2):
        for field in range(32):
            path = _field_path(root, arm, field)
            schedule = _schedule(adapter.storage, arm, field, 256)
            events = [event.key() for event in schedule.events]
            intent_path = path.with_suffix(".intent.json")
            intent = _bound({"schema_version": "targeted-lm-field-intent/v1",
                "plan_sha256": plan["plan_sha256"], "arm": arm, "field": field,
                "ber": BERS[arm], "seed": schedule.seed,
                "schedule_sha256": schedule.digest(), "events": events,
                "policy_names": sorted(policies)})
            if path.exists():
                item = _read_bound(path, plan["plan_sha256"])
                if not intent_path.exists() or _read_bound(intent_path, plan["plan_sha256"]) != intent:
                    raise PlanBlocked("completed final member cannot be verified for reuse")
                _verify_field(item, intent, schedule, policies, clean["risk"],
                              arm=arm, field=field, ber=BERS[arm])
                completed += 1
                continue
            if intent_path.exists():
                if _read_bound(intent_path, plan["plan_sha256"]) != intent:
                    raise PlanBlocked("partial field intent changed")
                raise PlanBlocked(f"partial field retained without final record: {intent_path}; manual same-identity recovery required")
            _write(intent_path, intent)
            results = {}
            for name, mask in policies.items():
                with adapter.injected(schedule, mask) as report:
                    risk = adapter.evaluate_language_model(final_ids, context=1024, stride=512)
                if name == "full" and asdict(risk) != clean["risk"]:
                    raise PlanBlocked("full logical protection did not reproduce clean final risk")
                results[name] = {"risk": asdict(risk), "report": asdict(report),
                                 "mask_sha256": mask.digest(),
                                 "expected_faults": expected_faults(96,mask.budget,BERS[arm])}
            item = _bound({"schema_version": "targeted-lm-final-field/v1",
                "plan_sha256": plan["plan_sha256"], "arm": arm, "field": field,
                "ber": BERS[arm], "seed": schedule.seed,
                "schedule_sha256": schedule.digest(), "events": events,
                "policies": results})
            _verify_field(item, intent, schedule, policies, clean["risk"],
                          arm=arm, field=field, ber=BERS[arm])
            _write(path, item)
            completed += 1
            print(f"BER arm {arm}/2 field {field+1}/32 sealed", flush=True)
    return {"status": "FINAL_MEMBERS_COMPLETE", "members": completed,
            "expected_members": 64}


def aggregate(plan_path: Path, run_root: Path) -> dict:
    plan = require_final_gate(plan_path, Path(run_root) / "selection" / "seal.json")
    validate_plan(plan)
    if any(sha256_file(ROOT / name) != value for name,value in plan["source_files_sha256"].items()):
        raise PlanBlocked("frozen LM campaign source changed")
    root = Path(run_root)
    keys = plan["candidate_bits"]
    bits = tuple(BitIdentity(key.split("[")[0], int(key.split("[")[1].split("]")[0]),
                             int(key.rsplit("#b",1)[1])) for key in keys)
    policies = _policies(plan, root, bits)
    class Offline:
        bit_count=96
        def bit_at(self, ordinal):return bits[ordinal]
    universe=Offline()
    clean = _read_bound(root / "final" / "clean.json", plan["plan_sha256"])
    if clean["risk"]["total_units"] != 256:
        raise PlanBlocked("clean final token count changed")
    rows=[];comparisons=[]
    for arm in (1,2):
        ber=BERS[arm]
        fields=[]
        for field in range(32):
            path=_field_path(root,arm,field)
            item=_read_bound(path,plan["plan_sha256"])
            schedule=_schedule(universe,arm,field,256)
            intent=_read_bound(path.with_suffix(".intent.json"),plan["plan_sha256"])
            _verify_field(item,intent,schedule,policies,clean["risk"],
                          arm=arm,field=field,ber=ber)
            fields.append(item)
        for name,mask in policies.items():
            risks=[field["policies"][name]["risk"] for field in fields]
            rows.append({"ber":ber,"budget":mask.budget,"method":name,
                "expected_proposed_faults":96*ber,
                "expected_applied_faults":(96-mask.budget)*ber,
                "mean_nll":statistics.mean(x["mean_nll"] for x in risks),
                "perplexity":math.exp(statistics.mean(x["mean_nll"] for x in risks)),
                "failed_token_fraction":statistics.mean(x["failed_units"]/256 for x in risks)})
        for budget in (12,24):
            for baseline in ("random","significance"):
                diffs=np.array([field["policies"][f"{baseline}_b{budget}"]["risk"]["mean_nll"]-
                    field["policies"][f"exact_isolated_b{budget}"]["risk"]["mean_nll"]
                    for field in fields])
                rng=np.random.default_rng(ANALYSIS_SEED+len(comparisons))
                draws=rng.integers(0,32,size=(10000,32))
                ci=np.quantile(diffs[draws].mean(axis=1),[.025,.975])
                signs=rng.choice(np.array([-1.0,1.0]),size=(10000,32))
                pv=(1+int(np.count_nonzero(np.abs((signs*diffs).mean(axis=1))>=abs(float(diffs.mean())))))/10001
                comparisons.append({"ber":ber,"budget":budget,
                    "comparison":f"exact_isolated_vs_{baseline}",
                    "mean_baseline_minus_isolated":float(diffs.mean()),
                    "ci95_percentile_bootstrap":[float(v) for v in ci],
                    "favorable_fields":int(np.count_nonzero(diffs>0)),
                    "adverse_fields":int(np.count_nonzero(diffs<0)),
                    "tied_fields":int(np.count_nonzero(diffs==0)),
                    "median_field_difference":float(np.median(diffs)),
                    "max_absolute_field_difference":float(np.max(np.abs(diffs))),
                    "p_unadjusted_signflip":pv})
    order=sorted(range(len(comparisons)),key=lambda i:comparisons[i]["p_unadjusted_signflip"])
    adjusted=0.0
    for rank,index in enumerate(order):
        adjusted=max(adjusted,min(1.0,(len(comparisons)-rank)*comparisons[index]["p_unadjusted_signflip"]))
        comparisons[index]["p_holm_family8"]=adjusted
    primary=[item for item in comparisons if item["ber"]==4/96 and
             item["comparison"]=="exact_isolated_vs_significance"]
    if len(primary)!=2 or {item["budget"] for item in primary}!={12,24}:
        raise PlanBlocked("prespecified LM primary family missing")
    adjusted=0.0
    for rank,item in enumerate(sorted(primary,key=lambda value:value["p_unadjusted_signflip"])):
        adjusted=max(adjusted,min(1.0,(2-rank)*item["p_unadjusted_signflip"]))
        item["p_holm_primary2"]=adjusted
    report=_bound({"schema_version":"targeted-lm-analysis/v1",
        "plan_sha256":plan["plan_sha256"],"scope":plan["scope"],
        "clean":clean["risk"],"rows":rows,"comparisons":comparisons,
        "uncertainty":"paired independent fault fields; same token cohort across arms; no independent-token replication claim"})
    path=root/"analysis.json"
    if path.exists():
        if _read_bound(path,plan["plan_sha256"])!=report:
            raise PlanBlocked("saved LM analysis differs from verified fields")
    else:_write(path,report)
    return report


def _svg(rows: list[dict], horizontal: str, target: str) -> str:
    """Small task-specific NLL plot; no cross-task metric mixing."""
    x_values = sorted({row[horizontal] for row in rows})
    low = min(row["mean_nll"] for row in rows)
    high = max(row["mean_nll"] for row in rows)
    if high == low:
        high = low + 1e-9
    names = sorted({row["method"] for row in rows})
    colors = ("#26547c", "#ef476f", "#06d6a0", "#ffd166", "#8338ec", "#ff7f11", "#3a86ff", "#555555")
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="500" viewBox="0 0 900 500">',
             '<rect width="900" height="500" fill="white"/>',
             f'<text x="40" y="28" font-size="17">{target}: targeted 96-bit mean token NLL</text>',
             '<path d="M70 45V410H650" fill="none" stroke="black"/>',
             f'<text x="320" y="475" font-size="14">{horizontal}</text>']
    for index, value in enumerate(x_values):
        x = 70 + 580 * index / max(1, len(x_values) - 1)
        parts.append(f'<text x="{x-15:.1f}" y="430" font-size="12">{value:.5g}</text>')
    for index in range(5):
        y = 410 - 365 * index / 4
        value = low + (high-low) * index / 4
        parts.append(f'<text x="5" y="{y+4:.1f}" font-size="11">{value:.4f}</text>')
    for index, name in enumerate(names):
        if horizontal == "ber":
            observations = [row for row in rows if row["method"] == name]
        else:
            # Budget panel shows the higher-BER arm; the lower arm is in CSV.
            observations = [row for row in rows if row["method"] == name and
                            row["ber"] == max(x["ber"] for x in rows)]
        points = []
        for row in sorted(observations, key=lambda value: value[horizontal]):
            x = 70 + 580 * x_values.index(row[horizontal]) / max(1, len(x_values)-1)
            y = 410 - 365 * (row["mean_nll"]-low)/(high-low)
            points.append(f"{x:.1f},{y:.1f}")
        if points:
            color = colors[index % len(colors)]
            parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"/>')
            parts.append(f'<text x="665" y="{55+index*24}" fill="{color}" font-size="11">{name}</text>')
    parts.append('</svg>')
    return "\n".join(parts) + "\n"


def export(plan_path: Path, snapshot_path: Path, registry_path: Path,
           run_root: Path, *, scp_host: str) -> Path:
    plan=json.loads(Path(plan_path).read_text());validate_plan(plan)
    if sha256_file(snapshot_path)!=plan["asset_manifest_sha256"] or \
       sha256_file(registry_path)!=plan["registry_sha256"]:
        raise PlanBlocked("frozen LM asset/registry snapshot changed")
    report=aggregate(plan_path,run_root)
    root=Path(run_root)
    destination=root.with_name(root.name+"_export")
    if destination.exists():
        checks=json.loads((destination/"checksums.json").read_text())
        if any(sha256_file(destination/name)!=value for name,value in checks.items()):
            raise PlanBlocked("existing LM export changed")
        with zipfile.ZipFile(destination/"complete.zip") as archive:
            if archive.testzip() is not None:raise PlanBlocked("existing LM ZIP corrupt")
        return destination
    stage=destination.with_name("."+destination.name+".incomplete")
    if stage.exists():raise FileExistsError("partial LM export retained")
    stage.mkdir()
    summary={"schema_version":"targeted-lm-summary/v1","plan_sha256":plan["plan_sha256"],
             "scope":report["scope"],"rows":report["rows"],"comparisons":report["comparisons"]}
    (stage/"summary.json").write_bytes(canonical_json_bytes(summary)+b"\n")
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=("ber", "budget", "method",
        "expected_proposed_faults", "expected_applied_faults", "mean_nll",
        "perplexity", "failed_token_fraction"))
    writer.writeheader(); writer.writerows(report["rows"])
    (stage/"table.csv").write_text(stream.getvalue())
    (stage/"ber_nll.svg").write_text(_svg(report["rows"],"ber",plan["target"]))
    (stage/"budget_nll.svg").write_text(_svg(report["rows"],"budget",plan["target"]))
    with zipfile.ZipFile(stage/"compact.zip","x",compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ("summary.json", "table.csv", "ber_nll.svg", "budget_nll.svg"):
            archive.write(stage/name,name)
    with zipfile.ZipFile(stage/"complete.zip","x",compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(plan_path,"plan.json")
        archive.write(snapshot_path,"assets/NETSURF_ASSET_READINESS.json")
        archive.write(registry_path,"assets/transformer_runtime_registry.json")
        for path in sorted(root.rglob("*")):
            if path.is_file():archive.write(path,"run/"+str(path.relative_to(root)))
        for name in ("summary.json", "table.csv", "ber_nll.svg", "budget_nll.svg"):
            archive.write(stage/name,"analysis/"+name)
        for name in SOURCE_FILES:archive.write(ROOT/name,"source/"+name)
    commands="\n".join(f"scp {scp_host}:{(destination/name).resolve()} ~/Downloads/{plan['target']}_{name}"
                       for name in ("compact.zip","complete.zip"))+"\n"
    (stage/"mac_scp_commands.txt").write_text(commands)
    checks={name:sha256_file(stage/name) for name in
            ("summary.json","table.csv","ber_nll.svg","budget_nll.svg",
             "compact.zip","complete.zip","mac_scp_commands.txt")}
    (stage/"checksums.json").write_bytes(canonical_json_bytes(checks)+b"\n")
    stage.rename(destination)
    return destination
