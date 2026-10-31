"""Frozen joint BER campaign using the M9.6 exact selector and paired evaluator.

The twelve source pools are copied by identity and hash; no candidate RNG is used.
Each arm remains an ordinary schema-6.6 campaign member, while this parent
enforces the all-arms selection boundary and binds the family before any final.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from .campaign_exposure import audit_exposure, check_exposure
from .cross_pool import (_bound_document, _read, _receipt, _toml, load_campaign,
                         operation_counts, verified_receipt, seal_selections, run_campaign)
from .manifests import digest_object, sha256_file
from .result_bundle import _write_bytes, _write_json

VERSION = "1.0.0"
BER_GRID = (0.001, 0.003, 0.01)
SOURCE = Path("/projects/netsurf/manuelbv/workspace/implementation-worktrees/m96/workspace/runs/m96_kws_cross_pool_k6_v1/campaign.json")
SOURCE_SHA256 = "46674ad4f90e8bc49fcee3ef826bbb57130005f239d192700cd09d8f106d3ff3"


def arm_id(probability):
    if probability not in BER_GRID:
        raise ValueError("BER is outside the prespecified family")
    return "ber_" + str(probability).replace(".", "p")


def _seed_plan(arm_index, pool_index, fields):
    base = 1_000_000_000 + arm_index * 3_000_000_000 + pool_index * 200_000_000
    selection = [base + i for i in range(4)]
    final = base + 20_000_000
    used = [seed + j * 1_000_003 for seed in selection for j in range(fields)]
    used.extend(final + j * 1_000_003 for j in range(128))
    used.append(base + 9_000_000)  # clean/isolated selection state
    return selection, final, used


def _check_seed_family(arms):
    all_seeds = []
    for arm in arms:
        design = arm["design"]
        for i, pool in enumerate(arm["pools"]):
            document = pool["selection_document"]
            seeds = document["restricted_oracle"]["calibration_replicate_seeds"]
            all_seeds.append(document["restricted_oracle"]["calibration_seed"] - 1)
            all_seeds.extend(s + j * 1_000_003 for s in seeds
                             for j in range(design["selection_fields_per_seed"]))
            all_seeds.extend(design["final_seeds"][i] + j * 1_000_003
                             for j in range(design["final_fields"]))
    if len(all_seeds) != len(set(all_seeds)):
        raise ValueError("selection/final seed collision anywhere in BER family")


def _window(audit, first, count, purpose):
    indices = list(range(first, first + count))
    overlap = sorted(set(indices) & set(audit["exposed_test_indices"]))
    if purpose == "confirmation" and overlap:
        raise ValueError(f"prior exposure overlap: {overlap}")
    return {"original_indices": indices, "sample_ids": [f"{audit['source_split']}:{i}" for i in indices],
            "prior_fault_outcome_overlap": overlap, "ledger_sha256": digest_object(audit),
            "identity_scope": "ordered pinned TFDS test indices; labels and encoded input hashes verified before freeze and at member execution"}


def _verify_windows(source, selection, final):
    """Bind label and encoded-input identities without sampling fault outcomes."""
    from .adapters.registry import create_adapter
    from .conditional_config import load_conditional
    from .conditional_damage import _sample_identity_records
    config = load_conditional(source.parent / "configs/pool_000_selection.toml")
    benchmark = create_adapter(config, None)
    benchmark.load()  # Includes the historically audited 16-clip clean reference gate.
    for window, role, start in ((selection, "conditional_model_selection", 2024),
                                (final, "conditional_final_evaluation", 3048)):
        first = window["original_indices"][0]
        inputs, labels, partition = benchmark.sample_partition(role, len(window["original_indices"]),
                                                               offset=first - start)
        if list(partition.indices) != window["original_indices"] or partition.source_split != "speech_commands_tfds_test_interleaved_blocks16":
            raise ValueError("KWS sample partition differs from planned identities")
        rows = _sample_identity_records(inputs, labels, partition)
        window["sample_sha256"] = partition.sample_sha256
        window["identity_records_sha256"] = digest_object(rows)
        window["identity_records"] = rows
        window["clean_reference_inference_previously_performed"] = True


def verify_partition(parent, name, data):
    from .conditional_damage import _sample_identity_records
    inputs, labels, partition = data
    expected = parent["selection_window" if name == "model_selection" else "final_window"]
    if list(partition.indices) != expected["original_indices"] or partition.sample_sha256 != expected["sample_sha256"]:
        raise ValueError("actual KWS partition differs from frozen joint identity")
    if digest_object(_sample_identity_records(inputs, labels, partition)) != expected["identity_records_sha256"]:
        raise ValueError("actual KWS labels or encoded inputs differ from frozen joint identity")


def _source_receipts(source, source_record):
    digest = sha256_file(source)
    result = {}
    for pool in source_record["pools"]:
        pid = pool["pool_id"]
        result[pid] = {}
        for phase in ("selection", "final"):
            path = _receipt(source.parent, pid, phase)
            receipt, _ = verified_receipt(path, digest, pid, phase)
            result[pid][phase] = {"receipt_sha256": sha256_file(path),
                                  "payload_sha256": receipt["payload_sha256"]}
    return result


def freeze_joint(source=SOURCE, output=None, *, engineering=False):
    """Freeze before inference. Engineering uses only previously exposed clips."""
    source = Path(source).resolve()
    if source != SOURCE or sha256_file(source) != SOURCE_SHA256:
        raise ValueError("M9.7 requires the exact frozen M9.6 source manifest")
    prior = load_campaign(source)
    if prior["design"]["logical_ber"] != .01 or prior["design"]["candidate_count"] != 6:
        raise ValueError("source is not the frozen M9.6 K=6 BER-0.01 campaign")
    if len(prior["pools"]) != 12 or [p["pool_id"] for p in prior["pools"]] != [f"pool_{i:03d}" for i in range(12)]:
        raise ValueError("source must contain all twelve ordered pools")
    audit = audit_exposure()
    purpose = "engineering" if engineering else "confirmation"
    levels = (.003,) if engineering else BER_GRID
    pools = prior["pools"][:1] if engineering else prior["pools"]
    selection_first, selection_count = (2088, 2) if engineering else (2152, 64)
    final_first, final_count = (3450, 2) if engineering else (3962, 512)
    selection_window = _window(audit, selection_first, selection_count, purpose)
    final_window = _window(audit, final_first, final_count, purpose)
    if not engineering:
        if selection_window["prior_fault_outcome_overlap"] or final_window["prior_fault_outcome_overlap"]:
            raise ValueError("confirmation window already exposed")
        if final_first + final_count != 4474:
            raise ValueError("reserved 4474–4889 must remain untouched")
    check_exposure(audit, final_first - 3048, final_count, purpose)
    source_receipts = _source_receipts(source, prior)
    _verify_windows(source, selection_window, final_window)
    root = Path(output).resolve()
    if root.exists():
        raise FileExistsError(root)
    arms = []
    for arm_index, probability in enumerate(levels):
        child = copy.deepcopy(prior)
        child["name"] = ("m97_engineering" if engineering else "m97_kws_multiber") + "_" + arm_id(probability)
        child["purpose"] = purpose
        child["joint_campaign"] = str(root / "campaign.json")
        child["source_m96"] = {"manifest": str(source), "sha256": sha256_file(source)}
        child["design"]["logical_ber"] = probability
        child["design"]["final_fields"] = 2 if engineering else 128
        child["design"]["selection_fields_per_seed"] = 1 if engineering else 8
        child["design"]["selection_replicates"] = 2 if engineering else 4
        child["samples"] = {"selection_offset": selection_first - 2024,
                            "selection_count": selection_count, "final_offset": final_first - 3048,
                            "final_count": final_count}
        child["exposure_inventory"] = audit
        child["exposure_audit"] = check_exposure(audit, final_first - 3048, final_count, purpose)
        child["pools"] = copy.deepcopy(pools)
        child["design"]["pool_seeds"] = [p["sampling_seed"] for p in pools]
        child["design"]["selection_seed_bases"] = []
        child["design"]["final_seeds"] = []
        for pool_index, pool in enumerate(child["pools"]):
            original = prior["pools"][pool_index]
            if digest_object(pool["candidates"]) != digest_object(original["candidates"]):
                raise ValueError("source candidate pool changed")
            doc = pool["selection_document"]
            seeds, final_seed, _ = _seed_plan(arm_index, pool_index, child["design"]["selection_fields_per_seed"])
            seeds = seeds[:child["design"]["selection_replicates"]]
            child["design"]["selection_seed_bases"].append(seeds[0])
            child["design"]["final_seeds"].append(final_seed)
            doc["experiment"]["name"] = child["name"] + "_" + pool["pool_id"] + "_selection"
            doc["samples"].update(model_selection_offset=child["samples"]["selection_offset"],
                                  model_selection_count=selection_count,
                                  evaluation_offset=child["samples"]["final_offset"],
                                  evaluation_count=final_count)
            doc["backgrounds"]["probabilities"] = [probability]
            doc["restricted_oracle"].update(calibration_background_count=child["design"]["selection_fields_per_seed"],
                                             calibration_replicate_seeds=seeds, calibration_seed=seeds[0] + 9_000_001)
            doc["paired_swap_evaluation"].update(seed=final_seed, background_count=child["design"]["final_fields"])
            doc["workflow"]["purpose"] = purpose
        child["operation_counts"] = operation_counts(6, child["design"]["selection_fields_per_seed"] * child["design"]["selection_replicates"],
                                                     selection_count, child["design"]["final_fields"], final_count, len(pools))
        arms.append(child)
    _check_seed_family(arms)
    root.mkdir(parents=True, exist_ok=False)
    arm_rows = []
    for child in arms:
        aid = arm_id(child["design"]["logical_ber"])
        directory = root / "arms" / aid
        (directory / "configs").mkdir(parents=True)
        manifest = directory / "campaign.json"
        _write_json(manifest, child)
        digest = sha256_file(manifest)
        for pool in child["pools"]:
            _write_bytes(directory / "configs" / (pool["pool_id"] + "_selection.toml"),
                         _toml(_bound_document(child, pool, manifest, digest)).encode())
        _write_json(directory / "checksums.json", {"schema_version": VERSION, "files": {
            str(p.relative_to(directory)): sha256_file(p) for p in sorted(directory.rglob("*")) if p.is_file()}})
        _write_bytes(directory / "FROZEN", (digest + "\n").encode())
        arm_rows.append({"id": aid, "logical_ber": child["design"]["logical_ber"],
                         "manifest": str(manifest), "manifest_sha256": digest,
                         "expected_fault_counts_per_field": {
                             "natural_model_proposed": 176128 * child["design"]["logical_ber"],
                             "natural_model_applied_after_22016_mask": 154112 * child["design"]["logical_ber"],
                             "selection_outside_mean": 154108 * child["design"]["logical_ber"],
                             "final_master_before_policy_mean": 154114 * child["design"]["logical_ber"],
                             "interpretation": "natural-model counts are expectations, not observed query counts; exact query events and suppression are saved by each member"},
                         "operation_counts": child["operation_counts"]})
    total = {key: sum(a["operation_counts"][key] for a in arm_rows) for key in (
        "selection_clip_inferences_total", "final_clip_inferences_upper_bound_total",
        "response_table_evaluations_total", "analytical_mask_integrations_total")}
    parent = {"schema_version": VERSION, "protocol": "joint-kws-multiber/v1", "purpose": purpose,
              "source_m96": {"manifest": str(source), "sha256": sha256_file(source),
                             "candidate_pools": [{"pool_id": p["pool_id"],
                                                  "candidate_order_sha256": p["candidate_order_sha256"],
                                                  "candidates_sha256": digest_object(p["candidates"])} for p in pools],
                             "verified_receipts": source_receipts},
              "checkpoint_sha256": prior["checkpoint_sha256"], "asset_lock_sha256": prior["asset_lock_sha256"],
              "source_mask_sha256": prior["source_mask_sha256"], "eligible_population": 176128,
              "protected_population": 22016, "candidate_budget": 2,
              "ber_grid": list(levels), "arms": arm_rows, "selection_window": selection_window,
              "final_window": final_window, "reserved_original_indices": [4474, 4889],
              "selection_fields_per_pool_ber": 2 if engineering else 32,
              "final_fields_per_pool_ber": 2 if engineering else 128,
              "statistics": {"confidence": .95, "bootstrap_seed": 970801, "bootstrap_replicates": 1000,
                             "primary_hypotheses": "two paired pool-level changes in selected-minus-exact-isolated CE: BER 0.001 minus 0.01 and BER 0.003 minus 0.01; two-sided Holm across two; simultaneous 97.5% t intervals",
                             "secondary_family": "nine within-BER two-sided pool-level mean CE contrasts, three policies by three BERs; Holm across all nine; simultaneous 99.4444% t intervals",
                             "known_anchor": "BER 0.01 M9.6 result is known; same-window arm is a comparison anchor, not untouched discovery",
                             "cross_ber": "paired pool vector; shared input-clips and independent fault-field streams",
                             "transfer": "optional descriptive BER-0.01 mask at lower BER; excluded from nine-test family"},
              "operation_counts_total": total,
              "exposure_inventory_sha256": digest_object(audit),
              "limitations": audit["limitations"]}
    _write_json(root / "campaign.json", parent)
    digest = sha256_file(root / "campaign.json")
    _write_json(root / "checksums.json", {"schema_version": VERSION, "files": {
        str(p.relative_to(root)): sha256_file(p) for p in sorted(root.rglob("*")) if p.is_file()}})
    _write_bytes(root / "FROZEN", (digest + "\n").encode())
    return {"campaign": str(root / "campaign.json"), "sha256": digest, "operation_counts_total": total}


def load_joint(path):
    path = Path(path).resolve()
    digest = sha256_file(path)
    if (path.parent / "FROZEN").read_text().strip() != digest:
        raise ValueError("joint campaign is not frozen")
    for relative, checksum in _read(path.parent / "checksums.json")["files"].items():
        if Path(relative).is_absolute() or ".." in Path(relative).parts or sha256_file(path.parent / relative) != checksum:
            raise ValueError("joint campaign inventory checksum mismatch")
    parent = _read(path)
    if parent.get("schema_version") != VERSION or parent.get("protocol") != "joint-kws-multiber/v1":
        raise ValueError("unsupported joint campaign")
    if [a["logical_ber"] for a in parent["arms"]] != parent["ber_grid"]:
        raise ValueError("BER arms differ from frozen grid")
    children = []
    for arm in parent["arms"]:
        member = Path(arm["manifest"])
        if member.parent.parent.parent != path.parent or sha256_file(member) != arm["manifest_sha256"]:
            raise ValueError("BER member manifest changed")
        child = load_campaign(member, arm["manifest_sha256"])
        children.append(child)
        if child.get("joint_campaign") != str(path) or child["design"]["logical_ber"] != arm["logical_ber"]:
            raise ValueError("BER member binding changed")
        for source_pool, pool in zip(parent["source_m96"]["candidate_pools"], child["pools"]):
            if pool["pool_id"] != source_pool["pool_id"] or digest_object(pool["candidates"]) != source_pool["candidates_sha256"]:
                raise ValueError("candidate pool differs across BER arms")
        if len(child["pools"]) != len(parent["source_m96"]["candidate_pools"]):
            raise ValueError("BER arm pool count differs")
    _check_seed_family(children)
    if parent["purpose"] == "confirmation":
        if parent["ber_grid"] != list(BER_GRID) or len(parent["source_m96"]["candidate_pools"]) != 12:
            raise ValueError("confirmation BER grid or pool count differs")
        if parent["selection_window"]["original_indices"] != list(range(2152, 2216)) or parent["final_window"]["original_indices"] != list(range(3962, 4474)):
            raise ValueError("confirmation windows differ from frozen protocol")
    for name in ("selection_window", "final_window"):
        window = parent[name]
        if digest_object(window["identity_records"]) != window["identity_records_sha256"]:
            raise ValueError("sample identity inventory changed")
    if sha256_file(parent["source_m96"]["manifest"]) != parent["source_m96"]["sha256"]:
        raise ValueError("M9.6 source manifest changed")
    source = load_campaign(parent["source_m96"]["manifest"], parent["source_m96"]["sha256"])
    for original, bound in zip(source["pools"], parent["source_m96"]["candidate_pools"]):
        if original["pool_id"] != bound["pool_id"] or digest_object(original["candidates"]) != bound["candidates_sha256"]:
            raise ValueError("M9.6 candidate pool provenance changed")
        for phase, receipt in parent["source_m96"]["verified_receipts"][original["pool_id"]].items():
            if sha256_file(_receipt(Path(parent["source_m96"]["manifest"]).parent,
                                    original["pool_id"], phase)) != receipt["receipt_sha256"]:
                raise ValueError("M9.6 finalized receipt changed")
    return parent


def verify_member(path, child, *, require_sealed=False):
    parent_path = Path(child["joint_campaign"]).resolve()
    parent = load_joint(parent_path)
    if not any(Path(row["manifest"]).resolve() == Path(path).resolve() and
               row["manifest_sha256"] == sha256_file(path) for row in parent["arms"]):
        raise ValueError("member is absent from frozen joint family")
    if require_sealed:
        marker = parent_path.parent / "SEALED"
        if not marker.is_file() or marker.read_text().strip() != digest_object(_sealed_registries(parent)):
            raise ValueError("all BER selections and final registries must be sealed before final execution")
    return parent


def require_all_selections(path):
    parent = load_joint(path)
    for arm in parent["arms"]:
        member = Path(arm["manifest"])
        child = load_campaign(member, arm["manifest_sha256"])
        for pool in child["pools"]:
            verified_receipt(_receipt(member.parent, pool["pool_id"], "selection"),
                             arm["manifest_sha256"], pool["pool_id"], "selection")
    return parent


def _sealed_registries(parent):
    result = {}
    for arm in parent["arms"]:
        directory = Path(arm["manifest"]).parent / "final-plan"
        registry = directory / "selections.json"
        digest = sha256_file(registry)
        if (directory / "SEALED").read_text().strip() != digest:
            raise ValueError("BER registry is not sealed")
        result[arm["id"]] = digest
    return result


def seal_joint(path):
    parent = require_all_selections(path)
    for arm in parent["arms"]:
        seal_selections(arm["manifest"])
    registries = _sealed_registries(parent)
    _write_bytes(Path(path).resolve().parent / "SEALED", (digest_object(registries) + "\n").encode())
    return {"sealed": registries, "sha256": digest_object(registries)}


def run_joint(path, phase, *, levels=None, pool_ids=None, progress=True):
    parent = load_joint(path)
    selected = set(parent["ber_grid"] if levels is None else levels)
    if not selected <= set(parent["ber_grid"]):
        raise ValueError("unprespecified BER")
    if phase == "final":
        verify_member(parent["arms"][0]["manifest"], load_campaign(parent["arms"][0]["manifest"]), require_sealed=True)
    results = {}
    for arm in parent["arms"]:
        if arm["logical_ber"] in selected:
            results[arm["id"]] = run_campaign(arm["manifest"], phase, pool_ids=pool_ids, progress=progress)
    return {"joint_campaign_sha256": sha256_file(path), "phase": phase, "arms": results}
