"""Prospective whole-eligible-INT8 reliability campaign, selection before final.

All production phases are foreground and write-once.  Freezing a plan performs
asset and input-identity validation but no faulted-model inference.  Final
fields cannot run before every ranking mask is sealed and hash-verified.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import zipfile

import numpy as np

from .manifests import canonical_json_bytes, digest_object, sha256_file
from .transformer_data import read_sst2_parquet, sst2_fault_partitions, sst2_records_digest
from .transformer_data import read_wikitext_parquet, wikitext_joined_text
from .w8a8_assets import load_ready_w8a8, TARGETS
from .w8a8_faults import W8A8FaultAdapter, W8A8Universe, BoundaryMask, MSBMask, bernoulli_field
from .w8a8_rank import PackedMask, write_top_masks, random_chunks, numerical_jump_chunks, gradient_score_chunks
from .w8a8_surrogate import build_surrogate_moments


ROOT = Path(__file__).resolve().parent.parent
SOURCE_FILES = ("netsurf/w8a8_assets.py", "netsurf/w8a8_faults.py", "netsurf/w8a8_rank.py",
                "netsurf/w8a8_surrogate.py", "netsurf/w8a8_campaign.py",
                "scripts/run_w8a8_campaign_once.sh", "scripts/run_distilbert_w8a8_once.sh",
                "scripts/run_pythia_w8a8_once.sh", "scripts/run_qwen_w8a8_once.sh")
METHODS = ("random", "msb_first", "numerical_bit_jump", "global_first_order",
           "per_example_absolute", "per_example_positive")
INTERIOR = (0.0625, 0.125, 0.1875, 0.25)
LAMBDAS = (4, 32, 256)
FIELDS = 24
SEED_BASE = 1203001
RESOURCES = {
    "distilbert_sst2_int8": {"foreground_hours": 10, "ram_gib": 8, "free_disk_gib": 5},
    "pythia160m_int8": {"foreground_hours": 8, "ram_gib": 12, "free_disk_gib": 10},
    "qwen3_0_6b_int8": {"foreground_hours": 24, "ram_gib": 64, "free_disk_gib": 40},
}


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as file:
        file.write(canonical_json_bytes(obj) + b"\n")


def _read(path: Path):
    return json.loads(path.read_text())


def _write_bound(path: Path, obj: dict) -> None:
    _write(path, {**obj, "payload_sha256": digest_object(obj)})


def _read_bound(path: Path) -> dict:
    item = _read(path)
    if digest_object({k: v for k, v in item.items() if k != "payload_sha256"}) != item.get("payload_sha256"):
        raise ValueError(f"immutable scientific member checksum changed: {path}")
    return item


def _cohort(asset, *, final: bool) -> dict:
    if asset.target == "distilbert_sst2_int8":
        root = asset.dataset_root / "sst2"
        train = read_sst2_parquet(root / "train-00000-of-00001.parquet", "train")
        validation = read_sst2_parquet(root / "validation-00000-of-00001.parquet", "validation")
        cohorts = sst2_fault_partitions(train, validation)
        rows = cohorts["evaluation"][:512] if final else cohorts["calibration"][:64]
        if len(rows) != (512 if final else 64):
            raise ValueError("frozen SST-2 cohort count changed")
        return {"role": "final" if final else "selection", "split": rows[0].split,
                "source_indices": [r.source_index for r in rows],
                "ordered_labels_and_text_sha256": sst2_records_digest(rows), "size": len(rows),
                "exposure_note": "previous cross-precision clean/fault studies disclosed; no virgin-cohort claim"}
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(asset.model_root), local_files_only=True,
                                               trust_remote_code=False, use_fast=True)
    split = "test" if final else "validation"
    rows = read_wikitext_parquet(asset.dataset_root / "wikitext-2-raw-v1" /
                                 f"{split}-00000-of-00001.parquet")
    text, text_sha = wikitext_joined_text(rows)
    ids = tuple(tokenizer(text, add_special_tokens=False)["input_ids"])
    start, end = (1024, 5121) if final else (1024, 9209)
    selected = ids[start:end]
    if len(selected) != end - start:
        raise ValueError("frozen WikiText-2 token window missing")
    return {"role": "final" if final else "selection", "split": split,
            "full_text_sha256": text_sha,
            "full_token_ids_sha256": hashlib.sha256(np.asarray(ids, dtype="<i4").tobytes()).hexdigest(),
            "input_start": start, "input_end_exclusive": end,
            "input_token_ids_sha256": hashlib.sha256(np.asarray(selected, dtype="<i4").tobytes()).hexdigest(),
            "scored_target_start": start + 1, "scored_targets": len(selected)-1,
            "context": 1024, "stride": 512, "bos": "none", "eos": "none",
            "exposure_note": "Kona clean full corpus and earlier narrow NetSuRF windows disclosed"}


def freeze(manifest: Path, target: str, destination: Path) -> dict:
    if Path(destination).exists():
        raise ValueError("frozen plan destination already exists")
    asset = load_ready_w8a8(manifest, target)
    universe = W8A8Universe(asset)
    selection, final = _cohort(asset, final=False), _cohort(asset, final=True)
    budgets = {str(x): int(universe.bits * x) for x in INTERIOR}
    if any(budget != universe.bits * fraction for fraction, budget in
           ((float(k), v) for k, v in budgets.items())):
        raise ValueError("requested whole-universe fractions are not exact cardinalities")
    fields = [{"lambda": lam, "ber": lam / universe.bits,
               "seeds": [SEED_BASE + 100000 * arm + i for i in range(FIELDS)]}
              for arm, lam in enumerate(LAMBDAS)]
    if len({s for arm in fields for s in arm["seeds"]}) != len(LAMBDAS) * FIELDS:
        raise ValueError("fault seeds collide")
    source = {name: sha256_file(ROOT / name) for name in SOURCE_FILES}
    plan = {"schema": "netsurf-w8a8-whole-qweight-campaign/v1", "target": target,
            "status": "FROZEN_SELECTION_NOT_RUN", "target_record_sha256": asset.record_sha256,
            "ready_manifest_sha256_at_freeze": asset.manifest_sha256,
            "quantization_manifest_sha256": sha256_file(asset.model_root / "quantization_manifest.json"),
            "model_commit": asset.record["model"]["commit"],
            "source_files_sha256": source, "kona_source_commit": "1aea52b",
            "eligible_universe": {"representation": "stored signed INT8 qweight codes only",
                "sha256": universe.digest, "bits": universe.bits, "codes": universe.codes,
                "modules": asset.modules, "tensor_manifest_digest": asset.quantization["tensors_digest"],
                "outside": asset.record["eligible_stored_int8_population"]["outside_population"]},
            "selection": selection, "final": final,
            "methods": METHODS,
            "ranking_seeds": {"random_hash_keys": 1203101},
            "random_policy": "independent seeded PCG64 uint64 priorities per global ordinal; exact top budget, global-ordinal tie break on vanishingly rare priority ties",
            "surrogate": {"model": "FP32 Linear weights = exact decoded stored INT8 codes times fixed channel scales",
                "gradient_target": "mean task cross-entropy on 64 SST-2 train records or mean token NLL on eight disjoint 1023-target validation windows",
                "per_example_unit": "SST-2 record" if target == "distilbert_sst2_int8" else "1023-target validation window",
                "not_derivative_of": "dynamic W8A8 torch._int_mm integer execution"},
            "budget_fractions": (0., *INTERIOR, 1.), "interior_budgets_bits": budgets,
            "fault_law": "independent Bernoulli over all eligible INT8 code bits, held for complete cohort trial",
            "fields": fields, "fields_per_ber": FIELDS,
            "nonfinite_penalty_nats": 30.,
            "primary_metric": "SST-2 mean cross-entropy" if target == "distilbert_sst2_int8" else "teacher-forced mean target-token NLL",
            "secondary_metric": "accuracy" if target == "distilbert_sst2_int8" else "perplexity",
            "statistics": {"unit": "independent whole-model fault field, paired across all methods and budgets",
                "comparison_family": "five nonrandom rankings versus random at four interior budgets and three BER arms = 60 two-sided comparisons per model",
                "adjustment": "Holm over 60 paired sign-flip p-values per model; 10000 draws at seed 1203901",
                "interval": "paired field bootstrap percentile 95%, 10000 draws at seed 1203902",
                "heavy_tail": "median, 10% trimmed mean, largest field, leave-largest-out"},
            "counts": {"selection_examples": 64 if target == "distilbert_sst2_int8" else 8,
                "selection_surrogate_backward_passes": 64 if target == "distilbert_sst2_int8" else 8,
                "final_fields": FIELDS * len(LAMBDAS),
                "policy_arms_per_field": 2 + len(METHODS)*len(INTERIOR),
                "final_model_requests": FIELDS*len(LAMBDAS)*(2+len(METHODS)*len(INTERIOR)),
                "clean_model_requests": 1,
                "scored_final_units_per_request": final["size"] if target == "distilbert_sst2_int8" else final["scored_targets"],
                "total_scored_final_unit_requests": FIELDS*len(LAMBDAS)*
                    (2+len(METHODS)*len(INTERIOR))*
                    (final["size"] if target == "distilbert_sst2_int8" else final["scored_targets"]),
                "mask_generation_score_key_visits": (5+3*4)*universe.bits,
                "mask_bytes_per_interior_budget": universe.bits//8,
                "stored_interior_mask_bytes": 5*len(INTERIOR)*(universe.bits//8),
                "surrogate_moment_bytes": 2*4*universe.codes},
            "resource_envelope": {**RESOURCES[target], "cpu_threads": 4,
                "basis": "bounded 8-sentence/511-target INT8 receipts scaled to 512 sentences/4096 targets; provisional, includes ranking/ZIP allowance",
                "contention_gate": "one model at a time; M11 workers must be absent"},
            "final_before_all_selection_sealed": "forbidden",
            "m12_handoff": "export immutable scientific payload and checksum receipt; M12 owns database ingest"}
    plan["plan_sha256"] = digest_object(plan)
    Path(destination).mkdir(parents=True, exist_ok=False)
    _write(Path(destination) / "plan.json", plan)
    _write(Path(destination) / "target_ready_record.json", asset.record)
    _write(Path(destination) / "quantization_manifest.json", asset.quantization)
    (Path(destination) / "FROZEN").write_text(sha256_file(Path(destination) / "plan.json") + "\n")
    return {"target": target, "plan_sha256": plan["plan_sha256"],
            "plan_file_sha256": sha256_file(Path(destination) / "plan.json"),
            "counts": plan["counts"]}


def validate(manifest: Path, plan_dir: Path) -> tuple[dict, object]:
    plan_path = Path(plan_dir) / "plan.json"
    plan = _read(plan_path)
    if (plan.get("schema") != "netsurf-w8a8-whole-qweight-campaign/v1" or
        sha256_file(plan_path) != (Path(plan_dir) / "FROZEN").read_text().strip() or
        digest_object({k: v for k, v in plan.items() if k != "plan_sha256"}) != plan.get("plan_sha256") or
        any(sha256_file(ROOT / name) != value for name, value in plan["source_files_sha256"].items())):
        raise ValueError("hash-bound W8A8 scientific plan or source changed")
    asset = load_ready_w8a8(manifest, plan["target"])
    universe = W8A8Universe(asset)
    if (asset.record_sha256 != plan["target_record_sha256"] or
        sha256_file(asset.model_root / "quantization_manifest.json") != plan["quantization_manifest_sha256"] or
        universe.digest != plan["eligible_universe"]["sha256"] or
        _read(Path(plan_dir) / "target_ready_record.json") != asset.record or
        _read(Path(plan_dir) / "quantization_manifest.json") != asset.quantization or
        _cohort(asset, final=False) != plan["selection"] or _cohort(asset, final=True) != plan["final"]):
        raise ValueError("target asset, eligible bits, source mapping or cohort changed")
    return plan, asset


def _budgets(plan) -> tuple[int, ...]:
    return tuple(plan["interior_budgets_bits"][str(x)] for x in INTERIOR)


def selection(manifest: Path, plan_dir: Path, run_dir: Path) -> dict:
    plan, asset = validate(manifest, plan_dir)
    root = Path(run_dir)
    if root.exists():
        if (root / "selection/seal.json").is_file():
            _verify_seal(plan, root)
            return {"status": "VERIFIED_SELECTION_REUSED", "target": plan["target"],
                    "selection_seal_sha256": sha256_file(root / "selection/seal.json")}
        raise ValueError("selection root exists without complete seal; inspect partial immutable state")
    adapter = W8A8FaultAdapter(asset)
    adapter.load()
    root.mkdir(parents=True, exist_ok=False)
    _write_bound(root / "run_identity.json", {"plan_sha256": plan["plan_sha256"],
                                              "target_record_sha256": asset.record_sha256})
    moments = build_surrogate_moments(adapter, root / "selection/moments")
    if asset.target == "distilbert_sst2_int8":
        identities_equal = moments["calibration"]["source_indices"] == plan["selection"]["source_indices"]
    else:
        identities_equal = moments["calibration"]["token_ids_sha256"] == plan["selection"]["input_token_ids_sha256"]
    if not identities_equal:
        raise ValueError("surrogate calibration identities differ from frozen plan")
    budgets = _budgets(plan)
    generated = {}
    definitions = {
        "random": lambda: random_chunks(adapter.universe.bits, 1203101),
        "numerical_bit_jump": lambda: numerical_jump_chunks(adapter.modules, adapter.universe),
        **{name: (lambda method=name: gradient_score_chunks(adapter.modules, adapter.universe,
                              root / "selection/moments", method)) for name in
           ("global_first_order", "per_example_absolute", "per_example_positive")},
    }
    for name, chunks in definitions.items():
        masks = write_top_masks(chunks, adapter.universe.bits, budgets,
                                root / "selection/masks" / name, name,
                                key_dtype=np.uint64 if name == "random" else np.uint32)
        generated[name] = {str(k): mask.sha256 for k, mask in masks.items()}
        print(f"{plan['target']} ranking {name} completed and hashed", flush=True)
    return {"status": "SELECTION_COMPLETE_UNSEALED", "target": plan["target"],
            "mask_sha256": generated, "moments_manifest_sha256": sha256_file(
                root / "selection/moments/manifest.json")}


def _verify_seal(plan, root: Path) -> dict:
    path = root / "selection/seal.json"
    if not path.is_file():
        raise ValueError("all ranking masks must be sealed before final evaluation")
    seal = _read_bound(path)
    if seal.get("plan_sha256") != plan["plan_sha256"] or seal.get("status") != "ALL_SELECTION_SEALED":
        raise ValueError("selection seal belongs to another scientific plan")
    if sha256_file(root / "selection/moments/manifest.json") != seal["moments_manifest_sha256"]:
        raise ValueError("surrogate moments changed after selection")
    for method in METHODS:
        if method == "msb_first":
            continue
        for budget in _budgets(plan):
            item = seal["masks"][method][str(budget)]
            path = root / item["path"]
            if sha256_file(path) != item["sha256"] or path.stat().st_size != plan["eligible_universe"]["bits"]//8:
                raise ValueError("sealed compact protection mask changed")
            mask = PackedMask(path, plan["eligible_universe"]["bits"], budget, item["sha256"])
            if mask.digest != item["digest"]:
                raise ValueError("sealed mask digest changed")
    return seal


def seal(manifest: Path, plan_dir: Path, run_dir: Path) -> dict:
    plan, _ = validate(manifest, plan_dir)
    root = Path(run_dir)
    if (root / "selection/seal.json").exists():
        return _verify_seal(plan, root)
    if _read_bound(root / "run_identity.json")["plan_sha256"] != plan["plan_sha256"]:
        raise ValueError("selection root belongs to a different plan")
    moments = root / "selection/moments/manifest.json"
    if not moments.is_file():
        raise ValueError("surrogate moments incomplete")
    mm = _read(moments)
    for item in mm["tensors"].values():
        for stem in ("mean_g", "mean_abs_g"):
            if sha256_file(moments.parent / item[stem]) != item[stem + "_sha256"]:
                raise ValueError("gradient moment file differs from selection receipt")
    masks = {}
    for method in METHODS:
        if method == "msb_first":
            continue
        folder = root / "selection/masks" / method
        meta = _read(folder / "manifest.json")
        if (meta.get("method") != method or meta.get("universe_bits") != plan["eligible_universe"]["bits"] or
            set(meta.get("masks", {})) != {str(b) for b in _budgets(plan)}):
            raise ValueError("ranked mask family incomplete")
        masks[method] = {}
        for budget in _budgets(plan):
            item = meta["masks"][str(budget)]
            path = folder / item["path"]
            if sha256_file(path) != item["sha256"] or path.stat().st_size != plan["eligible_universe"]["bits"]//8:
                raise ValueError("ranked mask bytes changed")
            masks[method][str(budget)] = {**item, "path": str(path.relative_to(root))}
    result = {"status": "ALL_SELECTION_SEALED", "plan_sha256": plan["plan_sha256"],
              "moments_manifest_sha256": sha256_file(moments), "masks": masks,
              "msb": {str(b): MSBMask(plan["eligible_universe"]["bits"], b).digest
                      for b in _budgets(plan)}}
    _write_bound(root / "selection/seal.json", result)
    return _verify_seal(plan, root)


def _policy_masks(plan, root: Path, seal: dict):
    n = plan["eligible_universe"]["bits"]
    masks = {"unprotected": BoundaryMask(n, False), "full_protection": BoundaryMask(n, True)}
    for method in METHODS:
        for budget in _budgets(plan):
            if method == "msb_first":
                mask = MSBMask(n, budget)
                if mask.digest != seal["msb"][str(budget)]:
                    raise ValueError("MSB analytical mask changed")
            else:
                item = seal["masks"][method][str(budget)]
                mask = PackedMask(root / item["path"], n, budget, item["sha256"])
            masks[f"{method}_b{budget}"] = mask
    if len(masks) != 26:
        raise ValueError("whole-universe policy family incomplete")
    return masks


def _evaluation(adapter, plan):
    cohort = plan["final"]
    if plan["target"] == "distilbert_sst2_int8":
        rows = adapter.sst2_cohorts()["evaluation"][:512]
        if [r.source_index for r in rows] != cohort["source_indices"] or \
           sst2_records_digest(rows) != cohort["ordered_labels_and_text_sha256"]:
            raise ValueError("frozen SST-2 final identities changed")
        return lambda: adapter.evaluate_sst2(rows)
    ids = adapter.token_sequence("test")
    selected = ids[cohort["input_start"]:cohort["input_end_exclusive"]]
    if hashlib.sha256(np.asarray(selected, dtype="<i4").tobytes()).hexdigest() != cohort["input_token_ids_sha256"]:
        raise ValueError("frozen WikiText-2 final token identities changed")
    return lambda: adapter.evaluate_lm(selected)


def _risk_record(risk) -> dict:
    return {"metrics": dict(risk.metrics), "failed_units": risk.failed_units,
            "total_units": risk.total_units, "logit_statistics_sha256": digest_object(risk.logits),
            "failure_kinds_sha256": digest_object(risk.failure_kinds)}


def final(manifest: Path, plan_dir: Path, run_dir: Path) -> dict:
    plan, asset = validate(manifest, plan_dir)
    root = Path(run_dir)
    sealed = _verify_seal(plan, root)
    masks = _policy_masks(plan, root, sealed)
    adapter = W8A8FaultAdapter(asset)
    adapter.load()
    evaluate = _evaluation(adapter, plan)
    clean_path = root / "final/clean.json"
    if clean_path.exists():
        clean = _read_bound(clean_path)
        if clean.get("plan_sha256") != plan["plan_sha256"]:
            raise ValueError("clean reference belongs to another plan")
    else:
        _write_bound(clean_path, {"plan_sha256": plan["plan_sha256"], "risk": _risk_record(evaluate())})
    completed = 0
    for arm, specification in enumerate(plan["fields"]):
        for field_index, seed in enumerate(specification["seeds"]):
            field = bernoulli_field(adapter.universe.bits, specification["ber"], seed,
                                    max_events=4096)
            identity = {"plan_sha256": plan["plan_sha256"], "selection_seal_sha256": sha256_file(
                root / "selection/seal.json"), "arm": arm, "field": field_index,
                "lambda": specification["lambda"], "ber": specification["ber"],
                "seed": seed, "field_sha256": field.sha256,
                "proposed_ordinals": list(field.ordinals)}
            path = root / "final" / f"arm_{arm:02d}" / f"field_{field_index:03d}.json"
            intent = path.with_suffix(".intent.json")
            if path.exists():
                record = _read_bound(path)
                if any(_read_bound(intent).get(k) != v for k, v in identity.items()) or \
                   any(record.get(k) != v for k, v in identity.items()) or \
                   set(record.get("policies", {})) != set(masks):
                    raise ValueError("completed field differs from frozen schedule or policy family")
                completed += 1
                continue
            if intent.exists():
                raise ValueError(f"partial final field retained; immutable recovery required: {intent}")
            _write_bound(intent, identity)
            policies = {}
            for name, mask in masks.items():
                with adapter.injected(field, mask) as events:
                    risk = evaluate()
                if events["proposed"] != events["applied"] + events["suppressed"]:
                    raise ValueError("fault-event accounting changed")
                policies[name] = {"mask_sha256": mask.digest, "budget_bits": mask.budget,
                                  "expected_applied_faults": (1-mask.budget/adapter.universe.bits)*specification["lambda"],
                                  "events": events, "risk": _risk_record(risk)}
            _write_bound(path, {**identity, "policies": policies})
            completed += 1
            print(f"{plan['target']} BER {arm+1}/3 field {field_index+1}/{FIELDS} sealed", flush=True)
    return {"target": plan["target"], "completed_fields": completed,
            "expected_fields": len(plan["fields"])*FIELDS}


def aggregate(manifest: Path, plan_dir: Path, run_dir: Path) -> dict:
    """Read-only field-level analysis; never calls a model or changes masks."""
    plan, _ = validate(manifest, plan_dir)
    root = Path(run_dir)
    seal_record = _verify_seal(plan, root)
    masks = _policy_masks(plan, root, seal_record)
    clean = _read_bound(root / "final/clean.json")
    if clean.get("plan_sha256") != plan["plan_sha256"]:
        raise ValueError("clean task reference differs from scientific plan")
    rows = []
    n = plan["eligible_universe"]["bits"]
    for arm, specification in enumerate(plan["fields"]):
        for field_index, seed in enumerate(specification["seeds"]):
            path = root / "final" / f"arm_{arm:02d}" / f"field_{field_index:03d}.json"
            if not path.is_file():
                raise ValueError(f"final field missing: {path}")
            record = _read_bound(path)
            intent = _read_bound(path.with_suffix(".intent.json"))
            field = bernoulli_field(n, specification["ber"], seed, max_events=4096)
            identity = {"plan_sha256": plan["plan_sha256"], "selection_seal_sha256": sha256_file(
                root / "selection/seal.json"), "arm": arm, "field": field_index,
                "lambda": specification["lambda"], "ber": specification["ber"],
                "seed": seed, "field_sha256": field.sha256,
                "proposed_ordinals": list(field.ordinals)}
            if any(intent.get(k) != v for k, v in identity.items()) or \
               any(record.get(k) != v for k, v in identity.items()) or \
               set(record.get("policies", {})) != set(masks):
                raise ValueError("final field identity, schedule or policy family changed")
            row = {"arm": arm, "field": field_index, "seed": seed,
                   "lambda": specification["lambda"], "ber": specification["ber"], "policies": {}}
            for name, mask in masks.items():
                item = record["policies"][name]
                expected_applied = sum(not mask.contains(x) for x in field.ordinals)
                events = item["events"]
                risk = item["risk"]
                if (item["mask_sha256"] != mask.digest or item["budget_bits"] != mask.budget or
                    item["expected_applied_faults"] != (1-mask.budget/n)*specification["lambda"] or
                    events["field_sha256"] != field.sha256 or events["mask_sha256"] != mask.digest or
                    (events["proposed"], events["applied"], events["suppressed"]) !=
                    (len(field.ordinals), expected_applied, len(field.ordinals)-expected_applied) or
                    risk["total_units"] != plan["counts"]["scored_final_units_per_request"] or
                    not 0 <= risk["failed_units"] <= risk["total_units"] or
                    not math.isfinite(risk["metrics"]["mean_nll"])):
                    raise ValueError("saved field event, mask or risk accounting changed")
                row["policies"][name] = {"mean_nll": risk["metrics"]["mean_nll"],
                                         "failed_units": risk["failed_units"],
                                         "accuracy": risk["metrics"].get("accuracy"),
                                         "perplexity": risk["metrics"].get("perplexity")}
            if row["policies"]["full_protection"]["mean_nll"] != clean["risk"]["metrics"]["mean_nll"] or \
               record["policies"]["full_protection"]["risk"]["logit_statistics_sha256"] != \
               clean["risk"]["logit_statistics_sha256"]:
                raise ValueError("full-protection control differs from clean reference")
            rows.append(row)
    comparisons = []
    for arm in range(len(LAMBDAS)):
        field_rows = [row for row in rows if row["arm"] == arm]
        for budget in _budgets(plan):
            random_key = f"random_b{budget}"
            for method in METHODS:
                if method == "random":
                    continue
                key = f"{method}_b{budget}"
                diffs = np.array([r["policies"][key]["mean_nll"] -
                                  r["policies"][random_key]["mean_nll"] for r in field_rows], dtype=np.float64)
                rng_boot = np.random.default_rng(1203902 + len(comparisons))
                rng_sign = np.random.default_rng(1203901 + len(comparisons))
                bootstrap = diffs[rng_boot.integers(0, len(diffs), size=(10000, len(diffs)))].mean(axis=1)
                permutations = (diffs * rng_sign.choice((-1., 1.), size=(10000, len(diffs)))).mean(axis=1)
                observed = float(diffs.mean())
                p = (1 + int(np.sum(np.abs(permutations) >= abs(observed)))) / 10001
                ordered = np.sort(diffs)
                trim = int(.1*len(diffs))
                leave = float(np.delete(diffs, np.argmax(np.abs(diffs))).mean())
                comparisons.append({"arm": arm, "lambda": plan["fields"][arm]["lambda"],
                    "ber": plan["fields"][arm]["ber"], "budget_bits": budget,
                    "method_minus_random": method, "mean_nll_difference": observed,
                    "ci95": [float(np.quantile(bootstrap, .025)), float(np.quantile(bootstrap, .975))],
                    "p_signflip": p, "favorable_fields": int(np.sum(diffs < 0)),
                    "adverse_fields": int(np.sum(diffs > 0)), "tied_fields": int(np.sum(diffs == 0)),
                    "median": float(np.median(diffs)),
                    "trimmed_10pct": float((ordered[trim:-trim] if trim else ordered).mean()),
                    "max_absolute_field": float(np.max(np.abs(diffs))),
                    "leave_largest_absolute_out": leave,
                    "failed_unit_difference": int(sum(r["policies"][key]["failed_units"]-
                        r["policies"][random_key]["failed_units"] for r in field_rows))})
    if len(comparisons) != 60:
        raise ValueError("prespecified 60-comparison family incomplete")
    order = sorted(range(len(comparisons)), key=lambda i: comparisons[i]["p_signflip"])
    adjusted = 0.
    for position, index in enumerate(order):
        adjusted = max(adjusted, min(1., comparisons[index]["p_signflip"]*(len(order)-position)))
        comparisons[index]["p_holm_60"] = adjusted
    report = {"schema": "netsurf-w8a8-field-analysis/v1", "plan_sha256": plan["plan_sha256"],
              "selection_seal_sha256": sha256_file(root / "selection/seal.json"),
              "independent_field_count": len(rows), "clean": clean["risk"],
              "field_rows": rows, "comparisons": comparisons,
              "limits": ["conditional on frozen model and input cohort", "fault fields are independent units",
                         "token predictions and policies within a field are paired, not independent replicates",
                         "surrogate rankings are not integer-execution derivatives",
                         "same expected fault count is not same physical reliability"]}
    analysis_path = root / "analysis.json"
    if analysis_path.exists():
        if {k: v for k, v in _read_bound(analysis_path).items() if k != "payload_sha256"} != report:
            raise ValueError("existing analysis differs from deterministic reconstruction")
    else:
        _write_bound(analysis_path, report)
    return {"status": "ANALYSIS_COMPLETE", "target": plan["target"],
            "fields": len(rows), "comparisons": len(comparisons),
            "analysis_sha256": sha256_file(analysis_path)}


def export(manifest: Path, plan_dir: Path, run_dir: Path, *, host: str = "kona") -> dict:
    plan, _ = validate(manifest, plan_dir)
    root = Path(run_dir)
    summary = aggregate(manifest, plan_dir, root)
    member_hashes = {str(p.relative_to(root)): sha256_file(p) for p in sorted((root / "final").rglob("*.json"))}
    payload = {"schema": "netsurf-w8a8-scientific-payload/v1", "plan_sha256": plan["plan_sha256"],
               "selection_seal_sha256": sha256_file(root / "selection/seal.json"),
               "analysis_sha256": sha256_file(root / "analysis.json"),
               "final_member_hashes": member_hashes,
               "field_count": plan["counts"]["final_fields"],
               "policy_arms_per_field": plan["counts"]["policy_arms_per_field"]}
    payload_path = root / "scientific_payload.json"
    if payload_path.exists():
        if _read_bound(payload_path) != {**payload, "payload_sha256": digest_object(payload)}:
            raise ValueError("existing scientific payload differs from immutable final evidence")
    else:
        _write_bound(payload_path, payload)
    destination = root.parent / (root.name + "_export")
    if destination.exists():
        zip_path = destination / "complete.zip"
        check_path = destination / "checksums.json"
        if not (zip_path.is_file() and check_path.is_file() and (destination / "FINALIZED").is_file()):
            raise ValueError("partial write-once evidence export retained")
        prior = _read(check_path)
        zip_hash = sha256_file(zip_path)
        if (prior != {name: sha256_file(root / name[4:]) if name.startswith("run/") else
                      sha256_file(Path(plan_dir) / name[5:]) if name.startswith("plan/") else
                      sha256_file(ROOT / name[7:]) for name in prior} or
            (destination / "FINALIZED").read_text().strip() != digest_object({
                "plan_sha256": plan["plan_sha256"], "zip_sha256": zip_hash, "members": prior})):
            raise ValueError("existing exported evidence differs from scientific source")
        with zipfile.ZipFile(zip_path) as archive:
            if archive.testzip() is not None:
                raise ValueError("existing complete evidence ZIP failed CRC")
        return {**summary, "status": "VERIFIED_FINALIZED_REUSED", "target": plan["target"],
                "scientific_payload_sha256": digest_object(payload), "zip_sha256": zip_hash,
                "archive": str(zip_path), "scp": (destination / "mac_scp_command.txt").read_text().strip()}
    destination.mkdir(parents=True, exist_ok=False)
    checks = {}
    with zipfile.ZipFile(destination / "complete.zip", "x", allowZip64=True) as archive:
        files = [(p, "run/" + str(p.relative_to(root))) for p in sorted(root.rglob("*")) if p.is_file()]
        files += [(p, "plan/" + str(p.relative_to(plan_dir))) for p in sorted(Path(plan_dir).rglob("*")) if p.is_file()]
        files += [(ROOT / name, "source/" + name) for name in SOURCE_FILES]
        for path, name in files:
            checks[name] = sha256_file(path)
            archive.write(path, name, compress_type=zipfile.ZIP_STORED if path.suffix == ".bits"
                          else zipfile.ZIP_DEFLATED)
        archive.writestr("evidence_checksums.json", canonical_json_bytes(checks) + b"\n")
    with zipfile.ZipFile(destination / "complete.zip") as archive:
        if archive.testzip() is not None:
            raise ValueError("complete evidence ZIP failed CRC verification")
    _write(destination / "checksums.json", checks)
    zip_hash = sha256_file(destination / "complete.zip")
    (destination / "FINALIZED").write_text(digest_object({"plan_sha256": plan["plan_sha256"],
                                                          "zip_sha256": zip_hash, "members": checks}) + "\n")
    command = f"scp {host}:{(destination / 'complete.zip').resolve()} ~/Downloads/{plan['target']}_w8a8_complete.zip"
    (destination / "mac_scp_command.txt").write_text(command + "\n")
    return {**summary, "status": "FINALIZED", "target": plan["target"],
            "scientific_payload_sha256": digest_object(payload), "zip_sha256": zip_hash,
            "archive": str(destination / "complete.zip"), "scp": command}


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "validate", "selection", "seal", "final", "aggregate", "export"))
    parser.add_argument("ready_manifest", type=Path)
    parser.add_argument("plan_or_target")
    parser.add_argument("destination", type=Path, nargs="?")
    args = parser.parse_args(argv)
    if args.phase == "freeze":
        if args.plan_or_target not in TARGETS or args.destination is None:
            parser.error("freeze needs TARGET and unused PLAN_DIR")
        result = freeze(args.ready_manifest, args.plan_or_target, args.destination)
    else:
        plan_dir = Path(args.plan_or_target)
        if args.phase == "validate":
            plan, _ = validate(args.ready_manifest, plan_dir)
            result = {"status": "VALIDATED", "target": plan["target"], "plan_sha256": plan["plan_sha256"],
                      "counts": plan["counts"]}
        else:
            if args.destination is None:
                parser.error(f"{args.phase} needs PLAN_DIR and RUN_DIR")
            action = {"selection": selection, "seal": seal, "final": final,
                      "aggregate": aggregate, "export": export}[args.phase]
            result = action(args.ready_manifest, plan_dir, args.destination)
    print(json.dumps(result, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
