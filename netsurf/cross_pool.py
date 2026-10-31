"""Prespecified probability-sampled campaigns around the existing exact selector.

No alternative selection, inference or fault-generation implementation lives here.
Campaigns freeze all pools/plans, then seal all selection artifacts before final use.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import json
import math
from pathlib import Path
import re
import tomllib

import numpy as np

from .campaign_exposure import KWS_SHA, audit_exposure, check_exposure
from .identities import BitIdentity
from .manifests import digest_object, sha256_file
from .result_bundle import _write_bytes, _write_json

VERSION = "1.0.0"
PROVENANCE = "independent_s5_status_stratified_uniform_without_replacement/v1"


def _read(path):
    return json.loads(Path(path).read_text())


def _strict(document, keys, name):
    if not isinstance(document, dict) or set(document) != set(keys):
        raise ValueError(f"{name} requires exactly {sorted(keys)}")


def _integer(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"expected integer >= {minimum}, got {value!r}")
    return value


def sample_pools(universe, protected, seeds, candidate_count):
    """Two independent uniform-without-replacement draws per published pool seed."""
    if candidate_count not in {6, 8} or len(set(universe)) != len(universe):
        raise ValueError("campaign requires unique universe and K=6 or K=8")
    protected = frozenset(protected)
    if not protected <= set(universe) or len(set(seeds)) != len(seeds):
        raise ValueError("ineligible source mask or repeated pool seed")
    strata = (tuple(sorted(protected)), tuple(sorted(set(universe) - protected)))
    quotas = (2, candidate_count - 2)
    results = []
    for number, seed in enumerate(seeds):
        _integer(seed)
        rng = np.random.Generator(np.random.PCG64(seed))
        chosen = []
        for population, quota in zip(strata, quotas):
            if quota > len(population):
                raise ValueError("pool quota exceeds source stratum")
            chosen.extend(population[int(i)] for i in rng.choice(len(population), quota, replace=False))
        candidates = tuple(sorted(chosen))
        rows = [{**asdict(bit), "identity": bit.key(), "source_protected": bit in protected,
                 "stratum": "protected" if bit in protected else "unprotected",
                 "inclusion_probability": (2 / len(strata[0]) if bit in protected
                                           else (candidate_count - 2) / len(strata[1]))}
                for bit in candidates]
        results.append({"pool_id": f"pool_{number:03d}", "sampling_seed": seed,
                        "candidates": rows, "candidate_order_sha256": digest_object([b.key() for b in candidates]),
                        "original_mask_indices": [i for i, b in enumerate(candidates) if b in protected]})
    return results


def operation_counts(k, fields, selection_inputs, final_fields, final_inputs, pools):
    selection = k + 1 + fields * 2**k
    final = final_fields * 3 * 2 * 2
    return {"pools": pools, "candidate_states_per_field": 2**k,
            "eligible_masks_per_pool": math.comb(k, 2),
            "selection_fields_per_pool": fields, "final_fields_per_pool": final_fields,
            "selection_state_requests_per_pool": selection,
            "selection_clip_inferences_total": pools * selection * selection_inputs,
            "final_state_requests_upper_bound_per_pool": final,
            "final_clip_inferences_upper_bound_total": pools * final * final_inputs,
            "response_table_evaluations_total": pools * fields * 2**k,
            "analytical_mask_integrations_total": pools * fields * math.comb(k, 2),
            "note": "Final upper bound assumes two swaps for each of three contrasts. Frozen paths and exact schedule reuse reduce actual work; 16 reference clips per adapter load are additional."}


def _toml(document):
    """Serialize this deliberately flat tables/array-of-tables configuration."""
    lines = []
    def scalar(v):
        return json.dumps(v, ensure_ascii=True)
    for key, value in document.items():
        if not isinstance(value, dict) and not (isinstance(value, list) and value and isinstance(value[0], dict)):
            lines.append(f"{key} = {scalar(value)}")
    for key, value in document.items():
        if isinstance(value, dict):
            lines += ["", f"[{key}]"] + [f"{k} = {scalar(v)}" for k, v in value.items()]
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            for row in value:
                lines += ["", f"[[{key}]]"] + [f"{k} = {scalar(v)}" for k, v in row.items()]
    return "\n".join(lines) + "\n"


def freeze_campaign(configuration, output):
    """Verify source assets, audit exposure, freeze all plans; no model inference."""
    from scipy.stats import t  # Required statistical backend; never install/fallback.
    source = Path(configuration).resolve()
    request = tomllib.loads(source.read_text())
    _strict(request, {"schema_version", "name", "purpose", "benchmark_template", "design", "samples", "statistics"}, "campaign")
    if request["schema_version"] != VERSION or request["purpose"] not in {"engineering", "confirmation"}:
        raise ValueError("unsupported campaign schema/purpose")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", request["name"]):
        raise ValueError("campaign name must be filesystem-safe")
    design, samples, stats = request["design"], request["samples"], request["statistics"]
    _strict(design, {"candidate_count", "pool_seeds", "selection_seed_bases", "selection_replicates", "selection_fields_per_seed", "final_seeds", "final_fields", "logical_ber"}, "design")
    _strict(samples, {"selection_offset", "selection_count", "final_offset", "final_count"}, "samples")
    _strict(stats, {"bootstrap_seed", "bootstrap_replicates", "confidence"}, "statistics")
    for key in samples:
        _integer(samples[key], 1 if key.endswith("count") else 0)
    for key in ("selection_replicates", "selection_fields_per_seed", "final_fields"):
        _integer(design[key], 1)
    if design["final_fields"] < 2 or samples["selection_offset"] + samples["selection_count"] > 1024:
        raise ValueError("need >=2 final fields and valid model-selection window")
    if stats["confidence"] != .95 or design["logical_ber"] != .01:
        raise ValueError("v1 prespecifies 95% confidence, CE minimization and BER 0.01")
    _integer(stats["bootstrap_seed"]); _integer(stats["bootstrap_replicates"], 100)
    n = len(design["pool_seeds"])
    if n < 2 or len(design["selection_seed_bases"]) != n or len(design["final_seeds"]) != n:
        raise ValueError("at least two pools and one selection base/final seed per pool required")
    if design["candidate_count"] not in {6, 8}:
        raise ValueError("campaign requires K=6 or K=8")
    all_seeds = []
    for i in range(n):
        for base in range(_integer(design["selection_seed_bases"][i]), design["selection_seed_bases"][i] + design["selection_replicates"]):
            all_seeds += [base + j * 1_000_003 for j in range(design["selection_fields_per_seed"])]
        all_seeds += [_integer(design["final_seeds"][i]) + j * 1_000_003 for j in range(design["final_fields"])]
    if len(set(all_seeds)) != len(all_seeds) or set(all_seeds) & set(design["pool_seeds"]):
        raise ValueError("pool/selection/final RNG seeds must be distinct, including derived field seeds")
    template = (source.parent / request["benchmark_template"]).resolve()
    base = tomllib.loads(template.read_text())
    from .asset_lock import VerifiedAssetLock
    from .tflite_storage import TFLiteModelStorage
    from .protection_context import ProtectionContextConfiguration, verify_protection_context
    from .conditional_config import load_conditional
    # Verify backend and scientific template semantics before generating any pool.
    template_config = load_conditional(template, check_inputs=False)
    if template_config.benchmark.adapter != "mlperf_tiny_kws_int8_v1":
        raise ValueError("campaign v1 is KWS-only")
    lock = VerifiedAssetLock.load(template_config.benchmark.asset_lock, expected_benchmark_id="mlperf-tiny/kws")
    storage = TFLiteModelStorage.from_path(template_config.benchmark.checkpoint, expected_sha256=KWS_SHA, eligible_roles=("kernel",))
    universe = storage.bit_identities()
    context = template_config.protection_context
    artifact = _read(context.source_bundle / context.artifact)
    from .protection_context import _selected, _identity
    provisional = tuple(_identity(b) for b in _selected(artifact, context.mask_key))
    # The verifier needs a mixed pool; this is just a deterministic metadata check,
    # not an experimental candidate draw. All source integrity checks precede RNG use.
    provisional_set = set(provisional)
    probe = (provisional[0], next(b for b in universe if b not in provisional_set))
    verified = verify_protection_context(context, universe, probe, expected_checkpoint_sha256=KWS_SHA)
    if (len(universe), len(verified.original)) != (176128, 22016):
        raise ValueError("pinned KWS S5 population/cardinality differs")
    source_payload = context.source_bundle / "scientific_payload.json"
    checks = _read(context.source_bundle / "checksums.json")
    if sha256_file(source_payload) != checks["files"]["scientific_payload.json"] or digest_object(_read(source_payload)) != context.scientific_payload_sha256:
        raise ValueError("source scientific payload corrupted")
    audit = audit_exposure()
    exposure = check_exposure(audit, samples["final_offset"], samples["final_count"], request["purpose"])
    pools = sample_pools(universe, verified.original, design["pool_seeds"], design["candidate_count"])
    values = storage.values()
    descriptors = {d.parameter: d for d in storage.descriptors}
    for pool in pools:
        for row in pool["candidates"]:
            desc = descriptors[row["parameter"]]
            channel = np.unravel_index(row["flat_index"], desc.shape)[desc.quantization.quantized_dimension]
            q = desc.quantization
            code = int(values[row["parameter"]].flat[row["flat_index"]])
            row.update({"tensor_index": desc.tensor_index, "layer": desc.original_name,
                        "stored_signed_code": code, "encoded_unsigned_word": code & 255,
                        "scale": q.scales[channel if len(q.scales) > 1 else 0],
                        "zero_point": q.zero_points[channel if len(q.zero_points) > 1 else 0],
                        "storage": "INT8 two's-complement kernel; bit 0 LSB"})
    overlap = [{"left": a["pool_id"], "right": b["pool_id"],
                "shared_identities": sorted(set(r["identity"] for r in a["candidates"]) & set(r["identity"] for r in b["candidates"]))}
               for i, a in enumerate(pools) for b in pools[i + 1:]]
    base["schema_version"] = "6.6.0"
    base["scientific_protocol"] = "cross-pool-frozen-protection/v7"
    base["samples"].update(model_selection_offset=samples["selection_offset"], model_selection_count=samples["selection_count"],
                           evaluation_offset=samples["final_offset"], evaluation_count=samples["final_count"])
    base["backgrounds"]["probabilities"] = [.01]
    base["execution"]["confidence"] = .95
    base["restricted_oracle"].update(budget=2, calibration_background_count=design["selection_fields_per_seed"])
    base["paired_swap_evaluation"].update(enabled=False, background_count=design["final_fields"])
    base["workflow"] = {"mode": "selection_only", "purpose": request["purpose"], "candidate_provenance": PROVENANCE, "record_per_input": True}
    for i, pool in enumerate(pools):
        doc = json.loads(json.dumps(base))
        doc["experiment"] = {"name": request["name"] + "_" + pool["pool_id"] + "_selection",
                             "description": "Independent status-stratified pool; exact model-selection-only enumeration.",
                             "objective": "Frozen cross-pool expected-CE comparisons; no radiation interpretation."}
        doc["candidates"] = [{k: r[k] for k in ("parameter", "flat_index", "bit_index")} for r in pool["candidates"]]
        doc["restricted_oracle"]["calibration_replicate_seeds"] = list(range(design["selection_seed_bases"][i], design["selection_seed_bases"][i] + design["selection_replicates"]))
        doc["paired_swap_evaluation"]["seed"] = design["final_seeds"][i]
        doc["output"]["root"] = ""
        pool["selection_document"] = doc
    record = {"schema_version": VERSION, "protocol": "cross-pool-frozen-protection/v1", "name": request["name"],
              "purpose": request["purpose"], "design": design, "samples": samples, "statistics": stats,
              "checkpoint_sha256": KWS_SHA, "asset_lock_sha256": sha256_file(lock.path),
              "universe_sha256": digest_object([b.key() for b in universe]),
              "eligible_population": 176128, "protected_population": 22016, "unprotected_population": 154112,
              "source_mask_sha256": verified.record["original_mask_sha256"],
              "source_context": context.to_dict(), "tensor_population": [d.to_dict() for d in storage.descriptors],
              "rng": {"algorithm": "NumPy Generator(PCG64); protected draw then unprotected draw; choice replace=False; sorted BitIdentity strata and final order", "numpy_version": np.__version__},
              "sampling_design": PROVENANCE, "pools": pools, "pool_overlap": overlap,
              "exposure_inventory": audit, "exposure_audit": exposure,
              "operation_counts": operation_counts(design["candidate_count"], design["selection_fields_per_seed"] * design["selection_replicates"], samples["selection_count"], design["final_fields"], samples["final_count"], n),
              "statistical_plan": {"primary": "restricted_selected_vs_original_s5", "secondary": ["restricted_selected_vs_exact_isolated", "exact_isolated_vs_original_s5"],
                  "principal_unit": "probability-sampled candidate pool; inputs shared; fields independent within pool",
                  "primary_ci": "Student-t on equally weighted pool means conditional on shared fixed inputs and selection data",
                  "secondary_multiplicity": "Holm across two cross-pool tests; Bonferroni 97.5% secondary CIs",
                  "diagnostics": "pool bootstrap; field bootstrap conditional on pools/inputs; shared-input bootstrap conditional on pools/fields; crossed pool/input/field bootstrap as an approximate sensitivity diagnostic",
                  "no_adaptation": "all pools retained; no response-dependent resampling, stopping, field-count change or candidate screening"}}
    destination = Path(output).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    _write_json(destination / "campaign.json", record)
    digest = sha256_file(destination / "campaign.json")
    (destination / "configs").mkdir()
    for pool in pools:
        doc = _bound_document(record, pool, destination / "campaign.json", digest)
        _write_bytes(destination / "configs" / (pool["pool_id"] + "_selection.toml"), _toml(doc).encode())
    _write_json(destination / "checksums.json", {"schema_version": VERSION, "files": {
        str(p.relative_to(destination)): sha256_file(p) for p in sorted(destination.rglob("*")) if p.is_file()}})
    _write_bytes(destination / "FROZEN", (digest + "\n").encode())
    return {"campaign": str(destination / "campaign.json"), "sha256": digest, "operation_counts": record["operation_counts"]}


def _bound_document(record, pool, manifest, digest, *, final_record=None):
    doc = json.loads(json.dumps(pool["selection_document"]))
    doc["output"]["root"] = str(Path(manifest).parent / "runs")
    doc["workflow"].update(campaign_manifest=str(manifest), campaign_sha256=digest, pool_id=pool["pool_id"])
    if final_record is not None:
        doc["experiment"]["name"] = doc["experiment"]["name"].removesuffix("_selection") + "_final"
        doc["workflow"].update(final_record, mode="final_only")
        doc["paired_swap_evaluation"]["enabled"] = True
    return doc


def load_campaign(path, expected=None):
    path = Path(path).resolve()
    digest = sha256_file(path)
    if expected is not None and digest != expected:
        raise ValueError("campaign manifest hash mismatch")
    if (path.parent / "FROZEN").read_text().strip() != digest:
        raise ValueError("campaign is not finalized/frozen")
    for relative, checksum in _read(path.parent / "checksums.json")["files"].items():
        if Path(relative).is_absolute() or ".." in Path(relative).parts or sha256_file(path.parent / relative) != checksum:
            raise ValueError("campaign inventory checksum mismatch")
    record = _read(path)
    if record.get("schema_version") != VERSION or record.get("protocol") != "cross-pool-frozen-protection/v1":
        raise ValueError("unsupported campaign schema/protocol")
    return record


def validate_pool_workflow(raw, samples, backgrounds, oracle, paired, context, source):
    from .selection_preflight import validate_workflow
    required = {"campaign_manifest", "campaign_sha256", "pool_id", "record_per_input"}
    extra = {"final_registry", "final_registry_sha256"}
    if not required <= raw.keys() or raw.get("candidate_provenance") != PROVENANCE or raw["record_per_input"] is not True:
        raise ValueError("schema 6.6 requires frozen campaign binding and per-input diagnostics")
    proxy = {k: v for k, v in raw.items() if k not in required | extra}
    proxy["candidate_provenance"] = "exploratory_frozen_m953_pool"
    validated = validate_workflow(proxy, samples, backgrounds, oracle, paired, context, source)
    validated["candidate_provenance"] = PROVENANCE
    validated.update({k: raw[k] for k in required | extra if k in raw})
    for key in ("campaign_manifest", "final_registry"):
        if key in validated:
            validated[key] = str((source.parent / validated[key]).resolve())
    if validated["mode"] == "final_only" and not extra <= raw.keys():
        raise ValueError("final execution requires an all-pool frozen selection registry")
    return validated


def validate_campaign(path):
    """Validate every frozen member configuration without accessing model inputs."""
    from .conditional_config import load_conditional
    path = Path(path).resolve()
    record = load_campaign(path)
    for pool in record["pools"]:
        configuration = load_conditional(path.parent / "configs" / (pool["pool_id"] + "_selection.toml"))
        if sha256_file(configuration.benchmark.checkpoint) != record["checkpoint_sha256"] or sha256_file(configuration.benchmark.asset_lock) != record["asset_lock_sha256"]:
            raise ValueError("frozen campaign asset identity changed")
    return {"name": record["name"], "pools": len(record["pools"]),
            "campaign_sha256": sha256_file(path), "operation_counts": record["operation_counts"]}


def verify_pool_configuration(configuration, *, execution):
    workflow = configuration.workflow
    path = Path(workflow["campaign_manifest"])
    record = load_campaign(path, workflow["campaign_sha256"])
    parent = None
    if "joint_campaign" in record:
        from .multiber_campaign import verify_member
        parent = verify_member(path, record, require_sealed=execution and workflow["mode"] == "final_only")
    pools = {p["pool_id"]: p for p in record["pools"]}
    if workflow["pool_id"] not in pools:
        raise ValueError("unprespecified pool")
    pool = pools[workflow["pool_id"]]
    # Compare every resolved scientific field, not just candidate identities.
    expected = _bound_document(record, pool, path, workflow["campaign_sha256"])
    actual = configuration.resolved_document()
    for key in ("benchmark", "samples", "risk", "candidates", "backgrounds", "execution", "restricted_oracle", "protection_context", "paired_swap_evaluation"):
        left = json.loads(json.dumps(actual[key]))
        right = expected[key]
        if key == "benchmark":
            left.pop("ranking_samples"); left.pop("evaluation_samples")
            right = {**right, **{k: str(Path(right[k]).resolve()) for k in ("checkpoint", "dataset", "asset_lock")}}
        if key == "protection_context":
            right = {**right, "source_bundle": str(Path(right["source_bundle"]).resolve())}
        if key == "paired_swap_evaluation":
            left["enabled"] = False
        if left != right:
            raise ValueError(f"frozen campaign {key} changed")
    if workflow["purpose"] != record["purpose"]:
        raise ValueError("campaign purpose changed")
    if configuration.output.root != path.parent / "runs":
        raise ValueError("campaign outputs must remain in its isolated runs directory")
    if workflow["mode"] == "final_only":
        if Path(workflow["final_registry"]).resolve() != (path.parent / "final-plan/selections.json").resolve():
            raise ValueError("final registry must be the campaign's unique sealed registry")
        registry = _read(workflow["final_registry"])
        if sha256_file(workflow["final_registry"]) != workflow["final_registry_sha256"] or registry["campaign_sha256"] != workflow["campaign_sha256"]:
            raise ValueError("final selection registry hash mismatch")
        if set(registry["selections"]) != set(pools):
            raise ValueError("all pool selections must be frozen before final evaluation")
        if execution and (path.parent / "final-plan/SEALED").read_text().strip() != workflow["final_registry_sha256"]:
            raise ValueError("final selection registry is not sealed")
        for key, value in registry["selections"][pool["pool_id"]].items():
            if workflow.get(key) != value:
                raise ValueError("selected mask source differs from all-pool frozen registry")
    if execution:
        if sha256_file(configuration.benchmark.checkpoint) != record["checkpoint_sha256"]:
            raise ValueError("frozen campaign checkpoint identity changed")
        if sha256_file(configuration.benchmark.asset_lock) != record["asset_lock_sha256"]:
            raise ValueError("frozen campaign asset lock identity changed")
        allowed = ([arm["manifest_sha256"] for arm in parent["arms"]] if parent is not None
                   else workflow["campaign_sha256"])
        check_exposure(audit_exposure(allowed_campaign=allowed),
                       record["samples"]["final_offset"], record["samples"]["final_count"], record["purpose"])
    return record


def _receipt(root, pool_id, phase):
    return root / "receipts" / f"{pool_id}_{phase}.json"


def verified_receipt(path, digest, pool_id, phase):
    from .conditional_bundle import verify_conditional_bundle
    value = _read(path)
    if value["campaign_sha256"] != digest or value["pool_id"] != pool_id or value["phase"] != phase:
        raise ValueError("receipt campaign/pool/phase mismatch")
    manifest = verify_conditional_bundle(value["bundle"])
    payload = _read(Path(value["bundle"]) / "scientific_payload.json")
    if (Path(value["bundle"]) / "FINALIZED").read_text().strip() != f"scientific_payload_sha256={value['payload_sha256']}":
        raise ValueError("receipt finalization marker mismatch")
    if digest_object(payload) != value["payload_sha256"] or manifest["scientific_payload_sha256"] != value["payload_sha256"] or payload.get("campaign_binding") != {"campaign_sha256": digest, "pool_id": pool_id} or payload["workflow"]["mode"] != phase + "_only":
        raise ValueError("receipt scientific payload binding mismatch")
    return value, payload


@contextmanager
def campaign_lock(root):
    import fcntl
    with (root / "execution.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("campaign already executing; simultaneous access is prohibited") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def seal_selections(path):
    path = Path(path).resolve(); record = load_campaign(path); root = path.parent
    if "joint_campaign" in record:
        from .multiber_campaign import require_all_selections
        require_all_selections(Path(record["joint_campaign"]))
    digest = sha256_file(path)
    from .conditional_config import load_conditional
    with campaign_lock(root):
        selections = {}
        for pool in record["pools"]:
            receipt, payload = verified_receipt(_receipt(root, pool["pool_id"], "selection"), digest, pool["pool_id"], "selection")
            selections[pool["pool_id"]] = {"selection_bundle": receipt["bundle"], "selection_payload_sha256": receipt["payload_sha256"],
                "selection_artifact_sha256": sha256_file(Path(receipt["bundle"]) / "artifacts/frozen_selection.json")}
        destination = root / "final-plan"
        destination.mkdir(exist_ok=False)
        registry = destination / "selections.json"
        _write_json(registry, {"schema_version": VERSION, "campaign_sha256": digest, "selections": selections})
        for pool in record["pools"]:
            binding = {**selections[pool["pool_id"]], "final_registry": str(registry), "final_registry_sha256": sha256_file(registry)}
            doc = _bound_document(record, pool, path, digest, final_record=binding)
            cfgpath = destination / (pool["pool_id"] + "_final.toml")
            _write_bytes(cfgpath, _toml(doc).encode())
            load_conditional(cfgpath)
        _write_bytes(destination / "SEALED", (sha256_file(registry) + "\n").encode())
    return {"registry": str(registry), "sha256": sha256_file(registry), "pools": len(selections)}


def run_campaign(path, phase, *, pool_ids=None, progress=True):
    """Sequential, resumable at completed pool boundaries. Interrupted work fails closed."""
    path = Path(path).resolve(); record = load_campaign(path); root = path.parent
    digest = sha256_file(path)
    if "joint_campaign" in record:
        from .multiber_campaign import verify_member
        verify_member(path, record, require_sealed=phase == "final")
    if phase not in {"selection", "final"}:
        raise ValueError("phase must be selection or final")
    pools = record["pools"]
    ids = {p["pool_id"] for p in pools}
    if pool_ids is not None:
        if not set(pool_ids) <= ids or len(set(pool_ids)) != len(pool_ids):
            raise ValueError("unknown/duplicate requested pool")
        pools = [p for p in pools if p["pool_id"] in pool_ids]
    if phase == "final":
        registry = root / "final-plan/selections.json"
        if (root / "final-plan/SEALED").read_text().strip() != sha256_file(registry):
            raise ValueError("all selections must be sealed before final execution")
    from .conditional_config import load_conditional
    from .conditional_damage import execute_conditional_damage
    from .conditional_bundle import create_conditional_bundle
    from .cross_pool_progress import CampaignProgress
    completed = []
    planned = record["operation_counts"]["selection_state_requests_per_pool" if phase == "selection" else "final_state_requests_upper_bound_per_pool"]
    with campaign_lock(root), CampaignProgress([p["pool_id"] for p in pools], enabled=progress, planned=planned) as reporter:
        (root / "receipts").mkdir(exist_ok=True)
        (root / "attempts").mkdir(exist_ok=True)
        for index, pool in enumerate(pools):
            pid = pool["pool_id"]; receipt = _receipt(root, pid, phase)
            if receipt.exists():
                value, _ = verified_receipt(receipt, digest, pid, phase)
                completed.append(value); reporter.complete(index, resumed=True); continue
            configpath = (root / "configs" / (pid + "_selection.toml") if phase == "selection"
                          else root / "final-plan" / (pid + "_final.toml"))
            configuration = load_conditional(configpath)
            verify_pool_configuration(configuration, execution=True)
            # An unfinished attempt can have consumed confirmation inputs. Never
            # silently rerun it or choose new seeds; explicit recovery audit required.
            _write_json(root / "attempts" / f"{pid}_{phase}.json", {"schema_version": VERSION,
                "campaign_sha256": digest, "pool_id": pid, "phase": phase, "configuration_sha256": sha256_file(configpath)})
            reporter.start(index)
            try:
                outcome = execute_conditional_damage(configuration, progress=reporter.callback(index))
                bundle = create_conditional_bundle(outcome, repository_root=Path(__file__).resolve().parents[1])
                value = {"schema_version": VERSION, "campaign_sha256": digest, "pool_id": pid,
                         "phase": phase, "bundle": str(bundle), "payload_sha256": outcome.scientific_payload_sha256}
                _write_json(receipt, value)
                verified_receipt(receipt, digest, pid, phase)
                completed.append(value); reporter.complete(index)
            except BaseException as error:
                reporter.fail(index, error); raise
    return {"campaign_sha256": digest, "phase": phase, "completed": completed}
