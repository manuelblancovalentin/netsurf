"""Versioned task/partition capabilities for schema 6.7; no new fault engine.

Older schema-6 paths deliberately retain their original KWS contracts and hashes.
Windows are prespecified source positions, not a claim of identical distributions.
"""
from contextlib import contextmanager
from dataclasses import dataclass, replace
import json
from pathlib import Path

import numpy as np

from .manifests import digest_object, sha256_file


@dataclass(frozen=True)
class ConditionalCapability:
    target: str
    adapter: str
    benchmark_id: str
    lock_id: str
    split: str
    source_split: str
    windows: tuple[tuple[int, int], ...]
    risk: str = "categorical_crossentropy"
    reference_inputs: int = 200

    def indices(self, role, offset, count):
        if self.risk == "one_minus_file_roc_auc":
            raise ValueError("ToyCar requires explicit complete-file cohorts, not positional windows")
        roles = ("conditional_calibration", "conditional_model_selection", "conditional_final_evaluation")
        if role not in roles or type(offset) is not int or type(count) is not int:
            raise ValueError("unsupported conditional role or noninteger sample request")
        start, stop = self.windows[roles.index(role)]
        if count < 1 or offset < 0 or start + offset + count > stop:
            raise ValueError(f"{role} offset/count exceeds fixed [{start},{stop}) window")
        return tuple(range(start + offset, start + offset + count))

    def record(self):
        from dataclasses import asdict
        if self.target == "toycar_int8":
            from .cohort_risk import CONTRACT
            return {"schema_version": "2.0.0", **asdict(self), "risk_contract": CONTRACT,
                    "partitioning": "explicit complete-file identities, never contiguous window offsets"}
        return {"schema_version": "1.0.0", **asdict(self),
                "window_interpretation": "disjoint source positions; no iid or identical-class-composition claim",
                "risk_units": "natural-log nats per image/clip; integer softmax, probability floor 1e-12"}


CAPABILITIES = (
    ConditionalCapability("kws_dscnn_int8", "mlperf_tiny_kws_int8_v1",
        "mlperf_tiny_kws_dscnn_int8_v1", "mlperf-tiny/kws", "full_test_4890",
        "speech_commands_tfds_test_interleaved_blocks16", ((1000, 2024), (2024, 3048), (3048, 4890)),
        reference_inputs=16),
    ConditionalCapability("resnet8_int8", "mlperf_tiny_ic_int8_v1",
        "mlperf_tiny_ic_resnet8_int8_v1", "mlperf-tiny/ic", "test", "cifar10_test",
        ((0, 1000), (1000, 3000), (3000, 10000))),
    ConditionalCapability("toycar_int8", "mlperf_tiny_ad_int8_v1",
        "mlperf_tiny_ad_toycar_int8_v1", "mlperf-tiny/ad", "official_248_files",
        "dcase2020_toycar_official_development_test", ((0, 248),) * 3,
        risk="one_minus_file_roc_auc", reference_inputs=392),
)


def conditional_capability(adapter):
    for value in CAPABILITIES:
        if adapter in {value.adapter, value.target}:
            return value
    raise ValueError(f"no validated conditional capability for {adapter}")


def validate_benchmark(benchmark, *, check_inputs):
    cap = conditional_capability(benchmark.adapter)
    if (benchmark.id, benchmark.split, benchmark.quantization) != (cap.benchmark_id, cap.split, "tflite_affine_int8"):
        raise ValueError("conditional benchmark identity/split/precision does not match its capability")
    if check_inputs:
        verified_storage(benchmark)
    return cap


def verified_storage(benchmark):
    """Bind the current portfolio pin; an external lock change is not auto-accepted."""
    from .asset_lock import VerifiedAssetLock
    from .benchmarks.capabilities import capability_record
    from .tflite_storage import TFLiteModelStorage
    cap = conditional_capability(benchmark.adapter)
    record = capability_record(cap.target)
    lock = VerifiedAssetLock.load(benchmark.asset_lock, expected_benchmark_id=cap.lock_id)
    if lock.digest != record["reference_lock"]["canonical_digest"]:
        raise ValueError("conditional asset-lock identity drift; audit required")
    model = record["checkpoint"]["files"][0]
    if lock.file(model["path"]) != benchmark.checkpoint.resolve():
        raise ValueError("conditional checkpoint does not resolve to pinned lock artifact")
    storage = TFLiteModelStorage.from_path(benchmark.checkpoint, expected_sha256=model["sha256"], eligible_roles=("kernel",))
    if sum(d.size * d.bit_width for d in storage.descriptors) != record["eligible_bits"]:
        raise ValueError("conditional eligible kernel population differs from portfolio")
    return cap, lock, storage


def create_conditional_adapter(configuration):
    import tensorflow as tf
    if tf.__version__ != "2.14.1":
        raise ValueError("schema 6.7 requires pinned TensorFlow 2.14.1 BUILTIN_REF")
    verified_storage(configuration.benchmark)
    if configuration.schema_version == "6.8.0":
        from .adapters.toycar_conditional import ConditionalToyCarAdapter
        return ConditionalToyCarAdapter(configuration)
    if configuration.benchmark.adapter == "mlperf_tiny_ic_int8_v1":
        from .adapters.mlperf_tiny import MlperfTinyImageClassificationAdapter

        class ConditionalResNetAdapter(MlperfTinyImageClassificationAdapter):
            def sample_partition(self, role, count, *, offset=0):
                if not role.startswith("conditional_"):
                    return super().sample_partition(role, count, offset=offset)
                cap = conditional_capability("resnet8_int8")
                indices = cap.indices(role, offset, count)
                x, y, part = super().sample_partition("final_evaluation", count, offset=indices[0])
                return x, y, replace(part, role=role, provenance=(
                    "M98 fixed disjoint CIFAR-10 test-batch source windows; no identical-distribution assertion"))

            @contextmanager
            def capture_input_losses(self):
                if getattr(self, "_capture", None) is not None:
                    raise RuntimeError("nested input loss capture")
                self._capture = []
                try:
                    yield self._capture
                finally:
                    self._capture = None
                    if hasattr(self, "_captured_probabilities"):
                        del self._captured_probabilities

            def predict_outputs(self, inputs):
                result = super().predict_outputs(inputs)
                if getattr(self, "_capture", None) is not None:
                    self._captured_probabilities = result
                return result

            def evaluate(self, inputs, labels, *, batch_size=256):
                # Preserve the authoritative scalar reduction, capture from the same call.
                result = super().evaluate(inputs, labels, batch_size=batch_size)
                if getattr(self, "_capture", None) is not None:
                    p = self._captured_probabilities
                    self._capture.append(-np.log(np.maximum(p[np.arange(len(labels)), labels], 1e-12)))
                    del self._captured_probabilities
                return result

        b = configuration.benchmark
        return ConditionalResNetAdapter(b.checkpoint, b.dataset, asset_lock=b.asset_lock,
            inference_backend=b.inference_backend, split=b.split, injectable_parameters=b.injectable_parameters)
    from .adapters.registry import create_adapter
    return create_adapter(configuration, None)


def _indices(value, source):
    found = set()
    if isinstance(value, dict):
        aliases = {source}
        if source == "cifar10_test":
            aliases.add("cifar10_test_mlperf_tiny_ic01_official_200")
        if value.get("source_split") in aliases:
            found.update(int(i) for i in value.get("indices", ()))
            if "original_index" in value:
                found.add(int(value["original_index"]))
        for child in value.values():
            found.update(_indices(child, source))
    elif isinstance(value, list):
        for child in value:
            found.update(_indices(child, source))
    return found


def audit_exposure(cap, *, roots=None, allowed_campaign=None):
    if cap.target == "kws_dscnn_int8":
        from .campaign_exposure import audit_exposure as kws_audit
        return kws_audit(roots, allowed_campaign=allowed_campaign)
    from .campaign_exposure import known_roots
    from .benchmarks.capabilities import capability_record
    model_sha = capability_record(cap.target)["checkpoint"]["files"][0]["sha256"]
    exposed, runs, seen = set(), [], set()
    for root in known_roots() if roots is None else roots:
        for path in sorted(Path(root).rglob("manifest.json")):
            bundle = path.parent.resolve()
            if bundle in seen or not (bundle / "FINALIZED").is_file():
                continue
            seen.add(bundle)
            manifest = json.loads(path.read_text())
            if manifest.get("benchmark", {}).get("checkpoint_sha256") != model_sha:
                continue
            payload_file = bundle / "scientific_payload.json"
            if payload_file.stat().st_size > 128 * 1024**2:
                raise ValueError(f"exposure payload exceeds bounded metadata audit: {bundle}")
            checks = json.loads((bundle / "checksums.json").read_text())
            payload = json.loads(payload_file.read_text())
            digest = digest_object(payload)
            if (sha256_file(payload_file) != checks["files"]["scientific_payload.json"]
                    or digest != manifest["scientific_payload_sha256"]
                    or (bundle / "FINALIZED").read_text().strip() != f"scientific_payload_sha256={digest}"):
                raise ValueError(f"corrupt exposure evidence: {bundle}")
            # Payload, not unbound manifest fields, supplies original identities.
            indices = sorted(_indices(payload, cap.source_split))
            same = allowed_campaign is not None and payload.get("campaign_binding", {}).get("campaign_sha256") == allowed_campaign
            if not same:
                exposed.update(indices)
            runs.append({"bundle": str(bundle), "scientific_payload_sha256": digest,
                "checksums_sha256": sha256_file(bundle / "checksums.json"),
                "original_test_indices": indices, "same_frozen_campaign": same})
            if cap.target == "toycar_int8":
                from .toycar_exposure import role_inventory
                runs[-1]["role_inventory"] = role_inventory(payload)
                runs[-1]["workflow"] = payload.get("workflow", {})
    result = {"schema_version": "1.0.0", "checkpoint_sha256": model_sha,
        "source_split": cap.source_split, "runs": runs, "exposed_test_indices": sorted(exposed),
        "limitations": "Known finalized payloads only, including official-subset original-index aliases; "
        "failed/manual/concurrent/out-of-root work needs disclosure. Full clean reference already used all test images. "
        "Payload checksums verified; multi-GB prediction artifacts not reread."}
    if cap.target == "toycar_int8":
        result["limitations"] = result["limitations"].replace("test images", "248 official files")
    return result


def check_exposure(cap, audit, offset, count, purpose):
    requested = cap.indices("conditional_final_evaluation", offset, count)
    overlap = sorted(set(requested) & set(audit["exposed_test_indices"]))
    if purpose == "confirmation" and overlap:
        raise ValueError(f"prior exposure overlap in {cap.source_split}: {overlap}")
    return {"ledger_sha256": digest_object(audit), "requested_original_indices": list(requested),
            "source_split": cap.source_split, "prior_overlap": overlap, "purpose": purpose,
            "limitations": audit["limitations"]}


def configuration_exposure(configuration):
    cap = conditional_capability(configuration.benchmark.adapter)
    audit = audit_exposure(cap)
    return check_exposure(cap, audit, configuration.samples.evaluation_offset,
                          configuration.samples.evaluation_count, configuration.workflow.get("purpose", "engineering"))


def validate_workflow(raw, samples, backgrounds, oracle, paired, context, source, *, scalar_cohort=False):
    """Reuse separated-selection validation, without the historical KWS exposure gate."""
    from .selection_preflight import validate_workflow as validate
    from .cross_pool import PROVENANCE
    if raw.get("candidate_provenance") not in {PROVENANCE, "prespecified_explicit_engineering/v1"}:
        raise ValueError("schema 6.7 needs explicit engineering or probability-sampled provenance")
    if raw.get("candidate_provenance") == "prespecified_explicit_engineering/v1" and raw.get("purpose") != "engineering":
        raise ValueError("explicit engineering candidates cannot be labeled confirmation")
    extra = {"campaign_manifest", "campaign_sha256", "pool_id", "record_per_input", "final_registry", "final_registry_sha256"}
    proxy = {k: v for k, v in raw.items() if k not in extra}
    proxy["candidate_provenance"] = "exploratory_frozen_m953_pool"
    result = validate(proxy, samples, backgrounds, oracle, paired, context, source, audit_kws=False)
    result["candidate_provenance"] = raw["candidate_provenance"]
    result.update({k: v for k, v in raw.items() if k in extra})
    if type(raw.get("record_per_input", False)) is not bool:
        raise ValueError("record_per_input must be boolean")
    if scalar_cohort and raw.get("record_per_input", False):
        raise ValueError("scalar-cohort ROC-AUC has no additive per-input diagnostic; record_per_input must be false")
    campaign_keys = {"campaign_manifest", "campaign_sha256", "pool_id"}
    if raw.keys() & campaign_keys:
        if not campaign_keys <= raw.keys() or raw.get("record_per_input") is not (not scalar_cohort):
            raise ValueError("campaign binding incomplete or missing additive input diagnostics")
        if result["mode"] == "final_only" and not {"final_registry", "final_registry_sha256"} <= raw.keys():
            raise ValueError("campaign final requires sealed all-pool registry")
        for key in ("campaign_manifest", "final_registry"):
            if key in result:
                result[key] = str((source.parent / result[key]).resolve())
    elif raw["candidate_provenance"] == PROVENANCE:
        raise ValueError("probability-sampled pool requires a frozen campaign")
    return result
