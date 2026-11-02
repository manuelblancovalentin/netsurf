"""Fail-closed capability records over the existing 18-target portfolio.

Inspection is side-effect free; probe is an explicit, bounded engineering action.
No precision conversion, asset download, delegate substitution or training.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import resource
import sys
from time import perf_counter

import numpy as np

from ..asset_lock import VerifiedAssetLock, sha256_path
from ..faults import FaultEvent, FaultModel, FaultSchedule
from ..manifests import digest_object, git_provenance
from ..protection import ProtectionMask

ROOT = Path(__file__).resolve().parents[2]
PORTFOLIO = ROOT / "config/benchmark_expansion_registry.json"
PINNED_PYTHON = "/projects/kona-assets/manuelbv/.venv-mlperf/bin/python"
EXECUTABLE = {
    "kws_dscnn_int8": "mlperf_tiny_kws_dscnn_int8_v1",
    "toycar_int8": "mlperf_tiny_ad_toycar_int8_v1",
    "resnet8_int8": "mlperf_tiny_ic_resnet8_int8_v1",
    "streaming_dscnn1d_int8": "mlperf_tiny_sww_short_int8_v1",
}


class UnsupportedCapability(ValueError):
    pass


def storage_contract(precision, *, official_int8=False):
    if precision not in {"FP32", "BF16", "INT8", "INT4"}:
        raise UnsupportedCapability(f"unsupported storage precision: {precision}")
    if precision == "FP32":
        return {"word_bits": 32, "encoding": "IEEE-754 binary32", "sign": [31],
                "exponent": list(range(23,31)), "mantissa": list(range(23)),
                "endianness": "must verify acquired storage and runtime, not inferred",
                "exceptional": "NaN/Inf must be recorded as explicit failure categories; finite-risk convention not approved"}
    if precision == "BF16":
        return {"word_bits": 16, "encoding": "bfloat16, not IEEE binary16/FP16", "sign": [15],
                "exponent": list(range(7,15)), "mantissa": list(range(7)),
                "exceptional": "NaN/Inf recorded explicitly; implicit FP32/FP16 substitution forbidden"}
    if precision == "INT4":
        return {"word_bits": 4, "encoding": "BLOCKED: signed codebook and quantization recipe not pinned",
                "packing": "BLOCKED: nibble order, group scales, zero points, padding, tensor/word/bit mapping and runtime repacking unknown"}
    return {"word_bits": 8, "encoding": "two's-complement INT8" if official_int8 else "BLOCKED: no approved quantized checkpoint/recipe",
            "packing": "one signed code per byte; LSB index zero" if official_int8 else "unverified",
            "quantization": "actual per-tensor/channel affine metadata; zero allocated codes remain eligible" if official_int8 else "unverified scales/zero points/groups",
            "excluded": ["INT32 biases", "affine metadata", "activations", "runtime duplicate packs"]}


def capability_records():
    document = json.loads(PORTFOLIO.read_text())
    if len(document["entries"]) != 18 or len({r["family"] for r in document["entries"]}) != 11:
        raise ValueError("portfolio inventory differs from the preserved M96 contract")
    records = []
    for entry in document["entries"]:
        target = entry["id"]
        executable = target in EXECUTABLE
        streaming = target == "streaming_dscnn1d_int8"
        tiny = entry["model"]["source"] == "mlcommons/tiny"
        blockers = list(entry.get("blockers", []))
        capabilities = {
            "reference_inference": "VERIFIED" if executable else "BLOCKED",
            "exact_kernel_xor": "VERIFIED" if executable else "BLOCKED",
            "restoration": "VERIFIED" if executable else "BLOCKED",
            "static_s2_s3": "VERIFIED" if executable else "BLOCKED",
            "s4_s5_s6": "VERIFIED" if executable and not streaming else "BLOCKED",
            "continuous_hessian": "VERIFIED" if target == "kws_dscnn_int8" else "BLOCKED",
            "conditional_restricted": "VERIFIED" if target in {"kws_dscnn_int8", "resnet8_int8", "toycar_int8"} else "BLOCKED",
            "independent_study_partitions": "VERIFIED" if executable and not streaming else "BLOCKED",
        }
        if streaming:
            blockers = ["only pinned ten-second engineering fixture; full stream/MUSAN absent",
                "no independently calibrated gradient task loss or held-out partitions",
                "event FP/FN remain separate; no agreed scalar risk for restricted protection"]
        if target == "resnet8_int8":
            blockers += ["conditional studies require schema 6.7 / campaign 1.1, audited exposure and a verified S5 context; not an efficacy claim"]
        if target == "toycar_int8":
            blockers += ["conditional task risk requires schema 6.8 / campaign 1.2 complete-file cohorts, exposure audit and verified S5 context; no additive AUC losses",
                         "continuous classical Hessian not validated; reconstruction STE derivatives are distinct"]
        if not executable:
            blockers += ["no validated NetSuRF loading, stored-bit/runtime mapping and fault-restoration backend for this target"]
        record = {
            "schema_version": "1.0.0", "target": target, "family": entry["family"],
            "benchmark_adapter": EXECUTABLE.get(target), "scope": "short-fixture engineering only" if streaming else "registered pinned target",
            "source_portfolio_sha256": sha256_path(PORTFOLIO),
            "checkpoint": entry["model"], "dataset": entry["dataset"],
            "local_checkpoint_files_present": bool(entry["model"].get("local_root")) and all(
                (Path(entry["model"]["local_root"]) / value["path"]).is_file()
                for value in entry["model"]["files"]),
            "reference_lock": entry.get("reference_lock"), "precision": entry["target_precision"],
            "storage": storage_contract(entry["target_precision"], official_int8=tiny),
            "eligible_bits": entry["eligible_kernel_bits"], "population_status": "bounded exact-XOR validated kernel population" if executable else entry["bit_count_status"],
            "authoritative_backend": entry["runtime"], "primary_task_metrics": entry["task_metric"],
            "primary_task_risk": "1 - file ROC-AUC (requires both classes); reconstruction MSE is a separate surrogate" if target == "toycar_int8" else
                                 "BLOCKED: FP/FN event metrics have no approved scalar trade-off" if streaming else
                                 "mean categorical cross-entropy" if executable else "BLOCKED: task/backend contract not validated",
            "surrogate_loss": "float-target reconstruction MSE, not ROC-AUC" if target == "toycar_int8" else
                              "decoded-logit CE with STE for S4-S6; separate float64 continuous CE for FKeras" if target == "kws_dscnn_int8" else
                              "requantized-logit CE with STE" if target == "resnet8_int8" else "not validated",
            "capabilities": capabilities, "blockers": blockers,
            "fault_lifetime": "whole stream" if streaming else "persistent evaluation trial",
            "resource_plan": entry["resource_plan"], "cost_model": entry["cost_model"],
            "validation_command": f"{PINNED_PYTHON} -m netsurf benchmarks probe {target} --output workspace/runs/m97_probe_{target}",
            "evidence": ("docs/M99_TOYCAR_CONDITIONAL.md; complete-file schema-6.8 engineering receipts" if target == "toycar_int8" else
                         "docs/M98_CONDITIONAL_ADAPTERS.md; schema-6.7 engineering receipts"
                         if target == "resnet8_int8" else
                         "docs/M97_BENCHMARK_ADAPTER_STATUS.md; probe receipts required for this runtime"),
            "status_does_not_validate_current_environment": True,
        }
        records.append(record)
    return tuple(records)


def capability_record(target):
    matches = [r for r in capability_records() if r["target"] == target or r["benchmark_adapter"] == target]
    if len(matches) != 1:
        raise UnsupportedCapability(f"unknown benchmark capability target: {target}")
    return matches[0]


def require_capability(target, capability):
    record = capability_record(target)
    if record["capabilities"].get(capability) != "VERIFIED":
        raise UnsupportedCapability(f"{record['target']}: {capability} BLOCKED; " + "; ".join(record["blockers"]))
    return record


def load_capability_adapter(target):
    record = require_capability(target, "reference_inference")
    import tensorflow as tf
    if tf.__version__ != "2.14.1":
        raise UnsupportedCapability(f"use {PINNED_PYTHON}; required TensorFlow 2.14.1 BUILTIN_REF, observed {tf.__version__}")
    if record["target"] == "streaming_dscnn1d_int8":
        from ..adapters.mlperf_streaming import MlperfStreamingShortAdapter
        adapter = MlperfStreamingShortAdapter(record["checkpoint"]["local_root"])
    else:
        benchmark_id = {"kws_dscnn_int8":"mlperf-tiny/kws", "toycar_int8":"mlperf-tiny/ad", "resnet8_int8":"mlperf-tiny/ic"}[record["target"]]
        lock = VerifiedAssetLock.load(Path(record["reference_lock"]["path"]), expected_benchmark_id=benchmark_id)
        expected = record["reference_lock"]["canonical_digest"]
        if lock.digest != expected:
            raise UnsupportedCapability(f"asset lock drift: expected {expected}, observed {lock.digest}; review before execution")
        model_file = record["checkpoint"]["files"][0]
        if lock.file_record(model_file["path"])["sha256"] != model_file["sha256"]:
            raise UnsupportedCapability("checkpoint lock differs from portfolio pin")
        model = lock.file(model_file["path"])
        if record["target"] == "resnet8_int8":
            from ..adapters.mlperf_tiny import MlperfTinyImageClassificationAdapter
            adapter = MlperfTinyImageClassificationAdapter(model, lock.file("cifar-10-python.tar.gz"),
                asset_lock=lock.path, split="official_subset_200", injectable_parameters=("kernels",))
        else:
            from ..adapters.mlperf_tiny_multibench import MlperfTinyKwsDscnnAdapter, MlperfTinyToyCarAdapter
            cls = MlperfTinyKwsDscnnAdapter if record["target"] == "kws_dscnn_int8" else MlperfTinyToyCarAdapter
            datasets = list(lock.document["resolved"]["datasets"].values())
            if len(datasets) != 1:
                raise UnsupportedCapability("ambiguous locked dataset root")
            adapter = cls(model, Path(datasets[0]), asset_lock=lock.path)
    adapter.load()
    return adapter


def probe_capability(target, output):
    """Two exposed examples (or short stream), one two-bit fault, full suppression."""
    record = require_capability(target, "exact_kernel_xor")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    started = perf_counter()
    adapter = load_capability_adapter(target)
    load_seconds = perf_counter() - started
    if record["target"] == "streaming_dscnn1d_int8":
        inputs, labels = adapter.fixture()
        partition = adapter.dataset_identity_manifest()
    else:
        inputs, labels, identity = adapter.sample_partition("evaluation", 2)
        partition = identity.to_dict()
    bits = tuple(adapter.bit_identities())
    if len(set(bits)) != len(bits) or len(bits) != record["eligible_bits"]:
        raise ValueError("eligible identity population differs from capability contract")
    before = adapter.values()
    clean = adapter.evaluate(inputs, labels)
    first = bits[0]
    from ..identities import BitIdentity
    events = (FaultEvent(None, first), FaultEvent(None, BitIdentity(first.parameter, first.flat_index, 7)))
    law = FaultModel("weight", "logical_trainable_parameter_bits", "persistent_trial", "independent_bernoulli", 0.0)
    schedule = FaultSchedule(law, 970901, events, len(bits), len(inputs))
    none = ProtectionMask("none", frozenset(), len(bits), 0, {})
    full = ProtectionMask("full", frozenset(bits), len(bits), len(bits), {})
    fault_started = perf_counter()
    faulted, counts = adapter.evaluate_fault_schedule(inputs, labels, schedule, none)
    fault_seconds = perf_counter() - fault_started
    protected, suppressed = adapter.evaluate_fault_schedule(inputs, labels, schedule, full)
    restored = adapter.evaluate(inputs, labels)
    if protected != clean or restored != clean or any(not np.array_equal(before[k],adapter.values()[k]) for k in before):
        raise ValueError("exact full protection/restoration failed")
    if (counts.proposed_events, counts.applied_events, counts.suppressed_events,
            suppressed.proposed_events, suppressed.applied_events, suppressed.suppressed_events) != (2,2,0,2,0,2):
        raise ValueError("fault event accounting failed")
    # The forced events are a semantic test, not a sampled BER=0 trial.
    payload = {"schema_version":"1.0.0", "target":record["target"], "purpose":"bounded engineering, not publication",
        "representation":adapter.representation_manifest(), "partition":partition,
        "inputs_sha256":hashlib.sha256(inputs.tobytes()).hexdigest(),
        "clean":asdict(clean), "faulted":asdict(faulted), "full_protection":asdict(protected),
        "fault_schedule_kind":"explicit two-bit XOR test, not drawn from the configured Bernoulli law",
        "events":[e.key() for e in events], "applied":asdict(counts), "suppressed":asdict(suppressed),
        "restored":True, "eligible_bits":len(bits), "population_sha256":digest_object([b.key() for b in bits])}
    receipt = {"status":"complete", "scientific_payload_sha256":digest_object(payload), "scientific_payload":payload,
        "source":git_provenance(ROOT), "execution":{"python":sys.executable,
            "load_seconds":load_seconds, "fault_evaluation_seconds":fault_seconds,
            "elapsed_seconds":perf_counter()-started,
            "process_peak_rss_mb":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}}
    output.mkdir(parents=True, exist_ok=False)
    path=output/"probe.json"
    with path.open("x") as stream:
        json.dump(receipt,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write("\n")
    with (output/"checksums.json").open("x") as stream:
        json.dump({"schema_version":"1.0.0", "probe.json":sha256_path(path)},stream,indent=2)
    return receipt
