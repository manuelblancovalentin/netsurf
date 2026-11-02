"""Bounded DistilBERT engineering conformance with immutable receipts."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path

from .adapters.transformers_pinned import PinnedTransformerAdapter, read_sst2_jsonl
from .faults import FaultEvent, FaultModel, FaultSchedule
from .identities import BitIdentity
from .manifests import canonical_json_bytes, digest_object, git_provenance, sha256_file
from .protection import ProtectionMask
from .transformer_assets import load_ready_asset


def _write_exclusive(path: Path, value: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def read_smoke_receipt(directory: Path) -> dict:
    root = Path(directory)
    if not (root / "FINALIZED").is_file():
        raise ValueError(f"partial or absent smoke receipt: {root}")
    checksums = json.loads((root / "checksums.json").read_text())
    for relative, expected in checksums.items():
        if sha256_file(root / relative) != expected:
            raise ValueError(f"corrupt smoke receipt: {relative}")
    return json.loads((root / "payload.json").read_text())


def run_distilbert_smoke(manifest_path: Path, portfolio_path: Path,
                         output: Path, source_root: Path) -> dict:
    output = Path(output).resolve()
    staging = output.with_name("." + output.name + ".incomplete")
    if output.exists():
        return read_smoke_receipt(output)
    if staging.exists():
        raise FileExistsError(f"interrupted smoke receipt retained for audit: {staging}")
    asset = load_ready_asset(manifest_path, "distilbert_sst2_fp32", portfolio_path)
    all_examples = read_sst2_jsonl(asset)
    identities = asset.record["partitions"]["engineering"][:2]
    if len(identities) != 2:
        raise ValueError("two prespecified engineering examples required")
    by_id = {x.source_index: x for x in all_examples}
    examples = tuple(by_id[index] for index in identities)
    adapter = PinnedTransformerAdapter(asset)
    adapter.load()
    baseline = adapter.storage.hashes()
    clean = adapter.evaluate_sst2(examples)
    if clean != adapter.evaluate_sst2(examples):
        raise ValueError("clean reference is not deterministic")
    classifier = adapter.storage.parameters.get("classifier.weight")
    if classifier is None:
        raise ValueError("no verified DistilBERT classifier.weight in eligible population")
    # Choose by clean absolute stored weight only; no fault outcome guides this choice.
    import torch
    magnitudes = classifier.detach().abs().reshape(-1)
    first_index = int(torch.argmax(magnitudes))
    second_index = (first_index + 1) % classifier.numel()
    first = BitIdentity("classifier.weight", first_index, adapter.storage.codec.width - 1)
    second = BitIdentity("classifier.weight", second_index, adapter.storage.codec.width - 1)
    law = FaultModel("weight", "cpu_pytorch_linear_weight_storage", "persistent_trial",
                     "forced_engineering", 0.0)
    single = FaultSchedule(law, 1001, (FaultEvent(None, first),), adapter.storage.bit_count, len(examples))
    double = FaultSchedule(law, 1002, (FaultEvent(None, first), FaultEvent(None, second)),
                           adapter.storage.bit_count, len(examples))
    none = ProtectionMask("unprotected", frozenset(), adapter.storage.bit_count, 0, {})
    one = ProtectionMask("protect_first", frozenset((first,)), adapter.storage.bit_count, 1, {})
    full = ProtectionMask("full_protection_control", frozenset((first, second)),
                          adapter.storage.bit_count, 2, {})
    trials = {}
    for name, schedule, mask in (("one_bit", single, none), ("multi_bit", double, none),
                                 ("full_control", double, full), ("equal_budget_protected", double, one)):
        with adapter.storage.injected(schedule, mask) as report:
            if name == "one_bit" and adapter.storage.hashes() == baseline:
                raise ValueError("injected bytes did not reach runtime parameter storage")
            risk = adapter.evaluate_sst2(examples)
        if adapter.storage.hashes() != baseline:
            raise ValueError("runtime state was not restored")
        trials[name] = {"risk": asdict(risk), "report": asdict(report),
                        "schedule_events": [event.key() for event in schedule.events],
                        "schedule_seed": schedule.seed,
                        "schedule_full_sha256": digest_object({"model": asdict(schedule.model),
                            "seed": schedule.seed, "events": [event.key() for event in schedule.events],
                            "universe_size": schedule.universe_size,
                            "inference_count": schedule.inference_count}),
                        "mask_sha256": mask.digest()}
    if trials["full_control"]["risk"] != asdict(clean):
        raise ValueError("full suppression did not reproduce clean outputs")
    if trials["one_bit"]["risk"]["logits"] == asdict(clean)["logits"]:
        raise ValueError("injected classifier bit did not measurably affect model inference")
    if adapter.evaluate_sst2(examples) != clean:
        raise ValueError("post-trial clean inference differs from reference")
    payload = {"schema_version": "transformer-smoke/v1", "status": "engineering-only",
        "asset_manifest_sha256": asset.manifest_sha256, "asset_record_sha256": asset.record_sha256,
        "source": git_provenance(source_root), "target": asset.target,
        "eligible_population": "contiguous CPU nn.Linear.weight FP32 stored bytes only",
        "eligible_bits": adapter.storage.bit_count,
        "descriptors": [asdict(x) for x in adapter.storage.descriptors],
        "engineering_source_indices": identities, "risk_version": "transformer-risk/v1",
        "clean": asdict(clean), "trials": trials,
        "limitation": "two exposed engineering examples; no effectiveness or production claim"}
    output.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    _write_exclusive(staging / "payload.json", canonical_json_bytes(payload) + b"\n")
    manifest = {"schema_version": "transformer-smoke-manifest/v1",
                "payload_sha256": digest_object(payload), "source_asset": str(Path(manifest_path).resolve())}
    _write_exclusive(staging / "manifest.json", canonical_json_bytes(manifest) + b"\n")
    checksums = {name: sha256_file(staging / name) for name in ("payload.json", "manifest.json")}
    _write_exclusive(staging / "checksums.json", canonical_json_bytes(checksums) + b"\n")
    _write_exclusive(staging / "FINALIZED", (manifest["payload_sha256"] + "\n").encode("ascii"))
    staging.rename(output)
    return payload
