"""Bounded, immutable real-model LM storage-fault conformance evidence."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time
import zipfile

import numpy as np

from .adapters.transformers_pinned import PinnedTransformerAdapter
from .faults import FaultEvent, FaultModel, FaultSchedule
from .identities import BitIdentity
from .manifests import canonical_json_bytes, digest_object, sha256_file
from .protection import ProtectionMask
from .torch_storage import TorchStoredLinearWeights
from .transformer_assets import load_ready_asset


ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "netsurf/transformer_lm_smoke.py", "netsurf/transformer_assets.py",
    "netsurf/transformer_lm_smoke_cli.py", "scripts/run_transformer_lm_smoke_once.sh",
    "netsurf/adapters/transformers_pinned.py", "netsurf/transformer_data.py",
    "netsurf/transformer_risk.py", "netsurf/torch_storage.py",
    "netsurf/storage_bit_codec.py", "netsurf/faults.py",
    "netsurf/protection.py", "netsurf/manifests.py",
)


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush(); os.fsync(stream.fileno())


def _verify_existing(output: Path) -> dict:
    checksums = json.loads((output / "checksums.json").read_text())
    for name, expected in checksums.items():
        if sha256_file(output / name) != expected:
            raise ValueError(f"completed LM smoke artifact changed: {name}")
    with zipfile.ZipFile(output / "complete.zip") as archive:
        if archive.testzip() is not None:
            raise ValueError("completed LM evidence ZIP is corrupt")
    payload = json.loads((output / "payload.json").read_text())
    if (digest_object(payload) != (output / "FINALIZED").read_text().strip() or
        payload["asset_manifest_sha256"] != sha256_file(output / "assets/NETSURF_ASSET_READINESS.json")):
        raise ValueError("completed LM smoke payload binding changed")
    return {"status": "FINALIZED_VERIFIED", "target": payload["target"],
            "output": str(output), "payload_sha256": digest_object(payload)}


def run_lm_smoke(manifest_path: Path, registry_path: Path, target: str,
                 output: Path, *, scp_host: str = "kona") -> dict:
    """Ten one-window requests on validation tokens, never a production study."""
    output = Path(output)
    if output.exists():
        return _verify_existing(output)
    stage = output.with_name("." + output.name + ".incomplete")
    if stage.exists():
        raise FileExistsError(f"partial LM smoke retained: {stage}")
    source_dirty = subprocess.check_output(["git", "status", "--porcelain",
        "--untracked-files=no"], cwd=ROOT, text=True).strip()
    if source_dirty:
        raise ValueError("commit scientific source before real-model smoke")
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
        cwd=ROOT, text=True).strip()
    source_hashes = {name: sha256_file(ROOT / name) for name in SOURCE_FILES}
    readiness_bytes = Path(manifest_path).read_bytes()
    registry_bytes = Path(registry_path).read_bytes()
    asset = load_ready_asset(manifest_path, target, registry_path)
    if (hashlib.sha256(readiness_bytes).hexdigest() != asset.manifest_sha256 or
        hashlib.sha256(registry_bytes).hexdigest() != sha256_file(registry_path)):
        raise ValueError("READY manifest or reviewed registry changed during verification")
    if target not in {"pythia160m_fp32stored_fp32compute",
                      "pythia160m_bf16stored_fp32compute", "qwen3_0_6b_bf16"}:
        raise ValueError("only three reviewed LM runtime configurations are executable")
    started = time.monotonic()
    adapter = PinnedTransformerAdapter(asset)
    adapter.load()
    full_eligible_bits = adapter.storage.bit_count
    source_model = adapter.storage_model or adapter.model
    name = adapter.storage.descriptors[0].name
    width = adapter.storage.codec.width
    candidates = (BitIdentity(name, 0, width - 1),
                  BitIdentity(name, 1, width - 2),
                  BitIdentity(name, 2, width - 10))
    adapter.storage = TorchStoredLinearWeights(source_model, asset.precision,
                                                eligible_bits=candidates)
    source_before = adapter.storage.hashes()
    ids = adapter.token_sequence(split="validation")[:33]
    if len(ids) != 33:
        raise ValueError("validation token cohort has fewer than 33 tokens")
    clean = adapter.evaluate_language_model(ids, context=1024, stride=512)
    clean_repeat = adapter.evaluate_language_model(ids, context=1024, stride=512)
    if asdict(clean) != asdict(clean_repeat):
        raise ValueError("clean teacher-forced window is not bitwise repeatable")
    law = FaultModel("weight", "verified_stored_linear_weight",
                     "persistent_trial", "forced_engineering", 0.0)

    def mask(name: str, protected: tuple[BitIdentity, ...]) -> ProtectionMask:
        return ProtectionMask(name, frozenset(protected), len(candidates),
                              len(protected), {})

    def trial(name: str, selected: tuple[BitIdentity, ...], protected: tuple[BitIdentity, ...],
              seed: int) -> dict:
        schedule = FaultSchedule(law, seed, tuple(FaultEvent(None, bit) for bit in selected),
                                 len(candidates), len(ids) - 1)
        before_words = {bit.key(): _word(adapter.storage, bit) for bit in selected}
        with adapter.injected(schedule, mask(name, protected)) as report:
            stored_words = {bit.key(): _word(adapter.storage, bit) for bit in selected}
            consumed_words = {bit.key(): _compute_word(adapter, bit) for bit in selected}
            risk = adapter.evaluate_language_model(ids, context=1024, stride=512)
        if adapter.storage.hashes() != source_before:
            raise ValueError("LM stored bytes were not restored")
        return {"seed": seed, "schedule_sha256": schedule.digest(),
                "events": [event.key() for event in schedule.events],
                "mask_sha256": mask(name, protected).digest(),
                "protected": [bit.key() for bit in protected],
                "report": asdict(report), "source_words_before": before_words,
                "source_words_during": stored_words,
                "inference_words_during": consumed_words,
                "risk": asdict(risk), "risk_sha256": digest_object(asdict(risk))}

    sign, exponent, significand = candidates
    trials = {
        "sign": trial("sign", (sign,), (), 47101),
        "exponent": trial("exponent", (exponent,), (), 47102),
        "significand": trial("significand", (significand,), (), 47103),
        "multi_unprotected": trial("multi", (sign, exponent), (), 47104),
        "multi_protected": trial("multi_one_protected", (sign, exponent), (exponent,), 47104),
        "full_control": trial("full_control", (sign, exponent), (sign, exponent), 47104),
        "multi_repeat": trial("multi_repeat", (sign, exponent), (), 47104),
    }
    if (trials["multi_unprotected"]["risk"] != trials["multi_repeat"]["risk"] or
        trials["multi_unprotected"]["schedule_sha256"] != trials["multi_repeat"]["schedule_sha256"] or
        trials["full_control"]["risk"] != asdict(clean)):
        raise ValueError("same-schedule repeat or full-protection control differs")
    clean_after = adapter.evaluate_language_model(ids, context=1024, stride=512)
    if asdict(clean_after) != asdict(clean):
        raise ValueError("clean task output differs after state restoration")
    if (Path(manifest_path).read_bytes() != readiness_bytes or
        Path(registry_path).read_bytes() != registry_bytes or
        any(sha256_file(ROOT / name) != expected for name, expected in source_hashes.items()) or
        subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                cwd=ROOT, text=True).strip()):
        raise ValueError("asset handoff or scientific source changed during LM trial")
    if load_ready_asset(manifest_path, target, registry_path).record_sha256 != asset.record_sha256:
        raise ValueError("READY target bytes changed during LM trial")
    payload = {"schema_version": "transformer-lm-fault-smoke/v1", "status": "ENGINEERING_ONLY",
        "source_commit": source_commit, "source_files_sha256": source_hashes,
        "target": target, "asset_manifest_sha256": asset.manifest_sha256,
        "asset_record_sha256": asset.record_sha256,
        "registry_sha256": sha256_file(registry_path),
        "checkpoint_files": list(asset.model_files),
        "dataset_files": list(asset.dataset_files),
        "stored_precision": asset.precision, "computation_precision": asset.computation_precision,
        "storage_to_computation": ("BF16 source words widened into a distinct verified FP32 CPU copy before inference"
                                   if asset.precision != asset.computation_precision else
                                   "faulted stored CPU parameter bytes consumed directly"),
        "full_eligible_linear_weight_bits": full_eligible_bits,
        "bounded_candidate_bits": [bit.key() for bit in candidates],
        "validation_token_identity": adapter.last_token_sequence_identity,
        "bounded_token_slice": {"start": 0, "stop": len(ids),
            "token_ids_sha256": hashlib.sha256(np.asarray(
                ids, dtype="<i4").tobytes()).hexdigest()},
        "scored_target_tokens": len(ids) - 1, "clean": asdict(clean),
        "clean_sha256": digest_object(asdict(clean)), "trials": trials,
        "model_requests": 10, "scored_token_requests": 10 * (len(ids) - 1),
        "runtime_seconds": time.monotonic() - started,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "limitation": "forced three-bit engineering support on one 32-target validation window; no BER effectiveness claim"}
    stage.mkdir(parents=True)
    _write(stage / "payload.json", canonical_json_bytes(payload) + b"\n")
    manifest = {"schema_version": "transformer-lm-fault-smoke-manifest/v1",
                "payload_sha256": digest_object(payload), "source_commit": source_commit,
                "asset_manifest_sha256": asset.manifest_sha256,
                "registry_sha256": payload["registry_sha256"]}
    _write(stage / "manifest.json", canonical_json_bytes(manifest) + b"\n")
    _write(stage / "assets/NETSURF_ASSET_READINESS.json", readiness_bytes)
    _write(stage / "assets/transformer_runtime_registry.json", registry_bytes)
    with zipfile.ZipFile(stage / "complete.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ("payload.json", "manifest.json", "assets/NETSURF_ASSET_READINESS.json",
                     "assets/transformer_runtime_registry.json"):
            archive.write(stage / name, name)
        for name in SOURCE_FILES:
            archive.write(ROOT / name, "source/" + name)
    with zipfile.ZipFile(stage / "compact.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(stage / "payload.json", "payload.json")
        archive.write(stage / "manifest.json", "manifest.json")
    commands = "\n".join(f"scp {scp_host}:{(output/name).as_posix()} ~/Downloads/{target}_{name}"
                         for name in ("compact.zip", "complete.zip")) + "\n"
    _write(stage / "mac_scp_commands.txt", commands.encode())
    _write(stage / "FINALIZED", (digest_object(payload) + "\n").encode())
    files = ["payload.json", "manifest.json", "assets/NETSURF_ASSET_READINESS.json",
             "assets/transformer_runtime_registry.json", "compact.zip", "complete.zip",
             "mac_scp_commands.txt", "FINALIZED"]
    checksums = {name: sha256_file(stage / name) for name in files}
    _write(stage / "checksums.json", canonical_json_bytes(checksums) + b"\n")
    for path in stage.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    stage.rename(output)
    return _verify_existing(output)


def _word(storage: TorchStoredLinearWeights, bit: BitIdentity) -> str:
    raw = storage._bytes(storage.parameters[bit.parameter])
    width = storage.codec.byte_width
    return raw[bit.flat_index * width:(bit.flat_index + 1) * width].hex()


def _compute_word(adapter: PinnedTransformerAdapter, bit: BitIdentity) -> str:
    tensor = dict(adapter.model.named_parameters())[bit.parameter]
    raw = TorchStoredLinearWeights._bytes(tensor)
    width = 4 if adapter.asset.computation_precision == "FP32" else 2
    return raw[bit.flat_index * width:(bit.flat_index + 1) * width].hex()
