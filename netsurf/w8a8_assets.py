"""Read-only, target-specific consumer of Kona's pinned W8A8 READY records."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys

from .manifests import digest_object, sha256_file


TARGETS = frozenset({"distilbert_sst2_int8", "pythia160m_int8", "qwen3_0_6b_int8"})
SPEC = "kona-assets/int8-w8a8-perchannel-pertoken/1"
INTERPRETER = Path("/projects/kona-assets/manuelbv/envs/torch-2.9.1-cpu-py311/bin/python")
KONA_ROOT = Path("/projects/kona-assets/manuelbv")
KONA_COMMIT = "1aea52b"
KONA_SOURCE_SHA256 = {
    "int8_torch.py": "478772e11c508c84c360d1e2eef640d5d5cfa005b9090ae84e614f2c88d3aed3",
    "int8_w8a8.py": "92a6ba771f6e5f245de0e80f2e78b6c0a2d3014b0d7f018ae4b60e905debb9da",
}


class W8A8AssetBlocked(ValueError):
    pass


@dataclass(frozen=True)
class ReadyW8A8:
    target: str
    record: dict
    record_sha256: str
    manifest_sha256: str
    model_root: Path
    dataset_root: Path
    quantization: dict
    eligible_bits: int
    modules: int


def _verified(root: Path, files: list[dict], label: str) -> None:
    if not files or len({item.get("path") for item in files}) != len(files):
        raise W8A8AssetBlocked(f"{label}: absent or duplicate file inventory")
    for item in files:
        rel = item.get("path")
        if not isinstance(rel, str) or not rel or Path(rel).is_absolute() or ".." in Path(rel).parts:
            raise W8A8AssetBlocked(f"{label}: invalid relative file path")
        path = (root / rel).resolve()
        allowed = root.parent.parent if root.parent.name == "snapshots" else root
        if not path.is_relative_to(allowed.resolve()) or not path.is_file():
            raise W8A8AssetBlocked(f"{label}: missing or escaping {rel}")
        if path.stat().st_size != item.get("size") or sha256_file(path) != item.get("sha256"):
            raise W8A8AssetBlocked(f"{label}: changed size or SHA-256 for {rel}")


def load_ready_w8a8(manifest_path: Path, target: str, *, verify_files: bool = True,
                    require_interpreter: bool = True) -> ReadyW8A8:
    """Reject an unreviewed path even if checkpoint bytes happen to exist."""
    if target not in TARGETS:
        raise W8A8AssetBlocked(f"unsupported W8A8 target: {target}")
    path = Path(manifest_path)
    if not path.is_file():
        raise W8A8AssetBlocked("authoritative readiness manifest missing")
    doc = json.loads(path.read_text())
    if doc.get("schema") != "kona-assets/netsurf-asset-readiness/1" or doc.get("manifest_revision", 0) < 2:
        raise W8A8AssetBlocked("unreviewed Kona readiness schema or revision")
    records = doc.get("targets")
    if not isinstance(records, list) or len({r.get("target_id") for r in records}) != len(records):
        raise W8A8AssetBlocked("duplicate or invalid target records")
    record = next((r for r in records if r["target_id"] == target), None)
    if record is None or record.get("status") != "READY" or record.get("blockers"):
        raise W8A8AssetBlocked(f"{target}: no unblocked READY record")
    env, model, dataset = record["environment"], record["model"], record["dataset"]
    quant, eligible, ref = record["quantization"], record["eligible_stored_int8_population"], record["reference_results"]
    if (record.get("interpreter") != str(INTERPRETER) or env.get("interpreter") != str(INTERPRETER) or
        env.get("torch") != "2.9.1+cpu" or env.get("attention") != "eager" or
        env.get("device") != "cpu" or env.get("cuda_available") is not False or
        quant.get("spec") != SPEC or quant.get("operator", {}).get("op") != "torch._int_mm" or
        eligible.get("declared_faultable") != "qweight (int8 codes only)" or
        not ref.get("repeat_bitwise_identical") or not all(ref.get("gate", {}).values()) or
        not record.get("fault_mutation_receipt", {}).get("summary", {}).get("passed")):
        raise W8A8AssetBlocked(f"{target}: W8A8 runtime, precision or conformance gate changed")
    if require_interpreter and Path(sys.executable).resolve() != INTERPRETER.resolve():
        raise W8A8AssetBlocked(f"run with the pinned interpreter: {INTERPRETER}")
    if verify_files:
        commit = subprocess.run(["git", "-C", str(KONA_ROOT), "rev-parse", "--short", "HEAD"],
                                text=True, capture_output=True, check=True).stdout.strip()
        if commit != KONA_COMMIT or any(sha256_file(KONA_ROOT / "src/kona_assets" / name) != expected
                                       for name, expected in KONA_SOURCE_SHA256.items()):
            raise W8A8AssetBlocked("pinned Kona W8A8 executable source changed")
    root = Path(model["path"])
    qfile = next((x for x in model["files"] if x.get("path") == "quantization_manifest.json"), None)
    if qfile is None or qfile["sha256"] != model["artifact"]["manifest_sha256"]:
        raise W8A8AssetBlocked("quantization manifest hash missing or changed")
    q = json.loads((root / "quantization_manifest.json").read_text())
    if (q.get("spec") != SPEC or q.get("tensors_digest") != model["artifact"]["tensors_digest"] or
        digest_object(q.get("tensors")) != q.get("tensors_digest") or
        q.get("population", {}).get("bits") != eligible.get("bits") or
        q.get("population", {}).get("int8_codes") != eligible.get("int8_codes") or
        q.get("population", {}).get("modules") != eligible.get("modules") or
        eligible["bits"] != 8 * eligible["int8_codes"] or
        len({x["module"] for x in q["tensors"]}) != eligible["modules"] or
        any(x.get("faultable") != {"qweight": True, "scale": False} for x in q["tensors"])):
        raise W8A8AssetBlocked("eligible INT8 tensor universe differs from READY")
    if target == "qwen3_0_6b_int8":
        head = next((x for x in q["tensors"] if x["module"] == "lm_head"), None)
        if not head or "separately stored quantised copy" not in head.get("tied_to_input_embedding", ""):
            raise W8A8AssetBlocked("Qwen separately stored output head missing")
    if dataset.get("protocol") == "kona-assets/sst2-distilbert/1":
        dataset_root = Path(dataset["path"])
        if dataset.get("label_mapping") != {"0": "NEGATIVE", "1": "POSITIVE"}:
            raise W8A8AssetBlocked("SST-2 public label mapping changed")
    elif dataset.get("protocol") == "kona-assets/lm-sliding-window/1":
        dataset_root = Path("/data/datasets/wikitext2")
        if dataset.get("revision") != "b08601e04326c79dfdd32d625aee71d232d685c3":
            raise W8A8AssetBlocked("WikiText-2 revision changed")
    else:
        raise W8A8AssetBlocked("unreviewed task dataset protocol")
    if verify_files:
        _verified(root, model["files"], "checkpoint")
        _verified(root, record["tokenizer"], "tokenizer")
        _verified(dataset_root, dataset["files"], "dataset")
        for receipt in (ref, record["fault_mutation_receipt"]):
            rpath = Path(receipt.get("smoke_report") or receipt.get("report") or "")
            expected = receipt.get("smoke_report_sha256") or receipt.get("report_sha256")
            if not rpath.is_file() or sha256_file(rpath) != expected:
                raise W8A8AssetBlocked("clean or exact-fault conformance receipt changed")
        lock = Path(env["requirements_lock"])
        if not lock.is_file() or sha256_file(lock) != env["requirements_lock_sha256"]:
            raise W8A8AssetBlocked("pinned Python lock changed")
    return ReadyW8A8(target, record, digest_object(record), sha256_file(path), root,
                     dataset_root, q, eligible["bits"], eligible["modules"])
