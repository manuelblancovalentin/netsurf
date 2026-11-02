"""Fail-closed consumer for the explicit Kona transformer readiness handoff.

This module never acquires assets.  An on-disk path alone is not a READY record.
The v1 consumer schema is intentionally explicit so an incompatible handoff is
blocked until its mapping can be reviewed, instead of guessing asset meaning.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

from .manifests import digest_object, sha256_file


class AssetBlocked(ValueError):
    pass


_SHA = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ReadyTransformerAsset:
    target: str
    model_root: Path
    dataset_path: Path
    interpreter: Path
    precision: str
    revision: str
    manifest_sha256: str
    record_sha256: str
    model_files: tuple[dict, ...]
    tokenizer_files: tuple[dict, ...]
    dataset_sha256: str
    interpreter_sha256: str
    labels: dict | None
    backend: dict
    record: dict


def _verified_file(root: Path, item: dict, kind: str) -> None:
    relative = item.get("path")
    expected = item.get("sha256")
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise AssetBlocked(f"{kind}: invalid relative path")
    if not isinstance(expected, str) or not _SHA.fullmatch(expected):
        raise AssetBlocked(f"{kind}: missing SHA-256 for {relative}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise AssetBlocked(f"{kind}: missing or escaping file {relative}")
    if sha256_file(path) != expected:
        raise AssetBlocked(f"{kind}: changed bytes for {relative}")


def load_ready_asset(manifest_path: Path, target: str, portfolio_path: Path) -> ReadyTransformerAsset:
    path = Path(manifest_path).resolve()
    if not path.is_file():
        raise AssetBlocked(f"NETSURF_ASSET_READINESS.json is absent: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        portfolio = json.loads(Path(portfolio_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AssetBlocked(f"cannot read asset manifest/portfolio: {error}") from error
    if document.get("schema") == "kona-assets/netsurf-asset-readiness/1":
        targets = document.get("targets")
        if not isinstance(targets, list) or len({r.get("target_id") for r in targets if isinstance(r, dict)}) != len(targets):
            raise AssetBlocked("Kona readiness targets missing or duplicate")
        source = next((r for r in targets if r.get("target_id") == target), None)
        if source is None:
            raise AssetBlocked(f"{target}: absent from Kona readiness manifest")
        if source.get("status") != "READY":
            raise AssetBlocked(f"{target}: {source.get('status')}; " + "; ".join(source.get("blockers", [])))
        # A READY status still needs the complete NetSuRF-specific mapping.
        # In particular, a planned interpreter path or an unlabeled dataset
        # description cannot be guessed into an executable source identity.
        model = source.get("model") or {}
        dataset = source.get("dataset") or {}
        backend = source.get("execution_backend") or {}
        record = {
            "target": target, "status": "READY", "source": model.get("repository"),
            "revision": model.get("revision"), "precision": source.get("target_precision"),
            "model_root": model.get("path"), "model_files": model.get("files"),
            "tokenizer_files": source.get("tokenizer"),
            "dataset_path": dataset.get("executable_path"),
            "dataset_sha256": dataset.get("executable_sha256"),
            "interpreter": source.get("interpreter"),
            "interpreter_sha256": source.get("interpreter_sha256"),
            "backend": backend, "labels": dataset.get("label_mapping"),
            "dataset_format": dataset.get("netsurf_format"),
            "partitions": source.get("splits"),
            "token_sequence_sha256": dataset.get("token_sequence_sha256"),
            "bos_mode": dataset.get("bos_mode"), "eos_mode": dataset.get("eos_mode"),
        }
    elif document.get("schema_version") == "netsurf-transformer-readiness/v1":
        records = document.get("records")
        if not isinstance(records, list) or len({r.get("target") for r in records if isinstance(r, dict)}) != len(records):
            raise AssetBlocked("readiness records missing or duplicate")
        record = next((r for r in records if r.get("target") == target), None)
    else:
        raise AssetBlocked("unsupported asset-readiness schema; review handoff mapping")
    registry = next((r for r in portfolio["entries"] if r["id"] == target), None)
    if record is None or registry is None or record.get("status") != "READY":
        raise AssetBlocked(f"{target}: no explicit READY record")
    if record.get("source") != registry["model"]["source"] or record.get("revision") != registry["model"]["revision"]:
        raise AssetBlocked(f"{target}: unpinned or changed source revision")
    if record.get("precision") != registry["target_precision"] or record["precision"] not in {"FP32", "BF16"}:
        raise AssetBlocked(f"{target}: unapproved stored precision")
    if not record.get("model_root") or not record.get("dataset_path") or not record.get("interpreter"):
        raise AssetBlocked(f"{target}: READY record lacks executable model, dataset or interpreter mapping")
    model_root = Path(record["model_root"]).resolve()
    files = record.get("model_files")
    tokens = record.get("tokenizer_files")
    if not isinstance(files, list) or not files or not isinstance(tokens, list) or not tokens:
        raise AssetBlocked(f"{target}: incomplete checkpoint or tokenizer inventory")
    for item in files:
        _verified_file(model_root, item, "model")
    for item in tokens:
        _verified_file(model_root, item, "tokenizer")
    for pin in registry["model"]["files"]:
        if "sha256" in pin and not any(x.get("path") == pin["path"] and x.get("sha256") == pin["sha256"] for x in files + tokens):
            raise AssetBlocked(f"{target}: checkpoint/tokenizer differs from portfolio pin: {pin['path']}")
    dataset_path = Path(record.get("dataset_path", "")).resolve()
    dataset_hash = record.get("dataset_sha256")
    if not dataset_path.is_file() or not isinstance(dataset_hash, str) or sha256_file(dataset_path) != dataset_hash:
        raise AssetBlocked(f"{target}: missing or changed dataset identity")
    interpreter = Path(record.get("interpreter", "")).resolve()
    interpreter_hash = record.get("interpreter_sha256")
    if not interpreter.is_file() or not isinstance(interpreter_hash, str) or sha256_file(interpreter) != interpreter_hash:
        raise AssetBlocked(f"{target}: incompatible or changed Python interpreter")
    backend = record.get("backend")
    if not isinstance(backend, dict) or backend.get("device") != "cpu" or backend.get("attention") != "eager" or backend.get("threads") != 1 or not all(backend.get(key) for key in ("python_version", "torch_version", "transformers_version")):
        raise AssetBlocked(f"{target}: unvalidated numerical/attention backend")
    labels = record.get("labels")
    if target == "distilbert_sst2_fp32" and (
        not isinstance(labels, dict) or
        {key: str(value).lower() for key, value in labels.items()} !=
        {"0": "negative", "1": "positive"}
    ):
        raise AssetBlocked("SST-2 public class-label mapping missing or changed")
    if target != "distilbert_sst2_fp32" and record.get("token_sequence_sha256") is None:
        raise AssetBlocked(f"{target}: exact token-ID sequence not pinned")
    return ReadyTransformerAsset(target, model_root, dataset_path, interpreter,
        record["precision"], record["revision"], sha256_file(path), digest_object(record),
        tuple(files), tuple(tokens), dataset_hash, interpreter_hash, labels, backend, record)


def verify_running_interpreter(asset: ReadyTransformerAsset) -> None:
    """Validate the already selected absolute interpreter and package versions."""
    import sys
    import platform
    from importlib.metadata import PackageNotFoundError, version

    if Path(sys.executable).resolve() != asset.interpreter:
        raise AssetBlocked(f"use pinned interpreter {asset.interpreter}; observed {sys.executable}")
    if platform.python_version() != asset.backend["python_version"]:
        raise AssetBlocked("Python version differs from READY backend")
    for package, key in (("torch", "torch_version"), ("transformers", "transformers_version")):
        try:
            actual = version(package)
        except PackageNotFoundError as error:
            raise AssetBlocked(f"{package} absent from pinned interpreter") from error
        if actual != asset.backend[key]:
            raise AssetBlocked(f"{package} version differs from READY backend: {actual}")
