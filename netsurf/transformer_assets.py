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
    dataset_files: tuple[dict, ...] = ()
    computation_precision: str = "FP32"


def _verified_file(root: Path, item: dict, kind: str) -> None:
    relative = item.get("path")
    expected = item.get("sha256")
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise AssetBlocked(f"{kind}: invalid relative path")
    if not isinstance(expected, str) or not _SHA.fullmatch(expected):
        raise AssetBlocked(f"{kind}: missing SHA-256 for {relative}")
    path = (root / relative).resolve()
    # Hugging Face snapshots contain symlinks into their sibling blobs/ tree.
    # Keep that repository boundary, then verify the pinned content hash.
    allowed_root = root.parent.parent if root.parent.name == "snapshots" else root
    if not path.is_relative_to(allowed_root.resolve()) or not path.is_file():
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
    registry = next((r for r in portfolio["entries"] if r["id"] == target), None)
    if registry is None:
        raise AssetBlocked(f"{target}: absent from reviewed NetSuRF registry")
    if document.get("schema") == "kona-assets/netsurf-asset-readiness/1":
        targets = document.get("targets")
        runtime = document.get("runtime_configurations", [])
        if (not isinstance(targets, list) or not isinstance(runtime, list) or
            len({r.get("target_id") for r in targets + runtime if isinstance(r, dict)}) !=
            len(targets) + len(runtime)):
            raise AssetBlocked("Kona readiness targets missing or duplicate")
        source = next((r for r in targets + runtime if r.get("target_id") == target), None)
        if source is None:
            raise AssetBlocked(f"{target}: absent from Kona readiness manifest")
        if source.get("status") != "READY":
            raise AssetBlocked(f"{target}: {source.get('status')}; " + "; ".join(source.get("blockers", [])))
        # A READY status still needs the complete NetSuRF-specific mapping.
        # In particular, a planned interpreter path or an unlabeled dataset
        # description cannot be guessed into an executable source identity.
        model = source.get("model") or {}
        dataset = source.get("dataset") or {}
        environment = source.get("environment") or {}
        reference = source.get("reference_results") or {}
        provenance = reference.get("provenance") or {}
        if target == "distilbert_sst2_fp32":
            if source.get("precision", {}).get("source") != "float32 as stored" or source.get("precision", {}).get("derived") is not None:
                raise AssetBlocked("DistilBERT READY stored FP32 representation changed")
            precision = "FP32"
            computation_precision = "FP32"
        else:
            stored = reference.get("stored_dtype")
            compute = reference.get("computation_dtype")
            if (not isinstance(stored, dict) or len(stored) != 1 or
                not all(isinstance(n, int) and n > 0 for n in stored.values()) or
                (stored.get("F32") is None and stored.get("BF16") is None) or
                compute not in {"float32", "bfloat16"}):
                raise AssetBlocked(f"{target}: stored/computation dtype is not pinned")
            precision = "FP32" if "F32" in stored else "BF16"
            computation_precision = "FP32" if compute == "float32" else "BF16"
            if (target == "pythia160m_bf16stored_fp32compute" and
                (precision, computation_precision) != ("BF16", "FP32")):
                raise AssetBlocked("Pythia BF16-storage/FP32-computation contract changed")
            if (target == "pythia160m_fp32stored_fp32compute" and
                (precision, computation_precision) != ("FP32", "FP32")):
                raise AssetBlocked("Pythia FP32-storage/FP32-computation contract changed")
            if target == "qwen3_0_6b_bf16" and (precision, computation_precision) != ("BF16", "BF16"):
                raise AssetBlocked("Qwen native BF16 contract changed")
        if (not reference.get("gate") or not all(reference["gate"].values()) or
            not reference.get("repeat_bitwise_identical") or
            reference.get("protocol") != dataset.get("protocol")):
            raise AssetBlocked(f"{target}: READY clean conformance gate missing")
        report_path = Path(reference.get("smoke_report", ""))
        if not report_path.is_file() or sha256_file(report_path) != reference.get("smoke_report_sha256"):
            raise AssetBlocked(f"{target}: clean reference report changed")
        clean_report = json.loads(report_path.read_text())
        lock_path = Path(environment.get("requirements_lock", ""))
        if not lock_path.is_file() or sha256_file(lock_path) != environment.get("requirements_lock_sha256"):
            raise AssetBlocked(f"{target}: environment lock changed")
        interpreter_path = Path(source.get("interpreter", ""))
        if not interpreter_path.is_file() or environment.get("interpreter") != str(interpreter_path):
            raise AssetBlocked(f"{target}: pinned interpreter missing or changed")
        backend = {"device": environment.get("device"), "attention": environment.get("attention"),
                   "threads": provenance.get("threads"), "python_version": environment.get("python"),
                   "torch_version": environment.get("torch"),
                   "transformers_version": environment.get("packages", {}).get("transformers")}
        if (provenance.get("attention") != backend["attention"] or
            provenance.get("python") != backend["python_version"] or
            provenance.get("torch") != backend["torch_version"] or
            provenance.get("transformers") != backend["transformers_version"] or
            provenance.get("cuda_available") is not False or
            provenance.get("deterministic_algorithms") is not True):
            raise AssetBlocked(f"{target}: clean backend provenance differs from READY environment")
        is_lm = dataset.get("protocol") == "kona-assets/lm-sliding-window/1"
        if is_lm:
            windowing = dataset.get("windowing") or {}
            pinned_dataset = registry.get("dataset") or {}
            report_data = clean_report.get("dataset") or {}
            if (windowing.get("context") != 1024 or windowing.get("stride") != 512 or
                "no special tokens" not in windowing.get("text", "") or
                not str(windowing.get("bos", "")).startswith("none;") or
                report_data.get("split") != "test" or
                not report_data.get("token_ids_sha256") or
                not report_data.get("text_sha256") or
                clean_report.get("computation_dtype") != compute or
                clean_report.get("model", {}).get("commit") != model.get("commit") or
                pinned_dataset.get("revision") != dataset.get("revision")):
                raise AssetBlocked(f"{target}: WikiText clean protocol or token identity changed")
            dataset_path = pinned_dataset.get("local_root")
            if not dataset_path or not isinstance(dataset_path, str):
                raise AssetBlocked(f"{target}: reviewed local WikiText root missing")
        else:
            dataset_path = dataset.get("path")
        record = {
            "target": target, "status": "READY", "source": model.get("repository"),
            "revision": model.get("commit"), "precision": precision,
            "model_root": model.get("path"), "model_files": model.get("files"),
            "tokenizer_files": source.get("tokenizer"),
            "dataset_path": dataset_path,
            "dataset_files": dataset.get("files"),
            "dataset_sha256": digest_object(dataset.get("files")),
            "interpreter": source.get("interpreter"),
            "interpreter_sha256": sha256_file(interpreter_path),
            "requirements_lock_sha256": environment["requirements_lock_sha256"],
            "clean_report_sha256": reference["smoke_report_sha256"],
            "backend": backend, "labels": dataset.get("label_mapping"),
            "dataset_format": ("wikitext2-raw-parquet/v1" if is_lm else
                "sst2-labeled-parquet/v1" if dataset.get("protocol") == "kona-assets/sst2-distilbert/1" else None),
            "partitions": source.get("splits"),
            "token_sequence_sha256": ({"test": report_data["token_ids_sha256"]} if is_lm else
                                      dataset.get("token_sequence_sha256")),
            "bos_mode": "none" if is_lm else dataset.get("bos_mode"),
            "eos_mode": "none" if is_lm else dataset.get("eos_mode"),
        }
        if is_lm:
            record["text_sha256"] = {"test": report_data["text_sha256"]}
            record["computation_precision"] = computation_precision
    elif document.get("schema_version") == "netsurf-transformer-readiness/v1":
        records = document.get("records")
        if not isinstance(records, list) or len({r.get("target") for r in records if isinstance(r, dict)}) != len(records):
            raise AssetBlocked("readiness records missing or duplicate")
        record = next((r for r in records if r.get("target") == target), None)
    else:
        raise AssetBlocked("unsupported asset-readiness schema; review handoff mapping")
    if record is None or registry is None or record.get("status") != "READY":
        raise AssetBlocked(f"{target}: no explicit READY record")
    if record.get("source") != registry["model"]["source"] or record.get("revision") != registry["model"]["revision"]:
        raise AssetBlocked(f"{target}: unpinned or changed source revision")
    if record.get("precision") != registry["target_precision"] or record["precision"] not in {"FP32", "BF16"}:
        raise AssetBlocked(f"{target}: unapproved stored precision")
    if record.get("computation_precision", record["precision"]) != registry.get(
        "computation_precision", record["precision"]):
        raise AssetBlocked(f"{target}: unapproved computation precision")
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
    dataset_files = record.get("dataset_files")
    if record.get("dataset_format") in {"sst2-labeled-parquet/v1", "wikitext2-raw-parquet/v1"}:
        if not dataset_path.is_dir() or not isinstance(dataset_files, list) or not dataset_files:
            raise AssetBlocked(f"{target}: labeled Parquet inventory missing")
        if digest_object(dataset_files) != dataset_hash:
            raise AssetBlocked(f"{target}: changed dataset inventory")
        for item in dataset_files:
            _verified_file(dataset_path, item, "dataset")
        for pin in (registry.get("dataset") or {}).get("files", []):
            if not any(x.get("path") == pin.get("path") and x.get("sha256") == pin.get("sha256")
                       for x in dataset_files):
                raise AssetBlocked(f"{target}: WikiText dataset differs from reviewed pin")
    elif not dataset_path.is_file() or not isinstance(dataset_hash, str) or sha256_file(dataset_path) != dataset_hash:
        raise AssetBlocked(f"{target}: missing or changed dataset identity")
    # Preserve the environment entrypoint symlink: resolving it to the base
    # Python binary loses that environment's site-packages when executed.
    interpreter = Path(record.get("interpreter", ""))
    if not interpreter.is_absolute():
        raise AssetBlocked(f"{target}: interpreter path is not absolute")
    interpreter_hash = record.get("interpreter_sha256")
    if not interpreter.is_file() or not isinstance(interpreter_hash, str) or sha256_file(interpreter) != interpreter_hash:
        raise AssetBlocked(f"{target}: incompatible or changed Python interpreter")
    backend = record.get("backend")
    if not isinstance(backend, dict) or backend.get("device") != "cpu" or backend.get("attention") != "eager" or backend.get("threads") not in (1, 8, 16, 32) or not all(backend.get(key) for key in ("python_version", "torch_version", "transformers_version")):
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
        tuple(files), tuple(tokens), dataset_hash, interpreter_hash, labels, backend, record,
        tuple(dataset_files or ()), record.get("computation_precision", record["precision"]))


def verify_running_interpreter(asset: ReadyTransformerAsset) -> None:
    """Validate the already selected absolute interpreter and package versions."""
    import sys
    import platform
    from importlib.metadata import PackageNotFoundError, version

    if Path(sys.executable).resolve() != asset.interpreter.resolve():
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
