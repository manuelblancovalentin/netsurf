"""Fail-closed, separately named BF16-storage/FP32-computation handoff.

This consumer does not promote a standalone clean report into READY evidence.
The asset producer must publish the explicit configuration record described in
docs/PYTHIA_BF16_STORAGE.md. Existing native-BF16 and FP32 targets are unchanged.
"""
import json
from pathlib import Path

from .manifests import digest_object, sha256_file
from .transformer_assets import AssetBlocked, ReadyTransformerAsset, _verified_file

TARGET = "pythia160m_bf16_storage_fp32_compute"
CONTRACT = "bf16-storage-fp32-compute/v1"


def load_pythia_asset(manifest_path, pin_path):
    path = Path(manifest_path).resolve()
    document = json.loads(path.read_text())
    pin = json.loads(Path(pin_path).read_text())
    if document.get("schema") != "kona-assets/netsurf-asset-readiness/1":
        raise AssetBlocked("review the unsupported readiness schema before execution")
    matches = [r for r in document.get("targets", []) if r.get("target_id") == TARGET]
    if len(matches) != 1 or matches[0].get("status") != "READY":
        raise AssetBlocked(f"{TARGET}: a distinct finalized READY record is required; native BF16 is not a substitute")
    source = matches[0]
    execution = source.get("execution", {})
    if execution != {"contract": CONTRACT, "stored_precision": "BF16", "compute_precision": "FP32"}:
        raise AssetBlocked("READY record does not explicitly identify BF16 storage and FP32 computation")
    model, dataset = source.get("model", {}), source.get("dataset", {})
    reference, environment = source.get("reference_results", {}), source.get("environment", {})
    if (model.get("repository"), model.get("commit")) != (pin["repository"], pin["commit"]):
        raise AssetBlocked("derived Pythia checkpoint identity differs from the reviewed pin")
    root = Path(model.get("path", "")).resolve()
    files = model.get("files", [])
    if {r["path"]: r["sha256"] for r in files} != pin["files"]:
        raise AssetBlocked("derived checkpoint inventory changed")
    for item in files:
        _verified_file(root, item, "model")
    tokens = source.get("tokenizer", [])
    if {r["path"] for r in tokens} != {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"}:
        raise AssetBlocked("complete tokenizer inventory required")
    for item in tokens:
        _verified_file(root, item, "tokenizer")
    if source.get("derivation") != pin["derivation"]:
        raise AssetBlocked("FP32 parent and round-to-nearest-even derivation must be explicit")
    if (dataset.get("source"), dataset.get("revision"), dataset.get("config")) != (
            pin["dataset"]["source"], pin["dataset"]["revision"], "wikitext-2-raw-v1"):
        raise AssetBlocked("WikiText-2 source/configuration/revision changed")
    data_root = Path(dataset.get("path", "")).resolve()
    data_files = dataset.get("files", [])
    if {r["path"]: r["sha256"] for r in data_files} != pin["dataset"]["files"]:
        raise AssetBlocked("WikiText-2 inventory changed")
    for item in data_files:
        _verified_file(data_root, item, "dataset")
    report_path = Path(reference.get("smoke_report", ""))
    if not report_path.is_file() or sha256_file(report_path) != reference.get("smoke_report_sha256"):
        raise AssetBlocked("hash-bound final clean-reference report required")
    report = json.loads(report_path.read_text())
    if (report.get("computation_dtype") != "float32" or set(report.get("stored_dtype", {})) != {"BF16"}
            or report.get("repeat_bitwise_identical") is not True
            or report.get("protocol") != "kona-assets/lm-sliding-window/1"
            or report.get("model", {}).get("commit") != pin["commit"]
            or {r["path"]: r["sha256"] for r in report.get("model", {}).get("files", [])} != pin["files"]):
        raise AssetBlocked("clean report is not the exact BF16-storage/FP32-compute reference")
    if report.get("windowing", {}).get("context") != 1024 or report.get("windowing", {}).get("stride") != 512:
        raise AssetBlocked("reference context/stride changed")
    if report.get("windowing", {}).get("bos") != "none" or not report.get("per_target_nll_sha256"):
        raise AssetBlocked("reference token-loss identity/BOS convention missing")
    if not reference.get("gate") or not all(v is True for v in reference["gate"].values()):
        raise AssetBlocked("final task-quality gate has not passed")
    if reference.get("metrics") != report.get("metrics"):
        raise AssetBlocked("READY task metrics differ from final report")
    provenance = report["provenance"]
    if (provenance.get("cuda_available") is not False or provenance.get("deterministic_algorithms") is not True
            or provenance.get("attention") != "eager"):
        raise AssetBlocked("unvalidated computation backend")
    interpreter = Path(source.get("interpreter", ""))
    if not interpreter.is_absolute() or not interpreter.is_file() or environment.get("interpreter") != str(interpreter):
        raise AssetBlocked("pinned interpreter missing")
    lock = Path(environment.get("requirements_lock", ""))
    if not lock.is_file() or sha256_file(lock) != environment.get("requirements_lock_sha256"):
        raise AssetBlocked("environment lock missing/changed")
    for key in ("python", "torch"):
        if environment.get(key) != provenance.get(key):
            raise AssetBlocked(f"clean-reference {key} environment changed")
    if (environment.get("packages", {}).get("transformers") != provenance.get("transformers")
            or environment.get("device") != "cpu" or environment.get("attention") != "eager"
            or not isinstance(provenance.get("threads"), int) or provenance["threads"] < 1):
        raise AssetBlocked("pinned backend provenance incomplete")
    # The final test sequence is pinned by the clean reference. Other splits
    # must be tokenized from pinned files and their identities sealed in a plan.
    token_identity = report["dataset"]
    if token_identity.get("split") != "test" or not token_identity.get("token_ids_sha256"):
        raise AssetBlocked("final clean-reference token sequence is not pinned")
    backend = {"device": "cpu", "attention": "eager", "threads": provenance["threads"],
               "python_version": provenance["python"], "torch_version": provenance["torch"],
               "transformers_version": provenance["transformers"]}
    record = {"execution": execution, "dataset_format": "wikitext2-raw-parquet/v1",
              "bos_mode": "none", "eos_mode": "none",
              "token_sequence_sha256": {"test": token_identity["token_ids_sha256"]},
              "text_sha256": {"test": token_identity["text_sha256"]},
              "clean_report_sha256": reference["smoke_report_sha256"],
              "clean_report": report, "requirements_lock_sha256": environment["requirements_lock_sha256"],
              "producer_record_sha256": digest_object(source)}
    return ReadyTransformerAsset(TARGET, root, data_root, interpreter, "BF16", pin["commit"],
        sha256_file(path), digest_object(source), tuple(files), tuple(tokens), digest_object(data_files),
        sha256_file(interpreter), None, backend, record, tuple(data_files))
