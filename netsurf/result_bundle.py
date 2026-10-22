"""Atomic, immutable scientific result-bundle creation and reading."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import uuid
from typing import Any, Iterable, Mapping

from . import __version__
from .experiment_config import MANIFEST_SCHEMA_VERSION, MEASUREMENT_SCHEMA_VERSION, MODERN_MEASUREMENT_SCHEMA_VERSION, SUMMARY_SCHEMA_VERSION
from .experiment_engine import ExperimentOutcome, TrialMeasurement
from .manifests import canonical_json_bytes, git_provenance, runtime_provenance, sha256_file
from .plotting import plot_protection_results
from .reporting import render_report


class ResultBundleError(RuntimeError):
    pass


@dataclass(frozen=True)
class ResultBundle:
    path: Path
    manifest: Mapping[str, Any]
    summary: Mapping[str, Any]
    resolved_config: Mapping[str, Any]

    @property
    def scientific_payload_sha256(self) -> str:
        return str(self.summary["scientific_payload_sha256"])

    def trials(self) -> tuple[TrialMeasurement, ...]:
        with (self.path / "measurements/trials.csv").open(newline="", encoding="utf-8") as stream:
            return tuple(TrialMeasurement(
                int(row["trial"]), row["policy"], row["schedule_sha256"], float(row["accuracy"]),
                float(row["categorical_crossentropy"]), row["schedule_hash"], int(row["proposed_events"]),
                int(row["applied_events"]), int(row["suppressed_events"]), int(row["unique_applied_bits"]),
            ) for row in csv.DictReader(stream))


def _write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def _write_json(path: Path, value: Any) -> None:
    _write_bytes(path, canonical_json_bytes(value) + b"\n")


def _write_csv(path: Path, fields: list[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())


def _inventory(root: Path) -> list[dict[str, Any]]:
    return [{"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(root.rglob("*")) if path.is_file()]


def _summary(outcome: ExperimentOutcome) -> dict[str, Any]:
    payload = outcome.scientific_payload
    result = {"schema_version": SUMMARY_SCHEMA_VERSION, "scientific_protocol": outcome.configuration.scientific_protocol,
            "experiment_name": outcome.configuration.name, "benchmark_id": outcome.configuration.benchmark.id,
            "trial_count": outcome.configuration.execution.trials, "baseline": payload["baseline"],
            "protection_results": payload["policies"], "checks": payload["checks"],
            "scientific_payload_sha256": outcome.scientific_payload_sha256}
    if outcome.estimator_runtimes_seconds:
        result["estimator_runtimes_seconds"] = dict(outcome.estimator_runtimes_seconds)
    if "paired_comparisons" in payload:
        result["paired_comparisons"] = payload["paired_comparisons"]
    return result


def create_result_bundle(outcome: ExperimentOutcome, *, repository_root: Path, run_id: str | None = None) -> Path:
    """Serialize an outcome through staging and atomically finalize it."""
    config = outcome.configuration
    now = datetime.now(timezone.utc)
    identifier = run_id or f"{now.strftime('%Y%m%dT%H%M%S.%fZ')}-{uuid.uuid4().hex[:8]}"
    experiment_root = config.output.root / config.name
    destination, staging = experiment_root / identifier, experiment_root / f".{identifier}.incomplete"
    experiment_root.mkdir(parents=True, exist_ok=True)
    if destination.exists() or staging.exists():
        raise ResultBundleError(f"run directory already exists: {destination}")
    staging.mkdir()
    _write_bytes(staging / "config.toml", config.source_text.encode("utf-8"))
    resolved = config.resolved_document()
    _write_json(staging / "resolved_config.json", resolved)
    _write_json(staging / "scientific_payload.json", outcome.scientific_payload)
    summary = _summary(outcome)
    measurement_version = (
        MODERN_MEASUREMENT_SCHEMA_VERSION
        if config.schema_version == "2.0.0"
        else MEASUREMENT_SCHEMA_VERSION
    )
    _write_json(staging / "summary.json", summary)
    trial_fields = ["schema_version", "trial", "policy", "schedule_sha256", "accuracy", "categorical_crossentropy", "schedule_hash", "proposed_events", "applied_events", "suppressed_events", "unique_applied_bits"]
    _write_csv(staging / "measurements/trials.csv", trial_fields,
               ({"schema_version": measurement_version, **row.to_dict()} for row in outcome.trials))
    _write_csv(staging / "measurements/estimator_scores.csv",
               ["schema_version", "estimator", "parameter", "flat_index", "bit_index", "score"],
               ({"schema_version": measurement_version, "estimator": estimator, "parameter": row[0], "flat_index": row[1], "bit_index": row[2], "score": row[3]}
                for estimator, rows in outcome.rankings.items() for row in rows))
    policy_rows = []
    for name, value in summary["protection_results"].items():
        accuracy, loss = value["accuracy"], value["categorical_crossentropy"]
        policy_rows.append({"schema_version": measurement_version, "policy": name, "protected_bits": value["protected_bits"], "coverage": value["coverage"],
            "accuracy_mean": accuracy["mean"], "accuracy_ci_low": accuracy["confidence_interval"][0], "accuracy_ci_high": accuracy["confidence_interval"][1],
            "crossentropy_mean": loss["mean"], "crossentropy_ci_low": loss["confidence_interval"][0], "crossentropy_ci_high": loss["confidence_interval"][1], "independent_trials": accuracy["count"]})
    _write_csv(staging / "measurements/protection_results.csv", list(policy_rows[0]), policy_rows)
    if outcome.estimator_runtimes_seconds:
        _write_csv(
            staging / "measurements/estimator_runtime.csv",
            ["schema_version", "estimator", "runtime_seconds"],
            ({"schema_version": measurement_version, "estimator": name, "runtime_seconds": value}
             for name, value in outcome.estimator_runtimes_seconds.items()),
        )
    _write_json(staging / "artifacts/protection_masks.json", outcome.protection_masks)
    _write_json(staging / "artifacts/fault_schedules.json", outcome.fault_schedules)
    figures: tuple[str, ...] = ()
    if config.output.plots:
        figures = tuple(path.relative_to(staging).as_posix() for path in plot_protection_results(summary, staging / "plots"))
    if config.output.report:
        _write_bytes(staging / "report.md", render_report(summary, resolved, figure_paths=figures).encode("utf-8"))
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION, "run_id": identifier, "status": "complete",
        "source": git_provenance(Path(repository_root).resolve()), "runtime": runtime_provenance(),
        "versions": {"netsurf": __version__, "experiment_schema": config.schema_version, "manifest_schema": MANIFEST_SCHEMA_VERSION,
                     "summary_schema": SUMMARY_SCHEMA_VERSION, "measurement_schema": measurement_version,
                     "scientific_protocol": config.scientific_protocol,
                     "estimators": {value.result_name: value.implementation_version for value in config.estimators}},
        "benchmark": {**resolved["benchmark"], "checkpoint_sha256": sha256_file(config.benchmark.checkpoint), "dataset_sha256": sha256_file(config.benchmark.dataset)},
        "experiment": {**resolved["experiment"], "requested_configuration": "config.toml", "resolved_configuration": "resolved_config.json"},
        "execution": {**resolved["execution"], "fault": resolved["fault"], "protection": resolved["protection"], "estimators": resolved["estimators"],
                      "started_utc": now.isoformat(), "completed_utc": datetime.now(timezone.utc).isoformat()},
        "outputs": _inventory(staging), "scientific_payload_sha256": outcome.scientific_payload_sha256,
    }
    _write_json(staging / "manifest.json", manifest)
    _write_json(staging / "checksums.json", {"schema_version": "1.0.0", "files": {entry["path"]: entry["sha256"] for entry in _inventory(staging)}})
    _write_bytes(staging / "FINALIZED", f"scientific_payload_sha256={outcome.scientific_payload_sha256}\n".encode("ascii"))
    staging.rename(destination)
    return destination


def read_result_bundle(path: str | Path, *, verify: bool = True) -> ResultBundle:
    root = Path(path).expanduser().resolve()
    if not (root / "FINALIZED").is_file():
        raise ResultBundleError(f"not a finalized result bundle: {root}")
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        resolved = json.loads((root / "resolved_config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ResultBundleError(f"cannot read result bundle {root}: {error}") from error
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ResultBundleError(f"unsupported manifest schema_version: {manifest.get('schema_version')!r}")
    if summary.get("schema_version") != SUMMARY_SCHEMA_VERSION:
        raise ResultBundleError(f"unsupported summary schema_version: {summary.get('schema_version')!r}")
    if verify:
        checksums = json.loads((root / "checksums.json").read_text(encoding="utf-8"))
        for relative, expected in checksums["files"].items():
            target = root / relative
            if not target.is_file():
                raise ResultBundleError(f"missing bundled artifact: {relative}")
            actual = sha256_file(target)
            if actual != expected:
                raise ResultBundleError(f"checksum mismatch for {relative}: expected {expected}, got {actual}")
    return ResultBundle(root, manifest, summary, resolved)
