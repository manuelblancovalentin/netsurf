"""Export a verified follow-up source bundle without changing its finalized files."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from netsurf.manifests import sha256_file
from netsurf.reliability_database import _verified_bundle


COMPACT_OMIT = frozenset({"measurements/estimator_predictions.csv"})


def export_source_bundle(source: dict, destination: Path) -> dict:
    root, _, _, checks = _verified_bundle(source)
    destination = destination.resolve()
    if not destination.is_dir():
        raise ValueError("database export directory must exist before source export")
    receipt_path = destination / "source_export_receipt.json"
    if receipt_path.exists():
        existing = json.loads(receipt_path.read_text())
        if (existing["source_scientific_payload_sha256"] != source["scientific_payload_sha256"]
                or existing["source_checksums_sha256"] != source["checksums_sha256"]):
            raise ValueError("existing source export belongs to a different finalized bundle")
        for record in existing["archives"].values():
            archive = Path(record["path"])
            if not archive.is_file() or sha256_file(archive) != record["sha256"]:
                raise ValueError(f"source archive changed: {archive}")
        return existing
    if any((destination / f"source_{kind}.zip").exists() for kind in ("compact", "complete")):
        raise ValueError("source ZIP exists without a matching export receipt")
    archives = {}
    included = tuple(sorted(checks["files"])) + ("checksums.json", "FINALIZED")
    for kind in ("compact", "complete"):
        target = destination / f"source_{kind}.zip"
        with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output:
            for relative in included:
                if kind == "compact" and relative in COMPACT_OMIT:
                    continue
                output.write(root / relative, arcname=str(Path("source_bundle") / relative))
            if kind == "compact":
                output.writestr(
                    "source_bundle/COMPACT_ARCHIVE_NOTICE.txt",
                    "The estimator_predictions.csv table is omitted. This is not a complete "
                    "scientific bundle; use source_complete.zip for independent artifact "
                    "verification. The original finalized source bundle remains unchanged.\n",
                )
        with zipfile.ZipFile(target) as archive:
            if archive.testzip() is not None:
                raise ValueError(f"source archive failed integrity check: {target}")
        archives[kind] = {"path": str(target), "sha256": sha256_file(target),
                          "bytes": target.stat().st_size}
    receipt = {
        "schema_version": "1.0.0", "source_bundle": str(root),
        "source_scientific_payload_sha256": source["scientific_payload_sha256"],
        "source_checksums_sha256": source["checksums_sha256"],
        "compact_omitted_artifacts": sorted(COMPACT_OMIT & set(checks["files"])),
        "archives": archives,
    }
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    commands = destination / "mac_scp_commands.txt"
    current = commands.read_text()
    lines = [f"scp kona:{record['path']} ~/Downloads/" for record in archives.values()]
    commands.write_text(current.rstrip("\n") + "\n" + "\n".join(lines) + "\n")
    return receipt
