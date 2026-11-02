"""Read-only source-to-Kona readiness audit, independent of model loading."""

from __future__ import annotations

import json
import os
from pathlib import Path

from .manifests import canonical_json_bytes, sha256_file


def audit_handoff(manifest_path: Path, portfolio_path: Path, output: Path) -> dict:
    manifest_path, portfolio_path, output = (Path(x).resolve() for x in
        (manifest_path, portfolio_path, output))
    document = json.loads(manifest_path.read_text())
    portfolio = json.loads(portfolio_path.read_text())
    if document.get("schema") != "kona-assets/netsurf-asset-readiness/1":
        raise ValueError("unsupported Kona readiness schema")
    records = document.get("targets")
    expected = portfolio.get("entries")
    if not isinstance(records, list) or not isinstance(expected, list) or len(records) != 18 or len(expected) != 18:
        raise ValueError("readiness/portfolio target count differs from 18")
    by_id = {record["target_id"]: record for record in records}
    if set(by_id) != {row["id"] for row in expected}:
        raise ValueError("readiness target identities differ from portfolio")
    rows = []
    for registry in expected:
        record = by_id[registry["id"]]
        model = record.get("model") or {}
        rows.append({"target": registry["id"], "asset_status": record["status"],
            "source_matches": model.get("repository") == registry["model"].get("source"),
            "revision_matches": model.get("commit", model.get("revision")) == registry["model"].get("revision"),
            "blockers": record.get("blockers", []),
            "interpreter_exists": bool(record.get("interpreter")) and Path(record["interpreter"]).is_file()})
    payload = {"schema_version": "transformer-handoff-audit/v1",
        "readiness_sha256": sha256_file(manifest_path),
        "portfolio_sha256": sha256_file(portfolio_path), "targets": rows}
    if output.exists():
        existing = json.loads(output.read_text())
        if existing != payload:
            raise ValueError("existing handoff audit differs; write-once output")
        return existing
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_json_bytes(payload) + b"\n")
        stream.flush(); os.fsync(stream.fileno())
    return payload
