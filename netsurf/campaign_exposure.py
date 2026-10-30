"""Metadata-only KWS exposure inventory; never reads new model inputs."""
from __future__ import annotations

import json
from pathlib import Path

from .manifests import digest_object, sha256_file

KWS_SHA = "aeea436800704fce17b17292e4412630ad856e9d777c044c64ef748a880bd0ae"
SPLIT = "speech_commands_tfds_test_interleaved_blocks16"
CANONICAL = Path("/projects/netsurf/manuelbv")


def known_roots():
    return [CANONICAL / "workspace/runs", *sorted(
        (CANONICAL / "workspace/implementation-worktrees").glob("*/workspace/runs")),
        *sorted((CANONICAL / "workspace/implementation-worktrees").glob("*/workspace/scratch"))]


def _identities(value):
    found = set()
    if isinstance(value, dict):
        if value.get("source_split") == SPLIT:
            found.update(int(i) for i in value.get("indices", ()))
        sid = value.get("sample_id", "")
        if isinstance(sid, str) and sid.startswith(SPLIT + ":"):
            found.add(int(sid.rsplit(":", 1)[1]))
        for child in value.values():
            found.update(_identities(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_identities(child))
    return found


def audit_exposure(roots=None, *, allowed_campaign=None):
    """Verify small scientific payloads, not multi-GB prediction inventories.

    Previously inventoried indices remain exposed even if an old run is moved.
    No assertion is possible about failed, manual, concurrent, or out-of-root work.
    """
    ledger = Path(__file__).parent / "data/kws_replication_exposure_v1.json"
    prior = json.loads(ledger.read_text())
    exposed = set(prior["exposed_test_indices"])
    records, seen = [], set()
    from .conditional_bundle import verify_conditional_bundle
    for root in known_roots() if roots is None else roots:
        for path in sorted(Path(root).rglob("manifest.json")):
            bundle = path.parent.resolve()
            if bundle in seen or not (bundle / "FINALIZED").is_file():
                continue
            seen.add(bundle)
            manifest = json.loads(path.read_text())
            if manifest.get("benchmark", {}).get("checkpoint_sha256") != KWS_SHA:
                continue
            checks = json.loads((bundle / "checksums.json").read_text())
            payload_file = bundle / "scientific_payload.json"
            if payload_file.stat().st_size > 128 * 1024**2:
                raise ValueError(f"exposure payload exceeds bounded audit size: {bundle}")
            if sha256_file(payload_file) != checks["files"]["scientific_payload.json"]:
                raise ValueError(f"exposure payload corrupted: {bundle}")
            payload = json.loads(payload_file.read_text())
            digest = digest_object(payload)
            if digest != manifest["scientific_payload_sha256"] or (bundle / "FINALIZED").read_text().strip() != f"scientific_payload_sha256={digest}":
                raise ValueError(f"exposure payload identity mismatch: {bundle}")
            intentional_corruption_fixture = (
                bundle.name == "copied-selection"
                and bundle.parent.name == "test_original_source_plan_is_v0"
                and "scratch" in bundle.parts
                and digest == "6acf2a45452432e64186f52c3ffa6dfd2f298765106c1282ee220ef5823f0ff7")
            # The named M957 regression copies the real selection bundle and
            # deliberately corrupts only its frozen mask. It performs no inference.
            # Keep its indices conservatively exposed, explicitly record its
            # invalid status, and never treat it as verified scientific evidence.
            if str(manifest.get("schema_version", "")).startswith("6.") and not intentional_corruption_fixture:
                try:
                    verify_conditional_bundle(bundle)
                except ValueError as error:
                    raise ValueError(f"exposure inventory failure at {bundle}: {error}") from error
            indices = sorted(_identities(payload) | set(manifest.get("m956_final_original_indices", ())))
            same = allowed_campaign is not None and payload.get("campaign_binding", {}).get("campaign_sha256") == allowed_campaign
            if not same:
                exposed.update(indices)
            records.append({"bundle": str(bundle), "scientific_payload_sha256": digest,
                            "checksums_sha256": sha256_file(bundle / "checksums.json"),
                            "original_test_indices": indices, "same_frozen_campaign": same,
                            "verification": "deliberately corrupted copied-selection fixture; payload only; conservatively exposed" if intentional_corruption_fixture else "payload verified; conditional inventory verified"})
    return {"schema_version": "1.0.0", "source_split": SPLIT,
            "prior_ledger_sha256": sha256_file(ledger), "runs": records,
            "exposed_test_indices": sorted(exposed),
            "limitations": "Fault-outcome exposure in known finalized evidence only; aggregate clean reference and label audit covered all clips. Failed/manual/concurrent/out-of-root evaluations require researcher disclosure."}


def check_exposure(audit, offset, count, purpose):
    requested = list(range(3048 + offset, 3048 + offset + count))
    if count < 1 or offset < 0 or requested[-1] >= 4890:
        raise ValueError("final window is outside locked KWS test indices")
    overlap = sorted(set(requested) & set(audit["exposed_test_indices"]))
    if purpose == "confirmation" and overlap:
        raise ValueError(f"prior exposure overlap: {overlap}")
    return {"ledger_sha256": digest_object(audit), "requested_original_indices": requested,
            "prior_overlap": overlap, "purpose": purpose, "limitations": audit["limitations"]}
