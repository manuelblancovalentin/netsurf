"""Verified whole-network protection contexts for bounded mask exchanges."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

from .identities import BitIdentity
from .manifests import digest_object, sha256_file


class ProtectionContextError(ValueError):
    pass


@dataclass(frozen=True)
class ProtectionContextConfiguration:
    mode: str = "none"
    source_bundle: Path | None = None
    artifact: str = "artifacts/protection_masks.json"
    artifact_sha256: str = ""
    scientific_payload_sha256: str = ""
    mask_key: str = ""
    expected_population_size: int = 0
    expected_global_cardinality: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "source_bundle": str(self.source_bundle) if self.source_bundle else "",
            "artifact": self.artifact,
            "artifact_sha256": self.artifact_sha256,
            "scientific_payload_sha256": self.scientific_payload_sha256,
            "mask_key": self.mask_key,
            "expected_population_size": self.expected_population_size,
            "expected_global_cardinality": self.expected_global_cardinality,
        }


@dataclass(frozen=True)
class VerifiedProtectionContext:
    mode: str
    original: frozenset[BitIdentity]
    outside: frozenset[BitIdentity]
    original_inside_indices: tuple[int, ...]
    record: dict[str, Any]


def _identity(value: object) -> BitIdentity:
    if not isinstance(value, list) or len(value) != 3:
        raise ProtectionContextError("mask identities must be [parameter, flat_index, bit_index]")
    return BitIdentity(str(value[0]), int(value[1]), int(value[2]))


def _selected(document: object, key: str) -> object:
    value = document
    for component in key.split("/"):
        if not isinstance(value, dict) or component not in value:
            raise ProtectionContextError(f"mask key is absent from artifact: {key!r}")
        value = value[component]
    return value


def verify_protection_context(
    configuration: ProtectionContextConfiguration,
    universe: Sequence[BitIdentity],
    candidates: Sequence[BitIdentity],
    *, expected_checkpoint_sha256: str,
) -> VerifiedProtectionContext:
    universe_set = set(universe)
    candidate_set = set(candidates)
    if configuration.mode == "none":
        record = {
            "mode": "none",
            "identity_sha256": digest_object({"mode": "none"}),
            "eligible_population_size": len(universe),
            "global_cardinality": 0,
            "candidate_protected_count": 0,
            "original_mask": [],
            "outside_mask": [],
        }
        return VerifiedProtectionContext("none", frozenset(), frozenset(), (), record)
    if configuration.mode != "verified_global_mask":
        raise ProtectionContextError(f"unsupported protection-context mode: {configuration.mode!r}")
    root = configuration.source_bundle
    if root is None or not (root / "FINALIZED").is_file():
        raise ProtectionContextError(f"source bundle is not finalized: {root}")
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        checksums = json.loads((root / "checksums.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProtectionContextError(f"cannot read source bundle metadata: {error}") from error
    if manifest.get("status") != "complete":
        raise ProtectionContextError("source bundle status is not complete")
    if manifest.get("scientific_payload_sha256") != configuration.scientific_payload_sha256:
        raise ProtectionContextError("source scientific payload hash differs from configuration")
    if manifest.get("benchmark", {}).get("checkpoint_sha256") != expected_checkpoint_sha256:
        raise ProtectionContextError("source mask checkpoint differs from current benchmark")
    finalized = (root / "FINALIZED").read_text(encoding="ascii").strip()
    if finalized != f"scientific_payload_sha256={configuration.scientific_payload_sha256}":
        raise ProtectionContextError("source FINALIZED marker differs from configured payload")
    relative_artifact = Path(configuration.artifact)
    if relative_artifact.is_absolute() or ".." in relative_artifact.parts:
        raise ProtectionContextError("protection-mask artifact must be a safe relative path")
    expected = checksums.get("files", {}).get(configuration.artifact)
    if expected != configuration.artifact_sha256:
        raise ProtectionContextError("configured mask hash differs from finalized checksums")
    artifact_path = root / relative_artifact
    if sha256_file(artifact_path) != configuration.artifact_sha256:
        raise ProtectionContextError("source protection-mask artifact checksum mismatch")
    try:
        document = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProtectionContextError(f"cannot parse source protection-mask artifact: {error}") from error
    raw_mask = _selected(document, configuration.mask_key)
    if not isinstance(raw_mask, list):
        raise ProtectionContextError("selected mask record is not a list")
    identities = tuple(_identity(value) for value in raw_mask)
    original = frozenset(identities)
    if len(original) != len(identities):
        raise ProtectionContextError("source protection mask contains duplicate identities")
    if configuration.expected_population_size != len(universe):
        raise ProtectionContextError("eligible population size differs from configured source contract")
    if len(original) != configuration.expected_global_cardinality:
        raise ProtectionContextError("global mask cardinality differs from configured source contract")
    missing = original - universe_set
    if missing:
        raise ProtectionContextError(f"source mask contains ineligible bit: {min(missing).key()}")
    inside = tuple(index for index, bit in enumerate(candidates) if bit in original)
    if not inside or len(inside) == len(candidates):
        raise ProtectionContextError("candidate pool must contain protected and unprotected source bits")
    outside = original - candidate_set
    if len(outside) + len(inside) != len(original):
        raise ProtectionContextError("protection-context cardinality decomposition failed")
    identity_payload = {
        "mode": configuration.mode,
        "source_scientific_payload_sha256": configuration.scientific_payload_sha256,
        "artifact_sha256": configuration.artifact_sha256,
        "mask_key": configuration.mask_key,
        "original_mask_sha256": digest_object([value.key() for value in sorted(original)]),
        "outside_mask_sha256": digest_object([value.key() for value in sorted(outside)]),
        "candidate_keys": [value.key() for value in candidates],
        "original_inside_indices": list(inside),
    }
    record = {
        **configuration.to_dict(),
        "identity_sha256": digest_object(identity_payload),
        "eligible_population_size": len(universe),
        "global_cardinality": len(original),
        "candidate_protected_count": len(inside),
        "original_inside_indices": list(inside),
        "original_inside_bits": [candidates[index].key() for index in inside],
        "original_mask_sha256": identity_payload["original_mask_sha256"],
        "outside_mask_sha256": identity_payload["outside_mask_sha256"],
        "original_mask": [value.key() for value in sorted(original)],
        "outside_mask": [value.key() for value in sorted(outside)],
    }
    return VerifiedProtectionContext(configuration.mode, original, frozenset(outside), inside, record)
