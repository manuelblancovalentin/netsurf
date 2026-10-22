"""Explicit sample identities and disjoint-partition validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Sequence


@dataclass(frozen=True)
class SamplePartition:
    role: str
    source_split: str
    indices: tuple[int, ...]
    sample_sha256: str
    provenance: str

    def to_dict(self):
        return asdict(self)


def sample_identity_digest(source_split: str, indices: Sequence[int], labels) -> str:
    payload = {
        "source_split": source_split,
        "indices": [int(value) for value in indices],
        "labels": [int(value) for value in labels],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def require_disjoint(partitions: Sequence[SamplePartition]) -> None:
    seen: dict[tuple[str, int], str] = {}
    overlaps: list[str] = []
    for partition in partitions:
        for index in partition.indices:
            key = (partition.source_split, index)
            if key in seen:
                overlaps.append(f"{partition.role} overlaps {seen[key]} at {key[0]}[{key[1]}]")
            else:
                seen[key] = partition.role
    if overlaps:
        raise ValueError("disjoint sample protocol violated: " + "; ".join(overlaps[:10]))
