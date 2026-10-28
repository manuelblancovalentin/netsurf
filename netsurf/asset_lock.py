"""Standard-library reader for neutral Kona Assets JSON locks.

This module intentionally does not import :mod:`kona_assets`.  A lock is
portable data: its digest excludes timestamps and machine-local resolution,
while each resolved file is independently size- and SHA-256-verified here.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


_DIGEST_EXCLUDED = {"created_utc", "resolved", "digest"}


class AssetLockError(ValueError):
    """A neutral asset lock is malformed, modified, or no longer resolves."""


def _canonical(document: Mapping[str, Any]) -> bytes:
    body = {key: value for key, value in document.items() if key not in _DIGEST_EXCLUDED}
    return json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class VerifiedAssetLock:
    """Verified lock data and its machine-local resolved paths."""

    path: Path
    document: Mapping[str, Any]
    digest: str
    resolved_files: Mapping[tuple[str, str], Path]

    @classmethod
    def load(cls, path: Path, *, verify_files: bool = True) -> "VerifiedAssetLock":
        source = Path(path).expanduser().resolve()
        try:
            document = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise AssetLockError(f"cannot read asset lock {source}: {error}") from error
        if document.get("schema") != "kona-assets.lock" or document.get("schema_version") != "1.0":
            raise AssetLockError("expected kona-assets.lock schema_version 1.0")
        expected = "sha256:" + hashlib.sha256(_canonical(document)).hexdigest()
        if document.get("digest") != expected:
            raise AssetLockError(
                f"asset-lock digest mismatch: expected {expected}, observed {document.get('digest')!r}"
            )
        benchmark = document.get("benchmark")
        if not isinstance(benchmark, dict) or benchmark.get("id") != "mlperf-tiny/ic":
            raise AssetLockError("asset lock is not the MLPerf Tiny image-classification lock")
        mandatory = (
            "execution_backend",
            "reference_results",
            "sample_identities",
            "preprocessing",
            "verification",
        )
        missing = [name for name in mandatory if not benchmark.get(name)]
        if missing:
            raise AssetLockError(f"asset lock lacks mandatory benchmark sections: {missing}")

        recorded_paths = (document.get("resolved") or {}).get("file_paths") or {}
        resolved: dict[tuple[str, str], Path] = {}
        problems: list[str] = []
        for entry in document.get("files", []):
            asset, relative = entry.get("asset"), entry.get("path")
            if not isinstance(asset, str) or not isinstance(relative, str):
                problems.append("file entry lacks string asset/path")
                continue
            raw = recorded_paths.get(f"{asset}|{relative}")
            if not raw:
                problems.append(f"no resolved path for {asset}|{relative}")
                continue
            candidate = Path(raw).expanduser().resolve()
            resolved[(asset, relative)] = candidate
            if not verify_files:
                continue
            try:
                observed_size = candidate.stat().st_size
                if entry.get("size") is not None and observed_size != int(entry["size"]):
                    problems.append(f"size differs for {relative}")
                elif entry.get("sha256") and sha256_path(candidate) != entry["sha256"]:
                    problems.append(f"SHA-256 differs for {relative}")
            except OSError as error:
                problems.append(f"cannot verify {relative}: {error.__class__.__name__}")
        if problems:
            raise AssetLockError("asset-lock verification failed: " + "; ".join(problems))
        return cls(source, document, expected, resolved)

    @property
    def benchmark(self) -> Mapping[str, Any]:
        return self.document["benchmark"]

    def file(self, relative_path: str, *, asset_prefix: str | None = None) -> Path:
        matches = [
            path
            for (asset, relative), path in self.resolved_files.items()
            if relative == relative_path and (asset_prefix is None or asset.startswith(asset_prefix))
        ]
        if len(matches) != 1:
            raise AssetLockError(
                f"expected one resolved lock file for {relative_path!r}, observed {len(matches)}"
            )
        return matches[0]

    def file_record(self, relative_path: str) -> Mapping[str, Any]:
        matches = [entry for entry in self.document["files"] if entry["path"] == relative_path]
        if len(matches) != 1:
            raise AssetLockError(
                f"expected one lock record for {relative_path!r}, observed {len(matches)}"
            )
        return matches[0]

    def manifest(self) -> dict[str, Any]:
        reference = self.benchmark["reference_results"]
        backend = self.benchmark["execution_backend"]
        return {
            "schema": self.document["schema"],
            "schema_version": self.document["schema_version"],
            "digest": self.digest,
            "benchmark": self.benchmark["id"],
            "protocol_version": self.benchmark["protocol_version"],
            "verified_file_count": len(self.resolved_files),
            "reference_model_sha256": reference["model_sha256"],
            "preprocessing_id": reference["preprocessing_id"],
            "sample_index_sha256": reference["sample_index_sha256"],
            "reference_output_digests": dict(reference["output_digests"]),
            "reference_backend": {
                "runtime": backend["runtime"],
                "op_resolver": backend["op_resolver"],
                "delegates": backend["delegates"],
                "threads": backend["threads"],
            },
        }


@lru_cache(maxsize=8)
def load_verified_asset_lock(path: str) -> VerifiedAssetLock:
    """Verify an immutable published lock once per process."""

    return VerifiedAssetLock.load(Path(path), verify_files=True)
