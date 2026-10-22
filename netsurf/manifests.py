"""Canonical serialization, artifact hashing, and run manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import subprocess
from typing import Any


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def digest_object(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def git_provenance(root: Path) -> dict[str, Any]:
    def run(*arguments: str, binary: bool = False):
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return result.stdout if binary else result.stdout.decode("utf-8").strip()

    diff = run("diff", "--binary", binary=True)
    status = run("status", "--porcelain=v1", "--untracked-files=all")
    listed = run("ls-files", "-z", "--cached", "--others", "--exclude-standard", binary=True)
    source_fingerprint = hashlib.sha256()
    for encoded in sorted(value for value in listed.split(b"\0") if value):
        relative = encoded.decode("utf-8", errors="surrogateescape")
        path = Path(root) / relative
        if not path.is_file() or path.is_symlink():
            continue
        source_fingerprint.update(encoded)
        source_fingerprint.update(b"\0")
        source_fingerprint.update(bytes.fromhex(sha256_file(path)))
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "source_fingerprint_sha256": source_fingerprint.hexdigest(),
        "dirty": bool(status),
    }


def runtime_provenance() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": version("numpy"),
    }
    for distribution in ("tensorflow", "keras", "HGQ2", "hls4ml"):
        try:
            result[distribution.lower()] = version(distribution)
        except PackageNotFoundError:
            result[distribution.lower()] = "not-installed"
    return result
