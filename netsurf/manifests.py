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
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "dirty": bool(run("status", "--porcelain")),
    }


def runtime_provenance() -> dict[str, str]:
    import numpy

    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": numpy.__version__,
    }
    try:
        import tensorflow

        result["tensorflow"] = tensorflow.__version__
    except ImportError:
        result["tensorflow"] = "not-installed"
    return result
