"""Write-once filesystem result store for finalized v2 experiments."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .manifests import canonical_json_bytes, digest_object


class ResultStore:
    def __init__(self, root: Path, run_id: str):
        self.path = Path(root).expanduser().resolve() / run_id
        self.path.mkdir(parents=True, exist_ok=False)
        self._finalized = False

    def _write_once(self, relative: str, payload: bytes) -> Path:
        if self._finalized:
            raise RuntimeError("result store is finalized")
        destination = self.path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        return destination

    def write_json(self, relative: str, value: Any) -> Path:
        return self._write_once(relative, canonical_json_bytes(value) + b"\n")

    def write_text(self, relative: str, value: str) -> Path:
        return self._write_once(relative, value.encode("utf-8"))

    def finalize(self, scientific_payload: Any) -> str:
        digest = digest_object(scientific_payload)
        self._write_once("FINALIZED", f"scientific_payload_sha256={digest}\n".encode("ascii"))
        self._finalized = True
        return digest
