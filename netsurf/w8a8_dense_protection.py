"""Exact packed-field suppression for sealed whole-qweight protection masks.

The packed fault byte and protection byte share the stored INT8 word index.
Applied bits are ``fault & ~protection``; XOR composes simultaneous flips
within a word. This changes neither the Bernoulli law nor the stored-code
fault semantics of the existing sparse campaign.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .manifests import digest_object, sha256_file
from .w8a8_dense_ber_pilot import CHUNK_CODES, POPCOUNT, xor_packed_codes
from .w8a8_faults import MSBMask, W8A8FaultAdapter
from .w8a8_rank import PackedMask


@dataclass(frozen=True)
class Protection:
    kind: str
    universe_bits: int
    budget: int
    digest: str
    path: Path | None = None
    file_sha256: str | None = None

    def __post_init__(self):
        if self.kind not in {"none", "full", "msb", "packed"}:
            raise ValueError("unknown exact packed protection kind")
        if self.universe_bits <= 0 or self.universe_bits % 8:
            raise ValueError("invalid INT8 protection universe")
        if not 0 <= self.budget <= self.universe_bits:
            raise ValueError("invalid protection cardinality")
        if self.kind == "none" and self.budget != 0:
            raise ValueError("empty protection budget must be zero")
        if self.kind == "full" and self.budget != self.universe_bits:
            raise ValueError("full protection budget differs from universe")
        if self.kind == "msb" and self.digest != MSBMask(self.universe_bits, self.budget).digest:
            raise ValueError("MSB policy identity changed")
        if self.kind == "packed":
            if self.path is None or self.file_sha256 is None:
                raise ValueError("packed policy requires sealed file identity")
            expected = PackedMask(self.path, self.universe_bits, self.budget, self.file_sha256).digest
            if self.digest != expected:
                raise ValueError("packed policy identity changed")


def msb_protection_bytes(start_code: int, count: int, codes: int, budget: int) -> np.ndarray:
    """Exact word-relative descending-bit-tier mask, then global word order."""
    if not (codes > 0 and 0 <= budget <= 8 * codes and 0 <= start_code <= codes
            and 0 <= count <= codes - start_code):
        raise ValueError("invalid MSB byte slice")
    complete_tiers, partial = divmod(budget, codes)
    full_byte = sum(1 << (7 - tier) for tier in range(complete_tiers))
    out = np.full(count, full_byte, dtype=np.uint8)
    if partial:
        stop = max(0, min(count, partial - start_code))
        out[:stop] |= np.uint8(1 << (7 - complete_tiers))
    return out


def protection_chunk(protection: Protection, start_code: int, count: int,
                     packed_bytes: np.ndarray | None = None) -> np.ndarray:
    codes = protection.universe_bits // 8
    if not 0 <= start_code <= codes or not 0 <= count <= codes - start_code:
        raise ValueError("protection slice outside universe")
    if protection.kind == "none":
        return np.zeros(count, dtype=np.uint8)
    if protection.kind == "full":
        return np.full(count, 255, dtype=np.uint8)
    if protection.kind == "msb":
        return msb_protection_bytes(start_code, count, codes, protection.budget)
    if packed_bytes is None or packed_bytes.dtype != np.uint8 or packed_bytes.size != codes:
        raise ValueError("sealed packed protection bytes unavailable")
    return packed_bytes[start_code:start_code + count]


def verify_protection(protection: Protection) -> dict:
    """Hash and cardinality audit performed before a policy joins a campaign."""
    if protection.kind != "packed":
        return {"kind": protection.kind, "budget": protection.budget,
                "mask_digest": protection.digest}
    assert protection.path is not None and protection.file_sha256 is not None
    if protection.path.stat().st_size != protection.universe_bits // 8:
        raise ValueError("sealed protection-mask byte count differs")
    if sha256_file(protection.path) != protection.file_sha256:
        raise ValueError("sealed protection-mask SHA-256 differs")
    count = 0
    with protection.path.open("rb") as source:
        while raw := source.read(CHUNK_CODES):
            count += int(POPCOUNT[np.frombuffer(raw, dtype=np.uint8)].sum(dtype=np.uint64))
    if count != protection.budget:
        raise ValueError("sealed protection-mask cardinality differs")
    return {"kind": "packed", "budget": count, "mask_digest": protection.digest,
            "file_sha256": protection.file_sha256}


def prepared_protection_bytes(protection: Protection) -> np.ndarray | None:
    """Read-only mmap of a prevalidated mask, shared by every paired field."""
    if protection.kind == "packed":
        assert protection.path is not None
        return np.memmap(protection.path, dtype=np.uint8, mode="r")
    return None


@contextmanager
def injected_packed(adapter: W8A8FaultAdapter, field_bytes: np.ndarray,
                    field_receipt: dict, protection: Protection,
                    packed_protection: np.ndarray | None = None):
    """Apply saved proposed field, suppress protected bits, restore exact bytes."""
    if adapter.modules is None or adapter.baseline_hashes is None:
        raise RuntimeError("load the pinned W8A8 executable first")
    if (field_receipt.get("universe_bits") != adapter.universe.bits or
        field_bytes.dtype != np.uint8 or field_bytes.size != adapter.universe.codes or
        protection.universe_bits != adapter.universe.bits):
        raise ValueError("field, mask or loaded model population differs")
    from kona_assets import int8_torch as execution
    import torch
    before = {name: execution.tensor_sha256(module.qweight)
              for name, module in adapter.modules.items()}
    if before != adapter.baseline_hashes:
        raise RuntimeError("stored bytes differ from clean baseline before XOR")
    done: list[tuple[np.ndarray, int, int]] = []
    offset = 0
    applied_count = 0
    try:
        for entry in adapter.universe.entries:
            codes = adapter.modules[entry["module"]].qweight.view(torch.uint8).reshape(-1).numpy()
            for local in range(0, codes.size, CHUNK_CODES):
                n = min(CHUNK_CODES, codes.size - local)
                start = offset + local
                protected = protection_chunk(protection, start, n, packed_protection)
                applied = np.bitwise_and(field_bytes[start:start+n], np.bitwise_not(protected))
                applied_count += int(POPCOUNT[applied].sum(dtype=np.uint64))
                view = codes[local:local+n]
                xor_packed_codes(view, applied)
                done.append((view, start, n))
            offset += codes.size
        if offset != adapter.universe.codes:
            raise RuntimeError("packed XOR did not cover complete eligible population")
        proposed = field_receipt["proposed"]
        if not 0 <= applied_count <= proposed:
            raise RuntimeError("applied fault count exceeds proposed field")
        yield {"field_sha256": field_receipt["sha256"],
               "mask_sha256": protection.digest, "proposed": proposed,
               "applied": applied_count, "suppressed": proposed - applied_count}
    finally:
        for view, start, n in reversed(done):
            protected = protection_chunk(protection, start, n, packed_protection)
            applied = np.bitwise_and(field_bytes[start:start+n], np.bitwise_not(protected))
            xor_packed_codes(view, applied)
        after = {name: execution.tensor_sha256(module.qweight)
                 for name, module in adapter.modules.items()}
        if before != after or after != adapter.baseline_hashes:
            raise RuntimeError("stored INT8 bytes were not restored after protected field")


def boundary_protections(bits: int) -> dict[str, Protection]:
    return {
        "unprotected": Protection("none", bits, 0,
            digest_object({"version": "w8a8-boundary-mask/v1", "bits": bits, "full": False})),
        "full_protection": Protection("full", bits, bits,
            digest_object({"version": "w8a8-boundary-mask/v1", "bits": bits, "full": True})),
    }
