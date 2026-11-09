"""Write-once packed Bernoulli stress fields for whole-universe W8A8 pilots.

This deliberately separate engineering path makes BER 0.01/0.1 feasible without
materializing hundreds of millions of Python fault-coordinate objects.  One
byte contains the eight independent stored-bit XOR decisions for one INT8 code.
It does not change the production sparse-field engine or a frozen campaign.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import mmap
import os
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np

from .manifests import canonical_json_bytes, digest_object, sha256_file
from .w8a8_assets import INTERPRETER, load_ready_w8a8
from .w8a8_faults import W8A8FaultAdapter
from .w8a8_high_ber_pilot import _cohort, _risk, TARGETS


BERS = (0.01, 0.1)
FIELD_SEED_BASE = 1529501
CHUNK_CODES = 1_000_000
LAW_VERSION = "w8a8-packed-independent-bernoulli-pcg64-float64-threshold/v1"
POPCOUNT = np.asarray([i.bit_count() for i in range(256)], dtype=np.uint8)


def _write_json(path: Path, obj: dict) -> None:
    with path.open("xb") as out:
        out.write(canonical_json_bytes(obj) + b"\n")
        out.flush()
        os.fsync(out.fileno())


def packed_field(path: Path, bits: int, ber: float, seed: int,
                 *, chunk_codes: int = CHUNK_CODES) -> dict:
    """Sample and seal all independent Bernoulli bits in C-order byte layout."""
    if bits < 8 or bits % 8 or not 0 < ber < 1 or seed < 0 or chunk_codes < 1:
        raise ValueError("invalid dense Bernoulli field specification")
    if path.exists():
        raise FileExistsError("write-once dense field already exists")
    rng = np.random.default_rng(seed)
    digest = hashlib.sha256()
    proposed = 0
    codes = bits // 8
    fd, stage_name = tempfile.mkstemp(prefix=".packed-field-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as out:
            for start in range(0, codes, chunk_codes):
                count = min(chunk_codes, codes - start)
                # Axis 1 represents bit indices 0..7 of exactly one stored byte.
                draws = rng.random((count, 8), dtype=np.float64)
                mask = np.packbits(draws < ber, axis=1, bitorder="little").reshape(-1)
                raw = mask.tobytes()
                out.write(raw)
                digest.update(raw)
                proposed += int(POPCOUNT[mask].sum(dtype=np.uint64))
            out.flush()
            os.fsync(out.fileno())
        os.link(stage_name, path)  # write-once: never replace an existing field
    finally:
        if os.path.exists(stage_name):
            os.unlink(stage_name)
    return {"law_version": LAW_VERSION, "seed": seed, "ber": ber,
            "universe_bits": bits, "bytes": codes, "proposed": proposed,
            "expected_proposed": bits * ber, "sha256": digest.hexdigest(),
            "numpy_version": np.__version__, "bitorder": "little"}


def xor_packed_codes(codes: np.ndarray, packed_mask: np.ndarray) -> None:
    """XOR all bit decisions into live uint8/int8 stored codes, in place."""
    if codes.dtype not in (np.dtype("int8"), np.dtype("uint8")) or codes.ndim != 1:
        raise ValueError("expected flat stored INT8 codes")
    if packed_mask.dtype != np.uint8 or packed_mask.shape != codes.shape:
        raise ValueError("packed mask does not match stored-code shape")
    np.bitwise_xor(codes.view(np.uint8), packed_mask, out=codes.view(np.uint8))


@contextmanager
def injected_packed_unprotected(adapter: W8A8FaultAdapter, path: Path, receipt: dict):
    """Apply a saved exact packed XOR field and restore every touched byte."""
    if adapter.modules is None or adapter.baseline_hashes is None:
        raise RuntimeError("load the pinned W8A8 executable first")
    if (receipt.get("universe_bits") != adapter.universe.bits or
        path.stat().st_size != adapter.universe.codes or
        sha256_file(path) != receipt.get("sha256")):
        raise ValueError("packed field identity or population mismatch")
    from kona_assets import int8_torch as execution
    before = {name: execution.tensor_sha256(module.qweight)
              for name, module in adapter.modules.items()}
    if before != adapter.baseline_hashes:
        raise RuntimeError("stored bytes differ from clean baseline before dense XOR")
    done: list[tuple[np.ndarray, np.ndarray]] = []
    offset = 0
    try:
        with path.open("rb") as source:
            with mmap.mmap(source.fileno(), length=0, access=mmap.ACCESS_READ) as data:
                all_masks = np.frombuffer(data, dtype=np.uint8)
                current_mask = None
                try:
                    for entry in adapter.universe.entries:
                        module = adapter.modules[entry["module"]]
                        codes = module.qweight.view(__import__("torch").uint8).reshape(-1).numpy()
                        count = codes.size
                        for start in range(0, count, CHUNK_CODES):
                            n = min(CHUNK_CODES, count - start)
                            view = codes[start:start+n]
                            current_mask = all_masks[offset+start:offset+start+n]
                            xor_packed_codes(view, current_mask)
                            done.append((view, current_mask))
                        offset += count
                    if offset != adapter.universe.codes:
                        raise RuntimeError("packed XOR did not cover the full eligible universe")
                    yield {"proposed": receipt["proposed"], "applied": receipt["proposed"],
                           "suppressed": 0, "field_sha256": receipt["sha256"]}
                finally:
                    for view, mask in reversed(done):
                        xor_packed_codes(view, mask)
                    if done:
                        del mask
                    del done, current_mask, all_masks
    finally:
        after = {name: execution.tensor_sha256(module.qweight)
                 for name, module in adapter.modules.items()}
        if before != after or after != adapter.baseline_hashes:
            raise RuntimeError("dense XOR did not restore all original stored bytes")


def run(target: str, manifest_path: Path, output_dir: Path) -> dict:
    if target not in TARGETS or (output_dir / "FINALIZED").exists():
        raise ValueError("unsupported target or finalized output already exists")
    if Path(__import__("sys").executable).resolve() != INTERPRETER.resolve():
        raise ValueError(f"requires pinned W8A8 interpreter: {INTERPRETER}")
    asset = load_ready_w8a8(manifest_path, target)
    adapter = W8A8FaultAdapter(asset)
    adapter.load()
    cohort, identity, evaluate = _cohort(adapter)
    source_root = Path(__file__).resolve().parent.parent
    plan = {"schema": "netsurf-w8a8-dense-ber-exposure-pilot/v1",
            "status": "EXPLORATORY_CALIBRATION", "target": target,
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source_root,
                                                    text=True).strip(),
            "source_sha256": sha256_file(Path(__file__)),
            "sparse_pilot_source_sha256": sha256_file(Path(__import__("netsurf.w8a8_high_ber_pilot",
                                                            fromlist=["_cohort"]).__file__)),
            "target_record_sha256": asset.record_sha256,
            "ready_manifest_sha256": asset.manifest_sha256,
            "eligible_bits": adapter.universe.bits,
            "eligible_population_sha256": adapter.universe.digest,
            "cohort": identity, "bers": list(BERS),
            "field_seeds": [FIELD_SEED_BASE+i for i in range(len(BERS))],
            "fields_per_ber": 1, "law_version": LAW_VERSION,
            "numpy_version": np.__version__, "chunk_codes": CHUNK_CODES,
            "fault_lifetime": "persistent for entire cohort evaluation",
            "policies": ["unprotected"], "failure_penalty_nats": 30.0,
            "interpretation": "engineering saturation/feasibility pilot only; no ranking or significance claim"}
    plan["plan_sha256"] = digest_object(plan)
    if output_dir.exists():
        saved = json.loads((output_dir / "plan.json").read_text())
        if saved != plan:
            raise ValueError("dense pilot plan identity changed; refusing to resume")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
        _write_json(output_dir / "plan.json", plan)
    clean = _risk(evaluate(cohort))
    if (output_dir / "clean_start.json").exists():
        if json.loads((output_dir / "clean_start.json").read_text()) != clean:
            raise RuntimeError("clean reference changed on resume")
    else:
        _write_json(output_dir / "clean_start.json", clean)
    t0 = time.monotonic()
    for index, ber in enumerate(BERS):
        path = output_dir / f"ber_{index}.packed.u8"
        meta_path = output_dir / f"ber_{index}.field.json"
        result_path = output_dir / f"ber_{index}.result.json"
        if path.exists() or meta_path.exists():
            if not path.exists() or not meta_path.exists():
                raise RuntimeError("partial dense field requires inspection; not overwriting")
            field = json.loads(meta_path.read_text())
            if (field.get("law_version") != LAW_VERSION or field.get("seed") != FIELD_SEED_BASE+index or
                field.get("ber") != ber or field.get("universe_bits") != adapter.universe.bits or
                field.get("numpy_version") != np.__version__ or path.stat().st_size != field.get("bytes") or
                sha256_file(path) != field.get("sha256")):
                raise RuntimeError("saved dense field identity differs")
        else:
            field = packed_field(path, adapter.universe.bits, ber, FIELD_SEED_BASE+index)
            _write_json(meta_path, field)
        if result_path.exists():
            previous = json.loads(result_path.read_text())
            if previous.get("field_sha256") != field["sha256"]:
                raise RuntimeError("saved dense result field differs")
            print(json.dumps({"target": target, "ber": ber, "status": "VERIFIED_RESUME_NO_INFERENCE"}),
                  flush=True)
            continue
        started = time.monotonic()
        with injected_packed_unprotected(adapter, path, field) as counts:
            risk = _risk(evaluate(cohort))
        result = {"ber": ber, "field_seed": field["seed"], "field_sha256": field["sha256"],
                  "proposed": counts["proposed"], "applied": counts["applied"],
                  "suppressed": counts["suppressed"], "risk": risk,
                  "evaluation_seconds": time.monotonic()-started}
        _write_json(result_path, result)
        print(json.dumps({"target": target, "ber": ber, "proposed": result["proposed"],
                          "mean_nll": risk["mean_nll_nats"], "failed": risk["failed_units"],
                          "elapsed_s": round(result["evaluation_seconds"], 2)}), flush=True)
    clean_end = _risk(evaluate(cohort))
    if clean_end != clean:
        raise RuntimeError("clean risk changed after dense fields")
    _write_json(output_dir / "clean_end.json", clean_end)
    files = {p.name: sha256_file(p) for p in sorted(output_dir.iterdir()) if p.is_file()}
    summary = {"schema": "netsurf-w8a8-dense-ber-exposure-pilot-result/v1",
               "status": "FINALIZED_ENGINEERING", "target": target,
               "plan_sha256": plan["plan_sha256"], "clean_risk": clean,
               "fields": len(BERS), "elapsed_seconds": time.monotonic()-t0,
               "completed_utc": datetime.now(timezone.utc).isoformat(),
               "scientific_files_sha256": files}
    summary["payload_sha256"] = digest_object(summary)
    _write_json(output_dir / "manifest.json", summary)
    _write_json(output_dir / "FINALIZED", {"manifest_sha256": sha256_file(output_dir / "manifest.json")})
    return {"target": target, "fields": len(BERS), "output": str(output_dir),
            "manifest_sha256": sha256_file(output_dir / "manifest.json")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", choices=TARGETS)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.target, args.manifest, args.output_dir)), flush=True)


if __name__ == "__main__":
    main()
