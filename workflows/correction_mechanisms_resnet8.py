#!/usr/bin/env python3
"""Run L2 (frozen before any outcome): Run L repeated on the MLPerf Tiny ResNet-8 INT8 model with more fault
draws, through the authoritative TFLite BUILTIN_REF executable.

Same conditions as Run L (workflows/correction_mechanisms.py): unprotected; SEC-DED over 64/32/16/8 data bits
(8/7/6/5 check bits, one error per codeword repaired, two or more left as they are, check bits fault too,
miscorrection not modelled); row-checksum correction (one faulty weight per output channel of each kernel
repaired, checksum words fault-free); TMR of all sign bits (FORTUNE-style, 12.5% of bits, cost 25%); TMR of the
per-example absolute-gradient mask at 6.25/12.5/18.75/25% of bits (cost 2B), voted bit wrong with 3p^2 - 2p^3.

Model and data: the pinned ResNet-8, INT8 kernels only (618,880 bits, as in the main study); CIFAR-10 test
images 1000-1999.  The absolute-gradient masks are the engine's masks of Run A replication 1 (built on training
images 1000-1063), read from that finalized bundle.  Clean accuracy must equal Run A replication 1 (87.4%).
p in the study grid; 8 fault draws per p; every condition sees the same raw fault field within a draw.

Stages: `masks` extracts the masks once into a compact file; `run --ber-index i --draw d` evaluates all
conditions for one (p, draw) and writes one JSON.  Exploratory; one checkpoint.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ENGINE = Path("/projects/netsurf/manuelbv/workspace/implementation-worktrees/paper1-runA-tflite-confirm")
RUN_A = ENGINE / "workspace/runs/paper1_runA_resnet8_confirm_v1"
CHECKPOINT = Path("/data/models/models--mlcommons--tiny/blobs/3c002613d1b2475eb51dd78dfb85a546c8ae658dee71cf6ade43b022fe205415")
DATASET = Path("/data/datasets/cifar10/cifar-10-python.tar.gz")
LOCK = Path("/projects/kona-assets/manuelbv/locks/mlperf-tiny/ic.lock.json")
BERS = (1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
SECDED = {64: 8, 32: 7, 16: 6, 8: 5}
FRACTIONS = ("0.062500", "0.125000", "0.187500", "0.250000")
DRAWS, SEED, OFFSET, COUNT, CLEAN = 8, 20261118, 1000, 1000, 0.874
POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def extract_masks(out: Path) -> None:
    bundle = next(d for d in RUN_A.iterdir() if (d / "FINALIZED").is_file())
    data = json.loads((bundle / "artifacts/protection_masks.json").read_text())["replication_1"]
    masks = {}
    for frac in FRACTIONS:
        masks[frac] = data[f"per_example_absolute_global@{frac}"]
    del data
    np.save(out / "masks_replication_1.npy", np.array(json.dumps(masks), dtype=object), allow_pickle=True)
    print({k: len(v) for k, v in masks.items()}, flush=True)


def field(rng, n, p):
    return np.packbits(rng.random((n, 8)) < p, axis=1, bitorder="little")[:, 0]


def run(out: Path, ber_index: int, draw: int) -> None:
    target = out / f"ber_{ber_index}_draw_{draw}.json"
    assert not target.exists()
    sys.path.insert(0, str(ENGINE))
    from netsurf.adapters.mlperf_tiny import MlperfTinyImageClassificationAdapter
    adapter = MlperfTinyImageClassificationAdapter(CHECKPOINT, DATASET, asset_lock=LOCK, split="test", injectable_parameters=("kernels",))
    adapter.load()
    x, y, part = adapter.sample_partition("evaluation", COUNT, offset=OFFSET)
    assert part.indices[0] == OFFSET and len(part.indices) == COUNT
    clean = adapter.evaluate(x, y)
    assert abs(clean.accuracy - CLEAN) < 1e-12, clean.accuracy
    storage = adapter._require_loaded()
    values = storage.values()
    names = list(values)
    codes = {n: np.ascontiguousarray(values[n]).view(np.uint8).reshape(-1) for n in names}
    masks_json = json.loads(str(np.load(out / "masks_replication_1.npy", allow_pickle=True)))
    packed = {}
    for frac, items in masks_json.items():
        m = {n: np.zeros(codes[n].size, dtype=np.uint8) for n in names}
        for parameter, index, bit in items:
            m[parameter][index] |= np.uint8(1 << bit)
        packed[f"tmr_abs_{float(frac) * 100:g}"] = m
    packed["tmr_msb"] = {n: np.full(codes[n].size, 0x80, dtype=np.uint8) for n in names}
    total_bits = 8 * sum(c.size for c in codes.values())
    for name, m in packed.items():
        frac = sum(int(POP[v].sum()) for v in m.values()) / total_bits
        assert abs(frac - (0.125 if name == "tmr_msb" else float(name.split("_")[-1]) / 100)) < 1e-4, (name, frac)
    cost = {"unprotected": 0.0, "row_checksum": 0.01, "tmr_msb": 0.25, **{f"secded_{k}": r / k for k, r in SECDED.items()},
            **{f"tmr_abs_{float(f) * 100:g}": 2 * float(f) for f in FRACTIONS}}
    p = BERS[ber_index]; q = 3 * p * p - 2 * p ** 3
    rng = np.random.default_rng(SEED + 1000 * ber_index + draw)
    raw = {n: field(rng, codes[n].size, p) for n in names}
    voted = {n: field(rng, codes[n].size, q) for n in names}
    conditions = {"unprotected": raw}
    for k, r in SECDED.items():
        per = k // 8; out_ = {}
        for n in names:
            xb = raw[n]; pad = (-xb.size) % per
            e = POP[np.concatenate([xb, np.zeros(pad, dtype=np.uint8)])].reshape(-1, per).sum(axis=1, dtype=np.int32)
            e = e + rng.binomial(r, p, size=e.size)
            out_[n] = np.where(np.repeat(e >= 2, per)[: xb.size], xb, 0).astype(np.uint8)
        conditions[f"secded_{k}"] = out_
    out_ = {}
    for n in names:
        mat = raw[n].reshape(values[n].shape[0], -1); faulty = (mat != 0).sum(axis=1)
        out_[n] = np.where((faulty != 1)[:, None], mat, 0).astype(np.uint8).reshape(-1)
    conditions["row_checksum"] = out_
    for name, m in packed.items():
        conditions[name] = {n: (raw[n] & ~m[n]) | (voted[n] & m[n]) for n in names}
    rows = []
    for name, xbs in conditions.items():
        faulted = {n: (codes[n] ^ xbs[n]).view(np.int8).reshape(values[n].shape) for n in names}
        started = time.monotonic()
        with adapter.temporary_values(faulted):
            res = adapter.evaluate(x, y)
        rows.append(dict(condition=name, ber=p, draw=draw, storage_cost=cost[name], accuracy=res.accuracy,
                         cross_entropy=res.categorical_crossentropy,
                         faulty_bits_after_correction=int(sum(int(POP[v].sum()) for v in xbs.values())),
                         raw_faulty_bits=int(sum(int(POP[v].sum()) for v in raw.values())), seconds=time.monotonic() - started))
        print(json.dumps(rows[-1]), flush=True)
    with open(target, "x") as f:
        json.dump(dict(schema="paper1-runL2-correction-mechanisms-resnet8/1.0", status="EXPLORATORY", ber=p, draw=draw,
                       clean_accuracy=clean.accuracy, window=[OFFSET, OFFSET + COUNT - 1], rows=rows), f, indent=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["masks", "run"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ber-index", type=int); ap.add_argument("--draw", type=int)
    a = ap.parse_args()
    if a.stage == "masks":
        a.out.mkdir(parents=True, exist_ok=False); extract_masks(a.out)
    else:
        run(a.out, a.ber_index, a.draw)


if __name__ == "__main__":
    main()
