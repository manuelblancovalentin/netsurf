#!/usr/bin/env python3
"""Run N0 (frozen before any outcome): exhaustive single-flip damage of every eligible stored bit of the pinned
ResNet-8 (77,360 INT8 kernel words, 618,880 bits), through the TFLite BUILTIN_REF executable.

For every bit: flip it alone, evaluate the 64 calibration images of Run A replication 2 (CIFAR-10 training images
2000-2063, the images the gradient rankings of Run M use), record accuracy and mean cross-entropy, restore.
These are the per-fault accuracies A(j) of Thales (Tyagi et al., HPCA 2023) with a uniform fault probability per
bit, and the single-flip loss changes HarDNN (Mahmoud et al.) aggregates per feature map.

Sharding is by code range; each shard writes `shard_XX.npz` once at the end (arrays `accuracy`, `cross_entropy`,
shape (codes_in_shard, 8), bit axis least to most significant) and a partial file every 2,000 flips that is
overwritten and is not a result.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fault_aware_resnet8 as fa  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--shards", type=int, required=True)
    a = ap.parse_args()
    target = a.out / f"shard_{a.shard:02d}.npz"
    assert not target.exists()
    adapter, _ = fa.setup()
    storage, values, names, starts, absj, codes = fa.flat(adapter)
    x, y, part = adapter.sample_partition("ranking", 64, offset=fa.OFFSET)
    assert part.indices[0] == fa.OFFSET and len(part.indices) == 64
    clean = adapter.evaluate(x, y)
    n = codes.size
    lo, hi = (a.shard * n) // a.shards, ((a.shard + 1) * n) // a.shards
    acc = np.full((hi - lo, 8), np.nan, dtype=np.float32); ce = np.full((hi - lo, 8), np.nan, dtype=np.float32)
    started, done = time.time(), 0
    for bit in range(7, -1, -1):                                # most significant bits first
        for k in range(lo, hi):
            fc = codes.copy(); fc[k] ^= np.uint8(1 << bit)
            with adapter.temporary_values(fa.to_values(fc, values, names, starts)):
                r = adapter.evaluate(x, y)
            acc[k - lo, bit] = r.accuracy; ce[k - lo, bit] = r.categorical_crossentropy
            done += 1
            if done % 2000 == 0:
                np.savez(a.out / f"partial_{a.shard:02d}.npz", accuracy=acc, cross_entropy=ce, lo=lo, hi=hi)
                print(json.dumps(dict(shard=a.shard, done=done, total=8 * (hi - lo), seconds=round(time.time() - started))), flush=True)
    np.savez(target, accuracy=acc, cross_entropy=ce, lo=lo, hi=hi, clean_accuracy=clean.accuracy,
             clean_cross_entropy=clean.categorical_crossentropy)
    (a.out / f"partial_{a.shard:02d}.npz").unlink(missing_ok=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
