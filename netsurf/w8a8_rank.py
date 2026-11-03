"""Bounded-memory exact top-budget masks over W8A8 code bits.

Scores are nonnegative finite float32 or seeded uint32 hash keys.  Three
streaming passes select the exact top K using score-key histograms and
canonical global-ordinal tie breaking.  Only compact packed masks are stored.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from .manifests import digest_object, sha256_file


ScoreChunks = Callable[[], Iterator[tuple[int, np.ndarray]]]


def positive_float_keys(values: np.ndarray) -> np.ndarray:
    scores = np.asarray(values, dtype=np.float32)
    if not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("rank scores must be finite nonnegative float32")
    return scores.view(np.uint32)


def random_hash_keys(start: int, count: int, seed: int) -> np.ndarray:
    """Seeded 32-bit SplitMix key per global bit; ties break by ordinal."""
    x = np.arange(start, start + count, dtype=np.uint64) + np.uint64(seed)
    x = x + np.uint64(0x9e3779b97f4a7c15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xbf58476d1ce4e5b9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94d049bb133111eb)
    x ^= x >> np.uint64(31)
    return (x >> np.uint64(32)).astype(np.uint32)


@dataclass(frozen=True)
class PackedMask:
    path: Path
    universe_bits: int
    budget: int
    sha256: str

    @property
    def digest(self) -> str:
        return digest_object({"version": "w8a8-packed-mask/v1", "bits": self.universe_bits,
                              "budget": self.budget, "file_sha256": self.sha256})

    def contains(self, ordinal: int) -> bool:
        if not 0 <= ordinal < self.universe_bits:
            raise ValueError("mask ordinal outside universe")
        with self.path.open("rb") as f:
            f.seek(ordinal // 8)
            return bool(f.read(1)[0] & (1 << (ordinal % 8)))


def _checked_chunks(chunks: ScoreChunks, universe_bits: int, dtype):
    expected = 0
    for start, keys in chunks():
        keys = np.asarray(keys)
        if (start != expected or keys.dtype != dtype or keys.ndim != 1 or
            keys.size == 0 or keys.size % 8):
            raise ValueError("ranking chunks must cover aligned ordinals exactly once")
        expected += keys.size
        yield start, keys
    if expected != universe_bits:
        raise ValueError("ranking chunks omit part of the eligible population")


def write_top_masks(chunks: ScoreChunks, universe_bits: int, budgets: tuple[int, ...],
                    directory: Path, method: str, *, key_dtype=np.uint32) -> dict[int, PackedMask]:
    """Exact top-K in three float-score or five random-key streaming passes.

    Peak extra memory is O(chunk_bits + 4 * 65536 + number_of_masks * chunk_bits/8).
    Disk is one byte per eight eligible bits per requested interior budget.
    An existing output directory is rejected to keep masks write-once.
    """
    if (universe_bits <= 0 or universe_bits % 8 or not budgets or len(set(budgets)) != len(budgets) or
        any(not 0 < k < universe_bits for k in budgets) or not method or Path(directory).exists()):
        raise ValueError("invalid or already existing compact-ranking destination")
    dtype = np.dtype(key_dtype)
    if dtype not in (np.dtype("uint32"), np.dtype("uint64")):
        raise ValueError("only nonnegative uint32/uint64 ranking keys are supported")
    segments = dtype.itemsize // 2
    prefixes = {k: 0 for k in budgets}
    remaining = {k: k for k in budgets}
    for segment in range(segments):
        shift = 16*(segments-segment-1)
        histograms = {prefix: np.zeros(65536, dtype=np.int64) for prefix in set(prefixes.values())}
        for _, keys in _checked_chunks(chunks, universe_bits, dtype):
            upper = None if segment == 0 else keys >> (shift+16)
            for prefix, histogram in histograms.items():
                selected = keys if upper is None else keys[upper == prefix]
                if selected.size:
                    bins = ((selected >> shift) & 65535).astype(np.int32)
                    histogram += np.bincount(bins, minlength=65536)
        for k in budgets:
            histogram = histograms[prefixes[k]]
            cumulative = np.cumsum(histogram[::-1])
            if remaining[k] < 1 or remaining[k] > int(cumulative[-1]):
                raise ValueError("score histogram lost ranking coordinates")
            digit = 65535 - int(np.searchsorted(cumulative, remaining[k], side="left"))
            remaining[k] -= int(histogram[digit+1:].sum())
            prefixes[k] = (prefixes[k] << 16) | digit
    thresholds = {k: (prefixes[k], remaining[k]) for k in budgets}
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=False)
    handles = {k: (output / f"{method}_b{k}.bits").open("xb") for k in budgets}
    digests = {k: hashlib.sha256() for k in budgets}
    ties = {k: 0 for k in budgets}
    counts = {k: 0 for k in budgets}
    try:
        for _, keys in _checked_chunks(chunks, universe_bits, dtype):
            for k in budgets:
                threshold, tie_limit = thresholds[k]
                greater = keys > dtype.type(threshold)
                equal = keys == dtype.type(threshold)
                tie_rank = np.cumsum(equal, dtype=np.int64) + ties[k]
                chosen = greater | (equal & (tie_rank <= tie_limit))
                ties[k] += int(equal.sum())
                counts[k] += int(chosen.sum())
                block = np.packbits(chosen, bitorder="little").tobytes()
                handles[k].write(block)
                digests[k].update(block)
    finally:
        for handle in handles.values():
            handle.close()
    if any(counts[k] != k or (output / f"{method}_b{k}.bits").stat().st_size != universe_bits // 8
           for k in budgets):
        raise ValueError("exact ranking cardinality or compact mask size changed")
    result = {k: PackedMask(output / f"{method}_b{k}.bits", universe_bits, k,
                            digests[k].hexdigest()) for k in budgets}
    (output / "manifest.json").write_text(json.dumps({
        "version": "w8a8-exact-streaming-topk/v1", "method": method,
        "universe_bits": universe_bits, "tie_break": "global-ordinal-ascending",
        "score_key": f"{dtype.name}-descending", "thresholds": {str(k): thresholds[k] for k in budgets},
        "masks": {str(k): {"path": result[k].path.name, "sha256": result[k].sha256,
                           "digest": result[k].digest, "bytes": universe_bits // 8} for k in budgets}},
        sort_keys=True, separators=(",", ":")) + "\n")
    return result


def numerical_jump_chunks(modules: dict, universe, *, codes_per_chunk: int = 262144):
    """Absolute physical decoded-value jump of the stored signed INT8 code."""
    if codes_per_chunk < 1:
        raise ValueError("positive code chunk required")
    for i, entry in enumerate(universe.entries):
        module = modules[entry["module"]]
        q = module.qweight.reshape(-1).detach().cpu().numpy()
        scale = module.scale.detach().cpu().numpy()
        width = entry["shape"][1]
        for start in range(0, q.size, codes_per_chunk):
            end = min(start + codes_per_chunk, q.size)
            old = q[start:end]
            unsigned = old.view(np.uint8)
            values = np.empty((end - start, 8), dtype=np.float32)
            for bit in range(8):
                altered = np.bitwise_xor(unsigned, np.uint8(1 << bit)).view(np.int8)
                delta = altered.astype(np.int16) - old.astype(np.int16)
                values[:, bit] = np.abs(delta.astype(np.float32) * scale[np.arange(start, end)//width])
            yield universe.offsets[i] + start * 8, positive_float_keys(values.reshape(-1))


def random_chunks(universe_bits: int, seed: int, *, bits_per_chunk: int = 2097152):
    if universe_bits <= 0 or universe_bits % 8 or bits_per_chunk <= 0 or bits_per_chunk % 8:
        raise ValueError("aligned random ranking universe and chunk required")
    rng = np.random.default_rng(seed)
    for start in range(0, universe_bits, bits_per_chunk):
        count = min(bits_per_chunk, universe_bits - start)
        yield start, rng.bit_generator.random_raw(count)


def gradient_score_chunks(modules: dict, universe, moments_dir: Path, method: str, *,
                          codes_per_chunk: int = 262144):
    """Rank exact INT8 XOR deltas using declared FP32 decoded-code surrogate moments.

    `mean_g` is d(loss)/d(decoded weight) averaged over independent calibration
    examples. `mean_abs_g` is the mean per-example absolute gradient.  The
    actual integer operator has no derivative; these are surrogate rankings.
    """
    if method not in {"global_first_order", "per_example_absolute", "per_example_positive"}:
        raise ValueError("unsupported decoded-code surrogate method")
    meta = json.loads((Path(moments_dir) / "manifest.json").read_text())
    if (meta.get("version") != "w8a8-decoded-code-surrogate-moments/v1" or
        meta.get("universe_sha256") != universe.digest):
        raise ValueError("surrogate moments belong to a different INT8 population")
    for i, entry in enumerate(universe.entries):
        name = entry["module"]
        item = meta["tensors"].get(name)
        count = int(np.prod(entry["shape"]))
        if not item or item.get("codes") != count:
            raise ValueError("missing exact tensor gradient moments")
        mean = np.memmap(Path(moments_dir) / item["mean_g"], dtype="<f4", mode="r", shape=(count,))
        mean_abs = np.memmap(Path(moments_dir) / item["mean_abs_g"], dtype="<f4", mode="r", shape=(count,))
        if sha256_file(Path(moments_dir) / item["mean_g"]) != item["mean_g_sha256"] or \
           sha256_file(Path(moments_dir) / item["mean_abs_g"]) != item["mean_abs_g_sha256"]:
            raise ValueError("surrogate gradient bytes changed")
        module = modules[name]
        q = module.qweight.reshape(-1).detach().cpu().numpy()
        scale = module.scale.detach().cpu().numpy()
        width = entry["shape"][1]
        for start in range(0, count, codes_per_chunk):
            end = min(start + codes_per_chunk, count)
            old = q[start:end]
            unsigned = old.view(np.uint8)
            row_scale = scale[np.arange(start, end) // width]
            g = np.asarray(mean[start:end])
            a = np.asarray(mean_abs[start:end])
            values = np.empty((end - start, 8), dtype=np.float32)
            for bit in range(8):
                changed = np.bitwise_xor(unsigned, np.uint8(1 << bit)).view(np.int8)
                delta = (changed.astype(np.int16) - old.astype(np.int16)).astype(np.float32) * row_scale
                if method == "global_first_order":
                    values[:, bit] = np.abs(g * delta)
                elif method == "per_example_absolute":
                    values[:, bit] = a * np.abs(delta)
                else:
                    pos = np.maximum((a + g) * .5, 0.)
                    neg = np.maximum((a - g) * .5, 0.)
                    values[:, bit] = np.where(delta >= 0, pos * delta, neg * -delta)
            yield universe.offsets[i] + 8 * start, positive_float_keys(values.reshape(-1))
