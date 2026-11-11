#!/usr/bin/env python3
"""Run L (EXPLORATORY; frozen before any outcome): selective triplication against blanket detect-and-correct
schemes for stored-weight faults on DistilBERT W8A8, each with its real failure modes and its storage cost.

Question (from the advisor meeting): at which fault rates does a protection mask do better than applying a
correcting scheme blindly to everything, and at what storage cost?

Fault model: independent persistent flips of every stored bit with probability p, redundant and check bits
included.  Conditions, all evaluated on the same raw fault field per draw:

  unprotected
  secded_K          K in {64, 32, 16, 8}: a single-error-correcting, double-error-detecting Hamming code over
                    every K data bits (8, 7, 6, 5 check bits).  A codeword with exactly one flipped bit
                    (data or check) is repaired; with two or more it is left as it is.  Miscorrection of
                    three or more errors is not modelled, which favours the code slightly.
                    Storage cost 12.5%, 21.9%, 37.5%, 62.5%.
  row_checksum      checksum-based correction in the ABFT style, applied to the stored matrix: two checksum
                    words per output row locate and repair one faulty weight in that row; a row with two or
                    more faulty weights is left as it is.  Checksum words are assumed fault-free.  Cost < 1%.
  tmr_abs_B         B in {6.25, 12.5, 18.75, 25}%: the sealed absolute-gradient mask, protected bits stored
                    three times with a majority vote, so a protected bit is wrong with probability
                    3p^2 - 2p^3; unprotected bits flip with p.  Storage cost 2B.
  tmr_msb           the most significant (sign) bit of every weight triplicated: FORTUNE-style selection,
                    12.5% of bits, cost 25%.

p in the study grid (1e-5 ... 1e-1), two draws, the study's frozen 512 sentences, exact float path of the
earlier prototypes (asserted equal to the integer executable on the clean model).  One model, one checkpoint.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import meanfield_drift_prototype as mf  # noqa: E402

BERS = (1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
SECDED = {64: 8, 32: 7, 16: 6, 8: 5}
FRACTIONS = (0.0625, 0.125, 0.1875, 0.25)
DRAWS, SEED = 2, 20261117
POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def field(rng, n, p):
    out = np.empty(n, dtype=np.uint8)
    for s in range(0, n, 1 << 21):
        e = min(s + (1 << 21), n)
        out[s:e] = np.packbits(rng.random((e - s, 8)) < p, axis=1, bitorder="little")[:, 0]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ber-index", type=int, required=True)
    args = ap.parse_args()
    p = BERS[args.ber_index]; q = 3 * p * p - 2 * p ** 3
    target = args.out / f"ber_{args.ber_index}.json"
    assert args.out.is_dir() and not target.exists()
    torch, plan, adapter, evaluate = mf.setup()
    modules, n_codes = mf.per_module_arrays(adapter)
    qmods = [adapter.modules[m["name"]] for m in modules]
    for qm in qmods:
        qm._mf_delta = None
    exact_clean = dict(evaluate().metrics)
    for qm, m in zip(qmods, modules):
        qm._mf_delta = torch.zeros(m["shape"], dtype=torch.float32)
    assert dict(evaluate().metrics) == exact_clean
    bits = plan["eligible_bits"]
    packed_masks = {"tmr_msb": [np.full(m["n"], 0x80, dtype=np.uint8) for m in modules]}
    for frac in FRACTIONS:
        pm = mf.mask_bits("per_example_absolute", int(round(frac * bits)), n_codes)
        packed = np.packbits(pm, axis=1, bitorder="little")[:, 0]
        packed_masks[f"tmr_abs_{100 * frac:g}"] = [packed[m["start"]: m["start"] + m["n"]] for m in modules]
    cost = {"unprotected": 0.0, "row_checksum": 0.01, "tmr_msb": 0.25, **{f"secded_{k}": r / k for k, r in SECDED.items()},
            **{f"tmr_abs_{100 * f:g}": 2 * f for f in FRACTIONS}}

    def run(xbs):
        applied = 0
        for qm, m, xb in zip(qmods, modules, xbs):
            flips = np.unpackbits(xb[:, None], axis=1, bitorder="little").astype(bool)
            applied += int(flips.sum())
            delta = np.where(flips, m["jumps"], 0).sum(axis=1, dtype=np.int32).astype(np.float32)
            qm._mf_delta = torch.from_numpy(delta.reshape(m["shape"]))
        started = time.monotonic()
        metrics = dict(evaluate().metrics)
        return metrics, applied, time.monotonic() - started

    rows = []
    for draw in range(DRAWS):
        rng = np.random.default_rng(SEED + 1000 * args.ber_index + draw)
        raw = [field(rng, m["n"], p) for m in modules]                 # every stored data bit flips with p
        voted = [field(rng, m["n"], q) for m in modules]               # a triplicated bit after the majority vote
        conditions = {"unprotected": raw}
        for k, r in SECDED.items():
            per = k // 8; out = []
            for xb in raw:
                pad = (-xb.size) % per
                e = POP[np.concatenate([xb, np.zeros(pad, dtype=np.uint8)])].reshape(-1, per).sum(axis=1, dtype=np.int32)
                e = e + rng.binomial(r, p, size=e.size)                 # flipped check bits of the same codeword
                keep = np.repeat(e >= 2, per)[: xb.size]
                out.append(np.where(keep, xb, 0).astype(np.uint8))
            conditions[f"secded_{k}"] = out
        out = []
        for xb, m in zip(raw, modules):
            mat = xb.reshape(m["shape"]); faulty = (mat != 0).sum(axis=1)
            out.append(np.where((faulty != 1)[:, None], mat, 0).astype(np.uint8).reshape(-1))
        conditions["row_checksum"] = out
        for name, pms in packed_masks.items():
            conditions[name] = [(xb & ~pm) | (vb & pm) for xb, vb, pm in zip(raw, voted, pms)]
        for name, xbs in conditions.items():
            metrics, applied, seconds = run(xbs)
            row = dict(condition=name, ber=p, draw=draw, storage_cost=cost[name], accuracy=metrics["accuracy"], mean_nll=metrics["mean_nll"],
                       faulty_bits_after_correction=applied, raw_faulty_bits=int(sum(int(POP[x].sum()) for x in raw)), seconds=seconds)
            rows.append(row)
            print(json.dumps(row), flush=True)
    with open(target, "x") as f:
        json.dump(dict(schema="paper1-runL-correction-mechanisms/1.0", status="EXPLORATORY", ber=p, clean=exact_clean, rows=rows), f, indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
