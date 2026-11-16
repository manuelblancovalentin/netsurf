#!/usr/bin/env python3
"""Run T, stage T2/T3: the non-ideal protection comparison for the W8A8 language models (DistilBERT, Pythia-160M,
Qwen3-0.6B) with the exact INT8 executable.  Protocol: review/RUN_T_T1_PROTOCOL_2026-11-15.md, addendum T2.

Same fault model and conditions as runT_tflite.py (netsurf/nonideal_protection.py of this worktree, loaded by file
path because the W8A8 code has its own `netsurf` package): every stored bit, copies and check bits included, flips
with probability p; unprotected, SEC-DED over 64/32/16/8 data bits per tensor, ABFT per output row with fault-free and
with faulty checksums, and triplication of the top 6.25/12.5/18.75/25% of bits by each of the six main-grid rankings,
using the sealed masks of the frozen wide-BER campaign (msb_first, random, numerical_bit_jump, global_first_order,
per_example_absolute, per_example_positive).  Faults are XORed into the stored INT8 codes of the live executable and
removed after every evaluation; the clean model is re-checked by hash after the job.

One job = (target, draw, BER index).  Output (write-once): <out>/<target>/draw<D>/ber<I>/{results.csv, DONE}.
Interpreter: the pinned W8A8 environment (/projects/kona-assets/manuelbv/envs/torch-2.9.1-cpu-py311).

    runT_w8a8.py --target distilbert_sst2_int8 --draw 0 --ber-index 3 [--out DIR] [--threads 4]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
WT = HERE.parents[2]
W8A8 = WT / "w8a8-ber-stress"
CAMPAIGN = W8A8 / "workspace/runs/w8a8_wide_ber_matched_v1"
MANIFEST = Path("/projects/kona-assets/manuelbv/NETSURF_ASSET_READINESS.json")
OUT = WT / "paper1-tecs-revision-v1/workspace/runs/paper1_runT_w8a8_v1"
BERS = (1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
SHARES = (0.0625, 0.125, 0.1875, 0.25)
RANKERS = {"msb_first": "msb", "per_example_absolute": "abs", "per_example_positive": "pos",
           "global_first_order": "global", "numerical_bit_jump": "jump", "random": "random"}
SEED = 20261222
RAW, VOTED, CODE, ABFT = 1, 2, 3, 4
PINNED: dict = {}

spec = importlib.util.spec_from_file_location("nonideal_protection", HERE.parents[1] / "netsurf/nonideal_protection.py")
N = importlib.util.module_from_spec(spec)
spec.loader.exec_module(N)


def setup(target, threads):
    sys.path.insert(0, str(W8A8))
    import subprocess
    import torch
    from netsurf import w8a8_assets
    # The frozen campaign pins kona-assets commit 1aea52b; later commits only added unrelated assets.  Accept the
    # current commit only if the two executable source files are byte-identical to their pinned hashes.
    for fname, expected in w8a8_assets.KONA_SOURCE_SHA256.items():
        if hashlib.sha256((w8a8_assets.KONA_ROOT / "src/kona_assets" / fname).read_bytes()).hexdigest() != expected:
            raise SystemExit(f"kona-assets {fname} differs from the pinned executable source")
    head = subprocess.run(["git", "-C", str(w8a8_assets.KONA_ROOT), "rev-parse", "--short", "HEAD"],
                          text=True, capture_output=True, check=True).stdout.strip()
    PINNED["kona_commit_pinned"], PINNED["kona_commit_used"] = w8a8_assets.KONA_COMMIT, head
    w8a8_assets.KONA_COMMIT = head
    from netsurf import w8a8_wide_ber_campaign as wide
    from netsurf.w8a8_campaign import _evaluation
    from netsurf.w8a8_faults import W8A8FaultAdapter
    plan, asset = wide.validate(MANIFEST, CAMPAIGN / target, verify_masks=False)
    adapter = W8A8FaultAdapter(asset)
    adapter.load()
    torch.set_num_threads(threads)
    evaluate = _evaluation(adapter, {"target": target, "final": plan["final_cohort"]})
    return plan, adapter, evaluate


def masks(plan, codes, starts):
    """{f"{ranker}@{share}": packed (codes,) uint8} from the frozen campaign's policies."""
    bits = 8 * codes
    out = {}
    for pol in plan["policies"]:
        name = pol["name"]
        method = next((m for m in RANKERS if name.startswith(m + "_b")), None)
        if method is None:
            continue
        share = pol["budget"] / bits
        share = min(SHARES, key=lambda s: abs(s - share))
        if pol["kind"] == "msb":
            rank = (7 - np.arange(8))[None, :] * codes + np.arange(codes)[:, None]      # tier first, then global code
            mask = N.pack_mask(rank < pol["budget"])
        else:
            raw = np.fromfile(pol["path"], dtype=np.uint8)
            if hashlib.sha256(raw.tobytes()).hexdigest() != pol["file_sha256"]:
                raise SystemExit(f"sealed mask {name} differs from its frozen hash")
            mask = raw
        assert mask.shape == (codes,) and int(N.POP[mask].sum()) == pol["budget"], name
        out[f"{RANKERS[method]}@{share}"] = mask
    if len(out) != 24:
        raise SystemExit(f"expected 24 ranked masks, found {len(out)}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=["distilbert_sst2_int8", "pythia160m_int8", "qwen3_0_6b_int8"])
    ap.add_argument("--draw", type=int, required=True)
    ap.add_argument("--ber-index", type=int, required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--only", nargs="*", default=None, help="smoke runs: evaluate only these conditions")
    a = ap.parse_args()
    if a.out == OUT and a.only:
        raise SystemExit("--only is for smoke runs outside the Run T output")
    out = a.out / a.target / f"draw{a.draw}" / f"ber{a.ber_index}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "results.csv").exists() or (out / "DONE").exists():
        raise SystemExit("write-once: results exist")
    t0 = time.time()
    plan, adapter, evaluate = setup(a.target, a.threads)
    names = [e["module"] for e in adapter.universe.entries]
    mods = [adapter.modules[n] for n in names]
    views = [m.qweight.numpy().view(np.uint8).reshape(-1) for m in mods]          # aliases the live INT8 buffers
    shapes = [tuple(m.qweight.shape) for m in mods]
    sizes = [v.size for v in views]
    starts = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    codes = int(starts[-1])
    assert 8 * codes == plan["eligible_bits"] == adapter.universe.bits
    clean_hash = hashlib.sha256(b"".join(v.tobytes() for v in views)).hexdigest()
    ranked = masks(plan, codes, starts)
    clean = dict(evaluate().metrics)
    p = BERS[a.ber_index]
    cid = zlib.crc32(a.target.encode())
    seed = lambda purpose, *more: [SEED, cid, a.draw, a.ber_index, purpose, *more]  # noqa: E731
    raw = N.raw_field(np.random.default_rng(seed(RAW)), codes, p)
    voted = N.raw_field(np.random.default_rng(seed(VOTED)), codes, N.tmr_residual(p))
    channels = sum(s[0] for s in shapes)

    def by_module(fn):
        x = np.empty(codes, np.uint8)
        for i in range(len(views)):
            x[starts[i]:starts[i + 1]] = fn(i, raw[starts[i]:starts[i + 1]])
        return x

    conds = [("unprotected", lambda: raw, 0.0, 0.0)]
    for k in N.SECDED_CHECK_BITS:
        conds.append((f"secded_{k}", lambda k=k: by_module(lambda i, r: N.secded(r, k, p, np.random.default_rng(seed(CODE, k, i)))),
                      0.0, N.storage_cost("secded", data_bits=k)))
    abft_cost = N.storage_cost("abft", channels=channels, weight_bits=8 * codes)
    conds.append(("abft_ideal", lambda: by_module(lambda i, r: N.abft_ideal(r, shapes[i], 0)), 0.0, abft_cost))
    conds.append(("abft_faulty", lambda: by_module(lambda i, r: N.abft_faulty(views[i], r, shapes[i], 0, p,
                                                                              np.random.default_rng(seed(ABFT, i)))), 0.0, abft_cost))
    for name, mask in ranked.items():
        share = float(name.split("@")[1])
        conds.append((f"tmr_{name}", lambda mask=mask: N.tmr(raw, voted, mask), share, N.storage_cost("tmr", share=share)))
    if a.only:
        conds = [c for c in conds if c[0] in a.only]
    rows = []
    for i, (cname, make, share, cost) in enumerate(conds):
        xor = make()
        try:
            for j, v in enumerate(views):
                np.bitwise_xor(v, xor[starts[j]:starts[j + 1]], out=v)
            s0 = time.time()
            metrics = dict(evaluate().metrics)
            sec = time.time() - s0
        finally:
            for j, v in enumerate(views):
                np.bitwise_xor(v, xor[starts[j]:starts[j + 1]], out=v)
        rows.append(dict(target=a.target, draw=a.draw, ber=p, condition=cname, share=share, storage_cost=round(cost, 6),
                         raw_faulty_bits=int(N.POP[raw].sum()), residual_faulty_bits=int(N.POP[xor].sum()),
                         seconds=round(sec, 1), **{f"metric_{k}": v for k, v in metrics.items()}))
        el = time.time() - t0
        print(json.dumps(dict(target=a.target, draw=a.draw, ber=p, done=i + 1, total=len(conds), seconds=round(el),
                              eta_minutes=round(el / (i + 1) * (len(conds) - i - 1) / 60, 1), condition=cname,
                              **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()})), flush=True)
    after = hashlib.sha256(b"".join(v.tobytes() for v in views)).hexdigest()
    if after != clean_hash:
        raise SystemExit("stored INT8 codes were not restored")
    with open(out / "results.csv", "x", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (out / "DONE").write_text(json.dumps(dict(seconds=round(time.time() - t0), evaluations=len(rows), clean=clean,
                                              clean_codes_sha256=clean_hash, threads=a.threads, executable_source=PINNED,
                                              masks_sha256={k: hashlib.sha256(v.tobytes()).hexdigest()[:16] for k, v in ranked.items()})))


if __name__ == "__main__":
    main()
