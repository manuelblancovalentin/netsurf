#!/usr/bin/env python3
"""Run T, stage T1: one comparison under non-ideal protection for an INT8 TFLite checkpoint (card).
Plan: review/RUN_T_FULL_COMPARISON_PLAN_2026-11-15.md and its T1 protocol note (paper worktree).

Every stored bit flips independently with probability p, including the copies of triplicated bits and the check bits
of codes (netsurf/nonideal_protection.py).  For one card, replication and draw, every fault rate is evaluated with
the TFLite reference kernels on the replication's evaluation window under

    unprotected
    tmr_<ranker>@<share>   the top <share> of bits by <ranker> stored three times (extra storage 2 x share)
    secded_<k>             SEC-DED over k = 64, 32, 16, 8 data bits
    abft_ideal             checksum correction per output channel, fault-free checksums
    abft_faulty            the same with faulty 32-bit checksums and wrong repairs

Rankers (masks built from the replication's locked Phase 1 statistics, 1,024 calibration inputs, exactly as the
gradient-trust fault runs built theirs): abs (per-example absolute gradient), msb (bit position, ties by storage
order; FORTUNE-style at a share of 12.5%), pos, global, jump, random.  Further rankers (FKeras) are added with
--extra-masks: an npz of packed (n,) uint8 masks named <ranker>@<share>; they see the same fault fields, because
every random stream is seeded by (card, replication, draw, rate, purpose) and never by the set of arms.

Output (write-once): <out>/<id>/rep<R>/draw<D>/{results.csv, DONE}.  --arms-tag names a second, separate file set for
arms added later.  Interpreter: the TensorFlow reference environment.

    runT_tflite.py --id jet-e100 --rep 0 --draw 0 [--out DIR] [--extra-masks FILE --arms-tag fkeras]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from netsurf import nonideal_protection as N  # noqa: E402
from runG_gradtrust import STAT_KEYS, card_model, load_card, load_xy  # noqa: E402
from site_paths import WT  # noqa: E402

RUNS = WT / "paper1-tecs-revision-v1/workspace/runs"
OUT = RUNS / "paper1_runT_tflite_v1"
BERS = (1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)            # the study grid
SHARES = (0.0625, 0.125, 0.1875, 0.25)
RANKERS = ("abs", "msb", "pos", "global", "jump", "random")
SEED = 20261221
RAW, VOTED, CODE, ABFT, RANDOM = 1, 2, 3, 4, 5                       # purposes of the random streams


def build_masks(m, st, cid, rep):
    """{f"{ranker}@{share}": packed (n,) uint8} for the six built-in rankers."""
    from netsurf.mean_variance import top_budget
    J = m.jumps
    sc = m.scores(st)
    scores = dict(abs=sc["abs"], pos=sc["pos"], **{"global": np.abs(st["g"])[:, None] * np.abs(J)}, jump=np.abs(J),
                  msb=np.broadcast_to(np.arange(8, dtype=float), (m.n, 8)),
                  random=np.random.default_rng([SEED, cid, rep, RANDOM]).random((m.n, 8)))
    out = {}
    for share in SHARES:
        c = int(np.floor(share * m.n * 8))
        for r in RANKERS:
            out[f"{r}@{share}"] = N.pack_mask(top_budget(scores[r], c))
    return out


def channel_axes(m):
    """Output-channel axis of every kernel: its per-channel quantization axis (axis 0 for a single scale)."""
    by = {d.parameter: d for d in m.storage.descriptors}
    return {nm: (by[nm].quantization.quantized_dimension if len(by[nm].quantization.scales) > 1 else 0) for nm in m.names}


def conditions(m, masks, axes, p, seeds):
    """Ordered {condition: (xor (n,) uint8, share, storage cost)} for one rate; `seeds(purpose, *more)` gives a seed."""
    codes_u8 = m.codes.view(np.uint8)
    raw = N.raw_field(np.random.default_rng(seeds(RAW)), m.n, p)
    out = {}
    if "abs@0.125" in masks:                                          # the base arm set (not an --extra-masks call)
        out["unprotected"] = (raw, 0.0, 0.0)
        for k in N.SECDED_CHECK_BITS:
            x = np.empty(m.n, np.uint8)
            for i, nm in enumerate(m.names):
                s, e = m.starts[i], m.starts[i + 1]
                x[s:e] = N.secded(raw[s:e], k, p, np.random.default_rng(seeds(CODE, k, i)))
            out[f"secded_{k}"] = (x, 0.0, N.storage_cost("secded", data_bits=k))
        xi, xf, channels = np.empty(m.n, np.uint8), np.empty(m.n, np.uint8), 0
        for i, nm in enumerate(m.names):
            s, e = m.starts[i], m.starts[i + 1]
            shape, ax = tuple(m.shapes[nm]), axes[nm]
            channels += shape[ax]
            xi[s:e] = N.abft_ideal(raw[s:e], shape, ax)
            xf[s:e] = N.abft_faulty(codes_u8[s:e], raw[s:e], shape, ax, p, np.random.default_rng(seeds(ABFT, i)))
        cost = N.storage_cost("abft", channels=channels, weight_bits=8 * m.n)
        out["abft_ideal"], out["abft_faulty"] = (xi, 0.0, cost), (xf, 0.0, cost)
    voted = N.raw_field(np.random.default_rng(seeds(VOTED)), m.n, N.tmr_residual(p))
    for name, mask in masks.items():
        share = float(name.split("@")[1])
        out[f"tmr_{name}"] = (N.tmr(raw, voted, mask), share, N.storage_cost("tmr", share=share))
    return out, raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--rep", type=int, required=True)
    ap.add_argument("--draw", type=int, required=True)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--bers", type=float, nargs="+", default=list(BERS))
    ap.add_argument("--extra-masks", type=Path, default=None)
    ap.add_argument("--arms-tag", default="")
    a = ap.parse_args()
    if a.out == OUT and tuple(a.bers) != BERS:
        raise SystemExit("the Run T output takes the frozen rate grid only; use --out for anything else")
    if bool(a.extra_masks) != bool(a.arms_tag):
        raise SystemExit("--extra-masks and --arms-tag go together")
    card = load_card(RUNS / f"paper1_gradtrust_inputs_v1/{a.id}/card.json")
    m = card_model(card, a.threads)
    cid = zlib.crc32(card["id"].encode())
    if a.extra_masks:
        z = np.load(a.extra_masks)
        masks = {k: np.asarray(z[k], dtype=np.uint8) for k in z.files}
        assert all(v.shape == (m.n,) for v in masks.values()), "extra masks must be packed (n,) uint8"
    else:
        zs = np.load(RUNS / f"paper1_gradtrust_diag_v1/{a.id}/stats.npz")
        masks = build_masks(m, {k: zs[f"{k}@{a.rep}"] for k in STAT_KEYS}, cid, a.rep)
    for name, mask in masks.items():
        want = int(np.floor(float(name.split("@")[1]) * m.n * 8))
        assert int(N.POP[mask].sum()) == want, (name, int(N.POP[mask].sum()), want)
    axes = channel_axes(m)
    xe, ye = load_xy(card["evaluation"][a.rep])
    out = a.out / card["id"] / f"rep{a.rep}" / f"draw{a.draw}"
    out.mkdir(parents=True, exist_ok=True)
    suffix = f"_{a.arms_tag}" if a.arms_tag else ""
    target = out / f"results{suffix}.csv"
    if target.exists() or (out / f"DONE{suffix}").exists():
        raise SystemExit(f"{target} exists (write-once)")
    zero = np.zeros((m.n, 8), dtype=bool)
    t0 = time.time()
    clean_acc, clean_ce, _, _ = m.evaluate(zero, zero, xe, ye)
    rows, total = [], None
    for p in a.bers:
        bi = BERS.index(p) if p in BERS else 1000 + int(round(-1000 * np.log10(p)))
        conds, raw = conditions(m, masks, axes, p, lambda purpose, *more: [SEED, cid, a.rep, a.draw, bi, purpose, *more])
        total = total or len(conds) * len(a.bers)
        for name, (xor, share, cost) in conds.items():
            flips = np.unpackbits(xor[:, None], axis=1, bitorder="little").astype(bool)
            acc, ce, _, _ = m.evaluate(flips, zero, xe, ye)
            rows.append(dict(checkpoint=card["id"], rep=a.rep, draw=a.draw, ber=p, condition=name, share=share,
                             storage_cost=round(cost, 6), accuracy=acc, cross_entropy=ce, window=len(ye),
                             raw_faulty_bits=int(N.POP[raw].sum()), residual_faulty_bits=int(N.POP[xor].sum())))
            el = time.time() - t0
            if len(rows) % 10 == 0 or len(rows) == total:
                print(json.dumps(dict(id=a.id, rep=a.rep, draw=a.draw, done=len(rows), total=total, seconds=round(el),
                                      eta_minutes=round(el / len(rows) * (total - len(rows)) / 60, 1))), flush=True)
    with open(target, "x", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (out / f"DONE{suffix}").write_text(json.dumps(dict(
        seconds=round(time.time() - t0), evaluations=len(rows), clean_accuracy=clean_acc, clean_cross_entropy=clean_ce,
        model_sha256=m.sha256, masks_sha256={k: hashlib.sha256(v.tobytes()).hexdigest()[:16] for k, v in masks.items()},
        seconds_per_evaluation=round((time.time() - t0) / max(len(rows), 1), 3))))


if __name__ == "__main__":
    main()
