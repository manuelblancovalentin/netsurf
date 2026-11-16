#!/usr/bin/env python3
"""Run T, stage T1b: the non-ideal comparison of runT_tflite.py for the ResNet-8 checkpoints and the DS-CNN, through
their engine adapters (TFLite reference kernels).  Protocol: review/RUN_T_T1_PROTOCOL_2026-11-15.md, section T1b.

Same fault model, conditions, seeds-by-purpose and output layout as runT_tflite.py (its `conditions` is imported).
Rankers:
    abs, pos          the saved 1,024-input masks of the earlier fault runs (Stage 1 for ResNet-8, Stage 4 preparation
                      for the DS-CNN)
    abs64, msb        rebuilt from the replication's first 64 calibration inputs with the engine's ordering code
    fkeras            FKeras as published: bit tier first, then the Hessian word score (engine code, two eigenpairs,
                      the same 1,024 calibration inputs as abs and pos)
    fkeras_weighted   the FKeras source's `rank_bits` (pinned commit 68e775c, defined there but not used by its
                      ranking path): |word score| x 2^bit, all bits in one list
    jump, random      numerical jump; seeded random order
Checks before any fault: the rebuilt msb and fkeras masks at 12.5% protect exactly the sign bits.

Output (write-once): <out>/<id>/rep<R>/draw<D>/{results.csv, DONE}; id = resnet8-<name> or dscnn.

    runT_engine.py --model pinned --rep 1 --draw 0 [--out DIR] [--bers ...]
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
from runR_stage1 import MASKS, NAMES, bkey  # noqa: E402
from runP_phase12 import unpack  # noqa: E402
from runT_tflite import BERS, OUT, RANDOM, SEED, SHARES, channel_axes, conditions  # noqa: E402

FK_SEED, FK_OPTIONS = 970201, dict(eigenpairs=2, max_iterations=100, tolerance=1e-5, max_products=300)


TOYCAR_MODEL = Path("/data/models/models--mlcommons--tiny/blobs/87cf24194ef93d1d9b11a591d805526b98008e351655d29883c825c9c106ba24")
TOYCAR_DATA = Path("/data/datasets/dcase2020-toycar")
TOYCAR_LOCK = Path("/projects/kona-assets/manuelbv/locks/mlperf-tiny/ad.lock.json")
TOYCAR_FILES, TOYCAR_WINDOW = 16, 62       # 16 normal training files for the masks (as in the main grid); 62 test files


class ToyModel:
    """ToyCar engine adapter and differentiable companion, flattened in adapter storage order."""

    def __init__(self, threads):
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(threads)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        from netsurf.adapters.mlperf_tiny_multibench import MlperfTinyToyCarAdapter
        from netsurf.mlperf_multibench_companion import MlperfToyCarDifferentiableCompanion
        self.adapter = MlperfTinyToyCarAdapter(TOYCAR_MODEL, TOYCAR_DATA, asset_lock=TOYCAR_LOCK, injectable_parameters=("kernels",))
        self.adapter.load()
        storage = self.adapter._require_loaded()
        values = storage.values()
        self.names = list(values)
        self.shapes = {n: values[n].shape for n in self.names}
        self.starts = np.concatenate([[0], np.cumsum([values[n].size for n in self.names])]).astype(np.int64)
        self.codes = np.concatenate([np.asarray(values[n]).reshape(-1) for n in self.names]).astype(np.int8)
        jumps = self.adapter.bit_perturbations()
        self.jumps = np.concatenate([np.asarray(jumps[n]).reshape(-1, 8) for n in self.names]).astype(np.float64)
        self.comp = MlperfToyCarDifferentiableCompanion(storage.reference_content)
        self.n = int(self.starts[-1])
        self.sha256 = storage.sha256

    def per_example(self, x, y):
        g = self.comp.per_example_gradients(x, y)
        return np.concatenate([np.asarray(g[n]).reshape(len(x), -1) for n in self.names], axis=1).astype(np.float64)


def load(model, rep, threads):
    if model == "toycar":
        m = ToyModel(threads)
        xe, ye, _ = m.adapter.sample_partition("evaluation", TOYCAR_WINDOW, offset=TOYCAR_WINDOW * rep)
        xc, yc, _ = m.adapter.sample_partition("ranking", TOYCAR_FILES, offset=TOYCAR_FILES * rep)
        m.storage = m.adapter._require_loaded()
        return m, None, (xe, np.asarray(ye)), (xc, np.asarray(yc)), (xc, np.asarray(yc)), "toycar"
    if model == "kws":
        from runR_kws import CAL_COUNT, OUT as PREP, KwsModel
        from runR_stage4_kws import WINDOW
        m = KwsModel(threads)
        saved = np.load(PREP / "masks" / f"rep{rep}" / "masks.npz")
        xe, ye, _ = m.clips("evaluation", WINDOW, WINDOW * rep)
        xc, yc = m.clips("ranking", 64, CAL_COUNT * rep)[:2]
        xf, yf = m.clips("ranking", CAL_COUNT, CAL_COUNT * rep)[:2]
        cid = "dscnn"
    else:
        from runP_phase0 import Model
        m = Model(model, threads)
        saved = np.load(MASKS / model / f"rep{rep}" / "masks.npz")
        xe, ye, _ = m.images("evaluation", 1000, 1000 * rep)
        xc, yc = m.images("ranking", 64, 1000 * rep)[:2]
        xf, yf = m.images("ranking", 1024, 1000 * rep)[:2]
        cid = f"resnet8-{model}"
    m.storage = m.adapter._require_loaded()
    return m, saved, (xe, np.asarray(ye)), (xc, np.asarray(yc)), (xf, np.asarray(yf)), cid


def build_masks(m, saved, xc, yc, xf, yf, seed_id, rep):
    """{f"{ranker}@{share}": packed (n,) uint8} and the FKeras solver record."""
    from netsurf.fkeras_estimator import fkeras_family
    from netsurf.mean_variance import top_budget
    from netsurf.ordering import compose_ordering
    from netsurf.protection import ProtectionPolicy
    from netsurf.ranking import Ranking
    from runQ_calibration import scores
    ids, specs = m.adapter.bit_identities(), m.adapter.parameter_specs()
    row = {n: int(m.starts[i]) for i, n in enumerate(m.names)}
    SC = scores(m.per_example(xc, yc), m.jumps)
    S = SC["abs"]
    t0 = time.time()
    fk = fkeras_family(m.adapter, xf, yf, ids, seed=FK_SEED, options=FK_OPTIONS, requested=["fkeras_paper"])
    word = dict(fk["fkeras_paper"].scores)
    ordered = {"abs64": ({b: float(S[row[b.parameter] + b.flat_index, b.bit_index]) for b in ids},
                         ("estimator_score_desc", "stable_identity")),
               "msb": ({b: float(b.bit_index) for b in ids}, ("estimator_score_desc", "stable_identity")),
               "fkeras": (word, ("bit_significance_desc", "estimator_score_desc", "stable_identity"))}
    out = {}
    for arm, (sc, criteria) in ordered.items():
        ordering = compose_ordering(arm, Ranking(arm, sc, {}), ids, specs, criteria)
        for share in SHARES:
            mk = np.zeros((m.n, 8), dtype=bool)
            for b in ProtectionPolicy.ordered(ordering, share).protected:
                mk[row[b.parameter] + b.flat_index, b.bit_index] = True
            out[f"{arm}@{share}"] = N.pack_mask(mk)
    wscore = np.zeros(m.n)
    for b in ids:                                   # the engine keys the word score by bit identity (same for 8 bits)
        wscore[row[b.parameter] + b.flat_index] = abs(float(word[b]))
    weighted = wscore[:, None] * (2.0 ** np.arange(8))[None, :]
    rnd = np.random.default_rng([SEED, seed_id, rep, RANDOM]).random((m.n, 8))
    for share in SHARES:
        c = int(np.floor(share * m.n * 8))
        out[f"fkeras_weighted@{share}"] = N.pack_mask(top_budget(weighted, c))
        out[f"jump@{share}"] = N.pack_mask(top_budget(np.abs(m.jumps), c))
        out[f"random@{share}"] = N.pack_mask(top_budget(rnd, c))
        for arm in ("abs", "pos"):
            if saved is None:                                  # ToyCar: no saved 1,024-input masks; same files as FKeras
                out[f"{arm}@{share}"] = N.pack_mask(top_budget(SC[arm], c))
            else:
                out[f"{arm}@{share}"] = N.pack_mask(unpack(saved[f"{arm}1024@{bkey(share)}"], m.n))
    sign = np.full(m.n, 0x80, dtype=np.uint8)
    assert np.array_equal(out["msb@0.125"], sign) and np.array_equal(out["fkeras@0.125"], sign), "full-tier masks must be the sign bits"
    return out, round(time.time() - t0, 1)


def evaluate(m, xor, x, y):
    codes = (m.codes.view(np.uint8) ^ xor).view(np.int8)
    faulted = {n: codes[m.starts[i]: m.starts[i + 1]].reshape(m.shapes[n]) for i, n in enumerate(m.names)}
    with m.adapter.temporary_values(faulted):
        res = m.adapter.evaluate(x, y)
    if hasattr(res, "roc_auc"):                             # ToyCar: file-level AUC in the accuracy column
        return float(res.roc_auc), float(res.reconstruction_mse)
    return float(res.accuracy), float(res.categorical_crossentropy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["kws", "toycar"] + list(NAMES))
    ap.add_argument("--rep", type=int, required=True)
    ap.add_argument("--draw", type=int, required=True)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--bers", type=float, nargs="+", default=list(BERS))
    a = ap.parse_args()
    if a.out == OUT and tuple(a.bers) != BERS:
        raise SystemExit("the Run T output takes the frozen rate grid only; use --out for anything else")
    m, saved, (xe, ye), (xc, yc), (xf, yf), name = load(a.model, a.rep, a.threads)
    cid = zlib.crc32(name.encode())
    out = a.out / name / f"rep{a.rep}" / f"draw{a.draw}"
    out.mkdir(parents=True, exist_ok=True)
    target = out / "results.csv"
    if target.exists() or (out / "DONE").exists():
        raise SystemExit(f"{target} exists (write-once)")
    t0 = time.time()
    masks, fk_seconds = build_masks(m, saved, xc, yc, xf, yf, cid, a.rep)
    # the base-arm switch of `conditions` looks for abs@0.125
    for nm, mask in masks.items():
        want = int(np.floor(float(nm.split("@")[1]) * m.n * 8))
        assert int(N.POP[mask].sum()) == want, (nm, int(N.POP[mask].sum()), want)
    axes = channel_axes(m)
    clean_acc, clean_ce = evaluate(m, np.zeros(m.n, np.uint8), xe, ye)
    rows, total = [], None
    for p in a.bers:
        bi = BERS.index(p) if p in BERS else 1000 + int(round(-1000 * np.log10(p)))
        conds, raw = conditions(m, masks, axes, p, lambda purpose, *more: [SEED, cid, a.rep, a.draw, bi, purpose, *more])
        total = total or len(conds) * len(a.bers)
        for cname, (xor, share, cost) in conds.items():
            acc, ce = evaluate(m, xor, xe, ye)
            rows.append(dict(checkpoint=name, rep=a.rep, draw=a.draw, ber=p, condition=cname, share=share,
                             storage_cost=round(cost, 6), accuracy=acc, cross_entropy=ce, window=len(ye),
                             raw_faulty_bits=int(N.POP[raw].sum()), residual_faulty_bits=int(N.POP[xor].sum())))
            el = time.time() - t0
            if len(rows) % 10 == 0 or len(rows) == total:
                print(json.dumps(dict(id=name, rep=a.rep, draw=a.draw, done=len(rows), total=total, seconds=round(el),
                                      eta_minutes=round(el / len(rows) * (total - len(rows)) / 60, 1))), flush=True)
    with open(target, "x", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (out / "DONE").write_text(json.dumps(dict(
        seconds=round(time.time() - t0), evaluations=len(rows), clean_accuracy=clean_acc, clean_cross_entropy=clean_ce,
        model_sha256=m.sha256, fkeras_seconds=fk_seconds,
        masks_sha256={k: hashlib.sha256(v.tobytes()).hexdigest()[:16] for k, v in masks.items()},
        seconds_per_evaluation=round((time.time() - t0) / max(len(rows), 1), 3))))


if __name__ == "__main__":
    main()
