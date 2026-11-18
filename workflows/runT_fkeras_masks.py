#!/usr/bin/env python3
"""Run T: FKeras masks for the generic TFLite checkpoints (jet MLP, MNIST MLP, VWW), one card and replication.
Protocol: review/RUN_T_T1_PROTOCOL_2026-11-15.md, addendum T1b.

Hessian of the mean continuous cross-entropy (float64 companion, no activation rounding) over the replication's
1,024 calibration inputs; two signed largest-magnitude eigenpairs with the same residual-checked Lanczos solver and
options as the engine's FKeras (seed 970201).  Word score: the published projection |sum_i lambda_i (v_i.theta) v_i|.
    fkeras            bits ordered by tier (sign bit first), then by word score, then storage order (as published)
    fkeras_weighted   |word score| x 2^bit over all bits (the FKeras source's `rank_bits`, defined but unused there)
Writes <out>/<id>/rep<R>/fkeras_masks.npz (packed (n,) uint8 masks named <arm>@<share>) and fkeras.json (write-once).

    runT_fkeras_masks.py --id vww --rep 0 [--out DIR] [--threads 8]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from netsurf import nonideal_protection as N  # noqa: E402
from runG_gradtrust import card_model, load_card, load_xy  # noqa: E402
from runT_tflite import OUT, RUNS, SHARES  # noqa: E402

FK_SEED, FK_OPTIONS = 970201, dict(eigenpairs=2, max_iterations=100, tolerance=1e-5, max_products=300)


def tier_first(word: np.ndarray, budget: int) -> np.ndarray:
    """Top `budget` bits: sign tier first, then by word score (descending), then storage order."""
    n = word.size
    bit = np.repeat(np.arange(8)[None, :], n, axis=0).reshape(-1)
    score = np.repeat(word[:, None], 8, axis=1).reshape(-1)
    order = np.lexsort((np.arange(8 * n), -score, -bit))
    mask = np.zeros(8 * n, dtype=bool)
    mask[order[:budget]] = True
    return mask.reshape(n, 8)


def main():
    from netsurf.fkeras_baseline import leading_hessian_eigenpairs, principal_weight_hessian_score
    from netsurf.mean_variance import top_budget
    from netsurf.tflite_graph_continuous import GraphHessian
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--rep", type=int, required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    card = load_card(RUNS / f"paper1_gradtrust_inputs_v1/{a.id}/card.json")
    m = card_model(card, a.threads)
    out = a.out / card["id"] / f"rep{a.rep}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "fkeras_masks.npz").exists():
        raise SystemExit("write-once: fkeras_masks.npz exists")
    xc, yc = load_xy(card["calibration"][a.rep])
    t0 = time.time()
    H = GraphHessian(Path(card["tflite"]).read_bytes(), m.names, xc, yc)
    assert np.allclose(H.theta, np.concatenate([m.clean[n].reshape(-1) for n in m.names]), rtol=1e-6, atol=1e-9)
    vectors, diag = leading_hessian_eigenpairs(H, len(H.theta), count=FK_OPTIONS["eigenpairs"], seed=FK_SEED,
                                               max_iterations=FK_OPTIONS["max_iterations"],
                                               tolerance=FK_OPTIONS["tolerance"], max_products=FK_OPTIONS["max_products"])
    word = np.abs(principal_weight_hessian_score(H.theta, vectors, np.asarray(diag.eigenvalues)))
    masks = {}
    for share in SHARES:
        c = int(np.floor(share * m.n * 8))
        masks[f"fkeras@{share}"] = N.pack_mask(tier_first(word, c))
        masks[f"fkeras_weighted@{share}"] = N.pack_mask(top_budget(word[:, None] * (2.0 ** np.arange(8))[None, :], c))
    assert np.all(masks["fkeras@0.125"] == 0x80), "the 12.5% FKeras mask must be the sign bits"
    np.savez(out / "fkeras_masks.npz", **masks)
    (out / "fkeras.json").write_text(json.dumps(dict(
        id=card["id"], rep=a.rep, calibration_inputs=int(len(yc)), seconds=round(time.time() - t0, 1),
        eigenvalues=[float(v) for v in diag.eigenvalues], residuals=[float(v) for v in diag.normalized_residuals],
        hessian_vector_products=int(diag.hvp_calls),
        options=FK_OPTIONS, seed=FK_SEED, companion=H.companion.implementation_version,
        word_score_sha256=hashlib.sha256(word.tobytes()).hexdigest(),
        masks_sha256={k: hashlib.sha256(v.tobytes()).hexdigest()[:16] for k, v in masks.items()}), indent=1))
    print(json.dumps(dict(done=str(out), seconds=round(time.time() - t0, 1), eigenvalues=[float(v) for v in diag.eigenvalues])))


if __name__ == "__main__":
    main()
