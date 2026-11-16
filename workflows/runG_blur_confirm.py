#!/usr/bin/env python3
"""Confirmation of the two-moment ("noise plus events") fault model on the gradient-trust checkpoints (jet MLP, MNIST
MLP, Visual Wake Words).  Plan: review/BLUR_CONFIRMATION_PLAN_2026-11-15.md (paper worktree), frozen before any pass.

For one card and replication this repeats the pilot's ``blur`` stage (runR_blur_pilot.py, 7057778) with the generic
INT8 companion: every (rate, budget, arm) is passed through the companion with the weights displaced by

    mean    mu                                    mu_w = p * sum of unprotected jumps
    gauss   mu + sqrt(V) * eps                    V_w  = p (1 - p) * sum of squared unprotected jumps
    hybrid  bits 6 and 7 as real flips, bits 0-5 by their two moments
    real    the flips of the finished TFLite runs' own fields (same seeds): a check of the companion path

The masks are rebuilt from the locked Phase 1 statistics exactly as the fault runs built them, the inputs are the
replication's evaluation window, and the fields use the fault runs' seeds.  ``run`` and ``moments`` are imported from
the pilot, so the model is computed by the pilot's own code.  No TFLite fault run is made.

Output (write-once): <out>/blur/<id>_rep<R>/{rows.csv, DONE}.  Interpreter: the TensorFlow reference environment.

    runG_blur_confirm.py --id jet-e1 --rep 0 [--out DIR] [--threads 4] [--rates ...] [--draws 8]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runG_gradtrust import FRACTIONS, SEED_FAULTS, STAT_KEYS, card_model, load_card, load_xy, masks_for  # noqa: E402
from runR_blur_pilot import PAPER, SEED_G, finish, moments, run  # noqa: E402

RUNS = PAPER / "workspace/runs"
OUT = RUNS / "paper1_blurconf_v1"
RATES = (3e-4, 1e-3, 3e-3, 5e-3, 1e-2, 3e-2)      # the pilot's six rates
ARMS = ("abs1024", "pos1024", "msb")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--rep", type=int, required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--rates", type=float, nargs="+", default=list(RATES))
    ap.add_argument("--fractions", type=float, nargs="+", default=list(FRACTIONS))
    ap.add_argument("--draws", type=int, default=8)
    a = ap.parse_args()
    if a.out == OUT and (tuple(a.rates) != RATES or tuple(a.fractions) != FRACTIONS or a.draws != 8):
        raise SystemExit("the confirmation output takes the frozen grid only; use --out for anything else")
    card = load_card(RUNS / f"paper1_gradtrust_inputs_v1/{a.id}/card.json")
    m = card_model(card, a.threads)
    z = np.load(RUNS / f"paper1_gradtrust_diag_v1/{a.id}/stats.npz")
    masks = masks_for(m, {k: z[f"{k}@{a.rep}"] for k in STAT_KEYS})
    xe, ye = load_xy(card["evaluation"][a.rep])
    xe, ye = np.asarray(xe), np.asarray(ye).astype(np.int64)
    assert np.all(np.abs(m.jumps[:, 7]) >= np.abs(m.jumps[:, 6])), "bit 7 is expected to be the sign bit"
    out = a.out / "blur" / f"{a.id}_rep{a.rep}"
    out.mkdir(parents=True, exist_ok=False)
    t0, J, rows, n_fw = time.time(), m.jumps, [], 0
    clean_loss, clean_acc, _ = run(m, np.zeros(m.n), xe, ye)
    cid = zlib.crc32(card["id"].encode())
    U = [np.random.default_rng([SEED_FAULTS, cid, a.rep, fi]).random((m.n, 8)) for fi in range(min(a.draws, 8))]
    total = len(a.rates) * len(a.fractions) * len(ARMS)
    for ri, p in enumerate(a.rates):
        gi = RATES.index(p) if p in RATES else 100 + ri        # the Gaussian draws depend on the rate, not on its position
        eps = [np.random.default_rng([SEED_G, gi, d]).standard_normal(m.n) for d in range(a.draws)]
        for frac in a.fractions:
            for arm in ARMS:
                unprot = ~masks[frac][arm]
                mu, var = moments(J, unprot, p)
                low = unprot.copy()
                low[:, 6:] = False
                mu_low, var_low = moments(J, low, p)
                todo = [("mean", 0, mu)]
                for d in range(a.draws):
                    todo.append(("gauss", d, mu + np.sqrt(var) * eps[d]))
                    if d < len(U):
                        fl = (U[d] < p) & unprot
                        top = fl.copy()
                        top[:, :6] = False
                        todo.append(("real", d, np.where(fl, J, 0.0).sum(1)))
                        todo.append(("hybrid", d, np.where(top, J, 0.0).sum(1) + mu_low + np.sqrt(var_low) * eps[d]))
                for variant, d, disp in todo:
                    loss, acc, _ = run(m, disp, xe, ye)
                    n_fw += 1
                    rows.append(dict(model=a.id, rep=a.rep, arm=arm, budget=frac, ber=p, variant=variant, draw=d,
                                     loss=loss, accuracy=acc))
                done = len(rows) // (1 + 3 * a.draws)
                el = time.time() - t0
                print(json.dumps(dict(id=a.id, rep=a.rep, done=done, total=total, seconds=round(el),
                                      eta_minutes=round(el / done * (total - done) / 60, 1))), flush=True)
    with open(out / "rows.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    finish(out, t0, n_fw, clean_loss=clean_loss, clean_accuracy=clean_acc, inputs=len(ye))


if __name__ == "__main__":
    main()
