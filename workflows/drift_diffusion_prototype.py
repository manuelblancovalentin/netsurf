#!/usr/bin/env python3
"""EXPLORATORY prototype (not a finalized study): two-moment ("drift + diffusion") surrogate of stored-bit faults.

Under independent stored-bit faults with probability p and a protection mask P, the faulted INT8 code of
a weight is  q' = q + sum_{unprotected bits b} B_b * jump_b  with independent B_b ~ Bernoulli(p), because
the two's-complement value is linear in the stored bits.  Hence, exactly,

    E[q']   = q + p * d,            d = sum of unprotected jumps          (drift)
    Var[q'] = p * (1 - p) * v,      v = sum of squared unprotected jumps  (diffusion)

and different weights are independent.  The surrogate keeps only these two moments: it replaces the
fault field by one Gaussian draw per stored weight, W' = q + p*d + sqrt(p*(1-p)*v) * eps.  The noise is
drawn in weight space, once per pass, so it is shared by every token and sentence exactly as a persistent
fault field is.  If this reproduces the measured Monte Carlo outcomes of all masks, then (d, v) per
weight are sufficient statistics of a mask at a given p, and the surrogate is differentiable in them.

Variants per (mask, BER):
    full        drift and diffusion, --draws Gaussian draws
    diffusion   diffusion only (drift set to zero), --diffusion-draws draws
(drift only is already in workspace/prototypes/meanfield_drift_v1.)
Added after v1 showed the Gaussian is too optimistic when sign bits are exposed (--variants):
    hybrid      the two highest bits (6 and the sign bit) as real independent flips, Gaussian for bits 0-5
    exact       all eight bits as real independent flips, i.e. a true fault field drawn here; this checks
                that the float path used by the prototype reproduces the measured Monte Carlo means
The same eps is used for every mask (seed depends on BER and draw only), mirroring the paired fault fields.

Read-only inputs: the frozen DistilBERT W8A8 snapshot, the sealed masks, the frozen 512-sentence final
cohort.  No bit-level fault field is sampled.  One process per shard; each writes shard_<i>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

DEFAULT_BERS = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
METHODS = ("msb_first", "random", "numerical_bit_jump", "global_first_order", "per_example_absolute", "per_example_positive")


def bit_jumps(codes: np.ndarray) -> np.ndarray:
    """Exact change of an INT8 code, in code units, when stored bit b is flipped; shape (n, 8)."""
    q = np.ascontiguousarray(codes, dtype=np.int8).reshape(-1)
    u = q.view(np.uint8)
    out = np.empty((q.size, 8), dtype=np.int16)
    for bit in range(8):
        out[:, bit] = np.bitwise_xor(u, np.uint8(1 << bit)).view(np.int8).astype(np.int16) - q.astype(np.int16)
    return out


def unprotected_moments(jumps: np.ndarray, protected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """d = sum of unprotected jumps, v = sum of squared unprotected jumps, per code.

    With independent flips of probability p the faulted code has mean q + p*d and variance p*(1-p)*v.
    """
    j = np.where(protected, 0, jumps).astype(np.int32)
    return j.sum(axis=1), (j * j).sum(axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--bers", type=float, nargs="+", default=DEFAULT_BERS)
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.125, 0.25])
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--diffusion-draws", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20261111)
    ap.add_argument("--variants", nargs="+", default=["full", "diffusion"], choices=["full", "diffusion", "hybrid", "exact"])
    ap.add_argument("--policies", nargs="+", default=None, help="keep only these, written as name@coverage")
    args = ap.parse_args()
    target = args.out / f"shard_{args.shard:02d}.json"
    assert args.out.is_dir() and not target.exists(), target

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import meanfield_drift_prototype as mf
    torch, plan, adapter, evaluate = mf.setup()
    modules, n_codes = mf.per_module_arrays(adapter)
    assert n_codes * 8 == plan["eligible_bits"]
    qmods = [adapter.modules[m["name"]] for m in modules]
    for qm, m in zip(qmods, modules):                                   # the helper above equals the v1 jumps
        assert np.array_equal(bit_jumps(qm.qweight.reshape(-1).numpy()), m["jumps"])

    policies = [("unprotected", 0, 0.0)]
    for frac in args.budgets:
        budget = int(round(frac * plan["eligible_bits"]))
        policies += [(name, budget, frac) for name in METHODS]
    if args.policies:
        policies = [pol for pol in policies if f"{pol[0]}@{pol[2]}" in set(args.policies)]
        assert len(policies) == len(set(args.policies)), "unknown policy requested"
    mine = [pol for i, pol in enumerate(policies) if i % args.shards == args.shard]
    need_bits = bool({"hybrid", "exact"} & set(args.variants))

    def run(delta):
        for qm, d in zip(qmods, delta):
            qm._mf_delta = d
        started = time.monotonic()
        risk = evaluate()
        return dict(risk.metrics), time.monotonic() - started

    clean, _ = run([None] * len(qmods))
    print(json.dumps({"shard": args.shard, "policies": [p[0] + f"@{p[2]}" for p in mine], "exact_clean": clean}), flush=True)
    rows = []
    for name, budget, frac in mine:
        protected = mf.mask_bits(name, budget, n_codes)
        d_t, s_t, dl_t, sl_t, j_t = [], [], [], [], []
        for mod in modules:
            prot = protected[mod["start"]: mod["start"] + mod["n"]]
            d, v = unprotected_moments(mod["jumps"], prot)
            d_t.append(torch.from_numpy(d.astype(np.float32).reshape(mod["shape"])))
            s_t.append(torch.from_numpy(np.sqrt(v.astype(np.float64)).astype(np.float32).reshape(mod["shape"])))
            if need_bits:
                low = prot.copy(); low[:, 6:] = True                    # treat bits 6-7 as absent: moments of bits 0-5
                dl, vl = unprotected_moments(mod["jumps"], low)
                dl_t.append(torch.from_numpy(dl.astype(np.float32).reshape(mod["shape"])))
                sl_t.append(torch.from_numpy(np.sqrt(vl.astype(np.float64)).astype(np.float32).reshape(mod["shape"])))
                j_t.append(torch.from_numpy(np.where(prot, 0, mod["jumps"]).astype(np.int16)))   # (codes, 8) unprotected jumps
        del protected
        for ber_index, p in enumerate(args.bers):
            sd = float(np.sqrt(p * (1.0 - p)))
            counts = {"full": args.draws, "diffusion": args.diffusion_draws, "hybrid": args.draws, "exact": args.draws}
            for variant in args.variants:
                for draw in range(counts[variant]):
                    gen = torch.Generator().manual_seed(args.seed + 1000 * int(round(1e7 * p)) + draw)
                    delta = []
                    for k, (d, s) in enumerate(zip(d_t, s_t)):         # fixed module order: same draws for every mask
                        if variant in ("full", "diffusion"):
                            eps = torch.randn(d.shape, generator=gen, dtype=torch.float32)
                            delta.append((p * d if variant == "full" else 0.0) + sd * s * eps)
                            continue
                        real = torch.zeros(d.numel(), dtype=torch.float32)
                        for bit in (range(8) if variant == "exact" else (6, 7)):   # real coin-toss flips of these bits
                            hit = torch.rand(d.numel(), generator=gen) < p
                            real += hit.float() * j_t[k][:, bit].float()
                        real = real.reshape(d.shape)
                        if variant == "hybrid":
                            eps = torch.randn(d.shape, generator=gen, dtype=torch.float32)
                            real = real + p * dl_t[k] + sd * sl_t[k] * eps
                        delta.append(real)
                    metrics, seconds = run(delta)
                    del delta
                    row = dict(policy=name, coverage=frac, budget_bits=budget, ber=p, variant=variant, draw=draw,
                               accuracy=metrics["accuracy"], nll=metrics["mean_nll"],
                               failure_fraction=metrics.get("failure_fraction", 0.0), seconds=seconds)
                    rows.append(row)
                    print(json.dumps(row), flush=True)
    for qm in qmods:
        qm._mf_delta = None
    with open(target, "x") as f:
        json.dump(dict(schema="paper1-drift-diffusion-prototype/1.0", status="EXPLORATORY", target=mf.TARGET,
                       shard=args.shard, shards=args.shards, bers=args.bers, draws=args.draws,
                       diffusion_draws=args.diffusion_draws, seed=args.seed, exact_clean=clean, rows=rows), f, indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
