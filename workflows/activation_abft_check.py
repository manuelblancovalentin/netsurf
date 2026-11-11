#!/usr/bin/env python3
"""Run H (EXPLORATORY; frozen before any outcome): stored-weight faults versus activation faults at the same
per-bit probability on DistilBERT W8A8, and what a per-layer checksum (ABFT) reports under each.

Why.  A reviewer of the earlier submission objected that "weights are persistent, activations are transient"
was asserted, not shown, and asked for a comparison with ABFT.  This run measures both on one model.

Fault classes, each with independent bit flips of probability p per bit:
  W  stored INT8 weight codes, drawn once per pass and kept for all 512 sentences (persistent);
  A  the INT8 activation codes entering every Linear layer, drawn anew for every forward call (transient).
Equal p for W and A is an equal-per-cell comparison (defective or low-voltage memory).  Under an
exposure-time model a transient value is at risk only while it is held, so its per-inference probability is
far lower; that conversion is an analysis step, not part of this run.

Conditions per p (two independent draws each):
  A_only            clean weights, activation faults
  W_only            weight faults, no protection
  W_only_masked     weight faults, sealed absolute-gradient mask at 12.5%
  W_and_A           both, no protection
  W_masked_and_A    both, weights protected by the 12.5% mask
p in the study grid: 1e-5 ... 1e-1.  Cohort: the study's frozen 512 sentences.

Checksum (ABFT) check, the standard output-sum form for a linear layer: before deployment store, for each
Linear layer, the column sums of the fault-free weight matrix.  At run time compare the sum of the layer's
integer outputs with (activation codes) . (stored column sums) for every token.  The stored sums are assumed
fault-free; the check reads the same activation codes as the multiplier.  Reported: the fraction of sentences
with at least one failed check, and the fraction of all (layer, token) checks that fail.  The check only
detects; no recovery is modelled.

Arithmetic: the float path of the drift prototypes, exact for integer codes (it reproduces the integer
executable on the clean model, which is asserted first).  No evaluation sentence is used to build anything.
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
CONDITIONS = ("A_only", "W_only", "W_only_masked", "W_and_A", "W_masked_and_A")
DRAWS, SEED = 2, 20261116


class State:
    act_ber = 0.0
    gen = None
    check = False
    batch_flag = None
    sentence_flags: list = []
    checks = 0
    failed = 0
    act_bits = 0


def close_batch():
    if State.batch_flag is not None:
        State.sentence_flags.extend(State.batch_flag.tolist())
    State.batch_flag = None


def install(torch, execution, model):
    """Replace QLinear.forward by an exact float path with optional activation faults and checksum checks."""
    model.register_forward_pre_hook(lambda module, inputs: close_batch())     # one model call = one batch of sentences

    def forward(self, x):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).float()
        a = (x2.abs().amax(dim=1, keepdim=True) / 127).clamp_min(float(execution.spec.SCALE_FLOOR))
        xq = torch.round(x2 / a).clamp(execution.spec.CODE_MIN, execution.spec.CODE_MAX)
        if State.batch_flag is None:
            State.batch_flag = torch.zeros(shape[0], dtype=torch.bool)
        if State.act_ber > 0:
            codes = xq.to(torch.int8).view(-1)
            State.act_bits += 8 * codes.numel()
            p = State.act_ber
            if p >= 3e-3:
                m = torch.zeros(codes.numel(), dtype=torch.uint8)
                for bit in range(8):
                    m |= (torch.rand(codes.numel(), generator=State.gen) < p).to(torch.uint8) << bit
            else:                                   # sparse draw: Poisson number of flipped bits, uniform positions
                n = int(torch.poisson(torch.tensor(8.0 * codes.numel() * p), generator=State.gen))
                pos = torch.randint(0, 8 * codes.numel(), (n,), generator=State.gen)
                m = torch.zeros(codes.numel(), dtype=torch.uint8)
                for bit in range(8):
                    sel = pos[pos % 8 == bit] // 8
                    m[sel] ^= (1 << bit)
            xq = (codes.view(torch.uint8) ^ m).view(torch.int8).float().reshape(xq.shape)
        delta = getattr(self, "_mf_delta", None)
        w = self.qweight.float() if delta is None else self.qweight.float() + delta
        acc = xq @ w.t()
        if State.check:
            bad = acc.double().sum(dim=1) != xq.double() @ self._golden_colsum
            State.checks += bad.numel(); State.failed += int(bad.sum())
            State.batch_flag |= bad.reshape(shape[0], -1).any(dim=1)
        y = acc * a * self.scale[None, :]
        if self.bias is not None:
            y = y + self.bias.float()
        return y.reshape(*shape[:-1], self.out_features)

    execution.QLinear.forward = forward


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ber-index", type=int, required=True)
    args = ap.parse_args()
    p = BERS[args.ber_index]
    target = args.out / f"ber_{args.ber_index}.json"
    assert args.out.is_dir() and not target.exists()
    torch, plan, adapter, evaluate = mf.setup()            # installs the prototype forward; replaced below
    from kona_assets import int8_torch as execution
    modules, n_codes = mf.per_module_arrays(adapter)
    qmods = [adapter.modules[m["name"]] for m in modules]
    for qm in qmods:
        qm._mf_delta = None
        qm._golden_colsum = qm.qweight.double().sum(dim=0)
    exact_clean = dict(evaluate().metrics)                  # prototype float path, zero drift == integer executable (checked in v1)
    install(torch, execution, adapter.model)
    State.check = True
    clean = dict(evaluate().metrics); close_batch()
    assert len(State.sentence_flags) == 512, len(State.sentence_flags)
    assert clean == exact_clean and State.failed == 0, "float path or checksum is not exact on the clean model"
    budget = int(round(0.125 * plan["eligible_bits"]))
    masks = {"none": mf.mask_bits("unprotected", 0, n_codes), "abs": mf.mask_bits("per_example_absolute", budget, n_codes)}
    jumps = {k: [torch.from_numpy(np.where(pm[m["start"]: m["start"] + m["n"]], 0, m["jumps"]).astype(np.int16)) for m in modules]
             for k, pm in masks.items()}
    rows = []
    for cond in CONDITIONS:
        for draw in range(DRAWS):
            gen = torch.Generator().manual_seed(SEED + 100000 * args.ber_index + 10 * CONDITIONS.index(cond) + draw)
            applied = 0
            if cond.startswith("W"):
                jm = jumps["abs" if "masked" in cond else "none"]
                wgen = torch.Generator().manual_seed(SEED + 100000 * args.ber_index + 7000 + draw)   # same weight field for all W conditions
                for qm, j in zip(qmods, jm):
                    real = torch.zeros(j.shape[0], dtype=torch.float32)
                    for bit in range(8):
                        hit = torch.rand(j.shape[0], generator=wgen) < p
                        col = j[:, bit].float()
                        real += hit.float() * col
                        applied += int((hit & (col != 0)).sum())
                    qm._mf_delta = real.reshape(qm.qweight.shape)
            else:
                for qm in qmods:
                    qm._mf_delta = None
            State.act_ber = p if cond.endswith("A") or cond == "A_only" else 0.0
            State.gen = gen; State.batch_flag = None; State.sentence_flags = []; State.checks = State.failed = State.act_bits = 0
            started = time.monotonic()
            metrics = dict(evaluate().metrics)
            close_batch()
            row = dict(condition=cond, ber=p, draw=draw, accuracy=metrics["accuracy"], mean_nll=metrics["mean_nll"],
                       weight_faults_applied=applied, activation_bits_exposed=State.act_bits,
                       sentences=len(State.sentence_flags), sentences_flagged=int(sum(State.sentence_flags)),
                       checks=State.checks, checks_failed=State.failed, seconds=time.monotonic() - started)
            rows.append(row)
            print(json.dumps(row), flush=True)
    with open(target, "x") as f:
        json.dump(dict(schema="paper1-runH-activations-abft/1.0", status="EXPLORATORY", ber=p, clean=clean,
                       eligible_weight_bits=plan["eligible_bits"], rows=rows), f, indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
