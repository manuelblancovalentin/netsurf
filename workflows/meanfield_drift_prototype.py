#!/usr/bin/env python3
"""EXPLORATORY prototype (not a finalized study): mean-field "drift" objective for protection masks.

Under independent stored-bit faults with probability p and a protection mask P, every stored
INT8 code has an exactly known expected value

    E[q'] = q + p * sum_{unprotected bits j} (jump of bit j, in code units).

The mean-field model replaces each stored weight by that expectation and runs ONE forward pass.
It captures the coherent, mask-dependent mean displacement ("drift") of the weights and ignores
the zero-mean fluctuation around it.  This script

  stage "score":  evaluates the mean-field loss/accuracy of the existing frozen DistilBERT masks at
                  every BER of the main grid, plus two cheap closed-form summaries (first-order
                  drift g.m and a diffusion proxy sum (E|g|)^2 * jump^2 over unprotected bits),
                  so they can be compared with the measured Monte Carlo outcomes offline.

The executable's arithmetic is kept (per-token INT8 activation quantization, per-channel scales);
only the integer matmul is replaced by a float matmul on the fractional mean codes.  With zero
drift this reproduces the clean executable up to float32 rounding, which is checked first.

Read-only inputs: the frozen W8A8 snapshot, the sealed masks and saved gradient moments of the
selection run, and the frozen 512-sentence final cohort.  No fault field is sampled here.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

W = Path("/projects/netsurf/manuelbv/workspace/implementation-worktrees")
CODE_ROOT = W / "w8a8-ber-stress"
MANIFEST = Path("/projects/kona-assets/manuelbv/NETSURF_ASSET_READINESS.json")
SEL = W / "transformer-integration/workspace/runs/distilbert_sst2_w8a8_reliability_v2_run1/selection"
SOURCE_RUN = W / "w8a8-ber-stress/workspace/runs/w8a8_wide_ber_matched_v1/distilbert_sst2_int8"
TARGET = "distilbert_sst2_int8"
METHODS = ("random", "numerical_bit_jump", "global_first_order", "per_example_absolute", "per_example_positive")


def setup():
    sys.path.insert(0, str(CODE_ROOT))
    import torch
    from netsurf import w8a8_wide_ber_campaign as wide
    from netsurf.w8a8_assets import KONA_ROOT
    if str(KONA_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(KONA_ROOT / "src"))
    from kona_assets import int8_torch as execution
    from netsurf.w8a8_campaign import _evaluation
    from netsurf.w8a8_faults import W8A8FaultAdapter
    plan, asset = wide.validate(MANIFEST, SOURCE_RUN, verify_masks=False)
    adapter = W8A8FaultAdapter(asset)
    adapter.load()
    evaluate = _evaluation(adapter, {"target": TARGET, "final": plan["final_cohort"]})
    original_forward = execution.QLinear.forward

    def meanfield_forward(self, x):
        delta = getattr(self, "_mf_delta", None)
        if delta is None:
            return original_forward(self, x)
        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).float()
        a = (x2.abs().amax(dim=1, keepdim=True) / 127).clamp_min(float(execution.spec.SCALE_FLOOR))
        xq = torch.round(x2 / a).clamp(execution.spec.CODE_MIN, execution.spec.CODE_MAX)
        acc = xq @ (self.qweight.float() + delta).t()          # fractional mean codes
        y = acc * a * self.scale[None, :]
        if self.bias is not None:
            y = y + self.bias.float()
        return y.reshape(*shape[:-1], self.out_features)

    execution.QLinear.forward = meanfield_forward
    return torch, plan, adapter, evaluate


def per_module_arrays(adapter):
    """Stored codes, exact per-bit jumps (code units), scales and saved gradient moments per module."""
    mom = json.loads((SEL / "moments/manifest.json").read_text())
    out = []
    offset = 0
    for entry in adapter.universe.entries:
        module = adapter.modules[entry["module"]]
        q = module.qweight.reshape(-1).numpy().copy()
        u = q.view(np.uint8)
        jumps = np.empty((q.size, 8), dtype=np.int16)
        for bit in range(8):
            jumps[:, bit] = np.bitwise_xor(u, np.uint8(1 << bit)).view(np.int8).astype(np.int16) - q.astype(np.int16)
        scale = np.repeat(module.scale.numpy(), module.qweight.shape[1]).astype(np.float64)
        item = mom["tensors"][entry["module"]]
        g = np.fromfile(SEL / "moments" / item["mean_g"], dtype="<f4").astype(np.float64)
        a = np.fromfile(SEL / "moments" / item["mean_abs_g"], dtype="<f4").astype(np.float64)
        out.append(dict(name=entry["module"], shape=tuple(module.qweight.shape), start=offset, n=q.size,
                        jumps=jumps, scale=scale, g=g, a=a))
        offset += q.size
    return out, offset


def mask_bits(name: str, budget: int, n_codes: int) -> np.ndarray:
    """Protected-bit indicator, shape (codes, 8); bit j of byte k is bit j of code k."""
    if name == "unprotected":
        return np.zeros((n_codes, 8), dtype=bool)
    if name == "msb_first":
        m = np.zeros((n_codes, 8), dtype=bool)
        tiers, rem = divmod(budget, n_codes)
        for k in range(tiers):
            m[:, 7 - k] = True
        if rem:
            m[:rem, 7 - tiers] = True
        return m
    raw = np.fromfile(SEL / "masks" / name / f"{name}_b{budget}.bits", dtype=np.uint8)
    m = np.unpackbits(raw[:, None], axis=1, bitorder="little").astype(bool)
    assert m.sum() == budget
    return m


def summarize(protected: np.ndarray, modules) -> tuple[list[np.ndarray], dict]:
    """Per-module net unprotected jump (code units) and closed-form drift/diffusion summaries."""
    net, g_dot_m, diffusion, energy = [], 0.0, 0.0, 0.0
    for mod in modules:
        unprot = ~protected[mod["start"]: mod["start"] + mod["n"]]
        j = np.where(unprot, mod["jumps"], 0)
        d = j.sum(axis=1, dtype=np.int32)                         # sum of unprotected jumps per weight
        v = (j.astype(np.int64) ** 2).sum(axis=1)                 # sum of squared jumps per weight
        net.append(d.astype(np.float32).reshape(mod["shape"]))
        g_dot_m += float((mod["g"] * mod["scale"] * d).sum())
        diffusion += float((mod["a"] ** 2 * mod["scale"] ** 2 * v).sum())
        energy += float((mod["scale"] ** 2 * v).sum())
    return net, dict(first_order_drift_per_unit_p=g_dot_m, diffusion_proxy_per_unit_p=diffusion,
                     jump_energy_per_unit_p=energy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.125, 0.25])
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch, plan, adapter, evaluate = setup()
    modules, n_codes = per_module_arrays(adapter)
    assert n_codes * 8 == plan["eligible_bits"]
    qmods = [adapter.modules[m["name"]] for m in modules]

    def run(net, p):
        for index, qm in enumerate(qmods):
            qm._mf_delta = None if net is None else torch.from_numpy(net[index] * np.float32(p))
        started = time.monotonic()
        risk = evaluate()
        return dict(risk.metrics), time.monotonic() - started

    rows = []
    for qm in qmods:
        qm._mf_delta = None
    exact_clean, _ = run(None, 0.0)                                        # integer executable
    zero = [np.zeros(m["shape"], dtype=np.float32) for m in modules]
    float_clean, seconds = run(zero, 0.0)                                  # float path, zero drift
    print(json.dumps({"exact_clean": exact_clean, "float_path_zero_drift": float_clean, "seconds": seconds}), flush=True)
    policies = [("unprotected", 0, 0.0)]
    for frac in args.budgets:
        budget = int(round(frac * plan["eligible_bits"]))
        policies += [(name, budget, frac) for name in ("msb_first",) + METHODS]
    for name, budget, frac in policies:
        protected = mask_bits(name, budget, n_codes)
        net, summary = summarize(protected, modules)
        del protected
        for p in plan["bers"]:
            metrics, seconds = run(net, p)
            row = dict(policy=name, coverage=frac, budget_bits=budget, ber=p,
                       meanfield_accuracy=metrics["accuracy"], meanfield_nll=metrics["mean_nll"],
                       meanfield_failure_fraction=metrics.get("failure_fraction", 0.0),
                       seconds=seconds, **summary)
            rows.append(row)
            print(json.dumps(row), flush=True)
    for qm in qmods:
        qm._mf_delta = None
    with open(args.out / "meanfield_scores.json", "x") as f:
        json.dump(dict(schema="paper1-meanfield-drift-prototype/1.0", status="EXPLORATORY",
                       target=TARGET, exact_clean=exact_clean, float_path_zero_drift=float_clean,
                       note="mean-field forward at expected stored codes; no fault field sampled", rows=rows), f, indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
