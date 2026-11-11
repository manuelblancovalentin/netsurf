#!/usr/bin/env python3
"""Bounded offline engineering check (no inference): residual first-order drift under each
frozen DistilBERT W8A8 mask.

For a mask P and BER p, the unprotected coordinates fault independently, so the
first-order predicted mean loss change is  p * D1(P),  D1(P) = sum_{b not in P} g_b * delta_b,
with g the saved mean calibration gradient and delta_b the exact INT8 XOR jump of the
stored code.  We also report the one-sided parts, the residual jump energy
sum delta_b^2 (a curvature-scale proxy), and the mask anatomy by bit position.
Reads the frozen checkpoint, saved moments and saved masks read-only; writes one new JSON.
"""
import json, struct, sys, hashlib
from pathlib import Path
import numpy as np
ROOT = Path("/data/models/local--kona-assets--distilbert-sst2-int8/snapshots/1f07e0012b060f3658fa139839dab054478c5121")
SEL = Path("/projects/netsurf/manuelbv/workspace/implementation-worktrees/transformer-integration/workspace/runs/distilbert_sst2_w8a8_reliability_v2_run1/selection")
BUDGETS = {0.0625: 21529344, 0.125: 43058688, 0.1875: 64588032, 0.25: 86117376}
out = Path(sys.argv[1])
qm = json.loads((ROOT / "quantization_manifest.json").read_text())
mom = json.loads((SEL / "moments/manifest.json").read_text())
with open(ROOT / "model.safetensors", "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]; header = json.loads(f.read(n)); base = 8 + n
st = np.memmap(ROOT / "model.safetensors", dtype=np.uint8, mode="r")
def tensor(name, dtype):
    h = header[name]; a, b = h["data_offsets"]
    return np.frombuffer(st[base + a: base + b], dtype=dtype).reshape(h["shape"])
codes, scales, g_all = [], [], []
for t in qm["tensors"]:
    q = tensor(t["qweight_name"], np.int8); s = tensor(t["scale_name"], np.float32)
    assert hashlib.sha256(q.tobytes()).hexdigest() == t["qweight_sha256"], "stored codes changed"
    item = mom["tensors"][t["module"]]
    g = np.fromfile(SEL / "moments" / item["mean_g"], dtype="<f4")
    assert g.size == q.size
    codes.append(q.reshape(-1)); scales.append(np.repeat(s.reshape(-1), q.shape[1])); g_all.append(g)
q = np.concatenate(codes); sc = np.concatenate(scales); g = np.concatenate(g_all).astype(np.float64)
N = q.size
assert N * 8 == 344469504
u = q.view(np.uint8)
delta = np.empty((N, 8), dtype=np.float32)          # exact decoded jump for each stored bit
for bit in range(8):
    changed = np.bitwise_xor(u, np.uint8(1 << bit)).view(np.int8)
    delta[:, bit] = (changed.astype(np.int16) - q.astype(np.int16)).astype(np.float32) * sc
e = g[:, None] * delta                              # signed first-order effect per bit
def masks():
    yield "unprotected", None, np.zeros((N, 8), dtype=bool)
    for cov, B in BUDGETS.items():
        m = np.zeros((N, 8), dtype=bool); tiers, rem = divmod(B, N)
        for k in range(tiers): m[:, 7 - k] = True
        if rem: m[:rem, 7 - tiers] = True            # canonical global-ordinal tie-break
        yield "msb", cov, m
        for meth in ("random", "numerical_bit_jump", "global_first_order", "per_example_absolute", "per_example_positive"):
            raw = np.fromfile(SEL / "masks" / meth / f"{meth}_b{B}.bits", dtype=np.uint8)
            m = np.unpackbits(raw[:, None], axis=1, bitorder="little").astype(bool)
            assert m.sum() == B
            yield meth, cov, m
rows = []
for name, cov, m in masks():
    r = ~m
    er = np.where(r, e, 0.0)
    d2 = np.where(r, delta.astype(np.float64) ** 2, 0.0)
    rows.append(dict(method=name, coverage=cov, protected_bits=int(m.sum()),
        D1_signed=float(er.sum()), D1_positive_part=float(np.clip(er, 0, None).sum()), D1_negative_part=float(np.clip(er, None, 0).sum()),
        asymmetry=float(er.sum() / (np.abs(er).sum() + 1e-300)),
        residual_jump_energy=float(d2.sum()),
        residual_sign_bits_fraction=float(r[:, 7].mean()), residual_bit6_fraction=float(r[:, 6].mean()),
        protected_fraction_by_bit=[float(m[:, b].mean()) for b in range(8)],
        protected_share_harmful_direction=float((m & (e > 0)).sum() / max(1, m.sum())),
        protected_share_beneficial_direction=float((m & (e < 0)).sum() / max(1, m.sum()))))
    print(json.dumps(rows[-1]))
with open(out, "x") as f:
    json.dump(dict(schema="paper1-tecs-residual-drift/1.0", target="distilbert_sst2_int8", codes=int(N),
                   note="first-order predicted mean loss change at BER p is p*D1_signed; surrogate FP32 decoded-weight gradient on 64 training-split calibration sentences",
                   rows=rows), f, indent=1)
