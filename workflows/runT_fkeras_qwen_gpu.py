#!/usr/bin/env python3
"""Run T: FKeras masks for Qwen3-0.6B (W8A8 artifact) on one GPU; standalone (torch, transformers, safetensors, scipy).

Same continuous companion and estimator as runT_fkeras_w8a8.py, without the CPU-pinned kona_assets loader:
  * the model skeleton comes from the artifact's config.json (eager attention), every manifest module is replaced by a
    continuous linear layer computing x @ (scale * q + u)^T (+ b) in float64 with NO activation quantization, u a zero
    offset on the decoded weights; every stored tensor is checked against the manifest's sha256;
  * the loss is the mean next-token cross-entropy over the frozen selection windows (token ids shipped from kona with
    their sha256; teacher-forced windows of 1,024 with stride 512, the campaign's rule);
  * the Hessian-vector product is double backpropagation on the GPU; two signed largest-magnitude eigenpairs come from
    the same residual-checked ARPACK solver (netsurf/fkeras_baseline.py, copied), options and seed as the other FKeras
    masks; word score |sum_i lambda_i (v_i . theta) v_i| with theta the decoded weights.
Masks (budgets from budgets.json = the campaign's MSB-first budgets), packed per code in global code order:
  fkeras@s           sign tier first, then lower tiers; within the partial tier, codes by word score (ties: code order)
  fkeras_weighted@s  top budget of |word| * 2^bit (the FKeras source's rank_bits); ties at the threshold by (code, bit)
Writes <out>/fkeras_masks.npz (compressed) and fkeras.json.

    runT_fkeras_qwen_gpu.py --model DIR --tokens tokens.npy --budgets budgets.json --out DIR [--check-only]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "pkg"))
from netsurf.fkeras_baseline import leading_hessian_eigenpairs, principal_weight_hessian_score  # noqa: E402,F401

FK_SEED, FK_OPTIONS = 970201, dict(eigenpairs=2, max_iterations=100, tolerance=1e-5, max_products=300)
SHARES = (0.0625, 0.125, 0.1875, 0.25)


class Continuous(torch.nn.Module):
    def __init__(self, q, s, bias):
        super().__init__()
        self.register_buffer("w0", q.double() * s.double()[:, None])
        self.bias = None if bias is None else torch.nn.Parameter(bias.double(), requires_grad=False)
        self.u = torch.nn.Parameter(torch.zeros_like(self.w0))

    def forward(self, x):
        y = x.double() @ (self.w0 + self.u).t()
        return y if self.bias is None else y + self.bias


def tensor_sha256(t):                      # kona_assets.int8_torch.tensor_sha256
    return hashlib.sha256(t.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def load(model_dir: Path, device):
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM
    manifest = json.loads((model_dir / "quantization_manifest.json").read_text())
    cfg = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float32, attn_implementation="eager").eval()
    sd = load_file(str(model_dir / "model.safetensors"))
    mods = []
    for e in manifest["tensors"]:
        q, s = sd.pop(e["qweight_name"]), sd.pop(e["scale_name"])
        if tensor_sha256(q) != e["qweight_sha256"] or tensor_sha256(s) != e["scale_sha256"]:
            raise SystemExit(f"stored bytes differ from the manifest: {e['module']}")
        bias = sd.pop(e["module"] + ".bias", None) if e["has_bias"] else None
        m = Continuous(q, s, bias)
        parent, _, leaf = e["module"].rpartition(".")
        setattr(model.get_submodule(parent) if parent else model, leaf, m)
        mods.append(m)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        raise SystemExit(f"unexpected tensors: {unexpected[:5]}")
    allowed = {k for k in missing if k.endswith((".w0", ".u", ".bias"))}
    if set(missing) - allowed:
        raise SystemExit(f"missing tensors: {sorted(set(missing) - allowed)[:5]}")
    model = model.double().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    for m in mods:
        m.u.requires_grad_(True)
    return model, mods, manifest


def windows(n, context=1024, stride=512):
    """(start, end, first scored position) as in the campaign's teacher_forced_windows."""
    out, prev_end = [], 0
    for start in range(0, n, stride):
        end = min(start + context, n)
        first = max(start + 1, prev_end)
        if first < end:
            out.append((start, end, first))
        prev_end = end
        if end == n:
            break
    return out


def lanczos_eigenpairs(product, n, *, count, seed, tolerance, max_products, max_basis=48, check_every=6):
    """Largest-magnitude eigenpairs of a symmetric operator by Lanczos with full reorthogonalization (64-bit sizes).
    Same acceptance rules as netsurf.fkeras_baseline.leading_hessian_eigenpairs, which uses ARPACK; ARPACK's 32-bit
    work-array indexing overflows at n * ncv > 2^31 (Qwen: n = 596M).  Symmetry check on two random vectors, ordering
    by descending |lambda| (algebraic value breaks ties), sign rule (largest absolute component nonnegative), and
    acceptance only if every normalized residual ||Hx - lambda x|| / (||Hx|| + |lambda||) <= tolerance, with the
    residual computed from an explicit product."""
    rng = np.random.default_rng(seed)
    u, v = rng.normal(size=(2, n)); u /= np.linalg.norm(u); v /= np.linalg.norm(v)
    hu, hv = product(u), product(v)
    symmetry = float(abs(u @ hv - v @ hu) / max(np.linalg.norm(hu), np.linalg.norm(hv), 1e-30))
    if symmetry > max(tolerance, 1e-10):
        raise ValueError(f"HVP symmetry check failed: {symmetry}")
    del u, hu, hv
    Q = [v / np.linalg.norm(v)]
    alphas, betas = [], []
    calls = 2
    while True:
        w = product(Q[-1]); calls += 1
        a = float(Q[-1] @ w); alphas.append(a)
        for q in Q:                                    # full reorthogonalization, twice for stability
            w -= (q @ w) * q
        for q in Q:
            w -= (q @ w) * q
        b = float(np.linalg.norm(w))
        k = len(alphas)
        if k >= count and (k % check_every == 0 or k >= max_basis or b < 1e-14 or calls + count + 1 > max_products):
            T = np.diag(alphas) + np.diag(betas, 1) + np.diag(betas, -1)
            vals, S = np.linalg.eigh(T)
            order = sorted(range(len(vals)), key=lambda i: (-abs(vals[i]), -vals[i]))[:count]
            est = [abs(b * S[-1, i]) / max(abs(vals[i]), 1e-30) for i in order]   # Lanczos residual estimate
            print(json.dumps(dict(stage="lanczos", basis=k, ritz=[float(vals[i]) for i in order], est=est)), flush=True)
            if max(est) <= tolerance * 0.1 or k >= max_basis or b < 1e-14 or calls + count + 1 > max_products:
                vectors = np.stack([sum(S[j, i] * Q[j] for j in range(k)) for i in order])
                values = np.array([vals[i] for i in order])
                break
        if b < 1e-14:
            raise ValueError("Lanczos breakdown before convergence")
        betas.append(b)
        Q.append(w / b)
    del Q
    for x in vectors:
        x /= np.linalg.norm(x)
        if x[np.argmax(np.abs(x))] < 0:
            x *= -1
    products = np.stack([product(x) for x in vectors]); calls += count
    residuals = np.linalg.norm(products - values[:, None] * vectors, axis=1)
    normalized = residuals / np.maximum(np.linalg.norm(products, axis=1) + np.abs(values), 1e-30)
    orthogonality = float(np.linalg.norm(vectors @ vectors.T - np.eye(count), ord=2))
    if np.any(normalized > tolerance) or orthogonality > max(tolerance, 1e-10):
        raise ValueError(f"FKeras eigenpair validation failed: residual={normalized}, orthogonality={orthogonality}")
    diag = dict(eigenvalues=[float(x) for x in values], normalized_residuals=[float(x) for x in normalized],
                orthogonality_error=orthogonality, symmetry_error=symmetry, hvp_calls=calls,
                solver="numpy-lanczos-full-reorthogonalization/v1", basis=len(alphas))
    return vectors, diag


def tier_first(word, budget):
    codes = word.size
    full, rem = divmod(budget, codes)
    m = np.zeros((codes, 8), dtype=bool)
    for k in range(full):
        m[:, 7 - k] = True
    if rem:
        order = np.argsort(-word, kind="stable")[:rem]
        m[order, 7 - full] = True
    return m


def weighted(word, budget):
    """Top `budget` of word[c] * 2^b over (c, b); ties at the threshold broken by (code, bit)."""
    codes = word.size
    srt = np.sort(word)                                                  # ascending

    def count_gt(t):
        return sum(int(codes - np.searchsorted(srt, t / 2.0 ** b, side="right")) for b in range(8))
    lo, hi = 0.0, float(word.max()) * 128.0
    for _ in range(200):                                                 # bisection on the threshold value
        mid = (lo + hi) / 2
        if count_gt(mid) >= budget:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-15 * max(hi, 1e-300):
            break
    t = hi                                                               # count_gt(t) < budget <= count_gt(lo)
    m = np.zeros((codes, 8), dtype=bool)
    for b in range(8):
        m[:, b] = word * 2.0 ** b > t
    need = budget - int(m.sum())
    if need > 0:                                                         # values in (lo, t]: fill by (code, bit)
        for b in range(8):
            cand = np.flatnonzero((word * 2.0 ** b > lo) & ~m[:, b])
            take = cand[:need]
            m[take, b] = True
            need -= take.size
            if need == 0:
                break
    assert int(m.sum()) == budget
    return m


def pack(m):
    return np.packbits(m, axis=1, bitorder="little")[:, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--tokens", type=Path, required=True)
    ap.add_argument("--budgets", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--check-only", action="store_true", help="load, one loss and one HVP, then stop")
    ap.add_argument("--max-windows", type=int, default=0, help="checks only: use the first N windows")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if (a.out / "fkeras_masks.npz").exists():
        raise SystemExit("write-once: fkeras_masks.npz exists")
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False
    meta = json.loads(a.tokens.with_suffix(".json").read_text())
    ids = np.load(a.tokens)
    if hashlib.sha256(ids.astype("<i4").tobytes()).hexdigest() != meta["token_ids_sha256"]:
        raise SystemExit("token ids differ from their recorded sha256")
    model, mods, manifest = load(a.model, device)
    U = [m.u for m in mods]
    sizes = [u.numel() for u in U]
    theta = np.concatenate([m.w0.reshape(-1).cpu().numpy() for m in mods])
    W = [tuple(w) for w in meta["windows"]]                # the campaign's teacher_forced_windows, computed on kona
    assert W[-1][1] == len(ids)
    if a.max_windows:
        W = W[:a.max_windows]
    count = sum(end - first for _, end, first in W)
    log = lambda **k: print(json.dumps(dict(seconds=round(time.time() - t0), **k)), flush=True)  # noqa: E731
    log(stage="loaded", device=device, codes=int(theta.size), windows=len(W), targets=count)

    def loss_of(start, end, first):
        x = torch.tensor(ids[start:end][None, :], dtype=torch.long, device=device)
        logits = model(input_ids=x, use_cache=False).logits[0]
        tgt = torch.tensor(ids[first:end], dtype=torch.long, device=device)
        return torch.nn.functional.cross_entropy(logits[first - start - 1:end - start - 1], tgt, reduction="sum")

    calls = [0]

    def hvp(vec):
        v = torch.from_numpy(np.asarray(vec, dtype=np.float64)).to(device)
        parts = [p.reshape(u.shape) for p, u in zip(torch.split(v, sizes), U)]
        total = torch.zeros(theta.size, dtype=torch.float64, device=device)
        for w in W:
            g = torch.autograd.grad(loss_of(*w), U, create_graph=True)
            dot = sum((gi * pi).sum() for gi, pi in zip(g, parts))
            hv = torch.autograd.grad(dot, U)
            total += torch.cat([h.reshape(-1) for h in hv])
            del g, dot, hv
        calls[0] += 1
        log(stage="hvp", call=calls[0])
        return (total / count).cpu().numpy()

    if a.check_only:
        with torch.no_grad():
            first_loss = float(loss_of(*W[0])) / (W[0][1] - W[0][2])
        r = np.random.default_rng(0).normal(size=theta.size)
        hv = hvp(r)
        log(stage="check", mean_nll_first_window=first_loss, hvp_norm=float(np.linalg.norm(hv)),
            gpu_peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2) if device == "cuda" else None)
        return
    vectors, diag = lanczos_eigenpairs(hvp, theta.size, count=FK_OPTIONS["eigenpairs"], seed=FK_SEED,
                                       tolerance=FK_OPTIONS["tolerance"], max_products=FK_OPTIONS["max_products"])
    word = np.abs(principal_weight_hessian_score(theta, vectors, np.asarray(diag["eigenvalues"])))
    del vectors
    budgets = {float(k): int(v) for k, v in json.loads(a.budgets.read_text()).items()}
    masks = {}
    for share, budget in sorted(budgets.items()):
        masks[f"fkeras@{share}"] = pack(tier_first(word, budget))
        masks[f"fkeras_weighted@{share}"] = pack(weighted(word, budget))
        log(stage="masks", share=share)
    np.savez_compressed(a.out / "fkeras_masks.npz", **masks)
    (a.out / "fkeras.json").write_text(json.dumps(dict(
        target="qwen3_0_6b_int8", calibration_targets=count, windows=len(W), seconds=round(time.time() - t0, 1),
        device=device, gpu=torch.cuda.get_device_name(0) if device == "cuda" else None,
        torch=torch.__version__, eigensolver=diag,
        options=FK_OPTIONS, seed=FK_SEED, tokens_sha256=meta["token_ids_sha256"],
        word_score_sha256=hashlib.sha256(word.tobytes()).hexdigest(),
        masks_sha256={k: hashlib.sha256(v.tobytes()).hexdigest()[:16] for k, v in masks.items()}), indent=1))
    log(stage="done")


if __name__ == "__main__":
    main()
