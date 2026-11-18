#!/usr/bin/env python3
"""Run T: FKeras masks for the W8A8 language models (DistilBERT; Pythia when memory allows).

Continuous companion, as for the TFLite models: the loaded W8A8 model in float64, each QLinear computing
x @ (scale * q + u)^T + b with NO activation quantization, where u is a zero offset on the decoded weights.  The
Hessian of the mean calibration loss with respect to u (= the weight Hessian) is applied by double backpropagation;
two signed largest-magnitude eigenpairs come from the same residual-checked Lanczos solver, options and seed as the
other FKeras masks.  Calibration: the selection cohort of the frozen campaign (the inputs of the main-grid masks).
Word score: the published projection |sum_i lambda_i (v_i . theta) v_i| with theta the decoded weights.
    fkeras            bit tier first (sign bit first), then word score, then global code order (as published)
    fkeras_weighted   |word score| x 2^bit over all bits (the FKeras source's rank_bits; defined but unused there)
Budgets are those of the campaign's MSB-first policies.  Writes <out>/<target>/fkeras_masks.npz (packed per code)
and fkeras.json (write-once).

    runT_fkeras_w8a8.py --target distilbert_sst2_int8 [--out DIR] [--threads 16]
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runT_w8a8 as T  # noqa: E402

FK_SEED, FK_OPTIONS = 970201, dict(eigenpairs=2, max_iterations=100, tolerance=1e-5, max_products=300)


def keep_float64(model):
    """The Hessian needs float64 end to end: the Hugging Face attention computes softmax in float32 (Pythia, Qwen) and
    the Qwen RMSNorm normalizes in float32.  Keep float64 when the input is float64 (precision only, same function)."""
    import torch.nn.functional as F
    if not getattr(F.softmax, "_fk64", False):
        orig = F.softmax

        def softmax(input, dim=None, _stacklevel=3, dtype=None):
            if dtype is torch.float32 and input.dtype is torch.float64:
                dtype = torch.float64
            return orig(input, dim=dim, _stacklevel=_stacklevel, dtype=dtype)
        softmax._fk64 = True
        F.softmax = softmax
    for m in model.modules():
        if type(m).__name__.endswith("RMSNorm"):
            eps = getattr(m, "variance_epsilon", getattr(m, "eps", 1e-6))

            def fwd(x, m=m, eps=eps):
                return m.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))
            m.forward = fwd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=["distilbert_sst2_int8", "pythia160m_int8"])
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out", type=Path, default=T.OUT)
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()
    out = a.out / a.target
    out.mkdir(parents=True, exist_ok=True)
    if (out / "fkeras_masks.npz").exists():
        raise SystemExit("write-once: fkeras_masks.npz exists")
    t0 = time.time()
    plan, adapter, _ = T.setup(a.target, a.threads)
    import torch
    sys.path.insert(0, str(T.W8A8))
    from netsurf.fkeras_baseline import leading_hessian_eigenpairs, principal_weight_hessian_score
    from kona_assets import int8_torch as execution
    names = [e["module"] for e in adapter.universe.entries]
    mods = [adapter.modules[n] for n in names]
    model = adapter.model.double()
    keep_float64(model)
    U = [torch.zeros(tuple(m.qweight.shape), dtype=torch.float64, requires_grad=True) for m in mods]
    for m, u in zip(mods, U):
        m._fk_u = u

    def continuous(self, x):
        w = self.qweight.double() * self.scale.double()[:, None] + self._fk_u
        y = x.double() @ w.t()
        if self.bias is not None:
            y = y + self.bias.double()
        return y
    execution.QLinear.forward = continuous
    theta = np.concatenate([(m.qweight.double() * m.scale.double()[:, None]).reshape(-1).numpy() for m in mods])
    sizes = [u.numel() for u in U]
    torch.use_deterministic_algorithms(False)
    torch.set_num_threads(a.threads)

    if a.target == "distilbert_sst2_int8":
        coh = plan["selection_cohort"]                 # the 64 train sentences of the main-grid masks, in frozen order
        sys.path.insert(0, str(T.W8A8))
        from netsurf.transformer_data import read_sst2_parquet
        train = read_sst2_parquet(adapter.asset.dataset_root / "sst2" / "train-00000-of-00001.parquet", "train")
        by_index = {r.source_index: r for r in train}
        missing = [i for i in coh["source_indices"] if i not in by_index]
        if missing or coh.get("split") != "train":
            raise SystemExit(f"selection cohort not found in the train partitions ({len(missing)} missing)")
        rows = [by_index[i] for i in coh["source_indices"]]
        sys.path.insert(0, str(T.W8A8))
        from netsurf.transformer_data import sst2_records_digest
        if sst2_records_digest(rows) != coh["ordered_labels_and_text_sha256"]:
            raise SystemExit("selection cohort text or labels differ from the frozen campaign")
        batches = []
        for s in range(0, len(rows), a.batch):
            b = rows[s:s + a.batch]
            enc = adapter.tokenizer([r.text for r in b], return_tensors="pt", truncation=True, padding="max_length", max_length=128)
            batches.append((enc, torch.tensor([r.label for r in b])))
        count = len(rows)

        def batch_loss(enc, y):
            logits = model(**enc).logits.double()
            return torch.nn.functional.cross_entropy(logits, y, reduction="sum")
    else:
        coh = plan["selection_cohort"]
        ids = adapter.token_sequence("validation")[coh["input_start"]:coh["input_end_exclusive"]]
        from netsurf.transformer_risk import teacher_forced_windows
        batches = [(start, end, first) for start, end, first in teacher_forced_windows(ids, 1024, 512)]
        count = sum(end - first for _, end, first in batches)

        def batch_loss(item, _):
            start, end, first = item
            x = torch.tensor([ids[start:end]], dtype=torch.long)
            logits = model(input_ids=x, use_cache=False).logits[0].double()
            tgt = torch.tensor(ids[first:end], dtype=torch.long)
            return torch.nn.functional.cross_entropy(logits[first - start - 1:end - start - 1], tgt, reduction="sum")
        batches = [(b, None) for b in batches]

    def hvp(vec):
        v = torch.from_numpy(np.asarray(vec, dtype=np.float64))
        parts = [p.reshape(u.shape) for p, u in zip(torch.split(v, sizes), U)]
        total = np.zeros(len(theta))
        for item, y in batches:
            loss = batch_loss(item, y)
            g = torch.autograd.grad(loss, U, create_graph=True)
            dot = sum((gi * pi).sum() for gi, pi in zip(g, parts))
            hv = torch.autograd.grad(dot, U)
            total += torch.cat([h.reshape(-1) for h in hv]).numpy()
        return total / count

    vectors, diag = leading_hessian_eigenpairs(hvp, len(theta), count=FK_OPTIONS["eigenpairs"], seed=FK_SEED,
                                               max_iterations=FK_OPTIONS["max_iterations"],
                                               tolerance=FK_OPTIONS["tolerance"], max_products=FK_OPTIONS["max_products"])
    word = np.abs(principal_weight_hessian_score(theta, vectors, np.asarray(diag.eigenvalues)))
    codes = len(theta)
    budgets = {}
    for pol in plan["policies"]:
        if pol["name"].startswith("msb_first_b"):
            share = min(T.SHARES, key=lambda s: abs(s - pol["budget"] / (8 * codes)))
            budgets[share] = pol["budget"]
    masks = {}
    bitpos = np.arange(8)
    for share, budget in sorted(budgets.items()):
        order_word = np.argsort(-word, kind="stable")                       # word score, ties by global code order
        rank = np.empty(codes, np.int64); rank[order_word] = np.arange(codes)
        key = (7 - bitpos)[None, :] * codes + rank[:, None]                 # tier first, then word order
        masks[f"fkeras@{share}"] = T.N.pack_mask(key < budget)
        flat = (word[:, None] * (2.0 ** bitpos)[None, :]).reshape(-1)       # rank_bits: word score x 2^bit
        top = np.argpartition(-flat, budget - 1)[:budget]
        sel = np.zeros(flat.size, dtype=bool); sel[top] = True
        masks[f"fkeras_weighted@{share}"] = T.N.pack_mask(sel.reshape(codes, 8))
        del flat, sel
    for k, v in masks.items():
        assert int(T.N.POP[v].sum()) == budgets[float(k.split("@")[1])], k
    np.savez(out / "fkeras_masks.npz", **masks)
    (out / "fkeras.json").write_text(json.dumps(dict(
        target=a.target, calibration_units=count, seconds=round(time.time() - t0, 1),
        eigenvalues=[float(v) for v in diag.eigenvalues], residuals=[float(v) for v in diag.normalized_residuals],
        hessian_vector_products=int(diag.hvp_calls), options=FK_OPTIONS, seed=FK_SEED,
        word_score_sha256=hashlib.sha256(word.tobytes()).hexdigest(),
        masks_sha256={k: hashlib.sha256(v.tobytes()).hexdigest()[:16] for k, v in masks.items()}), indent=1))


if __name__ == "__main__":
    main()
