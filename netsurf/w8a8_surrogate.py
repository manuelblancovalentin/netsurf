"""Whole-qweight FP32 surrogate moments for static W8A8 bit rankings.

The authoritative W8A8 inference remains integer `torch._int_mm`.  This
module builds a separate differentiable model whose Linear weights are exactly
decoded from the stored INT8 codes and fixed per-row scales.  It never calls
the surrogate an integer-network derivative.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .manifests import sha256_file
from .w8a8_faults import W8A8FaultAdapter


def build_surrogate_moments(adapter: W8A8FaultAdapter, output: Path, *,
                            sst2_examples: int = 64, lm_windows: int = 8) -> dict:
    """Write exactly two float32 moment streams per stored code, write-once.

    For language models an example is a disjoint 1,023-target WikiText
    validation window.  Per-example estimators are per-window, not per-token.
    No final-evaluation input enters this function.
    """
    if adapter.model is None or adapter.modules is None or Path(output).exists():
        raise ValueError("loaded W8A8 model and unused output directory required")
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from kona_assets import int8_torch

    if adapter.asset.target == "distilbert_sst2_int8":
        rows = adapter.sst2_cohorts()["calibration"][:sst2_examples]
        if len(rows) != sst2_examples or sst2_examples < 2:
            raise ValueError("insufficient disjoint SST-2 calibration records")
        examples = [(adapter.tokenizer([row.text], return_tensors="pt", padding="max_length",
                                       truncation=True, max_length=128), row.label) for row in rows]
        cohort = {"split": "SST-2 train", "source_indices": [r.source_index for r in rows],
                  "labels": [r.label for r in rows]}
    else:
        ids = adapter.token_sequence("validation")
        selected = ids[1024:1024 + lm_windows * 1023 + 1]
        if lm_windows < 2 or len(selected) < lm_windows * 1023 + 1:
            raise ValueError("insufficient disjoint WikiText validation tokens")
        examples = [torch.tensor([selected[i*1023:i*1023+1024]], dtype=torch.long)
                    for i in range(lm_windows)]
        cohort = {"split": "WikiText-2 raw validation", "windows": lm_windows,
                  "tokens_per_window": 1024, "scored_targets_per_window": 1023,
                  "token_ids_sha256": hashlib.sha256(np.asarray(selected,
                                                        dtype="<i4").tobytes()).hexdigest()}

    # Replace each QLinear in this ranking-only process with an FP32 Linear
    # initialized from exactly decoded qweight * scale.  Final evaluation must
    # load a fresh authoritative INT8 model in a separate process.
    linear = {}
    for name, q in adapter.modules.items():
        layer = nn.Linear(q.in_features, q.out_features, bias=q.bias is not None)
        with torch.no_grad():
            layer.weight.copy_(q.qweight.float() * q.scale[:, None])
            if q.bias is not None:
                layer.bias.copy_(q.bias)
        parent, _, leaf = name.rpartition(".")
        setattr(adapter.model.get_submodule(parent) if parent else adapter.model, leaf, layer)
        linear[name] = layer
    for parameter in adapter.model.parameters():
        parameter.requires_grad_(False)
    for layer in linear.values():
        layer.weight.requires_grad_(True)
    adapter.model.eval()

    dest = Path(output)
    dest.mkdir(parents=True, exist_ok=False)
    streams = {}
    metadata = {}
    for ordinal, (name, layer) in enumerate(linear.items()):
        count = layer.weight.numel()
        prefix = f"tensor_{ordinal:03d}"
        mean_path, abs_path = dest / f"{prefix}_mean_g.f32", dest / f"{prefix}_mean_abs_g.f32"
        mean = np.memmap(mean_path, dtype="<f4", mode="w+", shape=(count,))
        mean_abs = np.memmap(abs_path, dtype="<f4", mode="w+", shape=(count,))
        mean[:] = 0
        mean_abs[:] = 0
        streams[name] = (mean, mean_abs)
        metadata[name] = {"codes": count, "mean_g": mean_path.name, "mean_abs_g": abs_path.name,
                          "surrogate_weight_sha256": int8_torch.tensor_sha256(layer.weight)}
    losses = []
    for example in examples:
        if adapter.asset.target == "distilbert_sst2_int8":
            encoded, label = example
            logits = adapter.model(**encoded).logits
            loss = F.cross_entropy(logits, torch.tensor([label], dtype=torch.long))
        else:
            token_ids = example
            logits = adapter.model(input_ids=token_ids, use_cache=False).logits
            loss = F.cross_entropy(logits[0, :-1].float(), token_ids[0, 1:])
        if not bool(torch.isfinite(loss)):
            raise ValueError("nonfinite FP32 decoded-code surrogate calibration loss")
        loss.backward()
        losses.append(float(loss.detach()))
        for name, layer in linear.items():
            grad = layer.weight.grad
            if grad is None or not bool(torch.isfinite(grad).all()):
                raise ValueError(f"missing or nonfinite surrogate gradient: {name}")
            values = grad.detach().contiguous().cpu().numpy().reshape(-1)
            mean, mean_abs = streams[name]
            mean += values
            mean_abs += np.abs(values)
            layer.weight.grad = None
    for name, (mean, mean_abs) in streams.items():
        mean /= len(examples)
        mean_abs /= len(examples)
        mean.flush()
        mean_abs.flush()
        info = metadata[name]
        info["mean_g_sha256"] = sha256_file(dest / info["mean_g"])
        info["mean_abs_g_sha256"] = sha256_file(dest / info["mean_abs_g"])
    result = {"version": "w8a8-decoded-code-surrogate-moments/v1",
              "status": "SELECTION_ONLY", "target": adapter.asset.target,
              "target_record_sha256": adapter.asset.record_sha256,
              "universe_sha256": adapter.universe.digest,
              "surrogate": "exact decoded stored qweight * fixed FP32 row scale; distinct from dynamic W8A8 integer execution",
              "per_example_unit": "one SST-2 sentence" if adapter.asset.target == "distilbert_sst2_int8"
                                  else "one 1,023-target WikiText validation window",
              "calibration": cohort, "examples": len(examples), "losses": losses,
              "moments_dtype": "float32", "tensors": metadata}
    (dest / "manifest.json").write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    return result
