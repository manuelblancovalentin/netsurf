#!/usr/bin/env python3
"""Bounded Qwen trace of a known valid INT8 flip with unchanged final logits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

from netsurf.manifests import digest_object, sha256_file
from netsurf.w8a8_assets import load_ready_w8a8
from netsurf.w8a8_faults import W8A8FaultAdapter, W8A8Field, FixedMask


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: check_w8a8_requantization_mask.py READY.json OUTPUT_DIR")
    destination = Path(sys.argv[2])
    if destination.exists():
        raise ValueError("immutable diagnostic destination already exists")
    asset = load_ready_w8a8(Path(sys.argv[1]), "qwen3_0_6b_int8")
    adapter = W8A8FaultAdapter(asset)
    adapter.load()
    ids = adapter.token_sequence("test")[:128]
    gate = adapter.modules["model.layers.27.mlp.gate_proj"]
    downstream = adapter.modules["model.layers.27.mlp.down_proj"]
    traces = []

    def digest(tensor):
        return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()

    def gate_hook(module, inputs, output):
        traces[-1]["gate_output_sha256"] = digest(output)

    def down_hook(module, inputs):
        import torch
        x = inputs[0].reshape(-1, inputs[0].shape[-1]).float()
        a = (x.abs().amax(dim=1, keepdim=True) / 127).clamp_min(1e-12)
        q = torch.round(x / a).clamp(-127, 127).to(torch.int8)
        traces[-1]["downstream_int8_codes_sha256"] = digest(q)
        traces[-1]["downstream_activation_scales_sha256"] = digest(a)

    hooks = (gate.register_forward_hook(gate_hook), downstream.register_forward_pre_hook(down_hook))
    try:
        traces.append({})
        clean = adapter.evaluate_lm(ids)
        coordinate = adapter.universe.ordinal("model.layers.27.mlp.gate_proj", 1540712, 6)
        field = W8A8Field(820302, 0.0, adapter.universe.bits, (coordinate,), "forced_engineering")
        mask = FixedMask(adapter.universe.bits, frozenset())
        traces.append({})
        with adapter.injected(field, mask) as accounting:
            faulted = adapter.evaluate_lm(ids)
        traces.append({})
        restored = adapter.evaluate_lm(ids)
    finally:
        for hook in hooks:
            hook.remove()
    assert clean == restored
    assert traces[0] == traces[2]
    result = {"schema": "netsurf-w8a8-requantization-trace/v1", "status": "ENGINEERING_ONLY",
              "target": asset.target, "target_record_sha256": asset.record_sha256,
              "source_receipt_sha256": asset.record["fault_mutation_receipt"]["report_sha256"],
              "input": "first 128 WikiText-2 test token IDs, matching Kona's exposed all-module sweep",
              "coordinate": {"module": "model.layers.27.mlp.gate_proj", "code_index": 1540712,
                             "bit": 6, "global_ordinal": coordinate},
              "field_sha256": field.sha256, "event_accounting": accounting,
              "traces": traces, "clean_logits_sha256": digest_object(clean.logits),
              "faulted_logits_sha256": digest_object(faulted.logits),
              "stored_bytes_restored": True,
              "gate_module_output_changed": traces[0].get("gate_output_sha256") != traces[1].get("gate_output_sha256"),
              "downstream_int8_codes_equal": traces[0].get("downstream_int8_codes_sha256") == traces[1].get("downstream_int8_codes_sha256"),
              "downstream_activation_scales_equal": traces[0].get("downstream_activation_scales_sha256") == traces[1].get("downstream_activation_scales_sha256"),
              "final_task_statistics_equal": clean == faulted}
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "receipt.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    (destination / "FINALIZED").write_text(sha256_file(destination / "receipt.json") + "\n")
    print(json.dumps({k: result[k] for k in ("gate_module_output_changed", "downstream_int8_codes_equal",
                                          "downstream_activation_scales_equal", "final_task_statistics_equal")}))


if __name__ == "__main__":
    main()
