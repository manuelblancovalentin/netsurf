#!/usr/bin/env python3
"""Bounded real-model NetSuRF W8A8 receipt on already exposed engineering inputs."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import resource
import sys
import time

from netsurf.manifests import digest_object, sha256_file
from netsurf.w8a8_assets import load_ready_w8a8, TARGETS
from netsurf.w8a8_faults import W8A8FaultAdapter, W8A8Field, FixedMask, MSBMask


PICKS = {
    "distilbert_sst2_int8": (("distilbert.transformer.layer.0.attention.q_lin", 12345, 7),
                             ("classifier", 1220, 7)),
    "pythia160m_int8": (("gpt_neox.layers.0.attention.query_key_value", 12345, 7),
                       ("embed_out", 392457, 7)),
    "qwen3_0_6b_int8": (("model.layers.0.self_attn.q_proj", 12345, 7),
                       ("lm_head", 1564469, 7)),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("target", choices=sorted(TARGETS))
    p.add_argument("manifest", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("engineering receipt destination already exists")
    asset = load_ready_w8a8(args.manifest, args.target)
    adapter = W8A8FaultAdapter(asset)
    adapter.load()
    if args.target == "distilbert_sst2_int8":
        rows = adapter.sst2_cohorts()["engineering"][:8]
        evaluate = lambda: adapter.evaluate_sst2(rows)
        cohort = {"split": "SST-2 validation", "source_indices": [r.source_index for r in rows],
                  "labels": [r.label for r in rows]}
    else:
        ids = adapter.token_sequence("test")[:512]
        evaluate = lambda: adapter.evaluate_lm(ids)
        cohort = {"split": "WikiText-2 test", "token_ids_sha256": digest_object(ids),
                  "input_tokens": len(ids), "scored_targets": len(ids)-1}
    picks = [adapter.universe.ordinal(*p) for p in PICKS[args.target]]
    field = W8A8Field(820301, 0.0, adapter.universe.bits, tuple(sorted(picks)), "forced_engineering")
    masks = {"unprotected": FixedMask(adapter.universe.bits, frozenset()),
             "one_protected": FixedMask(adapter.universe.bits, frozenset(picks[:1])),
             "msb_6_25": MSBMask(adapter.universe.bits, adapter.universe.bits // 16),
             "full_protection": FixedMask(adapter.universe.bits, frozenset(picks))}
    start_clean = time.perf_counter()
    clean = evaluate()
    clean_seconds = time.perf_counter() - start_clean
    arms = {}
    for name, mask in masks.items():
        start_arm = time.perf_counter()
        with adapter.injected(field, mask) as counts:
            risk = evaluate()
        arms[name] = {"risk": {"metrics": dict(risk.metrics), "failed_units": risk.failed_units,
                                "total_units": risk.total_units, "logits_sha256": digest_object(risk.logits),
                                "failure_kinds_sha256": digest_object(risk.failure_kinds)},
                      "events": counts, "mask_sha256": mask.digest,
                      "trial_seconds": time.perf_counter() - start_arm}
    repeat = evaluate()
    assert clean == repeat, "post-trial clean task output changed"
    assert arms["full_protection"]["risk"]["logits_sha256"] == digest_object(clean.logits)
    assert arms["unprotected"]["events"]["applied"] == 2
    assert arms["one_protected"]["events"]["applied"] == 1
    assert arms["full_protection"]["events"]["applied"] == 0
    result = {"schema": "netsurf-w8a8-engineering-receipt/v1", "status": "ENGINEERING_ONLY",
              "asset_manifest_sha256": asset.manifest_sha256, "target_record_sha256": asset.record_sha256,
              "target": args.target, "model_commit": asset.record["model"]["commit"],
              "quantization_manifest_sha256": next(x["sha256"] for x in asset.record["model"]["files"]
                                                   if x["path"] == "quantization_manifest.json"),
              "universe_sha256": adapter.universe.digest, "eligible_bits": adapter.universe.bits,
              "eligible_modules": adapter.universe.entries and len(adapter.universe.entries),
              "cohort": cohort, "exact_coordinates": [asdict(adapter.universe.coordinate(x)) for x in picks],
              "field_sha256": field.sha256, "clean": {"metrics": dict(clean.metrics),
                  "failed_units": clean.failed_units, "total_units": clean.total_units,
                  "logits_sha256": digest_object(clean.logits), "trial_seconds": clean_seconds}, "arms": arms,
              "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
              "restored_byte_exact": True, "clean_repeated_bitwise": True,
              "observable_at_task_statistics": arms["unprotected"]["risk"]["logits_sha256"] != digest_object(clean.logits)}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "receipt.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    digest = sha256_file(args.output / "receipt.json")
    (args.output / "FINALIZED").write_text(digest + "\n")
    print(json.dumps({"target": args.target, "status": "ENGINEERING_ONLY_FINALIZED",
                      "receipt_sha256": digest, "observable": result["observable_at_task_statistics"],
                      "clean": clean.metrics, "faulted": arms["unprotected"]["risk"]["metrics"]}))


if __name__ == "__main__":
    sys.exit(main())
