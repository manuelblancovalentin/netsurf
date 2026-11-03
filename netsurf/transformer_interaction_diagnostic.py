"""Bounded retrospective paired bit/background diagnostic on exposed SST-2 inputs.

The selected coordinate is disclosed as post hoc from the v4 masks. This
exercise cannot establish the cause of a finalized result on another cohort.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import zipfile

from .adapters.transformers_pinned import PinnedTransformerAdapter, read_sst2_cohorts
from .faults import FaultEvent, FaultModel, FaultSchedule
from .identities import BitIdentity
from .manifests import canonical_json_bytes, digest_object, sha256_file
from .protection import ProtectionMask
from .torch_storage import TorchStoredLinearWeights, sparse_bernoulli_schedule
from .transformer_assets import load_ready_asset

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "config/experiments/distilbert_sst2_classifier1024_v4.json"
SELECTION = ROOT / "workspace/runs/distilbert_sst2_classifier1024_v4_run1/selection"
SNAPSHOT = ROOT / "workspace/plans/distilbert_sst2_classifier1024_v4/NETSURF_ASSET_READINESS.json"
PORTFOLIO = ROOT / "config/benchmark_expansion_registry.json"
TARGET = BitIdentity("classifier.weight", 0, 30)
BER = 16 / 1024
SEEDS = (881001, 881002, 881003, 881004)
SOURCE_FILES = ("netsurf/transformer_interaction_diagnostic.py",
                "netsurf/adapters/transformers_pinned.py", "netsurf/torch_storage.py",
                "netsurf/transformer_risk.py", "netsurf/transformer_assets.py",
                "netsurf/faults.py", "netsurf/protection.py")


def interaction_contrast(clean: float, target: float, background: float,
                         combined: float) -> dict[str, float]:
    """Conditional bit effect minus isolated bit effect, on one paired field."""
    isolated = target - clean
    conditional = combined - background
    return {"isolated_bit_effect": isolated, "conditional_bit_effect": conditional,
            "interaction_difference": conditional - isolated}


def outside_background(storage, *, target: BitIdentity, seed: int,
                       ber: float, inference_count: int) -> FaultSchedule:
    if target not in {storage.bit_at(i) for i in range(storage.bit_count)}:
        raise ValueError("target bit is outside declared universe")
    master = sparse_bernoulli_schedule(storage, ber=ber, seed=seed,
        inference_count=inference_count, max_events=128)
    events = tuple(event for event in master.events if event.bit != target)
    law = FaultModel("weight", "cpu_classifier_first_32_words",
                     "persistent_trial", "forced_engineering", ber)
    return FaultSchedule(law, seed, events, storage.bit_count, inference_count)


def _verify_existing(output: Path) -> dict:
    checks = json.loads((output / "checksums.json").read_text())
    if any(sha256_file(output / name) != value for name, value in checks.items()):
        raise ValueError("completed diagnostic artifact changed")
    with zipfile.ZipFile(output / "complete.zip") as archive:
        if archive.testzip() is not None:
            raise ValueError("completed diagnostic ZIP corrupt")
    payload = json.loads((output / "payload.json").read_text())
    if digest_object(payload) != (output / "FINALIZED").read_text().strip():
        raise ValueError("completed diagnostic payload digest changed")
    return {"status": "FINALIZED_VERIFIED", "output": str(output),
            "payload_sha256": digest_object(payload)}


def run(output: Path) -> dict:
    output = Path(output)
    if output.exists():
        return _verify_existing(output)
    stage = output.with_name("." + output.name + ".incomplete")
    if stage.exists():
        raise FileExistsError(f"partial diagnostic retained: {stage}")
    subprocess.run(["git", "ls-files", "--error-unmatch", *SOURCE_FILES],
                   cwd=ROOT, capture_output=True, check=True)
    subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *SOURCE_FILES],
                   cwd=ROOT, check=True)
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
        cwd=ROOT, text=True).strip()
    source_hashes = {name: sha256_file(ROOT / name) for name in SOURCE_FILES}
    plan = json.loads(PLAN.read_text())
    if plan["plan_sha256"] != "735ad6e8531628059bf891075e37885026b220a586bc2d0dc640e507238dc7d7" or \
       sha256_file(SNAPSHOT) != plan["asset_manifest_sha256"]:
        raise ValueError("original DistilBERT plan or READY snapshot changed")
    masks = {name: json.loads((SELECTION / f"{name}.json").read_text())
             for name in ("significance", "exact_isolated")}
    if (TARGET.key() not in masks["significance"]["masks"]["128"] or
        TARGET.key() in masks["exact_isolated"]["masks"]["128"]):
        raise ValueError("post-hoc disputed bit is no longer in the sealed mask difference")
    for name, expected in (("significance", "830d76fd47a556b52f1f6181c8383effbe2a2f91fc4de52e7f27df26a1f2d67b"),
                           ("exact_isolated", "60f32624197a7c18769aa3ad8f13f83a20f593f6680eff212edc147620eddaf9")):
        if sha256_file(SELECTION / f"{name}.json") != expected:
            raise ValueError("sealed v4 selection member changed")
    asset = load_ready_asset(SNAPSHOT, "distilbert_sst2_fp32", PORTFOLIO)
    started = time.monotonic()
    adapter = PinnedTransformerAdapter(asset)
    adapter.load()
    bits = tuple(BitIdentity("classifier.weight", i, bit)
                 for i in range(32) for bit in range(32))
    adapter.storage = TorchStoredLinearWeights(adapter.model, "FP32", eligible_bits=bits)
    examples = read_sst2_cohorts(asset)["engineering"]
    if len(examples) != 32 or [row.source_index for row in examples] != list(range(32)):
        raise ValueError("only previously exposed validation engineering examples 0–31 allowed")
    original_hashes = adapter.storage.hashes()
    empty = ProtectionMask("unprotected", frozenset(), 1024, 0, {})
    law = FaultModel("weight", "cpu_classifier_first_32_words",
                     "persistent_trial", "forced_engineering", BER)

    def evaluate(label: str, events: tuple[FaultEvent, ...], seed: int) -> dict:
        schedule = FaultSchedule(law, seed, events, 1024, len(examples))
        with adapter.storage.injected(schedule, empty) as report:
            risk = adapter.evaluate_sst2(examples)
        if adapter.storage.hashes() != original_hashes:
            raise ValueError("stored classifier bytes failed restoration")
        return {"label": label, "seed": seed, "schedule_sha256": schedule.digest(),
            "events": [event.key() for event in events], "report": asdict(report),
            "risk": asdict(risk), "risk_sha256": digest_object(asdict(risk))}

    clean = evaluate("clean", (), 881000)
    target_event = FaultEvent(None, TARGET)
    isolated = evaluate("target_only", (target_event,), 881000)
    pairs = []
    for seed in SEEDS:
        background = outside_background(adapter.storage, target=TARGET,
            seed=seed, ber=BER, inference_count=len(examples))
        events = background.events
        outside = evaluate("outside_only", events, seed)
        combined = evaluate("outside_plus_target", events + (target_event,), seed)
        pairs.append({"seed": seed, "background_schedule_sha256": background.digest(),
            "outside_event_count": len(events), "outside": outside,
            "combined": combined, "contrast": interaction_contrast(
                clean["risk"]["mean_nll"], isolated["risk"]["mean_nll"],
                outside["risk"]["mean_nll"], combined["risk"]["mean_nll"])})
    clean_after = evaluate("clean_after", (), 881005)
    if clean_after["risk"] != clean["risk"]:
        raise ValueError("clean task output changed after diagnostic")
    if any(sha256_file(ROOT / name) != value for name, value in source_hashes.items()):
        raise ValueError("scientific source changed during diagnostic")
    payload = {"schema_version": "distilbert-retrospective-interaction-engineering/v1",
        "status": "ENGINEERING_ONLY", "source_commit": source_commit,
        "source_files_sha256": source_hashes, "v4_plan_sha256": plan["plan_sha256"],
        "ready_snapshot_sha256": sha256_file(SNAPSHOT),
        "asset_record_sha256": asset.record_sha256,
        "significance_mask_file_sha256": sha256_file(SELECTION / "significance.json"),
        "isolated_mask_file_sha256": sha256_file(SELECTION / "exact_isolated.json"),
        "posthoc_target_bit": TARGET.key(), "ber_for_outside_bits": BER,
        "background_sampling": "independent Bernoulli on all 1023 outside candidate bits; target event removed from master draw and forced only in paired counterfactual",
        "input_split": "SST-2 validation", "source_indices": list(range(32)),
        "model_requests": 11, "clip_inferences": 11 * 32,
        "clean": clean, "target_only": isolated, "paired_backgrounds": pairs,
        "clean_after": clean_after, "runtime_seconds": time.monotonic() - started,
        "limitation": "target selected after v4 mask review; four engineering backgrounds and previously exposed 32-example cohort; no attribution of v4 final effect"}
    stage.mkdir(parents=True, exist_ok=False)
    (stage / "payload.json").write_bytes(canonical_json_bytes(payload) + b"\n")
    (stage / "FINALIZED").write_text(digest_object(payload) + "\n")
    with zipfile.ZipFile(stage / "complete.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(stage / "payload.json", "payload.json")
        archive.write(PLAN, "source/v4_plan.json")
        archive.write(SNAPSHOT, "source/frozen_READY.json")
        for name in SOURCE_FILES:
            archive.write(ROOT / name, "source/" + name)
    checks = {name: sha256_file(stage / name)
              for name in ("payload.json", "FINALIZED", "complete.zip")}
    (stage / "checksums.json").write_bytes(canonical_json_bytes(checks) + b"\n")
    stage.rename(output)
    return _verify_existing(output)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.output), sort_keys=True))
