#!/usr/bin/env python3
"""Engineering-only full-eligible-qweight Hessian feasibility after Qwen finalization.

No faults, masks, ranking scores, final input examples or production comparisons
are generated.  The old six-method campaigns and asset locks are read-only.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
EVIDENCE_ROOT = Path(
    "/projects/netsurf/manuelbv/workspace/implementation-worktrees/transformer-integration")

from netsurf.manifests import canonical_json_bytes, digest_object, sha256_file
from netsurf.w8a8_campaign import validate
from netsurf.w8a8_faults import W8A8FaultAdapter
from netsurf.w8a8_fkeras_preflight import (
    bounded_eigenpair_probe, project_full_population, symmetry_probe,
    validate_full_mapping,
)

READY = Path("/projects/kona-assets/manuelbv/NETSURF_ASSET_READINESS.json")
PYTHON = Path("/projects/kona-assets/manuelbv/envs/torch-2.9.1-cpu-py311/bin/python")
STEMS = {
    "distilbert": "distilbert_sst2_w8a8_reliability_v2",
    "pythia": "pythia160m_w8a8_reliability_v3",
    "qwen": "qwen3_0_6b_w8a8_reliability_v3",
}
QWEN = EVIDENCE_ROOT / "workspace/runs/qwen3_0_6b_w8a8_reliability_v3_run1"


def qwen_finalized() -> dict:
    """Read-only finalization gate; accepts no partial or merely present ZIP."""
    export = QWEN.with_name(QWEN.name + "_export")
    runtime = QWEN.with_name(QWEN.name + "_runtime_receipt.json")
    if not runtime.is_file() or not (export / "FINALIZED").is_file() or not (export / "complete.zip").is_file():
        raise ValueError("Qwen W8A8 has not finalized; full-model FKeras preflight is deferred")
    data = json.loads(runtime.read_text())
    if data.get("exit_code") != 0 or len(data.get("stages", [])) != 6:
        raise ValueError("Qwen runtime receipt is incomplete")
    plan = json.loads((EVIDENCE_ROOT / "workspace/plans" / STEMS["qwen"] / "plan.json").read_text())
    members = json.loads((export / "checksums.json").read_text())
    zip_hash = sha256_file(export / "complete.zip")
    expected = digest_object({"plan_sha256": plan["plan_sha256"],
                              "zip_sha256": zip_hash, "members": members})
    if (export / "FINALIZED").read_text().strip() != expected:
        raise ValueError("Qwen finalized export marker differs from source archive")
    return {"archive_sha256": zip_hash, "finalized_digest": expected,
            "runtime_receipt_sha256": sha256_file(runtime)}


def _peak_bytes() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def prepare_companion(adapter: W8A8FaultAdapter):
    """Replace every eligible QLinear by decoded FP32 Linear, in universe order."""
    import torch
    import torch.nn as nn

    if adapter.model is None or adapter.modules is None:
        raise ValueError("fresh loaded W8A8 executable required")
    modules = adapter.modules
    shapes = {name: tuple(modules[name].qweight.shape) for name in modules}
    mapping = validate_full_mapping(adapter.universe.entries, shapes, adapter.universe.bits)
    linear = {}
    for name, _, _ in mapping:
        q = modules[name]
        layer = nn.Linear(q.in_features, q.out_features, bias=q.bias is not None)
        with torch.no_grad():
            decoded = q.qweight.to(torch.float32) * q.scale[:, None]
            layer.weight.copy_(decoded)
            if not torch.equal(layer.weight, decoded):
                raise ValueError("continuous companion differs from exact decoded stored codes")
            if q.bias is not None:
                layer.bias.copy_(q.bias)
        parent, _, leaf = name.rpartition(".")
        setattr(adapter.model.get_submodule(parent) if parent else adapter.model, leaf, layer)
        linear[name] = layer
    adapter.modules = None
    del modules
    gc.collect()
    for parameter in adapter.model.parameters():
        parameter.requires_grad_(False)
    for layer in linear.values():
        layer.weight.requires_grad_(True)
    adapter.model.eval()
    weights = tuple(linear[name].weight for name, _, _ in mapping)
    if sum(weight.numel() for weight in weights) * 8 != adapter.universe.bits:
        raise ValueError("continuous Hessian excludes eligible code coordinates")
    return mapping, weights


def task_loss(adapter: W8A8FaultAdapter, plan: dict):
    """One already exposed selection example, never a final-evaluation item."""
    import torch
    import torch.nn.functional as F

    if adapter.asset.target == "distilbert_sst2_int8":
        rows = adapter.sst2_cohorts()["calibration"]
        if [row.source_index for row in rows[:64]] != plan["selection"]["source_indices"]:
            raise ValueError("selection labels or source mapping changed")
        row = rows[0]
        encoded = adapter.tokenizer([row.text], return_tensors="pt", padding="max_length",
                                    truncation=True, max_length=128)
        label = torch.tensor([row.label], dtype=torch.long)

        def loss():
            return F.cross_entropy(adapter.model(**encoded).logits, label)

        identity = {"split": "SST-2 train", "source_index": row.source_index,
                    "selection_cohort_sha256": plan["selection"]["ordered_labels_and_text_sha256"]}
    else:
        ids = adapter.token_sequence("validation")
        if adapter.last_token_identity["token_ids_sha256"] != plan["selection"]["full_token_ids_sha256"]:
            raise ValueError("selection token identity changed")
        selected = tuple(ids[1024:1152])
        if len(selected) != 128:
            raise ValueError("bounded validation window unavailable")
        token_ids = torch.tensor([selected], dtype=torch.long)

        def loss():
            logits = adapter.model(input_ids=token_ids, use_cache=False).logits
            return F.cross_entropy(logits[0, :-1].float(), token_ids[0, 1:])

        identity = {"split": "WikiText-2 raw validation", "token_start": 1024,
                    "token_end_exclusive": 1152, "scored_targets": 127,
                    "token_ids_sha256": hashlib.sha256(
                        __import__("numpy").asarray(selected, dtype="<i4").tobytes()).hexdigest(),
                    "selection_cohort_sha256": plan["selection"]["input_token_ids_sha256"]}
    return loss, identity


def execute(model: str, destination: Path, *, max_products: int, tolerance: float,
            max_seconds: float, max_rss_gib: float) -> dict:
    if Path(sys.executable).resolve() != PYTHON.resolve():
        raise ValueError("pinned Kona interpreter required")
    if destination.exists():
        raise ValueError("engineering preflight output already exists; no silent restart")
    qwen = qwen_finalized()
    plan_dir = EVIDENCE_ROOT / "workspace/plans" / STEMS[model]
    plan, asset = validate(READY, plan_dir)
    projection = project_full_population(plan["eligible_universe"]["codes"])
    if projection.eligible_bits != plan["eligible_universe"]["bits"]:
        raise ValueError("full qweight census differs from frozen campaign")
    import numpy as np
    import torch

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    destination.mkdir(parents=True, exist_ok=False)
    identity = {"schema": "netsurf-w8a8-full-qweight-fkeras-preflight/v1",
                "target": asset.target, "status": "STARTED_ENGINEERING_ONLY",
                "plan_sha256": plan["plan_sha256"],
                "target_record_sha256": asset.record_sha256,
                "eligible_universe_sha256": plan["eligible_universe"]["sha256"],
                "eligible_bits": projection.eligible_bits,
                "qwen_finalization": qwen,
                "source_sha256": sha256_file(Path(__file__)),
                "preflight_primitives_sha256": sha256_file(
                    ROOT / "netsurf/w8a8_fkeras_preflight.py"),
                "plan_file_sha256": sha256_file(plan_dir / "plan.json"),
                "interpreter_sha256": sha256_file(PYTHON),
                "max_products": max_products, "tolerance": tolerance,
                "max_seconds": max_seconds, "max_rss_gib": max_rss_gib}
    (destination / "identity.json").write_bytes(canonical_json_bytes(identity) + b"\n")
    start = time.monotonic()
    adapter = W8A8FaultAdapter(asset)
    adapter.load()
    loss_fn, input_identity = task_loss(adapter, plan)
    mapping, weights = prepare_companion(adapter)
    dimension = sum(weight.numel() for weight in weights)
    initial_loss = float(loss_fn().detach())
    if not np.isfinite(initial_loss):
        raise ValueError("nonfinite continuous companion loss")
    timings = []

    def hvp(vector: np.ndarray) -> np.ndarray:
        if time.monotonic() - start > max_seconds or _peak_bytes() > max_rss_gib * 1024**3:
            raise RuntimeError("preflight wall-time or RSS bound exceeded before HVP")
        began = time.monotonic()
        subvectors = tuple(torch.from_numpy(vector[offset:offset+size]).reshape(weight.shape)
                           for (_, offset, size), weight in zip(mapping, weights))
        loss = loss_fn()
        gradients = torch.autograd.grad(loss, weights, create_graph=True)
        products = torch.autograd.grad(gradients, weights, grad_outputs=subvectors)
        result = np.empty(dimension, dtype=np.float32)
        for (_, offset, size), product in zip(mapping, products):
            result[offset:offset+size] = product.detach().contiguous().numpy().reshape(-1)
        del gradients, products, subvectors, loss
        gc.collect()
        timings.append(time.monotonic() - began)
        print(json.dumps({"hvp_complete": len(timings), "seconds": timings[-1],
                          "peak_rss_bytes": _peak_bytes()}), flush=True)
        if time.monotonic() - start > max_seconds or _peak_bytes() > max_rss_gib * 1024**3:
            raise RuntimeError("preflight wall-time or RSS bound exceeded after HVP")
        return result

    symmetry = symmetry_probe(hvp, dimension)
    probe = bounded_eigenpair_probe(hvp, dimension, max_products=max_products,
                                    tolerance=tolerance)
    # The paper requires multiple eigenpairs and a tier-first full ranking.
    # Even a converged one-pair probe is not a measured FKeras policy.
    report = {**identity, "status": "BOUNDED_PREFLIGHT_COMPLETE", "input": input_identity,
              "continuous_loss": initial_loss,
              "continuous_model": "FP32 exactly decoded stored INT8 qweights times fixed row scales; "
                                  "unquantized weights frozen; no dynamic activation requantization",
              "parameter_mapping": mapping,
              "parameter_mapping_sha256": digest_object(mapping),
              "mapped_parameter_count": dimension,
              "projection": asdict(projection),
              "symmetry_probe": symmetry,
              "eigenpair_probe": probe,
              "hvp_seconds": timings,
              "estimated_500_hvp_seconds_lower_bound": 500 * min(timings),
              "measured_seconds": time.monotonic() - start,
              "peak_rss_bytes": _peak_bytes(),
              "publication_fkera_cell": "UNMEASURED"}
    (destination / "report.json").write_bytes(canonical_json_bytes(report) + b"\n")
    (destination / "FINALIZED").write_text(sha256_file(destination / "report.json") + "\n")
    return {"status": report["status"], "report": str(destination / "report.json"),
            "report_sha256": sha256_file(destination / "report.json"),
            "probe_converged": probe["converged"], "hvp_calls": probe["hvp_calls"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=tuple(STEMS))
    parser.add_argument("destination", type=Path)
    parser.add_argument("--max-products", type=int, default=6)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--max-seconds", type=float, default=7200)
    parser.add_argument("--max-rss-gib", type=float, default=128)
    args = parser.parse_args()
    if args.max_products < 2 or args.max_seconds <= 0 or args.max_rss_gib <= 0:
        parser.error("positive bounded resource limits required")
    print(json.dumps(execute(args.model, args.destination, max_products=args.max_products,
                             tolerance=args.tolerance, max_seconds=args.max_seconds,
                             max_rss_gib=args.max_rss_gib), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
