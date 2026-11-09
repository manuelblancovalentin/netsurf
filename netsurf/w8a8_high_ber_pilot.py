"""Bounded, unprotected W8A8 BER stress pilot; never constructs ranking masks.

This is an engineering/calibration study on exposed SST-2 engineering examples
or WikiText-2 validation tokens. It is not a publication protection campaign.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from .manifests import canonical_json_bytes, digest_object, sha256_file
from .transformer_data import sst2_records_digest
from .w8a8_assets import INTERPRETER, load_ready_w8a8
from .w8a8_faults import BoundaryMask, W8A8FaultAdapter, bernoulli_field


BERS = (1e-5, 1e-4, 1e-3)
FIELDS_PER_BER = 2
SEED_BASE = 1529001
MAX_EVENTS = 6_000_000
TARGETS = ("distilbert_sst2_int8", "pythia160m_int8", "qwen3_0_6b_int8")


def planned_fields(bits: int) -> tuple[dict, ...]:
    """Prespecified, exact Bernoulli fields; reject any unsafe event envelope."""
    arms = []
    for arm, ber in enumerate(BERS):
        expected = bits * ber
        if expected + 8 * math.sqrt(expected) > MAX_EVENTS:
            raise ValueError("pilot event bound cannot cover planned Bernoulli field")
        arms.append({"ber": ber, "expected_proposed": expected,
                     "seeds": [SEED_BASE + arm * 100_000 + i for i in range(FIELDS_PER_BER)]})
    return tuple(arms)


def _write(path: Path, obj: dict) -> None:
    with path.open("xb") as out:
        out.write(canonical_json_bytes(obj) + b"\n")
        out.flush()
        os.fsync(out.fileno())


def _risk(risk) -> dict:
    return {"mean_nll_nats": risk.mean_nll, "accuracy": risk.accuracy,
            "perplexity": risk.perplexity, "failed_units": risk.failed_units,
            "total_units": risk.total_units, "failure_fraction": risk.failure_fraction,
            "failure_kinds": {kind: risk.failure_kinds.count(kind) for kind in
                              ("nonfinite_logits", "inference_failure")},
            "outputs_sha256": digest_object(risk.logits)}


def _cohort(adapter: W8A8FaultAdapter):
    if adapter.asset.target == "distilbert_sst2_int8":
        records = adapter.sst2_cohorts()["engineering"][:8]
        identity = {"split": "validation", "source_indices": [r.source_index for r in records],
                    "ordered_labels_and_text_sha256": sst2_records_digest(records),
                    "units": len(records), "role": "previously exposed engineering only"}
        return records, identity, adapter.evaluate_sst2
    ids = adapter.token_sequence("validation")[:129]
    if len(ids) != 129:
        raise ValueError("WikiText-2 validation engineering token slice changed")
    identity = {"split": "validation", "token_offset": 0, "input_tokens": len(ids),
                "scored_targets": len(ids) - 1,
                "token_ids_sha256": hashlib.sha256(np.asarray(ids, dtype="<i4").tobytes()).hexdigest(),
                "role": "calibration-only; disjoint source split from frozen test evaluation"}
    return ids, identity, adapter.evaluate_lm


def run(target: str, manifest_path: Path, output_dir: Path) -> dict:
    if target not in TARGETS or output_dir.exists():
        raise ValueError("unsupported target or write-once output already exists")
    if Path(__import__("sys").executable).resolve() != INTERPRETER.resolve():
        raise ValueError(f"requires pinned W8A8 interpreter: {INTERPRETER}")
    asset = load_ready_w8a8(manifest_path, target)
    arms = planned_fields(asset.eligible_bits)
    adapter = W8A8FaultAdapter(asset)
    adapter.load()
    cohort, cohort_identity, evaluate = _cohort(adapter)
    source_root = Path(__file__).resolve().parent.parent
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source_root, text=True).strip()
    plan = {"schema": "netsurf-w8a8-high-ber-exposure-pilot/v1", "status": "EXPLORATORY_CALIBRATION",
            "target": target, "source_commit": source_commit,
            "source_sha256": sha256_file(Path(__file__)),
            "target_record_sha256": asset.record_sha256,
            "ready_manifest_sha256": asset.manifest_sha256,
            "eligible_bits": adapter.universe.bits,
            "eligible_population_sha256": adapter.universe.digest,
            "cohort": cohort_identity,
            "fault_law": "independent Bernoulli over complete eligible stored INT8 qweight population",
            "fault_lifetime": "persistent across every input in one field evaluation",
            "policies": ["unprotected"], "ber_arms": arms,
            "fields_per_ber": FIELDS_PER_BER, "max_events_per_field": MAX_EVENTS,
            "failure_penalty_nats": 30.0,
            "analysis_warning": "engineering exposure search only; two fields cannot establish protection effectiveness",
            "stopping_scope": "this run stops at BER 0.001; 0.01 and 0.1 require measured feasibility and new plans"}
    plan["plan_sha256"] = digest_object(plan)
    output_dir.mkdir(parents=True, exist_ok=False)
    _write(output_dir / "plan.json", plan)
    started = time.monotonic()
    clean = _risk(evaluate(cohort))
    _write(output_dir / "clean_start.json", clean)
    results = []
    zero_mask = BoundaryMask(adapter.universe.bits, False)
    for arm, specification in enumerate(arms):
        for index, seed in enumerate(specification["seeds"]):
            t0 = time.monotonic()
            field = bernoulli_field(adapter.universe.bits, specification["ber"], seed,
                                    max_events=MAX_EVENTS)
            ordinals = np.asarray(field.ordinals, dtype="<u8")
            data_path = output_dir / f"ber_{arm}_field_{index}.ordinals.u64le"
            with data_path.open("xb") as out:
                out.write(ordinals.tobytes())
                out.flush()
                os.fsync(out.fileno())
            field_sha = field.sha256
            with adapter.injected(field, zero_mask) as accounting:
                faulted = _risk(evaluate(cohort))
            if accounting["proposed"] != accounting["applied"] or accounting["suppressed"] != 0:
                raise RuntimeError("unprotected field accounting changed")
            record = {"ber": specification["ber"], "expected_proposed": specification["expected_proposed"],
                      "field_index": index, "seed": seed, "field_sha256": field_sha,
                      "ordinals_u64le_sha256": sha256_file(data_path),
                      "proposed": accounting["proposed"], "applied": accounting["applied"],
                      "suppressed": accounting["suppressed"], "risk": faulted,
                      "elapsed_seconds": time.monotonic() - t0}
            _write(output_dir / f"ber_{arm}_field_{index}.json", record)
            results.append(record)
            print(json.dumps({"target": target, "ber": specification["ber"], "field": index,
                              "applied": record["applied"], "mean_nll": faulted["mean_nll_nats"],
                              "failed": faulted["failed_units"], "elapsed_s": round(record["elapsed_seconds"], 2)}),
                  flush=True)
    restored = _risk(evaluate(cohort))
    _write(output_dir / "clean_end.json", restored)
    if clean != restored:
        raise RuntimeError("clean task output changed after exact fault restoration")
    files = sorted(p for p in output_dir.iterdir() if p.is_file())
    checksums = {p.name: sha256_file(p) for p in files}
    summary = {"schema": "netsurf-w8a8-high-ber-exposure-pilot-result/v1",
               "status": "FINALIZED_ENGINEERING", "target": target,
               "plan_sha256": plan["plan_sha256"], "clean_risk": clean,
               "fields": len(results), "elapsed_seconds": time.monotonic() - started,
               "completed_utc": datetime.now(timezone.utc).isoformat(),
               "scientific_files_sha256": checksums}
    summary["payload_sha256"] = digest_object(summary)
    _write(output_dir / "manifest.json", summary)
    _write(output_dir / "FINALIZED", {"manifest_sha256": sha256_file(output_dir / "manifest.json")})
    return {"target": target, "output": str(output_dir), "fields": len(results),
            "manifest_sha256": sha256_file(output_dir / "manifest.json")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", choices=TARGETS)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.target, args.manifest, args.output_dir)), flush=True)


if __name__ == "__main__":
    main()
