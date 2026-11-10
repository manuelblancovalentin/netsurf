"""Fresh Pythia high-BER campaign with independent fields and clean gates.

The fault law, frozen masks, task loss and condition analysis are those of the
validated wide-BER engine. This module adds a cross-process clean preflight and
exact clean-output checks before and after every proposed field. It never
modifies the earlier failed campaign.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from .manifests import digest_object, sha256_file
from .w8a8_assets import INTERPRETER
from .w8a8_campaign import _evaluation, _risk_record
from .w8a8_dense_protection import injected_packed, prepared_protection_bytes
from .w8a8_faults import W8A8FaultAdapter
from . import w8a8_wide_ber_campaign as wide


TARGET = "pythia160m_int8"
SEED_BASE = 1538001
SHARDS = 4
REPEATS = 2
SCHEMA = "netsurf-w8a8-pythia-wide-clean-gated/v2"
SOURCE = Path(__file__).resolve()


def _write(path: Path, value: dict) -> None:
    wide._write(path, value)


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _write_or_verify(path: Path, value: dict) -> None:
    if path.exists():
        if _read(path) != value:
            raise ValueError(f"existing clean record differs: {path}")
    else:
        _write(path, value)


def _reference_clean() -> tuple[Path, dict]:
    _, run, original, _ = wide._source(TARGET)
    path = run / "final/clean.json"
    value = wide._checked(path)
    if value["plan_sha256"] != original["plan_sha256"]:
        raise ValueError("historical finalized clean reference has wrong plan")
    export = run.parent / (run.name + "_export")
    checks_path = export / "checksums.json"
    archive = export / "complete.zip"
    marker = export / "FINALIZED"
    if not (checks_path.is_file() and archive.is_file() and marker.is_file()):
        raise ValueError("historical clean reference has no finalized export")
    checks = _read(checks_path)
    if (checks.get("run/final/clean.json") != sha256_file(path) or
        marker.read_text().strip() != digest_object({
            "plan_sha256": original["plan_sha256"],
            "zip_sha256": sha256_file(archive), "members": checks})):
        raise ValueError("historical clean reference disagrees with finalized source export")
    return path, value["risk"]


def freeze(manifest: Path, root: Path) -> dict:
    if root.exists() or Path(sys.executable).resolve() != INTERPRETER.resolve():
        raise ValueError("fresh root and pinned W8A8 interpreter required")
    historical, clean = _reference_clean()
    old_seed, old_files = wide.SEED_BASE, wide.SOURCE_FILES
    try:
        wide.SEED_BASE = SEED_BASE
        wide.SOURCE_FILES = old_files + ("netsurf/w8a8_pythia_wide_confirm.py",)
        result = wide.freeze(manifest, TARGET, root)
    finally:
        wide.SEED_BASE, wide.SOURCE_FILES = old_seed, old_files
    plan = _read(root / "plan.json")
    if (plan["field_seeds"][0][0] != SEED_BASE or
        plan["counts"]["fields"] != len(wide.BERS) * wide.FIELDS_PER_BER or
        plan["counts"]["planned_faulted_model_evaluations"] != 4800):
        raise ValueError("fresh Pythia wide-BER plan is incomplete")
    protocol = {"schema": SCHEMA, "plan_sha256": plan["plan_sha256"],
                "source_module_sha256": sha256_file(SOURCE),
                "historical_clean_path": str(historical),
                "historical_clean_file_sha256": sha256_file(historical),
                "expected_clean_risk": clean,
                "preflight_processes": SHARDS, "clean_repeats_per_process": REPEATS,
                "field_gate": "exact clean task-risk identity before and after every proposed field",
                "new_field_seed_base": SEED_BASE,
                "planned_extra_clean_requests": SHARDS * REPEATS + 2 * len(wide.BERS) * wide.FIELDS_PER_BER + 2 * SHARDS,
                "interpretation": "separate new whole-model matched campaign; no repair or pooling of failed source shards"}
    protocol["payload_sha256"] = digest_object(protocol)
    _write(root / "clean_protocol.json", protocol)
    _write(root / "CLEAN_PROTOCOL_FROZEN.json", {"file_sha256": sha256_file(root / "clean_protocol.json")})
    return {"plan_sha256": plan["plan_sha256"], "clean_protocol_sha256": protocol["payload_sha256"],
            "faulted_model_evaluations": 4800, "extra_clean_requests": protocol["planned_extra_clean_requests"]}


def _protocol(root: Path) -> dict:
    protocol = _read(root / "clean_protocol.json")
    if (protocol.get("schema") != SCHEMA or
        protocol.get("payload_sha256") != digest_object({k: v for k, v in protocol.items()
                                                        if k != "payload_sha256"}) or
        sha256_file(root / "clean_protocol.json") !=
        _read(root / "CLEAN_PROTOCOL_FROZEN.json")["file_sha256"] or
        protocol["source_module_sha256"] != sha256_file(SOURCE) or
        protocol["historical_clean_file_sha256"] != sha256_file(Path(protocol["historical_clean_path"]))):
        raise ValueError("Pythia clean-conformance protocol changed")
    return protocol


def _assert_clean(risk, expected: dict, *, root: Path, label: str, baseline_logits=None) -> None:
    record = _risk_record(risk)
    if record == expected:
        return
    first = None
    if baseline_logits is not None:
        first = next((index for index, (a, b) in enumerate(zip(baseline_logits, risk.logits))
                      if a != b), None)
    diagnostic = {"label": label, "expected": expected, "observed": record,
                  "first_differing_token": first,
                  "observed_token_stats": risk.logits,
                  "recorded_utc": datetime.now(timezone.utc).isoformat()}
    _write(root / "clean_drift" / f"{label}.json", diagnostic)
    raise RuntimeError(f"Pythia exact clean conformance failed at {label}; no result promoted")


def preflight_worker(manifest: Path, root: Path, shard: int) -> dict:
    if Path(sys.executable).resolve() != INTERPRETER.resolve() or not 0 <= shard < SHARDS:
        raise ValueError("pinned W8A8 interpreter and valid shard required")
    protocol = _protocol(root)
    plan, asset = wide.validate(manifest, root, verify_masks=True)
    if plan["plan_sha256"] != protocol["plan_sha256"]:
        raise ValueError("clean protocol and scientific plan differ")
    receipt_path = root / "preflight" / f"shard_{shard:02d}.json"
    if receipt_path.exists():
        prior = _read(receipt_path)
        if (prior.get("shard") != shard or prior.get("plan_sha256") != plan["plan_sha256"] or
            prior.get("payload_sha256") != digest_object({k: v for k, v in prior.items()
                                                           if k != "payload_sha256"}) or
            len(prior.get("records", [])) != REPEATS or
            any(row["risk"] != protocol["expected_clean_risk"] for row in prior["records"])):
            raise ValueError("existing preflight receipt differs")
        return {"shard": shard, "clean_requests": REPEATS, "reused_verified": True}
    adapter = W8A8FaultAdapter(asset)
    adapter.load()
    evaluate = _evaluation(adapter, {"target": TARGET, "final": plan["final_cohort"]})
    from kona_assets import int8_torch as execution
    before = {name: execution.tensor_sha256(module.qweight)
              for name, module in adapter.modules.items()}
    if before != adapter.baseline_hashes:
        raise ValueError("stored qweights differ before clean preflight")
    rows = []
    for repeat in range(REPEATS):
        risk = evaluate()
        _assert_clean(risk, protocol["expected_clean_risk"], root=root,
                      label=f"preflight_{shard:02d}_{repeat:02d}")
        rows.append({"repeat": repeat, "risk": _risk_record(risk),
                     "token_stats_sha256": digest_object(risk.logits),
                     "token_stats": risk.logits})
    after = {name: execution.tensor_sha256(module.qweight)
             for name, module in adapter.modules.items()}
    if before != after:
        raise ValueError("clean preflight mutated qweights")
    result = {"schema": SCHEMA, "plan_sha256": plan["plan_sha256"],
              "shard": shard, "pid": os.getpid(), "qweight_hashes_sha256": digest_object(before),
              "records": rows}
    result["payload_sha256"] = digest_object(result)
    _write(receipt_path, result)
    print(json.dumps({"stage": "preflight", "shard": shard, "clean_requests": REPEATS,
                      "nll": rows[0]["risk"]["metrics"]["mean_nll"]}), flush=True)
    return {"shard": shard, "clean_requests": REPEATS}


def seal_preflight(root: Path) -> dict:
    protocol = _protocol(root)
    plan = _read(root / "plan.json")
    if plan["plan_sha256"] != protocol["plan_sha256"]:
        raise ValueError("preflight plan differs")
    rows = []
    for shard in range(SHARDS):
        path = root / "preflight" / f"shard_{shard:02d}.json"
        item = _read(path)
        if (item["shard"] != shard or item["plan_sha256"] != plan["plan_sha256"] or
            item.get("payload_sha256") != digest_object({k: v for k, v in item.items()
                                                        if k != "payload_sha256"}) or
            len(item["records"]) != REPEATS):
            raise ValueError("incomplete Pythia preflight worker")
        rows.extend(item["records"])
    if any(row["risk"] != protocol["expected_clean_risk"] for row in rows):
        raise ValueError("preflight clean risk differs from finalized historical reference")
    if len({row["token_stats_sha256"] for row in rows}) != 1:
        raise ValueError("preflight per-token clean outputs differ across processes")
    if (root / "preflight" / "SEALED.json").exists():
        prior = _read(root / "preflight" / "SEALED.json")
        if prior["token_stats_sha256"] != rows[0]["token_stats_sha256"]:
            raise ValueError("sealed preflight differs")
        return prior
    seal = {"schema": SCHEMA, "plan_sha256": plan["plan_sha256"],
            "clean_risk": protocol["expected_clean_risk"],
            "token_stats_sha256": rows[0]["token_stats_sha256"],
            "worker_file_sha256": {f"shard_{i:02d}.json": sha256_file(root / "preflight" / f"shard_{i:02d}.json")
                                   for i in range(SHARDS)},
            "clean_requests": len(rows)}
    seal["payload_sha256"] = digest_object(seal)
    _write(root / "preflight" / "SEALED.json", seal)
    return seal


def _verified_seal(root: Path, protocol: dict) -> dict:
    seal = _read(root / "preflight" / "SEALED.json")
    if (seal["plan_sha256"] != protocol["plan_sha256"] or
        seal["clean_risk"] != protocol["expected_clean_risk"] or
        seal.get("payload_sha256") != digest_object({k: v for k, v in seal.items()
                                                    if k != "payload_sha256"}) or
        any(sha256_file(root / "preflight" / name) != digest
            for name, digest in seal["worker_file_sha256"].items())):
        raise ValueError("Pythia cross-process clean preflight is not sealed")
    return seal


def run_shard(manifest: Path, root: Path, shard: int) -> dict:
    if Path(sys.executable).resolve() != INTERPRETER.resolve() or not 0 <= shard < SHARDS:
        raise ValueError("pinned W8A8 interpreter and valid shard required")
    protocol = _protocol(root)
    seal = _verified_seal(root, protocol)
    plan, asset = wide.validate(manifest, root, verify_masks=True)
    if plan["plan_sha256"] != protocol["plan_sha256"] or plan["execution_shards"] != SHARDS:
        raise ValueError("fresh plan and clean protocol differ")
    adapter = W8A8FaultAdapter(asset)
    adapter.load()
    evaluate = _evaluation(adapter, {"target": TARGET, "final": plan["final_cohort"]})
    clean = evaluate()
    _assert_clean(clean, seal["clean_risk"], root=root, label=f"shard_{shard:02d}_start")
    clean_record = _risk_record(clean)
    _write_or_verify(root / "clean" / f"shard_{shard:02d}.json", clean_record)
    protections = [(record, wide._protection(record, plan["eligible_bits"]))
                   for record in plan["policies"]]
    protected_bytes = {record["name"]: prepared_protection_bytes(protection)
                       for record, protection in protections}
    completed = 0
    for index in range(shard, plan["fields_per_ber"], SHARDS):
        for arm, ber in enumerate(plan["bers"]):
            field_path, meta_path, directory = wide._field_paths(root, arm, index)
            gate_path = directory / "clean_gate.json"
            if gate_path.exists():
                gate = _read(gate_path)
                if (gate["plan_sha256"] != plan["plan_sha256"] or
                    gate.get("payload_sha256") != digest_object({k: v for k, v in gate.items()
                                                                 if k != "payload_sha256"}) or
                    gate["clean_risk_sha256"] != digest_object(clean_record) or
                    any(not (directory / f"{record['name']}.json").is_file()
                        for record in plan["policies"])):
                    raise ValueError("saved complete field clean gate differs")
                completed += 1
                continue
            if any((directory / f"{record['name']}.json").exists() for record in plan["policies"]):
                raise RuntimeError("partial field with no post-clean gate requires explicit audit")
            pre = evaluate()
            _assert_clean(pre, seal["clean_risk"], root=root,
                          label=f"shard_{shard:02d}_arm_{arm:02d}_field_{index:03d}_pre",
                          baseline_logits=clean.logits)
            field_path, meta, directory = wide._field(root, plan, arm, index)
            proposed = np.memmap(field_path, dtype=np.uint8, mode="r")
            for record, protection in protections:
                started = time.monotonic()
                if protection.kind == "full":
                    event = {"field_sha256": meta["sha256"], "mask_sha256": protection.digest,
                             "proposed": meta["proposed"], "applied": 0,
                             "suppressed": meta["proposed"]}
                    risk = clean_record
                    origin = "derived_exact_zero_applied_faults"
                else:
                    with injected_packed(adapter, proposed, meta, protection,
                                         protected_bytes[record["name"]]) as event:
                        risk = _risk_record(evaluate())
                    origin = "authoritative_faulted_inference"
                if event["proposed"] != event["applied"] + event["suppressed"]:
                    raise ValueError("Pythia protected-event accounting differs")
                observation = {"plan_sha256": plan["plan_sha256"], "arm": arm,
                    "field_index": index, "ber": ber, "field_seed": meta["seed"],
                    "field_sha256": meta["sha256"], "policy": record["name"],
                    "mask_sha256": protection.digest, "budget_bits": protection.budget,
                    "events": event, "risk": risk, "origin": origin,
                    "elapsed_seconds": time.monotonic() - started}
                observation["payload_sha256"] = digest_object(observation)
                _write(directory / f"{record['name']}.json", observation)
            del proposed
            post = evaluate()
            _assert_clean(post, seal["clean_risk"], root=root,
                          label=f"shard_{shard:02d}_arm_{arm:02d}_field_{index:03d}_post",
                          baseline_logits=clean.logits)
            gate = {"schema": SCHEMA, "plan_sha256": plan["plan_sha256"],
                    "arm": arm, "field_index": index, "field_sha256": meta["sha256"],
                    "clean_risk_sha256": digest_object(clean_record),
                    "pre_post_exact": True, "policy_observations": len(protections)}
            gate["payload_sha256"] = digest_object(gate)
            _write(gate_path, gate)
            completed += 1
            print(json.dumps({"stage": "fault_fields", "shard": shard,
                              "completed_fields_this_shard": completed,
                              "ber": ber, "field": index}), flush=True)
    end = evaluate()
    _assert_clean(end, seal["clean_risk"], root=root, label=f"shard_{shard:02d}_end",
                  baseline_logits=clean.logits)
    _write_or_verify(root / "clean" / f"shard_{shard:02d}_end.json", _risk_record(end))
    return {"shard": shard, "completed_fields": completed}


def finalize(manifest: Path, root: Path) -> dict:
    """Offline integrity and statistics; never loads a model or samples a fault."""
    protocol = _protocol(root)
    _verified_seal(root, protocol)
    plan, _ = wide.validate(manifest, root, verify_masks=True)
    if plan["plan_sha256"] != protocol["plan_sha256"]:
        raise ValueError("final plan differs from clean protocol")
    gates = 0
    for arm in range(len(plan["bers"])):
        for index in range(plan["fields_per_ber"]):
            directory = root / "fields" / f"arm_{arm:02d}" / f"field_{index:03d}"
            gate = _read(directory / "clean_gate.json")
            receipt = _read(directory / "field.json")
            if (gate["arm"] != arm or gate["field_index"] != index or
                gate["plan_sha256"] != plan["plan_sha256"] or
                gate["field_sha256"] != receipt["sha256"] or
                gate["clean_risk_sha256"] != digest_object(protocol["expected_clean_risk"]) or
                gate["pre_post_exact"] is not True or
                gate["policy_observations"] != len(plan["policies"]) or
                gate["payload_sha256"] != digest_object({k: v for k, v in gate.items()
                                                         if k != "payload_sha256"})):
                raise ValueError("Pythia field-level clean gate differs")
            gates += 1
    result = wide.finalize(manifest, root)
    if gates != result["fields"]:
        raise ValueError("not all finalized fields had exact clean gates")
    return {**result, "exact_clean_gated_fields": gates,
            "clean_preflight_requests": SHARDS * REPEATS}


def status(root: Path) -> dict:
    if not (root / "plan.json").is_file():
        return {"status": "NOT_STARTED"}
    gates = sum(1 for p in (root / "fields").glob("arm_*/field_*/clean_gate.json") if p.is_file())
    return {"status": "FINALIZED" if (root / "FINALIZED").is_file() else
                      "RUNNING" if (root / "preflight/SEALED.json").is_file() else "PREFLIGHT",
            "preflight_workers": sum((root / "preflight" / f"shard_{i:02d}.json").is_file()
                                     for i in range(SHARDS)),
            "completed_clean_gated_fields": gates, "planned_fields": 192}


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("freeze", "preflight-worker", "run-shard", "finalize"):
        command = sub.add_parser(name)
        command.add_argument("manifest", type=Path)
        command.add_argument("root", type=Path)
        if name in ("preflight-worker", "run-shard"):
            command.add_argument("shard", type=int)
    for name in ("seal-preflight", "status"):
        sub.add_parser(name).add_argument("root", type=Path)
    args = parser.parse_args()
    result = {"freeze": lambda: freeze(args.manifest, args.root),
              "preflight-worker": lambda: preflight_worker(args.manifest, args.root, args.shard),
              "seal-preflight": lambda: seal_preflight(args.root),
              "run-shard": lambda: run_shard(args.manifest, args.root, args.shard),
              "finalize": lambda: finalize(args.manifest, args.root),
              "status": lambda: status(args.root)}[args.command]()
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
