"""Bounded, write-once W8A8 follow-ups; never modifies a frozen source run.

Qwen is an exploratory BER crossover diagnostic with independently seeded
paired packed fields. Pythia is a clean-execution conformance probe only.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from .manifests import canonical_json_bytes, digest_object, sha256_file
from .w8a8_assets import INTERPRETER, load_ready_w8a8
from .w8a8_campaign import _evaluation, _risk_record
from .w8a8_dense_ber_pilot import LAW_VERSION, packed_field
from .w8a8_dense_protection import injected_packed, prepared_protection_bytes, verify_protection
from .w8a8_faults import W8A8FaultAdapter
from .w8a8_wide_ber_campaign import _source, _policy_records, _protection


ROOT = Path(__file__).resolve().parents[1]
QWEN = "qwen3_0_6b_int8"
PYTHIA = "pythia160m_int8"
BERS = (0.05, 0.075)
FIELDS = 24
SEED_BASE = 1537001
POLICIES = ("unprotected", "msb_first", "per_example_absolute")
BUDGET_FRACTION = 0.125
SOURCE_FILES = ("netsurf/w8a8_followup.py", "netsurf/w8a8_dense_ber_pilot.py",
                "netsurf/w8a8_dense_protection.py", "netsurf/w8a8_campaign.py",
                "netsurf/w8a8_wide_ber_campaign.py", "netsurf/w8a8_faults.py",
                "netsurf/transformer_risk.py")


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical_json_bytes(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _verify_plan(path: Path, schema: str) -> dict:
    plan = _read(path)
    if plan.get("schema") != schema or plan.get("plan_sha256") != digest_object(
        {k: v for k, v in plan.items() if k != "plan_sha256"}
    ) or any(sha256_file(ROOT / name) != digest for name, digest in
             plan["source_files_sha256"].items()):
        raise ValueError("frozen follow-up plan or source differs")
    if sha256_file(path) != (path.parent / "FROZEN").read_text().strip():
        raise ValueError("frozen plan file differs")
    return plan


def _risk_diagnostic(risk) -> dict:
    """Separate finite cap saturation from nonfinite/inference failures."""
    row = _risk_record(risk)
    finite = [float(logsumexp - target) for item in risk.logits if item is not None
              for target, logsumexp in (item,)]
    if len(finite) + risk.failed_units != risk.total_units or not all(map(math.isfinite, finite)):
        raise ValueError("risk sufficient statistics are incomplete or nonfinite")
    capped = sum(value >= 30.0 for value in finite)
    row["finite_nll_at_or_above_30_count"] = capped
    row["finite_nll_uncapped_mean"] = float(np.mean(finite)) if finite else None
    row["finite_nll_uncapped_p95"] = float(np.quantile(finite, .95)) if finite else None
    row["finite_nll_uncapped_max"] = max(finite) if finite else None
    row["cap_or_failure_fraction"] = (capped + risk.failed_units) / risk.total_units
    return row


def freeze_qwen(manifest: Path, root: Path) -> dict:
    if root.exists():
        raise FileExistsError(root)
    if Path(sys.executable).resolve() != INTERPRETER.resolve():
        raise ValueError(f"requires pinned W8A8 interpreter: {INTERPRETER}")
    asset = load_ready_w8a8(manifest, QWEN)
    adapter = W8A8FaultAdapter(asset)
    source_plan_path, source_run, source_plan, seal = _source(QWEN)
    records = _policy_records(adapter.universe.bits, source_run, source_plan, seal)
    budget = int(adapter.universe.bits * BUDGET_FRACTION)
    selected = [record for record in records if record["name"] == "unprotected" or
                record["name"] in (f"msb_first_b{budget}", f"per_example_absolute_b{budget}")]
    if len(selected) != 3:
        raise ValueError("required source masks are not sealed")
    for record in selected:
        verify_protection(_protection(record, adapter.universe.bits))
    seeds = [[SEED_BASE + arm * 100000 + index for index in range(FIELDS)]
             for arm in range(len(BERS))]
    plan = {"schema": "netsurf-w8a8-qwen-crossover/v1", "purpose": "exploratory_posthoc_ber_diagnostic",
            "target": QWEN, "bers": list(BERS), "fields_per_ber": FIELDS,
            "seed_base": SEED_BASE, "field_seeds": seeds, "policies": selected,
            "budget_bits": budget, "budget_fraction": BUDGET_FRACTION,
            "eligible_bits": adapter.universe.bits, "eligible_population_sha256": adapter.universe.digest,
            "source_plan_path": str(source_plan_path),
            "source_plan_sha256": sha256_file(source_plan_path),
            "source_selection_seal_sha256": sha256_file(source_run / "selection/seal.json"),
            "ready_record_sha256": asset.record_sha256,
            "ready_manifest_sha256": asset.manifest_sha256,
            "final_cohort": source_plan["final"], "fault_law": LAW_VERSION,
            "fault_lifetime": "persistent stored INT8 XOR over complete final cohort",
            "mitigation": "ideal logical protected-event suppression",
            "loss_contract": "token NLL capped at 30 nats; finite uncapped statistics reported separately",
            "counts": {"proposed_fields": len(BERS) * FIELDS,
                       "faulted_model_evaluations": len(BERS) * FIELDS * len(selected),
                       "scored_tokens_per_evaluation": source_plan["final"]["scored_targets"],
                       "packed_field_bytes": len(BERS) * FIELDS * adapter.universe.codes},
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_files_sha256": {name: sha256_file(ROOT / name) for name in SOURCE_FILES}}
    plan["plan_sha256"] = digest_object(plan)
    root.mkdir(parents=True)
    _write(root / "plan.json", plan)
    (root / "FROZEN").write_text(sha256_file(root / "plan.json") + "\n")
    return {"plan_sha256": plan["plan_sha256"], "counts": plan["counts"]}


def _check_qwen(manifest: Path, root: Path):
    plan = _verify_plan(root / "plan.json", "netsurf-w8a8-qwen-crossover/v1")
    asset = load_ready_w8a8(manifest, QWEN)
    adapter = W8A8FaultAdapter(asset)
    source_plan_path, source_run, source_plan, seal = _source(QWEN)
    policies = _policy_records(adapter.universe.bits, source_run, source_plan, seal)
    if (plan["ready_record_sha256"] != asset.record_sha256 or
        plan["ready_manifest_sha256"] != asset.manifest_sha256 or
        plan["eligible_bits"] != adapter.universe.bits or
        plan["eligible_population_sha256"] != adapter.universe.digest or
        plan["source_plan_path"] != str(source_plan_path) or
        plan["source_plan_sha256"] != sha256_file(source_plan_path) or
        plan["source_selection_seal_sha256"] != sha256_file(source_run / "selection/seal.json") or
        plan["final_cohort"] != source_plan["final"] or
        any(record not in policies for record in plan["policies"])):
        raise ValueError("READY model, source cohort or sealed policy differs")
    for record in plan["policies"]:
        verify_protection(_protection(record, adapter.universe.bits))
    return plan, adapter


def run_qwen(manifest: Path, root: Path) -> dict:
    if Path(sys.executable).resolve() != INTERPRETER.resolve():
        raise ValueError(f"requires pinned W8A8 interpreter: {INTERPRETER}")
    plan, adapter = _check_qwen(manifest, root)
    adapter.load()
    evaluate = _evaluation(adapter, {"target": QWEN, "final": plan["final_cohort"]})
    clean = _risk_diagnostic(evaluate())
    clean_path = root / "clean_start.json"
    if clean_path.exists():
        if _read(clean_path) != clean:
            raise ValueError("clean reference differs on resume")
    else:
        _write(clean_path, clean)
    protections = [(record, _protection(record, plan["eligible_bits"]))
                   for record in plan["policies"]]
    mask_bytes = {record["name"]: prepared_protection_bytes(protection)
                  for record, protection in protections}
    complete = 0
    for arm, ber in enumerate(plan["bers"]):
        for index, seed in enumerate(plan["field_seeds"][arm]):
            field_dir = root / "fields" / f"arm_{arm:02d}" / f"field_{index:03d}"
            field_dir.mkdir(parents=True, exist_ok=True)
            field_path = field_dir / "proposed.u8"
            receipt_path = field_dir / "field.json"
            if receipt_path.exists():
                receipt = _read(receipt_path)
                if (receipt["ber"] != ber or receipt["seed"] != seed or
                    receipt["law_version"] != LAW_VERSION or
                    receipt["universe_bits"] != plan["eligible_bits"] or
                    field_path.stat().st_size != receipt["bytes"] or
                    sha256_file(field_path) != receipt["sha256"]):
                    raise ValueError("saved paired field differs")
            else:
                if field_path.exists():
                    raise RuntimeError("incomplete field requires audit")
                receipt = packed_field(field_path, plan["eligible_bits"], ber, seed)
                _write(receipt_path, receipt)
            proposed = np.memmap(field_path, dtype=np.uint8, mode="r")
            for record, protection in protections:
                result_path = field_dir / f"{record['name']}.json"
                if result_path.exists():
                    result = _read(result_path)
                    if (result.get("payload_sha256") != digest_object({k: v for k, v in result.items()
                                                                        if k != "payload_sha256"}) or
                        result["plan_sha256"] != plan["plan_sha256"] or
                        result["field_sha256"] != receipt["sha256"] or
                        result["mask_sha256"] != protection.digest):
                        raise ValueError("saved policy outcome differs")
                    continue
                started = time.monotonic()
                with injected_packed(adapter, proposed, receipt, protection,
                                     mask_bytes[record["name"]]) as event:
                    risk = _risk_diagnostic(evaluate())
                if event["proposed"] != event["applied"] + event["suppressed"]:
                    raise ValueError("fault accounting does not close")
                result = {"plan_sha256": plan["plan_sha256"], "arm": arm,
                          "field_index": index, "ber": ber, "field_seed": seed,
                          "field_sha256": receipt["sha256"], "policy": record["name"],
                          "mask_sha256": protection.digest, "events": event,
                          "risk": risk, "elapsed_seconds": time.monotonic() - started}
                result["payload_sha256"] = digest_object(result)
                _write(result_path, result)
            del proposed
            complete += 1
            print(json.dumps({"phase": "QWEN", "completed_fields": complete,
                              "total_fields": len(plan["bers"]) * plan["fields_per_ber"],
                              "ber": ber, "field": index}), flush=True)
    clean_end = _risk_diagnostic(evaluate())
    if clean_end != clean:
        raise ValueError("Qwen clean output changed after paired fields")
    if not (root / "clean_end.json").exists():
        _write(root / "clean_end.json", clean_end)
    return {"completed_fields": complete, "clean_restored": True}


def finalize_qwen(root: Path) -> dict:
    """Verify saved fields and observations; never loads a model or infers."""
    if (root / "FINALIZED").exists():
        raise ValueError("Qwen follow-up already finalized")
    plan = _verify_plan(root / "plan.json", "netsurf-w8a8-qwen-crossover/v1")
    clean = _read(root / "clean_start.json")
    if _read(root / "clean_end.json") != clean:
        raise ValueError("clean reference was not restored")
    rows, comparisons = [], []
    for arm, ber in enumerate(plan["bers"]):
        for index, seed in enumerate(plan["field_seeds"][arm]):
            directory = root / "fields" / f"arm_{arm:02d}" / f"field_{index:03d}"
            receipt = _read(directory / "field.json")
            field_path = directory / "proposed.u8"
            if (receipt["ber"] != ber or receipt["seed"] != seed or
                receipt["universe_bits"] != plan["eligible_bits"] or
                receipt["law_version"] != LAW_VERSION or
                field_path.stat().st_size != receipt["bytes"] or
                sha256_file(field_path) != receipt["sha256"]):
                raise ValueError("paired Qwen proposed field differs")
            paired = {}
            for policy in plan["policies"]:
                item = _read(directory / f"{policy['name']}.json")
                event = item["events"]
                if (item.get("payload_sha256") != digest_object({k: v for k, v in item.items()
                                                                    if k != "payload_sha256"}) or
                    item["plan_sha256"] != plan["plan_sha256"] or item["arm"] != arm or
                    item["field_index"] != index or item["ber"] != ber or
                    item["field_seed"] != seed or item["field_sha256"] != receipt["sha256"] or
                    item["policy"] != policy["name"] or
                    item["mask_sha256"] != policy["mask_digest"] or
                    event["proposed"] != receipt["proposed"] or
                    event["proposed"] != event["applied"] + event["suppressed"] or
                    item["risk"]["total_units"] != plan["counts"]["scored_tokens_per_evaluation"]):
                    raise ValueError("saved Qwen policy outcome differs")
                row = {"ber": ber, "field_index": index, "field_seed": seed,
                       "field_sha256": receipt["sha256"], "policy": policy["name"],
                       "mask_sha256": policy["mask_digest"],
                       "mean_nll_nats": item["risk"]["metrics"]["mean_nll"],
                       "finite_nll_at_or_above_30_count": item["risk"]["finite_nll_at_or_above_30_count"],
                       "finite_nll_uncapped_mean": item["risk"]["finite_nll_uncapped_mean"],
                       "cap_or_failure_fraction": item["risk"]["cap_or_failure_fraction"],
                       "failed_units": item["risk"]["failed_units"],
                       "proposed": event["proposed"], "applied": event["applied"],
                       "suppressed": event["suppressed"]}
                rows.append(row)
                paired[policy["name"]] = row
            absolute = next(p for p in plan["policies"] if p["name"].startswith("per_example_absolute"))
            msb = next(p for p in plan["policies"] if p["name"].startswith("msb_first"))
            comparisons.append({"ber": ber, "field_index": index,
                                "field_sha256": receipt["sha256"],
                                "absolute_minus_msb_nats": paired[absolute["name"]]["mean_nll_nats"] -
                                                           paired[msb["name"]]["mean_nll_nats"],
                                "absolute_minus_msb_cap_fraction": paired[absolute["name"]]["cap_or_failure_fraction"] -
                                                                 paired[msb["name"]]["cap_or_failure_fraction"]})
    if len(rows) != len(BERS) * FIELDS * len(plan["policies"]):
        raise ValueError("Qwen follow-up grid incomplete")
    derived = root / "derived"
    derived.mkdir(exist_ok=True)
    for name, values in (("trials.csv", rows), ("paired.csv", comparisons)):
        with (derived / name).open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader(); writer.writerows(values)
    aggregates = []
    for ber in plan["bers"]:
        for policy in plan["policies"]:
            selected = [r for r in rows if r["ber"] == ber and r["policy"] == policy["name"]]
            values = np.asarray([r["mean_nll_nats"] for r in selected], dtype=np.float64)
            aggregates.append({"ber": ber, "policy": policy["name"], "fields": len(values),
                               "mean_nll_nats": float(values.mean()),
                               "mean_cap_or_failure_fraction": float(np.mean([r["cap_or_failure_fraction"]
                                                                                for r in selected])),
                               "mean_finite_uncapped_nll": float(np.mean([r["finite_nll_uncapped_mean"]
                                                                           for r in selected]))})
        diff = np.asarray([r["absolute_minus_msb_nats"] for r in comparisons if r["ber"] == ber])
        rng = np.random.default_rng(1537901 + plan["bers"].index(ber))
        boot = diff[rng.integers(0, len(diff), size=(10000, len(diff)))].mean(axis=1)
        aggregates.append({"ber": ber, "contrast": "absolute_minus_msb",
                           "fields": len(diff), "mean_difference_nats": float(diff.mean()),
                           "ci95_low": float(np.quantile(boot, .025)),
                           "ci95_high": float(np.quantile(boot, .975)),
                           "interpretation": "exploratory post-hoc BER interval; field uncertainty conditional on fixed masks and cohort"})
    _write(derived / "analysis.json", {"plan_sha256": plan["plan_sha256"],
                                       "clean": clean, "conditions_and_contrasts": aggregates})
    files = {str(p.relative_to(root)): sha256_file(p) for p in sorted(root.rglob("*"))
             if p.is_file() and p.name not in ("manifest.json", "FINALIZED")}
    manifest = {"schema": "netsurf-w8a8-qwen-crossover-result/v1",
                "status": "FINALIZED_EXPLORATORY", "plan_sha256": plan["plan_sha256"],
                "proposed_fields": len(BERS) * FIELDS, "policy_observations": len(rows),
                "paired_field_contrasts": len(comparisons), "files_sha256": files}
    manifest["payload_sha256"] = digest_object(manifest)
    _write(root / "manifest.json", manifest)
    (root / "FINALIZED").write_text(sha256_file(root / "manifest.json") + "\n")
    return {"status": manifest["status"], "fields": manifest["proposed_fields"],
            "observations": len(rows)}


def freeze_pythia(manifest: Path, root: Path, processes: int, repeats: int) -> dict:
    if root.exists() or processes < 2 or repeats < 2:
        raise ValueError("new destination, >=2 processes and >=2 repeats required")
    asset = load_ready_w8a8(manifest, PYTHIA)
    adapter = W8A8FaultAdapter(asset)
    source_plan_path, _, source_plan, _ = _source(PYTHIA)
    plan = {"schema": "netsurf-w8a8-pythia-clean-conformance/v1",
            "target": PYTHIA, "purpose": "bounded_shard_clean_diagnostic_no_faults",
            "processes": processes, "repeats_per_process": repeats,
            "ready_record_sha256": asset.record_sha256,
            "ready_manifest_sha256": asset.manifest_sha256,
            "eligible_population_sha256": adapter.universe.digest,
            "source_plan_path": str(source_plan_path),
            "source_plan_sha256": sha256_file(source_plan_path),
            "final_cohort": source_plan["final"],
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_files_sha256": {name: sha256_file(ROOT / name) for name in SOURCE_FILES}}
    plan["plan_sha256"] = digest_object(plan)
    root.mkdir(parents=True)
    _write(root / "plan.json", plan)
    (root / "FROZEN").write_text(sha256_file(root / "plan.json") + "\n")
    return {"plan_sha256": plan["plan_sha256"], "clean_requests": processes * repeats}


def probe_pythia(manifest: Path, root: Path, worker: int) -> dict:
    if Path(sys.executable).resolve() != INTERPRETER.resolve():
        raise ValueError(f"requires pinned W8A8 interpreter: {INTERPRETER}")
    plan = _verify_plan(root / "plan.json", "netsurf-w8a8-pythia-clean-conformance/v1")
    if not 0 <= worker < plan["processes"]:
        raise ValueError("worker outside frozen plan")
    asset = load_ready_w8a8(manifest, PYTHIA)
    adapter = W8A8FaultAdapter(asset)
    if (asset.record_sha256 != plan["ready_record_sha256"] or
        asset.manifest_sha256 != plan["ready_manifest_sha256"] or
        adapter.universe.digest != plan["eligible_population_sha256"]):
        raise ValueError("Pythia READY identity differs")
    adapter.load()
    evaluate = _evaluation(adapter, {"target": PYTHIA, "final": plan["final_cohort"]})
    from kona_assets import int8_torch as execution
    import torch
    original = {name: execution.tensor_sha256(module.qweight)
                for name, module in adapter.modules.items()}
    if original != adapter.baseline_hashes:
        raise ValueError("clean qweights differ from READY baseline")
    records = []
    for number in range(plan["repeats_per_process"]):
        started = time.monotonic()
        risk = evaluate()
        current = {name: execution.tensor_sha256(module.qweight)
                   for name, module in adapter.modules.items()}
        if current != original:
            raise ValueError("clean inference mutated stored qweights")
        records.append({"repeat": number, "risk": _risk_diagnostic(risk),
                        "token_stat_sha256": digest_object(risk.logits),
                        "token_stats": risk.logits,
                        "elapsed_seconds": time.monotonic() - started})
    result = {"plan_sha256": plan["plan_sha256"], "worker": worker,
              "pid": os.getpid(), "torch_version": torch.__version__,
              "torch_num_threads": torch.get_num_threads(),
              "openmp_threads_env": os.environ.get("OMP_NUM_THREADS"),
              "mkl_threads_env": os.environ.get("MKL_NUM_THREADS"),
              "qweight_hashes_sha256": digest_object(original),
              "records": records, "completed_utc": datetime.now(timezone.utc).isoformat()}
    result["payload_sha256"] = digest_object(result)
    _write(root / f"worker_{worker:02d}.json", result)
    print(json.dumps({"phase": "PYTHIA", "worker": worker,
                      "clean_requests": len(records), "mean_nll": [x["risk"]["metrics"]["mean_nll"]
                                                         for x in records]}), flush=True)
    return {"worker": worker, "clean_requests": len(records)}


def finalize_pythia(root: Path) -> dict:
    plan = _verify_plan(root / "plan.json", "netsurf-w8a8-pythia-clean-conformance/v1")
    records = []
    for worker in range(plan["processes"]):
        path = root / f"worker_{worker:02d}.json"
        item = _read(path)
        if (item["worker"] != worker or item["plan_sha256"] != plan["plan_sha256"] or
            item.get("payload_sha256") != digest_object({k: v for k, v in item.items()
                                                        if k != "payload_sha256"}) or
            len(item["records"]) != plan["repeats_per_process"]):
            raise ValueError("Pythia clean worker receipt incomplete")
        records.extend(item["records"])
    baseline = records[0]["token_stats"]
    differing = []
    for index, item in enumerate(records):
        if item["token_stats"] != baseline:
            first = next((i for i, (a, b) in enumerate(zip(baseline, item["token_stats"])) if a != b), None)
            differing.append({"record_index": index, "first_differing_token": first})
    summary = {"schema": "netsurf-w8a8-pythia-clean-result/v1",
               "plan_sha256": plan["plan_sha256"], "request_count": len(records),
               "distinct_token_stat_sha256": sorted({x["token_stat_sha256"] for x in records}),
               "first_differences": differing,
               "all_bitwise_identical": not differing,
               "scientific_files_sha256": {p.name: sha256_file(p) for p in sorted(root.iterdir())
                                           if p.is_file() and p.name != "manifest.json"}}
    summary["payload_sha256"] = digest_object(summary)
    _write(root / "manifest.json", summary)
    (root / "FINALIZED").write_text(sha256_file(root / "manifest.json") + "\n")
    return {"request_count": len(records), "all_bitwise_identical": not differing,
            "first_differences": differing}


def status(root: Path) -> dict:
    if not (root / "plan.json").is_file():
        return {"status": "NOT_STARTED"}
    plan = _read(root / "plan.json")
    if plan["schema"] == "netsurf-w8a8-qwen-crossover/v1":
        complete = sum(all((d / f"{p['name']}.json").is_file() for p in plan["policies"])
                       for d in (root / "fields").glob("arm_*/field_*") if d.is_dir())
        return {"status": "FINALIZED" if (root / "FINALIZED").is_file() else "RUNNING",
                "completed_fields": complete, "total_fields": len(plan["bers"]) * plan["fields_per_ber"]}
    complete = sum((root / f"worker_{i:02d}.json").is_file() for i in range(plan["processes"]))
    return {"status": "FINALIZED" if (root / "FINALIZED").is_file() else "RUNNING",
            "completed_workers": complete, "total_workers": plan["processes"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("freeze-qwen", "run-qwen", "freeze-pythia", "probe-pythia"):
        command = sub.add_parser(name)
        command.add_argument("manifest", type=Path)
        command.add_argument("root", type=Path)
        if name == "freeze-pythia":
            command.add_argument("--processes", type=int, default=4)
            command.add_argument("--repeats", type=int, default=2)
        if name == "probe-pythia":
            command.add_argument("worker", type=int)
    for name in ("finalize-pythia", "finalize-qwen", "status"):
        sub.add_parser(name).add_argument("root", type=Path)
    args = parser.parse_args()
    result = {
        "freeze-qwen": lambda: freeze_qwen(args.manifest, args.root),
        "run-qwen": lambda: run_qwen(args.manifest, args.root),
        "freeze-pythia": lambda: freeze_pythia(args.manifest, args.root, args.processes, args.repeats),
        "probe-pythia": lambda: probe_pythia(args.manifest, args.root, args.worker),
        "finalize-pythia": lambda: finalize_pythia(args.root),
        "finalize-qwen": lambda: finalize_qwen(args.root),
        "status": lambda: status(args.root),
    }[args.command]()
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
