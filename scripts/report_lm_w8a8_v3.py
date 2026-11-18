"""Offline, write-once LM W8A8 NLL/BER tables, figure, and compact export."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from netsurf.manifests import canonical_json_bytes, sha256_file  # noqa: E402
from netsurf.w8a8_campaign import aggregate, export, validate, _read_bound  # noqa: E402
import report_distilbert_w8a8_v2 as reference  # noqa: E402


READY = Path("/projects/kona-assets/manuelbv/NETSURF_ASSET_READINESS.json")
OBS_COLUMNS = ("target", "plan_sha256", "ber", "expected_proposed_faults", "field_index",
               "seed", "field_sha256", "field_record", "method", "protected_bits", "coverage",
               "mask_sha256", "proposed_faults", "applied_faults", "suppressed_faults",
               "expected_applied_faults", "mean_nll", "perplexity", "failed_tokens",
               "scored_tokens", "logit_statistics_sha256")
COND_COLUMNS = ("target", "plan_sha256", "ber", "expected_proposed_faults", "method",
                "protected_bits", "coverage", "independent_fields", "mean_nll",
                "mean_loss_degradation", "ci95_degradation_low", "ci95_degradation_high",
                "penalized_perplexity", "failed_fraction", "mean_proposed_faults",
                "mean_applied_faults", "mean_suppressed_faults", "expected_applied_faults")


def observations(plan: dict, run: Path) -> list[dict]:
    base = reference.observations(plan, run)
    cache = {}
    rows = []
    for item in base:
        file = item["field_record"]
        if file not in cache:
            cache[file] = _read_bound(run / file)
        policy = item["method"] if item["method"] in ("unprotected", "full_protection") else \
                 f'{item["method"]}_b{item["protected_bits"]}'
        metrics = cache[file]["policies"][policy]["risk"]["metrics"]
        row = {k: item[k] for k in OBS_COLUMNS if k in item}
        row.update(mean_nll=item["mean_cross_entropy"], perplexity=metrics["perplexity"],
                   failed_tokens=item["failed_examples"], scored_tokens=item["evaluated_examples"])
        if set(row) != set(OBS_COLUMNS):
            raise ValueError("LM field observation omitted a required outcome")
        rows.append(row)
    if len(rows) != plan["counts"]["final_model_requests"]:
        raise ValueError("LM field observations incomplete")
    return rows


def conditions(plan: dict, rows: list[dict], clean_nll: float) -> list[dict]:
    groups = {}
    for row in rows:
        groups.setdefault((row["ber"], row["method"], row["protected_bits"]), []).append(row)
    n = plan["eligible_universe"]["bits"]
    output = []
    for index, ((ber, method, budget), group) in enumerate(sorted(groups.items())):
        if len(group) != plan["fields_per_ber"] or \
           len({x["field_index"] for x in group}) != len(group):
            raise ValueError("LM condition has missing or duplicate fault fields")
        values = np.asarray([x["mean_nll"] for x in group], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("LM loss nonfinite after frozen failure penalty")
        rng = np.random.default_rng(3204001 + index)
        boot = values[rng.integers(0, len(values), size=(10000, len(values)))].mean(axis=1)-clean_nll
        mean = float(values.mean())
        output.append({"target": plan["target"], "plan_sha256": plan["plan_sha256"],
                       "ber": ber, "expected_proposed_faults": group[0]["expected_proposed_faults"],
                       "method": method, "protected_bits": budget, "coverage": budget/n,
                       "independent_fields": len(group), "mean_nll": mean,
                       "mean_loss_degradation": mean-clean_nll,
                       "ci95_degradation_low": float(np.quantile(boot, .025)),
                       "ci95_degradation_high": float(np.quantile(boot, .975)),
                       "penalized_perplexity": math.exp(mean),
                       "failed_fraction": sum(x["failed_tokens"] for x in group)/
                                          sum(x["scored_tokens"] for x in group),
                       "mean_proposed_faults": float(np.mean([x["proposed_faults"] for x in group])),
                       "mean_applied_faults": float(np.mean([x["applied_faults"] for x in group])),
                       "mean_suppressed_faults": float(np.mean([x["suppressed_faults"] for x in group])),
                       "expected_applied_faults": group[0]["expected_applied_faults"]})
    if len(output) != 78:
        raise ValueError("LM BER/budget/method condition grid incomplete")
    return output


def _csv(path: Path, columns: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("x", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _verify(destination: Path, run: Path) -> dict:
    receipt_path = destination / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if (destination / "FINALIZED").read_text().strip() != sha256_file(receipt_path):
        raise ValueError("LM report finalization checksum changed")
    if sha256_file(destination / "compact.zip") != receipt["compact_zip_sha256"] or \
       sha256_file(run.with_name(run.name + "_export") / "complete.zip") != receipt["complete_zip_sha256"]:
        raise ValueError("LM evidence ZIP changed")
    with zipfile.ZipFile(destination / "compact.zip") as archive:
        if archive.testzip() is not None:
            raise ValueError("LM compact ZIP CRC failed")
        if json.loads(archive.read("evidence_checksums.json")) != receipt["member_sha256"]:
            raise ValueError("LM compact member inventory changed")
        for name, digest in receipt["member_sha256"].items():
            import hashlib
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError("LM compact member hash changed: " + name)
    return receipt


def report(plan_dir: Path, run: Path) -> dict:
    plan, _ = validate(READY, plan_dir)
    if plan["target"] not in ("pythia160m_int8", "qwen3_0_6b_int8"):
        raise ValueError("LM W8A8 plan required")
    summary = aggregate(READY, plan_dir, run)
    full = export(READY, plan_dir, run)
    if summary["fields"] != 72 or full["status"] not in ("FINALIZED", "VERIFIED_FINALIZED_REUSED"):
        raise ValueError("LM scientific campaign incomplete before reporting")
    runtime_path = run.with_name(run.name + "_runtime_receipt.json")
    runtime = json.loads(runtime_path.read_text())
    if runtime.get("exit_code") != 0 or runtime.get("target") != plan["target"]:
        raise ValueError("LM foreground runtime did not complete successfully")
    destination = run.with_name(run.name + "_reporting_v3")
    if destination.exists():
        prior = _verify(destination, run)
        return {"status": "VERIFIED_FINALIZED_REUSED", "target": prior["target"],
                "plan_sha256": prior["plan_sha256"],
                "complete_zip_sha256": prior["complete_zip_sha256"],
                "compact_zip_sha256": prior["compact_zip_sha256"],
                "condition_count": prior["condition_count"],
                "field_observation_count": prior["field_observation_count"]}
    analysis = _read_bound(run / "analysis.json")
    rows = observations(plan, run)
    grouped = conditions(plan, rows, analysis["clean"]["metrics"]["mean_nll"])
    destination.mkdir(parents=True, exist_ok=False)
    _csv(destination / "condition_level.csv", COND_COLUMNS, grouped)
    _csv(destination / "field_observations.csv", OBS_COLUMNS, rows)
    figure_rows = [{**row, "mean_cross_entropy": row["mean_nll"]} for row in grouped]
    label = "Pythia-160M" if plan["target"] == "pythia160m_int8" else "Qwen3-0.6B-Base"
    figure = reference._svg(figure_rows, plan).replace("DistilBERT W8A8", label + " W8A8").replace(
        "Mean cross-entropy minus clean", "Mean token NLL minus clean")
    (destination / "loss_degradation_vs_ber.svg").write_text(figure)
    inputs = {"plan/plan.json": plan_dir / "plan.json",
              "plan/equivalence_audit.json": plan_dir / "equivalence_audit.json",
              "run/analysis.json": run / "analysis.json",
              "run/scientific_payload.json": run / "scientific_payload.json",
              "run/admission.json": run.with_name(run.name + "_admission.json"),
              "run/runtime_receipt.json": runtime_path,
              "core_export/checksums.json": run.with_name(run.name + "_export") / "checksums.json",
              "core_export/FINALIZED": run.with_name(run.name + "_export") / "FINALIZED",
              "report/condition_level.csv": destination / "condition_level.csv",
              "report/field_observations.csv": destination / "field_observations.csv",
              "report/loss_degradation_vs_ber.svg": destination / "loss_degradation_vs_ber.svg"}
    for name, expected in plan["source_files_sha256"].items():
        path = ROOT / name
        if sha256_file(path) != expected:
            raise ValueError("bound LM source changed: " + name)
        inputs["source/" + name] = path
    inputs["source/scripts/report_lm_w8a8_v3.py"] = Path(__file__)
    inputs["source/scripts/report_distilbert_w8a8_v2.py"] = Path(reference.__file__)
    hashes = {name: sha256_file(path) for name, path in inputs.items()}
    with zipfile.ZipFile(destination / "compact.zip", "x", compression=zipfile.ZIP_DEFLATED,
                         allowZip64=True) as archive:
        for name, path in inputs.items():
            archive.write(path, name)
        archive.writestr("evidence_checksums.json", canonical_json_bytes(hashes)+b"\n")
    with zipfile.ZipFile(destination / "compact.zip") as archive:
        if archive.testzip() is not None:
            raise ValueError("LM compact ZIP CRC failed")
    receipt = {"schema": "netsurf-w8a8-lm-reporting/v3", "target": plan["target"],
               "plan_sha256": plan["plan_sha256"],
               "complete_zip_sha256": full["zip_sha256"],
               "compact_zip_sha256": sha256_file(destination / "compact.zip"),
               "condition_count": len(grouped), "field_observation_count": len(rows),
               "member_sha256": hashes}
    (destination / "receipt.json").write_bytes(canonical_json_bytes(receipt)+b"\n")
    (destination / "FINALIZED").write_text(sha256_file(destination / "receipt.json")+"\n")
    commands = [full["scp"],
                f"scp kona:{(destination / 'compact.zip').resolve()} ~/Downloads/{plan['target']}_w8a8_compact_v3.zip",
                f"scp kona:{(destination / 'receipt.json').resolve()} ~/Downloads/{plan['target']}_w8a8_reporting_receipt_v3.json"]
    (destination / "mac_scp_commands.txt").write_text("\n".join(commands)+"\n")
    return {"status": "FINALIZED_OFFLINE_REPORT", "target": plan["target"],
            "condition_count": len(grouped), "field_observation_count": len(rows),
            "complete_zip_sha256": full["zip_sha256"],
            "compact_zip_sha256": receipt["compact_zip_sha256"],
            "receipt_sha256": sha256_file(destination / "receipt.json"),
            "commands": commands}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("plan_dir", type=Path)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(report(args.plan_dir, args.run_dir), sort_keys=True))
