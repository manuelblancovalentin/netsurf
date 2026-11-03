"""Read-only attribution audit of finalized targeted LM archives.

The structural mask is data-free. This module never loads a model or changes a
finalized run. Its derived output is descriptive, not a new held-out trial.
"""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import zipfile

from .manifests import canonical_json_bytes, digest_object, sha256_file

BIT = re.compile(r"^(?P<parameter>.+)\[(?P<word>\d+)\]#b(?P<position>\d+)$")
POLICIES = ("random_b12", "significance_b12", "exact_isolated_b12",
            "random_b24", "significance_b24", "exact_isolated_b24",
            "unprotected", "full")


def bit_parts(key: str, width: int) -> tuple[str, int, int, str]:
    if width not in (16, 32):
        raise ValueError("only exact FP32 and BF16 encodings are supported")
    match = BIT.fullmatch(key)
    if match is None:
        raise ValueError(f"invalid stored bit identity: {key}")
    position = int(match["position"])
    if not 0 <= position < width:
        raise ValueError("bit position outside stored representation")
    exponent_start = 23 if width == 32 else 7
    category = ("sign" if position == width - 1 else
                "exponent" if position >= exponent_start else "significand")
    return match["parameter"], int(match["word"]), position, category


def structural_mask(keys: list[str], width: int, budget: int) -> list[str]:
    """Exponent-first, then sign, then significand; canonical identity ties."""
    if len(set(keys)) != len(keys) or not 0 <= budget <= len(keys):
        raise ValueError("unique bit population and legal budget required")
    priority = {"exponent": 0, "sign": 1, "significand": 2}
    ordered = sorted(keys, key=lambda key: (
        priority[bit_parts(key, width)[3]], *bit_parts(key, width)[:3]))
    return sorted(ordered[:budget], key=lambda key: bit_parts(key, width)[:3])


def risk_decomposition(risk: dict, penalty: float) -> dict:
    """Separate failed-token penalty, finite cap and ordinary finite loss."""
    n = risk["total_units"]
    if (n <= 0 or len(risk["logits"]) != n or
        len(risk["failure_kinds"]) != n or
        risk["logit_record_kind"] != "target_logit_and_logsumexp"):
        raise ValueError("incomplete saved token statistics")
    failed = capped = 0
    failed_sum = capped_sum = ordinary_sum = 0.0
    kinds = Counter()
    for stats, kind in zip(risk["logits"], risk["failure_kinds"]):
        if stats is None:
            if kind not in {"inference_failure", "nonfinite_logits"}:
                raise ValueError("missing token statistics without failure identity")
            failed += 1; kinds[kind] += 1; failed_sum += penalty
        else:
            if kind is not None or len(stats) != 2 or not all(map(math.isfinite, stats)):
                raise ValueError("invalid finite token statistics")
            raw = stats[1] - stats[0]
            if raw >= penalty:
                capped += 1; capped_sum += penalty
            else:
                ordinary_sum += max(0.0, raw)
    mean = (failed_sum + capped_sum + ordinary_sum) / n
    if failed != risk["failed_units"] or not math.isclose(mean, risk["mean_nll"], abs_tol=2e-5):
        raise ValueError("saved mean NLL or failure count disagrees with token statistics")
    state = ("fully_nonfinite_or_failed" if failed == n else
             "partially_nonfinite_or_failed" if failed else
             "finite_catastrophic" if capped else "ordinary_finite")
    return {"state": state, "failed_tokens": failed,
            "nonfinite_tokens": kinds["nonfinite_logits"],
            "inference_failed_tokens": kinds["inference_failure"],
            "finite_capped_tokens": capped,
            "failed_penalty_mean_nll": failed_sum / n,
            "finite_cap_mean_nll": capped_sum / n,
            "ordinary_finite_mean_nll": ordinary_sum / n,
            "saved_mean_nll": risk["mean_nll"]}


def _read_json(archive: zipfile.ZipFile, name: str, plan_hash: str | None = None) -> dict:
    document = json.loads(archive.read(name))
    payload = document.get("payload_sha256")
    if payload is not None and payload != digest_object({k: v for k, v in document.items()
                                                         if k != "payload_sha256"}):
        raise ValueError(f"payload digest changed: {name}")
    if plan_hash is not None and document.get("plan_sha256") != plan_hash:
        raise ValueError(f"plan identity changed: {name}")
    return document


def _event_bit(event: str) -> str:
    if not event.startswith("persistent:"):
        raise ValueError("unexpected nonpersistent event")
    return event.split(":", 1)[1]


def _mask_digest(keys: list[str], width: int) -> str:
    # ProtectionMask orders parsed BitIdentity tuples, not displayed strings.
    ordered = sorted(keys, key=lambda key: bit_parts(key, width)[:3])
    return hashlib.sha256(json.dumps(ordered, separators=(",", ":")).encode()).hexdigest()


def _schedule_digest(events: list[str]) -> str:
    return hashlib.sha256(json.dumps(events, separators=(",", ":")).encode()).hexdigest()


def _normalized(keys: list[str], width: int) -> list[str]:
    return [f"{parameter}[{word}]#{category}" for key in keys
            for parameter, word, _, category in (bit_parts(key, width),)]


def audit_archive(path: Path) -> dict:
    """Verify embedded identities and derive descriptive field-level tables."""
    path = Path(path)
    checks_path = path.parent / "checksums.json"
    checks = json.loads(checks_path.read_text())
    archive_hash = sha256_file(path)
    if checks["complete.zip"] != archive_hash:
        raise ValueError("complete archive differs from finalized export checksum")
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError("archive CRC failure")
        plan = _read_json(archive, "plan.json")
        plan_hash = plan["plan_sha256"]
        if plan_hash != digest_object({k: v for k, v in plan.items() if k != "plan_sha256"}):
            raise ValueError("frozen plan digest changed")
        if (hashlib.sha256(archive.read("assets/NETSURF_ASSET_READINESS.json")).hexdigest()
                != plan["asset_manifest_sha256"] or
            hashlib.sha256(archive.read("assets/transformer_runtime_registry.json")).hexdigest()
                != plan["registry_sha256"]):
            raise ValueError("frozen asset or runtime snapshot changed")
        for source, expected in plan["source_files_sha256"].items():
            if hashlib.sha256(archive.read("source/" + source)).hexdigest() != expected:
                raise ValueError(f"frozen scientific source changed: {source}")
        if plan["protocol_variant"] != "targeted-lm-96/v1" or len(plan["candidate_bits"]) != 96:
            raise ValueError("unexpected campaign protocol or bit population")
        width = 32 if plan["stored_precision"] == "FP32" else 16
        keys = plan["candidate_bits"]
        if len(set(keys)) != 96 or digest_object(keys) != plan["eligible_population_sha256"]:
            raise ValueError("candidate population hash changed")
        classes = Counter(bit_parts(key, width)[3] for key in keys)
        selections = {}
        seal = _read_json(archive, "run/selection/seal.json", plan_hash)
        for method in ("random", "significance", "exact_isolated"):
            name = f"run/selection/{method}.json"
            member = _read_json(archive, name, plan_hash)
            if hashlib.sha256(archive.read(name)).hexdigest() != seal["artifacts"][method]["sha256"]:
                raise ValueError("selection seal member hash changed")
            if len(member["signed_scores"]) != 96 or member["bit_order_sha256"] != plan["eligible_population_sha256"]:
                raise ValueError("selection scores changed")
            selections[method] = member
        for ordinal, key in enumerate(keys):
            item = _read_json(archive, f"run/selection/isolated/bit_{ordinal:03d}.json", plan_hash)
            if item["bit"] != key or item["ordinal"] != ordinal:
                raise ValueError("isolated response identity changed")
            if hashlib.sha256(archive.read(f"run/selection/isolated/bit_{ordinal:03d}.json")).hexdigest() != selections["exact_isolated"]["isolated_member_file_sha256"][ordinal]:
                raise ValueError("isolated response file hash changed")
        selected = {f"{method}_b{budget}": selections[method]["masks"][str(budget)]
                    for method in selections for budget in (12, 24)}
        selected.update({f"exponent_first_b{budget}": structural_mask(keys, width, budget)
                         for budget in (12, 24)})
        selected["unprotected"] = []
        selected["full"] = sorted(keys)
        for name, mask in selected.items():
            if name not in {"unprotected", "full"} and len(mask) != int(name.rsplit("b", 1)[1]):
                raise ValueError("selection mask cardinality changed")
            if not set(mask) <= set(keys):
                raise ValueError("selection mask outside frozen population")
        mask_names = [f"{name}_b{budget}" for budget in (12, 24)
                      for name in ("random", "significance", "exponent_first", "exact_isolated")]
        overlap = {first: {second: len(set(selected[first]) & set(selected[second]))
                           for second in mask_names} for first in mask_names}
        clean = _read_json(archive, "run/final/clean.json", plan_hash)["risk"]
        analysis = _read_json(archive, "run/analysis.json", plan_hash)
        field_rows = []
        for arm in (1, 2):
            for field in range(32):
                prefix = f"run/final/ber_{arm:02d}/field_{field:03d}"
                record = _read_json(archive, prefix + ".json", plan_hash)
                intent = _read_json(archive, prefix + ".intent.json", plan_hash)
                if any(record[k] != intent[k] for k in ("arm", "ber", "field", "seed", "schedule_sha256", "events")):
                    raise ValueError("field intent and outcome differ")
                if (record["arm"] != arm or record["field"] != field or
                    record["seed"] != 130001 + arm * 100000 + field or
                    record["ber"] != plan["ber"][arm] or
                    record["schedule_sha256"] != _schedule_digest(record["events"]) or
                    set(record["policies"]) != set(POLICIES)):
                    raise ValueError("final field schedule or policy identity changed")
                event_bits = [_event_bit(event) for event in record["events"]]
                if len(set(event_bits)) != len(event_bits) or not set(event_bits) <= set(keys):
                    raise ValueError("invalid or duplicate final fault events")
                for policy in POLICIES:
                    entry = record["policies"][policy]
                    mask = set(selected[policy])
                    applied = [key for key in event_bits if key not in mask]
                    report = entry["report"]
                    if (entry["mask_sha256"] != _mask_digest(selected[policy], width) or
                        report["schedule_hash"] != record["schedule_sha256"] or
                        report["proposed_events"] != len(event_bits) or
                        report["applied_events"] != len(applied) or
                        report["suppressed_events"] != len(event_bits) - len(applied)):
                        raise ValueError("policy mask or event accounting changed")
                    if policy == "full" and entry["risk"] != clean:
                        raise ValueError("full protection does not reproduce clean risk")
                    detail = risk_decomposition(entry["risk"], plan["failure_penalty_nats"])
                    field_rows.append({"arm": arm, "ber": record["ber"], "field": field,
                        "seed": record["seed"], "policy": policy,
                        "proposed": len(event_bits), "applied": len(applied),
                        "applied_top_exponent": sum(bit_parts(key, width)[2] == width-2 for key in applied),
                        **detail})
        table = []
        for arm in (1, 2):
            for policy in POLICIES:
                rows = [row for row in field_rows if row["arm"] == arm and row["policy"] == policy]
                table.append({"arm": arm, "ber": plan["ber"][arm], "policy": policy,
                    "field_states": dict(Counter(row["state"] for row in rows)),
                    "fields_with_applied_top_exponent": sum(row["applied_top_exponent"] > 0 for row in rows),
                    "failed_tokens": sum(row["failed_tokens"] for row in rows),
                    "finite_capped_tokens": sum(row["finite_capped_tokens"] for row in rows),
                    **{key: statistics.mean(row[key] for row in rows) for key in
                       ("saved_mean_nll", "failed_penalty_mean_nll", "finite_cap_mean_nll",
                        "ordinary_finite_mean_nll")}})
                original = next(r for r in analysis["rows"] if r["ber"] == plan["ber"][arm] and r["method"] == policy)
                if not math.isclose(table[-1]["saved_mean_nll"], original["mean_nll"], abs_tol=1e-8):
                    raise ValueError("offline means disagree with finalized analysis")
        scores = selections["exact_isolated"]["signed_scores"]
        score_ties = {}
        for budget in (12, 24):
            ordered = sorted(zip(keys, scores), key=lambda item: (-item[1], bit_parts(item[0], width)[:3]))
            boundary = ordered[budget - 1][1]
            score_ties[str(budget)] = {"boundary_score": boundary,
                "boundary_tie_size": sum(score == boundary for score in scores),
                "boundary_tie_selected": sum(score == boundary for _, score in ordered[:budget])}
        return {"schema_version": "lm-representation-offline-audit/v1",
            "archive": str(path.resolve()), "archive_sha256": archive_hash,
            "plan_sha256": plan_hash, "source_commit": plan["source_commit"],
            "target": plan["target"], "stored_precision": plan["stored_precision"],
            "computation_precision": plan["computation_precision"],
            "candidate_classes": dict(classes), "candidate_bits": keys,
            "selected_masks": selected, "mask_overlap": overlap,
            "normalized_masks": {name: sorted(_normalized(mask, width)) for name, mask in selected.items()},
            "score_ties": score_ties, "max_score_tie_size": max(Counter(scores).values()),
            "field_rows": field_rows, "failure_table": table,
            "original_comparisons": analysis["comparisons"],
            "normalized_field_schedules": [
                {"arm": arm, "field": field, "seed": next(row["seed"] for row in field_rows
                    if row["arm"] == arm and row["field"] == field),
                 "events": sorted(_normalized([_event_bit(event) for event in
                     _read_json(archive, f"run/final/ber_{arm:02d}/field_{field:03d}.json", plan_hash)["events"]], width))}
                for arm in (1, 2) for field in range(32)]}


def audit_family(paths: list[Path]) -> dict:
    if len(paths) != 3:
        raise ValueError("all three frozen LM archives are required")
    audits = [audit_archive(path) for path in paths]
    expected = {"pythia160m_fp32stored_fp32compute",
                "pythia160m_bf16stored_fp32compute", "qwen3_0_6b_bf16"}
    if {item["target"] for item in audits} != expected:
        raise ValueError("missing or duplicate frozen target")
    fp = next(item for item in audits if item["target"] == "pythia160m_fp32stored_fp32compute")
    bf = next(item for item in audits if item["target"] == "pythia160m_bf16stored_fp32compute")
    pythia = {"same_normalized_masks": fp["normalized_masks"] == bf["normalized_masks"],
              "same_normalized_field_schedules": fp["normalized_field_schedules"] == bf["normalized_field_schedules"],
              "same_word_ordinals": [(bit_parts(key, 32)[0], bit_parts(key, 32)[1]) for key in fp["candidate_bits"][::3]] ==
                                    [(bit_parts(key, 16)[0], bit_parts(key, 16)[1]) for key in bf["candidate_bits"][::3]]}
    if not all(pythia.values()):
        raise ValueError("Pythia paired storage-precision identities do not reconcile")
    return {"schema_version": "lm-representation-family-audit/v1",
            "pythia_pairing": pythia, "audits": audits,
            "interpretation": "Offline descriptive associations; no multi-fault bit causality or new held-out exponent-first result."}


def write_derived(result: dict, destination: Path) -> Path:
    """Write a new, write-once audit bundle outside all historical runs."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("derived audit destination already exists")
    destination.mkdir(parents=True)
    source = Path(__file__).resolve()
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source.parents[1], text=True).strip()
    document = {**result, "auditor_source_sha256": sha256_file(source),
                "auditor_source_commit": source_commit}
    payload = {**document, "payload_sha256": digest_object(document)}
    (destination / "audit.json").write_bytes(canonical_json_bytes(payload) + b"\n")
    for name, key in (("field_rows.csv", "field_rows"), ("failure_table.csv", "failure_table")):
        rows = [{"target": audit["target"], **row} for audit in result["audits"] for row in audit[key]]
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
        (destination / name).write_text(stream.getvalue())
    with zipfile.ZipFile(destination / "complete.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ("audit.json", "field_rows.csv", "failure_table.csv"):
            archive.write(destination / name, name)
        archive.write(source, "source/netsurf/transformer_lm_representation_audit.py")
    checks = {name: sha256_file(destination / name) for name in
              ("audit.json", "field_rows.csv", "failure_table.csv", "complete.zip")}
    (destination / "checksums.json").write_bytes(canonical_json_bytes(checks) + b"\n")
    (destination / "FINALIZED").write_text(digest_object(checks) + "\n")
    for path in destination.iterdir():
        path.chmod(0o444)
    return destination


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs=3, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit_family(args.archives)
    write_derived(result, args.output)
    print(json.dumps({"status": "FINALIZED", "targets": [x["target"] for x in result["audits"]],
                      "output": str(args.output), "pythia_pairing": result["pythia_pairing"]}))


if __name__ == "__main__":
    main()
