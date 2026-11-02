"""Hash-bound campaign plans and all-selection-before-final gate."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re

from .manifests import canonical_json_bytes, digest_object, sha256_file


class PlanBlocked(ValueError):
    pass


def validate_frozen_plan(document: dict) -> str:
    if document.get("schema_version") != "transformer-campaign-plan/v1" or document.get("status") != "FROZEN":
        raise PlanBlocked("only a frozen transformer campaign plan may run")
    for key in ("target", "source_commit", "asset_manifest_sha256", "asset_record_sha256",
                "eligible_population_sha256", "partitions_sha256"):
        value = document.get(key)
        valid_hash = isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", value)
        if not isinstance(value, str) or (key != "target" and
            (not valid_hash or (key != "source_commit" and len(value) != 64))):
            raise PlanBlocked(f"missing or invalid binding: {key}")
    n = document.get("eligible_bits")
    budgets = document.get("budgets")
    bers = document.get("ber")
    if not isinstance(n, int) or n < 1 or not isinstance(budgets, list) or not budgets or any(not isinstance(x, int) for x in budgets) or budgets != sorted(set(budgets)) or budgets[0] != 0 or budgets[-1] != n:
        raise PlanBlocked("budget grid must be unique, sorted, and include zero/full")
    if not isinstance(bers, list) or not bers or any(not isinstance(p, (float, int)) or not math.isfinite(p) or not 0 <= p <= 1 for p in bers) or bers != sorted(set(bers)) or bers[0] != 0:
        raise PlanBlocked("BER grid must be unique, sorted, and include clean")
    methods = document.get("methods")
    if not isinstance(methods, list) or not methods or len(set(methods)) != len(methods):
        raise PlanBlocked("baseline methods must be prespecified")
    selection = document.get("selection")
    final = document.get("final")
    if not isinstance(selection, dict) or not isinstance(final, dict):
        raise PlanBlocked("selection and final schedules required")
    for stage, value in (("selection", selection), ("final", final)):
        if not isinstance(value.get("seed"), int) or not isinstance(value.get("fields"), int) or value["fields"] < 1 or not isinstance(value.get("sample_count"), int) or value["sample_count"] < 1:
            raise PlanBlocked(f"invalid {stage} count or seed")
    if selection["seed"] == final["seed"]:
        raise PlanBlocked("selection and final seed collision")
    if document.get("failure_penalty_nats") != 30.0:
        raise PlanBlocked("failure risk differs from approved version")
    analysis = document.get("analysis")
    if not isinstance(analysis, dict) or analysis.get("independent_unit") != "fault_field" or not analysis.get("multiple_comparison_procedure"):
        raise PlanBlocked("statistical unit or multiplicity procedure missing")
    digest = digest_object({key: value for key, value in document.items() if key != "plan_sha256"})
    if document.get("plan_sha256") != digest:
        raise PlanBlocked("scientific plan SHA-256 mismatch")
    return digest


def seal_selection(plan_path: Path, artifacts: dict[str, Path], output: Path) -> Path:
    plan = json.loads(Path(plan_path).read_text())
    plan_digest = validate_frozen_plan(plan)
    if set(artifacts) != set(plan["methods"]):
        raise PlanBlocked("every prespecified method must have one selection artifact")
    files = {}
    for method, path in artifacts.items():
        path = Path(path).resolve()
        if not path.is_file():
            raise PlanBlocked(f"missing selection artifact for {method}")
        files[method] = {"path": str(path), "sha256": sha256_file(path)}
    payload = {"schema_version": "transformer-selection-seal/v1", "plan_sha256": plan_digest,
               "artifacts": files}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_json_bytes(payload) + b"\n")
        stream.flush(); os.fsync(stream.fileno())
    return output


def require_final_gate(plan_path: Path, seal_path: Path) -> dict:
    plan = json.loads(Path(plan_path).read_text())
    digest = validate_frozen_plan(plan)
    seal = json.loads(Path(seal_path).read_text())
    if seal.get("schema_version") != "transformer-selection-seal/v1" or seal.get("plan_sha256") != digest or set(seal.get("artifacts", {})) != set(plan["methods"]):
        raise PlanBlocked("all selection artifacts are not sealed under this plan")
    for method, item in seal["artifacts"].items():
        if sha256_file(Path(item["path"])) != item["sha256"]:
            raise PlanBlocked(f"selection artifact changed: {method}")
    return plan
