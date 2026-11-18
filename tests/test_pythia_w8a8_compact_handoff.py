"""Synthetic offline checks for compact field-accounting audit."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from export_pythia_w8a8_scientific_handoff_v1 import METHODS, audit_observations  # noqa: E402


def _fixture():
    bits = 988545024
    budgets = (61784064, 123568128, 185352192, 247136256)
    plan = {"plan_sha256": "plan", "eligible_universe": {"bits": bits, "sha256": "population"},
            "interior_budgets_bits": {str(i): value for i, value in enumerate(budgets)},
            "fields": [{"ber": (arm+1)*1e-9, "lambda": (4,32,256)[arm],
                        "seeds": list(range(1000+arm*100, 1024+arm*100))} for arm in range(3)]}
    masks = [{"method": method, "protected_bits": budget, "mask_sha256": f"{method}:{budget}",
              "policy_id": f"{method}_b{budget}"} for method in METHODS for budget in budgets]
    masks += [{"method": "unprotected", "protected_bits": 0, "mask_sha256": "unprotected",
               "policy_id": "unprotected"},
              {"method": "full_protection", "protected_bits": bits, "mask_sha256": "full_protection",
               "policy_id": "full_protection"}]
    analysis = {"clean": {"metrics": {"mean_nll": 3.5, "perplexity": 33.11545195869231}},
                "field_rows": [{} for _ in range(72)]}
    rows = []
    for arm,spec in enumerate(plan["fields"]):
        for field_index,seed in enumerate(spec["seeds"]):
            for mask in masks:
                full = mask["method"] == "full_protection"
                proposed = (3,31,257)[arm]
                rows.append({"ber": str(spec["ber"]), "field_index": str(field_index),
                             "seed": str(seed), "expected_proposed_faults": str(spec["lambda"]),
                             "plan_sha256": "plan", "method": mask["method"],
                             "protected_bits": str(mask["protected_bits"]),
                             "mask_sha256": mask["mask_sha256"],
                             "proposed_faults": str(proposed),
                             "applied_faults": "0" if full else str(proposed),
                             "suppressed_faults": str(proposed) if full else "0",
                             "failed_tokens": "0", "scored_tokens": "4096",
                             "mean_nll": "3.5" if full else "3.51",
                             "perplexity": str(analysis["clean"]["metrics"]["perplexity"]),
                             "field_sha256": f"field-{arm}-{field_index}",
                             "field_record": f"final/arm_{arm:02d}/field_{field_index:03d}.json"})
    return plan, analysis, rows, masks


def test_complete_paired_field_audit():
    plan, analysis, rows, masks = _fixture()
    enriched, audit = audit_observations(plan, analysis, rows, masks)
    assert audit["independent_fields"] == 72
    assert audit["observations"] == 1872
    assert audit["ranking_methods"] == list(METHODS)
    assert all("failure_fraction" in row for row in enriched)


def test_rejects_changed_proposed_field_and_event_accounting():
    plan, analysis, rows, masks = _fixture()
    rows[1]["field_sha256"] = "different"
    with pytest.raises(ValueError, match="share proposed field"):
        audit_observations(plan, analysis, rows, masks)
    plan, analysis, rows, masks = _fixture()
    rows[1]["suppressed_faults"] = "1"
    with pytest.raises(ValueError, match="fault, failure, or budget"):
        audit_observations(plan, analysis, rows, masks)


def test_rejects_omitted_ranking_policy():
    plan, analysis, rows, masks = _fixture()
    rows.pop()
    with pytest.raises(ValueError, match="72 paired fields or 1872"):
        audit_observations(plan, analysis, rows, masks)
