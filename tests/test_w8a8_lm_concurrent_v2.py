"""LM concurrency revisions preserve every frozen scientific field."""

import json
from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import freeze_w8a8_lm_concurrent_v2 as frozen  # noqa: E402
import freeze_w8a8_lm_concurrent_v3 as expanded  # noqa: E402
import report_lm_w8a8_v3 as reporting  # noqa: E402


@pytest.mark.parametrize("model", ("pythia", "qwen"))
def test_lm_execution_plan_changes_only_scheduling_and_source_bindings(model):
    v1_dir, _ = frozen.locations(model)
    original = json.loads((v1_dir / "plan.json").read_text())
    derived, audit = frozen.derive(model, original)
    assert frozen.scientific_payload(derived) == frozen.scientific_payload(original)
    assert derived["fields"] == original["fields"]
    assert derived["selection"] == original["selection"]
    assert derived["final"] == original["final"]
    assert derived["eligible_universe"] == original["eligible_universe"]
    assert derived["methods"] == original["methods"]
    assert derived["interior_budgets_bits"] == original["interior_budgets_bits"]
    assert derived["source_files_sha256"].items() >= original["source_files_sha256"].items()
    assert audit["scientific_fields_equal"]
    assert all(x == "plan_sha256" or x.startswith("resource_envelope.") or
               x.startswith("source_files_sha256.") for x in audit["changed_paths"])
    derived["fields"][0]["seeds"][0] += 1
    assert frozen.scientific_payload(derived) != frozen.scientific_payload(original)


def test_lm_admission_preserves_original_resource_caps():
    assert frozen.admission("pythia")["max_new_process_rss_gib"] == 12
    assert frozen.admission("qwen")["max_new_process_rss_gib"] == 64
    assert frozen.admission("pythia")["cpu_threads"] == 4
    assert frozen.admission("qwen")["min_disk_free_gib"] == 40


@pytest.mark.parametrize("model", ("pythia", "qwen"))
def test_expanded_m12_admission_preserves_science(model):
    v1_dir, _ = expanded.locations(model)
    original = json.loads((v1_dir / "plan.json").read_text())
    derived, audit = expanded.derive(model, original)
    assert expanded.scientific_payload(derived) == expanded.scientific_payload(original)
    assert derived["source_files_sha256"].items() >= original["source_files_sha256"].items()
    assert audit["scientific_fields_equal"]
    assert expanded.admission(model)["max_other_major_model_cpu_cores"] == 4
    assert expanded.admission(model)["max_other_major_model_rss_gib"] == 16


def test_lm_condition_summary_keeps_field_as_unit_and_failure_fraction():
    plan = {"target": "pythia160m_int8", "plan_sha256": "a"*64,
            "eligible_universe": {"bits": 800}, "fields_per_ber": 24}
    rows = []
    for arm, ber in enumerate((.001, .002, .003)):
        for field in range(24):
            for method in ("unprotected", "full_protection", "random", "msb_first",
                           "numerical_bit_jump", "global_first_order",
                           "per_example_absolute", "per_example_positive"):
                budgets = (0,) if method == "unprotected" else (800,) if method == "full_protection" else \
                          (50, 100, 150, 200)
                for budget in budgets:
                    rows.append({"ber": ber, "method": method, "protected_bits": budget,
                                 "field_index": field, "mean_nll": 1. + arm*.1 + field*.001,
                                 "expected_proposed_faults": 4*(8**arm),
                                 "expected_applied_faults": 4*(8**arm)*(1-budget/800),
                                 "failed_tokens": int(field == 0), "scored_tokens": 10,
                                 "proposed_faults": 4, "applied_faults": 4,
                                 "suppressed_faults": 0})
    grouped = reporting.conditions(plan, rows, 1.)
    assert len(grouped) == 78
    assert all(item["independent_fields"] == 24 for item in grouped)
    assert all(item["failed_fraction"] == 1/240 for item in grouped)
    assert grouped[0]["penalized_perplexity"] > 1
    assert "per example positive" in reporting.reference._svg(grouped, plan)
