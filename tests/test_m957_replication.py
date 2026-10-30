from dataclasses import replace
import json
from pathlib import Path
import shutil
import tomllib

import numpy as np
import pytest

from netsurf.conditional_config import load_conditional, ConditionalConfigurationError, PAIRED_SWAP_COMPARISONS
from netsurf.conditional_damage import execute_conditional_damage
from netsurf.conditional_bundle import create_conditional_bundle, verify_conditional_bundle
from netsurf.manifests import sha256_file, digest_object
from netsurf.replication_preflight import (
    MASKS, LEDGER, load_replication_selection, replication_exposure_audit,
    verify_new_seeds, replication_plan, freeze_replication_plan,
    verify_replication_plan, reject_recorded_reuse,
)
from netsurf.replication_statistics import summarize_replication

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "config/examples/m957_kws_frozen_replication_k8_v1.toml"


def draft():
    c = load_conditional(TEMPLATE, check_inputs=False)
    return replace(c, workflow={k: v for k, v in c.workflow.items()
                               if k not in {"replication_plan", "replication_plan_sha256"}})


def local_draft():
    c = draft()
    if not Path(c.workflow["selection_bundle"]).is_dir() or not c.benchmark.checkpoint.is_file():
        pytest.skip("requires original M956 selection evidence and pinned local checkpoint (no inference)")
    return c


def rows(values):
    return [{"comparison": comparison, "logical_ber": .01, "field": i,
             "field_seed": 100 + i, "master_pairing_key": f"pair-{i}", "master_sha256": f"master-{i}",
             "sample_sha256": "fixed-samples", "protection_context_sha256": "context",
             "target_minus_baseline": float(value * (j + 1))}
            for j, comparison in enumerate(PAIRED_SWAP_COMPARISONS) for i, value in enumerate(values)]


def test_whole_field_mean_ci_pvalues_and_tail_influence():
    from scipy.stats import ttest_1samp, t
    x = np.array([-1.] * 7 + [8.])
    result = summarize_replication(rows(x), 8)
    primary = result["comparisons"][0]
    assert result["independent_field_count"] == 8  # not 24 policy observations, steps or inputs
    e = primary["mean_effect"]
    assert e["mean"] == np.mean(x) and e["count"] == 8
    assert primary["median"] == -1 and e["mean"] > 0
    assert primary["favorable_fields"] == 7 and primary["unfavorable_fields"] == 1
    assert np.allclose(e["confidence_interval"], np.mean(x) + np.array([-1, 1]) * t.ppf(.975, 7) * np.std(x, ddof=1) / np.sqrt(8))
    assert primary["p_value_two_sided_unadjusted"] == pytest.approx(ttest_1samp(x, 0).pvalue)
    for extreme in primary["extreme_contributions"]:
        assert extreme["contribution_to_full_mean"] + extreme["complement_contribution_to_full_mean"] == pytest.approx(e["mean"])
    assert primary["worst_adverse_fields"][0]["field"] == 7
    assert primary["all_fields_retained"]
    for secondary in result["comparisons"][1:]:
        assert secondary["p_value_secondary_holm"] >= secondary["p_value_two_sided_unadjusted"]
        assert secondary["secondary_family_mean_ci"]["confidence"] == .975


@pytest.mark.parametrize("alteration", ["duplicate", "missing", "seed", "ber", "samples", "nonfinite"])
def test_invalid_replication_records_rejected(alteration):
    r = rows([-1., 0., 2.])
    if alteration == "duplicate": r.append(dict(r[0]))
    if alteration == "missing": r.pop()
    if alteration == "seed": r[3]["field_seed"] += 1000
    if alteration == "ber": r[3]["logical_ber"] = .1
    if alteration == "samples": r[3]["sample_sha256"] = "different"
    if alteration == "nonfinite": r[3]["target_minus_baseline"] = float("nan")
    with pytest.raises(ValueError): summarize_replication(r, 3)


def test_degenerate_zero_and_nonzero_variance_is_explicit():
    for values, expected in (([0., 0.], 1.), ([1., 1.], None)):
        result = summarize_replication(rows(values), 2)
        assert result["comparisons"][0]["p_value_two_sided_unadjusted"] == expected
        assert result["comparisons"][0]["degenerate_variance"]


def test_proposed_window_and_exposure_rejection():
    c = draft()
    a = replication_exposure_audit(c.samples, "confirmation")
    assert a["requested_original_indices"] == list(range(3194, 3450)) and not a["prior_overlap"]
    for index in (0, 3048, 3060, 3066, 3193):
        # Exposed indices outside the final-role window are still rejected by this gate.
        with pytest.raises(ConditionalConfigurationError, match="exposure overlap"):
            replication_exposure_audit(replace(c.samples, evaluation_offset=index - 3048, evaluation_count=1), "confirmation")
    audit = json.loads(LEDGER.read_text())
    assert any(r["scientific_payload_sha256"] == "d1726e5be00efcb14382dfc2d26fd5921cee6a8e43fc9a590c7ca178ac8cbf93" for r in audit["runs"])


def test_seed_audit_checks_derived_collisions_not_just_base():
    c = draft()
    record = {"contract": {"selection_seeds": [956711], "selection_fields_per_seed": 8,
                          "final_plan": {"paired": {"seed": 956901, "background_count": 64}}}}
    assert len(verify_new_seeds(c, record)["new_derived_seeds"]) == 256
    for seed in (956901, 956711 + 1_000_003, 956901 + 63 * 1_000_003):
        with pytest.raises(ConditionalConfigurationError, match="derived field seeds"):
            verify_new_seeds(replace(c, paired_swap_evaluation=replace(c.paired_swap_evaluation, seed=seed)), record)


def test_configuration_schema_and_costs(tmp_path):
    c = draft()
    assert c.operation_counts()["total_requested_evaluations"] == 3072
    for old, new in (("6.5.0", "6.6.0"), ("replication_only", "final_only"),
                     ("22016", "22017"), ("176128", "176129"), ("confidence = 0.95", "confidence = 0.90")):
        path = tmp_path / "bad.toml"
        path.write_text(TEMPLATE.read_text().replace(old, new))
        with pytest.raises(ConditionalConfigurationError):
            altered = load_conditional(path, check_inputs=False)
            # Confidence is also enforced at the strict source/plan boundary.
            load_replication_selection(altered)


def test_original_source_plan_is_verified_but_new_final_plan_is_independent(tmp_path):
    c = local_draft()
    record = load_replication_selection(c)
    assert record["masks"] == [{"logical_ber": .01, **MASKS}]
    assert record["contract"]["final_plan"]["offset"] == 18
    plan = replication_plan(c)
    assert plan["final_plan"]["offset"] == 146
    assert plan["operation_counts"]["exact_path_requests_before_cache"] == 1536
    assert plan["operation_counts"]["clip_inferences_upper_bound"] == 786432
    assert plan["final_plan"]["procedure"]["primary"] == PAIRED_SWAP_COMPARISONS[0]
    for altered in (
        replace(c, candidates=tuple(reversed(c.candidates))),
        replace(c, protection_context=replace(c.protection_context, mask_key="different")),
        replace(c, benchmark=replace(c.benchmark, inference_backend="another")),
        replace(c, workflow={**c.workflow, "selection_payload_sha256": "0" * 64}),
    ):
        with pytest.raises(ConditionalConfigurationError): load_replication_selection(altered)
    source = Path(c.workflow["selection_bundle"])
    copy = tmp_path / "copied-selection"
    shutil.copytree(source, copy)
    artifact = copy / "artifacts/frozen_selection.json"
    artifact.chmod(0o644)  # Only our isolated synthetic copy.
    forged = json.loads(artifact.read_text())
    forged["masks"][0]["selected"] = [2, 7]
    artifact.write_text(json.dumps(forged))
    with pytest.raises(ConditionalConfigurationError, match="checksum mismatch"):
        load_replication_selection(replace(c, workflow={**c.workflow, "selection_bundle": str(copy)}))


def test_frozen_plan_hash_overwrite_and_changed_config_rejection(tmp_path):
    c = local_draft()
    with pytest.raises(ConditionalConfigurationError, match="freeze and bind"):
        verify_replication_plan(c)
    path = tmp_path / "plan.json"
    info = freeze_replication_plan(c, path)
    bound = replace(c, workflow={**c.workflow, "replication_plan": str(path),
                               "replication_plan_sha256": info["replication_plan_sha256"]})
    assert verify_replication_plan(bound) == replication_plan(c)
    assert sha256_file(path) == info["replication_plan_sha256"]
    with pytest.raises(FileExistsError): freeze_replication_plan(c, path)
    for modified in (
        replace(bound, samples=replace(c.samples, evaluation_count=255)),
        replace(bound, paired_swap_evaluation=replace(c.paired_swap_evaluation, seed=957902)),
        replace(bound, workflow={**bound.workflow, "replication_plan_sha256": "0" * 64}),
    ):
        with pytest.raises(ConditionalConfigurationError, match="plan/hash differs"):
            verify_replication_plan(modified)


def test_new_finalized_exposure_is_rejected(tmp_path):
    c = replace(draft(), output=replace(draft().output, root=tmp_path))
    p = tmp_path / "previous/run"
    p.mkdir(parents=True)
    (p / "FINALIZED").write_text("fixture")
    (p / "manifest.json").write_text(json.dumps({"benchmark": {"adapter": c.benchmark.adapter},
                                                "m956_final_original_indices": [3194]}))
    with pytest.raises(ConditionalConfigurationError, match="exposure overlap"):
        reject_recorded_reuse(c)


def test_exposure_and_unbound_plan_fail_before_model_access():
    class Forbidden:
        def load(self): raise AssertionError("must reject before model or input access")
    c = local_draft()
    for changed in (c, replace(c, samples=replace(c.samples, evaluation_offset=18))):
        with pytest.raises(ConditionalConfigurationError):
            execute_conditional_damage(changed, adapter=Forbidden())


def test_machine_schema():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((ROOT / "netsurf/schemas/conditional/v1_5.schema.json").read_text())
    doc = tomllib.loads(TEMPLATE.read_text())
    jsonschema.validate(doc, schema)
    doc["workflow"]["mode"] = "final_only"
    with pytest.raises(jsonschema.ValidationError): jsonschema.validate(doc, schema)


def test_pinned_kws_replication_no_selection_reserved_inputs_or_state_drift(tmp_path, monkeypatch):
    tf = pytest.importorskip("tensorflow")
    if tf.__version__ != "2.14.1": pytest.skip("requires pinned TF 2.14.1 BUILTIN_REF")
    from netsurf.adapters.registry import create_adapter
    text = TEMPLATE.read_text()
    text = "\n".join(line for line in text.splitlines() if not line.startswith("replication_plan")) + "\n"
    text = text.replace('name = "m957_kws_frozen_replication_k8_v1"', 'name = "m957_engineering_replication"')
    text = text.replace('purpose = "confirmation"', 'purpose = "engineering"')
    text = text.replace('evaluation_offset = 146', 'evaluation_offset = 12').replace('evaluation_count = 256', 'evaluation_count = 2')
    text = text.replace('background_count = 256', 'background_count = 3').replace('seed = 957901', 'seed = 957211')
    text = text.replace('root = ""', f'root = "{tmp_path / "runs"}"')
    path = tmp_path / "engineering.toml"
    path.write_text(text)
    config = load_conditional(path)
    info = freeze_replication_plan(config, tmp_path / "engineering-plan.json")
    path.write_text(text + f'replication_plan = "{info["replication_plan"]}"\nreplication_plan_sha256 = "{info["replication_plan_sha256"]}"\n')
    config = load_conditional(path)
    adapter = create_adapter(config, None)
    partition = adapter.sample_partition
    def guarded(role, count, *, offset=0):
        if role == "evaluation": assert count <= 16 and offset == 0
        else:
            assert role == "conditional_final_evaluation" and offset == 12 and count == 2
        return partition(role, count, offset=offset)
    adapter.sample_partition = guarded
    def forbidden(*a, **k): raise AssertionError("replication cannot select or enumerate")
    monkeypatch.setattr("netsurf.conditional_damage.subcube_states", forbidden)
    monkeypatch.setattr("netsurf.selection_preflight.selection_summary", forbidden)
    outcome = execute_conditional_damage(config, adapter=adapter)
    repeat = execute_conditional_damage(config, adapter=adapter)
    assert outcome.scientific_payload_sha256 == repeat.scientific_payload_sha256
    assert outcome.scientific_payload["checks"]["state_restored"]
    assert outcome.scientific_payload["frozen_selection"]["masks"] == [{"logical_ber": .01, **MASKS}]
    assert set(outcome.partitions) == {"held_out_evaluation"}
    assert outcome.partitions["held_out_evaluation"].indices == (3060, 3061)
    assert not outcome.oracle_response_records and not outcome.measurements
    assert len(outcome.paired_swap_field_records) == 9
    for field in range(3):
        effects = {r["comparison"]: r["target_minus_baseline"] for r in outcome.paired_swap_field_records if r["field"] == field}
        assert effects[PAIRED_SWAP_COMPARISONS[0]] == pytest.approx(
            effects[PAIRED_SWAP_COMPARISONS[1]] + effects[PAIRED_SWAP_COMPARISONS[2]], abs=1e-15)
    inputs = outcome.scientific_payload["paired_input_diagnostics"]
    assert len(inputs["records"]) == 9 and len(inputs["sample_identities"]) == 2
    assert all(abs(r["mean_roundoff_residual"]) < 1e-12 for r in inputs["records"])
    assert not hasattr(adapter, "_input_loss_capture")
    assert all(r["mean_effect"]["count"] == 3 for r in outcome.scientific_payload["replication_analysis"]["comparisons"])
    assert outcome.protection_context_record["global_cardinality"] == 22016
    for step in outcome.paired_swap_step_records:
        assert step["removed_proposed_events"] == step["removed_applied_events"] + step["removed_suppressed_events"]
    root = create_conditional_bundle(outcome, repository_root=ROOT)
    verify_conditional_bundle(root)
    assert "Holm" in (root / "report.md").read_text()
    assert json.loads((root / "artifacts/replication_plan.json").read_text()) == verify_replication_plan(config)
    # Diagnostic capture must not change any authoritative scalar/path measurement.
    control = replace(config, name="m957_engineering_no_capture", workflow={
        **{k: v for k, v in config.workflow.items() if not k.startswith("replication_plan")},
        "record_per_input": False})
    binding = freeze_replication_plan(control, tmp_path / "control-plan.json")
    control = replace(control, workflow={**control.workflow,
        "replication_plan": binding["replication_plan"], "replication_plan_sha256": binding["replication_plan_sha256"]})
    without = execute_conditional_damage(control, adapter=adapter)
    assert without.paired_swap_field_records == outcome.paired_swap_field_records
    assert without.paired_swap_step_records == outcome.paired_swap_step_records
    print(f"bounded engineering bundle: {root}; payload: {outcome.scientific_payload_sha256}")
