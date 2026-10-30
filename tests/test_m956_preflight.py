from dataclasses import replace
import json
from pathlib import Path
import tomllib

import numpy as np
import pytest

from netsurf.conditional_bundle import create_conditional_bundle, verify_conditional_bundle
from netsurf.conditional_config import load_conditional, ConditionalConfigurationError, PAIRED_SWAP_COMPARISONS, PairedSwapEvaluationConfiguration
from netsurf.conditional_damage import execute_conditional_damage
from netsurf.manifests import sha256_file
from netsurf.samples import SamplePartition, sample_identity_digest
from netsurf.selection_preflight import exposure_audit, load_frozen_selection, selection_summary, reject_local_final_reuse
from netsurf.restricted_subcube import subcube_states, analyze_restricted_response_tables
from tests.test_m95_conditional_damage import _FakeExactAdapter, _oracle_configuration, _context_source

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "config/examples/m956_kws_s5_selection_k8.toml"
FINAL = ROOT / "config/examples/m956_kws_s5_final_k8.toml"


class GuardedAdapter(_FakeExactAdapter):
    def __init__(self, allowed):
        super().__init__()
        self.allowed = allowed
        self.roles = []

    def sample_partition(self, role, count, *, offset=0):
        assert role in self.allowed, f"forbidden sample access: {role}"
        self.roles.append(role)
        base = {"conditional_model_selection": 2024, "conditional_final_evaluation": 3048}[role]
        indices = tuple(range(base + offset, base + offset + count))
        labels = np.zeros(count, dtype=np.int64)
        split = "speech_commands_tfds_test_interleaved_blocks16"
        return np.zeros((count, 1), dtype=np.int8), labels, SamplePartition(
            role, split, indices, sample_identity_digest(split, indices, labels), "synthetic")

    def same_distribution_window_diagnostics(self):
        raise AssertionError("must not scan all final labels")


def configuration(tmp_path):
    base = _oracle_configuration(tmp_path, .25)
    return replace(base, schema_version="6.4.0", scientific_protocol="frozen-selection-protection/v5",
        protection_context=_context_source(tmp_path, base),
        restricted_oracle=replace(base.restricted_oracle, calibration_replicate_seeds=(301, 302)),
        paired_swap_evaluation=PairedSwapEvaluationConfiguration(False, 3, 505, PAIRED_SWAP_COMPARISONS),
        workflow={"mode": "selection_only", "purpose": "engineering",
                  "candidate_provenance": "exploratory_frozen_m953_pool"})


def finalize_selection(config):
    outcome = execute_conditional_damage(config, adapter=GuardedAdapter({"conditional_model_selection"}))
    root = create_conditional_bundle(outcome, repository_root=ROOT)
    return root, outcome, replace(config,
        paired_swap_evaluation=replace(config.paired_swap_evaluation, enabled=True),
        workflow={**config.workflow, "mode": "final_only", "selection_bundle": str(root),
                  "selection_payload_sha256": outcome.scientific_payload_sha256,
                  "selection_artifact_sha256": sha256_file(root / "artifacts/frozen_selection.json")})


def test_selection_only_no_final_access_and_complete_stability(tmp_path, monkeypatch):
    config = configuration(tmp_path)
    import netsurf.conditional_damage as engine
    master = engine._master_backgrounds
    def guarded(universe, p, count, seed):
        assert seed in config.restricted_oracle.selection_seeds
        return master(universe, p, count, seed)
    monkeypatch.setattr(engine, "_master_backgrounds", guarded)
    root, outcome, _ = finalize_selection(config)
    repeat = execute_conditional_damage(config, adapter=GuardedAdapter({"conditional_model_selection"}))
    assert repeat.scientific_payload_sha256 == outcome.scientific_payload_sha256
    assert set(outcome.partitions) == {"model_selection"}
    assert not outcome.paired_swap_field_records
    assert not outcome.measurements
    assert len(outcome.oracle_response_records) == 3 + 4 * 4
    assert {r["phase"] for r in outcome.oracle_response_records} == {"model_selection", "model_selection_isolated"}
    row = outcome.oracle_summaries[0]
    assert len(row["eligible_masks"]) == 2
    assert all(len(m["field_risks"]) == 4 for m in row["eligible_masks"])
    assert len(row["model_selection_stability"]["replicates"]) == 2
    assert row["model_selection_stability"]["field_count"] == 4
    assert not outcome.scientific_payload["checks"]["final_samples_accessed"]
    verify_conditional_bundle(root)
    assert "No final-evaluation samples were loaded" in (root / "report.md").read_text()


def test_final_only_loads_masks_and_never_reselects(tmp_path, monkeypatch):
    _, selection, config = finalize_selection(configuration(tmp_path))
    import netsurf.conditional_damage as engine
    def forbidden(*a, **k):
        raise AssertionError("final execution must not select or enumerate response tables")
    monkeypatch.setattr(engine, "selection_summary", forbidden, raising=False)
    monkeypatch.setattr("netsurf.selection_preflight.selection_summary", forbidden)
    monkeypatch.setattr(engine, "analyze_restricted_response_tables", forbidden)
    outcome = execute_conditional_damage(config, adapter=GuardedAdapter({"conditional_final_evaluation"}))
    repeat = execute_conditional_damage(config, adapter=GuardedAdapter({"conditional_final_evaluation"}))
    assert outcome.scientific_payload_sha256 == repeat.scientific_payload_sha256
    assert set(outcome.partitions) == {"held_out_evaluation"}
    assert not outcome.oracle_response_records
    assert outcome.scientific_payload["frozen_selection"] == selection.scientific_payload["frozen_selection"]
    assert len(outcome.paired_swap_field_records) == 9
    assert all(r["target_minus_baseline"]["count"] == 3 for r in outcome.paired_swap_summaries)
    assert all("familywise_target_minus_baseline" in r for r in outcome.paired_swap_summaries)
    root = create_conditional_bundle(outcome, repository_root=ROOT)
    verify_conditional_bundle(root)
    assert "selection was not rerun" in (root / "report.md").read_text()


def test_frozen_artifact_rejects_changed_plan_masks_and_corruption(tmp_path):
    root, _, config = finalize_selection(configuration(tmp_path))
    load_frozen_selection(config)
    for modified in (
        replace(config, candidates=tuple(reversed(config.candidates))),
        replace(config, samples=replace(config.samples, evaluation_count=3)),
        replace(config, paired_swap_evaluation=replace(config.paired_swap_evaluation, seed=999)),
        replace(config, execution=replace(config.execution, confidence=.9)),
    ):
        with pytest.raises(ConditionalConfigurationError, match="contract differs"):
            load_frozen_selection(modified)
    artifact = root / "artifacts/frozen_selection.json"
    artifact.chmod(0o644)  # Synthetic temporary test bundle only.
    artifact.write_text("{}")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_frozen_selection(config)


def test_exact_exposure_ledger_and_overlap_rejection():
    for offset in (0, 5, 6, 11, 12, 17):
        with pytest.raises(ValueError, match="exposure overlap"):
            exposure_audit(offset, 1, "confirmation")
    audit = exposure_audit(18, 128, "confirmation")
    assert audit["requested_original_indices"] == list(range(3066, 3194))
    assert not audit["prior_overlap"]
    assert exposure_audit(12, 6, "engineering")["prior_overlap"] == list(range(3060, 3066))


def test_new_finalized_exposures_rejected_without_reading_measurements(tmp_path):
    config = configuration(tmp_path)
    config = replace(config, samples=replace(config.samples, evaluation_offset=18),
                     workflow={**config.workflow, "purpose": "confirmation"})
    root = config.output.root / "previous" / "run"
    root.mkdir(parents=True)
    (root / "FINALIZED").write_text("synthetic")
    (root / "manifest.json").write_text(json.dumps({"m956_final_original_indices": [3066]}))
    with pytest.raises(ConditionalConfigurationError, match="finalized bundle"):
        reject_local_final_reuse(config)


def test_exposure_rejected_before_adapter_is_loaded(tmp_path):
    config = configuration(tmp_path)
    config = replace(config, workflow={**config.workflow, "purpose": "confirmation"})
    class NoLoad:
        def load(self):
            raise AssertionError("exposure gate must run before adapter initialization")
    with pytest.raises(ConditionalConfigurationError, match="exposure overlap"):
        execute_conditional_damage(config, adapter=NoLoad())


def test_templates_counts_and_unbound_final_rejection(tmp_path):
    config = load_conditional(PILOT, check_inputs=False)
    assert config.operation_counts()["total_requested_evaluations"] == 8201
    assert config.operation_counts()["model_selection_outside_fields"] == 32
    with pytest.raises(ConditionalConfigurationError, match="SHA-256"):
        load_conditional(FINAL, check_inputs=False)
    path = tmp_path / "bad.toml"
    path.write_text(PILOT.read_text().replace("evaluation_offset = 18", "evaluation_offset = 12"))
    with pytest.raises(ConditionalConfigurationError, match="exposure overlap"):
        load_conditional(path, check_inputs=False)
    # A different base seed can collide with another seed's second field.
    path.write_text(PILOT.read_text().replace("956712", "1956714"))
    with pytest.raises(ConditionalConfigurationError, match="field seeds"):
        load_conditional(path, check_inputs=False)


@pytest.mark.parametrize("kind", ["additive", "beneficial", "higher_order"])
def test_selection_summary_matches_existing_joint_selector(kind):
    candidates = _FakeExactAdapter().bit_identities()[:4]
    def risk(s, r):
        return (sum((i + 1) * (1 if kind != "beneficial" else -1) for i in s)
                + (10 + r) * (kind == "higher_order" and all(i in s for i in (0, 1, 3))))
    tables = [{s: risk(s, r) for s in subcube_states(4)} for r in range(4)]
    isolated = {s: risk(s, 0) for s in [(), (0,), (1,), (2,), (3,)]}
    new = selection_summary(tables, isolated, candidates, 2, .1, (0, 1), (101, 102), 2)
    old = analyze_restricted_response_tables(tables, tables, isolated,
        candidate_count=4, budget=2, probability=.1, original_mask=(0, 1))
    assert tuple(new["selected_mask_indices"]) == old["selected_mask"]
    assert tuple(new["isolated_mask_indices"]) == old["isolated_mask"]
    for row in new["eligible_masks"]:
        assert row["mean_risk"] == old["calibration_mean_risks"][tuple(row["indices"])]


def test_machine_schema_and_python_validator_agree():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((ROOT / "netsurf/schemas/conditional/v1_4.schema.json").read_text())
    doc = tomllib.loads(PILOT.read_text())
    jsonschema.validate(doc, schema)
    doc["paired_swap_evaluation"]["enabled"] = True
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, schema)


def test_pinned_kws_bounded_selection_and_frozen_final(tmp_path):
    tf = pytest.importorskip("tensorflow")
    if tf.__version__ != "2.14.1":
        pytest.skip("requires pinned MLPerf TF2.14.1")
    from netsurf.adapters.registry import create_adapter
    config = load_conditional(PILOT)
    # Two candidate bits, one originally protected; never consume reserved confirmation data.
    config = replace(config, name="m956_kws_engineering",
        candidates=(config.candidates[0], config.candidates[6]),
        samples=replace(config.samples, model_selection_count=2, evaluation_offset=12, evaluation_count=2),
        restricted_oracle=replace(config.restricted_oracle, budget=1,
            calibration_background_count=1, calibration_replicate_seeds=(956101, 956102)),
        paired_swap_evaluation=replace(config.paired_swap_evaluation, background_count=2, seed=956201),
        output=replace(config.output, root=tmp_path / "runs"),
        workflow={**config.workflow, "purpose": "engineering"})
    adapter = create_adapter(config, None)
    partition = adapter.sample_partition
    allowed = {"conditional_model_selection"}
    def guarded(role, *args, **kwargs):
        if role == "evaluation":  # Mandatory adapter clean-conformance gate, NOT the final window.
            result = partition(role, *args, **kwargs)
            assert all(i < 1000 for i in result[2].indices)
            return result
        assert role in allowed
        return partition(role, *args, **kwargs)
    adapter.sample_partition = guarded
    selected = execute_conditional_damage(config, adapter=adapter)
    assert len(selected.oracle_response_records) == 11
    root = create_conditional_bundle(selected, repository_root=ROOT)
    final = replace(config, paired_swap_evaluation=replace(config.paired_swap_evaluation, enabled=True),
        workflow={**config.workflow, "mode": "final_only", "selection_bundle": str(root),
                  "selection_payload_sha256": selected.scientific_payload_sha256,
                  "selection_artifact_sha256": sha256_file(root / "artifacts/frozen_selection.json")})
    allowed.clear(); allowed.add("conditional_final_evaluation")
    outcome = execute_conditional_damage(final, adapter=adapter)
    assert outcome.scientific_payload["checks"]["state_restored"]
    assert outcome.scientific_payload["frozen_selection"] == selected.scientific_payload["frozen_selection"]
    assert len(outcome.paired_swap_field_records) == 6
    verify_conditional_bundle(create_conditional_bundle(outcome, repository_root=ROOT))
