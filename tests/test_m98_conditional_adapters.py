"""Bounded capability, exposure, task-risk and unchanged-selector checks."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from netsurf.conditional_benchmarks import conditional_capability, check_exposure, _indices
from netsurf.conditional_config import load_conditional
from netsurf.conditional_damage import execute_conditional_damage, exact_bernoulli_risk
from netsurf.conditional_bundle import create_conditional_bundle, verify_conditional_bundle
from netsurf.manifests import sha256_file, digest_object
from netsurf.samples import SamplePartition, sample_identity_digest, require_disjoint
from netsurf.restricted_subcube import subcube_states, integrate_response_table
from netsurf.paired_swap import paired_swap_field_contrast
from tests.test_m956_preflight import configuration, GuardedAdapter

ROOT = Path(__file__).resolve().parents[1]


def test_large_exact_event_csv_is_not_truncated(tmp_path):
    import csv
    from netsurf.cross_pool_analysis import read_field_records
    path = tmp_path / "fields.csv"
    events = json.dumps([f"tensor/kernel[{i}]#b7" for i in range(8000)])
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["master_events"])
        writer.writeheader(); writer.writerow({"master_events": events})
    previous = csv.field_size_limit()
    assert read_field_records(path)[0]["master_events"] == events
    assert csv.field_size_limit() == previous


def test_campaign_cli_prints_machine_readable_result(monkeypatch, capsys, tmp_path):
    from netsurf.__main__ import main
    monkeypatch.setattr("netsurf.cross_pool.validate_campaign", lambda _: {"verified": True})
    assert main(["campaign", "validate", str(tmp_path / "manifest.json")]) == 0
    assert json.loads(capsys.readouterr().out) == {"verified": True}


def test_capability_windows_and_alias_exposure():
    cap = conditional_capability("resnet8_int8")
    assert cap.indices("conditional_model_selection", 24, 32) == tuple(range(1024, 1056))
    assert cap.indices("conditional_final_evaluation", 225, 128) == tuple(range(3225, 3353))
    with pytest.raises(ValueError, match="exceeds"):
        cap.indices("conditional_model_selection", 1999, 2)
    # M99 replaces the M98 blocker with an explicitly nonadditive capability.
    assert conditional_capability("toycar_int8").risk == "one_minus_file_roc_auc"
    assert _indices({"x": [{"source_split": "cifar10_test_mlperf_tiny_ic01_official_200", "indices": [3225]}]}, cap.source_split) == {3225}
    audit = {"exposed_test_indices": [3225], "limitations": "test"}
    with pytest.raises(ValueError, match="exposure overlap"):
        check_exposure(cap, audit, 225, 128, "confirmation")
    assert check_exposure(cap, audit, 225, 128, "engineering")["prior_overlap"] == [3225]


def test_exposure_ledger_verifies_payload_and_same_campaign_identity(tmp_path):
    from netsurf.conditional_benchmarks import audit_exposure
    from netsurf.benchmarks.capabilities import capability_record
    cap = conditional_capability("resnet8_int8")
    payload = {"partition": {"source_split": "cifar10_test_mlperf_tiny_ic01_official_200", "indices": [3225]},
               "campaign_binding": {"campaign_sha256": "a" * 64}}
    path = tmp_path / "scientific_payload.json"
    path.write_text(json.dumps(payload))
    digest = digest_object(payload)
    (tmp_path / "manifest.json").write_text(json.dumps({
        "benchmark": {"checkpoint_sha256": capability_record(cap.target)["checkpoint"]["files"][0]["sha256"]},
        "scientific_payload_sha256": digest}))
    (tmp_path / "checksums.json").write_text(json.dumps({"files": {"scientific_payload.json": sha256_file(path)}}))
    (tmp_path / "FINALIZED").write_text(f"scientific_payload_sha256={digest}\n")
    assert audit_exposure(cap, roots=[tmp_path])["exposed_test_indices"] == [3225]
    assert audit_exposure(cap, roots=[tmp_path], allowed_campaign="a" * 64)["exposed_test_indices"] == []
    assert audit_exposure(cap, roots=[tmp_path], allowed_campaign="b" * 64)["exposed_test_indices"] == [3225]
    path.write_text("{}")
    with pytest.raises(ValueError, match="corrupt exposure"):
        audit_exposure(cap, roots=[tmp_path])


class ResNetFixture(GuardedAdapter):
    def sample_partition(self, role, count, *, offset=0):
        assert role in self.allowed
        indices = conditional_capability("resnet8_int8").indices(role, offset, count)
        y = np.zeros(count, dtype=np.int64)
        return np.zeros((count, 1), dtype=np.int8), y, SamplePartition(
            role, "cifar10_test", indices, sample_identity_digest("cifar10_test", indices, y), "synthetic")


def modern_config(tmp_path, monkeypatch):
    import netsurf.conditional_benchmarks as module
    monkeypatch.setattr(module, "audit_exposure", lambda *a, **k: {"exposed_test_indices": [], "limitations": "synthetic"})
    base = configuration(tmp_path)
    return replace(base, schema_version="6.7.0", scientific_protocol="capability-conditional-protection/v8",
        benchmark=replace(base.benchmark, adapter="mlperf_tiny_ic_int8_v1", id="mlperf_tiny_ic_resnet8_int8_v1", split="test"),
        workflow={"mode": "selection_only", "purpose": "engineering", "candidate_provenance": "prespecified_explicit_engineering/v1"})


def test_resnet_selection_final_contract_and_immutable_bundle(tmp_path, monkeypatch):
    cfg = modern_config(tmp_path, monkeypatch)
    selection = execute_conditional_damage(cfg, adapter=ResNetFixture({"conditional_model_selection"}))
    assert selection.partitions["model_selection"].source_split == "cifar10_test"
    assert not selection.scientific_payload["checks"]["final_samples_accessed"]
    assert not selection.paired_swap_field_records
    repeat = execute_conditional_damage(cfg, adapter=ResNetFixture({"conditional_model_selection"}))
    assert repeat.scientific_payload_sha256 == selection.scientific_payload_sha256
    root = create_conditional_bundle(selection, repository_root=ROOT)
    verify_conditional_bundle(root)
    final = replace(cfg, paired_swap_evaluation=replace(cfg.paired_swap_evaluation, enabled=True),
        workflow={**cfg.workflow, "mode": "final_only", "selection_bundle": str(root),
                  "selection_payload_sha256": selection.scientific_payload_sha256,
                  "selection_artifact_sha256": sha256_file(root / "artifacts/frozen_selection.json")})
    before = sha256_file(root / "scientific_payload.json")
    outcome = execute_conditional_damage(final, adapter=ResNetFixture({"conditional_final_evaluation"}))
    assert outcome.scientific_payload["frozen_selection"] == selection.scientific_payload["frozen_selection"]
    assert outcome.scientific_payload["checks"]["state_restored"]
    assert outcome.scientific_payload["checks"]["global_cardinality_preserved"]
    assert len(outcome.paired_swap_field_records) == 9
    assert sha256_file(root / "scientific_payload.json") == before
    require_disjoint([*selection.partitions.values(), *outcome.partitions.values()])
    from netsurf.selection_preflight import load_frozen_selection
    with pytest.raises(ValueError, match="contract differs"):
        load_frozen_selection(replace(final, candidates=tuple(reversed(final.candidates))))
    artifact = root / "artifacts/frozen_selection.json"
    artifact.chmod(0o644)  # Deliberately corrupt only this synthetic temporary fixture.
    artifact.write_text("{}")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_frozen_selection(final)


def test_schema67_rejects_wrong_task_precision_backend_and_roles(tmp_path, monkeypatch):
    import tomllib
    from netsurf.cross_pool import _toml
    doc = tomllib.loads((ROOT / "config/examples/m98_resnet8_selection_template.toml").read_text())
    monkeypatch.setattr("netsurf.conditional_benchmarks.configuration_exposure", lambda *_: {})
    for section, key, value in [("benchmark", "split", "official_subset_200"),
                                ("risk", "metric", "reconstruction_mse"),
                                ("benchmark", "quantization", "q<6,2,1>"),
                                ("benchmark", "inference_backend", "optimized"),
                                ("samples", "model_selection_offset", 2000)]:
        bad = json.loads(json.dumps(doc)); bad[section][key] = value
        path = tmp_path / f"{section}-{key}.toml"; path.write_text(_toml(bad))
        with pytest.raises(ValueError):
            load_conditional(path, check_inputs=False)
    good = tmp_path / "valid.toml"; good.write_text(_toml(doc))
    cfg = load_conditional(good, check_inputs=False)
    assert cfg.schema_version == "6.7.0"
    assert cfg.operation_counts()["total_requested_evaluations"] > 0


def test_resnet_joint_conditional_mode_preserves_pairing(tmp_path, monkeypatch):
    cfg = modern_config(tmp_path, monkeypatch)
    cfg = replace(cfg, workflow={}, paired_swap_evaluation=replace(cfg.paired_swap_evaluation, enabled=True))
    a = ResNetFixture({"conditional_calibration", "conditional_model_selection", "conditional_final_evaluation"})
    a.same_distribution_window_diagnostics = lambda: {}
    outcome = execute_conditional_damage(cfg, adapter=a)
    assert set(outcome.partitions) == {"calibration", "model_selection", "held_out_evaluation"}
    require_disjoint(tuple(outcome.partitions.values()))
    assert outcome.measurements and outcome.oracle_response_records and outcome.paired_swap_field_records
    for r in outcome.measurements:
        assert r["background_proposed_events"] == r["background_applied_events"] + r["background_suppressed_events"]
        assert r["candidate_proposed_events"] == r["candidate_applied_events"] + r["candidate_suppressed_events"]
    verify_conditional_bundle(create_conditional_bundle(outcome, repository_root=ROOT))


def test_toycar_auc_task_risk_is_not_window_mse_and_retains_all_orders():
    from netsurf.adapters.mlperf_tiny_multibench import MlperfTinyToyCarAdapter
    from netsurf.identities import BitIdentity
    bits = tuple(BitIdentity("kernel", 0, b) for b in range(4))
    labels = np.array([0, 0, 1, 1])
    def risk(state):
        s = set(state)
        scores = np.array([0.2, 0.5, 0.4, 0.8])
        # A beneficial fault, an adverse single fault, and a pure higher-order change.
        scores[2] += .3 * (0 in s)
        scores[0] += .6 * (1 in s)
        scores[3] -= .9 * ({1, 2, 3} <= s)
        return 1. - MlperfTinyToyCarAdapter._roc_auc(labels, scores)
    table = {s: risk(s) for s in subcube_states(4)}
    assert table[(0,)] < table[()]
    p = .13
    for mask in [(0, 1), (2, 3)]:
        assert integrate_response_table(table, candidate_count=4, protected=frozenset(mask), probability=p) == pytest.approx(
            exact_bernoulli_risk(risk, 4, p, protected=frozenset(mask)), abs=1e-14)
    weighted = 0.
    for state in subcube_states(4):
        d = paired_swap_field_contrast(lambda b: risk([x.bit_index for x in b]),
            [bits[i] for i in state], frozenset(bits[:2]), frozenset(bits[2:]), probability=p)
        weighted += p**len(state) * (1-p)**(4-len(state)) * d["target_minus_baseline"]
    expected = exact_bernoulli_risk(risk, 4, p, protected=frozenset((2, 3))) - exact_bernoulli_risk(risk, 4, p, protected=frozenset((0, 1)))
    assert weighted == pytest.approx(expected, abs=1e-14)
    # The expectation is over complete fixed-cohort AUC, not AUC of mean scores.


def test_pinned_resnet_partition_capture_and_xor_if_available():
    import importlib.metadata
    versions = []
    for package in ("tensorflow", "tensorflow-cpu"):
        try:
            versions.append(importlib.metadata.version(package))
        except importlib.metadata.PackageNotFoundError:
            pass
    if "2.14.1" not in versions:
        pytest.skip("requires authoritative pinned TensorFlow 2.14.1")
    from netsurf.conditional_benchmarks import create_conditional_adapter
    from netsurf.conditional_damage import _schedule, _state_sha256
    from netsurf.protection import ProtectionMask
    cfg = load_conditional(ROOT / "config/examples/m98_resnet8_selection_template.toml")
    a = create_conditional_adapter(cfg); a.load()
    # Already designated engineering identities; never the confirmation window.
    x, y, part = a.sample_partition("conditional_model_selection", 2)
    assert part.indices == (1000, 1001)
    before = _state_sha256(a.values()); clean = a.evaluate(x, y)
    bits = a.bit_identities(); pair = (bits[0], bits[7])
    schedule = _schedule(.01, 989999, pair, len(bits), len(x))
    none = ProtectionMask("none", frozenset(), len(bits), 0., {})
    full = ProtectionMask("full", frozenset(pair), len(bits), 2., {})
    with a.capture_input_losses() as captured:
        _, event = a.evaluate_fault_schedule(x, y, schedule, none)
    assert captured[0].shape == (2,)
    assert (event.proposed_events, event.applied_events, event.suppressed_events) == (2, 2, 0)
    restored, suppressed = a.evaluate_fault_schedule(x, y, schedule, full)
    assert restored == clean
    assert (suppressed.proposed_events, suppressed.applied_events, suppressed.suppressed_events) == (2, 0, 2)
    assert _state_sha256(a.values()) == before
    conformance = a.reference_conformance()
    assert conformance["exact_output_digest_match"]


def test_pinned_toycar_whole_cohort_response_table_if_available():
    import importlib.metadata
    try:
        version = importlib.metadata.version("tensorflow-cpu")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("requires pinned TensorFlow CPU 2.14.1")
    if version != "2.14.1":
        pytest.skip("requires pinned TensorFlow CPU 2.14.1")
    from netsurf.benchmarks.capabilities import load_capability_adapter
    from netsurf.conditional_damage import _schedule, _state_sha256
    from netsurf.protection import ProtectionMask
    a = load_capability_adapter("toycar_int8")
    x, y, partition = a.sample_partition("evaluation", 2)  # Already exposed reference-gate files.
    assert partition.indices == (1, 141)
    bits = a.bit_identities(); candidates = (bits[0], bits[7])
    before = _state_sha256(a.values())
    none = ProtectionMask("none", frozenset(), len(bits), 0., {})
    table = {}
    for state in subcube_states(2):
        schedule = _schedule(.01, 989991, [candidates[i] for i in state], len(bits), len(x))
        result, events = a.evaluate_fault_schedule(x, y, schedule, none)
        assert result.sample_count == 2 and result.window_count == len(x)
        assert events.proposed_events == events.applied_events + events.suppressed_events == len(state)
        assert _state_sha256(a.values()) == before
        table[state] = 1. - result.roc_auc  # Whole fixed cohort, never per-window MSE.
    for protected_index in (0, 1):
        mask = ProtectionMask("one", frozenset([candidates[protected_index]]), len(bits), 1., {})
        direct = 0.
        for state in subcube_states(2):
            result, _ = a.evaluate_fault_schedule(x, y, _schedule(.01, 989991,
                [candidates[i] for i in state], len(bits), len(x)), mask)
            direct += .01**len(state) * .99**(2-len(state)) * (1.-result.roc_auc)
        integrated = integrate_response_table(table, candidate_count=2, protected=frozenset([protected_index]), probability=.01)
        assert direct == pytest.approx(integrated, abs=1e-15)
    assert _state_sha256(a.values()) == before
