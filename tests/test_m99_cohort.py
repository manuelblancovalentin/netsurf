"""Nonadditive ROC-AUC algebra and protocol tests; no ML dependency."""
from dataclasses import replace
import itertools
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from netsurf.cohort_risk import (CONTRACT, ROLES, SOURCE, cohort_risk, ordering_diagnostics,
    score_digest, validate_cohorts, exposure_check)
from netsurf.cohort_analysis import scalar_pool_statistics, verify_score_evidence
from netsurf.conditional_damage import _risk_value, exact_bernoulli_risk, exact_conditional_damage, execute_conditional_damage
from netsurf.restricted_subcube import subcube_states, integrate_response_table
from netsurf.paired_swap import paired_swap_field_contrast
from netsurf.identities import BitIdentity
from netsurf.adapters.mlperf_tiny_multibench import AnomalyEvaluation
from netsurf.manifests import digest_object, sha256_file


def test_auc_ties_beneficial_faults_and_nonadditivity():
    labels = [0, 0, 1, 1]
    scores = [1., 2., 2., 3.]
    assert cohort_risk(labels, scores) == .125
    assert cohort_risk(labels, [1, 2, 3, 4]) == 0
    assert cohort_risk(labels, [1, 1, 1, 1]) == .5
    d = ordering_diagnostics(labels, [3, 2, 1, 4], scores)
    assert d["strict_reversals"] > 0 and d["risk_minus_clean"] > 0
    # Mean of cohort risks is not risk of averaged file scores.
    assert .5 * (cohort_risk([0, 1], [0, 1]) + cohort_risk([0, 1], [4, 0])) != cohort_risk([0, 1], [2, .5])
    with pytest.raises(ValueError): cohort_risk([0, 0], [1, 2])
    with pytest.raises(ValueError): cohort_risk([0, 1], [1, float('nan')])
    e = AnomalyEvaluation(1000., .75, 2, 392, (1., 2.))
    assert _risk_value(e, "one_minus_file_roc_auc") == .25
    assert _risk_value(e, "reconstruction_mse") == 1000.


@pytest.mark.parametrize("kind", ["isolated", "beneficial", "higher_order", "ties"])
def test_complete_cohort_response_integration_and_paired_swap_all_equal_budget_masks(kind):
    labels = [0, 0, 1, 1]
    bits = tuple(BitIdentity("kernel", 0, b) for b in range(4))
    def risk(state):
        s = set(state)
        scores = [.2, .5, .4, .8]
        if kind == "isolated": scores[0] += .6 * (0 in s)
        if kind == "beneficial": scores[2] += .3 * (0 in s)
        if kind == "higher_order": scores[3] -= 1. * ({0, 1, 2} <= s)
        if kind == "ties": scores = [0., 1., float(0 in s), 1.]
        return cohort_risk(labels, scores)
    table = {s: risk(s) for s in subcube_states(4)}
    p = .17
    masks = [frozenset(m) for m in itertools.combinations(range(4), 2)]
    for mask in masks:
        exact = exact_bernoulli_risk(risk, 4, p, protected=mask)
        assert integrate_response_table(table, candidate_count=4, protected=mask, probability=p) == pytest.approx(exact, abs=1e-14)
    for a, b in itertools.combinations(masks, 2):
        expectation = 0.
        for state in subcube_states(4):
            row = paired_swap_field_contrast(lambda active: risk([x.bit_index for x in active]),
                [bits[i] for i in state], frozenset(bits[i] for i in a), frozenset(bits[i] for i in b), probability=p)
            expectation += p**len(state) * (1-p)**(4-len(state)) * row["target_minus_baseline"]
        assert expectation == pytest.approx(exact_bernoulli_risk(risk, 4, p, protected=b) - exact_bernoulli_risk(risk, 4, p, protected=a), abs=1e-14)
    if kind == "higher_order":
        assert risk((0,)) == risk(()) and exact_conditional_damage(risk, 4, 0, p) > 0


def test_cohort_validation_exposure_and_statistical_units():
    s = SimpleNamespace(calibration_count=2, model_selection_count=2, evaluation_count=2,
                        calibration_offset=0, model_selection_offset=0, evaluation_offset=0)
    cohorts = dict(zip(ROLES, ([2, 142], [0, 140], [1, 141])))
    assert validate_cohorts(cohorts, s) == cohorts
    for bad in ({**cohorts, ROLES[2]: [0, 141]}, {**cohorts, ROLES[1]: [3, 4]}, {**cohorts, ROLES[2]: [True, 141]}):
        with pytest.raises(ValueError): validate_cohorts(bad, s)
    audit = {"exposed_test_indices": [1], "limitations": "test"}
    with pytest.raises(ValueError, match="exposure"):
        exposure_check(audit, cohorts, "confirmation")
    assert exposure_check(audit, cohorts, "engineering")["prior_overlap_by_role"][ROLES[2]] == [1]
    x = np.array([[[.1, .2, .3], [.3, .4, .5]], [[-.1, -.2, -.3], [-.3, -.4, -.5]]])
    stats = scalar_pool_statistics(x)
    assert stats["pool_count"] == 2 and stats["fields_per_pool"] == 2
    assert stats["comparisons"][0]["pool_mean_effect"]["mean"] == 0
    assert stats["comparisons"][0]["pool_mean_effect"]["count"] == 2
    assert stats["comparisons"][0]["field_intervals_by_pool"][0]["count"] == 2
    with pytest.raises(ValueError): scalar_pool_statistics(x[:, :, None, :])
    from scipy.stats import t
    mean_se = np.std([.2, -.2], ddof=1) / np.sqrt(2)
    assert stats["comparisons"][0]["pool_mean_effect"]["confidence_interval"] == pytest.approx(
        [-t.ppf(.975, 1) * mean_se, t.ppf(.975, 1) * mean_se])


def test_streamed_mask_selection_matches_original_parser(tmp_path, monkeypatch):
    from netsurf.protection_context import read_mask_identities
    path = tmp_path / "mask.json"
    path.write_text(json.dumps({"replication_0": {"S5": [["x", 1, 7], ["x", 2, 0]]}, "replication_1": {"S5": [["x", 3, 5]]}}))
    expected = read_mask_identities(path, "replication_0/S5")
    original = Path.stat
    monkeypatch.setattr(Path, "stat", lambda p, **kw: SimpleNamespace(st_size=20*1024**2) if p == path else original(p, **kw))
    assert read_mask_identities(path, "replication_0/S5") == expected
    with pytest.raises(ValueError, match="missing or empty"): read_mask_identities(path, "replication_2/S5")


@pytest.mark.parametrize("chunk", [3, 11, 37, 1000])
def test_stream_tokens_boundaries(tmp_path, chunk):
    from netsurf.mask_stream import json_tokens
    from netsurf.mechanism_audit import _json_tokens
    p = tmp_path / "mask.json"
    # Tiny chunks still need enough capacity for the longest indivisible token.
    p.write_text('{"x": [["a", 12345, 7], ["b", -1, 0]], "y": []}')
    expected = list(_json_tokens(p))
    assert list(json_tokens(p, chunk)) == expected


def test_schema68_configuration_and_rejected_additive_semantics(tmp_path, monkeypatch):
    import tomllib
    from netsurf.conditional_config import load_conditional
    from netsurf.cross_pool import _toml
    from netsurf.conditional_benchmarks import conditional_capability
    root = Path(__file__).resolve().parents[1]
    doc = tomllib.loads((root / "config/examples/m99_toycar_selection_template.toml").read_text())
    monkeypatch.setattr("netsurf.cohort_risk.configuration_exposure", lambda *a, **k: {})
    path = tmp_path / "config.toml"
    path.write_text(_toml(doc))
    conf = load_conditional(path, check_inputs=False)
    assert conf.schema_version == "6.8.0" and conf.cohorts[ROLES[2]] == [1, 141]
    assert conf.operation_counts()["restricted_oracle_requested_evaluations"] == 135
    for key, value in (("metric", "reconstruction_mse"), ("metric", "categorical_crossentropy")):
        bad = json.loads(json.dumps(doc)); bad["risk"][key] = value
        path.write_text(_toml(bad))
        with pytest.raises(ValueError, match="one_minus_file_roc_auc"):
            load_conditional(path, check_inputs=False)
    bad = json.loads(json.dumps(doc)); bad["workflow"]["record_per_input"] = True
    path.write_text(_toml(bad))
    with pytest.raises(ValueError, match="no additive"):
        load_conditional(path, check_inputs=False)
    bad = json.loads(json.dumps(doc)); bad["schema_version"] = "6.7.0"; bad["scientific_protocol"] = "capability-conditional-protection/v8"
    path.write_text(_toml(bad))
    with pytest.raises(ValueError, match="requires scalar-cohort"):
        load_conditional(path, check_inputs=False)


def test_pinned_complete_files_xor_restore_capture_if_available():
    from importlib.metadata import version, PackageNotFoundError
    try:
        if version("tensorflow-cpu") != "2.14.1": pytest.skip("pinned TF2.14.1 required")
    except PackageNotFoundError:
        pytest.skip("pinned TF2.14.1 required")
    from netsurf.conditional_config import load_conditional
    from netsurf.conditional_benchmarks import create_conditional_adapter
    from netsurf.conditional_damage import _schedule, _state_sha256
    from netsurf.protection import ProtectionMask
    root = Path(__file__).resolve().parents[1]
    cfg = load_conditional(root / "config/examples/m99_toycar_selection_template.toml")
    a = create_conditional_adapter(cfg); a.load()
    x, y, part = a.sample_partition(ROLES[2], 2)
    assert part.indices == (1, 141) and x.shape == (392, 640)
    before = _state_sha256(a.values()); clean = a.evaluate(x, y)
    bits = a.bit_identities(); pair = (bits[0], bits[7])
    schedule = _schedule(.001, 999999, pair, len(bits), len(x))
    none = ProtectionMask("none", frozenset(), len(bits), 0., {})
    full = ProtectionMask("full", frozenset(pair), len(bits), 2., {})
    first, report = a.evaluate_fault_schedule(x, y, schedule, none)
    repeat, _ = a.evaluate_fault_schedule(x, y, schedule, none)
    restored, suppressed = a.evaluate_fault_schedule(x, y, schedule, full)
    assert first == repeat and restored == clean
    assert (report.proposed_events, report.applied_events, report.suppressed_events) == (2, 2, 0)
    assert (suppressed.proposed_events, suppressed.applied_events, suppressed.suppressed_events) == (2, 0, 2)
    assert before == _state_sha256(a.values())
    evidence = a.cohort_evidence(); verify_score_evidence(evidence)
    assert evidence["evaluations"][0]["scores_sha256"] == "cd135be4abd970a9856891741a8fb09b7f67af6bd0552a9c23be28c848ef9fb1"
    assert evidence["evaluations"][1]["scores_sha256"] == evidence["evaluations"][2]["scores_sha256"]
    assert all(r["window_count"] == 392 for r in evidence["cohorts"].values())


def test_scalar_engine_selection_final_hashes_and_frozen_masks(tmp_path, monkeypatch):
    from tests.test_m956_preflight import configuration
    from tests.test_m95_conditional_damage import _FakeExactAdapter
    from netsurf.samples import SamplePartition, sample_identity_digest
    from netsurf.faults import InjectionReport
    from netsurf.conditional_bundle import create_conditional_bundle, verify_conditional_bundle
    from netsurf.selection_preflight import load_frozen_selection
    from netsurf.conditional_benchmarks import conditional_capability
    cfg = configuration(tmp_path)
    cfg = replace(cfg, schema_version="6.8.0", scientific_protocol="scalar-cohort-conditional-protection/v9",
        benchmark=replace(cfg.benchmark, adapter="mlperf_tiny_ad_int8_v1"), risk_metric="one_minus_file_roc_auc",
        cohorts=dict(zip(ROLES, ([2, 142], [0, 140], [1, 141]))))
    monkeypatch.setattr("netsurf.cohort_risk.configuration_exposure", lambda c, **k: {
        "source_split": SOURCE, "requested_original_indices": c.cohorts[ROLES[2]], "limitations": "synthetic"})
    class CohortFixture(_FakeExactAdapter):
        def __init__(self, role):
            super().__init__(); self.role = role; self.seen = []; self.cohorts = {}; self.rows = []
        def sample_partition(self, role, count, *, offset=0):
            assert role == self.role  # Selection must not load final files or vice versa.
            self.seen.append(role); indices = cfg.cohorts[role]
            x, y = np.zeros((2, 1), np.int8), np.array([0, 1])
            files = [{"sample_id": f"{SOURCE}:{i}", "source_split": SOURCE, "original_index": i,
                      "label": int(label), "window_count": 196, "ordered_window_ids": list(range(196))}
                     for i, label in zip(indices, y)]
            r = {"role": role, "files": files, "file_count": 2, "window_count": 392}
            r["cohort_sha256"] = digest_object(r); self.cohorts[r["cohort_sha256"]] = r
            return x, y, SamplePartition(role, SOURCE, tuple(indices), sample_identity_digest(SOURCE, indices, y), "synthetic complete-file scoring fixture")
        def sample_identity_records(self, *data): return next(iter(self.cohorts.values()))["files"]
        def partition_diagnostics(self, *data): return next(iter(self.cohorts.values()))
        def evaluate(self, x, y): return self.record([.2, .4])
        def record(self, scores):
            risk = cohort_risk([0, 1], scores)
            self.rows.append({"cohort_sha256": next(iter(self.cohorts)), "fault": None,
                "scores": scores, "scores_float64_hex": [v.hex() for v in scores], "scores_sha256": score_digest(scores),
                "risk": risk, "pair_ordering": ordering_diagnostics([0, 1], scores)})
            return AnomalyEvaluation(100., 1 - risk, 2, 392, tuple(scores))
        def evaluate_fault_schedule(self, x, y, schedule, protection):
            active = {e.bit for e in schedule.events if e.bit not in protection.protected}
            scores = [.2 + .4 * (cfg.candidates[0] in active), .4 - .3 * (cfg.candidates[1] in active)]
            return self.record(scores), InjectionReport(schedule.digest(), len(schedule.events), len(active), len(schedule.events)-len(active), len(active))
        def cohort_evidence(self): return {"schema_version": "1.0.0", "risk_contract": CONTRACT, "cohorts": self.cohorts, "evaluations": self.rows}
    first = execute_conditional_damage(cfg, adapter=CohortFixture(ROLES[1]))
    repeat = execute_conditional_damage(cfg, adapter=CohortFixture(ROLES[1]))
    assert first.scientific_payload_sha256 == repeat.scientific_payload_sha256
    verify_score_evidence(first.scientific_payload["scalar_cohort_evidence"])
    corrupted = json.loads(json.dumps(first.scientific_payload["scalar_cohort_evidence"]))
    corrupted["evaluations"][0]["scores"][0] += 1
    with pytest.raises(ValueError, match="scores"):
        verify_score_evidence(corrupted)
    root = create_conditional_bundle(first, repository_root=Path(__file__).resolve().parents[1])
    verify_conditional_bundle(root)
    assert "Mean 1 − file ROC-AUC" in (root / "report.md").read_text()
    final = replace(cfg, paired_swap_evaluation=replace(cfg.paired_swap_evaluation, enabled=True),
        workflow={**cfg.workflow, "mode": "final_only", "selection_bundle": str(root),
            "selection_payload_sha256": first.scientific_payload_sha256,
            "selection_artifact_sha256": sha256_file(root / "artifacts/frozen_selection.json")})
    load_frozen_selection(final)
    monkeypatch.setattr("netsurf.selection_preflight.selection_summary", lambda *a, **k: pytest.fail("final tried to select masks"))
    held = execute_conditional_damage(final, adapter=CohortFixture(ROLES[2]))
    assert held.scientific_payload["frozen_selection"] == first.scientific_payload["frozen_selection"]
    assert not held.oracle_response_records and len(held.paired_swap_field_records) == 9
    assert held.scientific_payload["benchmark"]["task_metrics"]["application_metric"] == "file_roc_auc"
    assert "paired_input_diagnostics" not in held.scientific_payload
    assert held.scientific_payload["checks"]["state_restored"]
    with pytest.raises(ValueError, match="contract differs"):
        load_frozen_selection(replace(final, cohorts={**final.cohorts, ROLES[2]: [3, 143]}))
    from netsurf.result_bundle import ResultBundleError
    with pytest.raises(ResultBundleError, match="already exists"):
        create_conditional_bundle(first, repository_root=Path(__file__).resolve().parents[1], run_id=root.name)
