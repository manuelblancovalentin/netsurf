from dataclasses import replace
import io
import json
from pathlib import Path

import numpy as np
import pytest

from netsurf.cross_pool import sample_pools, operation_counts, load_campaign, verify_pool_configuration, PROVENANCE, _toml
from netsurf.cross_pool_analysis import cross_pool_statistics
from netsurf.campaign_exposure import check_exposure
from netsurf.identities import BitIdentity
from netsurf.manifests import digest_object, sha256_file
from netsurf.restricted_subcube import subcube_states, fixed_budget_masks, integrate_response_table
from netsurf.conditional_damage import exact_bernoulli_risk
from netsurf.paired_swap import paired_swap_field_contrast
from netsurf.selection_preflight import selection_summary

ROOT = Path(__file__).resolve().parents[1]


def test_uniform_status_sampling_and_reproducible_order():
    universe = tuple(BitIdentity("kernel", i, b) for i in range(100) for b in range(8))
    protected = frozenset(universe[::8])
    pools = sample_pools(universe, protected, [31, 32, 33], 6)
    assert pools == sample_pools(universe, protected, [31, 32, 33], 6)
    assert pools[0]["candidate_order_sha256"] != pools[1]["candidate_order_sha256"]
    for pool in pools:
        rows = pool["candidates"]
        assert len(rows) == len({r["identity"] for r in rows}) == 6
        assert sum(r["source_protected"] for r in rows) == 2
        assert [BitIdentity(r["parameter"], r["flat_index"], r["bit_index"]) for r in rows] == sorted(BitIdentity(r["parameter"], r["flat_index"], r["bit_index"]) for r in rows)
        assert all(r["inclusion_probability"] == (2/100 if r["source_protected"] else 4/700) for r in rows)
    assert len(sample_pools(universe, protected, [41], 8)[0]["candidates"]) == 8
    with pytest.raises(ValueError, match="repeated"):
        sample_pools(universe, protected, [31, 31], 6)


def test_operation_counts_and_exposure_rejection():
    counts = operation_counts(6, 32, 64, 128, 512, 12)
    assert counts["selection_state_requests_per_pool"] == 2055
    assert counts["response_table_evaluations_total"] == 24576
    assert counts["selection_clip_inferences_total"] == 1578240
    assert counts["final_clip_inferences_upper_bound_total"] == 9437184
    ledger = {"exposed_test_indices": list(range(3194, 3450)), "limitations": "test"}
    with pytest.raises(ValueError, match="exposure overlap"):
        check_exposure(ledger, 146, 256, "confirmation")
    assert check_exposure(ledger, 402, 512, "confirmation")["requested_original_indices"] == list(range(3450, 3962))
    assert check_exposure(ledger, 146, 2, "engineering")["prior_overlap"] == [3194, 3195]


@pytest.mark.parametrize("kind", ["additive", "pairwise", "beneficial", "higher_order"])
def test_all_equal_budget_masks_and_paired_identity(kind):
    candidates = tuple(BitIdentity("kernel", 0, i) for i in range(6))
    def risk(state):
        s = set(state)
        if kind == "higher_order":
            return 8. * set(range(6)).issubset(s)
        return sum((i + 1) * (-1 if kind == "beneficial" else 1) for i in s) + (13. * {1, 2}.issubset(s) if kind == "pairwise" else 0.)
    table = {s: risk(s) for s in subcube_states(6)}
    from netsurf.restricted_subcube import select_response_tables
    isolated = {s: risk(s) for s in [(), *((i,) for i in range(6))]}
    selected = selection_summary([table, table], isolated, candidates, 2, .1, (0, 1), [101, 102], 1)
    assert len(selected["eligible_masks"]) == 15
    for row in selected["eligible_masks"]:
        mask = frozenset(row["indices"])
        direct = exact_bernoulli_risk(risk, 6, .1, protected=mask)
        assert row["mean_risk"] == pytest.approx(direct, abs=1e-12)
        assert 22014 + len(mask) == 22016
    # Existing complete-field telescoping identity, including beneficial effects.
    target = frozenset(candidates[i] for i in (2, 3)); base = frozenset(candidates[i] for i in (0, 1))
    exact = 0.
    for s in subcube_states(6):
        mass = .1**len(s) * .9**(6-len(s))
        result = paired_swap_field_contrast(lambda bits: risk([b.bit_index for b in bits]),
            tuple(candidates[i] for i in s), base, target, probability=.1)
        exact += mass * result["target_minus_baseline"]
    expected = exact_bernoulli_risk(risk, 6, .1, protected=frozenset((2, 3))) - exact_bernoulli_risk(risk, 6, .1, protected=frozenset((0, 1)))
    assert exact == pytest.approx(expected, abs=1e-12)


def test_pool_statistics_do_not_flatten_or_drop_harmful_pools():
    x = np.zeros((3, 4, 5, 3))
    x[0] = -1.; x[1] = 0.; x[2] = 4.
    a = cross_pool_statistics(x, seed=15, replicates=100)
    b = cross_pool_statistics(x, seed=15, replicates=100)
    assert a == b
    row = a["comparisons"][0]
    assert row["pool_mean_effect"]["count"] == 3
    assert row["pool_mean_effect"]["mean"] == 1.
    from scipy.stats import t
    se = np.std([-1., 0., 4.], ddof=1) / np.sqrt(3)
    assert row["pool_mean_effect"]["confidence_interval"] == pytest.approx((1-t.ppf(.975, 2)*se, 1+t.ppf(.975, 2)*se))
    assert row["estimated_harmful_pools"] == row["estimated_zero_pools"] == row["estimated_improving_pools"] == 1
    assert a["bootstrap"]["diagnostics"]["fields_conditional"]["standard_deviation"] == [0., 0., 0.]
    # Shared input shocks must not vanish by resampling each pool's clips independently.
    common = np.broadcast_to(np.arange(5)[None, None, :, None], (3, 4, 5, 3)).copy()
    c = cross_pool_statistics(common, seed=15, replicates=100)
    assert c["bootstrap"]["diagnostics"]["inputs_shared_conditional"]["standard_deviation"][0] > .3
    with pytest.raises(ValueError):
        cross_pool_statistics(x[:1], seed=15, replicates=100)


def test_manifest_inventory_tampering_rejected(tmp_path):
    record = {"schema_version": "1.0.0", "protocol": "cross-pool-frozen-protection/v1"}
    path = tmp_path / "campaign.json"; path.write_text(json.dumps(record))
    digest = sha256_file(path)
    (tmp_path / "FROZEN").write_text(digest)
    (tmp_path / "checksums.json").write_text(json.dumps({"files": {"campaign.json": digest}}))
    assert load_campaign(path, digest) == record
    path.write_text(json.dumps({**record, "changed": True}))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_campaign(path, digest)


def test_campaign_progress_uses_existing_single_queue(monkeypatch):
    from netsurf.cross_pool_progress import CampaignProgress
    from netsurf.progress import HierarchicalProgressReporter
    import netsurf.cross_pool_progress as module
    stream = io.StringIO()
    monkeypatch.setattr(module, "HierarchicalProgressReporter", lambda items: HierarchicalProgressReporter(items, stream=stream, interactive=False))
    with CampaignProgress(["pool_000", "pool_001"], planned=10) as progress:
        progress.start(0); progress.callback(0)("selection", 5, 10); progress.complete(0)
        progress.start(1); progress.fail(1, "synthetic failure")
    assert "\x1b" not in stream.getvalue()
    assert "synthetic failure" in stream.getvalue()


def test_setup_does_not_contaminate_authoritative_state_timing(monkeypatch):
    import netsurf.cross_pool_progress as module
    from types import SimpleNamespace
    events = []
    monkeypatch.setattr(module, "HierarchicalProgressReporter", lambda _: SimpleNamespace(update=lambda i, e: events.append(e)))
    progress = module.CampaignProgress(["pool"], planned=10)
    progress.callback(0)("selection", 2, 10)
    progress.callback(0)("selection", 3, 10)
    assert [e.work_class for e in events] == ["setup", "initial state batch", "authoritative states"]
    assert [e.completed for e in events] == [1, 3, 4]
    assert events[-1].total == 12


def test_roadmap_registry_preserves_complete_portfolio_and_gates():
    registry = json.loads((ROOT / "config/benchmark_expansion_registry.json").read_text())
    entries = {r["id"]: r for r in registry["entries"]}
    assert registry["schema_version"] == "1.0.0"
    assert len(entries) == 18 and len({r["family"] for r in entries.values()}) == 11
    assert sum(r["status"] == "Ready" for r in entries.values()) == 3
    assert [r["id"] for r in entries.values() if r["conditional_protocol_ready"]] == ["kws_dscnn_int8"]
    assert entries["pythia160m_bf16"]["status"] == "Research-stage"
    assert entries["vww_mobilenetv1_int8"]["status"] == "Asset acquisition required"
    assert entries["toycar_int8"]["eligible_kernel_bits"] == 2113536
    assert all(r["eligible_kernel_bits"] is None for r in entries.values() if r["track"] != "A")


def test_schema66_requires_campaign_binding():
    import tomllib
    doc = tomllib.loads((ROOT / "config/examples/m956_kws_s5_selection_k8.toml").read_text())
    doc["schema_version"] = "6.6.0"; doc["scientific_protocol"] = "cross-pool-frozen-protection/v7"
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((ROOT / "netsurf/schemas/conditional/v1_6.schema.json").read_text())
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, schema)
    doc["workflow"].update(candidate_provenance=PROVENANCE, campaign_manifest="campaign.json", campaign_sha256="a" * 64, pool_id="pool_000", record_per_input=True)
    jsonschema.validate(doc, schema)


def test_campaign_request_schema_and_fail_closed_validation(tmp_path):
    import tomllib
    from netsurf.cross_pool import freeze_campaign
    request = tomllib.loads((ROOT / "config/examples/m96_kws_cross_pool_k6.toml").read_text())
    schema = json.loads((ROOT / "netsurf/schemas/campaign/v1.schema.json").read_text())
    assert set(request) == set(schema["required"])
    for table in ("design", "samples", "statistics"):
        assert set(request[table]) == set(schema["properties"][table]["required"])
    for change in ("version", "unknown", "duplicate_field_seed", "unsupported_ber"):
        bad = json.loads(json.dumps(request))
        if change == "version":
            bad["schema_version"] = "2.0.0"
        elif change == "unknown":
            bad["unrecognized"] = True
        elif change == "duplicate_field_seed":
            bad["design"]["final_seeds"][1] = bad["design"]["final_seeds"][0]
        else:
            bad["design"]["logical_ber"] = .1
        source = tmp_path / (change + ".toml")
        source.write_text(_toml(bad))
        output = tmp_path / change
        with pytest.raises(ValueError):
            freeze_campaign(source, output)
        assert not output.exists()


def test_campaign_binding_rejects_changed_science_before_loading(tmp_path):
    from tests.test_m956_preflight import configuration
    from netsurf.cross_pool import _bound_document
    config = configuration(tmp_path)
    manifest = tmp_path / "campaign.json"
    config = replace(config, schema_version="6.6.0", scientific_protocol="cross-pool-frozen-protection/v7",
        output=replace(config.output, root=tmp_path / "runs"),
        workflow={"mode": "selection_only", "purpose": "engineering", "candidate_provenance": PROVENANCE,
                  "record_per_input": True, "pool_id": "pool_000", "campaign_manifest": str(manifest)})
    doc = config.resolved_document()
    for key in ("ranking_samples", "evaluation_samples"):
        doc["benchmark"].pop(key)
    doc.pop("operation_counts")
    for key in ("campaign_manifest", "pool_id"):
        doc["workflow"].pop(key)
    record = {"schema_version": "1.0.0", "protocol": "cross-pool-frozen-protection/v1", "purpose": "engineering",
              "pools": [{"pool_id": "pool_000", "selection_document": doc}]}
    manifest.write_text(json.dumps(record)); digest = sha256_file(manifest)
    (tmp_path / "FROZEN").write_text(digest)
    (tmp_path / "checksums.json").write_text(json.dumps({"files": {"campaign.json": digest}}))
    config = replace(config, workflow={**config.workflow, "campaign_sha256": digest})
    verify_pool_configuration(config, execution=False)
    for changed in (replace(config, candidates=tuple(reversed(config.candidates))),
                    replace(config, samples=replace(config.samples, evaluation_count=99)),
                    replace(config, paired_swap_evaluation=replace(config.paired_swap_evaluation, seed=1001)),
                    replace(config, restricted_oracle=replace(config.restricted_oracle, calibration_background_count=20))):
        with pytest.raises(ValueError, match="frozen campaign"):
            verify_pool_configuration(changed, execution=False)


def test_incomplete_selection_registry_and_interrupted_attempt_fail_closed(tmp_path, monkeypatch):
    import netsurf.cross_pool as campaign
    from types import SimpleNamespace
    path = tmp_path / "campaign.json"; path.write_text('{}')
    record = {"pools": [{"pool_id": "pool_000"}, {"pool_id": "pool_001"}],
              "operation_counts": {"selection_state_requests_per_pool": 10}}
    monkeypatch.setattr(campaign, "load_campaign", lambda *_: record)
    with pytest.raises(FileNotFoundError):
        campaign.seal_selections(path)
    assert not (tmp_path / "final-plan").exists()
    config_path = tmp_path / "configs/pool_000_selection.toml"
    config_path.parent.mkdir(); config_path.write_text("synthetic")
    (tmp_path / "attempts").mkdir(); (tmp_path / "attempts/pool_000_selection.json").write_text('{}')
    monkeypatch.setattr("netsurf.conditional_config.load_conditional", lambda *_: SimpleNamespace())
    monkeypatch.setattr(campaign, "verify_pool_configuration", lambda *a, **k: record)
    def forbidden(*a, **k):
        raise AssertionError("interrupted attempt must not rerun inference")
    monkeypatch.setattr("netsurf.conditional_damage.execute_conditional_damage", forbidden)
    with pytest.raises(FileExistsError):
        campaign.run_campaign(path, "selection", pool_ids=["pool_000"], progress=False)


def test_completed_receipt_resume_does_not_execute(tmp_path, monkeypatch):
    import netsurf.cross_pool as campaign
    path = tmp_path / "campaign.json"; path.write_text('{}')
    (tmp_path / "receipts").mkdir(); (tmp_path / "receipts/pool_000_selection.json").write_text('{}')
    record = {"pools": [{"pool_id": "pool_000"}], "operation_counts": {"selection_state_requests_per_pool": 10}}
    monkeypatch.setattr(campaign, "load_campaign", lambda *_: record)
    monkeypatch.setattr(campaign, "verified_receipt", lambda *a: ({"verified": True}, {}))
    def forbidden(*a, **k):
        raise AssertionError("completed phase must not access adapter")
    monkeypatch.setattr("netsurf.conditional_config.load_conditional", forbidden)
    assert campaign.run_campaign(path, "selection", progress=False)["completed"] == [{"verified": True}]


def test_field_pairing_regression():
    from netsurf.replication_statistics import summarize_replication
    from netsurf.conditional_config import PAIRED_SWAP_COMPARISONS
    rows = [{"comparison": comparison, "field": f, "field_seed": f + 100,
             "logical_ber": .01, "master_pairing_key": str(f), "master_sha256": str(f),
             "sample_sha256": "same", "protection_context_sha256": "same",
             "target_minus_baseline": .01 * (f + i)}
            for i, comparison in enumerate(PAIRED_SWAP_COMPARISONS) for f in range(3)]
    assert summarize_replication(rows, 3)["independent_field_count"] == 3
    rows[-1]["master_sha256"] = "different"
    with pytest.raises(ValueError, match="identical master"):
        summarize_replication(rows, 3)


def test_completed_engineering_evidence_if_available(tmp_path):
    from netsurf.cross_pool import verified_receipt
    from netsurf.conditional_config import load_conditional
    root = ROOT / "workspace/runs/m96_engineering_campaign_v1"
    if not (root / "receipts/pool_001_final.json").exists():
        pytest.skip("optional real KWS engineering evidence not available")
    digest = sha256_file(root / "campaign.json")
    for pid in ("pool_000", "pool_001"):
        _, selected = verified_receipt(root / f"receipts/{pid}_selection.json", digest, pid, "selection")
        _, final = verified_receipt(root / f"receipts/{pid}_final.json", digest, pid, "final")
        assert not selected["checks"]["final_samples_accessed"]
        assert selected["frozen_selection"] == final["frozen_selection"]
        assert final["checks"]["state_restored"]
        assert final["checks"]["global_cardinality_preserved"]
        assert final["exposure_audit"]["requested_original_indices"] == [3060, 3061]
        load_conditional(root / f"final-plan/{pid}_final.toml")
    from netsurf.cross_pool_analysis import aggregate_campaign
    first = aggregate_campaign(root / "campaign.json", tmp_path / "first", plots=False)
    second = aggregate_campaign(root / "campaign.json", tmp_path / "second", plots=False)
    assert first["scientific_payload_sha256"] == second["scientific_payload_sha256"]
    assert "previously exposed inputs" in (tmp_path / "first/report.md").read_text()
    with pytest.raises(FileExistsError):
        aggregate_campaign(root / "campaign.json", tmp_path / "first", plots=False)
