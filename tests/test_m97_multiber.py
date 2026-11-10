"""Deterministic scientific and lifecycle checks for the joint BER extension."""
import copy
import itertools
from pathlib import Path

import numpy as np
import pytest

from netsurf.campaign_exposure import check_exposure
from netsurf.conditional_damage import _master_backgrounds
from netsurf.identities import BitIdentity
from netsurf.manifests import digest_object
from netsurf.multiber_campaign import (PILOT_BER_GRID, _check_seed_family, _seed_plan, _window, arm_id,
                                       require_all_selections, verify_partition)
from netsurf.restricted_subcube import (bernoulli_state_mass, conditional_marginal_from_table,
                                        fixed_budget_masks, integrate_response_table,
                                        subcube_states)


def _table(k, risk):
    return {state: risk(set(state)) for state in subcube_states(k)}


def test_zero_ber_conditional_equals_exact_isolated_signed_damage():
    table = _table(4, lambda s: 1. - 3. * (2 in s) + 11. * {0, 1, 2}.issubset(s))
    for i in range(4):
        marginal = conditional_marginal_from_table(table, 4, 0., i)
        assert marginal == pytest.approx(table[(i,)] - table[()])
    assert table[(2,)] - table[()] < 0  # beneficial faults retain their sign


def test_common_and_heterogeneous_ber_normalize_and_integrate():
    states = subcube_states(4)
    for probabilities in (.003, (.001, .003, .01, .2)):
        assert sum(bernoulli_state_mass(s, (0, 1, 2, 3), probabilities) for s in states) == pytest.approx(1.)
    table = _table(4, lambda s: 2. + 7. * (2 in s))
    assert integrate_response_table(table, 4, (.001, .003, .01, .2)) == pytest.approx(2.07)
    assert integrate_response_table(table, 4, (.001, .003, .01, .2), protected=(2,)) == pytest.approx(2.)
    with pytest.raises(ValueError):
        bernoulli_state_mass((0,), (0, 1), (.1, 1.1))


def test_exact_candidate_reweighting_is_ber_specific():
    table = _table(3, lambda s: 5. * {0, 1, 2}.issubset(s) - 2. * (1 in s))
    for p in (.001, .003, .01):
        assert integrate_response_table(table, 3, p) == pytest.approx(5*p**3 - 2*p)
        assert integrate_response_table(table, 3, p, protected=(1,)) == pytest.approx(0.)


def test_outside_backgrounds_are_fresh_bernoulli_fields():
    universe = tuple(BitIdentity("kernel", i, 0) for i in range(3000))
    fields = _master_backgrounds(universe, .003, 32, 976543)
    assert len({seed for seed, _ in fields}) == 32
    assert fields == _master_backgrounds(universe, .003, 32, 976543)
    assert fields != _master_backgrounds(universe, .01, 32, 976543)
    rate = sum(len(bits) for _, bits in fields) / (len(universe) * len(fields))
    assert abs(rate - .003) < .0007
    # Match the exact PCG64 draw; a reused BER-0.01 table would fail this.
    first_seed, first = fields[0]
    rng = np.random.default_rng(first_seed)
    assert first == tuple(bit for bit, hit in zip(universe, rng.random(len(universe)) < .003) if hit)


def test_pure_higher_order_interaction_changes_conditional_vulnerability():
    table = _table(3, lambda s: 9. if {0, 1, 2}.issubset(s) else 0.)
    assert table[(0,)] == table[(1,)] == table[(2,)] == 0.
    assert conditional_marginal_from_table(table, 3, .1, 0) == pytest.approx(.09)
    assert conditional_marginal_from_table(table, 3, .01, 0) == pytest.approx(.0009)


def test_budget_and_identical_pool_seed_plan_across_arms():
    assert len(fixed_budget_masks(6, 2)) == 15
    assert all(22014 + len(mask) == 22016 for mask in fixed_budget_masks(6, 2))
    arms = []
    same_candidates = [f"bit{i}" for i in range(6)]
    for b in range(3):
        pools = []
        for pool in range(12):
            selection, final, _ = _seed_plan(b, pool, 8)
            pools.append({"selection_document": {"restricted_oracle": {
                "calibration_replicate_seeds": selection, "calibration_seed": selection[0] + 9_000_001}},
                "candidates": copy.deepcopy(same_candidates)})
        arms.append({"design": {"selection_fields_per_seed": 8,
                                 "final_seeds": [_seed_plan(b, i, 8)[1] for i in range(12)],
                                 "final_fields": 128}, "pools": pools})
    _check_seed_family(arms)
    assert all(a["pools"][i]["candidates"] == arms[0]["pools"][i]["candidates"]
               for a in arms for i in range(12))
    duplicate = copy.deepcopy(arms)
    duplicate[1]["design"]["final_seeds"][0] = duplicate[0]["design"]["final_seeds"][0]
    with pytest.raises(ValueError, match="seed collision"):
        _check_seed_family(duplicate)


def test_higher_ber_pilot_grid_seeds_and_exposed_input_label():
    assert PILOT_BER_GRID == (.02, .05, .075, .1)
    assert [arm_id(p) for p in PILOT_BER_GRID] == ["ber_0p02", "ber_0p05", "ber_0p075", "ber_0p1"]
    old, new = set(), set()
    for b in range(4):
        for pool in range(12):
            old_selection, old_final, _ = _seed_plan(b, pool, 8)
            selection, final, _ = _seed_plan(b + 10, pool, 4)
            old.update(s + j * 1_000_003 for s in old_selection for j in range(8))
            old.update(old_final + j * 1_000_003 for j in range(128))
            new.update(s + j * 1_000_003 for s in selection[:2] for j in range(4))
            new.update(final + j * 1_000_003 for j in range(32))
    assert old.isdisjoint(new)
    assert len(new) == 4 * 12 * (8 + 32)
    audit = {"source_split": "speech_commands_tfds_test_interleaved_blocks16",
             "exposed_test_indices": [2152, 3962], "limitations": "known evidence only"}
    exposed = _window(audit, 3962, 512, "engineering")
    assert exposed["prior_fault_outcome_overlap"] == [3962]
    with pytest.raises(ValueError, match="prior exposure overlap"):
        _window(audit, 3962, 512, "confirmation")


def test_exposure_windows_and_reserved_indices_rejected():
    audit = {"source_split": "speech_commands_tfds_test_interleaved_blocks16",
             "exposed_test_indices": [2151, 3450, 3961], "limitations": "known evidence only"}
    selection = _window(audit, 2152, 64, "confirmation")
    final = _window(audit, 3962, 512, "confirmation")
    assert selection["original_indices"] == list(range(2152, 2216))
    assert final["original_indices"] == list(range(3962, 4474))
    with pytest.raises(ValueError, match="exposure overlap"):
        _window({**audit, "exposed_test_indices": [3962]}, 3962, 512, "confirmation")
    with pytest.raises(ValueError, match="outside locked"):
        check_exposure(audit, 4474 - 3048, 417, "confirmation")


def test_same_joint_family_final_exposure_is_allowed_but_foreign_is_rejected(tmp_path):
    import json
    from netsurf.campaign_exposure import KWS_SHA, audit_exposure
    from netsurf.manifests import sha256_file
    root = tmp_path / "runs"
    bundle = root / "member"
    bundle.mkdir(parents=True)
    payload = {"campaign_binding": {"campaign_sha256": "arm_one"},
               "sample_identities": [{"sample_id": "speech_commands_tfds_test_interleaved_blocks16:3962"}]}
    (bundle / "scientific_payload.json").write_text(json.dumps(payload))
    digest = digest_object(payload)
    (bundle / "manifest.json").write_text(json.dumps({"schema_version": "test", "benchmark": {
        "checkpoint_sha256": KWS_SHA}, "scientific_payload_sha256": digest}))
    (bundle / "checksums.json").write_text(json.dumps({"files": {
        "scientific_payload.json": sha256_file(bundle / "scientific_payload.json")}}))
    (bundle / "FINALIZED").write_text(f"scientific_payload_sha256={digest}\n")
    allowed = audit_exposure([root], allowed_campaign=["arm_one", "arm_two"])
    assert 3962 not in allowed["exposed_test_indices"]
    foreign = audit_exposure([root], allowed_campaign=["arm_two"])
    assert 3962 in foreign["exposed_test_indices"]


def test_joint_input_identity_binding_and_cross_ber_reuse():
    from netsurf.samples import SamplePartition
    from netsurf.conditional_damage import _sample_identity_records
    inputs = np.arange(8, dtype=np.int8).reshape(2, 4)
    labels = np.array([1, 2])
    partition = SamplePartition("conditional_model_selection", "speech_commands_tfds_test_interleaved_blocks16",
                                (2152, 2153), "identity-hash", "synthetic")
    expected = {"original_indices": [2152, 2153], "sample_sha256": "identity-hash",
                "identity_records_sha256": digest_object(_sample_identity_records(inputs, labels, partition))}
    parent = {"selection_window": expected}
    for _ in (.001, .003, .01):
        verify_partition(parent, "model_selection", (inputs, labels, partition))
    wrong = inputs.copy(); wrong[0, 0] += 1
    with pytest.raises(ValueError, match="encoded inputs"):
        verify_partition(parent, "model_selection", (wrong, labels, partition))
    # The same clips induce covariance; pool-level paired differences retain it.
    shared = np.array([1., 2., 3.])
    low, anchor = shared + .1, shared
    assert np.var(low - anchor) < np.var(low) + np.var(anchor)


def test_all_selection_before_final_enforced(monkeypatch, tmp_path):
    import netsurf.multiber_campaign as module
    arm = {"manifest": str(tmp_path / "arm.json"), "manifest_sha256": "abc"}
    monkeypatch.setattr(module, "load_joint", lambda _: {"arms": [arm]})
    monkeypatch.setattr(module, "load_campaign", lambda *_: {"pools": [{"pool_id": "pool_000"}]})
    def fail(*_):
        raise FileNotFoundError("selection receipt missing")
    monkeypatch.setattr(module, "verified_receipt", fail)
    with pytest.raises(FileNotFoundError, match="selection receipt missing"):
        require_all_selections(tmp_path / "campaign.json")


def test_joint_hash_inventory_rejects_mutation(tmp_path):
    import json
    from netsurf.manifests import sha256_file
    from netsurf.multiber_campaign import load_joint
    root = tmp_path / "campaign"
    root.mkdir()
    manifest = root / "campaign.json"
    manifest.write_text(json.dumps({"schema_version": "1.0.0", "protocol": "joint-kws-multiber/v1",
                                    "ber_grid": [], "arms": [], "source_m96": {
                                        "manifest": str(root / "source.json"), "sha256": "placeholder"}}))
    (root / "FROZEN").write_text(sha256_file(manifest))
    (root / "checksums.json").write_text(json.dumps({"files": {"campaign.json": sha256_file(manifest)}}))
    manifest.write_text(manifest.read_text() + " ")
    with pytest.raises(ValueError, match="not frozen"):
        load_joint(manifest)


def test_engineering_receipts_show_deterministic_restoration_when_present():
    import json
    root = Path(__file__).resolve().parents[1] / "workspace/runs/m97_engineering_v1"
    receipt_dir = root / "arms/ber_0p003/receipts"
    if not receipt_dir.is_dir():
        pytest.skip("local engineering evidence is not distributed with source")
    for phase in ("selection", "final"):
        receipt = json.loads((receipt_dir / f"pool_000_{phase}.json").read_text())
        payload = json.loads((Path(receipt["bundle"]) / "scientific_payload.json").read_text())
        assert payload["checks"]["state_restored"]
        assert payload["checks"]["initial_state_sha256"] == payload["checks"]["final_state_sha256"]
        assert payload["checks"]["global_cardinality_preserved"]
        assert payload["checks"]["final_samples_accessed"] == (phase == "final")
