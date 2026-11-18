"""Bounded clean-gate checks; no model inference or new fault fields."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from netsurf.manifests import digest_object, sha256_file
from netsurf.w8a8_campaign import _risk_record
from netsurf import w8a8_pythia_wide_confirm as campaign


def _risk(value: float):
    return SimpleNamespace(
        metrics={"mean_nll": value, "failure_fraction": 0.0},
        failed_units=0, total_units=2,
        logits=((value,), (value + 1,)), failure_kinds=(None, None),
    )


def test_exact_clean_gate_records_drift_and_never_accepts_changed_output(tmp_path):
    expected = _risk_record(_risk(1.0))
    campaign._assert_clean(_risk(1.0), expected, root=tmp_path, label="clean")
    with pytest.raises(RuntimeError, match="clean conformance failed"):
        campaign._assert_clean(_risk(1.1), expected, root=tmp_path,
                               label="field_0_post", baseline_logits=_risk(1.0).logits)
    diagnostic = json.loads((tmp_path / "clean_drift/field_0_post.json").read_text())
    assert diagnostic["first_differing_token"] == 0
    assert diagnostic["observed"]["metrics"]["mean_nll"] == 1.1


def test_existing_clean_receipt_is_write_once(tmp_path):
    path = tmp_path / "clean.json"
    campaign._write_or_verify(path, {"mean_nll": 1.0})
    first_hash = sha256_file(path)
    campaign._write_or_verify(path, {"mean_nll": 1.0})
    assert sha256_file(path) == first_hash
    with pytest.raises(ValueError, match="existing clean record differs"):
        campaign._write_or_verify(path, {"mean_nll": 2.0})
    assert sha256_file(path) == first_hash


def test_seal_requires_all_four_independent_identical_clean_outputs(tmp_path, monkeypatch):
    plan = {"plan_sha256": "plan"}
    expected = _risk_record(_risk(1.0))
    campaign._write(tmp_path / "plan.json", plan)
    monkeypatch.setattr(campaign, "_protocol", lambda _root: {
        "plan_sha256": "plan", "expected_clean_risk": expected})
    for shard in range(campaign.SHARDS):
        rows = [{"risk": expected, "token_stats_sha256": "same"}
                for _ in range(campaign.REPEATS)]
        receipt = {"shard": shard, "plan_sha256": "plan", "records": rows}
        receipt["payload_sha256"] = digest_object(receipt)
        campaign._write(tmp_path / "preflight" / f"shard_{shard:02d}.json", receipt)
    seal = campaign.seal_preflight(tmp_path)
    assert seal["clean_requests"] == 8
    assert campaign._verified_seal(tmp_path, {"plan_sha256": "plan",
                                                 "expected_clean_risk": expected}) == seal
    tampered = tmp_path / "preflight/shard_03.json"
    changed = json.loads(tampered.read_text())
    changed["records"][0]["token_stats_sha256"] = "different"
    tampered.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="not sealed"):
        campaign._verified_seal(tmp_path, {"plan_sha256": "plan",
                                                 "expected_clean_risk": expected})


def test_new_field_seeds_do_not_overlap_failed_wide_run():
    from netsurf import w8a8_wide_ber_campaign as original
    fresh = {campaign.SEED_BASE + arm * 100_000 + i
             for arm in range(len(original.BERS)) for i in range(original.FIELDS_PER_BER)}
    failed = {original.SEED_BASE + arm * 100_000 + i
              for arm in range(len(original.BERS)) for i in range(original.FIELDS_PER_BER)}
    assert len(fresh) == len(failed) == 192
    assert fresh.isdisjoint(failed)
