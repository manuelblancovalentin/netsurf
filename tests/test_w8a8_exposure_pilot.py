"""Small deterministic tests; never use reserved pilot or final model inputs."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from netsurf.manifests import canonical_json_bytes, digest_object, sha256_file
from netsurf.w8a8_exposure_pilot import (
    FIELDS_PER_ARM, LAMBDAS, PILOT_INDICES, RESERVED_FINAL_INDICES,
    SEED_BASE, _field, validate,
)
from netsurf.w8a8_faults import bernoulli_field


def test_prespecified_exposure_grid_and_disjoint_inputs():
    assert LAMBDAS == (1024, 4096, 16384, 65536)
    assert FIELDS_PER_ARM == 16
    assert len(PILOT_INDICES) == 128 and len(RESERVED_FINAL_INDICES) == 200
    assert set(PILOT_INDICES).isdisjoint(RESERVED_FINAL_INDICES)
    assert min(PILOT_INDICES) == 544 and max(RESERVED_FINAL_INDICES) == 871
    assert all(32 <= x <= 543 for x in range(32, 544))


def test_schedule_identity_and_tampering():
    bits, lam = 4096, 64
    field = bernoulli_field(bits, lam/bits, SEED_BASE, max_events=256)
    plan = {"eligible_universe": {"bits": bits}, "max_events_per_field": 256,
            "arms": [{"ber": lam/bits, "fields": [{"index": 0, "seed": SEED_BASE,
                "sha256": field.sha256, "proposed": len(field.ordinals)}]}]}
    assert _field(plan, 0, 0) == field
    plan["arms"][0]["fields"][0]["sha256"] = "0"*64
    with pytest.raises(ValueError, match="schedule changed"):
        _field(plan, 0, 0)


def test_target_specific_ready_update_allowed_but_identity_change_rejected(tmp_path, monkeypatch):
    from netsurf import w8a8_exposure_pilot as pilot
    record = {"target_id": "distilbert_sst2_int8", "checkpoint": "abc"}
    quant = {"tensors_digest": "def"}
    original = {"schema": "test", "target_record_sha256": digest_object(record),
                "ready_manifest_sha256_at_freeze": "old_global_hash"}
    original["plan_sha256"] = digest_object(original)
    plan_dir = tmp_path / "frozen"
    plan_dir.mkdir()
    (plan_dir / "plan.json").write_bytes(canonical_json_bytes(original) + b"\n")
    (plan_dir / "FROZEN").write_text(sha256_file(plan_dir / "plan.json") + "\n")
    (plan_dir / "target_ready_record.json").write_text(json.dumps(record))
    (plan_dir / "quantization_manifest.json").write_text(json.dumps(quant))

    expected = dict(original, ready_manifest_sha256_at_freeze="new_global_hash")
    expected["plan_sha256"] = digest_object({k: v for k, v in expected.items() if k != "plan_sha256"})
    monkeypatch.setattr(pilot, "_expected_plan", lambda _: (expected, SimpleNamespace(record=record, quantization=quant)))
    assert validate(Path("unused"), plan_dir)[0] == original

    changed = dict(record, checkpoint="changed")
    monkeypatch.setattr(pilot, "_expected_plan", lambda _: (expected, SimpleNamespace(record=changed, quantization=quant)))
    with pytest.raises(ValueError, match="identity changed"):
        validate(Path("unused"), plan_dir)


def test_independent_seed_grid_and_ber_normalization():
    bits = 344469504
    seeds = [SEED_BASE + 100000*arm + i for arm in range(len(LAMBDAS)) for i in range(FIELDS_PER_ARM)]
    assert len(seeds) == len(set(seeds)) == 64
    assert all((lam/bits)*bits == lam for lam in LAMBDAS)
    assert (LAMBDAS[0]/bits) < (LAMBDAS[-1]/bits) < 1


def test_resource_gate_reserves_entire_pinned_affinity():
    from scripts.w8a8_exposure_pilot_gate_v2 import _occupied_physical, _parse_affinity
    assert _parse_affinity("2,4-8") == {2, 4, 5, 6, 7, 8}
    physical = lambda cpu: (0, cpu % 40)
    pinned = {"affinity": "2,4,6,8", "last_cpu": 8}
    assert _occupied_physical(pinned, physical) == {(0, 2), (0, 4), (0, 6), (0, 8)}
    broad = {"affinity": "0-79", "last_cpu": 15}
    assert _occupied_physical(broad, physical) == {(0, 15)}
