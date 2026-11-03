"""Small pseudo-network campaign: seal every mask before any paired final field."""

from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from netsurf.manifests import sha256_file
from netsurf.transformer_risk import TaskRisk
from netsurf.w8a8_faults import MSBMask
import netsurf.w8a8_campaign as campaign


def test_all_selection_sealed_before_final_and_immutable_recovery(tmp_path, monkeypatch):
    n = 64
    budgets = {str(f): int(n*f) for f in campaign.INTERIOR}
    plan = {"plan_sha256": "a"*64, "target": "synthetic", "eligible_universe": {"bits": n},
            "interior_budgets_bits": budgets,
            "fields": [{"lambda": lam, "ber": lam/n, "seeds": [300+arm*10+i for i in range(2)]}
                       for arm, lam in enumerate((1, 2, 3))],
            "counts": {"scored_final_units_per_request": 2, "final_fields": 6,
                       "policy_arms_per_field": 26}}
    plan_dir = tmp_path / "plan"; plan_dir.mkdir()
    (plan_dir / "plan.json").write_text(json.dumps(plan))
    root = tmp_path / "run"; (root / "selection/moments").mkdir(parents=True)
    moments = root / "selection/moments/manifest.json"; moments.write_text("{}\n")
    seal = {"status": "ALL_SELECTION_SEALED", "plan_sha256": plan["plan_sha256"],
            "moments_manifest_sha256": sha256_file(moments), "masks": {},
            "msb": {str(k): MSBMask(n, k).digest for k in budgets.values()}}
    for method in campaign.METHODS:
        if method == "msb_first": continue
        seal["masks"][method] = {}
        for budget in budgets.values():
            path = root / "selection/masks" / method / f"b{budget}.bits"
            path.parent.mkdir(parents=True, exist_ok=True)
            value = (1 << budget) - 1
            path.write_bytes(value.to_bytes(n//8, "little"))
            digest = sha256_file(path)
            from netsurf.w8a8_rank import PackedMask
            seal["masks"][method][str(budget)] = {"path": str(path.relative_to(root)),
                                                    "sha256": digest,
                                                    "digest": PackedMask(path,n,budget,digest).digest}
    class FakeAdapter:
        def __init__(self, asset):
            self.universe = SimpleNamespace(bits=n)
            self.active = None
        def load(self): pass
        @contextmanager
        def injected(self, field, mask):
            self.active = (field, mask)
            applied = sum(not mask.contains(x) for x in field.ordinals)
            try:
                yield {"field_sha256": field.sha256, "mask_sha256": mask.digest,
                       "proposed": len(field.ordinals), "applied": applied,
                       "suppressed": len(field.ordinals)-applied,
                       "touched_module_hashes_before": {}}
            finally:
                self.active = None
        def evaluate(self):
            applied = 0 if self.active is None else sum(not self.active[1].contains(x)
                                                       for x in self.active[0].ordinals)
            value = 1. + .1*applied
            return TaskRisk(value,None,None,0,2,((0.,value),(0.,value)),(None,None),
                            "target_logit_and_logsumexp")
    adapter = FakeAdapter(None)
    monkeypatch.setattr(campaign, "validate", lambda *args: (plan, SimpleNamespace()))
    monkeypatch.setattr(campaign, "W8A8FaultAdapter", lambda asset: adapter)
    monkeypatch.setattr(campaign, "_evaluation", lambda *args: adapter.evaluate)
    monkeypatch.setattr(campaign, "FIELDS", 2)
    with pytest.raises(ValueError, match="sealed"):
        campaign.final(tmp_path / "ready", plan_dir, root)
    campaign._write_bound(root / "selection/seal.json", seal)
    result = campaign.final(tmp_path / "ready", plan_dir, root)
    assert result["completed_fields"] == 6 and adapter.active is None
    assert campaign.final(tmp_path / "ready", plan_dir, root)["completed_fields"] == 6
    partial = root / "final/arm_00/field_000.json"
    original = partial.read_bytes(); partial.unlink()
    with pytest.raises(ValueError, match="partial final"):
        campaign.final(tmp_path / "ready", plan_dir, root)
    partial.write_bytes(original)
    analysis = campaign.aggregate(tmp_path / "ready", plan_dir, root)
    assert analysis["fields"] == 6 and analysis["comparisons"] == 60
    exported = campaign.export(tmp_path / "ready", plan_dir, root)
    assert exported["status"] == "FINALIZED"
    assert Path(exported["archive"]).is_file()
    assert campaign.export(tmp_path / "ready", plan_dir, root)["status"] == "VERIFIED_FINALIZED_REUSED"
    path = root / "final/arm_00/field_000.json"
    item = json.loads(path.read_text()); item["policies"]["unprotected"]["risk"]["metrics"]["mean_nll"] += 1
    path.write_text(json.dumps(item))
    with pytest.raises(ValueError, match="checksum"):
        campaign.aggregate(tmp_path / "ready", plan_dir, root)
