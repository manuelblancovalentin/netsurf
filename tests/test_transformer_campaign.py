"""Small synthetic campaign receipts; never load real model or reserved inputs."""

from dataclasses import asdict
import json
from pathlib import Path

import pytest

from netsurf.faults import InjectionReport
from netsurf.manifests import digest_object, sha256_file
from netsurf.torch_storage import expected_faults, sparse_bernoulli_schedule
from netsurf.transformer_campaign import (_bound, _member_path, _policy_masks,
    _selection_artifact, _write_json, aggregate, targeted_classifier_bits, SOURCE_FILES)
from netsurf.transformer_plan import PlanBlocked, seal_selection


def test_offline_aggregation_replays_every_field_and_rejects_changed_schedule(tmp_path):
    root = Path(__file__).resolve().parents[1]
    plan = json.loads((root / "config/experiments/distilbert_sst2_classifier1024_v3.json").read_text())
    plan["source_files_sha256"] = {name: sha256_file(root / name) for name in SOURCE_FILES}
    plan["plan_sha256"] = digest_object({key: value for key, value in plan.items() if key != "plan_sha256"})
    plan_path = tmp_path / "plan.json"
    _write_json(plan_path, plan)
    bits = targeted_classifier_bits()
    run = tmp_path / "run"
    selections = {}
    for method in plan["methods"]:
        scores = {bit: float(index) for index, bit in enumerate(bits)}
        path = run / "selection" / f"{method}.json"
        _write_json(path, _selection_artifact(plan, method, scores, bits))
        selections[method] = path
    seal_selection(plan_path, selections, run / "selection" / "seal.json")
    policies = _policy_masks(plan, run, bits)
    base_risk = {"mean_nll": 0.4, "accuracy": 0.8, "failed_units": 0,
                 "total_units": 256}
    _write_json(run / "final" / "clean.json", _bound({"plan_sha256": plan["plan_sha256"],
        "risk": base_risk}))

    class Universe:
        bit_count = 1024
        def bit_at(self, ordinal):
            return bits[ordinal]

    for arm, ber in enumerate(plan["ber"][1:], start=1):
        for field in range(plan["final"]["fields"]):
            seed = plan["final"]["seed"] + arm * 100000 + field
            schedule = sparse_bernoulli_schedule(Universe(), ber=ber, seed=seed,
                inference_count=256, max_events=128)
            rows = {}
            for name, mask in policies.items():
                applied = sum(event.bit not in mask.protected for event in schedule.events)
                report = InjectionReport(schedule.digest(), len(schedule.events), applied,
                                         len(schedule.events) - applied, applied)
                rows[name] = {"risk": base_risk, "report": asdict(report),
                    "mask_sha256": mask.digest(),
                    "expected_faults": expected_faults(1024, mask.budget, ber)}
            member = _bound({"plan_sha256": plan["plan_sha256"], "arm": arm,
                "ber": ber, "field": field, "seed": seed,
                "schedule_sha256": schedule.digest(),
                "schedule_events": [event.key() for event in schedule.events],
                "policies": rows})
            _write_json(_member_path(run, arm, field), member)
    report = aggregate(plan_path, run)
    assert len(report["comparisons"]) == 12
    assert len(report["rows"]) == 24
    assert all(item["mean_baseline_minus_isolated"] == 0 for item in report["comparisons"])
    first = _member_path(run, 1, 0)
    tampered = json.loads(first.read_text())
    tampered["schedule_events"] = ["persistent:classifier.weight[0]#b0", "forged"]
    tampered["payload_sha256"] = digest_object({k: v for k, v in tampered.items() if k != "payload_sha256"})
    first.write_text(json.dumps(tampered))
    with pytest.raises(PlanBlocked, match="event schedule"):
        aggregate(plan_path, run)
