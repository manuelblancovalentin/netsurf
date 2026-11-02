import hashlib
import json
import math
import struct

import numpy as np
import pytest

from netsurf.storage_bit_codec import StorageBitCodec
from netsurf.transformer_risk import (
    FAILURE_PENALTY_NATS, classification_risk, language_model_risk,
    language_model_risk_from_stats,
    teacher_forced_windows,
)
from netsurf.torch_storage import expected_faults, sparse_bernoulli_schedule
from netsurf.transformer_baselines import (UnsupportedEstimator, exact_isolated_scores,
    gradient_scores, numerical_displacement_scores, random_scores, select_equal_budget,
    significance_scores)
from netsurf.transformer_assets import AssetBlocked, load_ready_asset
from netsurf.transformer_plan import PlanBlocked, require_final_gate, seal_selection, validate_frozen_plan
from netsurf.transformer_handoff import audit_handoff
from netsurf.manifests import digest_object
from netsurf.transformer_conformance import conformance_cases
from netsurf.transformer_data import Sst2Row, sst2_fault_partitions, sst2_records_digest, wikitext_joined_text


@pytest.mark.parametrize("precision,positive,negative,zero,subnormal,inf,nan", [
    ("FP32", b"\x00\x00\x80\x3f", b"\x00\x00\x80\xbf", b"\x00\x00\x00\x00",
     b"\x01\x00\x00\x00", b"\x00\x00\x80\x7f", b"\x01\x00\xc0\x7f"),
    ("BF16", b"\x80\x3f", b"\x80\xbf", b"\x00\x00",
     b"\x01\x00", b"\x80\x7f", b"\xc1\x7f"),
])
def test_exact_float_storage_categories(precision, positive, negative, zero, subnormal, inf, nan):
    codec = StorageBitCodec(precision)
    assert codec.version == "stored-xor/v1"
    assert codec.fields(positive)["category"] == "normal"
    assert codec.value(positive) == 1.0
    assert codec.value(negative) == -1.0
    assert codec.fields(negative)["sign"] == 1
    assert codec.xor(positive, 0, codec.width - 1) == negative
    assert math.copysign(1, codec.value(codec.xor(zero, 0, codec.width - 1))) == -1
    assert codec.fields(subnormal)["category"] == "subnormal"
    assert codec.fields(inf)["category"] == "infinity"
    assert codec.fields(nan)["category"] == "nan"
    assert codec.finite_displacement(inf, 0) is None
    assert codec.finite_displacement(positive, 0) is not None
    assert codec.xor(codec.xor(positive, 0, 0), 0, 0) == positive
    with pytest.raises(IndexError):
        codec.xor(positive, 0, codec.width)


def test_exponent_flip_and_word_order():
    fp32 = StorageBitCodec("FP32")
    raw = struct.pack("<ff", 1.0, -0.0)
    changed = fp32.xor(raw, 1, 31)
    assert changed[:4] == raw[:4]
    assert fp32.fields(changed[4:])["sign"] == 0
    assert fp32.fields(fp32.xor(raw[:4], 0, 23))["exponent"] == 126


@pytest.mark.parametrize("precision,raw", [
    ("FP32", struct.pack("<f", -0.125)),
    ("BF16", b"\x00\xbe"),
])
def test_conformance_cases_encode_sign_exponent_significand_and_exceptions(precision, raw):
    codec = StorageBitCodec(precision)
    cases = conformance_cases("classifier.weight", 3, raw, codec)
    assert set(cases) == {"sign", "exponent", "significand", "infinity", "nan"}
    for name, events in cases.items():
        changed = raw
        for event in events:
            assert event.parameter == "classifier.weight" and event.flat_index == 3
            changed = codec.xor(changed, 0, event.bit_index)
        if name == "sign":
            assert codec.fields(changed)["sign"] != codec.fields(raw)["sign"]
        elif name == "exponent":
            assert codec.fields(changed)["exponent"] != codec.fields(raw)["exponent"]
        elif name == "significand":
            assert codec.fields(changed)["significand"] != codec.fields(raw)["significand"]
        else:
            assert codec.fields(changed)["category"] == name


def test_task_risk_failure_policy_and_exact_scoring_windows():
    result = classification_risk([(2.0, 0.0), None, (float("nan"), 0.0)], [0, 1, 0])
    assert result.failed_units == 2 and result.total_units == 3
    assert result.accuracy == pytest.approx(1 / 3)
    assert result.sample_count == 3 and result.metrics["failure_fraction"] == pytest.approx(2 / 3)
    assert result.mean_nll >= 2 * FAILURE_PENALTY_NATS / 3
    windows = teacher_forced_windows(list(range(12)), 5, 3)
    assert windows == ((0, 5, 1), (3, 8, 5), (6, 11, 8), (9, 12, 11))
    assert [target for _, end, first in windows for target in range(first, end)] == list(range(1, 12))
    assert any(windows[i][0] < windows[i - 1][1] for i in range(1, len(windows)))
    lm = language_model_risk([(0.0, 1.0), None], [1, 0])
    assert lm.failed_units == 1 and lm.perplexity == pytest.approx(math.exp(lm.mean_nll))
    stats = language_model_risk_from_stats([(1.0, math.log(math.e + 1)), None])
    assert stats.mean_nll == pytest.approx(lm.mean_nll)
    assert stats.logit_record_kind == "target_logit_and_logsumexp"
    with pytest.raises(ValueError):
        classification_risk([(0, 1)], [2])


def test_sst2_fault_cohorts_and_kona_record_digest():
    train = tuple(Sst2Row("train", i, f"train {i}", i % 2) for i in range(16))
    validation = tuple(Sst2Row("validation", i, f"validation {i}", i % 2) for i in range(8))
    parts = sst2_fault_partitions(train, validation, engineering_count=2,
                                  calibration_count=4, selection_count=3)
    assert [row.source_index for row in parts["engineering"]] == [0, 1]
    assert [row.source_index for row in parts["evaluation"]] == list(range(2, 8))
    assert not ({row.source_index for row in parts["calibration"]} &
                {row.source_index for row in parts["selection"]})
    assert parts == sst2_fault_partitions(train, validation, engineering_count=2,
                                           calibration_count=4, selection_count=3)
    expected = hashlib.sha256(b"0\t0\t" + hashlib.sha256(b"validation 0").hexdigest().encode()
                              + b"\n").hexdigest()
    assert sst2_records_digest(validation[:1]) == expected
    assert wikitext_joined_text(["alpha", "", "beta"])[0] == "alpha\n\n\n\nbeta"
    with pytest.raises(ValueError, match="source splits"):
        sst2_fault_partitions(validation, train, engineering_count=2,
                              calibration_count=4, selection_count=3)


def test_expected_faults_and_budget():
    assert expected_faults(100, 25, 0.01) == {"proposed": 1.0, "applied": 0.75, "suppressed": 0.25}
    with pytest.raises(ValueError):
        expected_faults(10, 11, 0.1)


def test_sparse_background_is_exact_bernoulli_and_seeded():
    from netsurf.identities import BitIdentity
    class SmallStorage:
        bit_count = 20
        def bit_at(self, ordinal):
            return BitIdentity("weight", ordinal, 0)
    storage = SmallStorage()
    schedules = [sparse_bernoulli_schedule(storage, ber=0.2, seed=seed,
                                           inference_count=2, max_events=20)
                 for seed in range(1000)]
    assert schedules[7] == sparse_bernoulli_schedule(storage, ber=0.2, seed=7,
                                                     inference_count=2, max_events=20)
    counts = np.array([len(x.events) for x in schedules])
    assert abs(float(counts.mean()) - 4.0) < 0.25
    positions = np.zeros(20)
    for schedule in schedules:
        for event in schedule.events:
            positions[event.bit.flat_index] += 1
    assert np.all(np.abs(positions / 1000 - 0.2) < 0.06)
    with pytest.raises(ValueError, match="cap"):
        sparse_bernoulli_schedule(storage, ber=1, seed=1, inference_count=1, max_events=2)


def test_bounded_baselines_preserve_signed_effects_and_nonfinite_gate():
    from netsurf.identities import BitIdentity
    codec = StorageBitCodec("FP32")
    raw = {"w": struct.pack("<f", 1.0)}
    bits = [BitIdentity("w", 0, 31), BitIdentity("w", 0, 0)]
    assert random_scores(bits, 7) == random_scores(bits, 7)
    assert significance_scores(bits)[bits[0]] > significance_scores(bits)[bits[1]]
    displacement = numerical_displacement_scores(bits, raw, codec)
    assert displacement[bits[0]] > displacement[bits[1]]
    gradients = {"w": np.array([[1.0], [-2.0]])}
    absolute = gradient_scores(bits, raw, codec, gradients, policy="per_example_absolute")
    positive = gradient_scores(bits, raw, codec, gradients, policy="per_example_positive")
    assert absolute[bits[0]] > positive[bits[0]] > 0
    assert gradient_scores(bits, raw, codec, gradients, policy="global_absolute")[bits[0]] > 0
    signed = exact_isolated_scores(bits, 1.0, lambda bit: 0.5 if bit == bits[0] else 1.5)
    assert signed[bits[0]] < 0 < signed[bits[1]]
    assert select_equal_budget(signed, 1) == frozenset((bits[1],))
    with pytest.raises(UnsupportedEstimator, match="nonfinite"):
        numerical_displacement_scores([BitIdentity("w", 0, 30)],
                                      {"w": struct.pack("<f", float("inf"))}, codec)


def test_readiness_rejects_absence_status_and_changed_bytes(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"fixture")
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_bytes(b"tokenizer")
    dataset = tmp_path / "sst2.jsonl"
    dataset.write_text('{"source_index":0,"text":"x","label":0}\n')
    interpreter = tmp_path / "python"
    interpreter.write_bytes(b"interpreter")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    registry = {"entries": [{"id": "distilbert_sst2_fp32", "target_precision": "FP32",
        "model": {"source": "example/model", "revision": "abc123", "files": [
            {"path": "model.safetensors", "sha256": digest(checkpoint)}]}}]}
    portfolio = tmp_path / "portfolio.json"
    portfolio.write_text(json.dumps(registry))
    record = {"target": "distilbert_sst2_fp32", "status": "BLOCKED", "source": "example/model",
        "revision": "abc123", "precision": "FP32", "model_root": str(tmp_path),
        "model_files": [{"path": "model.safetensors", "sha256": digest(checkpoint)}],
        "tokenizer_files": [{"path": "tokenizer.json", "sha256": digest(tokenizer)}],
        "dataset_path": str(dataset), "dataset_sha256": digest(dataset),
        "interpreter": str(interpreter), "interpreter_sha256": digest(interpreter),
        "backend": {"device": "cpu", "attention": "eager", "threads": 1,
                    "python_version": "3.11.17", "torch_version": "2.0", "transformers_version": "4.0"},
        "labels": {"0": "negative", "1": "positive"}}
    manifest = tmp_path / "NETSURF_ASSET_READINESS.json"
    with pytest.raises(AssetBlocked, match="absent"):
        load_ready_asset(manifest, "distilbert_sst2_fp32", portfolio)
    manifest.write_text(json.dumps({"schema_version": "netsurf-transformer-readiness/v1", "records": [record]}))
    with pytest.raises(AssetBlocked, match="READY"):
        load_ready_asset(manifest, "distilbert_sst2_fp32", portfolio)
    record["status"] = "READY"
    manifest.write_text(json.dumps({"schema_version": "netsurf-transformer-readiness/v1", "records": [record]}))
    assert load_ready_asset(manifest, "distilbert_sst2_fp32", portfolio).target == record["target"]
    checkpoint.write_bytes(b"changed")
    with pytest.raises(AssetBlocked, match="changed bytes"):
        load_ready_asset(manifest, "distilbert_sst2_fp32", portfolio)
    manifest.write_text(json.dumps({"schema": "kona-assets/netsurf-asset-readiness/1",
        "targets": [{"target_id": "distilbert_sst2_fp32",
                     "status": "PREPARED_AWAITING_APPROVAL", "blockers": ["acquisition approval"]}]}))
    with pytest.raises(AssetBlocked, match="PREPARED_AWAITING_APPROVAL.*acquisition approval"):
        load_ready_asset(manifest, "distilbert_sst2_fp32", portfolio)


def test_plan_binding_and_all_selection_before_final(tmp_path):
    plan = {"schema_version": "transformer-campaign-plan/v1", "status": "FROZEN",
        "target": "fixture", "source_commit": "a" * 64, "asset_manifest_sha256": "b" * 64,
        "asset_record_sha256": "c" * 64, "eligible_population_sha256": "d" * 64,
        "partitions_sha256": "e" * 64, "eligible_bits": 8, "budgets": [0, 2, 8],
        "ber": [0, 0.01], "methods": ["random", "isolated"],
        "selection": {"seed": 1, "fields": 2, "sample_count": 2},
        "final": {"seed": 2, "fields": 3, "sample_count": 2},
        "failure_penalty_nats": 30.0,
        "analysis": {"independent_unit": "fault_field", "multiple_comparison_procedure": "Holm"}}
    plan["plan_sha256"] = digest_object(plan)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    assert validate_frozen_plan(plan) == plan["plan_sha256"]
    with pytest.raises(FileNotFoundError):
        require_final_gate(path, tmp_path / "seal.json")
    first, second = tmp_path / "random.json", tmp_path / "isolated.json"
    first.write_text("a"); second.write_text("b")
    with pytest.raises(PlanBlocked, match="every"):
        seal_selection(path, {"random": first}, tmp_path / "seal.json")
    seal_selection(path, {"random": first, "isolated": second}, tmp_path / "seal.json")
    assert require_final_gate(path, tmp_path / "seal.json") == plan
    second.write_text("changed")
    with pytest.raises(PlanBlocked, match="changed"):
        require_final_gate(path, tmp_path / "seal.json")
    plan["final"]["seed"] = 1
    plan["plan_sha256"] = digest_object({k: v for k, v in plan.items() if k != "plan_sha256"})
    with pytest.raises(PlanBlocked, match="collision"):
        validate_frozen_plan(plan)


def test_optional_torch_runtime_storage_restore():
    torch = pytest.importorskip("torch")
    from netsurf.faults import FaultEvent, FaultModel, FaultSchedule
    from netsurf.identities import BitIdentity
    from netsurf.protection import ProtectionMask
    from netsurf.torch_storage import TorchStoredLinearWeights

    model = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False)).eval()
    store = TorchStoredLinearWeights(model, "FP32")
    first = BitIdentity("0.weight", 0, 31)
    law = FaultModel("weight", "stored_linear_weight", "persistent_trial", "forced_engineering", 0.0)
    schedule = FaultSchedule(law, 1, (FaultEvent(None, first),), store.bit_count, 1)
    none = ProtectionMask("none", frozenset(), store.bit_count, 0, {})
    full = ProtectionMask("full", frozenset((first,)), store.bit_count, 1, {})
    before = store.hashes()
    x = torch.tensor([[1.0, 0.0]])
    clean = model(x).detach().clone()
    with store.injected(schedule, none) as report:
        assert report.applied_events == 1
        assert store.hashes() != before
        assert not torch.equal(model(x), clean)
    assert store.hashes() == before and torch.equal(model(x), clean)
    with store.injected(schedule, full) as report:
        assert report.suppressed_events == 1
        assert torch.equal(model(x), clean)
    assert store.hashes() == before


def test_optional_tied_output_weight_is_explicitly_excluded():
    torch = pytest.importorskip("torch")
    from netsurf.torch_storage import TorchStoredLinearWeights

    class Tied(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(3, 2)
            self.output = torch.nn.Linear(2, 3, bias=False)
            self.output.weight = self.embedding.weight
            self.projection = torch.nn.Linear(2, 2, bias=False)

    model = Tied().eval()
    store = TorchStoredLinearWeights(model, "FP32")
    assert "output.weight" in store.excluded_tied_parameters
    assert "output.weight" not in store.parameters
    assert set(store.parameters) == {"projection.weight"}


def test_forced_engineering_events_cannot_be_misreported_as_bernoulli_draws():
    from netsurf.faults import FaultModel, FaultSchedule
    from netsurf.identities import BitIdentity
    model = FaultModel("weight", "stored_linear_weight", "persistent_trial",
                       "forced_engineering", 0.0)
    with pytest.raises(ValueError, match="cannot be sampled"):
        FaultSchedule.generate(model, [BitIdentity("w", 0, 0)], seed=1, inference_count=1)


def test_readonly_handoff_audit_binds_all_eighteen_targets(tmp_path):
    registry = {"entries": [{"id": f"t{i}", "model": {"source": "source", "revision": "r"}}
                            for i in range(18)]}
    handoff = {"schema": "kona-assets/netsurf-asset-readiness/1",
               "targets": [{"target_id": f"t{i}", "status": "READY" if i == 0 else "MISSING_ASSET",
                            "model": {"repository": "source", "revision": "r"},
                            "interpreter": str(tmp_path / "absent"), "blockers": []}
                           for i in range(18)]}
    portfolio, manifest, output = (tmp_path / name for name in ("registry.json", "readiness.json", "audit.json"))
    portfolio.write_text(json.dumps(registry)); manifest.write_text(json.dumps(handoff))
    result = audit_handoff(manifest, portfolio, output)
    assert len(result["targets"]) == 18
    assert result["targets"][0]["asset_status"] == "READY"
    assert not result["targets"][0]["interpreter_exists"]
    assert audit_handoff(manifest, portfolio, output) == result
    handoff["targets"][0]["model"]["revision"] = "other"
    manifest.write_text(json.dumps(handoff))
    with pytest.raises(ValueError, match="write-once"):
        audit_handoff(manifest, portfolio, output)
