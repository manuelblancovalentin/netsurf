from __future__ import annotations

from pathlib import Path
from dataclasses import replace
import itertools
import json

import numpy as np
import pytest

from netsurf.adapters.mlperf_tiny import ClassificationEvaluation
from netsurf.conditional_config import (
    ConditionalBackgrounds, ConditionalConfiguration, ConditionalExecution,
    ConditionalSamples, RestrictedOracleConfiguration,
)
from netsurf.conditional_bundle import create_conditional_bundle, verify_conditional_bundle
from netsurf.conditional_damage import (
    exact_bernoulli_risk, exact_conditional_damage, execute_conditional_damage,
)
from netsurf.experiment_config import BenchmarkConfiguration, OutputConfiguration
from netsurf.faults import InjectionReport
from netsurf.faults import FaultEvent, FaultModel, FaultSchedule
from netsurf.identities import BitIdentity
from netsurf.samples import SamplePartition, sample_identity_digest
from netsurf.result_bundle import ResultBundleError
from netsurf.manifests import sha256_file
from netsurf.protection_context import ProtectionContextConfiguration, ProtectionContextError
from netsurf.protection import ProtectionMask
from netsurf.restricted_subcube import integrate_response_table, subcube_states


def test_three_way_product_has_zero_isolated_and_pair_clean_but_nonzero_conditional_damage():
    def risk(faults):
        state = set(faults)
        return float(all(index in state for index in (0, 1, 2)))

    assert risk((0,)) - risk(()) == 0.0
    assert risk((0, 1)) - risk((1,)) == 0.0
    for candidate in range(3):
        assert np.isclose(exact_conditional_damage(risk, 3, candidate, 0.2), 0.2**2)


def test_mask_switch_oracle_changes_best_protected_bit_at_one_fifteenth():
    def risk(faults):
        z1, z2, z3 = (float(index in faults) for index in range(3))
        return 1.2 * z1 + z2 + 0.1 * z3 + 3.0 * z2 * z3

    def best(probability):
        values = {
            bit: exact_bernoulli_risk(risk, 3, probability, protected=frozenset({bit}))
            for bit in range(3)
        }
        return min(values, key=values.get)

    assert best(0.05) == 0
    assert best(0.10) == 1


class _FakeExactAdapter:
    def __init__(self):
        self._values = {"kernel": np.asarray([0, -1], dtype=np.int8)}
        self._bits = tuple(
            BitIdentity("kernel", word, bit) for word in range(2) for bit in range(8)
        )

    def load(self):
        return None

    def sample_partition(self, role, count, *, offset=0):
        bases = {
            "conditional_calibration": 1000,
            "conditional_model_selection": 2024,
            "conditional_final_evaluation": 3048,
        }
        split = "train" if role == "calibration" else "test"
        original_offset = bases.get(role, offset)
        indices = tuple(range(original_offset, original_offset + count))
        labels = np.zeros(count, dtype=np.int64)
        return (
            np.zeros((count, 1), dtype=np.int8), labels,
            SamplePartition(role, split, indices, sample_identity_digest(split, indices, labels), "test"),
        )

    def bit_identities(self):
        return self._bits

    def values(self):
        return {name: value.copy() for name, value in self._values.items()}

    def encoded_bit_transition(self, identity):
        original = int(self._values[identity.parameter].reshape(-1)[identity.flat_index])
        flipped = int(np.asarray(original, dtype=np.int8).view(np.uint8) ^ np.uint8(1 << identity.bit_index))
        flipped = int(np.asarray(flipped, dtype=np.uint8).view(np.int8))
        changed = self.values()
        changed[identity.parameter].reshape(-1)[identity.flat_index] = flipped
        return original, flipped, float(original), float(flipped), changed

    def evaluate_fault_schedule(self, inputs, labels, schedule, protection):
        del inputs, labels
        applied = [event.bit for event in schedule.events if event.bit not in protection.protected]
        # Includes a sign-bit interaction so conditional and isolated effects can differ.
        keys = set(applied)
        risk = 0.1 * len(keys)
        risk += 1.0 * (
            BitIdentity("kernel", 0, 7) in keys and BitIdentity("kernel", 1, 7) in keys
        )
        report = InjectionReport(
            schedule.digest(), len(schedule.events), len(applied),
            len(schedule.events) - len(applied), len(keys),
        )
        return ClassificationEvaluation(1.0 - min(risk, 1.0), risk, len(applied) + 1), report

    def representation_manifest(self):
        return {"storage": "test int8", "exact_xor": True}


def _configuration(tmp_path: Path, probability: float) -> ConditionalConfiguration:
    checkpoint = tmp_path / "model.tflite"
    dataset = tmp_path / "dataset"
    lock = tmp_path / "lock.json"
    checkpoint.write_bytes(b"model")
    dataset.mkdir()
    lock.write_text("{}", encoding="utf-8")
    benchmark = BenchmarkConfiguration(
        "mlperf_tiny_kws_int8_v1", "fake/kws", checkpoint, dataset,
        "official_subset_1000", 0, 0, "int8_tflite_affine",
        "tflite_affine_integer_v1", ("kernels",), lock,
        "tensorflow_lite_builtin_ref_v1",
    )
    candidates = (BitIdentity("kernel", 0, 7), BitIdentity("kernel", 1, 7))
    return ConditionalConfiguration(
        "6.0.0", "exact-conditional-damage/v1", "test", "test", "test",
        benchmark, ConditionalSamples(2, 2, 0, 0, True),
        "categorical_crossentropy", "minimize", candidates,
        ConditionalBackgrounds((probability,), 4, 5, 101, 202),
        ConditionalExecution(0.95),
        RestrictedOracleConfiguration(False, 1, 1, 1, 303, 404),
        OutputConfiguration(tmp_path / "runs", True, False),
        tmp_path / "config.toml", "schema_version = \"6.0.0\"\n",
    )


def _oracle_configuration(tmp_path: Path, probability: float) -> ConditionalConfiguration:
    configuration = _configuration(tmp_path, probability)
    return ConditionalConfiguration(
        "6.1.0", "exact-conditional-protection/v2", "oracle-test", "test", "test",
        BenchmarkConfiguration(
            configuration.benchmark.adapter, configuration.benchmark.id,
            configuration.benchmark.checkpoint, configuration.benchmark.dataset,
            "full_test_4890", 0, 0, configuration.benchmark.quantization,
            configuration.benchmark.quantization_backend,
            configuration.benchmark.injectable_parameters,
            configuration.benchmark.asset_lock, configuration.benchmark.inference_backend,
        ),
        ConditionalSamples(
            2, 2, 0, 0, True,
            "conditional_calibration", "conditional_model_selection", 2, 0,
            "conditional_final_evaluation",
        ),
        configuration.risk_metric, configuration.risk_direction,
        configuration.candidates, configuration.backgrounds, configuration.execution,
        RestrictedOracleConfiguration(True, 1, 2, 3, 303, 404),
        configuration.output, configuration.source_path,
        "schema_version = \"6.1.0\"\n",
    )


def test_zero_ber_equals_isolated_and_common_pairing_excludes_candidate(tmp_path):
    adapter = _FakeExactAdapter()
    before = adapter.values()
    outcome = execute_conditional_damage(_configuration(tmp_path, 0.0), adapter=adapter)
    assert all(row["conditional_damage"] == row["isolated_damage"] for row in outcome.measurements)
    assert all(not row["master_selected_candidate"] for row in outcome.measurements)
    assert all(row["background_proposed_events"] == 0 for row in outcome.measurements)
    assert all(row["candidate_proposed_events"] == 1 for row in outcome.measurements)
    assert outcome.scientific_payload["checks"]["state_restored"] is True
    for name in before:
        np.testing.assert_array_equal(adapter.values()[name], before[name])


def test_conditional_execution_is_deterministic_and_uses_independent_split_seeds(tmp_path):
    configuration = _configuration(tmp_path, 0.25)
    first = execute_conditional_damage(configuration, adapter=_FakeExactAdapter())
    second = execute_conditional_damage(configuration, adapter=_FakeExactAdapter())
    assert first.scientific_payload_sha256 == second.scientific_payload_sha256
    assert first.measurements == second.measurements
    calibration = {row["master_sha256"] for row in first.background_records if row["split"] == "calibration"}
    evaluation = {row["master_sha256"] for row in first.background_records if row["split"] == "held_out_evaluation"}
    assert calibration != evaluation


def test_int8_transition_uses_exact_twos_complement_xor():
    adapter = _FakeExactAdapter()
    assert adapter.encoded_bit_transition(BitIdentity("kernel", 0, 7))[:2] == (0, -128)
    assert adapter.encoded_bit_transition(BitIdentity("kernel", 1, 0))[:2] == (-1, -2)


def test_conditional_bundle_is_finalized_and_checksum_verified(tmp_path):
    outcome = execute_conditional_damage(_configuration(tmp_path, 0.0), adapter=_FakeExactAdapter())
    destination = create_conditional_bundle(
        outcome, repository_root=Path(__file__).resolve().parents[1], run_id="bounded-test"
    )
    manifest = verify_conditional_bundle(destination)
    assert manifest["scientific_payload_sha256"] == outcome.scientific_payload_sha256
    assert (destination / "measurements/conditional_damage.csv").is_file()
    assert (destination / "artifacts/background_fields.json").is_file()
    assert (destination / "report.md").is_file()
    with np.testing.assert_raises(ResultBundleError):
        create_conditional_bundle(
            outcome, repository_root=Path(__file__).resolve().parents[1], run_id="bounded-test"
        )


def test_restricted_oracle_execution_is_paired_deterministic_and_bundled(tmp_path):
    configuration = _oracle_configuration(tmp_path, 0.25)
    first = execute_conditional_damage(configuration, adapter=_FakeExactAdapter())
    second = execute_conditional_damage(configuration, adapter=_FakeExactAdapter())
    assert first.scientific_payload_sha256 == second.scientific_payload_sha256
    assert len(first.oracle_summaries) == 1
    summary = first.oracle_summaries[0]
    assert summary["requested_state_evaluations"] == 3 + (2 + 3) * 4
    assert summary["posthoc_held_out_optimum_is_selection_valid"] is False
    assert len(first.oracle_response_records) == summary["requested_state_evaluations"]
    assert first.scientific_payload["checks"]["state_restored"] is True
    identities = first.scientific_payload["sample_identities"]
    assert {row["original_index"] for row in identities["calibration"]}.isdisjoint(
        row["original_index"] for row in identities["model_selection"]
    )
    destination = create_conditional_bundle(
        first, repository_root=Path(__file__).resolve().parents[1], run_id="oracle-test"
    )
    verify_conditional_bundle(destination)
    assert (destination / "measurements/restricted_oracle_responses.csv").is_file()
    assert (destination / "artifacts/restricted_oracle_masks.json").is_file()


def _context_source(tmp_path, configuration):
    root = tmp_path / "source-run"
    artifact = root / "artifacts/protection_masks.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(json.dumps({"replication_0": {"s5@0.125": [
        ["kernel", 0, 0], ["kernel", 0, 7],
    ]}}), encoding="utf-8")
    payload = "a" * 64
    (root / "manifest.json").write_text(json.dumps({
        "status": "complete", "scientific_payload_sha256": payload,
        "benchmark": {"checkpoint_sha256": sha256_file(configuration.benchmark.checkpoint)},
    }), encoding="utf-8")
    artifact_sha = sha256_file(artifact)
    (root / "checksums.json").write_text(json.dumps({
        "files": {"artifacts/protection_masks.json": artifact_sha},
    }), encoding="utf-8")
    (root / "FINALIZED").write_text(
        f"scientific_payload_sha256={payload}\n", encoding="ascii"
    )
    return ProtectionContextConfiguration(
        "verified_global_mask", root, "artifacts/protection_masks.json",
        artifact_sha, payload, "replication_0/s5@0.125", 16, 2,
    )


def test_contextual_oracle_preserves_global_budget_and_reuses_response_table(tmp_path):
    base = _oracle_configuration(tmp_path, 0.25)
    configuration = replace(
        base, schema_version="6.2.0",
        scientific_protocol="contextual-exact-conditional-protection/v3",
        protection_context=_context_source(tmp_path, base),
    )
    outcome = execute_conditional_damage(configuration, adapter=_FakeExactAdapter())
    summary = outcome.oracle_summaries[0]
    assert summary["original_mask_indices"] == [0]
    assert summary["global_protection_cardinality"] == 2
    assert summary["global_protection_cardinality_preserved"] is True
    assert summary["held_out_selected_minus_original"]["count"] == 3
    assert len(outcome.oracle_derived_records) == 5
    assert all(row["pairing_key"] for row in outcome.oracle_derived_records)
    assert outcome.scientific_payload["checks"]["global_cardinality_preserved"] is True
    destination = create_conditional_bundle(
        outcome, repository_root=Path(__file__).resolve().parents[1], run_id="context-test"
    )
    verify_conditional_bundle(destination)
    assert (destination / "artifacts/protection_context.json").is_file()
    assert (destination / "artifacts/restricted_oracle_derived.json").is_file()


def test_contextual_oracle_rejects_tampered_finalized_mask(tmp_path):
    base = _oracle_configuration(tmp_path, 0.25)
    context = _context_source(tmp_path, base)
    (context.source_bundle / context.artifact).write_text("[]", encoding="utf-8")
    configuration = replace(
        base, schema_version="6.2.0",
        scientific_protocol="contextual-exact-conditional-protection/v3",
        protection_context=context,
    )
    with pytest.raises(ProtectionContextError, match="checksum mismatch"):
        execute_conditional_damage(configuration, adapter=_FakeExactAdapter())


def test_context_response_table_matches_direct_protected_fault_evaluation():
    """Table integration is identical to paired event suppression for every mask."""

    adapter = _FakeExactAdapter()
    candidates = (
        BitIdentity("kernel", 0, 7),
        BitIdentity("kernel", 1, 7),
        BitIdentity("kernel", 1, 0),
    )
    outside = BitIdentity("kernel", 0, 0)
    model = FaultModel(
        "weight", "encoded_parameter_storage", "persistent_trial",
        "independent_bernoulli", 0.25,
    )
    outside_events = (FaultEvent(None, outside),)
    base_protection = frozenset({outside})

    def evaluate(candidate_state, protected):
        schedule = FaultSchedule(
            model, 17,
            outside_events + tuple(FaultEvent(None, candidates[index]) for index in candidate_state),
            len(candidates) + 1, 1,
        )
        evaluation, report = adapter.evaluate_fault_schedule(
            np.zeros((1, 1), dtype=np.int8), np.zeros(1, dtype=np.int64), schedule,
            ProtectionMask("test", frozenset(protected), 16, float(len(protected)), {}),
        )
        return evaluation.categorical_crossentropy, report

    response_table = {
        state: evaluate(state, base_protection)[0] for state in subcube_states(len(candidates))
    }
    probability = 0.25
    for protected_indices in ((0,), (1,), (2,)):
        table_risk = integrate_response_table(
            response_table, len(candidates), probability, protected=protected_indices
        )
        available = tuple(index for index in range(len(candidates)) if index not in protected_indices)
        direct_risk = 0.0
        suppressed_events = 0
        for enabled in itertools.product((0, 1), repeat=len(available)):
            state = tuple(available[index] for index, present in enumerate(enabled) if present)
            count = len(state)
            mass = probability**count * (1.0 - probability) ** (len(available) - count)
            loss, report = evaluate(state, base_protection | {candidates[index] for index in protected_indices})
            direct_risk += mass * loss
            suppressed_events += report.suppressed_events
        assert table_risk == pytest.approx(direct_risk)
        assert suppressed_events >= 1  # the fixed outside-context bit is always suppressed
