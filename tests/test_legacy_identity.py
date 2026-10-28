import hashlib
import json

import numpy as np

from netsurf.__main__ import main
from netsurf.adapters.legacy_mnist import (
    LegacyMnistLiteAdapter,
    MnistLiteExactKernelBiasAdapter,
    _state_sha256,
)
from netsurf.legacy_identity import (
    IDENTITY_NAME,
    codec_conformance,
    fault_representation_comparison,
    fault_schedule_accounting_conformance,
    fold_alpha_beta_state,
    load_benchmark_identity,
    load_identity_manifest,
    parameter_quantization_statistics,
    verify_identity_files,
)
from netsurf.quantization import BitCodec, FixedPointFormat
from netsurf.samples import require_disjoint


def codec():
    return BitCodec(FixedPointFormat.parse("q<6,2,1>"))


def test_versioned_identity_population_accounting():
    identity = load_identity_manifest()
    assert identity["identity"] == IDENTITY_NAME
    tensors = identity["trainable_tensors"]
    populations = identity["populations"]
    assert sum(tensor["scalars"] for tensor in tensors) == 7_980
    assert sum(tensor["nonzero_scalars"] for tensor in tensors) == 7_845
    assert populations["allocated_trainable_scalars"] == 7_980
    assert populations["historical_all_trainable_logical_bits"] == 7_980 * 6
    assert populations["exact_q621_kernel_bias_logical_bits"] == 7_960 * 6
    assert populations["zero_valued_allocated_kernel_bias_scalars"] == 135
    assert populations["physically_stored_scalars_in_deployment"] is None
    assert "pending" in populations["unsupported_population"]


def test_identity_cli_is_side_effect_free_and_machine_readable(capsys):
    assert main(["benchmarks", "identity", "legacy_mnist_lite"]) == 0
    observed = json.loads(capsys.readouterr().out)
    assert observed["schema_version"] == "1.0.0"
    assert observed["artifacts"]["checkpoint"]["sha256"].startswith("ba48aee8")


def test_exact_benchmark_identity_is_versioned_and_scientifically_scoped(capsys):
    identity = load_benchmark_identity("mnist_lite_exact_kb_v1")
    assert identity["status"] == "accepted_bounded_inference_benchmark"
    assert identity["source"]["checkpoint_sha256"] == (
        "ba48aee8adbe7d04092d74f41958ec99112ae3e5f88ed56552ea9d1c4dba3006"
    )
    representation = identity["derived_representation"]
    assert representation["eligible_scalars"] == 7_960
    assert representation["eligible_bits"] == 47_760
    assert representation["uncorrupted_float_auxiliary_scalars"] == 20
    assert "not a reproduction" in identity["scientific_scope"]

    assert main(["benchmarks", "identity", "mnist_lite_exact_kb_v1"]) == 0
    observed = json.loads(capsys.readouterr().out)
    assert observed["identity"] == "mnist_lite_exact_kb_v1"


def test_candidate_identities_record_blockers_without_registering_false_readiness():
    keyword = load_benchmark_identity("keyword_spotting_candidate")
    autoencoder = load_benchmark_identity("toyadmos_autoencoder_candidate")
    assert keyword["checkpoint"]["output_logits"] == 49
    assert keyword["available_dataset"]["classes"] == 12
    assert keyword["status"].startswith("blocked_")
    assert autoencoder["status"] == "traceable_but_not_onboarded"
    assert autoencoder["checkpoint"]["kernel_bias_scalars"] == 7_568
    assert "MSE is reconstruction error" in autoencoder["bounded_256_anomalous_validation"]["note"]


def test_exact_identity_file_verifier_uses_derived_manifest_records(tmp_path):
    checkpoint = tmp_path / "checkpoint.keras"
    dataset = tmp_path / "mnist.npz"
    checkpoint.write_bytes(b"checkpoint")
    dataset.write_bytes(b"dataset")
    document = {
        "identity": "mnist_lite_exact_kb_v1",
        "source": {
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(b"checkpoint").hexdigest(),
            "dataset_path": str(dataset),
            "dataset_sha256": hashlib.sha256(b"dataset").hexdigest(),
        },
    }
    result = verify_identity_files(document)
    assert result["all_match"] is True


def test_quantization_statistics_keep_tensor_roles_separate():
    values = {
        "fc0/kernel:0": np.array([0.0, 0.125, 0.01], dtype=np.float32),
        "alpha0/alpha:0": np.array([0.006], dtype=np.float32),
    }
    result = parameter_quantization_statistics(values, codec())
    assert result["tensors"]["fc0/kernel:0"]["role"] == "kernel"
    assert result["tensors"]["alpha0/alpha:0"]["role"] == "alpha"
    assert result["totals"]["scalars"] == 4
    assert result["totals"]["on_lattice_scalars"] == 2


def test_exact_adapter_population_quantizes_only_kernel_bias_and_keeps_zeros():
    adapter = object.__new__(LegacyMnistLiteAdapter)
    adapter.codec = codec()
    source = {
        "fc0/kernel:0": np.array([0.0, 0.01], dtype=np.float32),
        "fc0/bias:0": np.array([0.06], dtype=np.float32),
        "alpha0/alpha:0": np.array([0.006], dtype=np.float32),
        "alpha0/beta:0": np.array([0.031], dtype=np.float32),
    }
    adapter.values = lambda: {name: value.copy() for name, value in source.items()}
    converted = adapter.exact_kernel_bias_values()
    np.testing.assert_array_equal(converted["fc0/kernel:0"], np.array([0.0, 0.0]))
    np.testing.assert_array_equal(converted["fc0/bias:0"], np.array([0.0]))
    np.testing.assert_array_equal(converted["alpha0/alpha:0"], source["alpha0/alpha:0"])
    np.testing.assert_array_equal(converted["alpha0/beta:0"], source["alpha0/beta:0"])
    identities = adapter.exact_kernel_bias_bit_identities()
    assert len(identities) == 18
    assert any(bit.parameter == "fc0/kernel:0" and bit.flat_index == 0 for bit in identities)
    assert all("alpha0" not in bit.parameter for bit in identities)


def test_v2_exact_adapter_exposes_only_exact_words_and_stable_state_identity():
    source = {
        "fc0/kernel:0": np.array([0.0, 0.125], dtype=np.float32),
        "fc0/bias:0": np.array([0.0], dtype=np.float32),
        "alpha0/alpha:0": np.array([0.006], dtype=np.float32),
        "alpha0/beta:0": np.array([0.031], dtype=np.float32),
    }
    adapter = object.__new__(MnistLiteExactKernelBiasAdapter)
    adapter.codec = codec()
    adapter.values = lambda: {name: value.copy() for name, value in source.items()}
    adapter._require_loaded = lambda: None
    adapter._converted_state_sha256 = _state_sha256(source)

    assert [spec.name for spec in adapter.parameter_specs()] == [
        "fc0/kernel:0",
        "fc0/bias:0",
    ]
    assert len(adapter.bit_identities()) == 18
    manifest = adapter.representation_manifest()
    assert manifest["eligible_scalars"] == 3
    assert manifest["eligible_bits"] == 18
    assert manifest["uncorrupted_float_auxiliary_scalars"] == 2
    assert manifest["zero_valued_allocated_words_included"] is True

    reverse_order = dict(reversed(list(source.items())))
    assert _state_sha256(reverse_order) == _state_sha256(source)
    changed = {name: value.copy() for name, value in source.items()}
    changed["fc0/kernel:0"][0] = 0.125
    assert _state_sha256(changed) != _state_sha256(source)


def test_v2_exact_adapter_rejects_identity_changing_assets_before_framework_import(tmp_path):
    checkpoint = tmp_path / "checkpoint.keras"
    dataset = tmp_path / "mnist.npz"
    checkpoint.write_bytes(b"different checkpoint")
    dataset.write_bytes(b"different dataset")
    adapter = MnistLiteExactKernelBiasAdapter(checkpoint, dataset, codec())
    try:
        adapter.load()
    except ValueError as error:
        assert "checkpoint identity mismatch" in str(error)
    else:
        raise AssertionError("an identity-changing checkpoint was accepted")


def test_legacy_analysis_partitions_are_explicit_and_disjoint():
    adapter = object.__new__(LegacyMnistLiteAdapter)
    inputs = np.arange(60_000, dtype=np.float32).reshape(-1, 1)
    labels = np.arange(60_000, dtype=np.int64) % 10
    adapter._mnist_split = lambda _split: (inputs, labels)
    ranking_x, _, ranking = adapter.sample_partition("ranking", 3)
    validation_x, _, validation = adapter.sample_partition("empirical_validation", 3)
    evaluation_x, _, evaluation = adapter.sample_partition("evaluation", 3)
    np.testing.assert_array_equal(ranking_x[:, 0], [50_000, 50_001, 50_002])
    np.testing.assert_array_equal(validation_x[:, 0], [40_000, 40_001, 40_002])
    np.testing.assert_array_equal(evaluation_x[:, 0], [0, 1, 2])
    require_disjoint((ranking, validation, evaluation))


def test_alpha_beta_folding_is_algebraically_equivalent_before_quantization():
    values = {
        "fc0/kernel:0": np.array([[1.0, -0.5], [0.25, 2.0]], dtype=np.float32),
        "fc0/bias:0": np.array([0.5, -0.25], dtype=np.float32),
        "alpha0/alpha:0": np.array([0.25, -0.5], dtype=np.float32),
        "alpha0/beta:0": np.array([0.125, 0.75], dtype=np.float32),
        "fc_out/kernel:0": np.eye(2, dtype=np.float32),
        "fc_out/bias:0": np.zeros(2, dtype=np.float32),
    }
    inputs = np.array([[1.0, -2.0]], dtype=np.float32)
    original_hidden = (
        inputs @ values["fc0/kernel:0"] + values["fc0/bias:0"]
    ) * values["alpha0/alpha:0"] + values["alpha0/beta:0"]
    folded = fold_alpha_beta_state(values)
    folded_hidden = (
        inputs @ folded["fc0/kernel:0"] + folded["fc0/bias:0"]
    ) * folded["alpha0/alpha:0"] + folded["alpha0/beta:0"]
    np.testing.assert_allclose(folded_hidden, original_hidden, rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(folded["alpha0/alpha:0"], np.ones(2))
    np.testing.assert_array_equal(folded["alpha0/beta:0"], np.zeros(2))


def test_codec_gate_exhausts_codes_single_flips_and_two_bit_composition():
    result = codec_conformance(codec())
    assert result == {
        "code_values": 64,
        "round_trip_exact": True,
        "single_bit_xor_checks": 384,
        "two_distinct_bit_xor_checks": 960,
    }


def test_fault_representation_reports_off_lattice_residual():
    values = {
        "fc0/kernel:0": np.array([0.0, 0.01], dtype=np.float32),
        "alpha0/alpha:0": np.array([0.006], dtype=np.float32),
    }
    result = fault_representation_comparison(values, codec())
    assert result["total_scalar_bit_comparisons"] == 12
    assert result["numerically_equal"] == 6
    assert result["different"] == 6
    np.testing.assert_allclose(result["maximum_absolute_state_difference"], 0.01)


def test_fault_schedule_gate_counts_zero_storage_and_protection_bounds():
    values = {
        "fc0/kernel:0": np.array([0.0, 0.125], dtype=np.float32),
        "fc0/bias:0": np.array([0.0], dtype=np.float32),
        "alpha0/alpha:0": np.array([0.006], dtype=np.float32),
    }
    result = fault_schedule_accounting_conformance(values, codec())
    assert result["unprotected"] == {
        "proposed_events": 2,
        "applied_events": 2,
        "suppressed_events": 0,
        "corrupted_value": -3.875,
    }
    assert result["full_protection"]["proposed_events"] == 2
    assert result["full_protection"]["applied_events"] == 0
    assert result["full_protection"]["suppressed_events"] == 2
    assert result["full_protection"]["state_unchanged"] is True
