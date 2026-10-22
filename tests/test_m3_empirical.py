from pathlib import Path

import numpy as np

from netsurf.analysis_m3 import ranking_quality
from netsurf.adapters.numpy_reference import NumpyReferenceAdapter
from netsurf.empirical import measure_single_bits, select_all
from netsurf.estimators_m3 import compute_estimator
from netsurf.experiment_config import load_experiment
from netsurf.quantization import BitCodec, FixedPointFormat
from netsurf.samples import require_disjoint


ROOT = Path(__file__).resolve().parents[1]


def _reference():
    configuration = load_experiment(ROOT / "config/experiments/numpy_reference.toml")
    codec = BitCodec(FixedPointFormat.parse(configuration.benchmark.quantization))
    adapter = NumpyReferenceAdapter(configuration.benchmark.checkpoint, configuration.benchmark.dataset, codec)
    adapter.load()
    return adapter, codec


def test_m3_partitions_are_explicit_and_disjoint():
    adapter, _ = _reference()
    partitions = [adapter.sample_partition("ranking", 2)[2], adapter.sample_partition("empirical_validation", 3)[2], adapter.sample_partition("evaluation", 3)[2]]
    require_disjoint(partitions)
    assert len({value.sample_sha256 for value in partitions}) == 3


def test_exact_empirical_reference_records_signed_damage_and_restores_state():
    adapter, codec = _reference()
    inputs, labels, partition = adapter.sample_partition("empirical_validation", 3)
    before = adapter.values()
    measurements = measure_single_bits(adapter, inputs, labels, select_all(adapter.bit_identities()), codec, evaluation_sample_sha256=partition.sample_sha256)
    assert len(measurements) == 102
    assert all(value.flipped_encoded == (value.original_encoded ^ (1 << value.bit_index)) for value in measurements)
    assert all(value.positive_loss_damage == max(0.0, value.signed_loss_difference) for value in measurements)
    assert any(value.signed_loss_difference < 0.0 for value in measurements)
    assert any(value.accuracy_damage > 0.0 for value in measurements)
    assert all(np.array_equal(before[name], adapter.values()[name]) for name in before)


def test_exact_hessian_hvp_and_estimator_metrics_are_numerically_characterized():
    adapter, codec = _reference()
    inputs, labels, _ = adapter.sample_partition("ranking", 2)
    identities = adapter.bit_identities()
    hessian = adapter.exact_hessian(inputs, labels)
    assert hessian.shape == (17, 17)
    assert np.allclose(hessian, hessian.T, atol=1e-8)
    vector = {name: np.ones_like(value) for name, value in adapter.values().items()}
    hvp = adapter.hessian_vector_product(inputs, labels, vector)
    flat_hvp = np.concatenate([hvp[name].reshape(-1) for name in adapter.values()])
    assert np.allclose(flat_hvp, hessian @ np.ones(17), atol=1e-8)
    exact = compute_estimator("exact_diagonal_second_order", adapter, inputs, labels, identities, codec)
    approximate = compute_estimator("hutchinson_diagonal_second_order", adapter, inputs, labels, identities, codec, probes=4, seed=42)
    assert exact.signed_predictions is not None
    assert approximate.metadata["probes"] == 4
    assert set(exact.scores) == set(identities) == set(approximate.scores)


def test_qpolar_variants_and_per_sample_aggregation_are_distinct_and_stateless():
    adapter, codec = _reference()
    inputs, labels, _ = adapter.sample_partition("ranking", 2)
    identities = adapter.bit_identities()
    before = adapter.values()
    legacy = compute_estimator("legacy_qpolargrad", adapter, inputs, labels, identities, codec)
    clean = compute_estimator("clean_activation_qpolargrad", adapter, inputs, labels, identities, codec)
    uncancelled = compute_estimator("uncancelled_activation_drift", adapter, inputs, labels, identities, codec)
    ordinary = compute_estimator("first_order_gradient_delta", adapter, inputs, labels, identities, codec)
    per_sample = compute_estimator("per_sample_first_order_magnitude", adapter, inputs, labels, identities, codec)
    assert legacy.scores != clean.scores
    assert clean.scores != uncancelled.scores
    assert ordinary.scores != per_sample.scores
    assert all(np.array_equal(before[name], adapter.values()[name]) for name in before)

    empirical_inputs, empirical_labels, partition = adapter.sample_partition("empirical_validation", 3)
    measurements = measure_single_bits(adapter, empirical_inputs, empirical_labels, select_all(identities), codec, evaluation_sample_sha256=partition.sample_sha256)
    rows = ranking_quality(measurements, (ordinary, per_sample, legacy, clean, uncancelled), top_ks=(5, 10))
    overall = [value for value in rows if value["scope"] == "overall"]
    assert len(overall) == 5
    assert all("spearman_positive_damage" in value for value in overall)
