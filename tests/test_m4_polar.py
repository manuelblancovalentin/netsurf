from __future__ import annotations

from pathlib import Path

import pytest

from netsurf.adapters.registry import create_adapter
from netsurf.incremental import DenseReluDenseIncrementalEvaluator, FaultState
from netsurf.polar import interaction_map, measure_polar_pair, measure_sparse_interactions
from netsurf.quantization import BitCodec, FixedPointFormat
from netsurf.study_config import load_study


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def polar_fixture():
    configuration = load_study(ROOT / "config/experiments/numpy_qpolar_cancellation.toml")
    codec = BitCodec(FixedPointFormat.parse(configuration.benchmark.quantization))
    adapter = create_adapter(configuration, codec); adapter.load()
    inputs, labels, _ = adapter.sample_partition("evaluation", 4)
    return adapter, DenseReluDenseIncrementalEvaluator(adapter, codec), inputs, labels


def test_final_layer_logit_responses_are_additive_and_bound_is_exact(polar_fixture):
    adapter, evaluator, inputs, labels = polar_fixture
    candidates = [bit for bit in adapter.bit_identities() if bit.parameter == "dense1/kernel"]
    result = measure_polar_pair(evaluator, inputs, labels, FaultState.from_bits(), candidates[0], candidates[7])
    assert result.logit_nonadditivity_mean_l2 == pytest.approx(0.0, abs=1e-14)
    assert result.exact_interaction == pytest.approx(result.finite_polar_interaction, abs=1e-14)
    assert result.bound_satisfied
    assert result.layer_relationship == "within_layer"


def test_cross_layer_exact_interaction_and_error_bound(polar_fixture):
    adapter, evaluator, inputs, labels = polar_fixture
    left = next(bit for bit in adapter.bit_identities() if bit.parameter == "dense0/kernel" and bit.bit_index == 5)
    right = next(bit for bit in adapter.bit_identities() if bit.parameter == "dense1/kernel" and bit.bit_index == 4)
    result = measure_polar_pair(evaluator, inputs, labels, FaultState.from_bits(), left, right)
    assert result.layer_relationship == "cross_layer"
    assert abs(result.interaction_prediction_error) <= result.analytical_error_bound + 1e-12
    assert result.bound_satisfied
    assert result.cancellation_score is None or result.cancellation_score >= 0


def test_background_conditioned_sparse_measurements_are_deterministic(polar_fixture):
    adapter, evaluator, inputs, labels = polar_fixture
    identities = tuple(adapter.bit_identities())
    pairs = ((identities[0], identities[-1]), (identities[1], identities[-2]))
    backgrounds = (FaultState.from_bits(), FaultState.from_bits((identities[12], identities[18])))
    first = measure_sparse_interactions(evaluator, inputs, labels, pairs, backgrounds)
    replay = measure_sparse_interactions(evaluator, inputs, labels, pairs, backgrounds)
    assert [value.to_dict() | {"wall_seconds": 0} for value in first] == [value.to_dict() | {"wall_seconds": 0} for value in replay]
    assert set(interaction_map(first)) == set(pairs)
    assert all(value.left_weight_delta != 0 and value.right_weight_delta != 0 for value in first)
