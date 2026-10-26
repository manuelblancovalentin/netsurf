from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

from netsurf.adapters.registry import create_adapter
from netsurf.incremental import DenseReluDenseIncrementalEvaluator, FaultState, metrics_from_logits
from netsurf.quantization import BitCodec, FixedPointFormat
from netsurf.study_config import load_study
from netsurf.topology import DenseReluDenseTopologyEvaluator, benchmark_topology_engines


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def topology_fixture():
    config = load_study(ROOT / "config/experiments/numpy_qpolar_cancellation.toml")
    codec = BitCodec(FixedPointFormat.parse(config.benchmark.quantization))
    adapter = create_adapter(config, codec)
    adapter.load()
    inputs, labels, _ = adapter.sample_partition("evaluation", 4)
    evaluator = DenseReluDenseIncrementalEvaluator(adapter, codec)
    return adapter, evaluator, inputs, labels


def _bits(adapter, parameter):
    return [bit for bit in adapter.bit_identities() if bit.parameter == parameter]


def test_exhaustive_single_bit_topology_responses_match_full_reference(topology_fixture):
    adapter, evaluator, inputs, _ = topology_fixture
    topology = DenseReluDenseTopologyEvaluator(evaluator, inputs)
    base = evaluator.full_forward(inputs).logits
    maximum = 0.0
    for bit in adapter.bit_identities():
        response = topology.response(bit)
        full = evaluator.full_forward(inputs, FaultState.from_bits((bit,))).logits
        maximum = max(maximum, float(np.max(np.abs(base + response.logits_delta - full))))
        encoded = evaluator.codec.encode(evaluator.base_values[bit.parameter]).reshape(-1)[bit.flat_index]
        assert response.original_encoded == int(encoded)
        assert response.flipped_encoded == int(evaluator.codec.flip(encoded, bit.bit_index))
    assert maximum < 1e-12
    assert topology.cache_misses == 102


def test_exact_pair_classes_and_finite_loss_reconstruction(topology_fixture):
    adapter, evaluator, inputs, labels = topology_fixture
    hidden = _bits(adapter, "dense0/kernel")
    output = _bits(adapter, "dense1/kernel")
    output_bias = _bits(adapter, "dense1/bias")
    shape0 = evaluator.base_values["dense0/kernel"].shape
    shape1 = evaluator.base_values["dense1/kernel"].shape
    h0 = next(bit for bit in hidden if np.unravel_index(bit.flat_index, shape0)[1] == 0 and bit.bit_index == 5)
    h0_other = next(bit for bit in hidden if bit.flat_index != h0.flat_index and np.unravel_index(bit.flat_index, shape0)[1] == 0 and bit.bit_index == 4)
    h1 = next(bit for bit in hidden if np.unravel_index(bit.flat_index, shape0)[1] == 1 and bit.bit_index == 5)
    o_same = next(bit for bit in output if np.unravel_index(bit.flat_index, shape1)[0] == 0 and bit.bit_index == 5)
    o_other = next(bit for bit in output if np.unravel_index(bit.flat_index, shape1)[0] == 1 and bit.bit_index == 4)
    cases = (
        (output[0], output[-1], "output_output_additive"),
        (h0, h1, "hidden_different_neurons_additive"),
        (h0, h0_other, "hidden_same_neuron"),
        (h0, o_same, "cross_layer_same_neuron"),
        (h0, o_other, "cross_layer_different_neurons_additive"),
        (h0, output_bias[0], "hidden_output_bias_additive"),
    )
    topology = DenseReluDenseTopologyEvaluator(evaluator, inputs)
    for left, right, expected_class in cases:
        pair = topology.pair(left, right, labels)
        full = evaluator.full_forward(inputs, FaultState.from_bits((left, right))).logits
        np.testing.assert_allclose(pair.exact_logits, full, rtol=1e-12, atol=1e-12)
        assert pair.topology_class == expected_class
        full_interaction = (metrics_from_logits(full, labels).per_example_loss
                            - metrics_from_logits(evaluator.full_forward(inputs, FaultState.from_bits((left,))).logits, labels).per_example_loss
                            - metrics_from_logits(evaluator.full_forward(inputs, FaultState.from_bits((right,))).logits, labels).per_example_loss
                            + metrics_from_logits(evaluator.full_forward(inputs).logits, labels).per_example_loss)
        np.testing.assert_allclose(pair.exact_loss_interaction_per_sample, full_interaction, rtol=1e-12, atol=1e-12)
        error = np.abs(pair.exact_loss_interaction_per_sample - pair.finite_polar_per_sample)
        assert np.all(error <= pair.analytical_bound_per_sample + 1e-12)
        if expected_class.endswith("additive"):
            np.testing.assert_allclose(pair.logit_residual, 0, atol=1e-12)
            np.testing.assert_allclose(pair.exact_loss_interaction_per_sample, pair.finite_polar_per_sample, atol=1e-12)


def test_random_multifault_background_add_remove_and_fallback(topology_fixture):
    adapter, evaluator, inputs, _ = topology_fixture
    generator = np.random.default_rng(4202)
    identities = tuple(adapter.bit_identities())
    for _ in range(24):
        background_bits = tuple(identities[int(i)] for i in generator.choice(len(identities), 7, replace=False))
        background = FaultState.from_bits(background_bits)
        toggles = tuple(identities[int(i)] for i in generator.choice(len(identities), 9, replace=False))
        topology = DenseReluDenseTopologyEvaluator(evaluator, inputs, background)
        actual = topology.multi_toggle_logits(toggles)
        expected = evaluator.full_forward(inputs, background.toggle(*toggles)).logits
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    fallback_evaluator = DenseReluDenseIncrementalEvaluator(adapter, evaluator.codec, cache_memory_limit_bytes=1)
    topology = DenseReluDenseTopologyEvaluator(fallback_evaluator, inputs)
    bit = identities[0]
    response = topology.response(bit)
    assert response.fallback
    expected = fallback_evaluator.full_forward(inputs, FaultState.from_bits((bit,))).logits - fallback_evaluator.full_forward(inputs).logits
    np.testing.assert_allclose(response.logits_delta, expected, rtol=1e-12, atol=1e-12)


def test_randomized_pair_reconstruction_on_nonempty_background(topology_fixture):
    adapter, evaluator, inputs, labels = topology_fixture
    generator = np.random.default_rng(4203)
    identities = tuple(adapter.bit_identities())
    checked = 0
    for _ in range(36):
        selected = tuple(identities[int(i)] for i in generator.choice(len(identities), 6, replace=False))
        background = FaultState.from_bits(selected[:4])
        left, right = selected[4:]
        topology = DenseReluDenseTopologyEvaluator(evaluator, inputs, background)
        pair = topology.pair(left, right, labels)
        expected = evaluator.full_forward(inputs, background.with_bit(left).with_bit(right)).logits
        np.testing.assert_allclose(pair.exact_logits, expected, rtol=1e-12, atol=1e-12)
        assert pair.maximum_bound_violation <= 1e-12
        checked += 1
    assert checked == 36


def test_response_cache_is_selective_and_invalidated_by_background(topology_fixture):
    adapter, evaluator, inputs, _ = topology_fixture
    bit = adapter.bit_identities()[0]
    clean = DenseReluDenseTopologyEvaluator(evaluator, inputs)
    first = clean.response(bit)
    replay = clean.response(bit)
    assert first is replay and clean.cache_hits == 1 and clean.cache_misses == 1
    background_bit = adapter.bit_identities()[-1]
    conditioned = DenseReluDenseTopologyEvaluator(evaluator, inputs, FaultState.from_bits((background_bit,)))
    assert conditioned.response(bit).original_encoded == first.original_encoded
    assert conditioned is not clean
    bounded = DenseReluDenseTopologyEvaluator(evaluator, inputs, response_cache_limit=1)
    bounded.response(adapter.bit_identities()[0]); bounded.response(adapter.bit_identities()[1])
    assert len(bounded._responses) == 1


def test_bounded_benchmark_measures_all_three_matched_swap_paths(topology_fixture):
    adapter, evaluator, inputs, labels = topology_fixture
    result = benchmark_topology_engines(
        evaluator, inputs, labels, tuple(adapter.bit_identities())[:8],
    )
    assert result["pair_count"] == 4
    for field in (
        "full_matched_swaps_per_second",
        "incremental_matched_swaps_per_second",
        "topology_matched_swaps_per_second",
    ):
        assert result[field] is not None
        assert result[field] > 0
    assert result["maximum_full_logit_absolute_error"] < 1e-12
