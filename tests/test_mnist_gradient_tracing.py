from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from netsurf.adapters.mnist_quantized import MnistQuantizedAdapter, PARAMETER_NAMES
from netsurf.quantization import BitCodec, FixedPointFormat


def test_per_example_gradient_graph_is_cached_across_batch_shapes():
    pytest.importorskip("tensorflow")
    adapter = MnistQuantizedAdapter(
        Path("unused-checkpoint.npz"),
        Path("unused-dataset.npz"),
        BitCodec(FixedPointFormat.parse("q<6,2,1>")),
    )
    adapter._values = {
        "dense0/kernel": np.zeros((784, 32), dtype=np.float32),
        "dense0/bias": np.zeros((32,), dtype=np.float32),
        "dense1/kernel": np.zeros((32, 10), dtype=np.float32),
        "dense1/bias": np.zeros((10,), dtype=np.float32),
    }

    results = []
    for batch_size in (1, 3, 2, 4, 1, 3):
        results.append(
            adapter.per_example_gradients(
                np.zeros((batch_size, 28, 28, 1), dtype=np.float32),
                np.zeros((batch_size,), dtype=np.int64),
            )
        )
    compiled = adapter._per_example_gradient_function

    assert compiled is adapter._per_example_gradient_function
    assert compiled.experimental_get_tracing_count() == 1
    assert tuple(results[0]) == PARAMETER_NAMES
    assert results[0]["dense1/bias"].shape == (1, 10)
    assert results[-1]["dense1/bias"].shape == (3, 10)


def test_per_example_gradient_mean_matches_aggregate_gradient():
    pytest.importorskip("tensorflow")
    rng = np.random.default_rng(20261027)
    adapter = MnistQuantizedAdapter(
        Path("unused-checkpoint.npz"),
        Path("unused-dataset.npz"),
        BitCodec(FixedPointFormat.parse("q<6,2,1>")),
    )
    adapter._values = {
        "dense0/kernel": rng.normal(0, 0.02, (784, 32)).astype(np.float32),
        "dense0/bias": rng.normal(0, 0.02, (32,)).astype(np.float32),
        "dense1/kernel": rng.normal(0, 0.02, (32, 10)).astype(np.float32),
        "dense1/bias": rng.normal(0, 0.02, (10,)).astype(np.float32),
    }
    inputs = rng.random((2, 28, 28, 1), dtype=np.float32)
    labels = np.asarray([1, 7], dtype=np.int64)

    per_example = adapter.per_example_gradients(inputs, labels)
    aggregate = adapter.gradients(inputs, labels)

    for name in PARAMETER_NAMES:
        np.testing.assert_allclose(per_example[name].mean(axis=0), aggregate[name], rtol=2e-5, atol=2e-6)
