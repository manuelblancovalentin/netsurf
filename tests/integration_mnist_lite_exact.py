"""Optional bounded integration for ``mnist_lite_exact_kb_v1``.

The historical checkpoint requires the preserved TensorFlow 2.11/QKeras
environment and is intentionally not collected by the ordinary Python 3.11
suite::

    proj activate netsurf-legacy
    python -m tests.integration_mnist_lite_exact
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import hashlib
import multiprocessing
from pathlib import Path

import numpy as np

from netsurf.adapters.legacy_mnist import MnistLiteExactKernelBiasAdapter
from netsurf.quantization import BitCodec, FixedPointFormat


CHECKPOINT = Path(
    "/ml/experiments/netsurf/legacy/benchmarks/mnist_fnn_lite/"
    "q6_2_1/pruned_0.125_159569_qflatten_dense10u_qapplyalpha_"
    "quantized_relu_dense10u_qsoftmax/models/"
    "pruned_0.125_159569_qflatten_dense10u_qapplyalpha_quantized_"
    "relu_dense10u_qsoftmax.20250414_152629.keras"
)
DATASET = Path("/ml/datasets/mnist/mnist.npz")
EXPECTED_STATE = "4d0d6ee025d8603592de1e67d4d0164ea3f2d9591fde75f4fdc230ac12ca5e32"
EXPECTED_OUTPUT = "c267e559958630913c37aecdf980ebbbbe6b5e1a3640ea8b696788ed6f97b82c"


def _worker_identity(_: int) -> tuple[str, str, int]:
    codec = BitCodec(FixedPointFormat.parse("q<6,2,1>"))
    adapter = MnistLiteExactKernelBiasAdapter(CHECKPOINT, DATASET, codec)
    adapter.load()
    inputs, _, _ = adapter.sample_partition("evaluation", 32)
    outputs = adapter.predict_outputs(inputs)
    return (
        adapter.representation_manifest()["converted_state_sha256"],
        hashlib.sha256(np.ascontiguousarray(outputs).tobytes()).hexdigest(),
        len(adapter.bit_identities()),
    )


def main() -> int:
    with ProcessPoolExecutor(
        max_workers=2, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        worker_results = tuple(executor.map(_worker_identity, range(2)))
    if worker_results[0] != worker_results[1]:
        raise RuntimeError(f"spawn-worker identity mismatch: {worker_results}")

    codec = BitCodec(FixedPointFormat.parse("q<6,2,1>"))
    adapter = MnistLiteExactKernelBiasAdapter(CHECKPOINT, DATASET, codec)
    adapter.load()
    representation = adapter.representation_manifest()
    if representation["converted_state_sha256"] != EXPECTED_STATE:
        raise RuntimeError(f"converted-state mismatch: {representation}")
    if representation["eligible_scalars"] != 7_960 or len(adapter.bit_identities()) != 47_760:
        raise RuntimeError(f"eligible-population mismatch: {representation}")

    inputs, labels, partition = adapter.sample_partition("evaluation", 10_000)
    first = adapter.predict_outputs(inputs)
    second = adapter.predict_outputs(inputs)
    output_digest = hashlib.sha256(np.ascontiguousarray(first).tobytes()).hexdigest()
    if output_digest != EXPECTED_OUTPUT or not np.array_equal(first, second):
        raise RuntimeError(
            f"inference identity mismatch: digest={output_digest}, repeated={np.array_equal(first, second)}"
        )
    result = adapter.evaluate(inputs, labels)
    if result.accuracy != 0.8568 or not np.isclose(
        result.categorical_crossentropy, 0.47477197647094727, rtol=0.0, atol=1e-12
    ):
        raise RuntimeError(f"clean metric mismatch: {result}")

    before = adapter.values()
    zero_identity = next(
        identity
        for identity in adapter.bit_identities()
        if before[identity.parameter].reshape(-1)[identity.flat_index] == 0.0
    )
    attacked = {name: value.copy() for name, value in before.items()}
    flat = attacked[zero_identity.parameter].reshape(-1)
    encoded = codec.encode(flat[zero_identity.flat_index])
    flat[zero_identity.flat_index] = codec.decode(
        codec.flip(codec.flip(encoded, 0), 5)
    ).item()
    with adapter.temporary_values(attacked):
        if np.array_equal(adapter.values()[zero_identity.parameter], before[zero_identity.parameter]):
            raise RuntimeError("two-bit XOR did not alter the selected zero-valued word")
    if any(not np.array_equal(before[name], adapter.values()[name]) for name in before):
        raise RuntimeError("adapter state was not restored")

    gradients = adapter.gradients(inputs[:4], labels[:4])
    if set(gradients) != set(before) or any(not np.all(np.isfinite(value)) for value in gradients.values()):
        raise RuntimeError("bounded gradient validation failed")

    print(
        "PASS mnist_lite_exact_kb_v1 "
        f"samples={result.sample_count} accuracy={result.accuracy:.4f} "
        f"loss={result.categorical_crossentropy:.15f} partition={partition.sample_sha256}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
