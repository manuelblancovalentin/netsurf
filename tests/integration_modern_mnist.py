"""Optional bounded integration check for the modern TensorFlow/Keras benchmark.

Run explicitly (it is intentionally not collected by the ordinary suite):

    conda run -n netsurf python -m tests.integration_modern_mnist

The prepared checkpoint is an input to this test. Training is not performed.
Two matched 256-sample, 12-trial experiments are finalized so stable payload
equivalence is checked through the same path researchers use.
"""

from __future__ import annotations

from pathlib import Path

from netsurf.api import read_result_bundle, run_experiment
from netsurf.manifests import sha256_file


EXPECTED_CHECKPOINT = "1c58daf1bb23d1cdbaf38e68e4d957145be62435fd379bc3716904d57aa77780"
EXPECTED_PAYLOAD = "0b1b7b0edb92cf7480ceb1c261cdf02cd948a70ce9e9d5f26b57905e71bd650f"


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    checkpoint = root / "workspace/benchmarks/mnist_quantized/checkpoint.npz"
    if not checkpoint.is_file():
        raise RuntimeError(
            "modern checkpoint is missing; run "
            "`netsurf benchmarks prepare mnist_quantized --epochs 3 --train-samples 20000`"
        )
    actual_checkpoint = sha256_file(checkpoint)
    if actual_checkpoint != EXPECTED_CHECKPOINT:
        raise RuntimeError(
            f"checkpoint mismatch: expected {EXPECTED_CHECKPOINT}, got {actual_checkpoint}"
        )

    configuration = root / "config/experiments/mnist_ranker_comparison.toml"
    first = read_result_bundle(run_experiment(configuration, repository_root=root))
    second = read_result_bundle(run_experiment(configuration, repository_root=root))
    digests = (first.scientific_payload_sha256, second.scientific_payload_sha256)
    if digests != (EXPECTED_PAYLOAD, EXPECTED_PAYLOAD):
        raise RuntimeError(f"scientific payload mismatch: {digests}")
    print(f"PASS repeated modern MNIST payload {EXPECTED_PAYLOAD}")
    print(first.path)
    print(second.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
