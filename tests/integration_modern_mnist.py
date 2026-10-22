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
EXPECTED_PAYLOAD = "f0fa9c19929b577fb8d47f2d479a8730b60606182bc564f00d3a07bf2ce8a32a"


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
    expected_environment = {
        "TF_ENABLE_ONEDNN_OPTS": "0",
        "TF_DETERMINISTIC_OPS": "1",
        "CUDA_VISIBLE_DEVICES": "-1",
    }
    if first.manifest["runtime"]["tensorflow_numerical_environment"] != expected_environment:
        raise RuntimeError("TensorFlow numerical environment was not recorded correctly")
    print(f"PASS repeated modern MNIST payload {EXPECTED_PAYLOAD}")
    print(first.path)
    print(second.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
