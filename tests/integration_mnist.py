"""Optional slow integration check for the validated TensorFlow 2.11 stack.

Run explicitly (it is intentionally not collected by the ordinary suite):

    conda run -n netsurf-legacy python -m tests.integration_mnist
"""

from __future__ import annotations

from pathlib import Path

from netsurf.api import read_result_bundle, run_experiment


EXPECTED = "07f4e88d7a36663fb120fafb97c28b80bab5dd3be13b91700755a4784b30bc1b"


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    configuration = root / "config/experiments/template.toml"
    first = read_result_bundle(run_experiment(configuration, repository_root=root))
    second = read_result_bundle(run_experiment(configuration, repository_root=root))
    digests = (first.scientific_payload_sha256, second.scientific_payload_sha256)
    if digests != (EXPECTED, EXPECTED):
        raise RuntimeError(f"scientific payload mismatch: {digests}")
    print(f"PASS repeated MNIST payload {EXPECTED}")
    print(first.path)
    print(second.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
