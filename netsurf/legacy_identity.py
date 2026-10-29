"""Versioned identity and bounded conformance tools for legacy MNIST-lite.

The manifest is evidence about one recovered file, not a declaration that the
file is the manuscript's Table-II benchmark.  TensorFlow is imported only by
``run_conformance``; reading or validating identity metadata is side-effect
free.
"""

from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .adapters.legacy_mnist import LegacyMnistLiteAdapter
from .faults import (
    FaultEvent,
    FaultModel,
    FaultSchedule,
    apply_persistent_schedule_encoded_state,
)
from .identities import BitIdentity
from .protection import ProtectionMask
from .quantization import BitCodec, FixedPointFormat


IDENTITY_SCHEMA_VERSION = "1.0.0"
IDENTITY_NAME = "legacy_mnist_lite.recovered_2025.v1"
IDENTITY_FILES = {
    "legacy_mnist_lite": "legacy_mnist_lite.v1.json",
    "mnist_lite_exact_kb_v1": "mnist_lite_exact_kb_v1.json",
    "keyword_spotting_candidate": "keyword_spotting_candidate.v1.json",
    "toyadmos_autoencoder_candidate": "toyadmos_autoencoder_candidate.v1.json",
    "mlperf_tiny_ic_resnet8_int8_v1": "mlperf_tiny_ic_resnet8_int8_v1.json",
    "mlperf_tiny_kws_dscnn_int8_v1": "mlperf_tiny_kws_dscnn_int8_v1.json",
    "mlperf_tiny_ad_toycar_int8_v1": "mlperf_tiny_ad_toycar_int8_v1.json",
    "mlperf_tiny_vww_mobilenetv1_int8_candidate_v1": "mlperf_tiny_vww_mobilenetv1_int8_candidate_v1.json",
}
SUPPORTED_ENCODED_ROLES = frozenset({"kernel", "bias"})


def load_identity_manifest() -> dict[str, Any]:
    resource = files("netsurf.benchmarks").joinpath(
        "identities/legacy_mnist_lite.v1.json"
    )
    document = json.loads(resource.read_text(encoding="utf-8"))
    if document.get("schema_version") != IDENTITY_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported legacy benchmark identity schema: {document.get('schema_version')!r}"
        )
    if document.get("identity") != IDENTITY_NAME:
        raise ValueError(f"unexpected legacy benchmark identity: {document.get('identity')!r}")
    return document


def load_benchmark_identity(name: str) -> dict[str, Any]:
    """Load any registered recovered/derived benchmark identity document."""

    try:
        filename = IDENTITY_FILES[name]
    except KeyError as error:
        raise KeyError(
            f"benchmark {name!r} has no versioned identity; available: "
            f"{', '.join(sorted(IDENTITY_FILES))}"
        ) from error
    resource = files("netsurf.benchmarks").joinpath(f"identities/{filename}")
    document = json.loads(resource.read_text(encoding="utf-8"))
    if document.get("schema_version") != IDENTITY_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported benchmark identity schema: {document.get('schema_version')!r}"
        )
    return document


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_identity_files(
    manifest: Mapping[str, Any] | None = None, *, repository_root: Path | None = None
) -> dict[str, Any]:
    """Verify immutable input hashes without importing TensorFlow."""

    document = dict(manifest or load_identity_manifest())
    root = Path(repository_root or Path(__file__).resolve().parents[1])
    results: dict[str, Any] = {}
    if "artifacts" in document:
        records = {
            key: (
                Path(document["artifacts"][key]["kona_path"]),
                document["artifacts"][key]["sha256"],
                document["artifacts"][key].get("repository_access_path"),
            )
            for key in ("checkpoint", "dataset")
        }
    elif document.get("identity") == "mnist_lite_exact_kb_v1":
        records = {
            key: (
                Path(document["source"][f"{key}_path"]),
                document["source"][f"{key}_sha256"],
                None,
            )
            for key in ("checkpoint", "dataset")
        }
    else:
        raise ValueError("identity does not define verifiable checkpoint/dataset records")
    for key, (path, expected, fallback) in records.items():
        if key == "checkpoint" and not path.is_file() and fallback:
            path = root / fallback
        present = path.is_file()
        observed = file_sha256(path) if present else None
        results[key] = {
            "path": str(path.resolve(strict=False)),
            "present": present,
            "expected_sha256": expected,
            "observed_sha256": observed,
            "matches": present and observed == expected,
        }
    results["all_match"] = all(item["matches"] for item in results.values())
    return results


def tensor_role(name: str) -> str:
    leaf = name.removesuffix(":0").rsplit("/", 1)[-1]
    if leaf in {"kernel", "bias", "alpha", "beta"}:
        return leaf
    return "other"


def parameter_quantization_statistics(
    values: Mapping[str, np.ndarray], codec: BitCodec
) -> dict[str, Any]:
    tensors: dict[str, Any] = {}
    totals = {
        "scalars": 0,
        "nonzero_scalars": 0,
        "on_lattice_scalars": 0,
        "out_of_range_scalars": 0,
    }
    for name, source in values.items():
        value = np.asarray(source)
        quantized = codec.quantize(value)
        difference = np.abs(value.astype(np.float64) - quantized)
        out_of_range = (value < codec.format.value_min) | (value > codec.format.value_max)
        on_lattice = difference == 0.0
        item = {
            "role": tensor_role(name),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "scalars": int(value.size),
            "nonzero_scalars": int(np.count_nonzero(value)),
            "on_lattice_scalars": int(np.count_nonzero(on_lattice)),
            "out_of_range_scalars": int(np.count_nonzero(out_of_range)),
            "at_lower_bound_scalars": int(
                np.count_nonzero(value == codec.format.value_min)
            ),
            "at_upper_bound_scalars": int(
                np.count_nonzero(value == codec.format.value_max)
            ),
            "quantized_to_lower_bound_scalars": int(
                np.count_nonzero(quantized == codec.format.value_min)
            ),
            "quantized_to_upper_bound_scalars": int(
                np.count_nonzero(quantized == codec.format.value_max)
            ),
            "clipped_due_to_range_violation_scalars": int(
                np.count_nonzero(out_of_range)
            ),
            "minimum": float(np.min(value)),
            "maximum": float(np.max(value)),
            "maximum_absolute_quantization_error": float(np.max(difference)),
            "mean_absolute_quantization_error": float(np.mean(difference)),
        }
        tensors[name] = item
        for total in totals:
            totals[total] += item[total]
    return {"tensors": tensors, "totals": totals}


def _kernel_bias_quantized_state(
    values: Mapping[str, np.ndarray], codec: BitCodec
) -> dict[str, np.ndarray]:
    return {
        name: (
            codec.quantize(value).astype(np.asarray(value).dtype)
            if tensor_role(name) in SUPPORTED_ENCODED_ROLES
            else np.asarray(value).copy()
        )
        for name, value in values.items()
    }


def _names_by_role(values: Mapping[str, np.ndarray]) -> dict[str, str]:
    selected: dict[str, str] = {}
    for name in values:
        role = tensor_role(name)
        if name.startswith("fc0/") and role in {"kernel", "bias"}:
            selected[f"fc0_{role}"] = name
        elif name.startswith("alpha0/") and role in {"alpha", "beta"}:
            selected[role] = name
    required = {"fc0_kernel", "fc0_bias", "alpha", "beta"}
    if set(selected) != required:
        raise ValueError(f"cannot identify foldable alpha path: found {selected}")
    return selected


def fold_alpha_beta_state(values: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Algebraically fold ``alpha * (xW+b) + beta`` into the first Dense.

    This transformation says nothing about a suitable storage precision for
    the folded tensors.  It exists to test numerical equivalence and candidate
    conversions explicitly.
    """

    result = {name: np.asarray(value).copy() for name, value in values.items()}
    names = _names_by_role(result)
    alpha = result[names["alpha"]]
    beta = result[names["beta"]]
    result[names["fc0_kernel"]] = (
        result[names["fc0_kernel"]] * alpha.reshape(1, -1)
    ).astype(result[names["fc0_kernel"]].dtype)
    result[names["fc0_bias"]] = (
        result[names["fc0_bias"]] * alpha + beta
    ).astype(result[names["fc0_bias"]].dtype)
    result[names["alpha"]] = np.ones_like(alpha)
    result[names["beta"]] = np.zeros_like(beta)
    return result


def fault_representation_comparison(
    values: Mapping[str, np.ndarray], codec: BitCodec
) -> dict[str, Any]:
    """Compare recovered delta-on-float with exact decoded XOR states."""

    by_tensor: dict[str, Any] = {}
    total_comparisons = 0
    equal_comparisons = 0
    maximum_difference = 0.0
    for name, source in values.items():
        if tensor_role(name) not in SUPPORTED_ENCODED_ROLES:
            continue
        original = np.asarray(source, dtype=np.float64)
        encoded = codec.encode(original)
        decoded = codec.decode(encoded)
        tensor_equal = 0
        tensor_total = original.size * codec.format.total_bits
        tensor_maximum = 0.0
        for bit_index in range(codec.format.total_bits):
            flipped = codec.decode(codec.flip(encoded, bit_index))
            legacy = original + (flipped - decoded)
            difference = np.abs(legacy - flipped)
            tensor_equal += int(np.count_nonzero(difference == 0.0))
            tensor_maximum = max(tensor_maximum, float(np.max(difference)))
        by_tensor[name] = {
            "scalar_bit_comparisons": tensor_total,
            "numerically_equal": tensor_equal,
            "different": tensor_total - tensor_equal,
            "maximum_absolute_state_difference": tensor_maximum,
        }
        total_comparisons += tensor_total
        equal_comparisons += tensor_equal
        maximum_difference = max(maximum_difference, tensor_maximum)
    return {
        "by_tensor": by_tensor,
        "total_scalar_bit_comparisons": total_comparisons,
        "numerically_equal": equal_comparisons,
        "different": total_comparisons - equal_comparisons,
        "maximum_absolute_state_difference": maximum_difference,
        "identity": (
            "legacy_corrupted - exact_xor_corrupted equals the original "
            "off-lattice residual for every applied bit"
        ),
    }


def codec_conformance(codec: BitCodec) -> dict[str, Any]:
    codes = np.arange(1 << codec.format.total_bits, dtype=np.uint64)
    decoded = codec.decode(codes)
    round_trip = np.array_equal(codec.encode(decoded), codes)
    single_flip_checks = 0
    for bit_index in range(codec.format.total_bits):
        flipped = codec.flip(codes, bit_index)
        if not np.array_equal(codec.flip(flipped, bit_index), codes):
            raise AssertionError("single-bit XOR is not involutive")
        single_flip_checks += codes.size
    pair_checks = 0
    for first in range(codec.format.total_bits):
        for second in range(first + 1, codec.format.total_bits):
            sequential = codec.flip(codec.flip(codes, first), second)
            combined = np.bitwise_xor(codes, np.uint64((1 << first) | (1 << second)))
            if not np.array_equal(sequential, combined):
                raise AssertionError("multiple XOR flips do not compose")
            pair_checks += codes.size
    return {
        "code_values": int(codes.size),
        "round_trip_exact": round_trip,
        "single_bit_xor_checks": single_flip_checks,
        "two_distinct_bit_xor_checks": pair_checks,
    }


def fault_schedule_accounting_conformance(
    values: Mapping[str, np.ndarray], codec: BitCodec
) -> dict[str, Any]:
    """Exercise multiple XORs, allocated zero state, and protection bounds."""

    exact = _kernel_bias_quantized_state(values, codec)
    exact = {
        name: value
        for name, value in exact.items()
        if tensor_role(name) in SUPPORTED_ENCODED_ROLES
    }
    zero_target = None
    for name, value in values.items():
        if tensor_role(name) in SUPPORTED_ENCODED_ROLES:
            indices = np.flatnonzero(np.asarray(value).reshape(-1) == 0.0)
            if indices.size:
                zero_target = (name, int(indices[0]))
                break
    if zero_target is None:
        raise ValueError("checkpoint contains no allocated zero-valued kernel/bias scalar")
    name, flat_index = zero_target
    bits = (
        BitIdentity(name, flat_index, 0),
        BitIdentity(name, flat_index, codec.format.total_bits - 1),
    )
    events = tuple(FaultEvent(None, bit) for bit in bits)
    model = FaultModel(
        target="weight",
        location="logical",
        persistence="persistent_trial",
        distribution="independent_bernoulli",
        probability=1.0,
    )
    schedule = FaultSchedule(model, 0, events, sum(v.size for v in exact.values()) * 6, 1)
    none = ProtectionMask("none", frozenset(), schedule.universe_size, 0.0, {})
    full = ProtectionMask("full", frozenset(bits), schedule.universe_size, 2.0, {})
    corrupted, unprotected = apply_persistent_schedule_encoded_state(
        exact, schedule, none, codec
    )
    restored, protected = apply_persistent_schedule_encoded_state(
        exact, schedule, full, codec
    )
    return {
        "allocated_zero_target": bits[0].key(),
        "multiple_flips_same_scalar": [bit.key() for bit in bits],
        "unprotected": {
            "proposed_events": unprotected.proposed_events,
            "applied_events": unprotected.applied_events,
            "suppressed_events": unprotected.suppressed_events,
            "corrupted_value": float(corrupted[name].reshape(-1)[flat_index]),
        },
        "full_protection": {
            "proposed_events": protected.proposed_events,
            "applied_events": protected.applied_events,
            "suppressed_events": protected.suppressed_events,
            "state_unchanged": all(
                np.array_equal(exact[tensor], restored[tensor]) for tensor in exact
            ),
        },
    }


def _scenario_metrics(
    adapter: LegacyMnistLiteAdapter,
    inputs: np.ndarray,
    labels: np.ndarray,
    reference_outputs: np.ndarray,
    reference_logits: np.ndarray,
    state: Mapping[str, np.ndarray] | None,
) -> dict[str, Any]:
    context = adapter.temporary_values(state) if state is not None else nullcontext()
    with context:
        evaluation = adapter.evaluate(inputs, labels)
        outputs = adapter.predict_outputs(inputs)
        logits = adapter.predict_logits(inputs)
    return {
        "accuracy": evaluation.accuracy,
        "categorical_crossentropy": evaluation.categorical_crossentropy,
        "prediction_disagreements_from_float_shadow": int(
            np.count_nonzero(np.argmax(outputs, axis=1) != np.argmax(reference_outputs, axis=1))
        ),
        "maximum_absolute_output_deviation": float(np.max(np.abs(outputs - reference_outputs))),
        "mean_absolute_output_deviation": float(np.mean(np.abs(outputs - reference_outputs))),
        "maximum_absolute_logit_deviation": float(np.max(np.abs(logits - reference_logits))),
        "mean_absolute_logit_deviation": float(np.mean(np.abs(logits - reference_logits))),
    }


def run_conformance(
    checkpoint: Path,
    dataset: Path,
    *,
    sample_count: int = 256,
    output: Path | None = None,
) -> dict[str, Any]:
    """Run the bounded checkpoint conversion comparison on held-out MNIST."""

    codec = BitCodec(FixedPointFormat.parse("q<6,2,1>"))
    adapter = LegacyMnistLiteAdapter(checkpoint, dataset, codec)
    adapter.load()
    inputs, labels, partition = adapter.sample_partition("evaluation", sample_count)
    original = adapter.values()
    reference_outputs = adapter.predict_outputs(inputs)
    reference_logits = adapter.predict_logits(inputs)

    folded = fold_alpha_beta_state(original)
    states = {
        "original_float_shadow": None,
        # No-fault inference under the recovered injector is exactly the
        # original float-shadow state.  It remains a separate named semantic.
        "historical_delta_on_float_clean": original,
        "q621_kernel_bias_with_float_auxiliaries": _kernel_bias_quantized_state(
            original, codec
        ),
        "float_alpha_beta_fold_equivalence_control": folded,
        "q621_folded_kernel_bias_with_identity_auxiliaries": _kernel_bias_quantized_state(
            folded, codec
        ),
        "common_q621_all_trainables_rejected_diagnostic": {
            name: codec.quantize(value).astype(value.dtype)
            for name, value in original.items()
        },
    }
    scenarios = {
        name: _scenario_metrics(
            adapter, inputs, labels, reference_outputs, reference_logits, state
        )
        for name, state in states.items()
    }
    repeated_outputs = adapter.predict_outputs(inputs)
    restored_values = adapter.values()
    result = {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "identity": IDENTITY_NAME,
        "input_verification": verify_identity_files(),
        "sample_partition": partition.to_dict(),
        "sample_count": sample_count,
        "quantization": codec.format.to_dict(),
        "parameter_quantization": parameter_quantization_statistics(original, codec),
        "fault_representation": fault_representation_comparison(original, codec),
        "codec_conformance": codec_conformance(codec),
        "schedule_accounting_conformance": fault_schedule_accounting_conformance(
            original, codec
        ),
        "scenarios": scenarios,
        "reproducibility": {
            "repeated_float_outputs_bitwise_equal": bool(
                np.array_equal(reference_outputs, repeated_outputs)
            ),
            "state_restored_after_scenarios": all(
                np.array_equal(original[name], restored_values[name]) for name in original
            ),
            "float_output_sha256": hashlib.sha256(
                reference_outputs.tobytes(order="C")
            ).hexdigest(),
        },
        "interpretation": {
            "historical_delta_on_float_clean": (
                "identical to original float-shadow inference when no fault is present"
            ),
            "q621_kernel_bias_with_float_auxiliaries": (
                "exact six-bit kernels/biases but unresolved float alpha/beta storage"
            ),
            "float_alpha_beta_fold_equivalence_control": (
                "algebraic conversion control; not a storage-format claim"
            ),
            "q621_folded_kernel_bias_with_identity_auxiliaries": (
                "candidate conversion whose adequacy must be judged from measured results"
            ),
            "common_q621_all_trainables_rejected_diagnostic": (
                "diagnostic only; alpha/beta do not share a justified q<6,2,1> storage contract"
            ),
        },
    }
    if output is not None:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
            stream.write("\n")
    return result


def _main(argv: list[str] | None = None) -> int:
    """Python-3.10-compatible gate CLI for the preserved legacy environment."""

    import argparse

    parser = argparse.ArgumentParser(prog="python -m netsurf.legacy_identity")
    commands = parser.add_subparsers(dest="command", required=True)
    identity = commands.add_parser("identity")
    identity.add_argument("--verify-files", action="store_true")
    conformance = commands.add_parser("conformance")
    conformance.add_argument("--samples", type=int, default=256)
    conformance.add_argument("--checkpoint", type=Path)
    conformance.add_argument("--dataset", type=Path)
    conformance.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    document = load_identity_manifest()
    if args.command == "identity":
        if args.verify_files:
            document = {**document, "file_verification": verify_identity_files(document)}
        print(json.dumps(document, indent=2, sort_keys=True))
        return 0
    checkpoint = args.checkpoint or Path(document["artifacts"]["checkpoint"]["kona_path"])
    dataset = args.dataset or Path(document["artifacts"]["dataset"]["kona_path"])
    result = run_conformance(
        checkpoint, dataset, sample_count=args.samples, output=args.output
    )
    print(
        json.dumps(
            {"identity": result["identity"], "samples": result["sample_count"], "output": str(args.output)},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
