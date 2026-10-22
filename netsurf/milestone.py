"""Bounded deterministic MNIST-lite scientific milestone."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from .adapters import LegacyMnistLiteAdapter
from .faults import FaultModel, FaultSchedule, apply_persistent_schedule
from .manifests import digest_object, git_provenance, runtime_provenance, sha256_file
from .paths import artifact_root
from .protection import ProtectionPolicy
from .quantization import BitCodec, FixedPointFormat
from .ranking import (
    FirstOrderEstimator,
    LegacyQPolarGradEstimator,
    MSBEstimator,
    RandomEstimator,
)
from .result_store import ResultStore
from .statistics import summarize
from .experiment_config import validate_experiment_document
from .experiment_engine import execute_experiment


CHECKPOINT_RELATIVE = Path(
    "benchmarks/mnist_fnn_lite/q6_2_1/"
    "pruned_0.125_159569_qflatten_dense10u_qapplyalpha_quantized_relu_dense10u_qsoftmax/"
    "models/pruned_0.125_159569_qflatten_dense10u_qapplyalpha_quantized_relu_dense10u_qsoftmax."
    "20250414_152629.keras"
)


def _arrays_digest(values: dict[str, np.ndarray]) -> str:
    payload = []
    for name, value in sorted(values.items()):
        array = np.asarray(value)
        payload.append(
            {
                "name": name,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "bytes_sha256": __import__("hashlib").sha256(array.tobytes(order="C")).hexdigest(),
            }
        )
    return digest_object(payload)


def _report_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# NetSuRF v2 MNIST-lite deterministic milestone",
        "",
        f"Scientific payload SHA-256: `{digest_object(payload)}`",
        "",
        "## Contract",
        "",
        "This bounded experiment uses a persistent logical weight-bit corruption per Monte Carlo trial. "
        "It is not a physical SEU-rate model. All policies receive the same fault schedule in each trial.",
        "",
        f"Checkpoint baseline: accuracy `{payload['baseline']['checkpoint']['accuracy']:.6f}`, "
        f"cross entropy `{payload['baseline']['checkpoint']['categorical_crossentropy']:.6f}`.",
        "",
        f"Maximum checkpoint-to-fixed-point distance: "
        f"`{payload['quantization_verification']['maximum_absolute_quantization_error']:.8f}`.",
        "",
        "## Paired Monte Carlo results",
        "",
        "| Policy | Protected bits | Coverage | Mean accuracy | 95% CI | Mean cross entropy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, result in payload["policies"].items():
        accuracy = result["accuracy"]
        cross_entropy = result["categorical_crossentropy"]
        low, high = accuracy["confidence_interval"]
        lines.append(
            f"| {name} | {result['protected_bits']} | {result['coverage']:.6f} | "
            f"{accuracy['mean']:.6f} | [{low:.6f}, {high:.6f}] | {cross_entropy['mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Reproducibility checks",
            "",
            f"- q<6,2,1> codec exhaustive check: `{payload['checks']['codec_exhaustive']}`.",
            f"- Full-MSB-tier qPolarGrad/MSB mask equivalence: `{payload['checks']['full_msb_mask_equivalent']}`.",
            f"- Model state restored after every trial: `{payload['checks']['state_restored']}`.",
            f"- Matched schedule hashes across policies: `{payload['checks']['matched_schedules']}`.",
        "- `legacy_qpolargrad` is labeled historical and uses globally attacked first-call activations; "
            "this run does not claim that it is an expected-loss estimator.",
            "- Exact bit-score rankings, protection masks, and fault events are stored as separate "
            "write-once JSON artifacts and bound into the scientific payload by SHA-256.",
            "",
        ]
    )
    return "\n".join(lines)


def _codec_exhaustive(codec: BitCodec) -> bool:
    format_ = codec.format
    encoded = np.arange(1 << format_.total_bits, dtype=np.uint64)
    decoded = codec.decode(encoded)
    if not np.array_equal(codec.encode(decoded), encoded):
        return False
    for bit in range(format_.total_bits):
        expected = codec.decode(codec.flip(encoded, bit)) - decoded
        actual = codec.perturbation(decoded, bit)
        if not np.array_equal(expected, actual):
            return False
    return True


def _legacy_execute_once_reference(
    root: Path,
    output_root: Path,
    *,
    repeat_index: int,
    checkpoint: Path | None = None,
    dataset: Path | None = None,
    sample_count: int = 256,
    ranking_count: int = 256,
    trials: int = 12,
    ber: float = 0.001,
    coverage: float = 0.10,
    random_ranking_seed: int = 23011,
    fault_seed: int = 910_000,
) -> tuple[Path, str, dict[str, Any]]:
    root = Path(root).resolve()
    if checkpoint is None:
        checkpoint = artifact_root() / "legacy" / CHECKPOINT_RELATIVE
    checkpoint = Path(checkpoint).expanduser().resolve()
    if dataset is None:
        raise ValueError("dataset path must be provided explicitly")
    dataset = Path(dataset).expanduser().resolve()
    codec = BitCodec(FixedPointFormat.parse("q<6,2,1>"))
    adapter = LegacyMnistLiteAdapter(checkpoint, dataset, codec)
    adapter.load()
    evaluation_inputs, evaluation_labels = adapter.dataset(sample_count)
    ranking_inputs, ranking_labels = adapter.dataset(ranking_count)

    checkpoint_state = adapter.values()
    checkpoint_baseline = adapter.evaluate(evaluation_inputs, evaluation_labels)
    quantized_state = adapter.quantized_values()
    quantization_errors = {
        name: float(np.max(np.abs(checkpoint_state[name] - quantized_state[name])))
        for name in checkpoint_state
    }
    with adapter.temporary_values(checkpoint_state):
        gradients = adapter.gradients(ranking_inputs, ranking_labels)
        deltas = {
            name: codec.perturbation_matrix(value)
            for name, value in checkpoint_state.items()
        }
        impacts = adapter.polar_impacts(
            ranking_inputs, activation_source="globally_attacked_first_call"
        )
        identities = adapter.bit_identities()
        qpolar = LegacyQPolarGradEstimator().estimate(
            identities,
            gradients,
            impacts,
            activation_source="globally_attacked_first_call",
        )
        msb = MSBEstimator().estimate(identities)
        random = RandomEstimator(random_ranking_seed).estimate(identities)
        first_order = FirstOrderEstimator().estimate(identities, gradients, deltas)

        masks = {
            "legacy_qpolargrad_position_first": ProtectionPolicy.position_first(qpolar, coverage),
            "msb_first": ProtectionPolicy.global_ranking(msb, coverage),
            "random": ProtectionPolicy.global_ranking(random, coverage),
            "first_order_gradient_delta": ProtectionPolicy.global_ranking(first_order, coverage),
        }
        full_tier = 1.0 / codec.format.total_bits
        qpolar_full_msb = ProtectionPolicy.position_first(qpolar, full_tier)
        msb_full = ProtectionPolicy.global_ranking(msb, full_tier)

        fault_model = FaultModel(
            target="weight",
            location="logical_trainable_parameter_bits",
            persistence="persistent_trial",
            distribution="independent_bernoulli",
            probability=ber,
        )
        trial_rows: list[dict[str, Any]] = []
        pre_trials_state = _arrays_digest(adapter.values())
        schedules: list[FaultSchedule] = []
        for trial in range(trials):
            schedule = FaultSchedule.generate(
                fault_model,
                identities,
                seed=fault_seed + trial,
                inference_count=sample_count,
            )
            schedules.append(schedule)
            for policy_name, mask in masks.items():
                corrupted, accounting = apply_persistent_schedule(
                    checkpoint_state, schedule, mask, codec
                )
                with adapter.temporary_values(corrupted):
                    evaluation = adapter.evaluate(evaluation_inputs, evaluation_labels)
                trial_rows.append(
                    {
                        "trial": trial,
                        "policy": policy_name,
                        "schedule_sha256": schedule.digest(),
                        "accuracy": evaluation.accuracy,
                        "categorical_crossentropy": evaluation.categorical_crossentropy,
                        **asdict(accounting),
                    }
                )
        post_trials_state = _arrays_digest(adapter.values())

    if _arrays_digest(adapter.values()) != _arrays_digest(checkpoint_state):
        raise RuntimeError("checkpoint state was not restored after quantized experiment")

    policy_results: dict[str, Any] = {}
    for policy_name, mask in masks.items():
        rows = [row for row in trial_rows if row["policy"] == policy_name]
        policy_results[policy_name] = {
            "protected_bits": len(mask.protected),
            "coverage": mask.coverage,
            "mask_sha256": mask.digest(),
            "accuracy": summarize([row["accuracy"] for row in rows], bounds=(0.0, 1.0)).to_dict(),
            "categorical_crossentropy": summarize(
                [row["categorical_crossentropy"] for row in rows]
            ).to_dict(),
            "mean_applied_events": float(np.mean([row["applied_events"] for row in rows])),
            "mean_suppressed_events": float(np.mean([row["suppressed_events"] for row in rows])),
        }

    schedule_by_trial = {
        trial: {row["schedule_sha256"] for row in trial_rows if row["trial"] == trial}
        for trial in range(trials)
    }
    exact_rankings = {
        ranking.estimator: [
            [identity.parameter, identity.flat_index, identity.bit_index, float(score)]
            for identity, score in sorted(ranking.scores.items())
        ]
        for ranking in (qpolar, msb, random, first_order)
    }
    exact_masks = {
        policy_name: [
            [identity.parameter, identity.flat_index, identity.bit_index]
            for identity in sorted(mask.protected)
        ]
        for policy_name, mask in masks.items()
    }
    exact_schedules = [
        {
            "trial": trial,
            "seed": schedule.seed,
            "sha256": schedule.digest(),
            "events": [
                [event.inference_index, event.bit.parameter, event.bit.flat_index, event.bit.bit_index]
                for event in schedule.events
            ],
        }
        for trial, schedule in enumerate(schedules)
    ]
    scientific_payload: dict[str, Any] = {
        "schema": "netsurf-v2-mnist-lite-milestone/v1",
        "benchmark": {
            "id": "mnist_fnn_lite-20250414_152629-q6_2_1-pruned_0_125",
            "checkpoint_sha256": sha256_file(checkpoint),
            "dataset_sha256": sha256_file(dataset),
            "split": "keras_mnist_test_first_n",
            "sample_count": sample_count,
            "ranking_sample_count": ranking_count,
            "normalization": {
                "source_range": [0, 255],
                "target_range": [codec.format.value_min, codec.format.value_max],
            },
            "trainable_scalars": sum(spec.size for spec in adapter.parameter_specs()),
            "logical_bits": len(identities),
        },
        "quantization": codec.format.to_dict(),
        "fault_model": asdict(fault_model),
        "configuration": {
            "trials": trials,
            "ber": ber,
            "coverage": coverage,
            "random_ranking_seed": random_ranking_seed,
            "fault_seed_first": fault_seed,
            "fault_seed_last": fault_seed + trials - 1,
        },
        "baseline": {
            "checkpoint": asdict(checkpoint_baseline),
        },
        "quantization_verification": {
            "maximum_absolute_quantization_error": max(quantization_errors.values()),
            "per_parameter_maximum_absolute_error": quantization_errors,
            "encoded_values_round_trip": all(
                np.array_equal(codec.encode(codec.decode(codec.encode(value))), codec.encode(value))
                for value in checkpoint_state.values()
            ),
        },
        "rankings": {
            qpolar.estimator: {"sha256": qpolar.digest(), **qpolar.metadata},
            msb.estimator: {"sha256": msb.digest(), **msb.metadata},
            random.estimator: {"sha256": random.digest(), **random.metadata},
            first_order.estimator: {"sha256": first_order.digest(), **first_order.metadata},
        },
        "policies": policy_results,
        "schedule_hashes": [schedule.digest() for schedule in schedules],
        "exact_artifact_sha256": {
            "rankings": digest_object(exact_rankings),
            "protection_masks": digest_object(exact_masks),
            "fault_schedules": digest_object(exact_schedules),
        },
        "checks": {
            "codec_exhaustive": _codec_exhaustive(codec),
            "full_msb_mask_equivalent": qpolar_full_msb.protected == msb_full.protected,
            "state_restored": pre_trials_state == post_trials_state,
            "matched_schedules": all(len(hashes) == 1 for hashes in schedule_by_trial.values()),
        },
    }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"mnist-lite-milestone-{timestamp}-r{repeat_index}"
    store = ResultStore(output_root, run_id)
    manifest = {
        "schema": "netsurf-v2-run-manifest/v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "scientific_payload_sha256": digest_object(scientific_payload),
        "source": git_provenance(root),
        "runtime": runtime_provenance(),
        "artifacts": {
            "checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint)},
            "dataset": {"path": str(dataset), "sha256": sha256_file(dataset)},
        },
    }
    store.write_json("manifest.json", manifest)
    store.write_json("scientific_payload.json", scientific_payload)
    store.write_json("trials.json", trial_rows)
    store.write_json("rankings.json", exact_rankings)
    store.write_json("protection_masks.json", exact_masks)
    store.write_json("fault_schedules.json", exact_schedules)
    store.write_text("report.md", _report_markdown(scientific_payload))
    digest = store.finalize(scientific_payload)
    return store.path, digest, scientific_payload


def execute_once(
    root: Path,
    output_root: Path,
    *,
    repeat_index: int,
    checkpoint: Path | None = None,
    dataset: Path | None = None,
    sample_count: int = 256,
    ranking_count: int = 256,
    trials: int = 12,
    ber: float = 0.001,
    coverage: float = 0.10,
    random_ranking_seed: int = 23011,
    fault_seed: int = 910_000,
) -> tuple[Path, str, dict[str, Any]]:
    """Run the historical file-layout API through the canonical engine.

    ``_legacy_execute_once_reference`` remains as audit evidence only. New CLI
    and API callers use versioned result bundles directly.
    """

    root = Path(root).resolve()
    output_root = Path(output_root).resolve()
    checkpoint = Path(checkpoint or artifact_root() / "legacy" / CHECKPOINT_RELATIVE).expanduser().resolve()
    if dataset is None:
        raise ValueError("dataset path must be provided explicitly")
    dataset = Path(dataset).expanduser().resolve()
    document = {
        "schema_version": "1.0.0",
        "scientific_protocol": "legacy-mnist-persistent-weight/v1",
        "experiment": {"name": "mnist_deterministic_milestone", "description": "Historical compatibility wrapper."},
        "benchmark": {"adapter": "legacy_mnist_lite", "id": "mnist_fnn_lite-20250414_152629-q6_2_1-pruned_0_125", "checkpoint": str(checkpoint), "dataset": str(dataset), "split": "keras_mnist_test_first_n", "evaluation_samples": sample_count, "ranking_samples": ranking_count, "quantization": "q<6,2,1>"},
        "estimators": [
            {"name": "legacy_qpolargrad", "implementation_version": "recovered-first-call/v1", "policy": "position_first", "result_name": "legacy_qpolargrad_position_first", "activation_source": "globally_attacked_first_call"},
            {"name": "msb_first", "implementation_version": "encoded-bit-index/v1", "policy": "global_ranking", "result_name": "msb_first"},
            {"name": "random", "implementation_version": "numpy-pcg64/v1", "policy": "global_ranking", "result_name": "random"},
            {"name": "first_order_gradient_delta", "implementation_version": "abs-gradient-times-exact-delta/v1", "policy": "global_ranking", "result_name": "first_order_gradient_delta"},
        ],
        "protection": {"coverage": coverage, "mitigation_model": "perfect_event_suppression"},
        "fault": {"target": "weight", "location": "logical_trainable_parameter_bits", "persistence": "persistent_trial", "distribution": "independent_bernoulli", "probability": ber},
        "execution": {"trials": trials, "confidence": 0.95, "random_ranking_seed": random_ranking_seed, "fault_seed": fault_seed},
        "output": {"root": str(output_root), "report": True, "plots": False},
    }
    configuration = validate_experiment_document(document, root / "compatibility-milestone.toml")
    outcome = execute_experiment(configuration)
    scientific_payload = dict(outcome.scientific_payload)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"mnist-lite-milestone-{timestamp}-r{repeat_index}"
    store = ResultStore(output_root, run_id)
    store.write_json("manifest.json", {"schema": "netsurf-v2-run-manifest/v1", "created_utc": datetime.now(timezone.utc).isoformat(), "run_id": run_id,
        "scientific_payload_sha256": digest_object(scientific_payload), "source": git_provenance(root), "runtime": runtime_provenance(),
        "artifacts": {"checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint)}, "dataset": {"path": str(dataset), "sha256": sha256_file(dataset)}}})
    store.write_json("scientific_payload.json", scientific_payload)
    store.write_json("trials.json", [row.to_dict() for row in outcome.trials])
    store.write_json("rankings.json", outcome.rankings)
    store.write_json("protection_masks.json", outcome.protection_masks)
    store.write_json("fault_schedules.json", outcome.fault_schedules)
    store.write_text("report.md", _report_markdown(scientific_payload))
    digest = store.finalize(scientific_payload)
    return store.path, digest, scientific_payload


def execute_repeated(
    root: Path,
    output_root: Path,
    *,
    repeats: int = 2,
    **kwargs,
) -> dict[str, Any]:
    if repeats < 2:
        raise ValueError("the milestone requires at least two executions")
    runs = [
        execute_once(root, output_root, repeat_index=index + 1, **kwargs)
        for index in range(repeats)
    ]
    digests = [digest for _, digest, _ in runs]
    if len(set(digests)) != 1:
        raise RuntimeError(f"repeated scientific payloads differ: {digests}")
    verification = {
        "schema": "netsurf-v2-repeat-verification/v1",
        "reproducible": True,
        "scientific_payload_sha256": digests[0],
        "runs": [str(path) for path, _, _ in runs],
    }
    verification_path = Path(output_root).resolve() / f"repeat-verification-{digests[0][:16]}.json"
    if verification_path.exists():
        raise FileExistsError(verification_path)
    verification_path.write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    verification_path.chmod(0o444)
    return verification
