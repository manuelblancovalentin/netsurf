"""Bounded PrioriFI scheduling and derived-allocation engineering smoke.

Run with ``python -m netsurf.priorifi_numpy_reference --output DIR``. This
experiment uses disjoint packaged calibration, bit-validation, and final input
sets. Its 102-bit size makes exhaustive source-style probing inexpensive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .adapters.numpy_reference import NumpyReferenceAdapter
from .faults import FaultModel, FaultSchedule, apply_persistent_schedule_encoded_state
from .fkeras_baseline import repository_weight_hessian_score
from .priorifi import (
    PRIORIFI_SCHEDULER_VERSION, PRIORIFI_SOURCE_COMMIT,
    measured_bit_allocation, probe_schedule, significance_tiers,
)
from .protection import ProtectionMask
from .quantization import BitCodec, FixedPointFormat


ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "benchmarks/data/numpy_reference_checkpoint.json"
DATASET = ROOT / "benchmarks/data/numpy_reference_dataset.json"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mask(name, identities, universe_size):
    members = frozenset(identities)
    if len(members) != len(tuple(identities)):
        raise ValueError("protection mask contains duplicates")
    return ProtectionMask(name, members, universe_size, float(len(members)), {})


def run(*, last_k: int = 3, budget: int = 17, probability: float = 0.05,
        trials: int = 12, fault_seed: int = 920000) -> dict:
    """Return a complete in-memory engineering receipt; never touch old runs."""

    if not 0 <= probability <= 1 or trials < 1:
        raise ValueError("probability must be in [0,1] and trials positive")
    start = perf_counter()
    codec = BitCodec(FixedPointFormat.parse("q<6,2,1>"))
    adapter = NumpyReferenceAdapter(CHECKPOINT, DATASET, codec)
    adapter.load()
    original = adapter.values()
    identities = adapter.bit_identities()
    calibration_x, calibration_y, calibration = adapter.sample_partition("ranking", 2)
    validation_x, validation_y, validation = adapter.sample_partition("empirical_validation", 3)
    final_x, final_y, final = adapter.sample_partition("evaluation", 3)
    clean_calibration_prediction = adapter.predict_classes(calibration_x)
    clean_correct = clean_calibration_prediction == calibration_y
    if not np.any(clean_correct):
        raise ValueError("PrioriFI prediction-change calibration has no clean-correct inputs")
    # This exact 17-dimensional dense reference implements the public source's
    # repository-form FKeras initialization, not the paper-form mask baseline.
    hessian = adapter.exact_hessian(calibration_x, calibration_y)
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    chosen = sorted(range(len(eigenvalues)), key=lambda index: -abs(eigenvalues[index]))[:8]
    selected_vectors = eigenvectors[:, chosen]
    residuals = np.linalg.norm(hessian @ selected_vectors -
                               selected_vectors * eigenvalues[chosen], axis=0)
    # The eighth selected mode is numerically null on two calibration samples.
    # Use a matrix-scale backward error so a tiny eigenvalue is not divided by
    # another tiny roundoff quantity; its contribution to the score is tiny.
    normalized_residuals = residuals / np.maximum(
        np.maximum(np.linalg.norm(hessian, ord=2),
                   np.linalg.norm(hessian @ selected_vectors, axis=0) + np.abs(eigenvalues[chosen])),
        1e-30)
    orthogonality = np.linalg.norm(selected_vectors.T @ selected_vectors - np.eye(len(chosen)))
    if np.any(normalized_residuals > 1e-10) or orthogonality > 1e-10:
        raise ValueError("NumPy Hessian seed eigenpairs failed numerical validation")
    names = tuple(original)
    theta = np.concatenate([original[name].reshape(-1) for name in names])
    scores = repository_weight_hessian_score(theta, selected_vectors.T, eigenvalues[chosen])
    words = [(name, flat) for name in names for flat in range(original[name].size)]
    ranked_words = tuple(word for _, word in sorted(
        zip(scores, words), key=lambda pair: (-float(pair[0]), pair[1])
    ))
    tiers = significance_tiers(identities, ranked_words)
    calibration_observations = {}

    def probe(bit):
        changed = adapter.values()
        scalar = changed[bit.parameter].reshape(-1)
        scalar[bit.flat_index] = codec.decode(codec.flip(
            codec.encode(scalar[bit.flat_index]), bit.bit_index
        ))
        with adapter.temporary_values(changed):
            prediction = adapter.predict_classes(calibration_x)
            damage = int(np.count_nonzero(prediction[clean_correct] !=
                                          clean_calibration_prediction[clean_correct]))
            loss_change = adapter.evaluate(calibration_x, calibration_y).categorical_crossentropy
        calibration_observations[bit] = loss_change
        return damage

    clean_calibration_loss = adapter.evaluate(calibration_x, calibration_y).categorical_crossentropy
    records = probe_schedule(tiers, probe, last_k=last_k)
    if len(records) != len(identities) or set(record.identity for record in records) != set(identities):
        raise RuntimeError("PrioriFI probe schedule did not cover all encoded bits")
    if any(not np.array_equal(original[name], adapter.values()[name]) for name in names):
        raise RuntimeError("state was not restored after PrioriFI probing")
    if not 0 <= budget <= len(records):
        raise ValueError("budget must be in [0,102]")
    derived = measured_bit_allocation(records, budget)
    msb = tuple(sorted(identities, key=lambda bit: (-bit.bit_index, bit)))[:budget]
    masks = {
        "unprotected": _mask("unprotected", (), len(identities)),
        "msb_first": _mask("msb_first", msb, len(identities)),
        "priorifi_derived_measured_prediction_change": _mask(
            "priorifi_derived_measured_prediction_change", derived, len(identities)),
        "full_protection": _mask("full_protection", identities, len(identities)),
    }
    validation_effects = adapter.exhaustive_single_bit_impacts(validation_x, validation_y)
    clean_final = adapter.evaluate(final_x, final_y)
    fault_model = FaultModel("weight", "logical_trainable_parameter_bits",
                             "persistent_trial", "independent_bernoulli", probability)
    fault_trials = []
    for trial in range(trials):
        schedule = FaultSchedule.generate(fault_model, identities, seed=fault_seed + trial,
                                          inference_count=1)
        for name, mask in masks.items():
            corrupted, accounting = apply_persistent_schedule_encoded_state(
                original, schedule, mask, codec)
            with adapter.temporary_values(corrupted):
                evaluation = adapter.evaluate(final_x, final_y)
            fault_trials.append({
                "trial": trial, "fault_seed": schedule.seed, "field_sha256": schedule.digest(),
                "method": name, "proposed": accounting.proposed_events,
                "applied": accounting.applied_events, "suppressed": accounting.suppressed_events,
                "loss": evaluation.categorical_crossentropy, "accuracy": evaluation.accuracy,
            })
    if any(not np.array_equal(original[name], adapter.values()[name]) for name in names):
        raise RuntimeError("state was not restored after trial evaluations")
    if any(row["proposed"] != row["applied"] + row["suppressed"] for row in fault_trials):
        raise RuntimeError("fault event accounting failed")
    means = {name: {
        "mean_loss": float(np.mean([row["loss"] for row in fault_trials if row["method"] == name])),
        "mean_accuracy": float(np.mean([row["accuracy"] for row in fault_trials if row["method"] == name])),
    } for name in masks}
    return {
        "study_type": "bounded_engineering_smoke_not_publication_evidence",
        "scheduler": PRIORIFI_SCHEDULER_VERSION,
        "source_commit": PRIORIFI_SOURCE_COMMIT,
        "initialization": "repository-form FKeras top-8 exact continuous 17x17 Hessian; descending score",
        "initialization_diagnostics": {
            "eigenvalues": [float(eigenvalues[index]) for index in chosen],
            "normalized_residuals": normalized_residuals.tolist(),
            "orthogonality_error": float(orthogonality),
        },
        "feedback": "clean-correct calibration prediction-change count per isolated exact XOR bit",
        "derived_allocation": "top observed prediction-change count; sequence tie-break; not published PrioriFI",
        "checkpoint_sha256": _digest(CHECKPOINT), "dataset_sha256": _digest(DATASET),
        "sample_partitions": {role: {"indices": list(part.indices), "sha256": part.sample_sha256}
                              for role, part in (("calibration", calibration),
                                                 ("bit_validation", validation), ("final", final))},
        "population_bits": len(identities), "probability": probability,
        "budget_bits": budget, "last_k": last_k, "fault_seed": fault_seed,
        "clean_calibration_loss": clean_calibration_loss,
        "clean_final_loss": clean_final.categorical_crossentropy,
        "clean_final_accuracy": clean_final.accuracy,
        "probe_records": [{"sequence": record.sequence, "bit": record.identity.key(),
                           "tier": record.tier, "prediction_changes": record.observed_metric,
                           "calibration_loss_change": calibration_observations[record.identity]
                           - clean_calibration_loss,
                           "validation_loss_change": validation_effects[record.identity]}
                          for record in records],
        "masks": {name: {"sha256": mask.digest(), "bits": [bit.key() for bit in sorted(mask.protected)]}
                  for name, mask in masks.items()},
        "fault_trials": fault_trials, "means": means,
        "runtime_seconds": perf_counter() - start,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new write-once directory")
    parser.add_argument("--last-k", type=int, default=3)
    parser.add_argument("--budget", type=int, default=17)
    parser.add_argument("--probability", type=float, default=0.05)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--fault-seed", type=int, default=920000)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; completed engineering evidence is write-once")
    receipt = run(last_k=args.last_k, budget=args.budget, probability=args.probability,
                  trials=args.trials, fault_seed=args.fault_seed)
    args.output.mkdir(parents=True, exist_ok=False)
    payload = json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    artifact = args.output / "engineering_receipt.json"
    artifact.write_text(payload, encoding="utf-8")
    (args.output / "checksums.json").write_text(
        json.dumps({"engineering_receipt.json": hashlib.sha256(payload.encode()).hexdigest()}, indent=2)
        + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "population_bits": receipt["population_bits"],
                      "probe_count": len(receipt["probe_records"]),
                      "fault_evaluations": len(receipt["fault_trials"]),
                      "runtime_seconds": receipt["runtime_seconds"], "means": receipt["means"]},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
