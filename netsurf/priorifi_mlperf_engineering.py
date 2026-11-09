"""Bounded, non-publication MLPerf test of PrioriFI-derived allocation.

Uses pinned TensorFlow Lite reference inference, a source-style FKeras repository
seed for probe order, exact isolated INT8 XOR, and a separate fixed-budget
NetSuRF mask-swap rule. Never reads untouched final-evaluation examples.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .adapters.mlperf_tiny_multibench import MlperfTinyKwsDscnnAdapter, MlperfTinyToyCarAdapter
from .adapters.mlperf_tiny import MlperfTinyImageClassificationAdapter
from .fkeras_baseline import leading_hessian_eigenpairs, repository_weight_hessian_score
from .fkeras_estimator import ContinuousTFLiteHessian
from .faults import FaultEvent, FaultModel, FaultSchedule
from .identities import BitIdentity
from .priorifi import (
    PRIORIFI_SCHEDULER_VERSION, PRIORIFI_SOURCE_COMMIT,
    candidate_tiers, cost_capped_local_allocation, probe_schedule,
)
from .protection import ProtectionMask


ASSETS = Path("/projects/kona-assets/manuelbv/locks/mlperf-tiny")
MODELS = Path("/data/models/models--mlcommons--tiny/snapshots/4addd0fa08d216e20637637874e084895f289da4")
MODEL = MODELS / "benchmark/training/keyword_spotting/trained_models/kws_ref_model.tflite"
DATA = Path("/data/datasets/speech_commands/0.0.3")
RESNET_MODEL = Path("/data/models/models--mlcommons--tiny/blobs/3c002613d1b2475eb51dd78dfb85a546c8ae658dee71cf6ade43b022fe205415")
RESNET_DATA = Path("/data/datasets/cifar10/cifar-10-python.tar.gz")
TOYCAR_MODEL = Path("/data/models/models--mlcommons--tiny/blobs/87cf24194ef93d1d9b11a591d805526b98008e351655d29883c825c9c106ba24")
TOYCAR_DATA = Path("/data/datasets/dcase2020-toycar")


def _adapter(benchmark: str):
    if benchmark == "kws":
        adapter = MlperfTinyKwsDscnnAdapter(MODEL, DATA, asset_lock=ASSETS / "kws.lock.json")
        lock = ASSETS / "kws.lock.json"
    elif benchmark == "resnet8":
        adapter = MlperfTinyImageClassificationAdapter(
            RESNET_MODEL, RESNET_DATA, asset_lock=ASSETS / "ic.lock.json",
            injectable_parameters=("kernels",),
        )
        lock = ASSETS / "ic.lock.json"
    elif benchmark == "toycar":
        adapter = MlperfTinyToyCarAdapter(
            TOYCAR_MODEL, TOYCAR_DATA, asset_lock=ASSETS / "ad.lock.json",
        )
        lock = ASSETS / "ad.lock.json"
    else:
        raise ValueError("bounded runner supports only kws, resnet8 or toycar")
    adapter.load()
    return adapter, lock


def _source_seed_word_scores(adapter, inputs, labels, identities):
    """Pinned PrioriFI source's repository-form Hessian seed, not a paper baseline."""
    operator = ContinuousTFLiteHessian(adapter, inputs, labels)
    vectors, diagnostics = leading_hessian_eigenpairs(
        operator, len(operator.theta), count=8, seed=120802,
        max_iterations=500, max_products=2000, tolerance=1e-5,
    )
    weights = repository_weight_hessian_score(
        operator.theta, vectors, np.asarray(diagnostics.eigenvalues),
    )
    offsets = {row["parameter"]: row["flat_offset"] for row in operator.mapping}
    words = {(bit.parameter, bit.flat_index) for bit in identities}
    return {word: abs(float(weights[offsets[word[0]] + word[1]])) for word in words}, asdict(diagnostics)


def _mask(name: str, bits, population: int) -> ProtectionMask:
    return ProtectionMask(name, frozenset(bits), population, float(len(bits)), {})


def run(*, benchmark: str = "kws", calibration_count: int = 32, candidate_seed: int = 120801,
        max_probes: int = 24, sign_per_status: int = 16, lower_per_tier: int = 4,
        last_k: int = 3, fault_seed: int = 121000,
        trials: int = 4, ber: float = 0.001) -> dict:
    candidate_count = 2 * sign_per_status + 7 * lower_per_tier
    if not (1 <= calibration_count <= 64 and 1 <= sign_per_status <= 256 and
            1 <= lower_per_tier <= 64 and 8 <= max_probes <= min(candidate_count, 512) and
            1 <= trials <= 16 and 0 <= ber <= 0.01):
        raise ValueError("engineering bounds exceeded")
    started = perf_counter()
    adapter, lock = _adapter(benchmark)
    source_content = adapter._model_content
    identities = adapter.bit_identities()
    # ToyCar needs both classes for AUC feedback. Its empirical-validation and
    # evaluation roles are disjoint halves of the already exposed official files.
    calibration_role = "empirical_validation" if benchmark == "toycar" else "ranking"
    calibration_x, calibration_y, calibration = adapter.sample_partition(calibration_role, calibration_count)
    engineering_count = 4 if benchmark == "toycar" else 8
    engineering_x, engineering_y, engineering = adapter.sample_partition("evaluation", engineering_count)
    if (calibration.source_split == engineering.source_split and
            set(calibration.indices) & set(engineering.indices)):
        raise ValueError("calibration and engineering-evaluation identities overlap")
    if benchmark == "toycar":
        clean_calibration_risk = 1.0 - adapter.evaluate(calibration_x, calibration_y).roc_auc
    else:
        clean_calibration_classes = np.argmax(adapter.predict_encoded_outputs(calibration_x), axis=1)
        clean_correct = clean_calibration_classes == calibration_y
        if not np.any(clean_correct):
            raise ValueError("no clean-correct calibration inputs")
    before_hessian = perf_counter()
    word_score, hessian_diagnostics = _source_seed_word_scores(
        adapter, calibration_x, calibration_y, identities,
    )
    if not all(np.isfinite(score) for score in word_score.values()):
        raise ValueError("nonfinite source-seed Hessian word score")
    hessian_seconds = perf_counter() - before_hessian
    ranked_words = tuple(sorted(word_score, key=lambda word: (-word_score[word], word)))
    baseline_order = tuple(sorted(identities, key=lambda bit: (-bit.bit_index, bit)))
    budget = len(identities) // 16  # 6.25%; an informative partial INT8 tier
    baseline = frozenset(baseline_order[:budget])
    rng = np.random.default_rng(candidate_seed)
    pool = []
    sampling = []
    for bit_position in range(7, -1, -1):
        for protected, count in ((True, sign_per_status if bit_position == 7 else 0),
                                 (False, sign_per_status if bit_position == 7 else lower_per_tier)):
            if count == 0:
                continue
            stratum = [bit for bit in identities
                       if bit.bit_index == bit_position and (bit in baseline) == protected]
            if len(stratum) < count:
                raise ValueError("candidate stratum has insufficient eligible bits")
            selected = rng.choice(len(stratum), count, replace=False)
            pool.extend(stratum[int(index)] for index in selected)
            sampling.append({"bit_position": bit_position, "baseline_protected": protected,
                             "population_count": len(stratum), "selected_count": count,
                             "inclusion_probability": count / len(stratum)})
    tiers = candidate_tiers(identities, pool, ranked_words)
    if sum(map(len, tiers)) != candidate_count:
        raise RuntimeError("unexpected candidate-pool cardinality")
    probes = []

    def exact_probe(bit: BitIdentity) -> float:
        _, _, _, _, changed = adapter.encoded_bit_transition(bit)
        with adapter.temporary_values(changed):
            if benchmark == "toycar":
                # Explicit task adaptation: source prediction-change feedback
                # is undefined for reconstruction. This is complete-file AUC.
                signed_damage = (1.0 - adapter.evaluate(calibration_x, calibration_y).roc_auc) - clean_calibration_risk
                feedback = max(0.0, signed_damage)
                probes.append({"bit": bit.key(), "auc_risk_change": signed_damage,
                               "nonnegative_feedback": feedback})
            else:
                classes = np.argmax(adapter.predict_encoded_outputs(calibration_x), axis=1)
                feedback = int(np.count_nonzero(classes[clean_correct] !=
                                                 clean_calibration_classes[clean_correct]))
                probes.append({"bit": bit.key(), "prediction_changes": feedback})
        return feedback

    before_probes = perf_counter()
    records = probe_schedule(tiers, exact_probe, last_k=last_k, max_probes=max_probes)
    probe_seconds = perf_counter() - before_probes
    derived = cost_capped_local_allocation(baseline_order, baseline, records)
    if len(derived) != budget or adapter._model_content != source_content:
        raise RuntimeError("mask cardinality or exact restoration failure")
    masks = {"msb_first": _mask("msb_first", baseline, len(identities)),
             "priorifi_derived_local": _mask("priorifi_derived_local", derived, len(identities))}
    clean = adapter.evaluate(engineering_x, engineering_y)
    fault_model = FaultModel("weight", "stored_int8_kernel_bits", "persistent_trial",
                             "independent_bernoulli", ber)
    evaluations = []
    for index in range(trials):
        schedule = FaultSchedule.generate(fault_model, identities,
                                          seed=fault_seed + index, inference_count=1)
        for name, mask in masks.items():
            measured, report = adapter.evaluate_fault_schedule(
                engineering_x, engineering_y, schedule, mask)
            row = {"trial": index, "method": name, "field_sha256": schedule.digest(),
                   "proposed": report.proposed_events,
                                "applied": report.applied_events,
                                "suppressed": report.suppressed_events}
            if benchmark == "toycar":
                row.update({"risk_1_minus_auc": 1.0 - measured.roc_auc,
                            "file_roc_auc": measured.roc_auc,
                            "reconstruction_mse": measured.reconstruction_mse})
            else:
                row.update({"loss": measured.categorical_crossentropy,
                            "accuracy": measured.accuracy})
            evaluations.append(row)
            if row["proposed"] != row["applied"] + row["suppressed"]:
                raise RuntimeError("fault-event accounting failure")
    if adapter._model_content != source_content or any(
        evaluations[i]["field_sha256"] != evaluations[i + 1]["field_sha256"]
        for i in range(0, len(evaluations), 2)
    ):
        raise RuntimeError("fault pairing or reference model restoration failure")
    changed = sorted(b.key() for b in baseline ^ derived)
    return {
        "status": "bounded_engineering_only_not_publication_effectiveness",
        "source_scheduler_commit": PRIORIFI_SOURCE_COMMIT,
        "scheduler": PRIORIFI_SCHEDULER_VERSION,
        "allocation": "cost-capped local measured-bit swaps around MSB; NetSuRF adaptation",
        "benchmark": adapter.benchmark_id,
        "checkpoint_sha256": (
            adapter.model_sha256 if benchmark != "resnet8"
            else hashlib.sha256(RESNET_MODEL.read_bytes()).hexdigest()
        ),
        "asset_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
        "population_bits": len(identities), "budget_bits": budget,
        "candidate_count": len(pool), "candidate_seed": candidate_seed,
        "candidate_sampling": sampling,
        "candidate_bits": [bit.key() for bit in sorted(pool)],
        "calibration": calibration.to_dict(), "engineering_evaluation": engineering.to_dict(),
        "hessian_seed": "repository-form top-8 continuous companion; only the probe order, not the published-form FKeras protection baseline",
        "hessian_seconds": hessian_seconds, "hessian_diagnostics": hessian_diagnostics,
        "probe_limit": max_probes, "last_k": last_k,
        "probe_seconds": probe_seconds,
        "probes": probes,
        "masks": {name: {"sha256": mask.digest(), "protected_bits": len(mask.protected)}
                  for name, mask in masks.items()},
        "changed_bit_identities": changed,
        "feedback_definition": ("positive complete-file AUC-risk change; task adaptation, not source metric"
                                if benchmark == "toycar" else
                                "changed predictions among initially correct calibration inputs; source metric"),
        "clean_engineering": (
            {"risk_1_minus_auc": 1.0 - clean.roc_auc, "file_roc_auc": clean.roc_auc,
             "reconstruction_mse": clean.reconstruction_mse}
            if benchmark == "toycar" else
            {"loss": clean.categorical_crossentropy, "accuracy": clean.accuracy}
        ),
        "fault": {"ber": ber, "seed": fault_seed, "trials": trials},
        "evaluations": evaluations,
        "elapsed_seconds": perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--benchmark", choices=("kws", "resnet8", "toycar"), default="kws")
    parser.add_argument("--calibration-count", type=int, default=32)
    parser.add_argument("--max-probes", type=int, default=24)
    parser.add_argument("--sign-per-status", type=int, default=16)
    parser.add_argument("--lower-per-tier", type=int, default=4)
    parser.add_argument("--trials", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; engineering receipts are write-once")
    receipt = run(benchmark=args.benchmark, calibration_count=args.calibration_count,
                  max_probes=args.max_probes, sign_per_status=args.sign_per_status,
                  lower_per_tier=args.lower_per_tier, trials=args.trials)
    args.output.mkdir(parents=True, exist_ok=False)
    raw = (json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    (args.output / "receipt.json").write_bytes(raw)
    (args.output / "checksums.json").write_text(json.dumps({"receipt.json": hashlib.sha256(raw).hexdigest()}) + "\n")
    print(json.dumps({"output": str(args.output), "elapsed_seconds": receipt["elapsed_seconds"],
                      "probe_count": len(receipt["probes"]),
                      "changed_bits": len(receipt["changed_bit_identities"]),
                      "evaluations": len(receipt["evaluations"])}, sort_keys=True))


if __name__ == "__main__":
    main()
