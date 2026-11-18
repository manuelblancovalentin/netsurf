"""Scalar complete-file risk, deliberately separate from additive input losses.

No TensorFlow import, inference, or surrogate loss is needed to replay these
statistics. Normal/anomaly comparison pairs are NOT independent observations.
"""
from __future__ import annotations

import hashlib
import numpy as np

from .manifests import digest_object

RISK = "one_minus_file_roc_auc"
SOURCE = "dcase2020_toycar_official_development_test"
ROLES = ("conditional_calibration", "conditional_model_selection", "conditional_final_evaluation")
CONTRACT = {
    "schema_version": "1.0.0", "metric": RISK, "direction": "minimize",
    "unit": "one scalar per fixed complete-file cohort and persistent fault field",
    "file_score": "pinned float64 mean reconstruction MSE across all 196 windows; 640 features/window",
    "auc": "mean(normal_score < anomaly_score) + half credit for equal scores",
    "risk": "1 - file_ROC_AUC", "additive_input_decomposition": False,
    "uncertainty": "independent fault fields conditional on the entire fixed cohort; pools are separate units",
    "input_population_uncertainty": "not estimated; comparison pairs and windows are not independent units",
}


def score_digest(scores):
    """Explicit little-endian float64 bytes, including exact representations."""
    return hashlib.sha256(np.asarray(scores, dtype="<f8").tobytes()).hexdigest()


def pair_ordering(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.shape != labels.shape or set(labels.tolist()) != {0, 1}:
        raise ValueError("complete-file ROC-AUC requires aligned scores and both binary classes")
    if not np.isfinite(scores).all():
        raise ValueError("nonfinite file anomaly score")
    return np.sign(scores[labels == 1, None] - scores[None, labels == 0]).astype(np.int8)


def cohort_risk(labels, scores):
    order = pair_ordering(labels, scores)
    return float(1 - np.mean((order > 0).astype(np.float64) + .5 * (order == 0)))


def ordering_diagnostics(labels, scores, clean_scores=None):
    order = pair_ordering(labels, scores)
    result = {"normal_anomaly_pairs": int(order.size),
        "concordant": int(np.count_nonzero(order > 0)),
        "discordant": int(np.count_nonzero(order < 0)),
        "ties": int(np.count_nonzero(order == 0)),
        "risk": cohort_risk(labels, scores),
        "independent_observation_count": None,
        "interpretation": "descriptive dependent pair counts, not binomial sample size"}
    if clean_scores is not None:
        clean = pair_ordering(labels, clean_scores)
        result.update(strict_reversals=int(np.count_nonzero(order * clean == -1)),
                      ordering_changes_including_ties=int(np.count_nonzero(order != clean)),
                      risk_minus_clean=cohort_risk(labels, scores) - cohort_risk(labels, clean_scores))
    return result


def validate_cohorts(raw, samples):
    if not isinstance(raw, dict) or set(raw) != set(ROLES):
        raise ValueError(f"cohorts must specify exactly {ROLES}")
    counts = (samples.calibration_count, samples.model_selection_count, samples.evaluation_count)
    offsets = (samples.calibration_offset, samples.model_selection_offset, samples.evaluation_offset)
    seen = set()
    for role, count, offset in zip(ROLES, counts, offsets):
        values = raw[role]
        if not isinstance(values, list) or len(values) != count or offset != 0:
            raise ValueError("explicit cohorts require matching file counts and zero offsets")
        if any(type(i) is not int or not 0 <= i < 248 for i in values) or len(set(values)) != count:
            raise ValueError("cohort indices must be unique original official file indices in [0,248)")
        if seen & set(values):
            raise ValueError("complete-file calibration/selection/final cohorts overlap")
        if not any(i < 140 for i in values) or not any(i >= 140 for i in values):
            raise ValueError("every complete-file cohort must contain both normal and anomalous files")
        seen.update(values)
    return {role: list(raw[role]) for role in ROLES}


def exposure_check(audit, cohorts, purpose):
    if purpose not in {"engineering", "confirmation"}:
        raise ValueError("unsupported cohort study purpose")
    overlaps = {role: sorted(set(cohorts[role]) & set(audit["exposed_test_indices"])) for role in ROLES[1:]}
    if purpose == "confirmation" and any(overlaps.values()):
        raise ValueError(f"previous ToyCar fault-outcome exposure overlaps confirmation cohorts: {overlaps}")
    return {"ledger_sha256": digest_object(audit), "source_split": SOURCE,
        "requested_original_indices": list(cohorts[ROLES[2]]),
        "cohorts": cohorts, "prior_overlap_by_role": overlaps, "purpose": purpose,
        "limitations": audit["limitations"]}


def configuration_exposure(configuration, *, allowed_campaign=None):
    from .conditional_benchmarks import conditional_capability, audit_exposure
    cap = conditional_capability(configuration.benchmark.adapter)
    return exposure_check(audit_exposure(cap, allowed_campaign=allowed_campaign),
                          configuration.cohorts, configuration.workflow.get("purpose", "engineering"))
