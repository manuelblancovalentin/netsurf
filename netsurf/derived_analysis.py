"""Presentation-only aggregation of saved protection-trial measurements.

This module never generates faults or changes the scientific payload. Its
inputs are immutable per-trial rows, so outputs can be regenerated beside an
existing finalized bundle.
"""

from __future__ import annotations

from collections import defaultdict
import itertools
from typing import Any, Iterable, Mapping

import numpy as np

from .analysis_m3 import holm_adjust, paired_permutation_test
from .statistics import summarize


def split_condition(label: str, default_ber: float | None = None) -> tuple[float, str, str, float]:
    condition, ber = label, default_ber
    if label.startswith("ber="):
        prefix, condition = label.split("/", 1)
        ber = float(prefix[4:])
    if ber is None:
        raise ValueError(f"trial label {label!r} does not identify a logical BER")
    if condition in {"no_protection", "full_protection"}:
        return float(ber), condition, condition, 0.0 if condition == "no_protection" else 1.0
    try:
        policy, coverage_text = condition.rsplit("@", 1)
        coverage = float(coverage_text)
    except (ValueError, TypeError) as error:
        raise ValueError(f"trial label {label!r} does not identify a protection coverage") from error
    return float(ber), condition, policy, coverage


def normalize_trials(rows: Iterable[Mapping[str, Any]], *, default_ber: float | None = None):
    normalized = []
    for source in rows:
        label = str(source["policy"])
        ber, condition, policy, coverage = split_condition(label, default_ber)
        row = {
            **source,
            "replication": str(source.get("replication") or "replication_0"),
            "logical_ber": ber,
            "condition": condition,
            "ranking_policy": policy,
            "coverage": coverage,
            "trial": int(source["trial"]),
            "accuracy": float(source["accuracy"]),
            "categorical_crossentropy": float(source["categorical_crossentropy"]),
            "proposed_events": int(source["proposed_events"]),
            "applied_events": int(source["applied_events"]),
            "suppressed_events": int(source["suppressed_events"]),
            "unique_applied_bits": int(source["unique_applied_bits"]),
            "schedule_sha256": str(source.get("schedule_sha256") or source.get("schedule_hash") or ""),
        }
        if not np.isfinite(row["accuracy"]) or not np.isfinite(row["categorical_crossentropy"]):
            raise ValueError("saved trial measurements contain a non-finite accuracy or loss")
        normalized.append(row)
    return tuple(normalized)


def _descriptive(values, confidence: float, *, bounds=None) -> dict[str, Any]:
    data = np.asarray(tuple(values), dtype=np.float64)
    result = summarize(data, confidence=confidence, bounds=bounds).to_dict()
    result.update({
        "median": float(np.median(data)), "quantile_05": float(np.quantile(data, 0.05)),
        "quantile_95": float(np.quantile(data, 0.95)),
        "minimum": float(np.min(data)), "maximum": float(np.max(data)),
    })
    return result


def _summary_row(rows: list[Mapping[str, Any]], confidence: float) -> dict[str, Any]:
    identities = {
        (row["replication"], row["logical_ber"], row["trial"], row["schedule_sha256"])
        for row in rows
    }
    mask_hashes = {str(row["ranking_mask_sha256"]) for row in rows if row.get("ranking_mask_sha256")}
    protected_counts = {int(row["protected_bits"]) for row in rows if row.get("protected_bits") is not None}
    realized_coverages = {float(row["realized_coverage"]) for row in rows if row.get("realized_coverage") is not None}
    return {
        "logical_ber": rows[0]["logical_ber"], "condition": rows[0]["condition"],
        "policy": rows[0]["ranking_policy"], "coverage": rows[0]["coverage"],
        "trial_count": len(rows), "fault_realization_count": len(identities),
        "replication_count": len({row["replication"] for row in rows}),
        "ranking_mask_count": len(mask_hashes) if mask_hashes else None,
        "ranking_mask_sha256": next(iter(mask_hashes)) if len(mask_hashes) == 1 else None,
        "protected_bits": next(iter(protected_counts)) if len(protected_counts) == 1 else None,
        "realized_coverage": next(iter(realized_coverages)) if len(realized_coverages) == 1 else rows[0]["coverage"],
        "accuracy": _descriptive((row["accuracy"] for row in rows), confidence, bounds=(0.0, 1.0)),
        "categorical_crossentropy": _descriptive((row["categorical_crossentropy"] for row in rows), confidence),
        "proposed_events": _descriptive((row["proposed_events"] for row in rows), confidence),
        "applied_events": _descriptive((row["applied_events"] for row in rows), confidence),
        "suppressed_events": _descriptive((row["suppressed_events"] for row in rows), confidence),
        "unique_applied_bits": _descriptive((row["unique_applied_bits"] for row in rows), confidence),
    }


def _paired(rows, *, confidence: float, permutation_seed: int):
    by_condition = defaultdict(dict)
    controls = defaultdict(dict)
    for row in rows:
        key = (row["replication"], row["logical_ber"], row["trial"], row["schedule_sha256"])
        if row["ranking_policy"] in {"no_protection", "full_protection"}:
            controls[(row["logical_ber"], row["ranking_policy"])][key] = row
        else:
            by_condition[(row["logical_ber"], row["coverage"], row["ranking_policy"])][key] = row
    specs, grouped = [], defaultdict(list)
    for ber, coverage, policy in sorted(by_condition):
        grouped[(ber, coverage)].append(policy)
        specs.append((ber, coverage, policy, "no_protection"))
    for (ber, coverage), policies in sorted(grouped.items()):
        specs.extend((ber, coverage, left, right) for left, right in itertools.combinations(sorted(policies), 2))
    pooled, replication_rows, differences = [], [], []
    raw_p = defaultdict(dict)
    for spec_index, (ber, coverage, left, right) in enumerate(specs):
        left_rows = by_condition[(ber, coverage, left)]
        right_rows = controls[(ber, right)] if right == "no_protection" else by_condition[(ber, coverage, right)]
        keys = sorted(set(left_rows) & set(right_rows))
        if not keys:
            continue
        comparison = f"{left}@{coverage:.6f}__minus__{right}" + ("" if right == "no_protection" else f"@{coverage:.6f}")
        for metric_index, metric in enumerate(("accuracy", "categorical_crossentropy")):
            values = [float(left_rows[key][metric]) - float(right_rows[key][metric]) for key in keys]
            estimate = _descriptive(values, confidence, bounds=(-1.0, 1.0) if metric == "accuracy" else None)
            name = f"{comparison}:{metric}"
            p_value = _derived_permutation_test(values, seed=permutation_seed + spec_index * 2 + metric_index)
            raw_p[(ber, metric)][name] = p_value
            pooled.append({
                "logical_ber": ber, "coverage": coverage, "metric": metric,
                "comparison": comparison, "left": left, "right": right,
                "matched_pairs": len(keys), "difference": estimate, "p_value_raw": p_value,
                "permutation_test": "paired two-sided sign-flip mean test",
                "pairing_identity": "replication, logical_ber, trial, schedule_sha256",
                "multiplicity_family": f"logical_ber={ber:.8g}, metric={metric}",
            })
            for key, difference in zip(keys, values):
                differences.append({
                    "logical_ber": ber, "coverage": coverage, "metric": metric,
                    "comparison": comparison, "left": left, "right": right,
                    "replication": key[0], "trial": key[2], "schedule_sha256": key[3],
                    "difference": difference,
                })
            by_replication = defaultdict(list)
            for key, difference in zip(keys, values):
                by_replication[key[0]].append(difference)
            for replication, rep_values in sorted(by_replication.items()):
                replication_rows.append({
                    "logical_ber": ber, "coverage": coverage, "metric": metric,
                    "comparison": comparison, "left": left, "right": right,
                    "replication": replication, "matched_pairs": len(rep_values),
                    "difference": _descriptive(rep_values, confidence, bounds=(-1.0, 1.0) if metric == "accuracy" else None),
                })
    for family, values in raw_p.items():
        adjusted = holm_adjust(values)
        for row in pooled:
            name = f"{row['comparison']}:{row['metric']}"
            if (row["logical_ber"], row["metric"]) == family and name in adjusted:
                row["p_value_holm"] = adjusted[name]
    return pooled, replication_rows, differences


def _derived_permutation_test(differences, *, seed: int, permutations: int = 10000) -> float:
    """Vectorized presentation-only equivalent of the paired sign-flip test."""
    data = np.asarray(differences, dtype=np.float64)
    if len(data) <= 20 and (1 << len(data)) <= permutations:
        return paired_permutation_test(data, seed=seed, permutations=permutations)
    observed = abs(float(np.mean(data)))
    generator = np.random.default_rng(int(seed))
    exceed = 0
    remaining = permutations
    while remaining:
        batch = min(1000, remaining)
        signs = generator.choice(np.asarray([-1.0, 1.0]), size=(batch, len(data)))
        exceed += int(np.count_nonzero(np.abs(np.mean(signs * data, axis=1)) >= observed - 1e-15))
        remaining -= batch
    return (exceed + 1) / (permutations + 1)


def derive_protection_analysis(
    trial_rows: Iterable[Mapping[str, Any]], *, confidence: float = 0.95,
    permutation_seed: int = 0, default_ber: float | None = None,
    expected_replications: int | None = None,
) -> dict[str, Any]:
    """Aggregate every saved replication without altering raw measurements."""
    rows = normalize_trials(trial_rows, default_ber=default_ber)
    if not rows:
        return {"pooled": [], "replications": [], "paired": [], "paired_replications": [], "paired_differences": []}
    observed = {row["replication"] for row in rows}
    if expected_replications is not None and len(observed) != expected_replications:
        raise ValueError(
            f"saved trials contain {len(observed)} replication identities; the resolved configuration requires {expected_replications}"
        )
    pooled_groups, replication_groups = defaultdict(list), defaultdict(list)
    for row in rows:
        pooled_groups[(row["logical_ber"], row["condition"])].append(row)
        replication_groups[(row["replication"], row["logical_ber"], row["condition"])].append(row)
    pooled = [_summary_row(group, confidence) for _, group in sorted(pooled_groups.items())]
    replications = [{"replication": key[0], **_summary_row(group, confidence)} for key, group in sorted(replication_groups.items())]
    paired, paired_replications, paired_differences = _paired(rows, confidence=confidence, permutation_seed=permutation_seed)
    return {
        "schema_version": "1.0.0",
        "statistical_unit": "fault realization identified by replication, logical BER, trial, and schedule SHA-256",
        "interval_interpretation": "fault-realization variability conditional on the recorded model, dataset, policies, and ranking masks",
        "pooled": pooled, "replications": replications, "paired": paired,
        "paired_replications": paired_replications, "paired_differences": paired_differences,
    }


def derive_task_protection_analysis(
    trial_rows: Iterable[Mapping[str, Any]], *, confidence: float = 0.95,
    permutation_seed: int = 0, default_ber: float | None = None,
    expected_replications: int | None = None,
) -> dict[str, Any]:
    """Aggregate saved task-generic trials without relabeling their metrics.

    The established replication/pairing implementation is reused internally,
    but its historical classification field names never escape this function.
    Every input row must describe the same primary and loss metric contract.
    """

    source = tuple(trial_rows)
    if not source:
        return {
            "pooled": [], "replications": [], "paired": [],
            "paired_replications": [], "paired_differences": [],
        }
    contracts = {
        (
            str(row["primary_metric_name"]), str(row["primary_metric_direction"]),
            str(row["loss_metric_name"]), str(row["loss_metric_direction"]),
        )
        for row in source
    }
    if len(contracts) != 1:
        raise ValueError("saved task trials contain inconsistent metric contracts")
    primary_name, primary_direction, loss_name, loss_direction = next(iter(contracts))
    mapped = tuple({
        **row,
        "accuracy": float(row["primary_metric_value"]),
        "categorical_crossentropy": float(row["loss_metric_value"]),
    } for row in source)
    result = derive_protection_analysis(
        mapped, confidence=confidence, permutation_seed=permutation_seed,
        default_ber=default_ber, expected_replications=expected_replications,
    )

    def aggregate(row):
        value = dict(row)
        value["primary_metric_name"] = primary_name
        value["primary_metric_direction"] = primary_direction
        value["loss_metric_name"] = loss_name
        value["loss_metric_direction"] = loss_direction
        value["primary_metric"] = value.pop("accuracy")
        value["loss_metric"] = value.pop("categorical_crossentropy")
        return value

    def comparison(row):
        value = dict(row)
        value["metric"] = primary_name if value["metric"] == "accuracy" else loss_name
        if "multiplicity_family" in value:
            value["multiplicity_family"] = (
                f"logical_ber={value['logical_ber']:.8g}, metric={value['metric']}"
            )
        return value

    result["pooled"] = [aggregate(row) for row in result["pooled"]]
    result["replications"] = [aggregate(row) for row in result["replications"]]
    result["paired"] = [comparison(row) for row in result["paired"]]
    result["paired_replications"] = [comparison(row) for row in result["paired_replications"]]
    result["paired_differences"] = [comparison(row) for row in result["paired_differences"]]
    result["metric_contract"] = {
        "primary": {"name": primary_name, "direction": primary_direction},
        "loss": {"name": loss_name, "direction": loss_direction},
    }
    return result
