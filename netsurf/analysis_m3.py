"""Ranking-quality, mask-composition, and paired-policy analysis."""

from __future__ import annotations

import itertools
import math
from typing import Mapping, Sequence

import numpy as np

from .empirical import SingleBitMeasurement, layer_name, parameter_class
from .identities import BitIdentity
from .statistics import summarize


def _average_ranks(values: Sequence[float]) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    order = np.argsort(data, kind="mergesort")
    ranks = np.empty(len(data), dtype=np.float64)
    start = 0
    while start < len(data):
        stop = start + 1
        while stop < len(data) and data[order[stop]] == data[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def spearman(values_a: Sequence[float], values_b: Sequence[float]) -> float:
    a, b = _average_ranks(values_a), _average_ranks(values_b)
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def kendall_tau_b(values_a: Sequence[float], values_b: Sequence[float]) -> float:
    a, b = np.asarray(values_a), np.asarray(values_b)
    concordant = discordant = ties_a = ties_b = 0
    for i in range(len(a)):
        da = np.sign(a[i + 1:] - a[i])
        db = np.sign(b[i + 1:] - b[i])
        concordant += int(np.sum((da * db) > 0))
        discordant += int(np.sum((da * db) < 0))
        ties_a += int(np.sum((da == 0) & (db != 0)))
        ties_b += int(np.sum((db == 0) & (da != 0)))
    denominator = math.sqrt((concordant + discordant + ties_a) * (concordant + discordant + ties_b))
    return float((concordant - discordant) / denominator) if denominator else float("nan")


def ranking_quality(
    measurements: Sequence[SingleBitMeasurement],
    estimator_outputs,
    *,
    top_ks: Sequence[int] = (5, 10, 20),
    tie_aware_baselines: bool = False,
) -> tuple[dict[str, object], ...]:
    measured = {value.identity: value for value in measurements}
    rows: list[dict[str, object]] = []
    scopes: list[tuple[str, tuple[BitIdentity, ...]]] = [("overall", tuple(measured))]
    for layer in sorted({value.layer for value in measurements}):
        scopes.append((f"layer:{layer}", tuple(value.identity for value in measurements if value.layer == layer)))
    for bit in sorted({value.bit_index for value in measurements}):
        scopes.append((f"bit:{bit}", tuple(value.identity for value in measurements if value.bit_index == bit)))
    for output in estimator_outputs:
        if not set(measured) <= set(output.scores):
            raise ValueError(f"estimator {output.name} does not cover every empirical candidate")
        for scope, identities in scopes:
            scores = [float(output.scores[value]) for value in identities]
            positive = [measured[value].positive_loss_damage for value in identities]
            absolute = [measured[value].absolute_loss_difference for value in identities]
            signed = [measured[value].signed_loss_difference for value in identities]
            unique, counts = np.unique(scores, return_counts=True)
            tied = int(np.sum(counts[counts > 1]))
            row: dict[str, object] = {
                "estimator": output.name,
                "implementation_version": output.implementation_version,
                "scope": scope,
                "candidate_count": len(identities),
                "spearman_positive_damage": spearman(scores, positive),
                "spearman_absolute_change": spearman(scores, absolute),
                "kendall_positive_damage": kendall_tau_b(scores, positive),
                "zero_score_fraction": float(np.mean(np.asarray(scores) == 0.0)),
                "tied_score_fraction": tied / len(scores),
                "runtime_seconds": output.runtime_seconds,
                "sample_limited": any(measured[value].sampling_probability < 1.0 for value in identities),
            }
            empirical_order = sorted(identities, key=lambda value: (-measured[value].positive_loss_damage, value))
            predicted_order = sorted(identities, key=lambda value: (-float(output.scores[value]), value))
            for requested in top_ks:
                k = min(requested, len(identities))
                overlap = len(set(empirical_order[:k]) & set(predicted_order[:k]))
                row[f"top_{requested}_recall"] = overlap / k
                row[f"top_{requested}_overlap"] = overlap
                if tie_aware_baselines:
                    threshold = float(output.scores[predicted_order[k - 1]])
                    strict = {
                        value for value in identities
                        if float(output.scores[value]) > threshold
                    }
                    tied_boundary = {
                        value for value in identities
                        if float(output.scores[value]) == threshold
                    }
                    empirical_top = set(empirical_order[:k])
                    draws = k - len(strict)
                    expected_overlap = len(strict & empirical_top)
                    if tied_boundary and draws:
                        expected_overlap += (
                            draws * len(tied_boundary & empirical_top) / len(tied_boundary)
                        )
                    row[f"top_{requested}_tie_aware_expected_recall"] = (
                        expected_overlap / k
                    )
                    row[f"top_{requested}_random_expected_recall"] = k / len(identities)
            if output.signed_predictions is not None:
                predicted = np.asarray([output.signed_predictions[value] for value in identities], dtype=np.float64)
                actual = np.asarray(signed, dtype=np.float64)
                row["signed_mae"] = float(np.mean(np.abs(predicted - actual)))
                row["signed_rmse"] = float(np.sqrt(np.mean((predicted - actual) ** 2)))
                row["signed_bias"] = float(np.mean(predicted - actual))
            else:
                row.update({"signed_mae": None, "signed_rmse": None, "signed_bias": None})
            rows.append(row)
    return tuple(rows)


def estimator_agreement(estimator_outputs) -> tuple[dict[str, object], ...]:
    """Pairwise raw-score and rank-order diagnostics on the full population."""

    rows = []
    for left, right in itertools.combinations(estimator_outputs, 2):
        identities = tuple(sorted(left.scores))
        if set(identities) != set(right.scores):
            raise ValueError("estimator populations differ")
        left_scores = [float(left.scores[value]) for value in identities]
        right_scores = [float(right.scores[value]) for value in identities]
        left_ranks, right_ranks = _average_ranks(left_scores), _average_ranks(right_scores)
        left_order = sorted(identities, key=lambda value: (-float(left.scores[value]), value))
        right_order = sorted(identities, key=lambda value: (-float(right.scores[value]), value))
        k = min(10, len(identities))
        rows.append({
            "left": left.name,
            "right": right.name,
            "candidate_count": len(identities),
            "spearman": spearman(left_scores, right_scores),
            "identical_scores": bool(np.array_equal(left_scores, right_scores)),
            "identical_order": left_order == right_order,
            "changed_average_rank_count": int(np.sum(left_ranks != right_ranks)),
            "maximum_average_rank_shift": float(np.max(np.abs(left_ranks - right_ranks))),
            "top_10_overlap": len(set(left_order[:k]) & set(right_order[:k])),
            "top_10_recall": len(set(left_order[:k]) & set(right_order[:k])) / k,
        })
    return tuple(rows)


def mask_composition(
    masks: Mapping[str, Sequence[BitIdentity]],
    universe: Sequence[BitIdentity],
) -> tuple[dict[str, object], ...]:
    dimensions = {
        "layer": lambda value: layer_name(value.parameter),
        "parameter_class": lambda value: parameter_class(value.parameter),
        "bit_position": lambda value: str(value.bit_index),
    }
    rows = []
    for policy, selected_values in masks.items():
        selected = tuple(selected_values)
        for dimension, getter in dimensions.items():
            universe_counts: dict[str, int] = {}
            selected_counts: dict[str, int] = {}
            for identity in universe:
                key = getter(identity)
                universe_counts[key] = universe_counts.get(key, 0) + 1
            for identity in selected:
                key = getter(identity)
                selected_counts[key] = selected_counts.get(key, 0) + 1
            for key in sorted(universe_counts):
                count = selected_counts.get(key, 0)
                rows.append({
                    "policy": policy,
                    "dimension": dimension,
                    "category": key,
                    "protected_count": count,
                    "protected_fraction_of_mask": count / len(selected) if selected else 0.0,
                    "coverage_within_category": count / universe_counts[key],
                    "eligible_count": universe_counts[key],
                })
    return tuple(rows)


def paired_permutation_test(differences: Sequence[float], *, seed: int, permutations: int = 10000) -> float:
    data = np.asarray(differences, dtype=np.float64)
    observed = abs(float(np.mean(data)))
    if len(data) <= 20 and (1 << len(data)) <= permutations:
        means = []
        for signs in itertools.product((-1.0, 1.0), repeat=len(data)):
            means.append(abs(float(np.mean(data * np.asarray(signs)))))
        return float(np.mean(np.asarray(means) >= observed - 1e-15))
    generator = np.random.default_rng(int(seed))
    exceed = 0
    for _ in range(permutations):
        signs = generator.choice(np.asarray([-1.0, 1.0]), size=len(data))
        exceed += abs(float(np.mean(data * signs))) >= observed - 1e-15
    return (exceed + 1) / (permutations + 1)


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted, running = {}, 0.0
    total = len(ordered)
    for index, name in enumerate(ordered):
        value = min(1.0, (total - index) * p_values[name])
        running = max(running, value)
        adjusted[name] = running
    return adjusted


def paired_policy_comparisons(policy_values: Mapping[str, Sequence[float]], *, confidence: float, seed: int):
    pairs = {}
    raw_p = {}
    for left, right in itertools.combinations(policy_values, 2):
        a, b = np.asarray(policy_values[left]), np.asarray(policy_values[right])
        if len(a) != len(b):
            raise ValueError("paired policy vectors differ in length")
        differences = a - b
        name = f"{left}__minus__{right}"
        pairs[name] = {
            "left": left,
            "right": right,
            "difference": summarize(differences, confidence=confidence, bounds=(-1.0, 1.0)).to_dict(),
            "permutation_test": "paired two-sided sign-flip mean test",
        }
        raw_p[name] = paired_permutation_test(differences, seed=seed)
    adjusted = holm_adjust(raw_p)
    for name in pairs:
        pairs[name]["p_value_raw"] = raw_p[name]
        pairs[name]["p_value_holm"] = adjusted[name]
    return pairs
