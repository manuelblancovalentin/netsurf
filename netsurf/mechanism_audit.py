"""Memory-bounded cross-benchmark vulnerability-mechanism diagnostics.

This module derives exploratory evidence from finalized bundles.  It does not
rerun ranking, fault injection, or authoritative inference, and its outputs are
separate derived analyses rather than replacements for finalized bundles.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import heapq
import json
import math
from pathlib import Path
import tempfile
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np

from .analysis_m3 import spearman
from .manifests import canonical_json_bytes, sha256_file
from .statistics import summarize


AUDIT_SCHEMA_VERSION = "1.0.0"

_S2 = "relative-word-bit-index/v1"
_S3 = "absolute-affine-xor-jump/v1"
_S4 = "signed-gradient-times-exact-delta/v2"
_S5 = "mean-absolute-per-example-gradient-times-exact-delta/v1"
_S6 = "mean-positive-per-example-gradient-times-exact-delta/v1"
_METHOD_BY_VERSION = {_S2: "S2", _S3: "S3", _S4: "S4", _S5: "S5", _S6: "S6"}


@dataclass(frozen=True)
class MechanismBundleSpec:
    name: str
    estimator_bundle: Path
    protection_bundle: Path


def _finalized(path: Path) -> Path:
    value = Path(path).expanduser().resolve()
    if not (value / "FINALIZED").is_file():
        raise ValueError(f"not a finalized NetSuRF bundle: {value}")
    return value


def _csv_rows(path: Path) -> Iterator[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        yield from csv.DictReader(stream)


def _identity(row: Mapping[str, str]) -> tuple[str, int, int]:
    return row["parameter"], int(row["flat_index"]), int(row["bit_index"])


def _optional_float(value: str | None) -> float | None:
    return None if value in {None, ""} else float(value)


def _read_empirical(bundle: Path) -> dict[tuple[str, int, int], dict[str, object]]:
    result = {}
    for row in _csv_rows(bundle / "measurements/empirical_bits.csv"):
        bit = int(row["bit_index"])
        delta = float(row["numerical_perturbation"])
        result[_identity(row)] = {
            "parameter": row["parameter"],
            "flat_index": int(row["flat_index"]),
            "bit_index": bit,
            "original_encoded": int(row["original_encoded"]),
            "flipped_encoded": int(row["flipped_encoded"]),
            "numerical_perturbation": delta,
            "inferred_affine_scale": abs(delta) / float(1 << bit),
            "signed_loss_difference": float(row["signed_loss_difference"]),
            "positive_loss_damage": float(row["positive_loss_damage"]),
            "absolute_loss_difference": float(row["absolute_loss_difference"]),
            "primary_metric_damage": _optional_float(
                row.get("primary_metric_damage", row.get("accuracy_damage"))
            ),
            "sampling_probability": float(row["sampling_probability"]),
            "score_quantile": None if not row.get("score_quantile") else int(row["score_quantile"]),
            "inclusion_rule": row["inclusion_rule"],
        }
    if not result:
        raise ValueError(f"bundle has no empirical bit measurements: {bundle}")
    return result


def _stream_candidate_scores(
    bundle: Path,
    candidates: Mapping[tuple[str, int, int], Mapping[str, object]],
) -> tuple[dict[tuple[str, int, int], dict[str, float]], dict[str, str]]:
    scores = {identity: {} for identity in candidates}
    names: dict[str, str] = {}
    path = bundle / "measurements/estimator_predictions.csv"
    for row in _csv_rows(path):
        if row.get("empirical_candidate", "").lower() != "true":
            continue
        identity = _identity(row)
        if identity not in candidates:
            continue
        method = _METHOD_BY_VERSION.get(row["implementation_version"])
        if method is None:
            continue
        names[method] = row["estimator"]
        scores[identity][method] = float(row["score"])
        if row.get("signed_prediction"):
            scores[identity][f"{method}_signed"] = float(row["signed_prediction"])
    return scores, names


def _strata(row: Mapping[str, object]) -> tuple[object, ...]:
    return row["parameter"], row["bit_index"], row["score_quantile"]


def _stratified_bootstrap(
    rows: Sequence[Mapping[str, object]],
    statistic,
    *,
    seed: int,
    replicates: int,
) -> tuple[float, float]:
    groups: dict[tuple[object, ...], list[Mapping[str, object]]] = {}
    for row in rows:
        groups.setdefault(_strata(row), []).append(row)
    generator = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        sample = []
        for key in sorted(groups, key=str):
            group = groups[key]
            sample.extend(group[int(index)] for index in generator.integers(0, len(group), len(group)))
        value = statistic(sample)
        if value is not None and math.isfinite(value):
            values.append(value)
    if not values:
        return float("nan"), float("nan")
    return tuple(float(value) for value in np.quantile(values, (0.025, 0.975)))


def _candidate_diagnostics(
    name: str,
    empirical: Mapping[tuple[str, int, int], dict[str, object]],
    scores: Mapping[tuple[str, int, int], Mapping[str, float]],
    *,
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows = []
    for identity in sorted(empirical):
        row = {"benchmark": name, **empirical[identity]}
        values = scores.get(identity, {})
        row.update({f"{key}_score": values.get(key) for key in ("S2", "S3", "S4", "S5", "S6")})
        mean_z = values.get("S4_signed")
        absolute = values.get("S5")
        positive = values.get("S6")
        if mean_z is not None and absolute is not None and positive is not None:
            row["mean_first_order_z"] = mean_z
            row["directional_coherence"] = None if absolute == 0.0 else mean_z / absolute
            row["directional_coherence_implied_by_s5_s6"] = (
                None if absolute == 0.0 else (2.0 * positive - absolute) / absolute
            )
            row["s5_s6_relation_residual"] = positive - 0.5 * (absolute + mean_z)
            row["s4_absolute_residual"] = values["S4"] - abs(mean_z)
        else:
            row["mean_first_order_z"] = None
            row["directional_coherence"] = None
            row["directional_coherence_implied_by_s5_s6"] = None
            row["s5_s6_relation_residual"] = None
            row["s4_absolute_residual"] = None
        rows.append(row)

    coherent = [row for row in rows if row["directional_coherence"] is not None]
    maximum_relation_error = max(
        (abs(float(row["s5_s6_relation_residual"])) for row in coherent),
        default=float("nan"),
    )
    maximum_s4_error = max(
        (abs(float(row["s4_absolute_residual"])) for row in coherent),
        default=float("nan"),
    )
    maximum_relative_relation_error = max(
        (
            abs(float(row["s5_s6_relation_residual"]))
            / max(
                abs(float(row["S6_score"])),
                abs(float(row["S5_score"])),
                abs(float(row["mean_first_order_z"])),
                np.finfo(np.float64).tiny,
            )
            for row in coherent
        ),
        default=float("nan"),
    )
    maximum_coherence_disagreement = max(
        (
            abs(
                float(row["directional_coherence"])
                - float(row["directional_coherence_implied_by_s5_s6"])
            )
            for row in coherent
        ),
        default=float("nan"),
    )
    correlations = {}
    coherence_values = [float(row["directional_coherence"]) for row in coherent]
    for metric in ("signed_loss_difference", "positive_loss_damage", "absolute_loss_difference"):
        correlations[metric] = (
            spearman(coherence_values, [float(row[metric]) for row in coherent])
            if len(coherent) > 1 else float("nan")
        )
    task_aligned = [
        row for row in coherent if row.get("primary_metric_damage") is not None
    ]
    task_metric_alignment = (
        spearman(
            [float(row["signed_loss_difference"]) for row in task_aligned],
            [float(row["primary_metric_damage"]) for row in task_aligned],
        ) if len(task_aligned) > 1 else float("nan")
    )

    harmful = [float(row["directional_coherence"]) for row in coherent if float(row["signed_loss_difference"]) > 0.0]
    beneficial = [float(row["directional_coherence"]) for row in coherent if float(row["signed_loss_difference"]) < 0.0]
    difference = (
        float(np.mean(harmful) - np.mean(beneficial))
        if harmful and beneficial else None
    )

    def difference_statistic(sample):
        left = [float(row["directional_coherence"]) for row in sample if float(row["signed_loss_difference"]) > 0.0]
        right = [float(row["directional_coherence"]) for row in sample if float(row["signed_loss_difference"]) < 0.0]
        return None if not left or not right else float(np.mean(left) - np.mean(right))

    difference_ci = (
        _stratified_bootstrap(
            coherent, difference_statistic, seed=bootstrap_seed,
            replicates=bootstrap_replicates,
        ) if difference is not None else (None, None)
    )
    return rows, {
        "candidate_count": len(rows),
        "coherence_defined_count": len(coherent),
        "zero_s5_denominator_count": len(rows) - len(coherent),
        "maximum_s5_s6_identity_residual": maximum_relation_error,
        "maximum_s5_s6_relative_residual": maximum_relative_relation_error,
        "maximum_coherence_disagreement": maximum_coherence_disagreement,
        "maximum_s4_absolute_identity_residual": maximum_s4_error,
        "coherence_spearman": correlations,
        "exact_loss_vs_primary_metric_damage_spearman": task_metric_alignment,
        "harmful_count": len(harmful),
        "beneficial_count": len(beneficial),
        "harmful_minus_beneficial_mean_coherence": difference,
        "harmful_minus_beneficial_stratified_bootstrap_ci": difference_ci,
        "bootstrap_replicates": bootstrap_replicates,
        "interpretation": (
            "S4 signed_prediction is the gradient-of-mean realization of E[z]; "
            "S5 and S6 are means of separately evaluated per-example gradients. "
            "Their mathematical identity and observed finite-precision residual "
            "are both reported. Results are "
            "sample-limited and stratified, not whole-population estimates."
        ),
    }


def _heterogeneity(name: str, rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    result = []
    dimensions = {
        "bit_index": lambda row: str(row["bit_index"]),
        "tensor": lambda row: str(row["parameter"]),
    }
    for dimension, key_function in dimensions.items():
        groups: dict[str, list[Mapping[str, object]]] = {}
        for row in rows:
            groups.setdefault(key_function(row), []).append(row)
        for category in sorted(groups):
            selected = groups[category]
            weights = np.asarray(
                [1.0 / float(row["sampling_probability"]) for row in selected],
                dtype=np.float64,
            )
            positive = np.asarray([float(row["positive_loss_damage"]) for row in selected])
            absolute = np.asarray([float(row["absolute_loss_difference"]) for row in selected])
            strata: dict[tuple[object, ...], list[float]] = {}
            for row in selected:
                strata.setdefault(_strata(row), []).append(float(row["positive_loss_damage"]))
            result.append({
                "benchmark": name,
                "dimension": dimension,
                "category": category,
                "candidate_count": len(selected),
                "positive_damage_mean_unweighted": float(np.mean(positive)),
                "positive_damage_mean_inverse_probability_weighted": float(np.average(positive, weights=weights)),
                "positive_damage_equal_recorded_stratum_mean": float(np.mean([np.mean(values) for values in strata.values()])),
                "positive_damage_standard_deviation": float(np.std(positive, ddof=1)) if len(positive) > 1 else 0.0,
                "absolute_damage_mean_unweighted": float(np.mean(absolute)),
                "affine_scale_mean": float(np.mean([float(row["inferred_affine_scale"]) for row in selected])),
                "encoded_value_mean": float(np.mean([int(row["original_encoded"]) for row in selected])),
                "sample_limited": True,
            })
    return result


def _read_small_table(bundle: Path, name: str) -> list[dict[str, str]]:
    path = bundle / "measurements" / name
    return list(_csv_rows(path)) if path.is_file() else []


def _method_from_policy(policy: str) -> str | None:
    value = policy.lower()
    if value.startswith("msb_global") or value.startswith("s2"):
        return "S2"
    if value.startswith("gradient_global") or value.startswith("s4"):
        return "S4"
    if value.startswith("per_example_absolute") or value.startswith("s5"):
        return "S5"
    if value.startswith("per_example_positive") or value.startswith("s6"):
        return "S6"
    return None


class _CharStream:
    def __init__(self, path: Path, chunk_size: int = 1 << 16):
        self.stream = path.open("r", encoding="utf-8")
        self.chunk_size = chunk_size
        self.buffer = ""
        self.index = 0
        self.pushed: list[str] = []

    def get(self) -> str:
        if self.pushed:
            return self.pushed.pop()
        if self.index >= len(self.buffer):
            self.buffer = self.stream.read(self.chunk_size)
            self.index = 0
            if not self.buffer:
                return ""
        value = self.buffer[self.index]
        self.index += 1
        return value

    def push(self, value: str) -> None:
        if value:
            self.pushed.append(value)

    def close(self) -> None:
        self.stream.close()


def _json_tokens(path: Path):
    source = _CharStream(path)
    punctuation = set("{}[]:,")
    try:
        while True:
            value = source.get()
            while value and value.isspace():
                value = source.get()
            if not value:
                return
            if value in punctuation:
                yield value
                continue
            if value == '"':
                raw = ['"']
                escaped = False
                while True:
                    char = source.get()
                    if not char:
                        raise ValueError(f"unterminated JSON string in {path}")
                    raw.append(char)
                    if char == '"' and not escaped:
                        break
                    escaped = char == "\\" and not escaped
                    if char != "\\":
                        escaped = False
                yield ("string", json.loads("".join(raw)))
                continue
            raw = [value]
            while True:
                char = source.get()
                if not char or char.isspace() or char in punctuation:
                    source.push(char)
                    break
                raw.append(char)
            text = "".join(raw)
            yield ("atom", json.loads(text))
    finally:
        source.close()


def _iter_mask_entries(path: Path, *, tokens=None) -> Iterator[tuple[str, str, str, int, int]]:
    tokens = iter(_json_tokens(path) if tokens is None else tokens)

    def take(expected=None):
        value = next(tokens)
        if expected is not None and value != expected:
            raise ValueError(f"invalid mask JSON: expected {expected!r}, got {value!r}")
        return value

    def string():
        value = take()
        if not isinstance(value, tuple) or value[0] != "string":
            raise ValueError(f"invalid mask JSON string token: {value!r}")
        return value[1]

    def mask(replication: str, policy: str):
        first = take()
        if first == "]":
            return
        while True:
            if first != "[":
                raise ValueError("invalid bit identity in mask artifact")
            parameter = string(); take(",")
            flat = int(take()[1]); take(",")
            bit = int(take()[1]); take("]")
            yield replication, policy, parameter, flat, bit
            separator = take()
            if separator == "]":
                return
            if separator != ",":
                raise ValueError("invalid mask-array separator")
            first = take()

    def object_body(replication: str | None = None):
        first = take()
        if first == "}":
            return
        while True:
            if not isinstance(first, tuple) or first[0] != "string":
                raise ValueError("invalid mask-object key")
            key = first[1]
            take(":")
            value = take()
            if value == "{":
                yield from object_body(key)
            elif value == "[":
                yield from mask(replication or "replication_0", key)
            else:
                raise ValueError("unexpected mask-object value")
            separator = take()
            if separator == "}":
                return
            if separator != ",":
                raise ValueError("invalid mask-object separator")
            first = take()

    take("{")
    yield from object_body()


def _read_identity_line(stream):
    value = stream.readline()
    if not value:
        return None
    parameter, flat, bit = value.rstrip("\n").split("\t")
    return parameter, int(flat), int(bit)


def _merge_overlap(left: Path, right: Path) -> tuple[int, int, int]:
    left_count = right_count = intersection = 0
    with left.open("r", encoding="utf-8") as a, right.open("r", encoding="utf-8") as b:
        av, bv = _read_identity_line(a), _read_identity_line(b)
        while av is not None or bv is not None:
            if bv is None or (av is not None and av < bv):
                left_count += 1; av = _read_identity_line(a)
            elif av is None or bv < av:
                right_count += 1; bv = _read_identity_line(b)
            else:
                left_count += 1; right_count += 1; intersection += 1
                av, bv = _read_identity_line(a), _read_identity_line(b)
    return left_count, right_count, intersection


def _externally_sort_identity_file(path: Path, *, chunk_lines: int = 100_000) -> None:
    """Sort one ranking-ordered mask file with bounded memory."""

    chunks = []
    with path.open("r", encoding="utf-8") as source:
        while True:
            values = []
            for _ in range(chunk_lines):
                value = _read_identity_line(source)
                if value is None:
                    break
                values.append(value)
            if not values:
                break
            values.sort()
            chunk = path.with_name(f"{path.name}.chunk-{len(chunks):04d}")
            with chunk.open("w", encoding="utf-8", newline="") as stream:
                for parameter, flat, bit in values:
                    stream.write(f"{parameter}\t{flat}\t{bit}\n")
            chunks.append(chunk)

    temporary = path.with_name(f"{path.name}.sorted")
    streams = [chunk.open("r", encoding="utf-8") for chunk in chunks]
    try:
        iterators = (
            iter(lambda stream=stream: _read_identity_line(stream), None)
            for stream in streams
        )
        with temporary.open("w", encoding="utf-8", newline="") as output:
            for parameter, flat, bit in heapq.merge(*iterators):
                output.write(f"{parameter}\t{flat}\t{bit}\n")
        temporary.replace(path)
    finally:
        for stream in streams:
            stream.close()
        for chunk in chunks:
            chunk.unlink(missing_ok=True)


def _mask_overlaps(bundle: Path, protection_rows: Sequence[Mapping[str, str]], scratch: Path):
    selected = {}
    for row in protection_rows:
        method = _method_from_policy(row["policy"])
        if method is not None and float(row["coverage"]) not in {0.0, 1.0}:
            selected[row["condition"]] = (method, float(row["coverage"]))
    if not selected:
        return []
    artifact = bundle / "artifacts/protection_masks.json"
    if not artifact.is_file():
        return []
    scratch.mkdir(parents=True, exist_ok=False)
    handles = {}
    paths = {}
    try:
        for replication, policy, parameter, flat, bit in _iter_mask_entries(artifact):
            if policy not in selected:
                continue
            method, coverage = selected[policy]
            key = replication, method, coverage
            if key not in handles:
                target = scratch / f"mask-{len(handles):04d}.tsv"
                handles[key] = target.open("w", encoding="utf-8", newline="")
                paths[key] = target
            handles[key].write(f"{parameter}\t{flat}\t{bit}\n")
    finally:
        for handle in handles.values():
            handle.close()
    # Masks are stored in ranking order, whereas the exact overlap merge-join
    # requires canonical identity order.
    for path in paths.values():
        _externally_sort_identity_file(path)
    rows = []
    groups: dict[tuple[str, float], list[str]] = {}
    for replication, method, coverage in paths:
        groups.setdefault((replication, coverage), []).append(method)
    for (replication, coverage), methods in sorted(groups.items()):
        methods = sorted(set(methods))
        for left_index, left in enumerate(methods):
            for right in methods[left_index + 1:]:
                left_count, right_count, intersection = _merge_overlap(
                    paths[(replication, left, coverage)],
                    paths[(replication, right, coverage)],
                )
                union = left_count + right_count - intersection
                rows.append({
                    "replication": replication,
                    "coverage": coverage,
                    "left": left,
                    "right": right,
                    "left_count": left_count,
                    "right_count": right_count,
                    "intersection": intersection,
                    "union": union,
                    "jaccard": intersection / union if union else 1.0,
                    "fraction_of_left": intersection / left_count if left_count else 1.0,
                })
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def _plot(output: Path, candidate_rows, paired_rows) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    figures = []
    benchmarks = sorted({row["benchmark"] for row in candidate_rows})
    figure, axes = plt.subplots(1, len(benchmarks), figsize=(5 * len(benchmarks), 4), squeeze=False)
    for axis, benchmark in zip(axes[0], benchmarks):
        rows = [row for row in candidate_rows if row["benchmark"] == benchmark and row["directional_coherence"] is not None]
        axis.scatter(
            [row["directional_coherence"] for row in rows],
            [row["signed_loss_difference"] for row in rows],
            c=[row["bit_index"] for row in rows], s=12, alpha=0.65,
        )
        axis.axhline(0.0, color="black", linewidth=0.7)
        axis.set(title=benchmark, xlabel="Directional coherence C", ylabel="Exact signed isolated loss change")
    figure.tight_layout()
    target = output / "coherence_vs_exact_damage.png"; figure.savefig(target, dpi=160); plt.close(figure)
    figures.append(target.name)

    selected = [row for row in paired_rows if row.get("right") in {"no_protection", "random_global"}]
    if selected:
        figure, axis = plt.subplots(figsize=(9, 5))
        labels = [f"{row['benchmark']}\n{row['left']}\nBER {float(row['logical_ber']):.0e}, {float(row['coverage']):.1%}" for row in selected]
        means = [float(row["mean_difference"]) for row in selected]
        low = [mean - float(row["ci_low"]) for mean, row in zip(means, selected)]
        high = [float(row["ci_high"]) - mean for mean, row in zip(means, selected)]
        axis.errorbar(range(len(selected)), means, yerr=[low, high], fmt="o")
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xticks(range(len(selected)), labels, rotation=75, ha="right")
        axis.set_ylabel("Saved paired metric difference (95% CI)")
        axis.set_title("Exploratory paired protection effects")
        figure.tight_layout()
        target = output / "paired_protection_effects.png"; figure.savefig(target, dpi=160); plt.close(figure)
        figures.append(target.name)
    return figures


def _s5_minus_s2_effects(paired_rows: Sequence[Mapping[str, str]]):
    result = []
    for row in paired_rows:
        if row.get("metric") not in {"accuracy", "roc_auc"}:
            continue
        if {row.get("left"), row.get("right")} != {
            "msb_global", "per_example_absolute_global"
        }:
            continue
        direction = 1.0 if row["left"] == "per_example_absolute_global" else -1.0
        low, high = float(row["ci_low"]), float(row["ci_high"])
        result.append({
            "benchmark": row["benchmark"],
            "logical_ber": float(row["logical_ber"]),
            "coverage": float(row["coverage"]),
            "metric": row["metric"],
            "matched_pairs": int(row["matched_pairs"]),
            "s5_minus_s2_mean": direction * float(row["mean_difference"]),
            "s5_minus_s2_ci_low": direction * (low if direction > 0 else high),
            "s5_minus_s2_ci_high": direction * (high if direction > 0 else low),
            "p_value_holm": _optional_float(row.get("p_value_holm")),
        })
    return sorted(result, key=lambda row: (
        row["benchmark"], row["logical_ber"], row["coverage"]
    ))


def _render_report(audit: Mapping[str, object], figures: Sequence[str]) -> str:
    lines = [
        "# NetSuRF cross-benchmark vulnerability mechanism audit",
        "",
        "This is a derived, exploratory analysis of immutable finalized bundles. It does not alter their scientific payloads and does not establish a causal or universal vulnerability mechanism.",
        "",
        "## Directional gradient coherence",
        "",
        "| Benchmark | Candidates | Defined C | Max S5/S6 identity residual | Spearman C vs signed exact loss | Harmful−beneficial mean C (95% stratified-bootstrap CI) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, value in audit["benchmarks"].items():
        coherence = value["coherence"]
        difference = coherence["harmful_minus_beneficial_mean_coherence"]
        interval = coherence["harmful_minus_beneficial_stratified_bootstrap_ci"]
        lines.append(
            f"| {name} | {coherence['candidate_count']} | {coherence['coherence_defined_count']} | "
            f"{coherence['maximum_s5_s6_identity_residual']:.3g} | "
            f"{coherence['coherence_spearman']['signed_loss_difference']:.3f} | "
            f"{difference if difference is not None else 'N/A'} {interval} |"
        )
    lines += [
        "",
        "The saved S4 signed prediction is the gradient-of-mean realization of `E[z]`; S5 and S6 come from separately reduced per-example gradients. The mathematical identity `S6=(S5+E[z])/2`, its finite-precision residual, and the S4-versus-S5/S6 coherence discrepancy are reported candidate by candidate. Zero S5 denominators are reported rather than divided. These results use the recorded stratified candidate samples and are not population-wide estimates.",
        "",
        "## Encoded-bit and tensor heterogeneity",
        "",
        "`heterogeneity.csv` reports unweighted, inverse-inclusion-probability-weighted, and equal-recorded-stratum summaries separately by bit index and tensor. Differences among these columns reveal sensitivity to the sampling estimand. `mask_overlap.csv` is computed by streaming exact saved mask identities through disk-backed temporary files, without loading the full mask JSON into memory.",
        "",
        "## Task-metric alignment",
        "",
        "ToyCar ranking differentiates per-window reconstruction MSE while final protection uses file-level ROC-AUC. The isolated-bit table permits association between reconstruction-MSE damage and ROC-AUC damage, but existing finalized bundles do not retain per-file clean/faulted scores. Therefore score-order reversals cannot be reconstructed retrospectively. The optional `write_toycar_file_score_artifact` API records bounded per-file clean/faulted scores, labels, raw float64 representations, event accounting, sample IDs, and checksums for future runs without enlarging ordinary bundles.",
        "",
        "## BER, budget, and interaction limits",
        "",
        "`paired_effects.csv` reproduces saved paired effect sizes and intervals using the original `(replication, BER, trial, schedule)` pairing. Variation with BER or budget is observational. It can be consistent with nonlinear interactions or redundancy, but does not isolate either mechanism because isolated faults and background-field marginal protection were not evaluated in one controlled factorial design.",
        "",
        "### Direct S5 versus corrected S2 effects",
        "",
        "Positive values favor S5 for the saved primary task metric. Intervals and Holm-adjusted p-values are copied from the finalized paired analyses; they are not inferred from marginal policy intervals.",
        "",
        "| Benchmark | BER | Coverage | Metric | Pairs | S5−S2 mean | 95% CI | Holm p |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in audit["s5_minus_s2_primary_effects"]:
        lines.append(
            f"| {row['benchmark']} | {row['logical_ber']:.0e} | {row['coverage']:.1%} | "
            f"{row['metric']} | {row['matched_pairs']} | {row['s5_minus_s2_mean']:.6f} | "
            f"[{row['s5_minus_s2_ci_low']:.6f}, {row['s5_minus_s2_ci_high']:.6f}] | "
            f"{row['p_value_holm'] if row['p_value_holm'] is not None else 'N/A'} |"
        )
    lines += [
        "",
        "## Redundancy and activation gating",
        "",
        "The finalized bundles expose stored weights, exact faults, estimator scores, masks, and application metrics, but not per-layer activation traces or intervention-matched gating changes. Cross-architecture differences cannot identify causal redundancy. Zero-valued stored weights remain physically allocated fault candidates. A defensible follow-up is a within-checkpoint, held-out intervention that records ReLU gate changes and layer outputs under the same isolated and matched-background fault identities; structured physical pruning must remain a distinct checkpoint identity.",
        "",
        "## Statistical scope",
        "",
        "All new relationships are exploratory. Candidate-sampling uncertainty, fault-field uncertainty, evaluation-file variability, and ranking-mask variability remain distinct. Existing single-mask studies condition on that mask. No result here establishes that one estimator is universally superior or that a correlation is causal.",
    ]
    if figures:
        lines += ["", "## Figures", ""] + [f"![{name}](plots/{name})" for name in figures]
    lines += ["", "## Source bundles", ""]
    for name, value in audit["benchmarks"].items():
        lines.append(f"- `{name}` estimator: `{value['estimator_bundle']}`")
        lines.append(f"- `{name}` protection: `{value['protection_bundle']}`")
    return "\n".join(lines) + "\n"


def run_mechanism_audit(
    specifications: Sequence[MechanismBundleSpec],
    output: Path,
    *,
    verify: bool = True,
    plots: bool = True,
    bootstrap_seed: int = 930_401,
    bootstrap_replicates: int = 1000,
) -> Path:
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    (destination / "plots").mkdir()
    all_candidates, all_heterogeneity, all_paired, all_composition, all_overlaps = [], [], [], [], []
    benchmarks = {}
    source_checksums = {}
    with tempfile.TemporaryDirectory(prefix="mask-audit-", dir=destination) as temporary:
        scratch = Path(temporary)
        for index, specification in enumerate(specifications):
            estimator = _finalized(specification.estimator_bundle)
            protection = _finalized(specification.protection_bundle)
            used = [
                estimator / "measurements/empirical_bits.csv",
                estimator / "measurements/estimator_predictions.csv",
                estimator / "measurements/ranking_metrics.csv",
                estimator / "measurements/estimator_runtime.csv",
                protection / "measurements/protection_results.csv",
                protection / "measurements/paired_comparisons.csv",
                protection / "measurements/mask_composition.csv",
                protection / "artifacts/protection_masks.json",
            ]
            expected_checksums = {}
            for bundle in (estimator, protection):
                document = json.loads((bundle / "checksums.json").read_text(encoding="utf-8"))
                expected_checksums[bundle] = document["files"]
            for path in used:
                if not path.is_file():
                    continue
                relative = path.relative_to(estimator if estimator in path.parents else protection).as_posix()
                root = estimator if estimator in path.parents else protection
                expected = expected_checksums[root].get(relative)
                if expected is None:
                    raise ValueError(f"used artifact is absent from bundle checksums: {path}")
                actual = sha256_file(path) if verify else expected
                if actual != expected:
                    raise ValueError(f"checksum mismatch: {path}")
                source_checksums[str(path)] = expected

            empirical = _read_empirical(estimator)
            scores, names = _stream_candidate_scores(estimator, empirical)
            candidate_rows, coherence = _candidate_diagnostics(
                specification.name, empirical, scores,
                bootstrap_seed=bootstrap_seed + index,
                bootstrap_replicates=bootstrap_replicates,
            )
            heterogeneity = _heterogeneity(specification.name, candidate_rows)
            protection_rows = _read_small_table(protection, "protection_results.csv")
            paired = _read_small_table(protection, "paired_comparisons.csv")
            composition = _read_small_table(protection, "mask_composition.csv")
            for row in paired:
                row["benchmark"] = specification.name
            for row in composition:
                row["benchmark"] = specification.name
            overlaps = _mask_overlaps(protection, protection_rows, scratch / f"masks-{index}")
            for row in overlaps:
                row["benchmark"] = specification.name
            ranking_metrics = _read_small_table(estimator, "ranking_metrics.csv")
            runtimes = _read_small_table(estimator, "estimator_runtime.csv")
            benchmarks[specification.name] = {
                "estimator_bundle": str(estimator),
                "protection_bundle": str(protection),
                "estimator_names": names,
                "coherence": coherence,
                "ranking_metrics_overall": [row for row in ranking_metrics if row.get("scope") == "overall"],
                "estimator_runtimes": runtimes,
                "protection_condition_count": len(protection_rows),
                "paired_comparison_count": len(paired),
                "mask_overlap_count": len(overlaps),
                "ranking_replication_limitation": "Interpret ranking_mask_count and replication_count from saved protection_results; repeated fault fields under one mask are not mask replications.",
            }
            all_candidates.extend(candidate_rows); all_heterogeneity.extend(heterogeneity)
            all_paired.extend(paired); all_composition.extend(composition); all_overlaps.extend(overlaps)

    audit = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "analysis_type": "exploratory_offline_mechanism_diagnostics",
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_replicates": bootstrap_replicates,
        "source_artifact_sha256": source_checksums,
        "benchmarks": benchmarks,
        "s5_minus_s2_primary_effects": _s5_minus_s2_effects(all_paired),
        "limitations": [
            "Stratified candidate samples are not uniform whole-network samples.",
            "S4/S5/S6 score identities establish saved aggregation consistency, not exact integer-program derivatives.",
            "Protection outcomes condition on frozen checkpoints, data partitions, masks, and logical fault models.",
            "Existing ToyCar bundles do not contain per-file faulted scores.",
            "No existing cross-architecture observation identifies a causal redundancy or activation-gating mechanism.",
        ],
    }
    _write_csv(destination / "candidate_diagnostics.csv", all_candidates)
    _write_csv(destination / "heterogeneity.csv", all_heterogeneity)
    _write_csv(destination / "paired_effects.csv", all_paired)
    _write_csv(destination / "mask_composition.csv", all_composition)
    _write_csv(destination / "mask_overlap.csv", all_overlaps)
    figures = _plot(destination / "plots", all_candidates, all_paired) if plots else []
    (destination / "audit.json").write_bytes(canonical_json_bytes(audit) + b"\n")
    (destination / "report.md").write_text(_render_report(audit, figures), encoding="utf-8")
    inventory = {
        path.relative_to(destination).as_posix(): sha256_file(path)
        for path in sorted(destination.rglob("*")) if path.is_file()
    }
    (destination / "checksums.json").write_bytes(canonical_json_bytes({
        "schema_version": "1.0.0", "files": inventory,
    }) + b"\n")
    return destination


def write_toycar_file_score_artifact(
    adapter,
    inputs,
    labels,
    sample_ids: Sequence[object],
    destination: Path,
    *,
    fault_cases: Mapping[str, tuple[object, object]] | None = None,
) -> Path:
    """Write an opt-in bounded clean/faulted ToyCar file-score artifact."""

    output = Path(destination).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    cases = {"clean": (None, None), **dict(fault_cases or {})}
    rows = []
    accounting = {}
    for name, (schedule, protection) in cases.items():
        records, report = adapter.file_score_records(
            inputs, labels, schedule=schedule, protection=protection
        )
        if len(records) != len(sample_ids):
            raise ValueError("sample_ids must identify every file in the bounded evaluation")
        for sample_id, record in zip(sample_ids, records):
            rows.append({"condition": name, "sample_id": sample_id, **record})
        if report is not None:
            accounting[name] = {
                "schedule_sha256": report.schedule_hash,
                "proposed_events": report.proposed_events,
                "applied_events": report.applied_events,
                "suppressed_events": report.suppressed_events,
                "unique_applied_bits": report.unique_applied_bits,
            }
    _write_csv(output / "file_scores.csv", rows)
    manifest = {
        "schema_version": "1.0.0",
        "sample_ids": list(sample_ids),
        "conditions": list(cases),
        "event_accounting": accounting,
        "interpretation": "bounded diagnostic only; ROC-AUC labels must not be used to fit a ranking on final-evaluation files",
    }
    (output / "manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
    files = {path.name: sha256_file(path) for path in output.iterdir() if path.is_file()}
    (output / "checksums.json").write_bytes(canonical_json_bytes({"schema_version": "1.0.0", "files": files}) + b"\n")
    return output
