"""Prespecified whole-field mean inference and non-excluding tail diagnostics."""
from __future__ import annotations

import math
import numpy as np

from .conditional_config import PAIRED_SWAP_COMPARISONS
from .statistics import summarize


def _field_statistics(rows):
    from scipy.stats import t  # Required here: never substitute a normal approximation.
    x = np.asarray([r["target_minus_baseline"] for r in rows], dtype=np.float64)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("replication inference requires >=2 finite whole-field contrasts")
    estimate = summarize(x, confidence=.95).to_dict()
    se, mean = estimate["standard_error"], estimate["mean"]
    p = float(2 * t.sf(abs(mean / se), df=len(x) - 1)) if se > 0 else (1.0 if mean == 0 else None)
    order = sorted(range(len(x)), key=lambda i: (-abs(x[i]), rows[i]["field"]))
    worst = sorted(range(len(x)), key=lambda i: (-x[i], rows[i]["field"]))
    extrema = []
    for fraction in (.01, .05, .10):
        n = max(1, math.ceil(fraction * len(x)))
        indices = order[:n]
        contribution = float(np.sum(x[indices]) / len(x))
        extrema.append({"top_absolute_fraction": fraction, "count": n,
            "fields": [rows[i]["field"] for i in indices],
            "contribution_to_full_mean": contribution,
            "complement_contribution_to_full_mean": mean - contribution,
            "fraction_of_absolute_mass": float(np.sum(np.abs(x[indices])) / np.sum(np.abs(x))) if np.any(x) else None})
    loo = (np.sum(x) - x) / (len(x) - 1)
    return {
        "mean_effect": estimate, "p_value_two_sided_unadjusted": p,
        "p_value_method": "paired Student-t on whole-field contrasts, df=n-1",
        "degenerate_variance": bool(se == 0),
        "favorable_fields": int(np.count_nonzero(x < 0)),
        "unfavorable_fields": int(np.count_nonzero(x > 0)), "zero_fields": int(np.count_nonzero(x == 0)),
        "median": float(np.median(x)),
        "quantiles": {str(q): float(np.quantile(x, q)) for q in (0, .01, .05, .25, .5, .75, .95, .99, 1)},
        "mean_minus_median": mean - float(np.median(x)),
        "worst_adverse_fields": [{"field": rows[i]["field"], "field_seed": rows[i]["field_seed"],
                                  "contrast": float(x[i])} for i in worst[:5] if x[i] > 0],
        "extreme_contributions": extrema,
        "leave_one_out_influence_only": {"min_mean": float(np.min(loo)), "max_mean": float(np.max(loo)),
            "max_absolute_mean_shift": float(np.max(np.abs(loo - mean))),
            "interpretation": "descriptive sensitivity only; all fields retained in every inferential result"},
        "all_fields_retained": True,
    }


def summarize_replication(records, expected_count):
    groups = {name: [] for name in PAIRED_SWAP_COMPARISONS}
    common = None
    for row in records:
        if row["comparison"] not in groups:
            raise ValueError("unprespecified replication comparison")
        groups[row["comparison"]].append(row)
    results = []
    for name, rows in groups.items():
        rows.sort(key=lambda r: r["field"])
        if len(rows) != expected_count or [r["field"] for r in rows] != list(range(expected_count)):
            raise ValueError("missing or duplicate whole-field observations")
        if len({r["field_seed"] for r in rows}) != expected_count:
            raise ValueError("duplicate master-field seed")
        keys = [(r["logical_ber"], r["field"], r["field_seed"], r["master_pairing_key"],
                 r["master_sha256"], r["sample_sha256"], r["protection_context_sha256"]) for r in rows]
        if common is not None and keys != common:
            raise ValueError("comparisons do not share identical master fields, samples and context")
        common = keys
        results.append({"comparison": name, "role": "primary" if name == PAIRED_SWAP_COMPARISONS[0] else "secondary",
                        **_field_statistics(rows)})
    secondaries = results[1:]
    # Undefined degenerate tests conservatively participate as p=1, but remain null in output.
    order = sorted(range(2), key=lambda i: secondaries[i]["p_value_two_sided_unadjusted"] if
                   secondaries[i]["p_value_two_sided_unadjusted"] is not None else 1.)
    previous = 0.
    for rank, i in enumerate(order):
        raw = secondaries[i]["p_value_two_sided_unadjusted"]
        adjusted = max(previous, min(1., (2 - rank) * (raw if raw is not None else 1.)))
        previous = adjusted
        secondaries[i]["p_value_secondary_holm"] = adjusted if raw is not None else None
        secondaries[i]["secondary_family_mean_ci"] = summarize(
            [r["target_minus_baseline"] for r in groups[secondaries[i]["comparison"]]], confidence=.975).to_dict()
    return {"schema_version": "1.0.0", "independent_field_count": expected_count,
            "comparisons": results, "secondary_family_size": 2,
            "interpretation": "mean expected risk is primary; median and favorable-field fraction are different estimands; approximate t inference may be sensitive to rare heavy tails; conditional on fixed inputs"}


def replication_report(outcome):
    payload = outcome.scientific_payload
    plan = payload["replication_plan"]
    analysis = payload["replication_analysis"]
    lines = [f"# {outcome.configuration.name}", "", "Independent frozen-mask replication; selection was not rerun.", "",
        "Exact TF 2.14.1 BUILTIN_REF INT8 inference; persistent independent Bernoulli logical BER 0.01; ideal event suppression, not hardware mitigation.", "",
        "Masks: selected (2,6), exact isolated (2,7), original S5 (6,7). Global budget 22,016/176,128.", "",
        f"Purpose: {outcome.configuration.workflow['purpose']}. Master fields: {analysis['independent_field_count']}; fixed input clips: {outcome.configuration.samples.evaluation_count}.", "",
        "Negative target-minus-baseline cross-entropy favors the target. Every path is summed within a master field before inference. Inputs and path steps are not independent fault replications.", "",
        "| Comparison | Role | Mean CE difference | 95% mean CI | Raw p | Secondary Holm p | Favorable / adverse / zero | Median |",
        "|---|---|---:|---|---:|---:|---|---:|"]
    for r in analysis["comparisons"]:
        e = r["mean_effect"]
        lines.append(f"| {r['comparison']} | {r['role']} | {e['mean']:.9g} | {e['confidence_interval']} | {r['p_value_two_sided_unadjusted']} | {r.get('p_value_secondary_holm', 'not applicable')} | {r['favorable_fields']} / {r['unfavorable_fields']} / {r['zero_fields']} | {r['median']:.9g} |")
    lines += ["", "Primary: two-sided 95% Student-t CI and unadjusted paired t-test. Secondary p-values use Holm across the two prespecified secondary comparisons; simultaneous Bonferroni secondary intervals are saved in replication_analysis.json. These are approximate mean-risk inferences, not guarantees under heavy tails.", "",
        "## Tail sensitivity (descriptive; no fields excluded)", ""]
    for r in analysis["comparisons"]:
        lines += [f"### {r['comparison']}", "", f"Empirical quantiles: {r['quantiles']}.", "",
                  f"Worst adverse fields: {r['worst_adverse_fields']}.", "",
                  f"Leave-one-field-out influence: {r['leave_one_out_influence_only']}.", "",
                  "| Largest absolute fraction | Fields | Contribution to full mean | Complement contribution |",
                  "|---|---:|---:|---:|"]
        for x in r["extreme_contributions"]:
            lines.append(f"| {x['top_absolute_fraction']:.0%} | {x['count']} | {x['contribution_to_full_mean']:.9g} | {x['complement_contribution_to_full_mean']:.9g} |")
        lines.append("")
    lines += ["## Provenance and limitations", "",
        f"Selection payload: `{plan['selection_payload_sha256']}`. Replication plan canonical digest: `{payload['replication_plan_sha256']}`.", "",
        f"Scientific payload: `{outcome.scientific_payload_sha256}`.", "",
        "Intervals condition on the fixed clips. Per-input contributions are saved only when explicitly requested, for later crossed input/field analysis; no input-resampling intervals are asserted here. Mean, median and probability of favorable background are not interchangeable. This is one exploratory eight-bit pool; it does not establish population-wide or cross-architecture superiority.", "",
        "Exposure clearance covers the hashed inventory and local finalized records, not unknown manual/failed or concurrent jobs. No final outcome may change the plan, masks or candidate pool.", "",
        "Reproduce with the bundle's config and frozen plan using `/projects/kona-assets/manuelbv/.venv-mlperf/bin/python -m netsurf conditional run /ABSOLUTE/PATH/config.toml`. A completed confirmation window cannot be reused as unexposed confirmation; an engineering replay requires a separately frozen plan.", ""]
    return "\n".join(lines)
