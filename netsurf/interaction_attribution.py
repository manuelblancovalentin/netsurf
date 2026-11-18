"""M910 offline decision attribution. No inference, optimizer or fault sampling.

All contrasts are target minus baseline. The clean additive model is evaluated
on the SAME selection cohort as the response table. Outside-field effects and
within-candidate interactions are algebraically distinct, not causal claims
about architectural redundancy.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from .restricted_subcube import fixed_budget_masks, integrate_response_table, subcube_states

VERSION = "1.0.0"
TIE_ATOL = 1e-12  # diagnostic near-ties only; never changes any frozen mask


def mean_interval(values, confidence=.95):
    """Descriptive paired whole-unit t interval; no invented n=1 precision."""
    from scipy.stats import t
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all() or not 0 < confidence < 1:
        raise ValueError("need finite whole-unit observations")
    result = {"count": len(x), "mean": float(x.mean()),
              "minimum": float(x.min()), "maximum": float(x.max()),
              "median": float(np.median(x)), "zero_count": int(np.count_nonzero(x == 0))}
    sd = float(x.std(ddof=1)) if len(x) > 1 else None
    se = sd / np.sqrt(len(x)) if sd is not None else None
    half = float(t.ppf((1 + confidence) / 2, len(x) - 1) * se) if se is not None else None
    result.update(standard_deviation=sd, standard_error=se,
                  confidence_interval=None if half is None else [result["mean"]-half, result["mean"]+half],
                  degenerate_empirical_variance=sd == 0,
                  uncertainty="whole-unit t approximation conditional on fixed cohort; zero empirical variance does not establish population equivalence")
    return result


def mobius_coefficients(table, k):
    """Anchored Boolean-subcube coefficients, NOT whole-network spectral modes."""
    if set(table) != set(subcube_states(k)) or not np.isfinite(list(table.values())).all():
        raise ValueError("need every finite candidate-state response exactly once")
    coefficients = np.zeros(2**k, dtype=np.float64)
    for state, risk in table.items():
        coefficients[sum(1 << i for i in state)] = risk
    for i in range(k):
        for mask in range(2**k):
            if mask & (1 << i):
                coefficients[mask] -= coefficients[mask ^ (1 << i)]
    return {tuple(i for i in range(k) if mask & (1 << i)): float(value)
            for mask, value in enumerate(coefficients)}


def decompose_field(table, clean_isolated, *, k, budget, probability):
    """Exact risk and all signed additive/interaction terms for every legal mask.

    J_G(Q) = R0 + p sum_{b notin Q} d0_b + (h_G(0)-R0)
             + p sum_{b notin Q}(dG_b-d0_b) + within_G(Q).
    The common background intercept cancels in every mask contrast.
    """
    if set(clean_isolated) != {(), *((i,) for i in range(k))}:
        raise ValueError("clean table must contain empty and all singletons on the same cohort")
    if not np.isfinite(list(clean_isolated.values())).all() or not 0 <= probability <= 1:
        raise ValueError("invalid clean response or Bernoulli probability")
    coefficients = mobius_coefficients(table, k)
    masks = fixed_budget_masks(k, budget)
    clean = float(clean_isolated[()]); base = float(table[()])
    d0 = np.array([clean_isolated[(i,)] - clean for i in range(k)])
    dg = np.array([table[(i,)] - base for i in range(k)])
    rows = []
    for mask in masks:
        available = set(range(k)) - set(mask)
        a0 = float(probability * sum(d0[i] for i in available))
        ag = float(probability * sum(dg[i] for i in available))
        exact = integrate_response_table(table, k, probability, protected=mask)
        # Center before integration to resolve small decision contrasts without
        # subtracting two large common intercepts. Original absolute risk checked.
        centered = integrate_response_table({s: v-base for s, v in table.items()}, k, probability, protected=mask)
        order = [float(sum(c * probability**len(s) for s, c in coefficients.items()
                           if len(s) == degree and set(s) <= available)) for degree in range(2, k+1)]
        within = centered - ag
        error = max(abs(exact - (base + centered)), abs(within - sum(order)))
        if error > 1e-11 * max(1., abs(exact), abs(base)):
            raise ArithmeticError("exact response decomposition failed")
        rows.append({"mask": list(mask), "exact_risk": exact,
                     "clean_additive_risk": clean+a0, "conditioned_additive_risk": base+ag,
                     "background_intercept_shift": base-clean,
                     "clean_term": a0, "outside_correction": ag-a0,
                     "within_correction": within, "centered_exact": centered,
                     "within_by_order": order, "decomposition_error": error})
    return rows


def contrast(target, baseline):
    terms = {key: float(target[key]-baseline[key]) for key in
             ("clean_term", "outside_correction", "within_correction", "centered_exact")}
    exact = terms.pop("centered_exact")
    residual = exact - sum(terms.values())
    direct = target["exact_risk"] - baseline["exact_risk"]
    if max(abs(residual), abs(direct-exact)) > 1e-11 * max(1., abs(target["exact_risk"]), abs(baseline["exact_risk"])):
        raise ArithmeticError("mask contrast attribution does not close")
    return {"target": target["mask"], "baseline": baseline["mask"], **terms,
            "exact_contrast": exact, "direct_absolute_risk_contrast": direct,
            "closure_error": residual, "within_by_order":
            [float(a-b) for a, b in zip(target["within_by_order"], baseline["within_by_order"])]}


def ordering_comparison(predicted, exact):
    from scipy.stats import spearmanr, kendalltau
    x, y = np.asarray(predicted), np.asarray(exact)
    if x.shape != y.shape or x.ndim != 1 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("aligned finite mask risks required")
    strict = reversed_pairs = prediction_ties = exact_ties = 0
    for i, j in itertools.combinations(range(len(x)), 2):
        a, b = float(x[i]-x[j]), float(y[i]-y[j])
        prediction_ties += abs(a) <= TIE_ATOL
        exact_ties += abs(b) <= TIE_ATOL
        if abs(a) > TIE_ATOL and abs(b) > TIE_ATOL:
            strict += 1; reversed_pairs += a*b < 0
    nonconstant = np.ptp(x) > TIE_ATOL and np.ptp(y) > TIE_ATOL
    return {"mask_count": len(x), "pair_count": len(x)*(len(x)-1)//2,
            "spearman_raw_tie_ranks": float(spearmanr(x, y).statistic) if nonconstant else None,
            "kendall_tau_b_raw_ties": float(kendalltau(x, y).statistic) if nonconstant else None,
            "strict_comparable_pairs": strict, "strict_reversals": int(reversed_pairs),
            "prediction_near_tie_pairs": int(prediction_ties), "exact_near_tie_pairs": int(exact_ties),
            "diagnostic_near_tie_atol": TIE_ATOL}


def selection_diagnostics(risks, masks, selected, *, seed, replicates=500):
    """Shared field resampling of all masks; descriptive post-selection stability."""
    from scipy.stats import t
    x = np.asarray(risks, dtype=np.float64)
    if (x.ndim != 2 or x.shape[1] != len(masks) or len(x) < 2
            or len(masks) < 2 or len(set(map(tuple, masks))) != len(masks)
            or not np.isfinite(x).all()):
        raise ValueError("need >=2 paired fields by >=2 unique legal masks")
    if replicates < 2:
        raise ValueError("need >=2 diagnostic bootstrap draws")
    means = x.mean(axis=0); order = np.argsort(means, kind="stable")
    best, second = order[:2]
    selected = list(map(tuple, masks)).index(tuple(selected))
    gaps = x[:, second] - x[:, best]
    gap = mean_interval(gaps)
    rng = np.random.Generator(np.random.PCG64(seed))
    counts = np.zeros(len(masks), dtype=int); selected_near_wins = 0
    for _ in range(replicates):
        m = x[rng.integers(len(x), size=len(x))].mean(axis=0)
        counts[int(np.argmin(m))] += 1
        selected_near_wins += m[selected] - m.min() <= TIE_ATOL
    prefix = []
    for n in range(1, len(x)+1):
        m = x[:n].mean(axis=0); idx = np.argsort(m, kind="stable")
        prefix.append({"fields": n, "winner": list(masks[idx[0]]), "margin": float(m[idx[1]]-m[idx[0]]),
                       "near_tied_best_count": int(np.count_nonzero(m-m.min() <= TIE_ATOL))})
    leave_one = (x.sum(axis=0)-x)/(len(x)-1)
    scenarios = [{"fields": n, "runner_up_gap_projected_t95_halfwidth":
                  float(t.ppf(.975, n-1)*gap["standard_deviation"]/np.sqrt(n))}
                 for n in (8, 16, 32, 64, 128, 256)]
    return {"fields": len(x), "masks": len(masks), "best": list(masks[best]),
            "runner_up": list(masks[second]), "frozen_selected": list(masks[selected]),
            "best_runner_up_gap": gap,
            "gap_to_paired_field_se": gap["mean"]/gap["standard_error"] if gap["standard_error"] else None,
            "exact_tied_best_count": int(np.count_nonzero(means == means.min())),
            "near_tied_best_count": int(np.count_nonzero(means-means.min() <= TIE_ATOL)),
            "per_field_winners": [list(masks[i]) for i in np.argmin(x, axis=1)],
            "prefix": prefix, "cumulative_switches": sum(a["winner"] != b["winner"] for a,b in zip(prefix,prefix[1:])),
            "leave_one_field_out_selected_fraction": float(np.mean(np.argmin(leave_one, axis=1) == selected)),
            "bootstrap": {"seed": seed, "replicates": replicates, "winner_counts": counts.tolist(),
                          "selected_near_best_fraction": selected_near_wins/replicates},
            "field_count_sensitivity": scenarios,
            "interpretation": "Descriptive selection-on-same-fields diagnostic, not a confidence probability of true optimality or an unbiased regret estimate. T widths hold empirical variance fixed, are not formal power, and miss unseen rare tails. Masks are paired, not independent observations."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--campaign", type=Path, action="append",
                        help="Frozen child campaign.json; repeat to include all BER arms")
    source.add_argument("--sources", type=Path, help="Versioned hash-pinned offline source inventory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=910001)
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    if args.sources:
        args.campaign = source_paths(args.sources)
    from .interaction_evidence import analyze_campaigns
    print(json.dumps(analyze_campaigns(args.campaign, args.output, seed=args.seed,
              replicates=args.bootstrap_replicates, plots=not args.no_plots), indent=2))


def source_paths(path):
    from .manifests import sha256_file
    doc = json.loads(Path(path).read_text())
    if set(doc) != {"schema_version", "description", "campaigns", "exclusions"} or doc["schema_version"] != VERSION:
        raise ValueError("unsupported offline source inventory")
    paths = []
    for row in doc["campaigns"]:
        if set(row) != {"path", "sha256"}:
            raise ValueError("every campaign requires an exact path and hash")
        p = Path(row["path"])
        if not p.is_absolute() or sha256_file(p) != row["sha256"]:
            raise ValueError("source campaign path/hash mismatch")
        paths.append(p)
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("source campaigns must be nonempty and unique")
    return paths


if __name__ == "__main__":
    main()
