"""M11 outside-conditioned, candidate-forced-clean responses (not full influence).

No model dependency. All risks are scalar, including complete-cohort AUC risk.
The finite-field oracle is restricted to K<=12; it is not a population optimum.
"""
from __future__ import annotations

import itertools
import math
from typing import Callable

import numpy as np


def exposure(population: int, budget: int, *, probability=None, expected=None):
    if not 0 <= budget < population or (probability is None) == (expected is None):
        raise ValueError("supply one exposure axis and 0 <= budget < population")
    p = float(probability if probability is not None else expected / (population-budget))
    if not 0 <= p <= 1:
        raise ValueError("Bernoulli probability must lie in [0,1]")
    return {"logical_ber": p, "expected_applied_flips": (population-budget)*p}


def error_bound(m: int, p: float, limit: float):
    if not isinstance(m, int) or m < 0 or not 0 <= p <= 1 or not math.isfinite(limit) or limit < 0:
        raise ValueError("invalid bounded-risk domain")
    return 0.0 if m < 2 else limit*m*p*(1-(1-p)**(m-1))


def finite_sample_bound(k, budget, p, limit, fields, alpha=.05):
    """Simultaneous Hoeffding coordinate bound, conditional on fixed inputs/pool.

    d is in [-L,L]. Equal-budget contrasts have at most 2 min(r,K-r)
    changed coordinates. No normal approximation or independent-bit assumption.
    """
    if not 0 <= budget <= k or fields < 1 or not 0 < alpha < 1 or k < 1:
        raise ValueError("invalid finite-sample domain")
    epsilon = error_bound(k-budget, p, limit)
    t = limit*math.sqrt(2*math.log(2*k/alpha)/fields)
    stochastic = 2*p*min(budget, k-budget)*t
    return {"approximation_error_bound": epsilon,
            "coordinate_sampling_error_bound": t,
            "regret_sampling_term": stochastic,
            "restricted_regret_bound": min(limit, 2*epsilon+stochastic),
            "confidence": 1-alpha, "independent_outside_fields": fields}


def measure_field(evaluate: Callable[[tuple[int, ...]], float], k: int, *, exhaustive=False):
    """Exactly K+1 calls, or exactly 2**K calls when auditing the oracle.

    evaluate(A) must persist the SAME outside G, force other candidates clean, and restore.
    """
    if not 1 <= k <= 64 or exhaustive and k > 12:
        raise ValueError("K must be <=64 (exhaustive oracle <=12)")
    states = [(), *((i,) for i in range(k))]
    if exhaustive:
        states.extend(s for n in range(2, k+1) for s in itertools.combinations(range(k), n))
    table = {}
    for state in states:
        value = float(evaluate(state))
        if not math.isfinite(value):
            raise ValueError("nonfinite authoritative task risk")
        table[state] = value
    return table


def additive_risks(table, masks, k, p):
    d = np.array([table[(i,)]-table[()] for i in range(k)])
    return np.array([table[()]+p*sum(d[i] for i in range(k) if i not in q) for q in masks])


def exact_mask_risks(table, masks, k, p):
    """Reuse the exact response table without extra inference; all orders retained."""
    if k > 12 or len(table) != 2**k or set(table) != {
        tuple(i for i in range(k) if state & (1 << i)) for state in range(2**k)
    }:
        raise ValueError("complete K<=12 response table required")
    risks = []
    for mask in masks:
        available = tuple(i for i in range(k) if i not in mask)
        value = 0.0
        for n in range(len(available)+1):
            mass = p**n*(1-p)**(len(available)-n)
            value += mass*math.fsum(table[s] for s in itertools.combinations(available, n))
        risks.append(value)
    return np.array(risks)


def choose(deltas, budget):
    values = np.asarray(deltas)
    if values.ndim != 1 or not np.isfinite(values).all() or not 0 <= budget <= len(values):
        raise ValueError("finite signed candidate scores and a legal budget required")
    # Largest damaging signed marginals are protected; beneficial negatives retained.
    return tuple(sorted(sorted(range(len(values)), key=lambda i: (-values[i], i))[:budget]))


def observed_curvature(table,k,p,masks):
    """Exact sup over candidate contexts for ONE OBSERVED outside G only."""
    kappas={}
    for i,j in itertools.combinations(range(k),2):
        values=[]
        for state in table:
            if i in state or j in state:continue
            values.append(abs(table[tuple(sorted((*state,i,j)))]-table[tuple(sorted((*state,i)))]-
                              table[tuple(sorted((*state,j)))]+table[state]))
        kappas[i,j]=max(values)
    return np.array([p*p*sum(v for (i,j),v in kappas.items() if i not in q and j not in q) for q in masks])


def compare_oracle(tables, clean, k, budget, p, original):
    from scipy.stats import spearmanr
    masks = tuple(itertools.combinations(range(k), budget))
    exact = np.mean([exact_mask_risks(t, masks, k, p) for t in tables], axis=0)
    outside = np.mean([additive_risks(t, masks, k, p) for t in tables], axis=0)
    isolated = additive_risks(clean, masks, k, p)
    # Center per field BEFORE integration: common large intercepts otherwise
    # obscure tiny mask contrasts. Absolute values are retained in all rows.
    centered_exact=np.mean([exact_mask_risks({s:v-t[()] for s,v in t.items()},masks,k,p) for t in tables],axis=0)
    centered_outside=np.mean([additive_risks({s:v-t[()] for s,v in t.items()},masks,k,p) for t in tables],axis=0)
    centered_clean=additive_risks({s:v-clean[()] for s,v in clean.items()},masks,k,p)
    best = int(np.argmin(centered_exact))
    rows = [{"mask": list(q), "exact": float(e), "outside_additive": float(a),
             "clean_additive": float(c), "outside_error": float(a-e), "clean_error": float(c-e)}
            for q,e,a,c in zip(masks,exact,outside,isolated)]
    diagnostics = {}
    for name, predicted, centered in (("outside", outside, centered_outside), ("clean", isolated, centered_clean)):
        winner = int(np.argmin(centered))
        reversals = wrong_signs = comparable = ties = 0
        for i in range(len(masks)):
            for j in range(i):
                a, e = centered[i]-centered[j], centered_exact[i]-centered_exact[j]
                comparable += int(e != 0)
                reversals += int(a*e < 0)
                wrong_signs += int(np.sign(a) != np.sign(e))
                ties += int(a == 0 or e == 0)
        rho = None if np.ptp(centered) == 0 or np.ptp(centered_exact) == 0 else float(spearmanr(centered,centered_exact).statistic)
        diagnostics[name] = {"selected_mask": list(masks[winner]),
            "max_absolute_error": float(np.max(np.abs(predicted-exact))),
            "mean_absolute_error": float(np.mean(np.abs(predicted-exact))),
            "finite_field_selection_regret": float(centered_exact[winner]-centered_exact[best]),
            "spearman": rho, "strict_reversals": reversals, "wrong_sign_or_tie": wrong_signs,
            "exact_nontied_pairs": comparable, "pairs_with_ties": ties}
    closure=float(np.max(np.abs(exact-centered_exact-np.mean([t[()] for t in tables]))))
    curvature=np.mean([observed_curvature(t,k,p,masks) for t in tables],axis=0)
    if np.any(np.abs(centered_exact-centered_outside)>curvature+1e-12):
        raise ArithmeticError("empirical-context curvature bound violated")
    return {"all_masks": rows, "diagnostics": diagnostics, "exact_empirical_optimum": list(masks[best]),
            "centering_closure_error":closure,
            "observed_curvature_bound_max":float(np.max(curvature)),
            "curvature_scope":"all candidate contexts on sampled outside fields ONLY; not a population certificate",
            "original_S5_empirical_regret": float(exact[masks.index(tuple(original))]-exact[best]),
            "interpretation": "same finite selection fields; NOT held-out regret or population optimum"}


def paired_summary(matrix, names, confidence=.95):
    """Whole-field paired t intervals; Bonferroni across all reported contrasts."""
    from scipy.stats import t
    a = np.asarray(matrix, dtype=float)
    if a.ndim != 2 or a.shape[1] != len(names) or not np.isfinite(a).all() or len(a) < 1:
        raise ValueError("finite field by frozen-mask matrix required")
    family = len(names)*(len(names)-1)//2
    result = []
    for i,j in itertools.combinations(range(len(names)),2):
        d = a[:,i]-a[:,j]
        se = float(np.std(d,ddof=1)/math.sqrt(len(d))) if len(d)>1 else None
        radius = float(t.ppf(1-(1-confidence)/(2*family),len(d)-1)*se) if se is not None else None
        mean = float(np.mean(d))
        result.append({"target":names[i], "baseline":names[j], "mean_difference":mean,
            "paired_fields":len(d), "standard_error":se,
            "simultaneous_interval":None if radius is None else [mean-radius,mean+radius],
            "minimum":float(np.min(d)), "maximum":float(np.max(d)),
            "negative_fraction":float(np.mean(d<0)), "positive_fraction":float(np.mean(d>0)),
            "method":"paired whole-field Student-t; Bonferroni family", "family_size":family,
            "conditions_on":"frozen masks, fixed pool, checkpoint and input cohort"})
    return result


def crossed_bootstrap(values, *, seed, replicates=1000, labels=None):
    """Paired field/input bootstrap. ToyCar values are FILE SCORES, not risks.

    Shape: fields x masks x inputs/files. For AUC, stratify complete files by
    class and recompute AUC. Same input and field draws for every method.
    Returns mask-minus-first contrasts. Assumes iid fields and exchangeable
    independent inputs (within class for files); does not resample selection.
    """
    a = np.asarray(values, dtype=float)
    if a.ndim != 3 or not np.isfinite(a).all() or replicates < 2:
        raise ValueError("finite field/mask/input array and >=2 resamples required")
    rng = np.random.default_rng(seed)
    if labels is not None:
        labels = np.asarray(labels)
        if labels.shape != (a.shape[2],) or set(labels.tolist()) != {0,1}:
            raise ValueError("complete normal and anomalous files required")
        n, z = np.flatnonzero(labels==0), np.flatnonzero(labels==1)
    result = []
    for _ in range(replicates):
        fields = rng.integers(a.shape[0],size=a.shape[0])
        if labels is None:
            inputs = rng.integers(a.shape[2],size=a.shape[2])
            risk = a[fields][:,:,inputs].mean(axis=(0,2))
        else:
            normal = a[fields][:,:,rng.choice(n,len(n),replace=True)]
            anomaly = a[fields][:,:,rng.choice(z,len(z),replace=True)]
            diff = anomaly[:,:,:,None]-normal[:,:,None,:]
            risk = (1-((diff>0)+.5*(diff==0)).mean(axis=(2,3))).mean(axis=0)
        result.append((risk-risk[0]).tolist())
    return np.asarray(result)
