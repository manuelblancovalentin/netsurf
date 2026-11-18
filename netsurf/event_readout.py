"""Classification accuracy under independent bit faults from measured single-flip effects (EXPLORATORY).

Side-effect free, NumPy and SciPy only.  Plan: ``review/TOPBIT_ANALYSIS_PLAN_2026-11-14.md`` (paper worktree).

Model.  The output of the network is a vector of INT8 codes (one per class).  For clip ``i`` and a fault pattern,

    scores_i = clean_i + background_i + sum over flipped event bits b of delta_ib            (additive model)

where ``delta_ib`` is the measured change of the output codes when bit ``b`` alone is flipped and ``background_i`` the
change caused by everything that is not an event bit.  Codes are integers, so every sum is exact; the final scores are
clipped to the INT8 range as the network's output is.  The clip is classified correctly when the label has the largest
score, with ties going to the lowest class index (``numpy.argmax``).

Readouts of the expected accuracy at bit-flip probability ``p`` use the margin against the clean runner-up class
``k*``:  M_i = m0_i + S_i + B_i,  m0 = clean[y] - clean[k*],  S = sum_b X_b a_ib,  a_ib = delta_ib[y] - delta_ib[k*],
X_b ~ Bernoulli(p) independent, B_i the background margin change.  This ignores a third class overtaking the label and
the clipping; the plan reports how much that matters by comparing with the full-score additive model.

  event_sum_pmf             exact distribution of S on the integers (grouped binomials, convolved; the mass dropped
                            by truncation is returned and bounded by ``eps``).  This is the characteristic-function
                            (Fourier) formula evaluated on its lattice.
  lattice_correct_probability   P(correct) from that distribution and a Gaussian background N(mu_B, var_B).
  gaussian_correct_probability  the same with S replaced by its two moments (bell-curve readout).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve
from scipy.special import ndtr
from scipy.stats import binom

INT8_MIN, INT8_MAX = -128, 127


def correct(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """``argmax(scores) == label`` per clip (ties to the lowest class index)."""
    return np.asarray(scores).argmax(-1) == np.asarray(labels)


def additive_scores(clean: np.ndarray, flipped_codes: np.ndarray | None = None, background: np.ndarray | None = None,
                    clip: bool = True) -> np.ndarray:
    """Scores under the additive model.

    clean          (C, K) int clean output codes
    flipped_codes  (n, C, K) int output codes measured with each event bit flipped alone (only the flipped events)
    background     (C, K) int output codes measured with the background faults alone, or None
    """
    clean = np.asarray(clean, dtype=np.int32)
    s = clean.copy()
    if flipped_codes is not None and len(flipped_codes):
        s += (np.asarray(flipped_codes, dtype=np.int32) - clean[None]).sum(0)
    if background is not None:
        s += np.asarray(background, dtype=np.int32) - clean
    return np.clip(s, INT8_MIN, INT8_MAX) if clip else s


def runner_up(clean: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Index of the highest-scoring class other than the label, per clip (ties to the lowest index)."""
    z = np.asarray(clean, dtype=np.float64).copy()
    z[np.arange(len(labels)), labels] = -np.inf
    return z.argmax(-1)


def tie_threshold(labels: np.ndarray, rival: np.ndarray) -> np.ndarray:
    """Smallest integer margin that is classified correctly: 0 when a tie goes to the label (label index lower)."""
    return np.where(np.asarray(labels) < np.asarray(rival), 0, 1)


def _binomial_on_lattice(value: int, count: int, p: float, eps: float):
    """(lo, pmf, dropped) of value * Binomial(count, p), truncated where the upper tail is below eps."""
    kmax = int(binom.isf(eps, count, p)) + 1 if p > 0 else 0
    kmax = min(max(kmax, 0), count)
    k = np.arange(kmax + 1)
    w = binom.pmf(k, count, p)
    dropped = max(0.0, 1.0 - float(w.sum()))
    step = abs(int(value))
    pmf = np.zeros(kmax * step + 1)
    pmf[k * step] = w
    if value < 0:
        return -kmax * step, pmf[::-1].copy(), dropped
    return 0, pmf, dropped


def _trim(lo: int, pmf: np.ndarray, eps: float):
    pmf = np.where(pmf < 0, 0.0, pmf)                     # FFT round-off
    c = np.cumsum(pmf)
    total = c[-1]
    left = int(np.searchsorted(c, eps / 2, side="right"))
    right = int(np.searchsorted(c, total - eps / 2, side="left"))
    right = min(max(right, left), len(pmf) - 1)
    dropped = float(c[left - 1]) if left > 0 else 0.0
    dropped += float(total - c[right])
    return lo + left, pmf[left:right + 1].copy(), dropped


def event_sum_pmf(effects, p: float, eps: float = 1e-13):
    """Distribution of S = sum_b X_b a_b for integer effects a_b and X_b ~ Bernoulli(p) independent.

    Returns (lo, pmf, dropped): pmf[j] = P(S = lo + j) up to the truncated mass ``dropped`` (at most a few times
    eps per distinct effect value).
    """
    a = np.asarray(effects, dtype=np.int64)
    a = a[a != 0]
    if a.size == 0 or p <= 0:
        return 0, np.array([1.0]), 0.0
    values, counts = np.unique(a, return_counts=True)
    lo, pmf, dropped = 0, np.array([1.0]), 0.0
    for v, c in sorted(zip(values.tolist(), counts.tolist()), key=lambda t: abs(t[0])):
        blo, bpmf, bd = _binomial_on_lattice(v, c, p, eps)
        dropped += bd
        if len(bpmf) == 1:
            continue
        conv = fftconvolve(pmf, bpmf) if min(len(pmf), len(bpmf)) > 64 else np.convolve(pmf, bpmf)
        lo, pmf, td = _trim(lo + blo, conv, eps)
        dropped += td
    return lo, pmf, dropped


def lattice_correct_probability(m0: int, effects, p: float, threshold: int, bg_mean: float = 0.0, bg_var: float = 0.0,
                                eps: float = 1e-13):
    """P(m0 + S + B >= threshold) with S from ``event_sum_pmf`` and B ~ N(bg_mean, bg_var).

    The background is a lattice variable approximated by a Gaussian, so a continuity correction of one half step is
    used: P(m0 + S + B > threshold - 1/2).  With bg_var = 0 the background is the constant bg_mean.
    Returns (probability, dropped mass).
    """
    lo, pmf, dropped = event_sum_pmf(effects, p, eps)
    s = lo + np.arange(len(pmf))
    x = m0 + s + bg_mean - (threshold - 0.5)
    if bg_var > 0:
        q = ndtr(x / np.sqrt(bg_var))
    else:
        q = (x > 0).astype(float)
    return float(np.dot(pmf, q)), dropped


def gaussian_correct_probability(m0, effects, p: float, threshold, bg_mean=0.0, bg_var=0.0):
    """Bell-curve readout: S replaced by N(p sum a, p (1 - p) sum a^2); same continuity correction.

    ``effects`` may be (n,) for one clip or (C, n) for C clips (with array m0, threshold, bg_mean, bg_var)."""
    a = np.asarray(effects, dtype=np.float64)
    mean = p * a.sum(-1) + np.asarray(bg_mean, dtype=np.float64)
    var = p * (1 - p) * (a * a).sum(-1) + np.asarray(bg_var, dtype=np.float64)
    x = np.asarray(m0, dtype=np.float64) + mean - (np.asarray(threshold) - 0.5)
    sd = np.sqrt(var)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(sd > 0, ndtr(x / np.where(sd > 0, sd, 1.0)), (x > 0).astype(float))
    return out
