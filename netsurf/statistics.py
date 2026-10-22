"""Small, explicit statistical summaries for paired Monte Carlo trials."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Estimate:
    count: int
    mean: float
    standard_deviation: float
    standard_error: float
    confidence: float
    confidence_interval: tuple[float, float]

    def to_dict(self):
        return asdict(self)


def summarize(
    values: Sequence[float],
    confidence: float = 0.95,
    bounds: tuple[float, float] | None = None,
) -> Estimate:
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or data.size == 0 or not np.all(np.isfinite(data)):
        raise ValueError("values must be a nonempty finite one-dimensional sequence")
    count = int(data.size)
    mean = float(np.mean(data))
    if count == 1:
        return Estimate(count, mean, 0.0, 0.0, confidence, (mean, mean))
    deviation = float(np.std(data, ddof=1))
    error = deviation / math.sqrt(count)
    try:
        from scipy.stats import t

        critical = float(t.ppf((1.0 + confidence) / 2.0, df=count - 1))
    except ImportError:
        from statistics import NormalDist

        critical = NormalDist().inv_cdf((1.0 + confidence) / 2.0)
    interval = (mean - critical * error, mean + critical * error)
    if bounds is not None:
        interval = (max(bounds[0], interval[0]), min(bounds[1], interval[1]))
    return Estimate(
        count,
        mean,
        deviation,
        error,
        confidence,
        interval,
    )
