"""Deterministic task risks for pinned transformer benchmark cohorts."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence
from typing import Mapping

import numpy as np


FAILURE_PENALTY_NATS = 30.0
RISK_VERSION = "transformer-risk/v1"


@dataclass(frozen=True)
class TaskRisk:
    mean_nll: float
    accuracy: float | None
    perplexity: float | None
    failed_units: int
    total_units: int
    logits: tuple[tuple[float, ...] | None, ...]
    failure_kinds: tuple[str | None, ...]
    logit_record_kind: str = "full_logits"

    @property
    def failure_fraction(self) -> float:
        return self.failed_units / self.total_units

    @property
    def sample_count(self) -> int:
        return self.total_units

    @property
    def metrics(self) -> Mapping[str, float]:
        result = {"mean_nll": self.mean_nll, "failure_fraction": self.failure_fraction}
        if self.accuracy is not None:
            result["accuracy"] = self.accuracy
        if self.perplexity is not None:
            result["perplexity"] = self.perplexity
        return result


def classification_risk(
    logits: Sequence[Sequence[float] | None], labels: Sequence[int], *,
    penalty_nats: float = FAILURE_PENALTY_NATS,
) -> TaskRisk:
    if len(logits) != len(labels) or not labels:
        raise ValueError("one logit row and public label required per example")
    if not math.isfinite(penalty_nats) or penalty_nats <= 0:
        raise ValueError("finite positive failure penalty required")
    losses, correct, failures, recorded, kinds = [], 0, 0, [], []
    for row, label in zip(logits, labels):
        if label not in (0, 1):
            raise ValueError("SST-2 labels must be 0 or 1")
        if row is None:
            failures += 1; losses.append(penalty_nats); recorded.append(None); kinds.append("inference_failure"); continue
        values = np.asarray(row, dtype=np.float64)
        if values.shape != (2,):
            raise ValueError("SST-2 requires two logits")
        if not np.all(np.isfinite(values)):
            failures += 1; losses.append(penalty_nats); recorded.append(None); kinds.append("nonfinite_logits"); continue
        recorded.append(tuple(float(v) for v in values)); kinds.append(None)
        maximum = float(np.max(values))
        nll = maximum + math.log(float(np.sum(np.exp(values - maximum)))) - float(values[label])
        losses.append(min(max(0.0, nll), penalty_nats))
        correct += int(np.argmax(values) == label)
    return TaskRisk(float(np.mean(losses)), correct / len(labels), None, failures,
                    len(labels), tuple(recorded), tuple(kinds))


def teacher_forced_windows(token_ids: Sequence[int], context: int, stride: int) -> tuple[tuple[int, int, int], ...]:
    """(start, end, first scored target) with every target 1..N-1 scored once."""
    if context < 2 or not 1 <= stride < context or len(token_ids) < 2:
        raise ValueError("context >= 2, 1 <= stride < context and >= 2 tokens required")
    result = []
    start, first = 0, 1
    while True:
        end = min(len(token_ids), start + context)
        if first <= start or first >= end:
            raise ValueError("window has no preceding context or scored target")
        result.append((start, end, first))
        if end == len(token_ids):
            break
        first = end
        start += stride
    return tuple(result)


def language_model_risk(
    target_logits: Sequence[Sequence[float] | None], target_ids: Sequence[int], *,
    penalty_nats: float = FAILURE_PENALTY_NATS,
) -> TaskRisk:
    if len(target_logits) != len(target_ids) or not target_ids:
        raise ValueError("one scored logit row required per target token")
    losses, failures, recorded, kinds = [], 0, [], []
    for row, target in zip(target_logits, target_ids):
        if row is None:
            losses.append(penalty_nats); failures += 1; recorded.append(None); kinds.append("inference_failure"); continue
        values = np.asarray(row, dtype=np.float64)
        if values.ndim != 1 or values.size < 2 or not 0 <= target < values.size:
            raise ValueError("invalid token logit row or target")
        if not np.all(np.isfinite(values)):
            losses.append(penalty_nats); failures += 1; recorded.append(None); kinds.append("nonfinite_logits"); continue
        recorded.append(tuple(float(v) for v in values)); kinds.append(None)
        maximum = float(np.max(values))
        nll = maximum + math.log(float(np.sum(np.exp(values - maximum)))) - float(values[target])
        losses.append(min(max(0.0, nll), penalty_nats))
    mean = float(np.mean(losses))
    return TaskRisk(mean, None, math.exp(mean), failures, len(target_ids), tuple(recorded), tuple(kinds))


def language_model_risk_from_stats(
    target_logit_and_logsumexp: Sequence[tuple[float, float] | None], *,
    penalty_nats: float = FAILURE_PENALTY_NATS,
) -> TaskRisk:
    """Score recorded sufficient logit statistics without retaining full vocab rows."""
    if not target_logit_and_logsumexp or not math.isfinite(penalty_nats) or penalty_nats <= 0:
        raise ValueError("nonempty token statistics and finite positive penalty required")
    losses, failures, recorded, kinds = [], 0, [], []
    for item in target_logit_and_logsumexp:
        if item is None:
            losses.append(penalty_nats); failures += 1; recorded.append(None); kinds.append("inference_failure"); continue
        target_logit, logsumexp = item
        if not math.isfinite(target_logit) or not math.isfinite(logsumexp):
            losses.append(penalty_nats); failures += 1; recorded.append(None); kinds.append("nonfinite_logits"); continue
        losses.append(min(max(0.0, logsumexp - target_logit), penalty_nats))
        recorded.append((float(target_logit), float(logsumexp))); kinds.append(None)
    mean = float(np.mean(losses))
    return TaskRisk(mean, None, math.exp(mean), failures, len(losses), tuple(recorded),
                    tuple(kinds), "target_logit_and_logsumexp")
