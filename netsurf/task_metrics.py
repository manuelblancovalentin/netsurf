"""Typed task-generic measurements introduced by study schema 3.4."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .identities import BitIdentity


@dataclass(frozen=True)
class TaskTrialMeasurement:
    trial: int
    policy: str
    schedule_sha256: str
    primary_metric_name: str
    primary_metric_value: float
    primary_metric_direction: str
    loss_metric_name: str
    loss_metric_value: float
    loss_metric_direction: str
    schedule_hash: str
    proposed_events: int
    applied_events: int
    suppressed_events: int
    unique_applied_bits: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TaskSingleBitMeasurement:
    parameter: str
    flat_index: int
    bit_index: int
    layer: str
    parameter_class: str
    original_encoded: int
    flipped_encoded: int
    numerical_perturbation: float
    loss_metric_name: str
    clean_loss: float
    faulted_loss: float
    signed_loss_difference: float
    positive_loss_damage: float
    absolute_loss_difference: float
    primary_metric_name: str
    primary_metric_direction: str
    clean_primary_metric: float
    faulted_primary_metric: float
    primary_metric_damage: float
    prediction_change_count: int | None
    evaluation_sample_sha256: str
    runtime_seconds: float
    inclusion_rule: str
    sampling_probability: float
    score_quantile: int | None

    @property
    def identity(self) -> BitIdentity:
        return BitIdentity(self.parameter, self.flat_index, self.bit_index)

    def to_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def evaluation_contract(evaluation) -> tuple[str, float, str, str, float, str]:
    """Return primary/loss names, values and directions without aliases."""

    if hasattr(evaluation, "accuracy") and hasattr(evaluation, "categorical_crossentropy"):
        return (
            "classification_accuracy", float(evaluation.accuracy), "maximize",
            "categorical_crossentropy", float(evaluation.categorical_crossentropy), "minimize",
        )
    required = (
        "primary_metric_name", "primary_metric_value",
        "loss_metric_name", "loss_metric_value",
    )
    if not all(hasattr(evaluation, name) for name in required):
        raise TypeError("adapter evaluation does not expose the task-metric contract")
    primary_name = str(evaluation.primary_metric_name)
    loss_name = str(evaluation.loss_metric_name)
    return (
        primary_name, float(evaluation.primary_metric_value), "maximize",
        loss_name, float(evaluation.loss_metric_value), "minimize",
    )
