"""Susceptibility estimators and immutable ranking results.

Estimators produce scores only. Ordering constraints and cost allocation belong
to :mod:`netsurf.protection`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib
import json
from typing import Mapping, Sequence

import numpy as np

from .identities import BitIdentity


@dataclass(frozen=True)
class Ranking:
    estimator: str
    scores: Mapping[BitIdentity, float]
    metadata: Mapping[str, object]

    def ordered(self, *, descending: bool = True) -> tuple[BitIdentity, ...]:
        direction = -1.0 if descending else 1.0
        return tuple(
            identity
            for identity, _ in sorted(
                self.scores.items(),
                key=lambda item: (direction * float(item[1]), item[0]),
            )
        )

    def digest(self) -> str:
        payload = [
            [identity.parameter, identity.flat_index, identity.bit_index, float(score)]
            for identity, score in sorted(self.scores.items())
        ]
        return hashlib.sha256(
            json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
        ).hexdigest()


class SusceptibilityEstimator(ABC):
    name: str

    @abstractmethod
    def estimate(self, *args, **kwargs) -> Ranking:
        raise NotImplementedError


class RandomEstimator(SusceptibilityEstimator):
    name = "random"

    def __init__(self, seed: int):
        self.seed = int(seed)

    def estimate(self, identities: Sequence[BitIdentity]) -> Ranking:
        generator = np.random.default_rng(self.seed)
        scores = {identity: float(score) for identity, score in zip(identities, generator.random(len(identities)))}
        return Ranking(self.name, scores, {"seed": self.seed})


class MSBEstimator(SusceptibilityEstimator):
    """Correct bit-position score: larger encoded indices are more significant."""

    name = "msb_first"

    def estimate(self, identities: Sequence[BitIdentity]) -> Ranking:
        return Ranking(self.name, {identity: float(identity.bit_index) for identity in identities}, {})


class GradientMagnitudeEstimator(SusceptibilityEstimator):
    """Absolute parameter-gradient score copied to every bit of a scalar."""

    name = "gradient_magnitude"

    def estimate(
        self,
        identities: Sequence[BitIdentity],
        gradients: Mapping[str, np.ndarray],
    ) -> Ranking:
        scores = {
            identity: float(abs(gradients[identity.parameter].reshape(-1)[identity.flat_index]))
            for identity in identities
        }
        return Ranking(self.name, scores, {"expression": "abs(dL/dw)"})


class FirstOrderEstimator(SusceptibilityEstimator):
    """Elementary first-order score ``abs(dL/dw * delta_w_bit)``."""

    name = "first_order_gradient_delta"

    def estimate(
        self,
        identities: Sequence[BitIdentity],
        gradients: Mapping[str, np.ndarray],
        deltas: Mapping[str, np.ndarray],
    ) -> Ranking:
        scores: dict[BitIdentity, float] = {}
        for identity in identities:
            gradient = gradients[identity.parameter].reshape(-1)[identity.flat_index]
            delta = deltas[identity.parameter].reshape(-1, deltas[identity.parameter].shape[-1])[
                identity.flat_index, identity.bit_index
            ]
            scores[identity] = float(abs(gradient * delta))
        return Ranking(self.name, scores, {"expression": "abs(dL/dw * delta_w_bit)"})


class LegacyQPolarGradEstimator(SusceptibilityEstimator):
    """Recovered historical qPolarGrad expression, isolated from its policy.

    The caller supplies the legacy activation-drift factor ``P``. This keeps the
    recovered formula ``abs(P * dL/dw)`` explicit and testable while avoiding
    the legacy ranker's mutable ``*_N`` state. Whether ``P`` is computed from
    clean or globally attacked activations is recorded by the caller.
    """

    name = "legacy_qpolargrad"

    def estimate(
        self,
        identities: Sequence[BitIdentity],
        gradients: Mapping[str, np.ndarray],
        polar_impacts: Mapping[str, np.ndarray],
        *,
        activation_source: str,
    ) -> Ranking:
        scores: dict[BitIdentity, float] = {}
        for identity in identities:
            gradient = gradients[identity.parameter].reshape(-1)[identity.flat_index]
            impact = polar_impacts[identity.parameter].reshape(
                -1, polar_impacts[identity.parameter].shape[-1]
            )[identity.flat_index, identity.bit_index]
            scores[identity] = float(abs(impact * gradient))
        return Ranking(
            self.name,
            scores,
            {
                "expression": "abs(P_bit * dL/dw)",
                "activation_source": activation_source,
                "warning": "historical heuristic; not an expected-loss estimator",
            },
        )
