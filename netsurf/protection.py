"""Protection policies that map scores and costs to immutable bit sets."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping, Sequence

from .identities import BitIdentity
from .ranking import Ranking


@dataclass(frozen=True)
class ProtectionMask:
    policy: str
    protected: frozenset[BitIdentity]
    universe_size: int
    budget: float
    metadata: Mapping[str, object]

    @property
    def coverage(self) -> float:
        return len(self.protected) / self.universe_size if self.universe_size else 0.0

    def digest(self) -> str:
        payload = [identity.key() for identity in sorted(self.protected)]
        return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode("utf-8")).hexdigest()


class ProtectionPolicy:
    """Deterministic allocation policies; ties use stable bit identity order."""

    @staticmethod
    def _count(universe_size: int, coverage: float) -> int:
        if not 0.0 <= coverage <= 1.0:
            raise ValueError("coverage must be in [0, 1]")
        return min(universe_size, max(0, int(math.floor(coverage * universe_size + 1e-12))))

    @classmethod
    def global_ranking(cls, ranking: Ranking, coverage: float) -> ProtectionMask:
        count = cls._count(len(ranking.scores), coverage)
        protected = frozenset(ranking.ordered()[:count])
        return ProtectionMask(
            policy="static_global_ranking",
            protected=protected,
            universe_size=len(ranking.scores),
            budget=float(count),
            metadata={"ranking": ranking.estimator, "requested_coverage": coverage},
        )

    @classmethod
    def position_first(cls, ranking: Ranking, coverage: float) -> ProtectionMask:
        count = cls._count(len(ranking.scores), coverage)
        ordered = sorted(
            ranking.scores,
            key=lambda identity: (
                -identity.bit_index,
                -float(ranking.scores[identity]),
                identity,
            ),
        )
        return ProtectionMask(
            policy="position_first_ranking",
            protected=frozenset(ordered[:count]),
            universe_size=len(ranking.scores),
            budget=float(count),
            metadata={"ranking": ranking.estimator, "requested_coverage": coverage},
        )

    @classmethod
    def nonuniform_cost(
        cls,
        ranking: Ranking,
        costs: Mapping[BitIdentity, float],
        budget: float,
    ) -> ProtectionMask:
        if budget < 0:
            raise ValueError("budget must be nonnegative")
        missing = set(ranking.scores) - set(costs)
        if missing:
            raise ValueError(f"missing costs for {len(missing)} bits")
        if any(costs[identity] <= 0 for identity in ranking.scores):
            raise ValueError("all costs must be positive")
        ordered = sorted(
            ranking.scores,
            key=lambda identity: (
                -float(ranking.scores[identity]) / float(costs[identity]),
                identity,
            ),
        )
        selected: list[BitIdentity] = []
        spent = 0.0
        for identity in ordered:
            cost = float(costs[identity])
            if spent + cost <= budget:
                selected.append(identity)
                spent += cost
        return ProtectionMask(
            policy="greedy_score_per_cost",
            protected=frozenset(selected),
            universe_size=len(ranking.scores),
            budget=float(budget),
            metadata={"ranking": ranking.estimator, "spent": spent, "optimality": "not claimed"},
        )
