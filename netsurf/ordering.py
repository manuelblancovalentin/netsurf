"""Composable, deterministic ordering policies for susceptibility scores."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping, Sequence

import numpy as np

from .identities import BitIdentity, ParameterSpec
from .ranking import Ranking


SUPPORTED_CRITERIA = {
    "bit_significance_desc",
    "estimator_score_desc",
    "canonical",
    "reverse_canonical",
    "random",
    "layer_input_first",
    "layer_output_first",
    "stable_identity",
}

ALIASES = {
    "ud": "layer_input_first",
    "du": "layer_output_first",
}


@dataclass(frozen=True)
class OrderedRanking:
    name: str
    estimator: str
    criteria: tuple[str, ...]
    identities: tuple[BitIdentity, ...]
    seed: int | None

    def digest(self) -> str:
        payload = [[value.parameter, value.flat_index, value.bit_index] for value in self.identities]
        return hashlib.sha256(
            json.dumps(payload, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


def _layer_name(parameter: str) -> str:
    return parameter.split("/", 1)[0]


def compose_ordering(
    name: str,
    ranking: Ranking,
    identities: Sequence[BitIdentity],
    parameter_specs: Sequence[ParameterSpec],
    criteria: Sequence[str],
    *,
    seed: int | None = None,
) -> OrderedRanking:
    """Apply lexicographic criteria without allowing later keys across tiers."""

    normalized = tuple(ALIASES.get(value, value) for value in criteria)
    unknown = set(normalized) - SUPPORTED_CRITERIA
    if unknown:
        raise ValueError(f"unsupported ordering criteria: {sorted(unknown)}")
    if not normalized:
        raise ValueError("at least one ordering criterion is required")
    universe = tuple(identities)
    if len(set(universe)) != len(universe):
        raise ValueError("candidate identities contain duplicates")
    if set(ranking.scores) != set(universe):
        raise ValueError("estimator and ordering candidate populations differ")
    if "random" in normalized and seed is None:
        raise ValueError("random ordering criterion requires a seed")

    canonical = {value: index for index, value in enumerate(sorted(universe))}
    parameter_order = {value.name: index for index, value in enumerate(parameter_specs)}
    layers: list[str] = []
    for specification in parameter_specs:
        layer = _layer_name(specification.name)
        if layer not in layers:
            layers.append(layer)
    layer_order = {value: index for index, value in enumerate(layers)}
    random_key: Mapping[BitIdentity, float] = {}
    if "random" in normalized:
        generator = np.random.default_rng(int(seed))
        random_key = {
            identity: float(value)
            for identity, value in zip(sorted(universe), generator.random(len(universe)))
        }

    def key(identity: BitIdentity):
        result: list[float | int] = []
        for criterion in normalized:
            if criterion == "bit_significance_desc":
                result.append(-identity.bit_index)
            elif criterion == "estimator_score_desc":
                result.append(-float(ranking.scores[identity]))
            elif criterion in {"canonical", "stable_identity"}:
                result.append(canonical[identity])
            elif criterion == "reverse_canonical":
                result.append(-canonical[identity])
            elif criterion == "random":
                result.append(random_key[identity])
            elif criterion == "layer_input_first":
                result.extend((layer_order[_layer_name(identity.parameter)], parameter_order[identity.parameter]))
            elif criterion == "layer_output_first":
                result.extend((-layer_order[_layer_name(identity.parameter)], -parameter_order[identity.parameter]))
        # Always provide a deterministic final tie break. It cannot cross an
        # earlier tier because Python tuple comparison is lexicographic.
        result.append(canonical[identity])
        return tuple(result)

    ordered = tuple(sorted(universe, key=key))
    if set(ordered) != set(universe) or len(ordered) != len(universe):
        raise RuntimeError("ordering lost or duplicated candidates")
    return OrderedRanking(name, ranking.estimator, normalized, ordered, seed)


def standard_policy_definition(name: str) -> tuple[tuple[str, ...], str]:
    """Return unambiguous criteria and required estimator for named policies."""

    definitions = {
        "msb+ordered": (("bit_significance_desc", "canonical"), "msb_first"),
        "msb+reversed": (("bit_significance_desc", "reverse_canonical"), "msb_first"),
        "msb+random": (("bit_significance_desc", "random"), "msb_first"),
        "msb+layer_input_first": (("bit_significance_desc", "layer_input_first", "stable_identity"), "msb_first"),
        "msb+layer_output_first": (("bit_significance_desc", "layer_output_first", "stable_identity"), "msb_first"),
        "msb+gradient": (("bit_significance_desc", "estimator_score_desc", "stable_identity"), "first_order_gradient_delta"),
        "gradient_global": (("estimator_score_desc", "stable_identity"), "first_order_gradient_delta"),
        "random_global": (("random", "stable_identity"), "random"),
    }
    try:
        return definitions[name]
    except KeyError as error:
        raise ValueError(f"unknown standard policy: {name}") from error
