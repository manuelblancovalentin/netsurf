"""PrioriFI's feedback-guided *probe order*, not a protection optimizer.

This is a deterministic, storage-bit-identity adaptation of the pinned public
smart-pixel implementation. The caller supplies a parameter ranking and an
isolated-fault metric; this module does not define either one.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median
from typing import Callable, Iterable, Sequence
import math

from .identities import BitIdentity


PRIORIFI_SOURCE_COMMIT = "6e7637b92315e5df75bb6ce258972bdf1131bc20"
PRIORIFI_SCHEDULER_VERSION = "priorifi-public-feedback/storage-identity-v1"


@dataclass(frozen=True)
class ProbeRecord:
    identity: BitIdentity
    tier: int
    sequence: int
    observed_metric: float


def significance_tiers(
    identities: Iterable[BitIdentity], ranked_words: Iterable[tuple[str, int]],
) -> tuple[tuple[BitIdentity, ...], ...]:
    """Convert a complete parameter ranking into word-relative MSB-to-LSB queues.

    The pinned source assumes homogeneous, contiguous words. Explicit identities
    let the same scheduling rule preserve the codec's actual bit positions.
    """

    bits = tuple(identities)
    words = tuple(ranked_words)
    if len(bits) != len(set(bits)) or len(words) != len(set(words)):
        raise ValueError("duplicate bit identity or ranked word")
    by_word: dict[tuple[str, int], dict[int, BitIdentity]] = {}
    for bit in bits:
        key = (bit.parameter, bit.flat_index)
        positions = by_word.setdefault(key, {})
        if bit.bit_index < 0 or bit.bit_index in positions:
            raise ValueError("invalid or duplicate encoded bit position")
        positions[bit.bit_index] = bit
    if set(words) != set(by_word):
        raise ValueError("ranked words must cover exactly the injectable population")
    widths = {len(positions) for positions in by_word.values()}
    if len(widths) != 1 or not widths:
        raise ValueError("PrioriFI homogeneous-tier adaptation requires equal word widths")
    width = widths.pop()
    if any(set(positions) != set(range(width)) for positions in by_word.values()):
        raise ValueError("each word must contain every encoded bit position exactly once")
    return tuple(tuple(by_word[word][position] for word in words)
                 for position in range(width - 1, -1, -1))


def probe_schedule(
    tiers: tuple[tuple[BitIdentity, ...], ...],
    probe: Callable[[BitIdentity], float],
    *, last_k: int,
    max_probes: int | None = None,
) -> tuple[ProbeRecord, ...]:
    """Probe one bit/tier, then select the tier with largest recent feedback.

    Reproduces the public implementation's unusual branch: use the median of
    the last ``k`` measurements when *remaining queue length* >= k, otherwise
    the mean of its retained measurements. Equal feedback resolves by tier
    order. ``max_probes`` limits experiments without changing the order prefix.
    """

    if last_k < 1:
        raise ValueError("last_k must be positive")
    queues = [list(tier) for tier in tiers]
    flattened = [bit for tier in queues for bit in tier]
    if len(flattened) != len(set(flattened)):
        raise ValueError("bit identity appears in multiple tiers")
    total = len(flattened)
    limit = total if max_probes is None else max_probes
    if not 0 <= limit <= total:
        raise ValueError("max_probes must be in [0, population size]")
    history: list[list[float]] = [[] for _ in queues]
    records: list[ProbeRecord] = []

    def one(tier_index: int) -> None:
        identity = queues[tier_index].pop(0)
        value = float(probe(identity))
        if not math.isfinite(value):
            raise ValueError(f"probe returned nonfinite feedback for {identity.key()}")
        history[tier_index].append(value)
        if len(history[tier_index]) > last_k:
            history[tier_index].pop(0)
        records.append(ProbeRecord(identity, tier_index, len(records), value))

    for index, queue in enumerate(queues):
        if queue and len(records) < limit:
            one(index)
    while len(records) < limit:
        eligible = (index for index, queue in enumerate(queues) if queue)
        choice = min(
            eligible,
            key=lambda index: (
                -(median(history[index][-last_k:]) if len(queues[index]) >= last_k
                  else mean(history[index])),
                index,
            ),
        )
        one(choice)
    return tuple(records)


def measured_bit_allocation(records: Iterable[ProbeRecord], budget: int) -> tuple[BitIdentity, ...]:
    """Separate NetSuRF adaptation: protect top observed nonnegative damages.

    This is **not** a PrioriFI output. It is undefined for unprobed bits and
    becomes an exact empirical isolated ranking when every bit is probed.
    """

    records = tuple(records)
    if not 0 <= budget <= len(records):
        raise ValueError("budget must not exceed the number of measured bits")
    if len({record.identity for record in records}) != len(records):
        raise ValueError("duplicate probe record")
    return tuple(record.identity for record in sorted(
        records, key=lambda record: (-max(0.0, record.observed_metric), record.sequence)
    )[:budget])


def candidate_tiers(
    eligible: Sequence[BitIdentity],
    candidate_pool: Sequence[BitIdentity],
    ranked_words: Sequence[tuple[str, int]],
) -> tuple[tuple[BitIdentity, ...], ...]:
    """Source-style tier queues restricted to a prespecified candidate pool.

    Unlike ``significance_tiers``, each candidate word may contribute only
    selected positions. The pool is a declared *subset* of the full eligible
    population, never implicitly the whole-network bit universe.
    """

    population, candidates, words = tuple(eligible), tuple(candidate_pool), tuple(ranked_words)
    if len(population) != len(set(population)) or len(candidates) != len(set(candidates)):
        raise ValueError("duplicate eligible or candidate bit")
    if len(words) != len(set(words)):
        raise ValueError("duplicate ranked word")
    if not set(candidates) <= set(population):
        raise ValueError("candidate pool contains an ineligible bit")
    eligible_words = {(bit.parameter, bit.flat_index) for bit in population}
    if set(words) != eligible_words:
        raise ValueError("ranked words do not cover the eligible population")
    positions = {bit.bit_index for bit in population}
    if positions != set(range(max(positions) + 1)):
        raise ValueError("eligible population has incomplete significance tiers")
    selected = set(candidates)
    return tuple(
        tuple(BitIdentity(word[0], word[1], bit_index) for word in words
              if BitIdentity(word[0], word[1], bit_index) in selected)
        for bit_index in sorted(positions, reverse=True)
    )


def cost_capped_local_allocation(
    prior_order: Sequence[BitIdentity],
    protected: Iterable[BitIdentity],
    records: Iterable[ProbeRecord],
) -> frozenset[BitIdentity]:
    """NetSuRF adaptation: measured-bit swaps over a frozen baseline mask.

    Only probed bits may change protection status. The number of protected
    probed bits is fixed to its prior value, so whole-network cardinality and
    all unprobed statuses remain unchanged. Equal feedback retains prior order;
    the probe *sequence* affects which bits are measured, not tie-breaking.
    This rule is NOT part of the published PrioriFI algorithm.
    """

    order, baseline, observed = tuple(prior_order), frozenset(protected), tuple(records)
    if len(order) != len(set(order)) or not baseline <= set(order):
        raise ValueError("invalid baseline ordering or protection mask")
    if len({row.identity for row in observed}) != len(observed):
        raise ValueError("duplicate measured bit")
    if not {row.identity for row in observed} <= set(order):
        raise ValueError("measured bit is not eligible")
    if any(not math.isfinite(row.observed_metric) for row in observed):
        raise ValueError("nonfinite measured feedback")
    rank = {bit: index for index, bit in enumerate(order)}
    # The prior order must actually describe the frozen baseline at this
    # budget. Otherwise tied observations could silently change its policy.
    if baseline != frozenset(order[:len(baseline)]):
        raise ValueError("baseline mask is not a prefix of prior_order")
    probed = {row.identity for row in observed}
    local_budget = len(probed & baseline)
    selected = sorted(observed, key=lambda row: (-max(0.0, row.observed_metric),
                                                       rank[row.identity]))[:local_budget]
    result = (baseline - probed) | {row.identity for row in selected}
    if len(result) != len(baseline) or (result - probed) != (baseline - probed):
        raise RuntimeError("cost-capped allocation violated the fixed-budget contract")
    return frozenset(result)
