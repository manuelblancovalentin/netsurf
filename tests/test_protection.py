from netsurf.identities import BitIdentity
from netsurf.protection import ProtectionPolicy
from netsurf.ranking import MSBEstimator, Ranking


def universe():
    return tuple(BitIdentity("w", scalar, bit) for scalar in range(4) for bit in range(6))


def test_msb_first_really_selects_msb():
    identities = universe()
    ranking = MSBEstimator().estimate(identities)
    mask = ProtectionPolicy.global_ranking(ranking, 1 / 6)
    assert len(mask.protected) == 4
    assert {identity.bit_index for identity in mask.protected} == {5}


def test_position_first_equals_msb_at_full_tier():
    identities = universe()
    arbitrary = Ranking(
        "arbitrary",
        {identity: float((identity.flat_index * 17 + identity.bit_index) % 11) for identity in identities},
        {},
    )
    position_mask = ProtectionPolicy.position_first(arbitrary, 1 / 6)
    msb_mask = ProtectionPolicy.global_ranking(MSBEstimator().estimate(identities), 1 / 6)
    assert position_mask.protected == msb_mask.protected


def test_exact_cardinality_and_digest_reproducibility():
    identities = universe()
    ranking = MSBEstimator().estimate(identities)
    first = ProtectionPolicy.global_ranking(ranking, 0.25)
    second = ProtectionPolicy.global_ranking(ranking, 0.25)
    assert len(first.protected) == 6
    assert first.digest() == second.digest()
