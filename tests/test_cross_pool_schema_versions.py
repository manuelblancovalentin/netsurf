"""Campaign schema versions: 1.1.0 belongs to the capability protocol, 1.3.0 to the KWS high-BER validation, and the
one high-BER campaign frozen while that format still carried the number 1.1.0 stays loadable."""
import pytest

from netsurf import cross_pool as C


def record(version, protocol, ber=.01, primary="restricted_selected_vs_original_s5"):
    return {"schema_version": version, "protocol": protocol, "design": {"logical_ber": ber},
            "statistical_plan": {"primary": primary}}


def test_requests_each_version_accepts_only_its_fault_rates():
    assert C.supported_ber("1.0.0", .01) and not C.supported_ber("1.0.0", .05)
    assert C.supported_ber("1.3.0", .05) and C.supported_ber("1.3.0", .075)
    assert not C.supported_ber("1.3.0", .01) and not C.supported_ber("1.3.0", .1)
    for version in ("1.1.0", "1.2.0"):                      # capability and cohort protocols: any rate in (0, 1)
        assert C.supported_ber(version, .001) and C.supported_ber(version, .05)
        assert not C.supported_ber(version, 0) and not C.supported_ber(version, 1)
    assert not C.supported_ber("9.9.9", .01)


def test_version_numbers_are_distinct_and_1_1_0_is_the_capability_protocol():
    assert C.CAPABILITY_VERSION == "1.1.0" and C.COHORT_VERSION == "1.2.0" and C.HIGH_BER_VERSION == "1.3.0"
    assert len({C.VERSION, C.CAPABILITY_VERSION, C.COHORT_VERSION, C.HIGH_BER_VERSION}) == 4


def test_saved_records_are_classified_by_protocol():
    assert C.campaign_kind(record("1.0.0", C.FROZEN_V1)) == "frozen"
    assert C.campaign_kind(record("1.1.0", C.CAPABILITY_V2)) == "capability"
    assert C.campaign_kind(record("1.1.0", C.CAPABILITY_V2, ber=.05)) == "capability"
    assert C.campaign_kind(record("1.2.0", C.COHORT_V3, ber=.001)) == "cohort"
    assert C.campaign_kind(record("1.3.0", C.FROZEN_V1, ber=.05, primary="restricted_selected_vs_exact_isolated")) == "high_ber"


def test_the_campaign_frozen_before_the_renumbering_still_loads_as_high_ber():
    legacy = record("1.1.0", C.FROZEN_V1, ber=.05, primary="restricted_selected_vs_exact_isolated")
    assert C.campaign_kind(legacy) == "high_ber"


@pytest.mark.parametrize("bad", [
    record("1.1.0", C.FROZEN_V1, ber=.01, primary="restricted_selected_vs_exact_isolated"),     # not a high-BER rate
    record("1.1.0", C.FROZEN_V1, ber=.05),                                                     # not the high-BER primary
    record("1.3.0", C.CAPABILITY_V2, ber=.05), record("1.2.0", C.CAPABILITY_V2), record("2.0.0", C.FROZEN_V1)])
def test_mismatched_version_and_protocol_are_refused(bad):
    with pytest.raises(ValueError):
        C.campaign_kind(bad)
