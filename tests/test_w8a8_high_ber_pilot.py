"""Pure planning tests for the bounded high-BER calibration pilot."""

import pytest

from netsurf.w8a8_high_ber_pilot import BERS, FIELDS_PER_BER, planned_fields


@pytest.mark.parametrize("bits", [344469504, 988545024, 4767875072])
def test_planned_high_ber_fields_are_distinct_and_bounded(bits):
    arms = planned_fields(bits)
    assert [arm["ber"] for arm in arms] == list(BERS)
    assert all(len(arm["seeds"]) == FIELDS_PER_BER for arm in arms)
    assert len(set(seed for arm in arms for seed in arm["seeds"])) == len(BERS) * FIELDS_PER_BER
    assert all(arm["expected_proposed"] == bits * arm["ber"] for arm in arms)


def test_unsafe_event_population_is_rejected():
    with pytest.raises(ValueError, match="event bound"):
        planned_fields(10**11)
