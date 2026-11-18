"""Pure matched-counterfactual and outside-field tests; no model inference."""
import pytest

from netsurf.identities import BitIdentity
from netsurf.transformer_interaction_diagnostic import interaction_contrast, outside_background


class TinyUniverse:
    bit_count = 128

    def bit_at(self, ordinal):
        return BitIdentity("tiny", ordinal // 8, ordinal % 8)


def test_pure_higher_order_interaction_changes_conditional_bit_effect():
    # R(A,B) has a signed additive bit effect and a separate pair interaction.
    def risk(target, background):
        return 1.0 + 0.2 * target - 0.1 * background + 0.7 * target * background
    result = interaction_contrast(risk(0, 0), risk(1, 0), risk(0, 1), risk(1, 1))
    assert result["isolated_bit_effect"] == pytest.approx(0.2)
    assert result["conditional_bit_effect"] == pytest.approx(0.9)
    assert result["interaction_difference"] == pytest.approx(0.7)
    additive = interaction_contrast(1.0, 1.2, 0.9, 1.1)
    assert additive["interaction_difference"] == pytest.approx(0.0)


def test_outside_background_is_deterministic_and_excludes_forced_target():
    universe = TinyUniverse()
    target = universe.bit_at(17)
    one = outside_background(universe, target=target, seed=7103, ber=0.3,
                             inference_count=4)
    two = outside_background(universe, target=target, seed=7103, ber=0.3,
                             inference_count=4)
    assert one.events == two.events and one.digest() == two.digest()
    assert target not in {event.bit for event in one.events}
    assert one.universe_size == 128 and one.inference_count == 4
    with pytest.raises(ValueError, match="outside"):
        outside_background(universe, target=BitIdentity("other", 0, 0),
                           seed=7103, ber=0.3, inference_count=4)
