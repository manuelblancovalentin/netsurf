"""The statement rules of the events-versus-background analysis, on hand-made rows."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workflows"))
from event_background_analyze import statements  # noqa: E402


def row(layer, mask, ber, events, share, ratio, ks):
    return dict(layer=layer, mask=mask, ber=ber, events_mean=events, event_share_measured=share, event_share_exact=share,
                event_over_background_median=ratio, ks_top=ks)


def test_statements_hold_on_a_clean_case():
    rows = [row(3, "unprotected", 1e-4, 0.01, 0.90, 20.0, 0.49), row(3, "unprotected", 1e-2, 1.0, 0.905, 4.0, 0.20),
            row(3, "unprotected", 1e-1, 10.0, 0.91, 1.0, 0.03)]
    s = statements(rows, figure_layer=3)
    assert s["grow_together"]["holds_everywhere"]
    assert s["one_event_stands_out"]["holds_everywhere"]
    assert s["many_events_bell_curve"]["figure_layer_holds"]


def test_statements_fail_when_they_should_and_skip_fully_protected_layers():
    rows = [row(3, "unprotected", 1e-4, 0.01, 0.80, 2.0, 0.30), row(3, "unprotected", 1e-1, 10.0, 0.95, 1.0, 0.40),
            row(0, "abs@0.125", 1e-4, 0.0, 0.0, float("nan"), float("nan"))]
    s = statements(rows, figure_layer=3)
    assert "layer0|abs@0.125" not in s["grow_together"]["per_layer_mask"]
    assert not s["grow_together"]["holds_everywhere"]                 # share moves by 0.15
    assert not s["one_event_stands_out"]["holds_everywhere"]          # median 2 < 3 at a sparse rate
    assert not s["many_events_bell_curve"]["figure_layer_holds"]      # the gap rises
