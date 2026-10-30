from netsurf.restricted_subcube import select_restricted_subcube_mask


def test_restricted_subcube_selects_only_on_calibration_and_reports_held_out_separately():
    def calibration(faults):
        return 10.0 * (0 in faults) + 1.0 * (1 in faults) + 0.1 * (2 in faults)

    def held_out(faults):
        return 1.0 * (0 in faults) + 9.0 * (1 in faults) + 0.1 * (2 in faults)

    result = select_restricted_subcube_mask(
        [calibration], [held_out], candidate_count=3, budget=1, probability=0.2
    )
    assert result.selected_mask == (0,)
    assert result.held_out_risks[(1,)] < result.held_out_risks[result.selected_mask]


def test_restricted_subcube_is_exact_and_stably_tie_broken():
    result = select_restricted_subcube_mask(
        [lambda faults: float(len(faults))], [lambda faults: float(len(faults))],
        candidate_count=3, budget=1, probability=0.25,
    )
    assert result.selected_mask == (0,)
    assert result.calibration_risks[(0,)] == 0.5
    assert result.held_out_risks[(0,)] == 0.5
