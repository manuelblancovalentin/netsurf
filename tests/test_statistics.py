from netsurf.statistics import summarize


def test_summary_counts_independent_trials_not_inference_samples():
    estimate = summarize([0.75, 0.80, 0.85])
    assert estimate.count == 3
    assert abs(estimate.mean - 0.8) < 1e-12
    assert estimate.standard_deviation > 0
    assert estimate.standard_error > 0


def test_single_trial_interval_and_physical_bounds():
    singleton = summarize([0.7], bounds=(0.0, 1.0))
    assert singleton.confidence_interval == (0.7, 0.7)
    bounded = summarize([0.99, 1.0, 1.0], bounds=(0.0, 1.0))
    assert bounded.confidence_interval[1] == 1.0
