"""The no-filter sensitivity keeps every grid rate in the contrast family and restores the module's margin."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workflows"))
import runT_analyze as A  # noqa: E402
import runT_sensitivity_allrates as S  # noqa: E402


def test_all_rates_kept_and_margin_restored():
    rng = np.random.default_rng(0)
    acc = {}
    for r, (left, right) in {1e-4: (90.0, 89.0), 1e-2: (40.0, 30.0)}.items():     # 1e-2: both unusable (clean 91)
        acc[("tmr_abs@0.125", r)] = {(0, d): left + rng.normal(0, 0.1) for d in range(8)}
        acc[("unprotected", r)] = {(0, d): right + rng.normal(0, 0.1) for d in range(8)}
    A.MARGIN = 5.0
    signs = np.random.default_rng(A.SEED).choice((-1.0, 1.0), size=(2000, 8))
    _, _, filtered = A.analyze(acc, 91.0, signs)
    rows = S.run(acc, 91.0, 8, 5.0, 0.5)
    assert {r["ber"] for r in filtered} == {1e-4}
    assert {r["ber"] for r in rows} == {1e-4, 1e-2}
    assert A.MARGIN == 5.0
