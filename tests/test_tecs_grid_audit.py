"""Deterministic checks for the TECS-revision offline audit helpers (no inference)."""
import importlib.util
import itertools
import sys
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "publications/paper1_review/scripts/tecs_grid_audit.py"
spec = importlib.util.spec_from_file_location("tecs_grid_audit", SCRIPT)
ga = importlib.util.module_from_spec(spec)
sys.modules["tecs_grid_audit"] = ga
spec.loader.exec_module(ga)


def test_holm_matches_hand_computation_and_is_monotone():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    adj = ga.holm(p)
    # sorted: .005*4=.02, .01*3=.03, .03*2=.06, .04*1=.04 -> running max .06
    assert np.allclose(adj, [0.03, 0.06, 0.06, 0.02])
    assert np.all(adj >= p) and np.all(adj <= 1)


def test_t_interval_contains_mean_and_matches_scipy():
    d = np.array([0.1, -0.2, 0.3, 0.05, 0.0, 0.4])
    m, lo, hi = ga.t_ci(d)
    assert lo < m < hi and np.isclose(m, d.mean())
    from scipy import stats
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    assert np.isclose(hi - m, h)


def test_signflip_is_seed_deterministic_and_respects_resolution_floor():
    d = np.linspace(0.5, 1.5, 24)  # all positive: most extreme possible statistic
    p1 = ga.signflip_p(d, np.random.default_rng(1))
    p2 = ga.signflip_p(d, np.random.default_rng(1))
    assert p1 == p2
    assert p1 >= 1.0 / (ga.SIGNFLIP_DRAWS + 1)
    assert p1 < 1e-3
    assert ga.signflip_p(np.zeros(8), np.random.default_rng(1)) == 1.0


def test_method_aliases_cover_both_storage_conventions():
    for name in ("msb_global", "msb_first", "per_example_absolute_global", "per_example_absolute",
                 "fkeras_paper", "no_protection", "unprotected", "full_protection"):
        assert name in ga.METHOD_ALIASES
    assert ga.METHOD_ALIASES["per_example_positive_global"] == ga.METHOD_ALIASES["per_example_positive"] == "pos_grad"
    assert set(ga.THRESH) == {"primary", "tight", "loose"}
    assert ga.THRESH["primary"] == dict(acc=0.05, auc=0.05, nll=0.5)


def _expected(risk, protected, p, n=3):
    free = [i for i in range(n) if i not in protected]
    total = 0.0
    for state in itertools.product((0, 1), repeat=len(free)):
        z = [0] * n
        for i, v in zip(free, state):
            z[i] = v
        total += p ** sum(state) * (1 - p) ** (len(free) - sum(state)) * risk(z)
    return total


def test_manuscript_examples_and_improvement_potential_identity():
    # Eq. (3): preferred single protected bit switches at p = 1/15.
    r3 = lambda z: 1.2 * z[0] + z[1] + 0.1 * z[2] + 3 * z[1] * z[2]
    for p in (0.01, 1 / 15, 0.2):
        assert np.isclose(_expected(r3, {0}, p), 1.1 * p + 3 * p * p)
        assert np.isclose(_expected(r3, {1}, p), 1.3 * p)
    assert _expected(r3, {0}, 0.05) < _expected(r3, {1}, 0.05)
    assert _expected(r3, {0}, 0.10) > _expected(r3, {1}, 0.10)
    # Non-nested optimum at p = 0.1.
    rn = lambda z: z[0] + 1.05 * z[1] + 1.05 * z[2] + z[0] * z[1] + z[0] * z[2]
    singles = {b: _expected(rn, {b}, 0.1) for b in range(3)}
    pairs = {s: _expected(rn, set(s), 0.1) for s in itertools.combinations(range(3), 2)}
    assert min(singles, key=singles.get) == 0 and np.isclose(singles[0], 0.210) and np.isclose(singles[1], 0.215)
    assert min(pairs, key=pairs.get) == (1, 2) and np.isclose(pairs[(1, 2)], 0.100) and np.isclose(pairs[(0, 1)], 0.105)
    # Eq. (2): J(P) - J(P u {b}) = p * E_G[R(G u {b}) - R(G)] for an arbitrary risk table.
    rng = np.random.default_rng(7)
    table = rng.normal(size=8)
    risk = lambda z: table[z[0] + 2 * z[1] + 4 * z[2]]
    for p in (0.03, 0.4):
        for b in range(3):
            lhs = _expected(risk, set(), p) - _expected(risk, {b}, p)
            others = [i for i in range(3) if i != b]
            v = 0.0
            for state in itertools.product((0, 1), repeat=2):
                z = [0, 0, 0]
                for i, s in zip(others, state):
                    z[i] = s
                z1 = list(z); z1[b] = 1
                v += p ** sum(state) * (1 - p) ** (2 - sum(state)) * (risk(z1) - risk(z))
            assert np.isclose(lhs, p * v)
