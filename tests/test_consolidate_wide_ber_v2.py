"""Bounded rendering checks for the offline comparative extension."""

import sqlite3

from scripts.consolidate_wide_ber_evidence_v2 import CONTRASTS, _plots
from scripts.consolidate_wide_ber_evidence import BER_GRID, BUDGETS, METHODS


def test_qwen_and_kws_plots_render_from_saved_scalar_tables(tmp_path):
    db = sqlite3.connect(":memory:")
    db.execute("""CREATE TABLE wide_ber_conditions(
        study_id TEXT, ber REAL, method_id TEXT, coverage REAL,
        mean_loss REAL, loss_ci_low REAL, loss_ci_high REAL)""")
    db.execute("""CREATE TABLE kws_high_ber_summary(
        ber REAL, comparison TEXT, mean_difference REAL,
        pool_ci_low REAL, pool_ci_high REAL)""")
    db.execute("""CREATE TABLE kws_high_ber_pool_contrasts(
        ber REAL, comparison TEXT, pool_id TEXT, mean_difference REAL)""")
    bits = 4_767_875_072
    for ber in BER_GRID:
        for method, coverage in (("unprotected", 0.), ("full_protection", 1.)):
            db.execute("INSERT INTO wide_ber_conditions VALUES (?,?,?,?,?,?,?)",
                       ("qwen3_0_6b_wide_ber", ber, method, coverage, 3., 2.9, 3.1))
        for method in METHODS:
            for coverage in BUDGETS:
                db.execute("INSERT INTO wide_ber_conditions VALUES (?,?,?,?,?,?,?)",
                           ("qwen3_0_6b_wide_ber", ber,
                            f"{method}_b{int(bits*coverage)}", coverage, 3., 2.9, 3.1))
    for ber in (.02, .05, .075, .1):
        for comparison in CONTRASTS:
            db.execute("INSERT INTO kws_high_ber_summary VALUES (?,?,?,?,?)",
                       (ber, comparison, -.001, -.002, 0.))
            for pool in range(12):
                db.execute("INSERT INTO kws_high_ber_pool_contrasts VALUES (?,?,?,?)",
                           (ber, comparison, f"pool_{pool:03d}", -.001))
    _plots(db, tmp_path)
    assert all((tmp_path / name).stat().st_size > 0 for name in (
        "qwen3_0_6b_loss_vs_ber.png", "qwen3_0_6b_loss_vs_budget.svg",
        "kws_high_ber_pool_effects.png", "kws_high_ber_pool_effects.svg"))
