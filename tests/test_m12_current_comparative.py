"""Offline current-study selection and plotting on available finalized evidence."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from netsurf.manifests import sha256_file
from workflows import build_m12_current_comparative as current


ROOT = Path(__file__).resolve().parents[1]
KWS_PRODUCT = ROOT / "workspace/m12_kws_seven_method_followup/derived_database_recovered_v2"
KWS_SOURCES = ROOT / "workspace/m12_kws_seven_method_followup/source_receipt.json"


@pytest.mark.skipif(not KWS_PRODUCT.is_dir(), reason="finalized KWS source product not present")
def test_current_view_selects_only_declared_study_without_changing_historical(tmp_path, monkeypatch):
    source_db = KWS_PRODUCT / "reliability_v3.sqlite"
    before = sha256_file(source_db)
    receipt = json.loads(KWS_SOURCES.read_text())
    selected = {item["id"]: item for item in receipt["studies"]}
    monkeypatch.setattr(current, "STUDIES", (
        ("kws", "DS-CNN / Speech Commands", "kws_partial_tier_fkeras_matched_v1_followup", 240),
    ))
    target = tmp_path / "current.sqlite"
    current._current_views(source_db, target, selected)
    with sqlite3.connect(target) as db:
        assert db.execute("SELECT count(*) FROM current_comparative_conditions").fetchone()[0] == 240
        assert db.execute("SELECT count(*) FROM primary_conditions").fetchone()[0] == 280
        assert db.execute("SELECT status FROM current_method_coverage WHERE method_id='priorifi_derived'").fetchone()[0] == "NOT_MEASURED"
        assert db.execute("SELECT status FROM current_method_coverage WHERE method_id='fkeras_paper'").fetchone()[0] == "FINALIZED_MEASURED"
    assert sha256_file(source_db) == before


@pytest.mark.skipif(not KWS_PRODUCT.is_dir(), reason="finalized KWS source product not present")
def test_current_plots_use_saved_measurements_only(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    receipt = json.loads(KWS_SOURCES.read_text())
    selected = {item["id"]: item for item in receipt["studies"]}
    monkeypatch.setattr(current, "STUDIES", (
        ("kws", "DS-CNN / Speech Commands", "kws_partial_tier_fkeras_matched_v1_followup", 240),
    ))
    target = tmp_path / "current.sqlite"
    current._current_views(KWS_PRODUCT / "reliability_v3.sqlite", target, selected)
    current._plots(target, tmp_path / "plots")
    assert (tmp_path / "plots/kws_loss_vs_ber.png").stat().st_size > 1000
    assert (tmp_path / "plots/kws_loss_vs_budget.png").stat().st_size > 1000
