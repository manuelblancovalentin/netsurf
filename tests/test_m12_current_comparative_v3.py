"""Regression checks for the write-once, offline six-benchmark database."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3

import pytest

from netsurf.manifests import sha256_file
from workflows import build_m12_current_comparative_v3 as current


ROOT = Path(__file__).resolve().parents[1]
PRODUCT = ROOT / "workspace/m12_current_comparative_v3"


def test_transformer_method_names_are_normalized_without_redefining_scores():
    assert current.ALIASES["unprotected"] == "no_protection"
    assert current.ALIASES["msb_first"] == "msb_global"
    assert current.ALIASES["per_example_absolute"] == "per_example_absolute_global"
    assert "fkeras_paper" not in current.ALIASES


@pytest.mark.skipif(not PRODUCT.is_dir(), reason="offline finalized v3 product unavailable")
def test_v3_product_preserves_historical_views_and_separates_studies():
    database = PRODUCT / "current_comparative.sqlite"
    checks = json.loads((PRODUCT / "checksums.json").read_text())["files"]
    assert all(sha256_file(PRODUCT / name) == digest for name, digest in checks.items())
    assert (PRODUCT / "FINALIZED").read_text().strip() == "database_sha256=" + checks[database.name]
    with sqlite3.connect(database) as db:
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert db.execute("SELECT count(*) FROM primary_conditions").fetchone()[0] == 280
        assert db.execute("SELECT count(*) FROM current_comparative_conditions_v2").fetchone()[0] == 552
        assert db.execute("SELECT count(*) FROM current_comparative_conditions").fetchone()[0] == 864
        assert db.execute("SELECT count(*) FROM current_trial_observations_v3").fetchone()[0] == 25776
        assert dict(db.execute("SELECT model_key,condition_count FROM current_study_registry_v3")) == {
            "resnet8": 240, "kws": 240, "toycar": 150,
            "distilbert": 78, "pythia": 78, "qwen": 78,
        }
        assert db.execute("""SELECT count(*) FROM current_method_coverage_v3 c
            JOIN current_study_registry_v3 s USING(study_id)
            WHERE s.model_key IN ('distilbert','pythia','qwen')
            AND c.method_id='fkeras_paper' AND c.status='NOT_IMPLEMENTED'""").fetchone()[0] == 3
        assert db.execute("""SELECT count(*) FROM transformer_trials_v3
            WHERE proposed_faults != applied_faults+suppressed_faults
            OR (method_id='full_protection' AND applied_faults!=0)""").fetchone()[0] == 0
        assert db.execute("""SELECT count(*) FROM (
            SELECT study_id,ber,field_index,COUNT(DISTINCT field_sha256) AS hashes
            FROM transformer_trials_v3 GROUP BY study_id,ber,field_index
            HAVING hashes != 1)""").fetchone()[0] == 0
