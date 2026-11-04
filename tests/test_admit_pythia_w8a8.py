"""Offline Pythia admission safeguards; no model or fault execution."""

import hashlib
import json
import sqlite3
import zipfile
from pathlib import Path

import pytest

from workflows.admit_pythia_w8a8 import METHODS, _same, build, verify_handoff


def test_pythia_method_mapping_preserves_controls_and_six_rankings():
    assert set(METHODS) == {
        "unprotected", "random", "msb_first", "numerical_bit_jump",
        "global_first_order", "per_example_absolute", "per_example_positive",
        "full_protection",
    }
    assert len(set(METHODS.values())) == len(METHODS)


def test_field_number_comparison_rejects_changed_source_values():
    _same(0.125, "0.125", "coverage")
    with pytest.raises(ValueError, match="source record differs"):
        _same(0.125, "0.25", "coverage")


def test_compact_member_checksum_is_checked_before_source_access(tmp_path):
    compact = tmp_path / "compact.zip"
    with zipfile.ZipFile(compact, "w") as archive:
        archive.writestr("handoff/audit.json", b"altered")
        archive.writestr("evidence_checksums.json", json.dumps({
            "handoff/audit.json": hashlib.sha256(b"original").hexdigest()
        }))
    with pytest.raises(ValueError, match="compact member checksum differs"):
        verify_handoff(compact, tmp_path / "unused-complete.zip")


def test_admission_refuses_to_overwrite_existing_product(tmp_path):
    output = tmp_path / "finalized"
    output.mkdir()
    with pytest.raises(FileExistsError, match="write-once"):
        build(tmp_path / "unused-base", tmp_path / "unused-compact",
              tmp_path / "unused-complete", output)


def test_admitted_database_keeps_tflite_and_language_model_studies_separate():
    product = Path(__file__).resolve().parents[1] / "workspace/m12_current_comparative_w8a8_v1"
    database = product / "reliability_current_w8a8.sqlite"
    if not database.exists():
        pytest.skip("write-once local admission product is not present")
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert connection.execute("SELECT count(*) FROM primary_conditions").fetchone()[0] == 280
        assert connection.execute("SELECT count(*) FROM current_comparative_conditions").fetchone()[0] == 552
        assert connection.execute("SELECT count(*) FROM current_w8a8_language_model_conditions").fetchone()[0] == 78
        assert connection.execute("SELECT count(*) FROM w8a8_language_model_trials").fetchone()[0] == 1872
        assert connection.execute("SELECT count(DISTINCT field_sha256) FROM w8a8_language_model_trials").fetchone()[0] == 72
        assert connection.execute("SELECT count(DISTINCT method_id) FROM w8a8_language_model_masks").fetchone()[0] == 8
        assert connection.execute("SELECT count(*) FROM w8a8_language_model_trials WHERE proposed_faults != applied_faults + suppressed_faults").fetchone()[0] == 0
        assert connection.execute("SELECT count(*) FROM current_comparative_conditions WHERE study_id LIKE 'pythia%'").fetchone()[0] == 0
