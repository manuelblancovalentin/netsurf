from __future__ import annotations

from pathlib import Path
import tempfile

from netsurf.api import regenerate_report
from netsurf.result_bundle import create_result_bundle, read_result_bundle
from tests.test_result_bundle_v2 import _outcome


def test_report_regenerates_from_saved_data_only():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        bundle_path = create_result_bundle(_outcome(root), repository_root=Path(__file__).resolve().parents[1], run_id="report")
        destination = root / "regenerated"
        report = regenerate_report(read_result_bundle(bundle_path), destination)
        text = report.read_text(encoding="utf-8")
        assert "Scientific payload SHA-256" in text
        assert "fault trial" in text
        assert not (destination / "scientific_payload.json").exists()
