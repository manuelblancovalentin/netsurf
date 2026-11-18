"""Bounded progress and higher-BER pilot safety checks; no model inference."""
import json
import os
from pathlib import Path

import pytest

from netsurf.multiber_campaign import PILOT_BER_GRID, arm_id
from scripts.watch_kws_higher_ber_pilot import duration, render, snapshot


def test_pilot_grid_and_duration():
    assert [arm_id(p) for p in PILOT_BER_GRID] == ["ber_0p02", "ber_0p05", "ber_0p075", "ber_0p1"]
    assert duration(None) == "Estimating..."
    assert duration(3661) == "01:01:01"


def test_receipt_progress_is_counted_only_after_persistence(tmp_path):
    root = tmp_path / "pilot"
    root.mkdir()
    arm = root / "arms/ber_0p02"
    for folder in ("attempts", "receipts"):
        (arm / folder).mkdir(parents=True)
    (root / "campaign.json").write_text(json.dumps({"arms": [{"manifest": str(arm / "campaign.json"),
        "logical_ber": .02}], "source_m96": {"candidate_pools": [{"pool_id": "pool_000"}, {"pool_id": "pool_001"}]}}))
    attempt = arm / "attempts/pool_000_selection.json"
    receipt = arm / "receipts/pool_000_selection.json"
    attempt.write_text("{}")
    os.utime(attempt, (100., 100.))
    receipt.write_text("{}")
    os.utime(receipt, (110., 110.))
    (arm / "attempts/pool_001_selection.json").write_text("{}")
    (root / "runner.stage").write_text("time|SELECTION")
    result = snapshot(root)
    assert (result["complete"], result["total"]) == (1, 4)
    assert result["rows"][0]["selection"] == 1
    assert result["rows"][0]["final"] == 0
    assert result["rows"][0]["current"] == "selection pool_001"
    assert result["remaining_seconds"] == pytest.approx(10. + 2 * 15.)
    assert not result["analysis_finalized"]
    rich = pytest.importorskip("rich")
    from rich.console import Console
    from io import StringIO
    output = StringIO()
    Console(file=output, width=76, force_terminal=False, color_system=None).print(render(result))
    assert max(map(len, output.getvalue().splitlines())) <= 76
    assert "1/4" in output.getvalue()


def test_no_campaign_is_not_reported_complete(tmp_path):
    result = snapshot(tmp_path / "absent")
    assert result["exists"] is False
