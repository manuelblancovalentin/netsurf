"""Bounded offline validation for wide-BER source consolidation."""

from __future__ import annotations

import csv
from pathlib import Path
import shutil

import pytest

from scripts import consolidate_wide_ber_evidence as wide


SOURCE = (Path(__file__).resolve().parents[1] / "workspace/runs" /
          "pythia_wide_ber_clean_shards_v2")


@pytest.fixture
def saved_subset() -> Path:
    if not (SOURCE / "DERIVED_FINALIZED.json").exists():
        pytest.skip("gitignored real-model recovery subset is not present")
    return SOURCE


def test_recovered_subset_audit_and_pairing(saved_subset: Path) -> None:
    study = wide.audit_study(saved_subset, full_source=False)
    assert study["field_count"] == 144
    assert len(study["trials"]) == 3744
    assert len(study["conditions"]) == 208
    assert len(study["pairs"]) == 160
    assert study["per_ber"] == 18


def test_manifest_tamper_rejected(saved_subset: Path, tmp_path: Path) -> None:
    for path in saved_subset.iterdir():
        if path.is_file():
            shutil.copyfile(path, tmp_path / path.name)
    with (tmp_path / "conditions.csv").open("a") as stream:
        stream.write("tampered\n")
    with pytest.raises(ValueError, match="scientific-file digest mismatch"):
        wide.audit_study(tmp_path, full_source=False)


def test_paired_difference_reconstructed_not_trusted(
    saved_subset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("manifest.json", "DERIVED_FINALIZED.json", "analysis.json",
                 "conditions.csv", "trials.csv", "paired_comparisons.csv"):
        shutil.copyfile(saved_subset / name, tmp_path / name)
    path = tmp_path / "paired_comparisons.csv"
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[0]["mean_nll_difference_nats"] = "123.0"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    original = wide._verify_manifest
    monkeypatch.setattr(wide, "_verify_manifest", lambda root, marker, full_source:
                        original(saved_subset, marker, full_source=full_source))
    with pytest.raises(ValueError, match="paired contrast differs"):
        wide.audit_study(tmp_path, full_source=False)
