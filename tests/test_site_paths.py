"""Machine-local roots of the workflows: unset, they are this machine's original paths; set, they move together;
a relocated asset lock keeps its digest."""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

WORKFLOWS = Path(__file__).resolve().parents[1] / "workflows"
ORIGINAL = {  # the literal paths the workflows used before site_paths.py
    "PINNED": "/data/models/models--mlcommons--tiny/blobs/3c002613d1b2475eb51dd78dfb85a546c8ae658dee71cf6ade43b022fe205415",
    "DATASET": "/data/datasets/cifar10/cifar-10-python.tar.gz",
    "LOCK": "/projects/kona-assets/manuelbv/locks/mlperf-tiny/ic.lock.json",
    "PAPER": "/projects/netsurf/manuelbv/workspace/implementation-worktrees/paper1-tecs-revision-v1",
}
VARS = ("NETSURF_SITE_WT", "NETSURF_SITE_DATA", "NETSURF_SITE_LOCKS")


def _fresh(name, monkeypatch, env):
    for v in VARS:
        monkeypatch.delenv(v, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.syspath_prepend(str(WORKFLOWS))
    for m in ("site_paths", name):
        sys.modules.pop(m, None)
    return importlib.import_module(name)


def test_defaults_are_the_original_paths(monkeypatch):
    p0 = _fresh("runP_phase0", monkeypatch, {})
    for k, v in ORIGINAL.items():
        assert str(getattr(p0, k)) == v, k


def test_overrides_move_every_root(monkeypatch, tmp_path):
    env = {"NETSURF_SITE_WT": str(tmp_path / "wt"), "NETSURF_SITE_DATA": str(tmp_path / "data"),
           "NETSURF_SITE_LOCKS": str(tmp_path / "locks")}
    p0 = _fresh("runP_phase0", monkeypatch, env)
    assert p0.PAPER == tmp_path / "wt/paper1-tecs-revision-v1"
    assert p0.PINNED == tmp_path / ("data" + ORIGINAL["PINNED"][len("/data"):])
    assert p0.DATASET == tmp_path / "data/datasets/cifar10/cifar-10-python.tar.gz"
    assert p0.LOCK == tmp_path / "locks/mlperf-tiny/ic.lock.json"
    for v in VARS:
        monkeypatch.delenv(v, raising=False)
    for m in ("site_paths", "runP_phase0"):
        sys.modules.pop(m, None)


@pytest.mark.skipif(not Path(ORIGINAL["LOCK"]).is_file(), reason="asset lock not on this machine")
def test_relocated_lock_keeps_its_digest(monkeypatch, tmp_path):
    st = _fresh("site_transfer", monkeypatch, {})
    out = tmp_path / "locks/mlperf-tiny/ic.lock.json"
    st.relocate(Path(ORIGINAL["LOCK"]), out, tmp_path / "site", verify=False)
    moved = json.loads(out.read_text())
    original = json.loads(Path(ORIGINAL["LOCK"]).read_text())
    assert moved["digest"] == original["digest"]
    assert all(v.startswith(str(tmp_path / "site/data") + "/") for v in moved["resolved"]["file_paths"].values())
    assert {k: v for k, v in moved.items() if k != "resolved"} == {k: v for k, v in original.items() if k != "resolved"}
