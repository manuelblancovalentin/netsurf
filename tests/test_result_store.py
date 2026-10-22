import json
from pathlib import Path
import tempfile

from netsurf.result_store import ResultStore


def test_result_store_is_write_once_and_finalized():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        store = ResultStore(root, "run-1")
        store.write_json("measurement.json", {"value": 3})
        digest = store.finalize({"scientific": "payload"})
        assert len(digest) == 64
        assert json.loads((store.path / "measurement.json").read_text()) == {"value": 3}
        try:
            store.write_text("late.txt", "forbidden")
        except RuntimeError as error:
            assert "finalized" in str(error)
        else:
            raise AssertionError("a finalized result store accepted a write")


def test_result_store_rejects_existing_run_directory():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        ResultStore(root, "same-run")
        try:
            ResultStore(root, "same-run")
        except FileExistsError:
            pass
        else:
            raise AssertionError("duplicate run directory was accepted")
