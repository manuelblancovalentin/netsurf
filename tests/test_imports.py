import os
from pathlib import Path
import subprocess
import sys
import tempfile


def test_import_is_side_effect_free_outside_repository():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        home = root / "home"
        work = root / "work"
        home.mkdir()
        work.mkdir()
        environment = dict(os.environ)
        environment["HOME"] = str(home)
        subprocess.run(
            [sys.executable, "-c", "import netsurf; print(netsurf.__version__)"],
            cwd=work,
            env=environment,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert list(home.iterdir()) == []
        assert list(work.iterdir()) == []
