from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile


def test_cli_help_and_init_work_outside_repository():
    with tempfile.TemporaryDirectory() as directory:
        destination = Path(directory) / "created.toml"
        environment = dict(os.environ)
        repository = str(Path(__file__).resolve().parents[1])
        environment["PYTHONPATH"] = repository + os.pathsep + environment.get("PYTHONPATH", "")
        help_result = subprocess.run([sys.executable, "-m", "netsurf", "--help"], cwd=directory, env=environment, text=True, capture_output=True)
        assert help_result.returncode == 0
        assert "init-experiment" in help_result.stdout
        initialize = subprocess.run([sys.executable, "-m", "netsurf", "init-experiment", "--template", "smoke", "--output", str(destination)], cwd=directory, env=environment, text=True, capture_output=True)
        assert initialize.returncode == 0, initialize.stderr
        assert destination.is_file()
        repeat = subprocess.run([sys.executable, "-m", "netsurf", "init-experiment", "--output", str(destination)], cwd=directory, env=environment, text=True, capture_output=True)
        assert repeat.returncode == 2
