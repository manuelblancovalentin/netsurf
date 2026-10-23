"""Read-only environment diagnostics for self-service experiments."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import platform
import tempfile
from typing import Any

from . import __version__
from .benchmarks import benchmark_list
from .paths import dataset_root, workspace_root


def _version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def doctor_report() -> dict[str, Any]:
    workspace = workspace_root()
    writable = False
    warning: str | None = None
    try:
        scratch = workspace / "scratch"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix="doctor-", dir=scratch, delete=True):
            writable = True
    except OSError as error:
        warning = str(error)
    packages = {name: _version(name) for name in (
        "numpy", "tensorflow", "keras", "HGQ2", "hls4ml", "matplotlib", "scipy", "pytest", "rich"
    )}
    warnings = []
    if packages["tensorflow"] is None or packages["keras"] is None:
        warnings.append("modern Keras benchmark unavailable")
    if packages["HGQ2"] is None:
        warnings.append("HGQ2 quantization/HLS exploration unavailable")
    if packages["matplotlib"] is None:
        warnings.append("plot generation unavailable")
    if packages["rich"] is None:
        warnings.append("Rich live progress unavailable; install with: python -m pip install 'rich>=13,<16'")
    if not writable:
        warnings.append(f"workspace is not writable: {warning}")
    return {
        "python": platform.python_version(),
        "netsurf": __version__,
        "environment": {
            "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "conda_prefix": os.environ.get("CONDA_PREFIX"),
            "proj_project": os.environ.get("PROJ_PROJECT") or os.environ.get("PROJ_NAME"),
        },
        "packages": packages,
        "dataset_root": {"path": str(dataset_root()), "readable": os.access(dataset_root(), os.R_OK)},
        "workspace": {"path": str(workspace), "writable": writable},
        "benchmarks": {item.name: item.readiness()[0] for item in benchmark_list()},
        "warnings": warnings,
        "eda_tools_required": False,
    }
