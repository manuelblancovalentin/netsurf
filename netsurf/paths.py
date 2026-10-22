"""External storage locations used by maintained NetSuRF workflows.

The library never creates these roots merely by importing this module.  Site
defaults match the Kona workstation and can be overridden for other systems.
"""

from __future__ import annotations

import os
from pathlib import Path


def dataset_root() -> Path:
    """Return the configured shared dataset root."""

    return Path(os.environ.get("NETSURF_DATASET_ROOT", "/ml/datasets")).expanduser()


def artifact_root() -> Path:
    """Return the configured persistent experiment-artifact root."""

    return Path(
        os.environ.get("NETSURF_ARTIFACT_ROOT", "/ml/experiments/netsurf")
    ).expanduser()


def legacy_source_root() -> Path:
    """Return the preserved historical package root used by adapters only."""

    configured = os.environ.get("NETSURF_LEGACY_SOURCE_ROOT")
    if configured:
        return Path(configured).expanduser()
    return Path(__file__).resolve().parents[1] / "DEPRECATED" / "legacy_source" / "netsurf"


def qkeras_package_root() -> Path:
    """Return the preserved external QKeras package directory."""

    configured = os.environ.get("NETSURF_QKERAS_ROOT")
    if configured:
        return Path(configured).expanduser()
    return Path(__file__).resolve().parents[1] / "qkeras"
