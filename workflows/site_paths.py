"""Machine-local roots of the Run P/Q/R workflows, so the same code runs on another machine (for example Quest).

Three roots, each overridable by one environment variable; unset, they are this machine's paths, so behaviour here
is unchanged:

    NETSURF_SITE_WT     the folder that holds the worktrees     (default /projects/netsurf/manuelbv/workspace/implementation-worktrees)
    NETSURF_SITE_DATA   the data root (models/, datasets/)      (default /data)
    NETSURF_SITE_LOCKS  the asset-lock root (mlperf-tiny/*.json) (default /projects/kona-assets/manuelbv/locks)

On another machine the asset locks must be relocated too (their ``resolved`` paths are machine-local and excluded from
the lock digest); see ``site_relocate_locks.py``.
"""
from __future__ import annotations

import os
from pathlib import Path

DEFAULTS = {
    "NETSURF_SITE_WT": "/projects/netsurf/manuelbv/workspace/implementation-worktrees",
    "NETSURF_SITE_DATA": "/data",
    "NETSURF_SITE_LOCKS": "/projects/kona-assets/manuelbv/locks",
}


def root(name: str) -> Path:
    value = os.environ.get(name) or DEFAULTS[name]
    return Path(value)


WT = root("NETSURF_SITE_WT")
DATA = root("NETSURF_SITE_DATA")
LOCKS = root("NETSURF_SITE_LOCKS")
