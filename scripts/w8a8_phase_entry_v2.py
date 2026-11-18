"""Set DistilBERT CPU scheduling before importing the frozen campaign engine."""

from __future__ import annotations

import json
import os
import sys

import torch


torch.set_num_interop_threads(1)
from netsurf.w8a8_campaign import main  # noqa: E402


if __name__ == "__main__":
    print(json.dumps({"phase_entry": "w8a8-cpu-schedule/v2",
                      "affinity": sorted(os.sched_getaffinity(0)),
                      "torch_interop_threads": torch.get_num_interop_threads(),
                      "torch_intraop_threads_at_entry": torch.get_num_threads()}), flush=True)
    main(sys.argv[1:])
