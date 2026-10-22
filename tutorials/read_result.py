"""Read, verify, inspect, and re-report a finalized NetSuRF run.

Run this as a script or copy the individual cells into Jupyter.  It never
loads TensorFlow and never reruns fault injection.
"""

# %%
from pathlib import Path

from netsurf.api import read_result_bundle, regenerate_report

# Replace this with a finalized directory printed by ``netsurf run``.
run_directory = Path("workspace/runs/mnist_deterministic_milestone/<run-id>")

# %%
result = read_result_bundle(run_directory)
print(result.scientific_payload_sha256)
print(result.summary["protection_results"])

# Typed, one-row-per-policy-per-independent-trial measurements.
trials = result.trials()
print(trials[0])

# %%
# Presentation is regenerated under scratch; the finalized bundle is untouched.
regenerate_report(result, Path("workspace/scratch/regenerated-report"), plots=False)
