"""Read, verify, inspect, and re-report a finalized NetSuRF run.

Run this as a script or copy the individual cells into Jupyter.  It never
loads TensorFlow and never reruns fault injection.
"""

# %%
from pathlib import Path

from netsurf.api import read_result_bundle, regenerate_report

# Replace this with a finalized directory printed by ``netsurf run``.
run_directory = Path("workspace/runs/numpy_reference_single_bit/<run-id>")

# %%
result = read_result_bundle(run_directory)
print(result.scientific_payload_sha256)
print(result.summary)

# Typed exact isolated-bit measurements for schema-3 empirical studies.
if (result.path / "measurements/empirical_bits.csv").is_file():
    empirical_bits = result.empirical_bits()
    print(empirical_bits[0])

# Typed, one-row-per-policy-per-independent-trial measurements for protection studies.
if (result.path / "measurements/trials.csv").is_file():
    trials = result.trials()
    print(trials[0])

# Any saved table can also be read without framework imports.
print(result.measurement_rows("estimator_runtime.csv"))

# %%
# Presentation is regenerated under scratch; the finalized bundle is untouched.
regenerate_report(result, Path("workspace/scratch/regenerated-report"), plots=False)
