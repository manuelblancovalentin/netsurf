# Workflows

NetSuRF's ML experiment engine is ordinary Python and is not represented as a
Wolf feature. Wolf may be integrated only at a documented hardware boundary
such as HLS/RTL handoff, synthesis, or report extraction that its installed
backend actually supports.

A future workflow handoff must carry immutable identifiers for the model,
dataset split, quantization, fault model, estimator, protection policy, seeds,
logical-to-physical mapping, source revision, and input artifact hashes. No
hardware workflow is executed as part of the repository migration.
