# MNIST compatibility dependency boundary

The deterministic milestone loads one historical Keras checkpoint through an
explicit adapter. That path needs TensorFlow, QKeras, tensorflow-model-
optimization, and the historical package's auxiliary modules. These are not
core NetSuRF dependencies and are not yet declared as a pip extra because the
external `qkeras` checkout and several historical modules do not yet have
verified immutable provenance for a fresh Python 3.11 installation.

The previously working environment snapshot is in `DEPRECATED/env/`. Validate
the new environment against that evidence without blindly replaying the entire
snapshot. Core tests remain independent of this stack. Historical Nodus import
state is routed to `workspace/legacy/` by default and can be relocated with
`NETSURF_LEGACY_WORKSPACE`.
