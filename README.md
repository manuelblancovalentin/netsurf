# NetSuRF

NetSuRF is a research framework for deterministic neural-network bit-fault
susceptibility experiments and resource-aware protection studies. The
canonical `netsurf` package is the reconstructed, side-effect-free v2 core.
The recovered 2025 implementation and its notebooks are preserved under
`DEPRECATED/` and are evidence, not the current scientific specification.

## Kona setup

The private `proj` project is named `netsurf` and points to this repository and
the Python 3.11 environment at
`/mnt/raid0/ml/conda_envs/manuelbv/envs/netsurf`.

```bash
proj activate netsurf
python -m pip install -e .
python -m tests.run
```

The core install requires only NumPy. The deterministic MNIST compatibility
milestone additionally needs the validated historical TensorFlow/QKeras stack;
see `dependencies/LEGACY_MNIST.md`.

## Storage configuration

Source code never requires datasets or generated runs to live in the Git tree.
The Kona defaults are:

- `NETSURF_DATASET_ROOT=/ml/datasets`
- `NETSURF_ARTIFACT_ROOT=/ml/experiments/netsurf`
- `NETSURF_LEGACY_SOURCE_ROOT=<repo>/DEPRECATED/legacy_source/netsurf`

Override these environment variables on another system. A reference layout is
provided in `config/storage.example.toml`.

## Deterministic milestone

After installing the compatibility dependencies and verifying the external
checkpoint and MNIST hashes, run the bounded experiment twice:

```bash
env HOME=/tmp/netsurf_home CUDA_VISIBLE_DEVICES=-1 \
  TF_DETERMINISTIC_OPS=1 TF_CPP_MIN_LOG_LEVEL=2 \
  netsurf milestone --repo "$PWD" --samples 256 --ranking-samples 256 \
  --trials 12 --ber 0.001 --coverage 0.10 --repeats 2
```

This is a logical persistent weight-bit experiment, not a calibrated physical
SEU model. Full prerequisites, paths, expected hash, and resource bounds are in
`docs/REPRODUCIBILITY.md`.

## Repository map

- `netsurf/`: canonical scientific core and explicit legacy adapters.
- `tests/`: deterministic regression suite.
- `config/`: source-controlled configuration examples and definitions.
- `workflows/`: workflow contracts; Wolf is restricted to supported hardware stages.
- `docs/`: scientific contracts, audit, architecture, provenance, and roadmap.
- `tutorials/`: maintained examples only.
- `workspace/`: ignored disposable local work.
- `DEPRECATED/`: recovered source, notebooks, environment snapshots, and evidence.

Generated benchmark outputs, checkpoints, rankings, and synthesis products are
not validated merely because they exist. Consult `docs/LEGACY_MAPPING.md` before
using historical material in a publication.
