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

Input datasets remain shared, while all new generated runs are local to the
ignored repository workspace. The Kona defaults are:

- `NETSURF_DATASET_ROOT=/ml/datasets`
- `NETSURF_WORKSPACE=<repo>/workspace`
- `NETSURF_LEGACY_SOURCE_ROOT=<repo>/DEPRECATED/legacy_source/netsurf`
- `NETSURF_QKERAS_ROOT=<repo>/DEPRECATED/external_links/qkeras`
- `NETSURF_LEGACY_WORKSPACE=<repo>/workspace/legacy`

Override these environment variables on another system. A reference layout is
provided in `config/storage.example.toml`.

## Self-service experiments

The CLI, Python API, and notebook-style tutorial all use the same engine:

```bash
netsurf --help
netsurf init-experiment --template smoke --output workspace/scratch/my_experiment.toml
netsurf validate config/experiments/template.toml --resolved
netsurf run config/experiments/template.toml
netsurf inspect workspace/runs/<experiment>/<run-id>
netsurf report workspace/runs/<experiment>/<run-id>
```

Finalized bundles are immutable and contain configuration, provenance,
measurements, exact masks and schedules, checksums, a Markdown report, and
supported plots. `tutorials/read_result.py` demonstrates the public Python API.
The maintained experiment is a logical persistent weight-bit experiment, not a
calibrated physical SEU model. See `docs/REPRODUCIBILITY.md` for its bounds and
expected digest.

## Repository map

- `netsurf/`: canonical scientific core and explicit legacy adapters.
- `tests/`: deterministic regression suite.
- `config/`: source-controlled configuration examples and definitions.
- `workflows/`: workflow contracts; Wolf is restricted to supported hardware stages.
- `docs/`: scientific contracts, audit, architecture, provenance, and roadmap.
- `tutorials/`: maintained examples only.
- `workspace/`: ignored runs, scratch data, caches, and legacy side effects.
- `DEPRECATED/`: recovered source, notebooks, environment snapshots, and evidence.

Generated benchmark outputs, checkpoints, rankings, and synthesis products are
not validated merely because they exist. Consult `docs/LEGACY_MAPPING.md` before
using historical material in a publication.
