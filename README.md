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
python -m pip install --no-deps -e .
python -m tests.run
```

The validated modern stack is recorded in `dependencies/modern-ml-direct.txt`;
see `dependencies/MODERN_ML.md` for setup without `proj`, compatibility notes,
and acceptance checks. Historical compatibility remains documented separately
in `dependencies/LEGACY_MNIST.md` and is not required by the new benchmarks.

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

## Self-service modern experiments

The CLI, Python API, and notebook-style tutorial all use the same engine:

```bash
netsurf doctor
netsurf benchmarks list
netsurf benchmarks info numpy_reference
netsurf benchmarks info mnist_quantized

netsurf validate config/experiments/numpy_reference.toml --resolved
netsurf run config/experiments/numpy_reference.toml

# One-time bounded preparation; refuses to overwrite an existing checkpoint.
netsurf benchmarks prepare mnist_quantized --epochs 3 --train-samples 20000

netsurf validate config/experiments/mnist_msb.toml
netsurf run config/experiments/mnist_msb.toml
netsurf run config/experiments/mnist_ranker_comparison.toml

# M3: exact isolated-bit reference and composable ranking policies.
netsurf run config/experiments/numpy_single_bit.toml
netsurf run config/experiments/mnist_single_bit.toml
netsurf run config/experiments/mnist_msb_policies.toml

# Larger researcher-run M3 protocols (bounded, but not smoke tests).
netsurf run config/experiments/mnist_estimator_comparison.toml
netsurf run config/experiments/mnist_multi_coverage.toml

# M3.1: cancellation control, uniform sampling, independent replications,
# and a multi-coverage/multi-logical-BER protection design.
netsurf run config/experiments/numpy_qpolar_cancellation.toml
netsurf run config/experiments/mnist_estimator_uniform.toml
netsurf run config/experiments/mnist_estimator_replicated.toml
netsurf run config/experiments/mnist_multi_coverage_ber.toml

# Use the run directory printed by either command.
netsurf inspect workspace/runs/<experiment>/<run-id>
netsurf report workspace/runs/<experiment>/<run-id> --plots
```

`netsurf run` displays live progress automatically when stderr is an
interactive terminal. Progress is written to stderr so the final JSON result
on stdout remains script-friendly. Use `--progress` to force occasional plain
lines when redirected, or `--no-progress` to disable presentation entirely:

```bash
netsurf run config/experiments/mnist_policy_ber_replication_v1.toml
netsurf run config/experiments/mnist_policy_ber_replication_v1.toml --no-progress
netsurf run config/experiments/mnist_policy_ber_replication_v1.toml --progress 2>progress.log
netsurf validate config/experiments/mnist_policy_ber_wide_v1.toml
netsurf run config/experiments/mnist_policy_ber_wide_v1.toml
```

The Rich display updates in place, adapts to terminal width, and reports an
overall experiment ETA from measured operation classes. The counter advances
only after declared operations actually complete. Shared
fault schedules are counted once, while every model evaluation under a control
or protection mask is counted separately. Setup, estimator, empirical-bit,
statistical, and bundle stages are also explicit. ETA remains `Estimating...`
until enough remaining work has a measured rate and changes to `Completed in
MM:SS` at completion. Progress timing is never serialized into the scientific
payload. If Rich is unavailable, NetSuRF retains compact plain output; the
environment owner can install it with `python -m pip install 'rich>=13,<16'`.

The preparation step trains `Flatten(28x28) -> Dense(32, ReLU) -> Dense(10)`
and writes an exact `q<6,2,1>` checkpoint under
`workspace/benchmarks/mnist_quantized/`. Experiments never retrain it. Kernels
and biases are the injectable population: 25,450 scalars and 152,700 logical
bits. MSB-first, seeded random, and first-order
`abs(dL/dw * exact_bit_delta)` rankers are available independently or together.
M3 additionally separates raw estimators, composable lexicographic ordering,
and budget selection; records disjoint ranking, empirical-validation, and final
evaluation samples; and supports exact isolated-bit damage, seeded Hutchinson
diagonal curvature, per-example first order, and three explicitly versioned
historical/qPolarGrad research variants.
Schema 3.1 additionally provides uniform candidate sampling, the separately
versioned positive per-example score, independent replication controls,
multiple Hutchinson probe-count aliases, and multiple logical-BER conditions.

Schema 4 adds fixed-budget conditional protection search for the two explicit
Dense→ReLU→Dense adapters. Existing M3 rankings can initialize masks and form
rank-mixture proposal priors. Uniform, static-guided, polar-guided,
conditional-greedy, and simulated-annealing searches use exact matched
three-state swap evaluations; independent held-out fault states are shared by
all final masks. The polar score is proposal guidance only.

```bash
netsurf validate config/experiments/numpy_pgcs_exact.toml
netsurf run config/experiments/numpy_pgcs_exact.toml
netsurf run config/experiments/mnist_incremental_validation.toml
# Researcher-sized studies; inspect their budgets before launching:
netsurf run config/experiments/mnist_static_guided_search.toml
netsurf run config/experiments/mnist_polar_guided_search.toml
```

NetSuRF's lazy TensorFlow import suppresses TensorFlow 2.21's native pre-absl
CPU/GPU discovery chatter by default. Set `NETSURF_TF_IMPORT_QUIET=0` to expose
it when debugging, and optionally set `NETSURF_TF_CPP_MIN_LOG_LEVEL` to choose
TensorFlow's normal C++ log threshold. This affects console presentation only;
it does not enable a GPU, change inference, or suppress Python exceptions.

For generic schema-1 template operations retained from M1.5:

```bash
netsurf --help
netsurf init-experiment --template smoke --output workspace/scratch/my_experiment.toml
netsurf validate config/experiments/template.toml --resolved
```

Finalized bundles are immutable and contain configuration, provenance,
measurements, exact masks and schedules, checksums, a Markdown report, and
supported plots. `tutorials/read_result.py` demonstrates the public Python API;
report regeneration does not import TensorFlow or rerun inference. These are
logical weight-bit experiments with ideal event suppression, not calibrated
physical SEU models or synthesized mitigation. See `docs/REPRODUCIBILITY.md`
for bounded commands and expected digests.

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

Schema 4.1 adds independent conditional-risk evaluation for closely related
final masks. Begin with
`config/experiments/numpy_pgcs_conditional_validation.toml`; matched MNIST
smoke and research configurations reuse the existing prepared checkpoint.
