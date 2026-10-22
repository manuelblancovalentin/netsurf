# Reproducibility

## Canonical environment and installation

The private `proj` project `netsurf` points to this repository and the dedicated
Python 3.11.17 environment at
`/mnt/raid0/ml/conda_envs/manuelbv/envs/netsurf`. Proj selects an environment;
it is not a dependency lock.

```bash
/opt/proj/bin/proj info --env netsurf
proj activate netsurf
python -m pip install -e .
python -m tests.run
```

Expected M1.5 core result: 30 passed, 0 failed. From outside the checkout,
`python -c 'import netsurf; print(netsurf.__version__)'` must print `2.0.0`.

On Kona, interactive shell startup must initialize Lmod, Conda, and the proj
shell adapter. The repository does not modify those site shell settings.

## Storage prerequisites

- Dataset: `/ml/datasets/mnist/mnist.npz`, SHA-256
  `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`.
- Checkpoint under `/ml/experiments/netsurf/legacy/benchmarks/`, SHA-256
  `ba48aee8adbe7d04092d74f41958ec99112ae3e5f88ed56552ea9d1c4dba3006`.
- Output root: `<repo>/workspace/runs/`.

The generated-work root can be changed with `NETSURF_WORKSPACE`; the normal
and documented destination remains repository-local. Historical external runs
are not moved. Exact input paths and migration evidence are in
`STORAGE_MIGRATION.md`.

## First scientific milestone

The exact historical compatibility stack is TensorFlow 2.11.1 on Python 3.10,
preserved as `netsurf-legacy`. TensorFlow 2.11 has no Python 3.11 wheel, so the
milestone must remain in that environment until a newer stack is independently
shown to reproduce the same payload. The canonical core itself is validated in
the Python 3.11 `netsurf` environment.

First validate in the canonical environment:

```bash
conda run -n netsurf python -m tests.run
conda run -n netsurf python -m netsurf validate config/experiments/template.toml --resolved
```

The core suite does not import TensorFlow. The full template is then executed
in the preserved `netsurf-legacy` environment because that is where the
validated TensorFlow 2.11 checkpoint reader exists:

```bash
conda run -n netsurf-legacy python -m netsurf run config/experiments/template.toml
conda run -n netsurf-legacy python -m netsurf run config/experiments/template.toml
```

Run `netsurf inspect <printed-run-directory>` for each output and compare the
reported scientific payload digests. This bounded command is CPU-only,
evaluates 256 held-out MNIST test images,
derives rankings from the same first 256 test images, runs 12 paired persistent-
fault trials at logical BER 0.001 and 10% equal-bit coverage. Each invocation
gets a unique run directory; no finalized bundle is overwritten.

Expected resource scale: CPU only, under one minute on the validated Kona run,
roughly 96 faulted model evaluations plus
ranking/gradient passes, and less than 10 MB of new result data. No training,
HLS, synthesis, or historical-result modification occurs.

Expected scientific payload SHA-256:
`07f4e88d7a36663fb120fafb97c28b80bab5dd3be13b91700755a4784b30bc1b`.

Each finalized run contains the exact TOML and resolved JSON configuration,
manifest and summary, scientific payload, full CSV measurements, exact masks
and schedules, checksums, report, supported plots, and `FINALIZED`. Exact
artifacts are write-once. Timestamps, source status, and run paths are excluded
from the scientific payload digest.

Reports can be regenerated without TensorFlow or inference:

```bash
conda run -n netsurf python -m netsurf report <run-directory> \
  --output workspace/scratch/reports/<run-id>
```

The canonical environment currently lacks optional `matplotlib`. Markdown
regeneration works there; `--plots` requires `matplotlib>=3.8,<4`. The user may
add it with `python -m pip install -e '.[plot]'`. The already-installed legacy
environment provides a usable Matplotlib for the bounded integration run.
