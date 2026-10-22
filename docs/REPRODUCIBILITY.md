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

Expected core result: 13 passed, 0 failed. From outside the checkout,
`python -c 'import netsurf; print(netsurf.__version__)'` must print `2.0.0`.

On Kona, interactive shell startup must initialize Lmod, Conda, and the proj
shell adapter. The repository does not modify those site shell settings.

## Storage prerequisites

- Dataset: `/ml/datasets/mnist/mnist.npz`, SHA-256
  `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`.
- Checkpoint under `/ml/experiments/netsurf/legacy/benchmarks/`, SHA-256
  `ba48aee8adbe7d04092d74f41958ec99112ae3e5f88ed56552ea9d1c4dba3006`.
- Output root: `/ml/experiments/netsurf/runs/`.

These roots can be changed with `NETSURF_DATASET_ROOT` and
`NETSURF_ARTIFACT_ROOT`. Exact paths and migration evidence are in
`STORAGE_MIGRATION.md`.

## First scientific milestone

The exact historical compatibility stack is TensorFlow 2.11.1 on Python 3.10,
preserved as `netsurf-legacy`. TensorFlow 2.11 has no Python 3.11 wheel, so the
milestone must remain in that environment until a newer stack is independently
shown to reproduce the same payload. The canonical core itself is validated in
the Python 3.11 `netsurf` environment.

This bounded command is CPU-only, evaluates 256 held-out MNIST test images,
derives rankings from the same first 256 test images, runs 12 paired persistent-
fault trials at logical BER 0.001 and 10% equal-bit coverage, repeats the entire
experiment, and checks the canonical payload hash:

```bash
cd /mnt/raid0/asic/projects/NU/netsurf/manuelbv
env HOME=/tmp/netsurf_milestone_home \
  CUDA_VISIBLE_DEVICES=-1 \
  TF_DETERMINISTIC_OPS=1 \
  TF_CPP_MIN_LOG_LEVEL=2 \
  /mnt/raid0/ml/conda_envs/manuelbv/envs/netsurf-legacy/bin/python \
  -m netsurf milestone \
  --repo /mnt/raid0/asic/projects/NU/netsurf/manuelbv \
  --output-root /ml/experiments/netsurf/runs/migration-validation-2026-10-22 \
  --dataset /ml/datasets/mnist/mnist.npz \
  --samples 256 \
  --ranking-samples 256 \
  --trials 12 \
  --ber 0.001 \
  --coverage 0.10 \
  --repeats 2
```

Expected resource scale: CPU only, under one minute on the validated Kona run,
roughly 96 faulted model evaluations plus
ranking/gradient passes, and less than 10 MB of new result data. No training,
HLS, synthesis, or historical-result modification occurs.

Expected scientific payload SHA-256:
`07f4e88d7a36663fb120fafb97c28b80bab5dd3be13b91700755a4784b30bc1b`.

Each finalized run contains `manifest.json`, `scientific_payload.json`,
`trials.json`, `rankings.json`, `protection_masks.json`,
`fault_schedules.json`, `report.md`, and `FINALIZED`. Exact artifacts are
write-once and bound into the scientific payload by SHA-256. Timestamps, source
status, and run paths are excluded from the scientific payload digest.
