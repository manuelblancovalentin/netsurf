# Compute environment for the Run R experiments (recorded 12 November 2026)

For the paper's cost table. Host name, user names and network details are deliberately left out.

## Machine

One shared CPU server; no GPU was used for any Run R stage.

| Item | Value |
|---|---|
| CPU | 2 x Intel Xeon Gold 6248 (Cascade Lake), 2.5 GHz base, 3.9 GHz max, AVX-512 |
| Cores | 40 physical (20 per socket), 80 hardware threads (2 per core), 2 NUMA nodes |
| Cache | L2 40 MiB in total (1 MiB per core), L3 55 MiB in total (27.5 MiB per socket) |
| Memory | 472 GiB |
| OS | Ubuntu 24.04.5 LTS, Linux 6.8 |

## Software

| Use | Interpreter and libraries |
|---|---|
| Fault injection, masks, searches (every number from a faulty model) | Python 3.11.18, TensorFlow 2.14.1 with the TFLite builtin reference kernels (`BUILTIN_REF`, one thread), NumPy 1.26.4 |
| Analysis, statistics, figures | Python 3.11.17, NumPy 2.4.6, SciPy 1.17.1 |

## How jobs were run

- Each job is one single-threaded process, except the Stage 1 mask rebuild (2 threads). Jobs were run up to 76 at a
  time with `xargs -P` (`workflows/runR_launch.sh`).
- 76 processes occupy hardware threads on 40 physical cores, so two processes usually share one core. Measured
  effect: one Stage 1 evaluation took 6.2 s in a single-job smoke and 12.2 s with 76 jobs running. Throughput is
  about the same either way (about 6 evaluations per second), so the per-job times in the ledger are times on a shared
  core, and CPU hours are counted as job seconds of one hardware thread.
- Per-phase jobs, CPU hours, time per mask and per evaluation, and wall time: `workflows/runR_timing.py` on the Run R
  branch, which reads the seconds that every job records in its output.
