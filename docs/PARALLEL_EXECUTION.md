# Deterministic multiprocess execution

NetSuRF uses Python's `spawn` process start method for independent CPU work.
One worker remains the compatibility default and scientific reference. Worker
assignment and completion order are execution details: every unit has a stable
logical identity, every scientific seed follows the existing configuration
contract, and outcomes return to canonical order before aggregation.

## Safe boundaries

| Work | Parallel boundary | Deliberately sequential |
|---|---|---|
| Schema-3 study | Independent configured replications | Operations inside one replication |
| Schema-4 search | Independent `(budget, BER, replication)` units | Accepted-mask updates and search iterations |
| General batch | Independent TOML configurations | No nested pool inside a batch worker |
| Schema-5 pruning batch | Independent configurations and destinations | Neuron-removal stages, optimizer steps, one training run |
| Schema-1/2 experiment | Sequential reference | Model-mutating trials remain in one process |

Conditional candidates sharing a fixed background field stay together, which
preserves common random numbers and ordered reductions. M4.4 factor tables and
calibrations retain complete scientific keys, but their in-memory cache is
process-local. Mutable mask aggregates are never shared.

## Resource policy

```bash
netsurf run config/experiments/study.toml --workers 8 --threads-per-worker 2
netsurf batch a.toml b.toml --workers 2 --threads-per-worker 4
netsurf pruning batch prune-a.toml prune-b.toml --workers 2 --threads-per-worker 2
```

Both controls default to one. NetSuRF reads CPU affinity when available and
rejects a process/thread request above the visible capacity. Before spawned
workers import numerical runtimes it limits OpenMP, OpenBLAS, MKL, NumExpr, and
TensorFlow threads. TensorFlow retains the established CPU-only, deterministic,
oneDNN-disabled contract.

Batch workers own whole configurations; nested pools are not created. A pruning
batch rejects duplicate checkpoint destinations before execution. Replicated
single-run outcomes return to the parent, and only the parent finalizes their
shared immutable result bundle.

## Determinism and failures

Task identities contain logical indices only. PIDs, worker numbers, wall time,
and Python's randomized hash are never scientific seeds. The coordinator
rejects duplicate or missing identities and prevents finalization after worker
failure. Pending tasks are cancelled and live children are terminated on an
error or interruption.

The manifest records worker/thread controls, spawn method, visible CPUs, timing,
and process memory. These are excluded from `scientific_payload.json`. Bounded
NumPy tests require exact fault-field, mask, proposal, and payload-hash equality
between serial and parallel execution; multiprocessing adds no output rounding.

Startup and serialization can dominate tiny tasks. Use one worker for one small
NumPy experiment. On Kona, begin measurements at 4--8 workers and one numerical
thread per worker; increase only after measuring whole-workload time and memory.
Using all visible CPUs is not a recommended default.
