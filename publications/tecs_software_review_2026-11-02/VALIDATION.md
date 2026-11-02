# Unified manuscript / campaign validation

Date: 2 November 2026. Branch: `paper1-expanded-benchmarks`.
Base manuscript: `589aaa67f485ae3eb36febe70132304a1b9f27ae`.
The delivery manifest binds the final committed source and archive contents.

## Manuscript and historical evidence

- 126 finalized source records / 612 consumed files reverified.
- 10,112 raw static rows recomputed into 358 summaries; 2,184 paired summaries
  checked using BER, replication, trial and schedule identities.
- Every artifact of the recovered 128-clip FKeras study was checksum-verified.
  Payload: `dd6ccdd8c000437082619ee67a79d25d067709eee210df8fc2c0daf5f712fb36`.
  Separate cohort, all six methods, all budgets/BERs; no outcome selection.
- Six saved plots plus native workflow figure; five numbered method/model/result
  tables plus analytical risk table. Thirteen bibliography entries, all cited.
- `latexmk -pdf main.tex`: complete 20-page ACM internal draft. No unresolved
  references/citations or overfull boxes. Cosmetic underfull spacing and the
  intentional ACM reference-format warning remain.
- Rendered definition/formula tables, paired results and main plot pages inspected
  for readability. Partial-tier ticks are explicit. No internal milestone/ranker
  codes appear in reader-facing sections.
- No inference is called by the manuscript evidence builder. Its plot data,
  source receipts and provenance are independently packaged.

## Executed bounded engineering and tests

- Canonical full suite: **345 passed, 12 skipped**, 83.24 s. Optional backend/slow
  gates remain. Five modern-TensorFlow Lite deprecation warnings are not hidden.
- Initial isolated-worktree run: 338 passed, 13 skipped, four failures. All four
  required the existing prepared MNIST checkpoint at a workspace-relative path.
  Read-only-use symlinks to that checkpoint and pinned upstream sources resolved
  the prerequisites. No checkpoint, test threshold or scientific logic changed.
- Pinned MLPerf suite: **39 passed**, 79.63 s. Includes exact stored XOR,
  depthwise kernels, ToyCar scores/restoration, KWS HVP/dense references, upstream
  FKeras source-form equality and streaming preprocessing/event equivalence.
  Four librosa/audioread deprecation warnings remain.
- New single/root versus replicated payload tests reject ambiguous missing
  replication identity and mismatched schedule pairing.
- Lifecycle tests cover sealed-stage reuse, corruption, incomplete stages,
  archive overwrite rejection, frozen TOML hashes/counts and all18 ledger entries.
  They do not establish that the unexecuted multi-hour campaigns completed.
- Final focused manuscript/reader/lifecycle/streaming tests: **19 passed**.
- Streaming comparison executed twice, identical payload:
  `c8ee6163da82e8cd4fee6f13e1e9ed9eb49da6daac1dd6dc490fe49618792674`.
  Two fields, 16 policy/control evaluations, same exposed short stream. Exact
  restoration and proposed/applied/suppressed accounting pass. Repeated execution
  is not another independent fault replication.
- VWW probe fails closed on image provenance: not a completed benchmark.
- All three prospective campaign preflights pass under the pinned interpreter.
  Verified-stage reuse is tested; code/config drift is rejected.
  No production campaign was started.

## Exact verification commands

Run from `/projects/netsurf/manuelbv/workspace/implementation-worktrees/paper1-expanded`.
No installation is needed:

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 \
 /mnt/raid0/ml/conda_envs/manuelbv/envs/netsurf/bin/python -m pytest -q

OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 \
 CUDA_VISIBLE_DEVICES=-1 TF_ENABLE_ONEDNN_OPTS=0 \
 /projects/kona-assets/manuelbv/.venv-mlperf/bin/python -m pytest -q \
 tests/test_m97_integration.py tests/test_m97_fkeras.py tests/test_m97_capabilities.py \
 tests/test_m93_multibench.py tests/test_mlperf_tiny_adapter.py \
 tests/test_streaming_protection_gate.py tests/test_paper1_campaign.py
```

Campaign and bounded streaming reproduction commands, costs, prerequisites and
blocked-model dependency requests are in `docs/PAPER1_EXPANDED_CAMPAIGN.md`.

## Preservation and unfulfilled acceptance criteria

Canonical v2 and the prior manuscript worktree are untouched. Researcher-owned
untracked files, M11 processes, original asset locks, models, datasets and
finalized historical runs are not modified. No installation, download, merge,
push, training or publication-scale execution was performed. Only this
worktree's own unfinalized presentation files were regenerated.
Earlier development preflight receipts were retained, not erased.

Expanded architecture acceptance remains **INCOMPLETE**: no verified transformer
assets/backend; VWW unresolved; streaming lacks a full independent task cohort.
The paper does not turn gaps into results. Checkpoint generalization, input
exposure, weak ToyCar precision and unresolved conditional-versus-isolated
benefit remain scientific limitations.
