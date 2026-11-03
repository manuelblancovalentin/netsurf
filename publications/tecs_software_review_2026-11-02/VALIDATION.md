# Unified manuscript / campaign validation

## 3 November targeted cross-precision revision

The previous 22-page revision remains the source baseline. This targeted
revision compiles to **24 pages** after three verified model/dataset citations;
the abstract, main INT8 reliability plots,
published-form FKeras series and supplementary heatmap are not regenerated.
The independent DistilBERT study and three targeted LM pilots are disclosed
in a separate cross-precision subsection. No scientific inference was run.

The read-only targeted-evidence checker passed against four finalized run
analyses and complete export ZIP hashes. It verified frozen plan/payload
identities, both DistilBERT high-BER paired contrasts, all six high-BER LM
table groups, every protected sign/exponent position, all 32 saved field
records per LM condition, and exact FP32-to-BF16 category mapping of both
Pythia precision runs' 64 nonzero-BER fault fields. Finite-token NLL was
recomputed from saved logits, not estimated from the 30-nat penalized means.
The full source and original result archives remain read-only.

Compilation completed without unresolved references or overfull boxes; page
17's table was inspected at 1600-pixel page scale and is legible. The first
full regression run had 353 passed, 12 skipped and one manuscript-only
failure because its claim-count test still expected 38 rather than 44 rows;
the assertion was updated and the rerun passed: **354 passed, 12 skipped,
5 existing TensorFlow Lite deprecation warnings**, 82.49 s. The historical validation
notes below describe earlier frozen gates, not the current evidence state.

Date: 2 November 2026. Branch: `paper1-expanded-benchmarks`.
Base manuscript: `589aaa67f485ae3eb36febe70132304a1b9f27ae`.
The delivery manifest binds the final committed source and archive contents.

## Current published-FKeras / DistilBERT revision

- Full canonical regression suite: **351 passed, 12 skipped**, 85.34 s. Five
  existing TensorFlow Lite deprecation warnings remain visible.
- Final focused manuscript, archive/statistics, evidence-reader and campaign
  contract checks: **22 passed**, 2.05 s.
- Compiled **22-page** ACM TECS internal-review PDF. No unresolved references,
  citations, overfull boxes or fatal errors. Existing optional bibliography-field
  warnings, underfull spacing and the intentional internal-review ACM metadata
  warning remain; no bibliographic field was fabricated to silence them.
- Main text uses one published-form FKeras definition/series. All original
  source evidence and broader comparison families remain unchanged.
- Original INT8 audit rerun: 126 records / 612 consumed files; 10,112 trial rows,
  358 summaries, 2,184 paired comparisons. The displayed FKeras filter changes
  neither the raw observations nor the historical scientific payload.
- DistilBERT complete/compact/checksum archives pinned and verified; all 122
  complete-archive members inventoried. Original READY snapshot, 14 source-file
  hashes, plan, selection seal and all 101 bound run records verified.
- 96 fault fields / 768 policy evaluations: schedules, masks, cardinalities,
  event accounting and zero-applied clean controls checked. All 12 original
  bootstrap/sign-flip/12-test Holm results recomputed. Additional accuracy
  intervals are descriptive, not a new testing family.
- Independent stored-logit/locked-label verification: **768/768** saved accuracy
  and capped-loss values exactly reproduced, comprising 196,608 dependent
  example outcomes. No inference was run by either manuscript reader.
- Rendered updated KWS figure, FP32 table/result pages and supplementary
  paired-effect heatmap inspected. A clipped prototype margin was corrected
  before export. Main reliability curves and scientific inputs are retained.
- No finalized common-grid KWS or matched adaptive-search result was found in
  inspected source trees. No Pythia/Qwen protection results are included.
- Historical ZIPs, finalized results, asset locks and researcher TOMLs were not
  modified. No environment change, download, production run, merge or push.

The figures/tables are manuscript derivatives. Full test output is preserved
as `validation/manuscript-revision-tests.log` in the new review archive. The
complete/compact original DistilBERT archives are included verbatim under
`distilbert-source/`, alongside their published export checksums.

## Prior manuscript and historical evidence (before this revision)

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

### Targeted correction gate (2 November 2026)

The authoritative 22-page manuscript was retained; its abstract, conclusion,
reliability-versus-BER/budget figures, published-form FKeras comparison and
supplementary heatmap were not regenerated or strengthened. The KWS v1 TOML
still has SHA-256 `8bb73b4de8503756374e5c94b86aab172fa45f3b08b53ce3b47fd9f833c0e8ac`.
The new seven-policy KWS v2 TOML validated under the pinned MLPerf interpreter,
and its bounded asset/partition preflight completed without fault trials.
It plans 128 fault fields, 3840 policy/control evaluations and 112 within-BER,
per-metric Holm comparisons. Preflight receipts and hashes are listed in
`docs/PAPER1_EXPANDED_CAMPAIGN.md`.

The hash-pinned DistilBERT safetensors header and label mapping establish that
the 32 selected classifier words all feed the NEGATIVE logit. The offline
table builder rechecked all 96 archived fields, 768 policy outcomes and 12
original paired tests, with zero new model inferences. All table effects and
intervals now use nats; no statistical comparison, 30-nat cap or original
historical payload was changed. Pythia/Qwen readiness was reconciled from the
current asset manifest and separately checksummed engineering bundles; neither
has a finalized publication protection result.

`/mnt/raid0/ml/conda_envs/manuelbv/envs/netsurf/bin/python -m pytest -q`:
**354 passed, 12 skipped** (five existing TensorFlow Lite deprecation warnings).
`latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`: **passed**, 22
pages, no overfull or undefined references. The revised table page was also
visually inspected. No historical model or fault experiment was rerun.

Canonical v2 and the prior manuscript worktree are untouched. Researcher-owned
untracked files, M11 processes, original asset locks, models, datasets and
finalized historical runs are not modified. No installation, download, merge,
push, training or publication-scale execution was performed. Only this
worktree's own unfinalized presentation files were regenerated.
Earlier development preflight receipts were retained, not erased.

Expanded architecture acceptance remains **INCOMPLETE**: targeted DistilBERT
FP32 protection is now verified, but whole-model/representative transformer
protection is not; VWW unresolved; streaming lacks a full independent task cohort.
The paper does not turn gaps into results. Checkpoint generalization, input
exposure, weak ToyCar precision and unresolved conditional-versus-isolated
benefit remain scientific limitations.
