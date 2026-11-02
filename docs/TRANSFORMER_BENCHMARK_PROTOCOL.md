# Transformer storage and benchmark integration protocol

**Status, 2 November 2026:** source and synthetic engineering stage. Kona's
`NETSURF_ASSET_READINESS.json` arrived during implementation. Its SHA-256 is
`a8e767e23450cfdaef7b09341768754a55754d084e26647dde0a8606ccc000ee`.
All three new FP32/BF16 target records say `PREPARED_AWAITING_APPROVAL`, not
`READY`; the interpreter is planned but absent. No new model is accepted, no
new effectiveness result is claimed, and no production fault campaign has
run. This work belongs to the existing software-only ACM TECS manuscript and
retains its three validated TinyML benchmarks as the evidence backbone.

## Scientific boundary and precision

The initial faultable universe consists **only** of contiguous CPU
`torch.nn.Linear.weight` parameters whose checkpoint and loaded runtime bytes
match exactly and whose observed stored dtype is FP32 or BF16. Embeddings,
layer-norm parameters, biases, activations, attention caches, optimizer state,
quantization metadata and any unsupported tensor representation are outside
this universe. The backend rejects aliases, tied parameter storage, tensor
views, extra shared storage, dtype conversion, and non-eager attention. The
loader uses local files and disables network access, training behavior,
autocast and non-deterministic algorithms. It records the source checkpoint
hashes and compares each eligible runtime tensor's bytes with its safetensors
source before faulting. Only CPU eager attention with one thread is admitted;
GPU, flash/SDPA substitution and compiled models need separate conformance.

`stored-xor/v1` numbers bits from least-significant to most-significant. FP32
has 1 sign, 8 exponent and 23 significand bits; BF16 has 1, 8 and 7. The
codec mutates the little-endian encoded byte, including signed zero,
subnormals, infinity and NaN. It does not calculate an approximate float
delta and re-encode it. INT8's existing two's-complement codec and TFLite
execution are unchanged. FP16 is unsupported here. INT8 and packed INT4
transformer variants remain blocked until the exact source codes, scale,
zero-point, grouping, physical nibble order and consumed inference bytes are
verified; a convenience conversion is not an approved asset.

A trial keeps the same altered parameter bytes throughout its evaluation
cohort. Protected coordinates suppress logical events. After each trial, the
code restores all eligible bytes, verifies their hashes and checks that
storage pointers have not changed. The one-bit smoke additionally requires
changed logits, establishing that the altered runtime tensor was consumed.
This is a logical stored-weight fault model, not a physical SEU rate or an ECC
guarantee.

## Task risks and input identities

SST-2 uses public binary labels, 0 negative and 1 positive, and reports
two-logit cross-entropy, accuracy, failure fraction and the exact logits for
each engineering example. The pinned labeled JSONL map carries original
source indices. `engineering`, `calibration`, `selection`, and `evaluation`
indices must all be explicit, unique and pairwise disjoint. The smoke uses
the first two prespecified engineering indices only. No unlabeled public test
rows are accepted as evaluation labels.

Pythia and Qwen use teacher-forced WikiText-2 raw text only after its exact
local bytes and token-ID sequence are pinned. The plan must specify tokenizer,
`bos_mode`, `eos_mode`, context length and stride. Windows overlap for context,
but each target token after the first is scored exactly once. The primary
risk is mean target-token negative log-likelihood; perplexity is its
exponential. Failed token fraction is separate. Overlapping windows do not
become independent statistical replicates. For each token, the evidence
retains the target logit and vocabulary log-sum-exp, the sufficient statistics
for exact NLL, rather than an enormous full-vocabulary logit vector. No
stochastic generation is used.

For every model, a nonfinite logit row or runtime inference failure receives
the prespecified **30-nat** loss penalty for its sample/token; finite NLLs
are capped at 30 nats as well. Failures have zero classification credit and
are counted separately. This bounded risk remains defined under NaN,
infinity and exceptions; the cap is a scientific choice to be reviewed before
a final campaign, never chosen from observed final effects.

## Asset gate

`NETSURF_ASSET_READINESS.json` is the sole real-execution handoff. The
consumer reads Kona's actual `kona-assets/netsurf-asset-readiness/1` schema
and rejects `PREPARED_AWAITING_APPROVAL`, `MISSING_ASSET`, `LICENSE_BLOCKED`
and `BACKEND_BLOCKED` records before opening model data. The NetSuRF-specific
executable mapping `netsurf-transformer-readiness/v1` requires one explicit
`READY` record per target with registry source/revision, exact precision,
model and tokenizer file inventories and SHA-256 values, dataset file and
hash, absolute interpreter and hash, Python/PyTorch/Transformers versions,
CPU/eager/one-thread backend, public class mapping or exact token-sequence
hash, and disjoint partition identities. It compares the registry's pinned
checkpoint hashes. Unsupported handoff schemas fail pending a reviewed
mapping; mere file presence never establishes readiness. The source team
does not edit asset manifests or locks.

A read-only 18-target handshake receipt is generated at
`workspace/runs/transformer_handoff_audit_v1.json` by
`python -m netsurf.transformer_cli audit-readiness`. It records the exact
Kona manifest hash, portfolio hash, target statuses, revision matches and
interpreter existence. The present receipt finds only the three previously
validated TinyML targets `READY`; all transformer interpreters are absent.

The present portfolio registry pins Pythia `50f5173d…` and Qwen
`c1899de2…`; Kona's prepared handoff specifies Pythia step143000
`b56d9bee…` and Qwen `da87bfb6…`. Those revision changes require an
explicit registry/protocol review before a READY record can execute. Pythia's
BF16 target also requires a verified derived artifact from its source FP32
weights. Qwen tied weights may be rejected by the first alias-safe backend.
Those failures are explicit, not silently coerced.

## Fault law, feasibility and baselines

Independent Bernoulli bit faults are drawn using geometric success gaps over
the ordered eligible universe. This is distributionally exact without
materializing every bit. The saved schedule includes seed, event identities,
law, proposed/applied/suppressed counts and event checksum. A prespecified
event cap fails the run; it never truncates a schedule. With `N` eligible bits,
`B` protected bits and BER `p`, expected proposed events are `Np`, applied
events are `(N-B)p`, and suppressed events are `Bp`. These expected counts
must accompany each result. Equal BER or equal expected event count does not
mean equal physical reliability across storage technologies.

The implemented bounded ranking primitives are seeded random, encoded MSB
order, finite numerical bit displacement, global mean gradient, per-example
absolute gradient, per-example positive gradient, and exact isolated-bit
task-risk difference. Signed harmful and beneficial isolated effects remain
signed. The numerical displacement and first-order rules reject nonfinite
transitions rather than assigning favorable scores. The gradient score
functions are implemented; autograd extraction and full-scale validation are
pending. FKeras/Hessian is blocked until a specific numerical HVP passes
conformance; autograd availability alone is insufficient. Historical
qPolarGrad is unchanged and is not transplanted to transformers.

## Execution stages and resource gate

The only executable real-model stage now is a **bounded DistilBERT
engineering smoke**. It loads once and performs seven two-example inference
passes: clean twice, one-bit, two-bit, full suppression of the same two
events, one-bit equal-budget protection against that two-bit field, and a
post-restoration clean check. The upper bound is **14 clip inferences**, one
checkpoint load, and two changed bit coordinates. The output is write-once,
has exact events, byte hashes, risk values, a finalized marker and checksums.
Its two events are explicitly tagged `forced_engineering`; they are not
represented as draws from a BER-zero Bernoulli law. It is not a publication
effectiveness comparison. Kona's prepared resource estimates are **272 MB
disk / 1.2 GB RAM** for DistilBERT, **1,351 MB disk / 2.5 GB RAM** for
Pythia, and **1,204 MB disk / 4 GB RAM** for Qwen, all CPU and zero VRAM.
These are asset-side estimates, not measured NetSuRF inference peaks; the
model load, checkpoint comparison and full-logit attention output add cost.

The next production plan must be frozen **after** the READY record and exact
eligible `N` are known, but **before** any final effects are inspected. A
candidate initial grid is `p ∈ {0, 1/N, 4/N}` with budgets `B ∈
{0, floor(0.01N), floor(0.05N), N}`. Its exact expected applied fields are
`(N-B)p`; this avoids blindly reusing INT8 BER levels in a much larger
floating-point universe. The final choice of clip/token counts, independent
field count, baseline availability, random seeds, candidate budget and all
request costs must be made in a hashed plan. Freeze calibration and every
selection artifact before final evaluation. Share fault fields across
protection methods; independent fields, not clips or overlapping LM windows,
form stochastic replicates. Cohort sampling uncertainty and field
uncertainty must be reported separately. Include clean, unprotected and full
suppression controls. Report paired loss contrasts, accuracy or perplexity
on separate axes, failure fractions and resource costs. Do not increase only
favorable conditions after inspecting final effects.

No production-run TOML or script is marked frozen while readiness and source
conformance are absent. The 18-target status ledger is
`config/transformer_target_status.json`; it does not change the historical
portfolio registry or the manuscript's existing evidence.

## Foreground commands and exports

First obtain Kona's explicit READY manifest. The following single foreground
command verifies it, selects its **verified absolute** interpreter, runs the
bounded DistilBERT smoke, verifies/finalizes evidence, generates compact and
complete ZIPs, and prints/saves Mac `scp` commands:

```bash
/projects/netsurf/manuelbv/workspace/implementation-worktrees/transformer-integration/scripts/run_distilbert_engineering_once.sh \
  /absolute/path/NETSURF_ASSET_READINESS.json \
  /projects/netsurf/manuelbv/workspace/implementation-worktrees/transformer-integration/workspace/runs/distilbert_sst2_smoke_v1
```

The export produces `compact.zip`, `complete.zip`, `checksums.json` and
`mac_scp_commands.txt` beside the smoke receipt. An existing finalized
receipt is checksum-verified and reused; a corrupt or interrupted receipt
fails closed. Offline aggregation:

```bash
/mnt/raid0/ml/conda_envs/manuelbv/envs/netsurf/bin/python -m netsurf.transformer_cli aggregate-smoke \
  /projects/netsurf/manuelbv/workspace/implementation-worktrees/transformer-integration/workspace/runs/distilbert_sst2_smoke_v1
```

No one-shot **production** launcher is issued for an unaccepted model. It
must follow a frozen source/asset/partition/statistical plan and independent
selection seal; issuing a runnable publication launcher now would bypass
those gates.

## Manuscript integration boundary

The current manuscript benchmark and methodology tables may cite the three
already validated TinyML rows as before. DistilBERT/Pythia/Qwen rows are
`IMPLEMENTING`, with no effectiveness cells or BER/budget curves. Once a
new model passes exact stored-bit conformance and a reproducible task-level
protection campaign, add a separately labeled model row, precision/storage
methodology, BER and budget figure panels, computational cost and limitations.
Accuracy, ROC-AUC and perplexity must use separate axes. This protocol is
manuscript-ready methods context, not a new manuscript or effectiveness claim.
