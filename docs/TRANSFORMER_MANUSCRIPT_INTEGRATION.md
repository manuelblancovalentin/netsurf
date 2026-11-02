# ACM TECS software manuscript: transformer integration evidence

This is a companion evidence note for the existing November 9 manuscript,
not a separate manuscript. It does not edit the paper1-engineering source or
its figures. The previously validated ResNet-8/CIFAR-10, DS-CNN/KWS and
ToyCar/AD results remain the established benchmark backbone.

| Candidate row | Storage and task | Kona status at handoff SHA-256 `a8e767e2…` | Eligible-effectiveness evidence |
|---|---|---|---|
| DistilBERT / SST-2 | source FP32, binary CE and accuracy | `PREPARED_AWAITING_APPROVAL` | None; synthetic codec demonstration only |
| Pythia-160M / WikiText-2 | proposed derived BF16, token NLL/perplexity | `PREPARED_AWAITING_APPROVAL` | None; proposed revision differs from registry |
| Qwen3-0.6B Base / WikiText-2 | source BF16, token NLL/perplexity | `PREPARED_AWAITING_APPROVAL` | None; proposed revision differs from registry |

The source-to-asset audit is an immutable read-only receipt at
`workspace/runs/transformer_handoff_audit_v1.json`; the tiny synthetic
FP32 encoded-bit demonstration is at
`workspace/runs/transformer_fixture_v1.json`. Neither is a model-level
effectiveness result. The exact storage methodology and task-risk/failure
rules are in `docs/TRANSFORMER_BENCHMARK_PROTOCOL.md`.

## Method text for integration after acceptance

For a verified transformer, fault coordinates index the actual stored FP32
or BF16 words of the declared contiguous CPU linear-weight population.
Each fault flips one encoded bit by XOR in the runtime tensor that inference
consumes. Candidate identities, checkpoint bytes, tokenizer, source indices,
token sequence, inference backend, fault events and post-trial restoration
are hashed. A protected coordinate suppresses the corresponding logical
event under a common paired fault field. SST-2 risk is bounded mean
cross-entropy with public binary labels; WikiText-2 risk is bounded
teacher-forced mean token NLL, with each target token scored once despite
overlapping context windows. Nonfinite and failed inferences receive a
prespecified 30-nat penalty and remain separately counted.

The paragraph above describes an **implemented interface and planned
protocol**, not a completed real-model experiment. Do not place any new
transformer in the paper's effectiveness tables or BER/budget figures until
the READY asset gate, exact stored-bit smoke, equal-budget protection trial,
frozen final protocol and checksum-verified result bundle all pass. When
eligible, use separate panels for accuracy, ROC-AUC and perplexity; report
runtime, bit population, BER, expected applied faults and failure fraction.

## Remaining integration blockers

The planned Torch environment and checkpoints are absent. DistilBERT
requires public labeled validation rows and an engineering cohort disjoint
from selection/final evaluation. Pythia requires the derived BF16 artifact
to be pinned and its source revision reconciled; Qwen requires revision
reconciliation and explicit tied-weight handling. Numerical backend and
autograd estimator conformance remain pending. No production launcher,
publication number or altered manuscript figure is justified yet.
