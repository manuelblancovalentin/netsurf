# Whole-eligible W8A8 transformer reliability gate

## Scientific identity and current status

Kona READY revision 2 at `<assets>/NETSURF_ASSET_READINESS.json` has SHA-256 `7128399521cef0d3e08f5d70bcb4f5b360fe2c85406c8ea8ac495e85be77b1b0`; its source commit is `1aea52b`. NetSuRF verifies the target-specific READY record, every checkpoint/tokenizer/dataset/receipt hash, pinned Python lock, the exact Kona W8A8 source bytes and the operator contract before loading. The three reviewed target-record digests are DistilBERT `e6d03c3bce030b060badb6d4e44e93008176848361e551e265f2c02b6ea06c68`, Pythia `85e7135f695a42a24669afe12483fa72ca97f4158b1e593074f470e712580723`, Qwen `c9bc0733982117e9ace2fab0403d3ae0458b7ae2c3e210f3590494187289bd88`. Later unrelated global manifest additions may not alter a frozen target record, its files or its scientific source.

These are **derived W8A8 executables**, not upstream pretrained INT8 checkpoints or integer-only networks. Every declared `QLinear.qweight` is a stored signed two's-complement INT8 code array, with FP32 scale per output row. Each Linear input is dynamically quantized per token; `torch._int_mm` consumes the live INT8 buffer and accumulates exactly in INT32 on the pinned CPU build. Embeddings, normalization, attention products, softmax, biases and dequantization remain floating point. The INT8 code bits alone are faultable. Per-channel scales, activation codes/scales, embeddings, normalization and biases remain clean **outside** the eligible population; no protection credit is assigned to them. Qwen's `lm_head.qweight` is a separately stored, separately faultable INT8 copy of the source-tied embedding matrix.

| Target | Eligible QLinear tensors | Stored codes | Eligible bits | Checkpoint INT8 storage → computation |
|---|---:|---:|---:|---|
| DistilBERT/SST-2 | 38 | 43,058,688 | 344,469,504 | INT8 weights and dynamic INT8 activations → exact INT32 Linear accumulation |
| Pythia-160M | 49 | 123,568,128 | 988,545,024 | same, with FP32 non-Linear path |
| Qwen3-0.6B Base | 197 | 595,984,384 | 4,767,875,072 | same, including separate INT8 output head |

`netsurf/w8a8_faults.py` maps a global integer ordinal to the exact `(module, C-order code index, bit index)` without materializing a bit-object list. Geometric-gap sampling gives independent Bernoulli events over the **entire** eligible bit universe and records every proposed ordinal and schedule hash. A trial applies encoded-byte XOR to the live `qweight` buffer, holds that state for the complete input cohort, suppresses mask-protected events, then reverses the XOR and verifies touched tensor hashes. Source and live state-dict aliases and QLinear module coverage are checked. A valid flip can be invisible at the final output. The [Qwen 128-token trace](<repo>/workspace/implementation-worktrees/transformer-integration/workspace/runs/qwen3_0_6b_w8a8_masked_fault_v2/receipt.json) shows a changed gate-projection output with identical downstream quantized activation codes, activation scales and final task statistics. Its prior 512-token trace is retained separately; on that longer input the same fault **did** change downstream codes and the final result. Neither trace is an effectiveness estimate.

## Whole-universe rankings and masks

Six fixed-cardinality methods share one eligible population: independent seeded PCG64 random priorities; descending encoded bit position; absolute decoded INT8 XOR jump; absolute mean first-order surrogate gradient times the exact decoded jump; mean per-example absolute surrogate damage; and mean per-example positive-part surrogate damage. For SST-2 an example is one labeled training sentence. For the LMs it is a prespecified nonoverlapping 1,023-target validation window. The differentiable surrogate replaces each integer Linear by an FP32 Linear initialized to **exactly decoded stored codes times fixed scales**. Its derivatives are **not** derivatives of dynamic W8A8 integer execution. Their signed means are retained; numerical bit-jump scores use exact two's-complement XOR and channel scales.

The exact top-budget engine streams `uint32` float-score keys in three passes, or `uint64` random priorities in five passes. Histogram refinement identifies the exact boundary score; ties break by global ordinal. Each completed mask has exactly the declared cardinality and a SHA-256 of its packed bits. A mask occupies one byte per eight eligible bits; no per-bit Python-object score list or JSON mask array is built. Analytical MSB and zero/full masks need no dense file. Four interior masks for each of the other five methods require the following bounded storage; the two FP32 surrogate moment streams are separate:

| Target | One packed mask | Twenty interior masks | Two float32 gradient-moment streams | Minimum free disk allowance including ZIP and receipts |
|---|---:|---:|---:|---:|
| DistilBERT | 43,058,688 bytes | 0.861 GB | 0.344 GB | 5 GiB |
| Pythia | 123,568,128 bytes | 2.471 GB | 0.989 GB | 10 GiB |
| Qwen | 595,984,384 bytes | 11.920 GB | 4.768 GB | 40 GiB |

A bounded 4,718,592-bit probe of the current five-pass random selector generated four exact masks in 0.402 s. A real DistilBERT-tensor probe generated four numerical-jump masks in 0.271 s. These timings are throughput probes, not whole-model guarantees. The full plans require 17 passes over every eligible bit for the five stored rankings: 5.856, 16.805 and 81.054 **billion score-key visits**, respectively. Whole-universe gradient moments and masks have **not** been generated yet; their first production execution is the selection stage.

FKeras and PrioriFI are outside these six supported W8A8 methods. Their published-form definitions and any future M12 cells require separate validated implementations; no substitute score is used here. Exact isolated-bit reference over billions of bits is computationally blocked. Historical qPolarGrad is unchanged.

## Prespecified reliability family

The initial equal-exposure family has expected **proposed** faults `λ = Np` of **4, 32 and 256**. These are pilot regimes spanning sparse to substantial eligible-code faults, not physical equivalent upset rates. The exact BERs are:

| Model | λ=4 | λ=32 | λ=256 |
|---|---:|---:|---:|
| DistilBERT | 1.16120584074e-8 | 9.28964672588e-8 | 7.43171738071e-7 |
| Pythia | 4.04635085190e-9 | 3.23708068152e-8 | 2.58966454521e-7 |
| Qwen | 8.38948156064e-10 | 6.71158524852e-9 | 5.36926819881e-8 |

Each model has 0%, 6.25%, 12.5%, 18.75%, 25% and 100% protection. A single proposed master field is shared across all 26 policy arms at each BER and field index. Expected applied faults are `λ × (1−coverage)`; for λ=256 they are 256, 240, 224, 208, 192 and 0 at those budgets. The four interior bit budgets are DistilBERT 21,529,344 / 43,058,688 / 64,588,032 / 86,117,376; Pythia 61,784,064 / 123,568,128 / 185,352,192 / 247,136,256; Qwen 297,992,192 / 595,984,384 / 893,976,576 / 1,191,968,768. Budgets are exact, not rounded approximations.

Selection uses 64 hash-ranked SST-2 **train** records or eight disjoint 1,023-target WikiText-2 **validation** windows. Final evaluation uses 512 public-label SST-2 **validation** records or 4,096 scored WikiText-2 **test** targets, with exact ordered-input and label/token digests frozen. Earlier clean and narrow fault exposure on these corpora is disclosed; this is a prospective W8A8 allocation family, not an untouched-corpus discovery claim. Failure/nonfinite outputs receive the existing 30-nat penalty and remain a separate failure fraction. SST-2 primary risk is mean cross-entropy; accuracy is secondary. LM primary risk is teacher-forced target-token mean NLL; perplexity is secondary. Overlapping contexts are not treated as independent fields.

There are **24 independent whole-model fields per BER**, **72 fields** and **1,872 policy/field inference requests per model**, plus one clean reference. DistilBERT has 958,464 scored-example requests; each LM has 7,667,712 scored-target requests. Field pairing is preserved across methods and budgets. The prespecified within-model family has 60 two-sided method-minus-random NLL comparisons (five other methods × four budgets × three BERs), with 10,000 paired sign flips and Holm adjustment across all 60. Paired field bootstrap intervals use 10,000 draws. Report favorable/adverse/tied fields, failures, accuracy/perplexity separately, medians, 10% trimmed means and leave-largest-field-out diagnostics. No BER or budget is selected post hoc for an unadjusted claim.

## Bounded conformance and resource gate

The finalized [DistilBERT](<repo>/workspace/implementation-worktrees/transformer-integration/workspace/runs/distilbert_sst2_w8a8_fault_smoke_v2/receipt.json), [Pythia](<repo>/workspace/implementation-worktrees/transformer-integration/workspace/runs/pythia160m_w8a8_fault_smoke_v2/receipt.json) and [Qwen](<repo>/workspace/implementation-worktrees/transformer-integration/workspace/runs/qwen3_0_6b_w8a8_fault_smoke_v2/receipt.json) receipts used already exposed engineering inputs and two faults in different tensors. Each verifies clean, unprotected, one-bit protected, a **whole-universe 6.25% MSB mask**, and suppression of both proposed events. The full byte hashes of touched stored tensors and the clean task output returned exactly after each trial. The unprotected fault changed task statistics in all three. No nonfinite outcome occurred in this tiny smoke; the task-risk code preserves and penalizes such outcomes when they occur. These are engineering tests, not reliability estimates.

| Target | Clean / unprotected mean loss | One-bit protected mean loss | Per-trial unprotected time | Peak RSS |
|---|---:|---:|---:|---:|
| DistilBERT, 8 sentences | 0.00132889 / 0.00154232 nats | 0.00140350 | 0.197 s | 1,815 MB |
| Pythia, 511 targets | 3.26546369 / 3.26350824 nats | 3.26544888 | 0.855 s | 1,814 MB |
| Qwen, 511 targets | 2.52644146 / 2.51931148 nats | 2.52646560 | 2.434 s | 3,658 MB |

The Pythia and Qwen two-fault effects are beneficial on these particular tokens. Signed effects are retained; they are not evidence of protection harm or benefit in the prospective family. Scaling the measured short trials to the declared cohorts suggests several foreground hours. Provision **10 h / 8 GiB RAM** for DistilBERT, **8 h / 12 GiB** for Pythia, and **24 h / 64 GiB** for Qwen, plus the free disk in the mask table. These are conservative allowances, not performance promises. All scripts use four CPU threads and refuse to start while the two M11 workers are active. Run one model at a time after researcher approval. Do not adapt the BER grid, budgets, cohorts or comparison family after observing final fields.

## Execution and central-database handoff

The three plan directories are `workspace/plans/{distilbert_sst2,pythia160m,qwen3_0_6b}_w8a8_reliability_v1`; each binds source files, the target READY record, checkpoint, tokenizer, dataset, exact input identities, budget counts, BER/seeds, statistics and resource envelope. Freezing and validating them executes **no faulted-model inference**. The foreground scripts are:

| Target | Frozen plan SHA-256 (`plan.json`) | Scientific `plan_sha256` |
|---|---|---|
| DistilBERT | `21c611dee0b68287fba1a7018973456b088080f90d9f46964e1f8f69a0816999` | `f1cb52004bd629ffdc8363106a30d535a4c670d7acd5f88bda5b98ddefaaf4bd` |
| Pythia | `434353aea4c35adfdd27f18b087a830811a7e822ddc8b59a90e909789f892979` | `26e62f1d960ea5127d7e1d7ad3149facb4df84caa90d05c8835c393ffabc54fc` |
| Qwen | `736bf8aac2bc2af84f811c92a00bfe513f33edd0487fcea45cb5617e49c435e8` | `17f65c350906b8044bb834b845eceb096a2689c355f815cb45150dfebef8d4f9` |

```sh
<repo>/workspace/implementation-worktrees/transformer-integration/scripts/run_distilbert_w8a8_once.sh
<repo>/workspace/implementation-worktrees/transformer-integration/scripts/run_pythia_w8a8_once.sh
<repo>/workspace/implementation-worktrees/transformer-integration/scripts/run_qwen_w8a8_once.sh
```

Each command validates READY and plan hashes, constructs and seals all masks from selection data, evaluates paired final fields in the foreground, aggregates offline, creates a complete ZIP with all mask bytes and field records, verifies its CRC and prints/saves the exact Mac `scp` command. A partial selection or field is retained and fails closed; it is never silently assigned a new seed. The executable uses the absolute pinned Kona interpreter with offline model loading. **No production selection, final field, ZIP or manuscript effectiveness result has been executed.**

The separate `m12-reliability-database` worktree owns the central database and its source registry. The machine-readable [handoff contract](<repo>/workspace/implementation-worktrees/transformer-integration/config/w8a8_m12_handoff_v1.json) records exact READY, plan and engineering-receipt hashes, method IDs and current stage. NetSuRF will hand over only finalized complete ZIPs plus `plan_sha256`, target-record SHA-256, quantization-manifest SHA-256, eligible-universe digest, mask-file SHA-256 values, field schedule hashes, analysis SHA-256, `FINALIZED` digest, and the `checksums.json` inventory. M12 must verify those before adding source rows; no database file or registry on that branch is modified here. The method map is `random→random_global`, `msb_first→msb_global`, `numerical_bit_jump→numerical_jump_global`, `global_first_order→gradient_global`, `per_example_absolute→per_example_absolute_global`, `per_example_positive→per_example_positive_global`, plus zero/full controls. M12 should retain the explicit decoded-code surrogate definition and mark these as a distinct W8A8 derived-executable cohort. It must not pool them with TFLite kernels, FP32/BF16 targeted panels, failed native-BF16 Pythia, or unimplemented FKeras/PrioriFI cells. M12 schema `1.2.0` has an asset-readiness overlay but no W8A8 effectiveness importer yet; that importer is M12's work, not a second NetSuRF database. Its current database rows still say the NetSuRF adapter is unvalidated; the handoff conveys the new bounded conformance evidence without rewriting that database.

Unresolved limits: whole-universe moment and mask generation has passed fixture and bounded tensor probes, but has not yet completed for any target; production wall time and disk throughput may differ. The six-method family omits exact full-universe isolated-bit, FKeras and PrioriFI rankings. The candidate universe is all verified INT8 Linear codes, not all model storage. Equal expected fault counts across architectures are not equivalent physical fault reliability.
