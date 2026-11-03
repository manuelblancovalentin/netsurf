# Pythia BF16 storage with FP32 computation

Status on 2 November 2026: implemented and synthetically tested, **not yet a
validated real-model NetSuRF backend**. The researcher explicitly requires
holding real Pythia fault execution until Kona Assets publishes a distinct,
hash-bound READY handoff. No real Pythia inference or fault experiment was run
for this change. DistilBERT and M11 processes/worktrees remain untouched.

## Three different numerical configurations

| Stored weights | Computation | Current evidence and disposition |
|---|---|---|
| FP32, upstream step143000 | FP32 | Kona clean-reference report, perplexity 25.378383485836594 |
| BF16, named round-to-nearest-even derivative | FP32 | Kona clean-reference report, perplexity 25.42878725701041; separate finalized READY handoff pending |
| Same BF16 derivative | Native BF16 | Researcher reports approximately 76.5 perplexity and VALIDATION_FAILED; not an execution fallback |

The first two numbers describe existing asset-validation reports, **not
finalized NetSuRF protection results or new manuscript effectiveness claims**.
Reports are under `/projects/kona-assets/manuelbv/validation/netsurf/`, named
`pythia-160m-final.fp32.test.smoke.json` and
`pythia-160m-bf16-rne.fp32compute.test.smoke.json`. The latter records all
288,729 scored test tokens, context 1,024, stride 512, eager CPU execution,
32 threads, repeat bitwise equality and 359.17 seconds total runtime.

`config/pythia_mixed_pin.json` records the separately derived checkpoint hash
`45c31786f3a41eb04c589fc5895d4b04c13e6e138975fb1d279e21269379bd4c`, its
FP32 parent at `b56d9bee36300031aeea723b73c4d62ac7fa71a2`, tokenizer and dataset
hashes. This does not silently replace the older portfolio's different `main`
revision. The pin is preparation metadata, not a fabricated READY record.

## Exact execution and fault contract

The adapter directly loads the BF16 checkpoint into the validated FP32 eager
model, matching Kona's loading route. Loading the network in BF16 and converting
it later is not assumed equivalent, because initialization of auxiliary state
could differ. Every checkpoint parameter must map to the runtime with identical
shape and exact widening. The fault population contains only non-aliased,
contiguous `nn.Linear.weight` tensors, including allocated zeros. Embeddings,
normalization, biases, rotary buffers, activations and caches are excluded.

Private BF16 words are the authoritative fault state. A 16-bit XOR is applied
there, followed by exact widening `uint32 = uint16 << 16` into the inference
parameter. The FP32 computation cache has no independent fault coordinates.
BF16 sign is bit 15, exponent bits 7–14, significand bits 0–6. Signed zero,
subnormal, infinity and NaN payloads survive the widening mapping. Repeated and
multi-bit events compose by XOR; duplicate Bernoulli events are rejected.
Protected events are suppressed with proposed = applied + suppressed. Original
BF16 and FP32 bytes are verified after ordinary and exceptional exits. Autocast,
off-lattice shadow parameters, aliases and nested transactions fail explicitly.

The checkpoint header has 49 eligible linear tensors, 123,568,128 words and
1,977,090,048 BF16 bits (247,136,256 bytes); the corresponding FP32 computation
cache is 494,272,512 bytes. These are **header-verified counts pending runtime
correspondence**, not a passed real fault gate. The full checkpoint has 148
BF16 tensors. Loading and transaction checks add memory/copy costs beyond
these two numbers; measure them during the first permitted engineering run.

Teacher-forced scoring reproduces the reference operation order: FP32
`log_softmax`, gather the true next-token log probability, then retain/reduce
float64 NLL. Source-window identities preserve the 512-token context overlap;
selecting a later window does not reset it to a shorter context. Nonfinite NLL
is recorded as null with an explicit failed-token count. Raw mean NLL and
perplexity are unavailable if any token failed; no survivor-only average is
used. A separately labeled 30-nat capped endpoint is available for a prospective
bounded study. Finite clipped tokens are counted separately from failures.
That endpoint is not reference perplexity and still requires researcher approval
before freezing a campaign. Runtime/backend exceptions abort rather than being
silently converted into apparently valid model measurements.

## Required READY handoff

The proposed consumer accepts `kona-assets/netsurf-asset-readiness/1` only with
a separate target `pythia160m_bf16_storage_fp32_compute`, status `READY`, and:

```json
"execution": {
  "contract": "bf16-storage-fp32-compute/v1",
  "stored_precision": "BF16",
  "compute_precision": "FP32"
}
```

Its model repository/commit/files must match the reviewed derivative pin.
`derivation` must match the pin's parent/rounding identity. Dataset source,
revision, config, absolute path and all Parquet hashes are required. The record
must bind the final clean report by `reference_results.smoke_report` and
`smoke_report_sha256`, its exact `metrics` and all-true explicit quality `gate`.
The report must identify BF16 storage, FP32 computation, full token hash,
per-token-NLL hash, deterministic repeat and exact eager-backend provenance.
The existing interpreter/environment-lock fields must bind the installed
Python/PyTorch/Transformers versions and CPU execution. The consumer does not
invent a quality threshold or authorize READY from a report's existence.
If Kona publishes a different field layout, review that mapping explicitly;
do not edit the producer's manifest to satisfy this consumer.

## Prepared bounded experiment

`config/experiments/pythia_bf16_storage_fp32_prepare_v1.json` is an explicit
**DRAFT**, not a `netsurf run` configuration or frozen production plan.
It targets the first eight words of the first query/key/value projection,
all 16 BF16 bits each: 128 eligible bits. The deterministic selection rule is
independent of fault outcomes. All other model bits are outside this proposed
fault law, not implicitly protected. Proposed expected events are 1, 4 and 16,
so BER is 0.0078125, 0.03125 and 0.125 on that targeted support. This is not
the exposure obtained by applying those probabilities to the entire model.
For the whole header population, one expected event instead means
approximately `5.058e-10` BER. No whole-population experiment is implemented here.

Budgets are 0, 16, 32 and 128 bits; random, encoded significance and empirical
isolated ranking are proposed. Selection uses validation window 0; engineering
uses test window 0; final proposes test windows 4–5, without context overlap.
Clean reference has already exposed the full test set. Fault-outcome exposure
must be re-audited and full token identities sealed after READY; these indices
are not yet certified untouched confirmation. No samples were consumed here.

At 16 independent fields per BER, shared zero/full controls and two interior
budgets, there are 129 selection and 770 final forward windows, 899 total,
with at most 920,576 forwarded tokens, excluding engineering/conformance and
replay checks. There are 48 independent fault fields, not 48 times the token
count. Using the existing clean-report 354.24/563 seconds per window gives a
rough **9.4-minute clean-forward-only floor**, not a runtime promise: storage
copy/hash transactions, exceptional states, ranking and competing jobs add cost.
No mixed-adapter real throughput or peak RSS has been measured. Estimated
serialized token results are tens of MB, subject to actual retained fields.

Pending gates: READY; real clean/widening/XOR/restoration conformance; exposure
and token seals; approval of targeted support and capped risk; integration of
selection/final sealing for this language-model endpoint. Until these pass,
there is deliberately no production-launch command or automatic fallback.

## Reproducible validation commands

```bash
cd /projects/netsurf/manuelbv/workspace/implementation-worktrees/pythia-bf16-storage
/mnt/raid0/ml/conda_envs/manuelbv/envs/netsurf/bin/python -m pytest -q
/projects/kona-assets/manuelbv/envs/torch-2.9.1-cpu-py311/bin/python -m unittest -v tests.test_pythia_mixed_runtime
/mnt/raid0/ml/conda_envs/manuelbv/envs/netsurf/bin/python -m netsurf.pythia_preflight \
  --pin config/pythia_mixed_pin.json \
  --checkpoint /data/models/local--kona-assets--pythia-160m-bf16-rne/snapshots/45c31786f3a41eb04c589fc5895d4b04c13e6e13 \
  --plan config/experiments/pythia_bf16_storage_fp32_prepare_v1.json
/mnt/raid0/ml/conda_envs/manuelbv/envs/netsurf/bin/python -m netsurf.pythia_preflight \
  --pin config/pythia_mixed_pin.json \
  --readiness /projects/kona-assets/manuelbv/NETSURF_ASSET_READINESS.json
```

The last command currently exits 2 as intended. The preflight never initializes
a model. Pytest is absent from the pinned torch environment; standard-library
unittest executes the actual torch synthetic checks without installing anything.
Optional pytest installation, only if the researcher wants that runner there:
`/projects/kona-assets/manuelbv/envs/torch-2.9.1-cpu-py311/bin/python -m pip install 'pytest>=7,<9'`.
It is not needed for the provided unittest command and was not performed.

Tests cover every 65,536 BF16 code and all 16 single-bit XOR transitions,
real synthetic FP32 inference consumption, multiple flips, exceptions,
protected controls, exact restoration, loss reduction, READY refusal and
context-overlap rejection. Existing transformer code and frozen DistilBERT plans
are unchanged. Synthetic passing tests do not promote Pythia to READY.
