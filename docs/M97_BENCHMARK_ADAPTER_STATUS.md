# M9.7-A: executable benchmark capabilities and remaining gates

Historical milestone record. M9.8 subsequently enables capability-bound
ResNet-8 conditional/restricted execution through schema 6.7 and campaign 1.1;
see `docs/M98_CONDITIONAL_ADAPTERS.md`. The original M96 18-target inventory and
the M97 measurements below remain unchanged. ToyCar conditional AUC support
still requires the documented file-cohort/reporting gate.

This is an additive interface over the unchanged M96 portfolio, **11 model
families and 18 precision targets**, not a declaration that all targets work.
Portfolio SHA-256: `289471536a49d87e6055d25c0deaed98b616132650a397c0c80d614d145339dc`.
`netsurf.benchmarks.capabilities` exposes version-1 records and explicit
`UnsupportedCapability` errors. It never downloads, converts precision, changes
a delegate, substitutes a dataset, or treats a checkpoint name as conformance.

## Interface and validation

```bash
cd <repo>/workspace/implementation-worktrees/m97
<netsurf-env>/bin/python -m netsurf benchmarks capabilities
<netsurf-env>/bin/python -m netsurf benchmarks capabilities kws_dscnn_int8
<assets>/.venv-mlperf/bin/python -m netsurf benchmarks probe kws_dscnn_int8 --output workspace/runs/m97_probe_kws_new
```

Replace the target with any row below. Blocked probes exit with an explicit
reason before loading an alternative or creating an output directory. A record's
`VERIFIED` status describes bounded implementation evidence, not a guarantee
that the caller's current machine has the files/backend. Executable probes check
the actual runtime, all source hashes, current lock identity, eligible population,
bit uniqueness, clean result, two same-word XOR faults (LSB and sign), exact event
suppression, full protection and state/output restoration. The forced events are
explicit semantic tests, not a falsely labelled random BER=0 draw.

Public Python entry points: `capability_records()`, `capability_record(target)`,
`require_capability(target, capability)`, `load_capability_adapter(target)`,
`probe_capability(target, new_output_path)`. Inspection imports no ML framework.
The existing CLI `benchmarks list` also exposes the streaming fixture as
`mlperf_tiny_sww_short_int8_v1`, explicitly experimental and probe-only.

Records include immutable source revision/files/checksums, dataset/preprocessing
contract, authoritative runtime, fault population/storage/packing, task metrics
and risk, surrogate objective, per-capability status, partition gates, blockers,
resource plan, validation command, and current checkpoint-file presence. A probe
adds exact tensor descriptors, quantization metadata, runtime identity, sample
identity, population hash, inputs hash, outcomes, event accounting, elapsed/load/
fault times and process peak RSS. `probe.json` is hashed by `checksums.json`;
its stable payload excludes runtime/source-location metadata. Existing bundle
engines are reused for actual supported experiments.

## Readiness by target

`V` means VERIFIED in its stated bounded scope; `B` means BLOCKED, not a silent
fallback. `S` means existing static S2–S6/isolated/protection experiments. `C`
means M96 conditional/restricted workflow. H means a validated **continuous**
loss Hessian, not merely a differentiable or STE forward model.

| Target ID | Reference / XOR / restoration | S | H | C | Gate / artifact prerequisites |
|---|---|---|---|---|---|
| `kws_dscnn_int8` | V | V | V | V | Pinned Speech Commands TFDS 0.0.3 and KWS lock; 176,128 kernel bits |
| `toycar_int8` | V | V | B | B | Pinned ToyCar lock; 2,113,536 bits; file AUC is not per-file decomposable risk |
| `resnet8_int8` | V | V | B | B | Pinned CIFAR/IC lock; 618,880 bits; schema-6 partition/metric gates remain KWS-specific |
| `vww_mobilenetv1_int8` | B | B | B | B | Pinned model exists, but local derivative-image provenance is unresolved |
| `streaming_dscnn1d_int8` | V, short fixture only | B | B | B | 368,320 bits; structural rankings only, no independent study splits or agreed scalar event risk |
| `resnet18_fp32` | B | B | B | B | Acquire pinned `microsoft/resnet-18`, licensed ImageNet-1k, explicit FP32 runtime/byte map |
| `resnet18_int8` | B | B | B | B | Above plus approved INT8 checkpoint/recipe, scales, calibration, runtime packing |
| `mobilenetv3_small_fp32` | B | B | B | B | Pinned source + ImageNet-1k absent; validate preprocessing and native dtype |
| `mobilenetv3_small_int8` | B | B | B | B | Above plus pinned INT8 conversion and stored-to-executed code equivalence |
| `vit_base_fp32` | B | B | B | B | Pinned ViT-Base + ImageNet-1k absent; tensor/attention/runtime mapping unimplemented |
| `vit_base_int8` | B | B | B | B | Above plus precision-specific calibrated recipe and packing contract |
| `distilbert_sst2_fp32` | B | B | B | B | Pinned fine-tuned checkpoint, tokenizer and SST-2 absent; explicit sequence/padding protocol |
| `distilbert_sst2_int8` | B | B | B | B | Above plus actual INT8 quantized executable and eligible storage mapping |
| `pythia160m_bf16` | B | B | B | B | Source pin is FP16, not BF16; source assets/WikiText-2 absent and conversion not approved |
| `pythia160m_int8` | B | B | B | B | No pinned INT8 derivative, tokenizer/window/NLL or packed-kernel mapping |
| `qwen3_0_6b_bf16` | B | B | B | B | BF16 source metadata exists; weights/WikiText-2 absent; tied-weight/cache/token protocol unvalidated |
| `qwen3_0_6b_int8` | B | B | B | B | No approved INT8 checkpoint/recipe/runtime layout |
| `qwen3_0_6b_int4` | B | B | B | B | No INT4 codebook, nibble packing, group/scales, zero points, padding or executable conformance |

Each record retains the original registry's **exact immutable revision and
prerequisite filenames/checksums**, not a moving model alias. `benchmarks
capabilities TARGET` prints these details and a concrete fail-closed validation
command for every row. `/data/models` currently contains only the MLCommons
Tiny snapshot. The audited dataset roots do not contain ImageNet-1k, SST-2 or
WikiText-2; Imagenette is not ImageNet-1k. No private alternate cache is searched
or treated as approved storage. Package versions for future PyTorch/Transformers
execution remain unvalidated; metadata mentioning a backend is not a package
installation recommendation or an implemented loader. No large-model Hessian
capability is inferred from architecture metadata.

FP32's logical word is IEEE binary32 (sign 31, exponent 23–30, fraction 0–22).
BF16 is sign 15, exponent 7–14, fraction 0–6, **not FP16**. Byte order, tied
storage, device copies and executed dtype must be verified after acquisition.
NaN/Inf faults require explicit failure categories; no finite penalty convention
has been approved, and silent clipping/replacement is forbidden. INT4 is not
established merely by claiming four-bit precision or loading a library: signed
code interpretation, nibble order, group metadata and any runtime repacking
must be pinned and exercised by actual XOR tests first.

## Existing model gates and ResNet lock reconciliation

KWS and ToyCar continue using their unchanged locks, exact affine INT8 kernels,
TF 2.14.1 `BUILTIN_REF`, one thread, and no delegates. KWS's ranking objective
is classification CE. ToyCar's differentiable objective is float-target
per-window reconstruction MSE; final task quality is file-level ROC-AUC.
`1−AUC` is a possible task-level risk, not an independent loss for each file.
The current schema-6 engine cannot simply substitute it for KWS CE. No such
substitution, new optimizer or cross-pool engine edit was made.

The prior ResNet adapter tests hardcoded an old canonical lock digest and old
`/ml` dataset/model paths. M96 already recorded the current lock digest
`sha256:d0daf561e681b5a31f23350192c6b5d7041b27024bf054d13a2eb793e581abd9`
(file SHA-256 `65226641813cf4192d4898293b09d3861a2ccbcae0e4aca04beb3d15bd3bbbc9`).
The model remains SHA-256
`3c002613d1b2475eb51dd78dfb85a546c8ae658dee71cf6ade43b022fe205415`.
The old test's full/subset output and CIFAR archive digests match the current
lock. Tests now resolve dataset/model paths from the verified lock and allow
only the old recorded or current audited lock fingerprints, while still
asserting the unchanged scientific content digests. The new capability loader
requires the exact **current portfolio pin**. No lock bytes, inference threshold,
checkpoint, data or backend were changed. The historical complete old lock is
not available here, so a full field-by-field provenance delta cannot be claimed.

Schema-3 static inference/gradient paths are verified for these three existing
models. M96 conditional and restricted protection remain KWS-only; extending
their partitions or task-risk definitions requires a separate reviewed change.
The scope distinction is executable (`require_capability`), not only prose.

## First additional executable: streaming DS-CNN short fixture

The available authoritative model and fixture win priority over unverified VWW
images. New adapter `netsurf/adapters/mlperf_streaming.py` uses MLCommons Tiny
commit `4addd0fa08d216e20637637874e084895f289da4`, the 10-second short WAV, its
source event windows and the exact published LFBE pipeline. This is not the
full streaming benchmark, Speech Commands KWS, or an approximately similar MFCC
pipeline. Upstream source is Apache-2.0, with attribution in the adapter and
`docs/THIRD_PARTY_M97.md`.

| Artifact | SHA-256 |
|---|---|
| `str_ww_ref_model.tflite` | `3af8550895ba7d5c584277102b5075c52dcfa63ba9d2b2240f37c4e6abd5dd2b` |
| `short_wav_easy10s.wav` | `741ab419e46087ce1b738b23ca12c0633e796234f81673c7554629c38a4b708a` |
| `sww_short_test.json` | `c1da7de188c118a2abcffedc459469ea250b55a504f63f17110827a1298a6c08` |
| INT8 input windows | `706fa76c1897de5732b8c281fa4ce55db81665a2435aa53520322666ecb8b5d8` |
| Clean exact outputs | `cedd76d2fafd974f2788ee206cf64ed8a6f756321719d671a1089b995a8da1ef` |

There are 46,040 kernel words across nine kernels, 368,320 eligible bits,
282 windows of shape (30,1,40), three output classes. Biases/quantizer state
remain excluded. The entire stream is one persistent fault window. Features
use the source float32 operation ordering, pre-emphasis, Hamming STFT,
mel/log-energy calculation and original truncating INT8 cast. The class-0
threshold is 0.95; event counting retains the source 0.5-second tolerance,
one-second suppression/debounce, and end-aligned padding. No fake per-window
accuracy, loss or training labels are generated.

The actual result is **2 true detections, 0 false positives, 1 false negative**.
An independent test extracts and runs the upstream preprocessing/event functions
and obtains exactly the same arrays and event result. The missed wakeword is
retained, not fixed by changing thresholds. The adapter's load gate checks exact
input/output digests. Same-word sign+LSB faults change the output digest; full
protection and restoration recover the exact clean digest and task metrics.
Static MSB, numerical jump and seeded random structural scores can operate on
its exact storage, but no scientifically chosen differentiable event objective
or independent partitions exist. Therefore it is a **probe-capable adapter**, not
yet accepted by static publication-study or conditional schema configurations.

Recommended next onboarding: obtain/verify the official full streaming test
inputs (including required noise sources), freeze event-level evaluation and
independent calibration/validation/test identities, then decide an explicit
surrogate objective and scalar protection risk. This reuses verified local
storage and preprocessing. VWW remains an alternative only after image
provenance is resolved. Do not begin with a massive model whose actual executable
precision and dataset are still absent.

## Bounded engineering performance

New probe receipts in this worktree, not historical publication numbers:

| Probe directory under workspace/runs | Load s | Fault evaluation s | Peak process RSS MiB | Unit |
|---|---:|---:|---:|---|
| `m97_probe_kws_v1` | 39.41 | 0.00664 | 486.6 | 2 clips |
| `m97_probe_toycar_v1` | 21.10 | 0.03835 | 1256.2 | 2 files, 392 windows |
| `m97_probe_resnet8_v1` | 7.16 | 0.01353 | 646.5 | 2 images |
| `m97_probe_streaming_v1` | 1.82 | 0.15618 | 565.3 | 282-window stream |

Load includes source verification/conformance; times were observed with other
bounded checks running and are not isolated hardware benchmarks. The streaming
v1 probe preceded the additional mandatory short-output gate, so its load time
does not include that extra clean pass. Runtime/memory scale with input count,
candidate count and stored records. At BER 0.1, expected proposed events are
17,612.8 (KWS), 211,353.6 (ToyCar), 61,888 (ResNet-8), and 36,832 (streaming)
per persistent field. The existing engine mutates the private buffer once per
field and permits several XORs within a word; it does not repeat inference per
event. Exact event JSON and per-bit estimator tables can nevertheless dominate
storage; no replay information was silently discarded or compressed away.
These are logical stress probabilities, not physical upset rates.

## Validation environment and limitations

Use the pinned MLPerf interpreter for authoritative inference and M97 FKeras;
the modern canonical environment can inspect/validate configs, run core tests
and regenerate plots without inference. The new capability command explicitly
rejects a different TF version. Optional upstream-source test dependencies are
documented in `M97_FKERAS_REPRODUCTION.md`. All 18 records are inspectable without
those dependencies. No environment or asset-lock modification was performed.
