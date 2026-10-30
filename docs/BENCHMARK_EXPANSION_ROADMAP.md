# NetSuRF benchmark expansion: audited portfolio, 2026-10-30

This is an engineering roadmap, not a claim that every target is implemented.
The versioned machine-readable companion is
`config/benchmark_expansion_registry.json`: 11 model families, 18 precision
targets, exact registered revisions/files/hashes, local-availability flags,
source/lock identities, task metrics, limitations and resource plans.
It is **not** the executable benchmark registry; it does not make new adapters
discoverable or silently approve a model conversion.

## Audit basis and status vocabulary

Inspected NetSuRF adapters/contracts, shared `/data/models` and `/data/datasets`,
and `/projects/kona-assets/manuelbv/src/kona_assets/registry/{models,datasets,benchmarks}`
plus `locks/mlperf-tiny/`. No downloads, training, package installation, precision
conversion or new model inference were performed for this roadmap. Private
user caches were not searched. Registry pins for unacquired models are evidence
of an acquisition specification, not proof that their tensor headers were read.
Hugging Face CLI metadata was read at pinned revisions for the Pythia/Qwen dtype
check; model cards and official MLPerf specifications supplied source context.

- **Ready**: pinned assets and a verified existing static fault-inference backend.
- **Adapter required**: usable bounded source assets exist; exact adapter/task
  conformance remains incomplete. Full research data can still be missing.
- **Asset acquisition required**: essential independently verified assets missing.
- **Research-stage**: new quantization or numerical fault contracts also required.

No non-TinyML target is currently ready. Even “Ready” does not mean the current
schema-6 restricted conditional protocol supports that benchmark.

## Complete portfolio

| Priority / target | Precision targets and status | Available / missing | Metric; faultable kernel bits |
|---|---|---|---|
| 1 ResNet-8 / CIFAR-10 | INT8: **Ready** | Official pinned TFLite, CIFAR-10, lock, exact adapter/companion | Accuracy and CE; 618,880 verified |
| 1 ToyCar autoencoder | INT8: **Ready** | Pinned TFLite, ToyADMOS/ToyCar files, lock, exact adapter/companion | File ROC-AUC, separate float-domain reconstruction MSE; 2,113,536 verified |
| 1 DS-CNN / Speech Commands | INT8: **Ready** | Pinned TFLite, 4,890 test clips, lock, exact adapter/companion | Accuracy and CE; 176,128 verified |
| 2 MobileNetV1-0.25 / VWW | INT8: **Asset acquisition required** | Model exists; local derivative images not verified against official source pixels | Person/not-person accuracy, CE; 1,664,896 metadata-only candidate bits |
| 2 streaming 1D DS-CNN | INT8: **Adapter required** | Model and official 10-second fixture exist; no conformance lock, long waveform/MUSAN missing | Event false positives/false negatives; 368,320 metadata-only candidate bits |
| 3 ResNet-18 / ImageNet-1k | FP32: **Asset acquisition required**; INT8: **Research-stage** | Registered model pin, no shared checkpoint or full ImageNet; no INT8 recipe | Accuracy/CE; eligible bit population unknown |
| 3 MobileNetV3-Small / ImageNet-1k | FP32: **Asset acquisition required**; INT8: **Research-stage** | Registered `small_100.lamb_in1k`, not an unspecified MobileNetV3; no weights/dataset/INT8 recipe | Accuracy/CE; unknown |
| 4 ViT-Base / ImageNet-1k | FP32: **Asset acquisition required**; INT8: **Research-stage** | Registered patch16-224 checkpoint fine-tuned for 1k classes; no acquired weights/data/INT8 recipe | Accuracy/CE; unknown |
| 4 DistilBERT / SST-2 | FP32: **Asset acquisition required**; INT8: **Research-stage** | Registered SST-2 fine-tuned checkpoint; data/tokenizer assets not acquired; no INT8 recipe | Binary sentiment accuracy/CE; unknown |
| 4 Pythia-160M / WikiText-2 | BF16: **Research-stage**; INT8: **Research-stage** | Source pin exists, but provider config declares **FP16**, not BF16; no acquired weights/data | Token-average NLL/perplexity; unknown |
| 4 Qwen3-0.6B / WikiText-2 | BF16: **Asset acquisition required**; INT8/INT4: **Research-stage** | Provider declares BF16; no acquired weights/data or approved quantized variants | Token-average NLL/perplexity; unknown |

The metadata-only VWW/streaming populations were counted from actual local
TFLite operators and kernel buffers without executing inference. Counts do not
certify eligible-buffer mutation/restoration or the task pipeline. Zero words
remain allocated. All verified TinyML counts exclude INT32 biases, quantizer
metadata and activations.

## Existing TinyML evidence and limits

All five source models use official MLCommons Tiny revision
`4addd0fa08d216e20637637874e084895f289da4`. Full SHA-256 values and file paths are
in the registry. Existing exact execution is CPU TensorFlow **2.14.1**,
`BUILTIN_REF`, one thread, no delegate; it must not be replaced by modern TF
because a newer runtime loads the model. Stored-code XOR is authoritative;
companions provide surrogate derivatives, not derivatives of the integer program.

| Existing reference | Locked clean result (not a new M96 evaluation) | Conditional readiness |
|---|---|---|
| ResNet-8 | 175/200 official subset; 8,707/10,000 full test | Static S2–S6/empirical supported; schema-6 partitions remain KWS-only |
| KWS | 901/1,000 official subset; 0.9161554192 over 4,890 | Existing KWS exact selection, context and paired-swap workflow |
| ToyCar | About 0.862037 file AUC over 248 files under locked float-domain scoring | Static task metrics supported; conditional file-AUC and input-resampling extension still needed |

The ToyCar EEMBC score (~0.861839) and float-domain reference are distinct
scoring definitions, not interchangeable rounded numbers. File AUC is
non-additive over files; a file-resampling analysis must recompute it, not average
per-file “AUC.” Reconstruction-MSE gradients remain a surrogate for that task
objective. Extending schema 6 must explicitly choose risk and retain both metrics.
ResNet-8 needs an explicit disjoint CIFAR sample plan rather than relabeling KWS
sample roles. Neither extension is implemented in M96.

Current static methods are random, corrected word-relative MSB, exact affine
numerical jump and S4/S5/S6 with mapped companion kernels. Reference inference,
ranker construction and full restoration were already tested in M93 and continue
to pass pinned tests here. This is software conformance, not a generalized
scientific efficacy result. See `MLPERF_TINY_M93_MULTIBENCH.md` and
`MLPERF_TINY_M71_CONFORMANCE.md` for backend and preprocessing evidence.

### VWW gate

The 1000 local derivatives have official sample IDs and balanced labels, but
matching IDs do not prove matching pixels. Provisional accuracy 0.858 must not
be described as official benchmark conformance. Independently verify the
original image archive (or an authoritative equivalent byte-level source and
documented reconstruction), preprocessing and split before an adapter readiness
claim. Prior archive access was blocked; a third-party SHA is not independent
verification. No substitute COCO derivative was created here. The model remains
available, and this asset gate can proceed independently of KWS work.

### Streaming gate and recommended next model

Recommend **streaming 1D DS-CNN**, initially the bounded reference onboarding
gate, because the actual INT8 model and a ten-second official waveform are
already present and its convolution/depthwise operations have a credible exact
TFLite path. This recommendation is not approval to claim long-stream quality.

Model: `benchmark/training/streaming_wakeword/trained_models/str_ww_ref_model.tflite`,
SHA `3af8550895ba7d5c584277102b5075c52dcfa63ba9d2b2240f37c4e6abd5dd2b`,
74,520 bytes. Parsed signature INT8 `[1,30,1,40]` → `[1,3]`; four Conv2D,
four depthwise kernels and one dense kernel: 46,040 stored kernel words.
This is the deployed representation of the named 1D architecture, not permission
to replace it with a generic KWS model.

Fixture under `benchmark/runner/sww_data_dir/`:

- `short_wav_easy10s.wav`, 640,044 bytes, SHA
  `741ab419e46087ce1b738b23ca12c0633e796234f81673c7554629c38a4b708a`.
- Corresponding JSON, 216 bytes, SHA
  `c1da7de188c118a2abcffedc459469ea250b55a504f63f17110827a1298a6c08`.

The medium waveform is absent. The long workload needs exact construction from
Speech Commands v0.02 and MUSAN, with constituent licenses/provenance and roughly
11 GB additional audio acquisition reviewed by the researcher. No download is
authorized by this roadmap. Upstream streaming specifications use event counts,
timing, thresholds and debounce, not independent frame classification. The
source's provisional INT8 result does not itself establish the official quality
gate. See the [official streaming reference](https://github.com/mlcommons/tiny/blob/master/benchmark/training/streaming_wakeword/README.md)
and [Tiny rules](https://github.com/mlcommons/tiny/blob/master/benchmark/MLPerfTiny_Rules.adoc);
implementation must use the pinned local revision, not mutable master.

## Registered non-Tiny checkpoints and acquisition gates

These source revisions are pinned in Kona Assets, but their weight files are
not in the audited shared model store. The registry records complete expected
file SHA-256 and sizes. A model card dtype is a clue, not a substitute for
checking the acquired tensor headers and executable dtypes.

| Family / source | Immutable revision | Registered weight size | Architecture/task checks |
|---|---|---:|---|
| [microsoft/resnet-18](https://huggingface.co/microsoft/resnet-18) | `65a5785d9156231087c481e0c7dd33a5ff6f7e3e` | 46.8 MB | ImageNet classification head; pin processor resize/crop and label order |
| [timm/mobilenetv3_small_100.lamb_in1k](https://huggingface.co/timm/mobilenetv3_small_100.lamb_in1k) | `1824797e7887cbec1990e4adbd6675960a36c589` | 10.2 MB | Small variant, timm recipe/preprocessing, hard-swish/SE operations |
| [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224) | `3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3` | 346.3 MB | ImageNet-21k pretraining then ImageNet-1k fine-tuning; correct 1k head |
| [DistilBERT SST-2](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) | `714eb0fa89d2f80546fda750413ed43d93601a13` | 267.8 MB | Uncased tokenizer, 0 negative / 1 positive, truncation/padding attention masks |
| [EleutherAI/pythia-160m](https://huggingface.co/EleutherAI/pythia-160m) | `50f5173d932e8e61f858120bcb800b97af589f46` | 375.0 MB | GPT-NeoX, 12 layers, context 2048; provider FP16, untied embeddings |
| [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | `c1899de289a04d12100db370d81485cdf75e47ca` | 1.503 GB | 28 layers, grouped-query attention, tied embeddings, BF16 config |

Pythia's [pinned configuration](https://huggingface.co/EleutherAI/pythia-160m/raw/50f5173d932e8e61f858120bcb800b97af589f46/config.json)
declares `float16`; BF16 would be a newly identified, approved conversion.
Qwen's [pinned configuration](https://huggingface.co/Qwen/Qwen3-0.6B/raw/c1899de289a04d12100db370d81485cdf75e47ca/config.json)
declares BF16; Qwen3 requires a compatible Transformers release (provider minimum
4.51.0), but no exact NetSuRF backend/version has yet been validated. Proposed
frameworks are PyTorch/Transformers and timm for MobileNetV3, not automatic
TensorFlow cloning. Any new environment is researcher-managed and separately
validated; do not force these models through TFLite or invent INT8 variants.

Source model registries/cards declare Apache-2.0; acquisition must retain exact
license files and notices. Dataset terms are independent:

- Full ImageNet-1k is not local. `ILSVRC/imagenet-1k` revision
  `49e2ee26f3810fb5a7536bbf732a7b07389a47b5` requires researcher acceptance of
  its custom access/research terms. Local Imagenette is **not** ImageNet-1k.
  See [dataset access terms](https://huggingface.co/datasets/ILSVRC/imagenet-1k).
- SST-2 is not local. `nyu-mll/glue` revision
  `bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c` registers 67,349 training, 872
  labeled validation and 1,821 withheld-label test rows. Review dataset terms
  separately, and prespecify calibration/validation roles without claiming the
  fine-tuned model's historical validation data are an untouched test set.
- WikiText-2 raw is not local. `Salesforce/wikitext` revision
  `b08601e04326c79dfdd32d625aee71d232d685c3`, `wikitext-2-raw-v1`, registers
  CC-BY-SA/GFDL provenance. Source rows (36,718/3,760/4,358) are not independent
  token counts. Preserve documents and notices when constructing a corpus.

No clean score is asserted for an unacquired checkpoint. Required gates are:
verified source tensors/dtypes, exact task preprocessing, deterministic baseline
outputs, declared numerical backend, eligible-storage-to-runtime mapping,
bit-flip outputs, full restoration, and calibration/validation/final partitions.

## Precision-aware contracts required before implementation

| Representation | Population and mutation | Exceptional values / interpretation |
|---|---|---|
| Current affine INT8 TFLite | Signed 8-bit kernel storage, per-channel scales/zero-points fixed, private byte XOR; bias excluded | Exact reference integer execution; output loss calculation defined by task; no float-shadow substitution |
| FP32 | Explicit tensor scope and 32-bit IEEE storage identity, endian/word mapping; disable implicit casts/autocast | Sign/exponent/mantissa flips can create NaN/Inf. Predeclare failure-category rates and risk treatment; do not silently clamp, discard, retry or assign finite CE |
| BF16 | Actual 16-bit stored codes, distinct from FP16; establish whether runtime expands/repackages them | Eight exponent bits, different range/precision from FP16; same exceptional-output contract and round-trip byte checks |
| Affine/other INT8 derivative | Pin recipe, calibration set, rounding/saturation, scale grouping, zero-points, backend packing and new checkpoint hash | Weight-only versus weight+activation quantization must be explicit; source FP32 quality is not quantized quality |
| Packed INT4 | Pin signed encoding, nibble order, group sizes, packed-word/tensor/bit identity, scales and padding | Padding and duplicated runtime packs are not silently counted as independent stored parameters; metadata inclusion requires its own population |

For every new population fix the lifetime (initially persistent parameter
storage), target Bernoulli law, replay identity and private-copy restoration.
Expected proposed count is Np for N eligible bits; with deterministic ideal mask
P, expected applied count is (N−|P|)p. Record realized counts and
proposed=applied+suppressed. Multi-bit flips in one word compose by XOR. Clean
restoration includes runtime caches/repacked weights, not only source arrays.
Equal BER or equal coverage across different N/precisions is not equal physical
exposure, area, or reliability cost. Include bit counts, absolute budgets and
inference/construction cost; no FIT conversion is justified.

### Language-model scoring is a separate task contract

Use checkpoint-specific tokenization and frozen raw-text token streams. Define
document concatenation/separators, BOS/EOS insertion, context length, stride,
target masking, burn-in, batch effects and denominators. Compare token-average
NLL and perplexity within a given tokenizer/protocol; do not directly equate
perplexities across Pythia and Qwen tokenizations. Qwen chat/“thinking” prompts
must not silently enter a raw WikiText likelihood task. The usual
[sliding-context perplexity procedure](https://huggingface.co/docs/transformers/perplexity)
illustrates why context and stride matter; it is not yet an approved local recipe.

Exclude activation, KV-cache and runtime-arithmetic faults from the parameter
storage population. Cache reset/replay and tied weight identities must be
explicit: tied storage is not two independent injectable copies. Independent
input uncertainty should respect document/sequence blocks, not treat every
overlapping token window as independent. Pretraining contamination and WikiText
representativeness remain limitations even with disjoint experiment partitions.

## Computation, memory and storage plan

Measured existing costs are evidence from M93/M956/M957, not promises for new
architectures. KWS M93: ~2.79 s load, 37.7 s for its 1000-example conformance,
~0.09 s/32-clip faulted inference, ~0.46 GiB peak; eight-example S4 ~0.379 s,
S5/S6 ~3.63 s. ToyCar: ~1.97 s load, 17.5 s/248-file full reference, ~1.55 GiB
peak; bounded two-file/392-window faulted inference ~0.038 s. These protocols
include different overheads and are not cross-task efficiency rankings.
M96 exact campaign costs are in `M96_CROSS_POOL_PROTOCOL.md`.

For new models, measure a no-training clean run, 16 candidate flips and tiny
gradient batches before selecting study budgets. Cost scales as unique
authoritative fault states × input count, with model loading/preprocessing
separated. Isolated studies add candidate count; exact K-pool enumeration adds
2^K×fields; S5/S6 add per-example gradient work. Do not extrapolate KWS clip
throughput to ImageNet or tokens.

| Tier | Planning resource allowance (not measured acceptance criteria) | Storage/cost hazards |
|---|---|---|
| Existing/next TinyML | CPU first; 1–2 GiB or more for full score/event maps | VWW source pixels and SWW long-wave dependencies dominate acquisition; exact events grow with BER×N×fields |
| ResNet-18/MobileNetV3 | 2–4 GiB RAM; modest GPU optional for gradients | Weights ~10–47 MB, full ImageNet dominates disk; per-bit Python objects can exceed model RAM |
| ViT/DistilBERT | 4–8 GiB RAM/GPU starting allowance | Weights ~0.27–0.35 GB; attention and per-example derivatives grow with sequence length/batch |
| Pythia-160M | 4–8 GiB RAM/GPU starting allowance | ~375 MB source; token sequences, tied/scope audit and gradient maps dominate beyond weights |
| Qwen3-0.6B | 8–16 GiB RAM/GPU starting allowance | ~1.50 GB source, context/KV costs; full per-bit score tables can be billions of entries |

Before large-model support, use compact tensor/word/bit iterators and bounded
candidate analyses; do not instantiate billions of Python identities. This is a
future scale gate, not an implemented optimization or license to compress away
replay information. Record peak RSS, device memory, thread/backend settings,
artifact bytes and score construction time per accepted adapter.

## Method comparison matrix and paper boundaries

| Method | Existing status / required distinction |
|---|---|
| Random, corrected MSB, absolute encoded numerical jump | Implemented for three exact TinyML static adapters |
| Global first-order | Signed mean contribution is recorded separately; S4 ranking is its absolute value, not signed-damage ranking |
| Per-example absolute / positive first-order | Existing S5/S6; companion loss and actual task metric must be distinguished |
| Historical qPolarGrad | Recovered explicitly named legacy/Dense expression; not automatically valid for convolution/attention; no new reinterpretation here |
| Exact isolated-bit oracle | Existing bounded empirical measurements and selection comparator; sampled candidates are not full-network optima |
| True weight-Hessian FKeras | Paper-form principal-Hessian score/HVP implementation and bounded KWS HVP gate exist; no validated full KWS eigensolve or protection efficacy. Not Hessian diagonal or integer-program Hessian |
| Unchanged static allocation | Frozen source baseline, supported |
| Exact restricted-mask selection | Existing finite 2^K response enumeration; M96 samples pools around a fixed context, not a scalable whole-network optimizer |
| Unguided swaps / random or shuffled guidance / conditional MC / gradient candidate groups | Earlier Dense search infrastructure has architecture-specific implementations; not a validated matched KWS/ToyCar/ResNet-8 publication suite |
| Attack proposals, importance sampling, MCMC/annealing, RL, new optimizer | Future only; no implementation in M96 |
| Factorized/Kronecker surrogates | Prior Dense-specific diagnostics; require architectural and mathematical validation before cross-model comparison |

The FKeras paper/code discrepancy remains documented in
`M95_CONDITIONAL_DAMAGE.md`: the recovered public accumulation appears to use
an additional eigenvalue factor; the implemented paper-form score uses one.
Do not rename it a faithful package reproduction or infer all Hessian methods
miss interactions. No new eigensolve was run here.

Every comparative study must match eligible storage, lifetime/law, protection
cardinality, data, task risk, schedules and construction/evaluation budgets.
Paper 1 asks when isolated susceptibility can miss effective allocation under
simultaneous faults. Counterexamples establish possibility; M952/M957 establish
bounded examples; M96 prepares a generalization test. It does not establish
universal failure of static rankings, an inability of RL to learn interactions,
or population superiority before the twelve pools are evaluated. No prospective
Paper 2 optimizer is disclosed or implemented.

## Staged next work and concrete follow-on implementation brief

1. Consolidate ResNet-8/KWS/ToyCar static evidence and source contracts. Extend
   conditional protocols separately: CIFAR partitions first; ToyCar explicit
   file-AUC risk/resampling without disguising MSE as the task objective.
2. Onboard streaming DS-CNN through the bounded specification below. In parallel,
   resolve VWW original pixels; no provisional-dataset workaround.
3. Acquire approved full ImageNet and pinned ResNet-18/MobileNetV3 weights; agree
   FP32 exceptional-fault semantics before INT8 recipes.
4. Onboard DistilBERT/ViT, then Pythia/Qwen with tokenizer and precision gates.
   Acquisition or GPU availability is not numerical validation.

### Follow-on task: streaming INT8 DS-CNN reference and exact-bit gate

Scope: a separate worktree, no production sweeps or downloads without researcher
authorization. Start from the pinned Tiny revision/model/short-wave hashes above.
Read the official preprocessing, calibration, evaluation and debounce code in
that revision. Produce a new neutral asset lock only after conformance; do not
edit existing locks or claim the ten-second fixture is the full benchmark.

Deliverables and acceptance sequence:

1. Inventory waveform PCM/sample-rate/channel handling, feature pipeline,
   window/hop boundaries, quantization and labels/event times. Match the source's
   64 ms feature window, 32 ms stride, 40 coefficients and full model window
   construction exactly; do not substitute a similar MFCC pipeline.
2. Execute only the pinned short waveform with authoritative TF 2.14.1
   `BUILTIN_REF`; save input/output/event digests and compare source event logic,
   including confidence threshold, suppression and latency allowance. Report
   quality failures honestly. Establish repeated-run equality before mutation.
3. Register a benchmark-specific adapter, enumerate nine kernel populations,
   validate per-channel metadata and all INT8 code/bit transitions, multi-bit
   composition, protected suppression, proposed/applied counts and byte/output
   restoration. Persist faults for a declared entire-stream window; preprocessing
   and event-state reset must not vary with software batching.
4. Validate random/S2/S3 first. Reconstruct a kernel-mapped companion only if
   source operations and stream aggregation permit a defensible loss. Frame
   CE derivatives do not automatically predict event FP/FN. Document that
   objective mismatch before claiming S4–S6 task alignment.
5. Bound tests to the short fixture and ≤16 selected single-bit faults, plus
   tiny synthetic stream cases. Measure load time, frames/s, score construction,
   peak RSS and evidence bytes. No long fault campaign or retraining.
6. Deliver preprocessing/conformance diagnostics, exact hashes, tests and a
   separate acquisition/validation plan for the official long stream. If source
   quality or preprocessing cannot be reproduced, stop at that gate rather than
   changing thresholds or data. Full-study readiness remains blocked until then.

This is an explicit architecture integration, not automatic arbitrary-model
conversion. It can be assigned as a separate task after the researcher approves
the scope; M96 does not start it.

## Separate future hardware workstream

Ideal logical event suppression is neither TMR nor SECDED. A future matched
hardware comparison needs actual stored-word packing and exposure/lifetime,
register versus SRAM placement, correlated-fault and voter assumptions, and
area/energy/latency as well as application quality. Parity/ABFT detectors need
explicit retry/recovery and uncorrectable-error handling before any correction
claim. SECDED needs actual word grouping; Ranger needs calibrated ranges and
hardware comparators, not software-overhead extrapolation. Transient activation,
datapath and cache faults require separate populations and timing. Report silent
corruption, detected uncorrectable events and recovery cost separately. None of
these mechanisms, synthesis flows or radiation-rate claims is implemented here.
