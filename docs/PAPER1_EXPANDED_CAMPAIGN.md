# Paper 1: unified manuscript and campaign delivery

Audit date: 2 November 2026. Target internal gate: 9 November; review/revisions:
10–12 November. This replaces the old manuscript-only optional-week split.
Branch: `paper1-expanded-benchmarks`, based on manuscript commit
`589aaa67f485ae3eb36febe70132304a1b9f27ae`. No merge or push.

## Decision and present gate

**The manuscript and three prospective static campaigns are deliverable; the
expanded-architecture acceptance gate is INCOMPLETE.** One exposed streaming
fixture now has an executed, repeatable protection comparison. The attempted
VWW gate fails explicitly on image provenance. No verified local transformer
checkpoint/data/backend combination was found. Neither blocked probes nor
source metadata are counted as completed benchmark experiments.

The authoritative 18-target status ledger is
`config/paper1_target_status.json`. Its portfolio hash binds all 11 families
and 18 precision targets to the existing registry. The ledger distinguishes
acquired-model hashes from advertised upstream files, records the implementation
base and links exact finalized evidence. The delivery manifest supplies the
final implementation commit and hashes; the bounded run also records its dirty
source fingerprint. The old registry's conditional-workflow notes are historical:
later ResNet/ToyCar conditional implementations remain in their own worktrees.
This branch does not transplant those engines.

| Family / precision targets | Current executable evidence | Exact remaining boundary |
|---|---|---|
| ResNet-8 INT8 | Frozen reference, exact XOR, restored static comparisons | New four-budget common grid not executed |
| DS-CNN KWS INT8 | Same; restored 128-clip FKeras experiment | Common 1000-clip FKeras/random/jump grid not executed |
| ToyCar INT8 | Exact static inference and complete-file ROC-AUC | More fault fields planned; exposed inputs remain exploratory |
| Streaming 1D DS-CNN INT8 | Two identical bounded comparison payloads on the official short fixture | No independent full corpus, MUSAN, gradient protocol or approved scalar FP/FN tradeoff |
| MobileNetV1 VWW INT8 | Model metadata available; executable probe rejects data provenance | Original image/split mapping and reference-quality evidence needed |
| ResNet-18 FP32 / INT8 | Capability/prerequisite records only | ImageNet terms/data, model, exact backend; INT8 recipe |
| MobileNetV3-small FP32 / INT8 | Records only | Same; Imagenette is not ImageNet-1k |
| ViT-Base FP32 / INT8 | Records only | Same plus attention execution/storage conformance |
| DistilBERT SST-2 FP32 / INT8 | Records only | Weights/tokenizer/labeled data and approved backend; INT8 recipe |
| Pythia-160M BF16 / INT8 | Records only | Source metadata is FP16, not BF16; corpus, conversion and storage validation |
| Qwen3-0.6B BF16 / INT8 / INT4 | Records only | Assets/corpus/backend; explicit INT8/INT4 packing and conversion |

## The omitted FKeras experiment is now part of the manuscript

Source: `config/paper1_fkeras_source.json`; original finalized
`m97_kws_fkeras_exploratory_v1/20261031T215413.151168Z-6bbb9223`.
Payload `dd6ccdd8c000437082619ee67a79d25d067709eee210df8fc2c0daf5f712fb36`.
Every artifact in that run's checksum inventory was rehashed. Raw trials and
paired summaries were independently recomputed; no inference or mask was rerun.

This is 128 evaluation clips, 32 calibration clips, one ranking realization,
32 paired fault fields per BER, three BERs and four budgets. It is **not**
the 1000-clip primary KWS cohort. The new six-panel figure retains all its
methods, budgets and BERs. At BER .01 and 6.25%, paper-form FKeras minus MSB
is +8.374 percentage points (field-t interval +3.337 to +13.411; source
within-BER Holm p=.04590). At 18.75% it is +0.049 points with an interval
crossing zero. Paper/repository differences do not establish one formula's
superiority after the recorded multiplicity correction. At complete tiers,
their masks and outcomes equal MSB exactly.

The printed score is the absolute projected Hessian vector; the repository sum
is the sum of componentwise absolute squared-eigenvalue contributions, not
simply an extra eigenvalue outside an otherwise unchanged formula. The same
signed-eigenvalue convention, continuous companion, tier-first policy and
validated eigensolver are retained. The 32-clip construction costs are .398 s
global gradient, 5.591 s shared per-example pass, and 1.461 s shared FKeras solve.
The complete 2496-evaluation run recorded 1464 s coordinator time and 1104 MiB
peak; these are historical measurements, not forecasts for a different cohort.

## Executed new bounded evidence

Command, in this worktree and using a new output name:

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 \
 TF_NUM_INTEROP_THREADS=1 CUDA_VISIBLE_DEVICES=-1 TF_ENABLE_ONEDNN_OPTS=0 \
 /projects/kona-assets/manuelbv/.venv-mlperf/bin/python \
 -m netsurf.streaming_validation --output workspace/runs/my_streaming_engineering_check
```

Actual runs: `workspace/runs/paper1_streaming_gate_v1` and
`workspace/runs/paper1_streaming_gate_repeat_v1`. Both payloads:
`c8ee6163da82e8cd4fee6f13e1e9ed9eb49da6daac1dd6dc490fe49618792674`.
Model `3af8550895ba7d5c584277102b5075c52dcfa63ba9d2b2240f37c4e6abd5dd2b`;
46,040 eligible INT8 kernel words / 368,320 bits. Biases are excluded.
The fixture is previously exposed, 10 seconds / 282 windows, with three true
events. Clean detection gives two true positives, zero false positives and one
false negative; that miss is preserved.

Two persistent Bernoulli fields at .001 propose 372 and 376 flips. MSB, random
and numerical-jump masks at 6.25% and 12.5% share those fields and zero/full
controls: 16 evaluations, 4512 faulted-window inferences. Including clean and
restored comparisons gives 5076; the adapter's initial 282-window conformance
adds another 282 (5358 total per run). Every full mask restores exact clean
output; all events satisfy proposed=applied+suppressed. Multiple bits in a word
use the existing XOR engine.

There is a beneficial random-mask outcome and damaging numerical-jump outcomes;
they are retained. No superiority, accuracy, or invented scalar FP/FN risk is
reported from two fields on one stream. The first run took 24.09 s, including
2.21 s load and 8.26 s ranking/mask work; faulted evaluation was .156–.168 s
per complete fixture; peak RSS 758 MiB. The ranking timer excludes the preceding
exact-jump extraction; total runtime includes it. These are engineering costs,
not extrapolations to the full streaming benchmark.

The maintained static schema cannot yet express full-stream task metrics without
a separately agreed protocol. Accordingly this gate has its own explicit,
bounded engineering artifact schema; it does not masquerade as a publication
campaign or alter the existing scientific engine.

## Frozen common-grid studies (not run)

Each study uses the same existing frozen model, exact INT8 kernel universe,
reference backend, logical persistent Bernoulli law and ideal event suppression.
Coverage is eligible-bit cardinality, not hardware cost. The public grid is
BER `[0.0001,0.001,0.003,0.01]`, interior budgets
`[0.0625,0.125,0.1875,0.25]`, and shared 0% / 100% controls.
Each has four mask/schedule replications × eight fields per BER: 32 fields/BER,
128 total. Random-mask variability has four realizations, not 32.
The primary task summaries remain accuracy/CE for classifiers and complete-file
ROC-AUC for ToyCar, with reconstruction MSE separate.

| Frozen configuration | Evaluation/calibration units | Policies | Mask evaluations | Faulted model invocations |
|---|---|---:|---:|---:|
| `paper1_resnet8_common_grid_v1.toml` | 1000 images / 64 images | 6 | 3328 | 3,328,000 images |
| `paper1_kws_common_grid_v1.toml` | 1000 clips / 64 clips | 8, including both FKeras forms | 4352 | 4,352,000 clips |
| `paper1_toycar_common_grid_v1.toml` | 64 files / 16 files | 6 | 3328 | 41,746,432 windows |

Formula: 128 × (policies × four interior budgets + two controls). ToyCar uses
196 windows/file. Loading, conformance, clean/calibration evaluation and score
construction add work; they are not included in the faulted counts. Independent
hardware trials are the 128 fields, not thousands of policy applications.
Theoretical proposed means are Np; at .0001/.001/.003/.01 these are:
ResNet 61.888/618.88/1856.64/6188.8; KWS 17.6128/176.128/528.384/1761.28;
ToyCar 211.3536/2113.536/6340.608/21135.36. Expected applied counts multiply
by (1−coverage); actual counts are recorded, not replaced by these expectations.

All ranking/empirical/evaluation identities are checked against already exposed,
hash-verified source cohorts, and the role partitions are disjoint. This is an
exploratory extension, not untouched-input confirmation. New fault and ranking
seeds are fixed in the TOMLs; no final outcome selects masks. Existing within-BER
paired comparisons and Holm families are preserved. The prospective grid may
duplicate existing cells to obtain a single matched schedule family with new
partial tiers; it does not replace old evidence or reinterpret old p-values.

Resource plan (two workers, measured historical throughput extrapolated, not
guaranteed): ResNet roughly 3–5 h, KWS 1.5–4 h, ToyCar 2–4 h. Allow at least
10 GiB free RAM per concurrent study and 20 GiB free disk for the classifier
studies; ToyCar at least 30 GiB RAM and 50 GiB disk. These are conservative
reservations, not measured future usage. Whole-population score/mask JSON and
replication copies dominate memory/storage, while exact events grow with Np.
Full evidence ZIP can require another bundle-sized allocation plus compression
time; do not launch all three simultaneously on the strength of these estimates.

## One-shot foreground commands

Use this worktree, not canonical v2:

```bash
cd /projects/netsurf/manuelbv/workspace/implementation-worktrees/paper1-expanded
bash workflows/run_paper1_resnet8.sh --preflight-only
bash workflows/run_paper1_kws.sh --preflight-only
bash workflows/run_paper1_toycar.sh --preflight-only
```

All three preflights have been exercised. They load/verify source inputs but do
not execute research fault trials. The ResNet new TOML uses the lock's resolved
`/data` paths rather than stale `/ml` aliases; the lock itself is unchanged.
Earlier development preflight receipts are retained under `workspace/scratch/`.

Researcher-run foreground commands (one at a time, e.g. inside tmux):

```bash
bash workflows/run_paper1_resnet8.sh
bash workflows/run_paper1_kws.sh
bash workflows/run_paper1_toycar.sh
```

Each script pins `/projects/kona-assets/manuelbv/.venv-mlperf/bin/python`,
thread settings and source root. No activation, installation or wrapper process
is needed. The driver validates TOML/lock/model hashes and the exact exposed
partitions; the existing engine constructs and validates calibration-only masks.
It runs in the foreground, verifies the finalized bundle, derives raw-based
paired tables/plots, and creates compact/full ZIPs. It prints and saves Mac
`scp` commands under
`workspace/paper1-campaigns/<experiment>/exports/MAC_DOWNLOAD.txt`.
Replace SSH alias `kona` only if your Mac uses a different configured alias.

Completed stages are reused only after file-hash verification. A changed
configuration/code fingerprint, corrupt stage, concurrent invocation, or prior
unsealed execution fails closed. Interrupted runs are not silently restarted;
inspect and explicitly verify the existing result before creating a new plan.
Do not edit a frozen TOML in place after running it: create a new version/name
and update its plan record. Compact ZIPs intentionally omit large replay files
and include a scope notice; use full-evidence ZIP for exact replay. Neither
archive contains input datasets or checkpoints. No experiment, selection,
inference or download is hidden inside the manuscript build.

## Asset and dependency boundary

Read-only checks found no approved local ImageNet-1k, SST-2 or WikiText-2 data
and no corresponding model weights in the audited shared roots. The HF cache
held only WikiText metadata. Canonical Python lacks torch, transformers,
datasets and safetensors. Other historical environments contain torch but no
verified complete transformer stack. No environment or shared asset was changed.

Recommended next onboarding is DistilBERT FP32: ~268 MB upstream safetensors,
small labeled SST-2 validation, and an explicit classification task. Before
implementation, the researcher must approve/provide the pinned checkpoint,
tokenizer, dataset and a separate validated interpreter. The source revision is
`714eb0fa89d2f80546fda750413ed43d93601a13`; the registry records weight SHA
`7c3919835e442510166d267fe7cbe847e0c51cd26d9ba07b89a57b952b49b8aa`.
SST-2 uses `nyu-mll/glue` revision
`bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c`; the withheld-label test set is not
an evaluation substitute. Previously used validation examples must not be
misrepresented as untouched confirmation.

Proposed user-managed dependency request, **not an accepted/pinned backend and
not executed here**: PyTorch >=2.6,<3; Transformers >=4.51,<5; safetensors >=0.5,<1;
datasets >=3,<5. In the researcher's separately chosen Python 3.11 environment:

```bash
python -m pip install "torch>=2.6,<3" "transformers>=4.51,<5" "safetensors>=0.5,<1" "datasets>=3,<5"
python -m pip check
python -c "import torch,transformers,safetensors,datasets; print(torch.__version__,transformers.__version__,safetensors.__version__,datasets.__version__)"
```

Actual resolved versions, CPU/device arithmetic and safetensor byte mappings
must pass clean, finite-fault and restoration tests before any result is accepted.
The proposed software route follows the
[versioned Transformers installation guidance](https://huggingface.co/docs/transformers/v4.51.3/installation)
and [DistilBERT API](https://huggingface.co/docs/transformers/v4.51.3/model_doc/distilbert);
those documents are not evidence that the proposed local combination was tested.
Do not install this into the pinned TF2.14.1 environment. An existing approved
interpreter may be supplied instead. User-approved acquisition can use
`hf download distilbert/distilbert-base-uncased-finetuned-sst-2-english --revision 714eb0fa89d2f80546fda750413ed43d93601a13 --include config.json model.safetensors tokenizer_config.json vocab.txt --local-dir <approved-staging-directory>`;
dataset acquisition and licensing are separate. This command is deliberately
not automated or claimed to complete onboarding.

FP32/BF16 exceptional outputs need an explicit nonfinite-failure contract;
INT8/INT4 derivatives need verified scales, packing, duplicate/tied storage and
padding semantics. A downloaded model alone is not fault-injectable support.
No target is silently replaced with another precision or smaller dataset.

## Scientific priorities through the integrated deadline

1. Review the complete manuscript now, including favorable and unfavorable
   restored FKeras results and the explicit prior FCCM/RL overlap.
2. Researcher executes the frozen common grids; inspect **all** conditions,
   paired effects, catastrophic trials and ranking/evaluation cost. Do not
   cherry-pick methods or retroactively narrow the multiplicity family.
3. Resolve assets/backend for one independent transformer task; then implement
   exact storage and bounded smoke before a prospective campaign. If unresolved
   on 9 November, report the gate incomplete rather than invent results.
4. Resolve full streaming data/protocol or authoritative VWW provenance. The
   short fixture is engineering evidence only; it cannot carry generalization.
5. Nov10–12: adversarial review, author approval and editorial corrections.
   No presumed TECS acceptance or universally superior protection policy.
