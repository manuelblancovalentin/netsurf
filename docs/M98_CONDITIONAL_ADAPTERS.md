# M9.8 — ResNet-8 conditional protection and ToyCar task-risk gate

Implementation branch `m98-conditional-adapters`, isolated at
`/projects/netsurf/manuelbv/workspace/implementation-worktrees/m98`, based on
consolidated `v2` commit `e3209f7`. No merge, push, asset/environment change,
FKeras change or publication campaign execution. Existing M97 worktrees and
their runs are untouched. This extends the existing exact selector; it does
not add an optimizer, proposal sampler or estimator.

Implementation commit: `b429e72e6f60180d4a07e828c31409eeb5fbefbd`.
Authorship and committer identity are the configured researcher
`manuelblancovalentin <manuel.blanco.valentin@gmail.com>`.

## Readiness and scientific boundary

| Capability | KWS | ResNet-8 | ToyCar |
|---|---|---|---|
| Pinned TF 2.14.1 BUILTIN_REF, exact kernel XOR, restoration | VERIFIED, unchanged | VERIFIED | VERIFIED, unchanged |
| Schema-6 conditional and fixed-budget restricted selection | VERIFIED, old hashes retained | VERIFIED, schema 6.7 | BLOCKED for production protocol |
| Frozen selection → independent paired final comparison | VERIFIED | VERIFIED | Needs reviewed complete-file cohort and nonadditive reporting contract |
| Cross-pool freeze/seal/aggregate | VERIFIED, request 1.0 unchanged | VERIFIED, request 1.1 | No additive per-input AUC decomposition allowed |
| Whole-cohort response-table identity | Existing CE evidence | Exact CE smoke | Synthetic AUC test and bounded real two-file test |

ResNet implementation readiness is not evidence of protection superiority.
The tiny engineering results are not a publication comparison. ToyCar's
blocked status is an orchestration/partition/reporting gate, **not** a claim
that the response-table or paired-swap mathematics fails for ROC-AUC.

## Frozen ResNet identity and lock reconciliation

- Official MLCommons Tiny source revision:
  `4addd0fa08d216e20637637874e084895f289da4`.
- Model: `benchmark/training/image_classification/trained_models/pretrainedResnet_quant.tflite`.
- Checkpoint SHA-256:
  `3c002613d1b2475eb51dd78dfb85a546c8ae658dee71cf6ade43b022fe205415`.
- Current IC lock: `/projects/kona-assets/manuelbv/locks/mlperf-tiny/ic.lock.json`.
  Canonical digest `sha256:d0daf561e681b5a31f23350192c6b5d7041b27024bf054d13a2eb793e581abd9`;
  file SHA-256 `65226641813cf4192d4898293b09d3861a2ccbcae0e4aca04beb3d15bd3bbbc9`.
- CIFAR archive SHA-256:
  `6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce`.
- Official 200-image reference output SHA-256:
  `f045f8f11e923bb37c467d1969e3e53c5b966061f1134688bd69b3e7f01338b9`.
- Eligible population: **77,360 allocated INT8 kernel words / 618,880 bits**.
  Zero codes remain faultable. INT32 biases, scales, zero points, activation
  buffers, graph metadata and runtime copies remain excluded.

M97 reconciled the old test's `f953…` lock expectation and obsolete `/ml`
paths by auditing the current lock against unchanged model/data/output pins.
M98 requires the exact current portfolio pin and verifies the official
output digest with the original adapter. A changed lock is rejected, not
automatically accepted. The historical complete old lock is still unavailable;
there is no claim to have reconstructed every historical metadata edit.
No lock bytes or test thresholds were changed here.

Inference is one-thread TensorFlow Lite 2.14.1 `BUILTIN_REF`, no delegates.
Input is pinned CIFAR-10 uint8 RGB converted exactly to INT8 by subtracting
128. Primary risk is the existing mean
`-ln(max((integer_softmax[label]+128)/256, 1e-12))`, in nats/image, with no
renormalization. Classification accuracy is descriptive. No surrogate or
gradient is evaluated during conditional selection/final evaluation.

## Verified whole-network S5 context

The source is a real finalized run, not an inferred mask location:

`/projects/netsurf/manuelbv/workspace/runs/m92_resnet8_static_test1000_v1/20261029T084524.633748Z-f0bcc564`

- Source payload:
  `3ed95baa04ede8047f5b5eb04f9024f2e5f856c39abcc84b2ca1fb894b22db32`.
- `artifacts/protection_masks.json` SHA-256:
  `f3045a36181c569d4ae7e5f4568dccc718ce105e3f20a97090a3807bc80557b8`.
- Key: `replication_0/per_example_absolute_global@0.125000`.
- Protected: **77,360 bits**, unprotected: **541,520 bits**.
- Its ranking used training indices 0–63; its ordinary final test slice was
  indices 0–999. It is one frozen model/ranking allocation, not independently
  trained networks or a fresh ranking replication.

Freezing verifies source completion, payload hash, exact mask artifact hash,
checkpoint, eligible identities, uniqueness and cardinality before sampling
any pool. Large source prediction tables are not needed and are not rescanned.
Each new pool draws two protected and four unprotected identities uniformly
without replacement within status strata, using the existing PCG64 routine.
Inclusion probabilities are `2/77360` and `4/541520`; they are not uniform
whole-network bit probabilities. Independent pools may overlap. No fault
outcome enters the draw. All exact identities, codes, bit positions, scales,
zero points, source protection status, seeds and overlaps are in `campaign.json`.

For every local budget-two mask, `P_out` contains 77,358 bits and
`P(Q)=P_out union Q` contains exactly 77,360. Original S5, exact isolated and
selected masks use the same global cardinality. The selector integrates all
64 candidate states per outside field for all 15 masks, retaining every
candidate interaction order, including beneficial faults. Final comparisons
load sealed masks; they cannot reselect them or choose a final optimum.

## Partitions and prior exposure

Schema 6.7 uses capability-defined source windows. Existing KWS windows are
unchanged. The new ResNet windows are:

| Role | Original CIFAR-10 test positions | Capacity |
|---|---|---:|
| `conditional_calibration` | 0–999 | 1,000 |
| `conditional_model_selection` | 1000–2999 | 2,000 |
| `conditional_final_evaluation` | 3000–9999 | 7,000 |

TOML offsets are relative to these windows. Sample identities use the original
`cifar10_test` index; the official-subset alias is normalized during exposure
audits because it represents the same original images, not a disjoint dataset.
No assertion of iid or identical class distributions follows from contiguous
windows. Execution records actual labels, class composition, input hashes,
partition hashes and ordering. Selection-only does not load final inputs.

The initial metadata audit found **14 finalized ResNet runs and 1,181 distinct
test identities** with recorded fault outcomes. Their paths/payload hashes,
checksum-inventory hashes and exact indices are in the frozen campaign's
`exposure_inventory`. Small payload identity is verified; multi-GB predictions
are not loaded. The source S5 ranking uses train identities, which are disjoint
from these test windows. Model-selection samples are not presented as unseen
confirmation samples.

The production plan reserves selection **1024–1055** (32 images) and final
**3225–3352** (128 images). Final start 3225 is the first 128-image contiguous
gap in the audited final window, selected from identity exposure only, **not
from labels, accuracy or damage**. The earlier 200-image scattered fault
subset is included in that audit. No production-window inference was performed
to choose or tune the plan. The full source test split has previously undergone
clean-reference evaluation; “unexposed” means no recorded fault-outcome use,
not inference-naive data.

Engineering explicitly designates test **1000–1001** for selection and
**3000–3001** for final. These now count as exposed. Their observed class
counts are `{0:1,5:1}` and `{1:1,5:1}` respectively—two images are not
representative class strata. They are outside the production windows.

Execution rescans known canonical/worktree run and scratch roots. Confirmation
overlap rejects before inference; completed members of the same frozen campaign
may share its planned final set. This is not a global reservation service:
failed/manual/concurrent/out-of-root exposure must be disclosed. Do not run
other studies on these confirmation identities concurrently.

## Versioned interface and unchanged mathematics

New experiment schema **6.7.0**, protocol
`capability-conditional-protection/v8`; new campaign request **1.1.0**, protocol
`capability-cross-pool/v2`. Schema 6.0–6.6 and campaign 1.0 behavior is retained.
The underlying `restricted_subcube`, `paired_swap`, storage codec, estimators
and fault-law implementations are unchanged.

- `netsurf.conditional_benchmarks`: explicit capability/partition contracts,
  current asset pin verification, exposure audit, architecture-specific ResNet
  partition adapter and opt-in authoritative per-image CE capture.
- `conditional_config`, `conditional_damage`, `conditional_bundle`: additive
  6.7 dispatch, capability-bound identity checks and versioned output reading.
- `cross_pool`: additive 1.1 freeze/seal/run path; existing sampling and lifecycle.
- `cross_pool_analysis`: same paired field/pool statistics; capability metadata
  and a bounded CSV reader for long exact event lists. ResNet BER .01 exceeded
  Python's default 128 KiB field limit; 64 MiB per field is allowed and the
  previous process limit restored. **No events are truncated or discarded.**
- `benchmarks.capabilities`: ResNet conditional readiness now explicit; the
  historical 18-target registry remains unchanged as the M96 inventory.
- `__main__`: restores the missing final JSON print for `campaign` commands.

The TOML keys keep their existing meanings:

| Section | Supported fields |
|---|---|
| `benchmark` | `adapter`, `id`, `checkpoint`, `dataset`, `asset_lock`, `inference_backend`, `split`, `quantization`, `quantization_backend`, `injectable_parameters` |
| `samples` | `calibration_role/count/offset`, `model_selection_role/count/offset`, `evaluation_role/count/offset`, `require_disjoint=true` |
| `risk` | `metric="categorical_crossentropy"`, `direction="minimize"` |
| `candidates` | explicit `parameter`, `flat_index`, `bit_index`; 2–12 unique eligible bits for restriction |
| `backgrounds` | `probabilities`, calibration/evaluation `count` and `seed` |
| `restricted_oracle` | existing enabled/budget/count/seed fields and explicit `calibration_replicate_seeds` |
| `protection_context` | existing verified source bundle, payload/artifact hashes, mask key, population and cardinality |
| `paired_swap_evaluation` | `enabled`, `background_count`, `seed`, ordered existing three `comparisons` |
| `workflow` | selection/final-only mode, purpose, provenance, optional per-input capture; existing frozen selection/campaign binding fields |
| `execution`, `output` | confidence; root/report/plots (schema-6 plots remain offline) |

Supported 6.7 adapters: `mlperf_tiny_ic_int8_v1` and
`mlperf_tiny_kws_int8_v1`. Other capabilities fail explicitly, including ToyCar.
Omitting workflow retains joint conditional/restricted/paired diagnostic
execution; it is not a separated confirmation run. Separated explicit pools
use `prespecified_explicit_engineering/v1`, engineering only. Probability-sampled
pool provenance requires the frozen campaign binding. Final-only requires the
source selection hashes; campaign finals additionally require the complete
sealed registry. Different model/metric/data/candidate/seed/budget plans reject.

Campaign 1.1 retains all request keys from 1.0: `benchmark_template`, `name`,
`purpose`; `design.candidate_count`, `pool_seeds`, `selection_seed_bases`,
`selection_replicates`, `selection_fields_per_seed`, `final_seeds`,
`final_fields`, `logical_ber`; selection/final sample offsets/counts; bootstrap
seed/replicates and 0.95 confidence. It allows an explicitly prespecified BER
in `(0,1)` instead of hardcoding .01. The delivered plan uses .01.

## Frozen production plan — NOT EXECUTED

Editable request: `config/examples/m98_resnet8_cross_pool_k6.toml`.
Binding template: `config/examples/m98_resnet8_selection_template.toml`.
Engineering request: `config/examples/m98_resnet8_engineering.toml`.
Use campaign commands, not the source-binding template as a publication run.

Frozen manifest: `workspace/runs/m98_resnet8_cross_pool_k6_v1/campaign.json`.
SHA-256: `5bee01e08717c06535142e6ec44e52cb2327dd427af828c7f2fd4045c32bf1da`.
Four K=6 pools, local budget 2, global S5 coverage 12.5%, logical BER .01.
Pool seeds 980101–980104; selection seed pairs 980201–980208;
final seeds 980501–980504; bootstrap seed 980801.
Selection has 2 batches ×4 independent fields =8 fields/pool; final has
32 independent whole-path master fields/pool. Derived-seed overlap is rejected.
All four selections must finalize before the final registry can be sealed.

| Work | Per pool | Four pools |
|---|---:|---:|
| Selection response states | 8×64 =512 | 2,048 |
| Clean + isolated selection requests | 7 | 28 |
| Total selection requests | 519 | 2,076 |
| Selection image inferences (32 images) | 16,608 | 66,432 |
| Analytical mask integrations | 8×15 =120 | 480 |
| Final master fields | 32 | 128 |
| Final state request bound (3 contrasts ×2 swaps ×2 states) | 384 | 1,536 |
| Final image inference bound (128 images) | 49,152 | 196,608 |

Total: **3,612** state requests / **263,040** image inferences before reuse,
plus 200 previously exposed reference images per adapter initialization
(normally eight loads =1,600 images). Actual final calls depend on frozen path
length and schedule-cache reuse; identical masks have zero contrast.
Selection outside universe is 541,516 bits (four unprotected candidates removed).
Expected outside faults at .01 are 5,415.16; natural proposed full-universe
events are 6,188.8 and applied after full static mask 5,415.2. Forced candidate
states are not natural BER draws. Actual events remain in the bundles.

This is a modest prespecified cross-pool pilot, not a power calculation.
Primary selected−S5 and two secondary contrasts use the existing whole-field
and equal-pool procedures. Pools, fields and shared images are not interchangeable
replications. Four pools give very limited between-pool inference. Input-resampling
diagnostics are assumption-limited; no causal mechanism or global optimum follows.

## Bounded engineering evidence

Campaign `workspace/runs/m98_resnet8_engineering_v1/campaign.json`, SHA-256
`e936ec77557886d519fa437b118a084cbca0776ad455d30afc3d8287a203a4ca`.
Two K=6 pools, two selection fields and two final fields each, two images/role.
Selection used 135 state requests per pool; final used six and eight unique
authoritative states. Every bundle verified restoration, cardinality, sample
disjointness, exact event accounting and sealed mask identity.

| Pool / phase | Runtime s | Scientific payload SHA-256 |
|---|---:|---|
| 000 selection | 38.04 | `9efe2b2a39cdae684ea59d720be7932b6a1582fb41efc3a62b96973528667806` |
| 001 selection | 37.01 | `d1dd2b8371576d7e35f2fdc5b8c69cea8844496459c1c1924e3b3d69e31096c6` |
| 000 final | 24.60 | `31a8e07f5069ce096b01151a746a462ca61d2543c5384195c8a26d853cc50b1a` |
| 001 final | 21.45 | `2ae04ab732de232b0d9aec9d8a938a28834b729e83ad9da55801023d7a4504be` |

Pool 000 original/isolated/selected local masks were `[0,1]`, `[1,3]`, `[1,2]`;
pool 001 `[0,3]`, `[1,2]`, `[1,2]`. Selected−S5 mean CE was −0.00406279
and +0.000086347 respectively. **These are software checks, not effectiveness
evidence**: two fields and two images cannot establish superiority.

Offline report/plot: `workspace/runs/m98_resnet8_engineering_analysis_v1/report.md`
and `pool_effects.png`. Independent regeneration in `..._analysis_repeat`
gave identical analysis payload
`a6459d4d2cae0d8430a89274ec5af3e5f7992c7cc302685763167cdfc67085f6`.
An additional engineering selection replay reproduced pool 000's payload
exactly. Its timing receipt is `workspace/runs/m98_resnet8_timing.json`:
134 additional requests on two images took 17.956 s after first progress;
total 38.153 s, peak RSS **2,472.8 MiB**. Verification/materializing the
source mask and eligible identities dominate initialization memory; reserve
**3–4 GiB**, not the initial 1–2 GiB estimate.

A conservative input-linear extrapolation from this small run gives about
74 minutes selection and 220 minutes final at the request upper bound, roughly
**five CPU hours** plus setup. It scales fixed per-state overhead as if it
were per-image work and may substantially overpredict; it is neither a fitted
performance model nor a guarantee. Machine contention also matters. Engineering
evidence occupies 73 MiB; allow **1–3 GiB** for production exact event/context
records and reports, with further growth at higher BER/field counts.

The old KWS schema-6.4 engineering final was replayed into a new M98 directory.
It exactly retained historical payload
`5b4efe74ccf9b81f9e685b1203d88168d9a4ba7acb2054faedf12d857b0d3556`.
The source bundle/checksum inventory was verified unchanged. Replay evidence
is under `workspace/runs/m98_kws_compatibility/`; no historical file was rewritten.

## ToyCar audit and next implementation boundary

Pinned model SHA-256:
`87cf24194ef93d1d9b11a591d805526b98008e351655d29883c825c9c106ba24`;
canonical AD lock digest
`sha256:2e362d617a104ca3749d47e1fd6a24ba0558b2be996bd16fb2fdd3cefa499b8c`.
Population remains 2,113,536 kernel bits; INT32 biases excluded. Existing
TF2.14.1 inference, float-domain target reconstruction, two-thread preprocessing
gate, raw float64 score digest and exact restoration are unchanged.

For a fixed set of complete files and labels, let `s_f(F)` be each file's
mean reconstruction MSE over **all** its windows under persistent fault set F.
Then task risk is `R(F)=1−AUC({s_f(F)},labels)`, with half credit for ties.
This is a scalar set function. Exact Bernoulli response-table integration and
the paired-swap telescoping identity hold for it without requiring additivity
over files. All normal/anomalous ordering interactions are retained. In
general `E[AUC(scores(F))] != AUC(E[scores(F)])`.

The synthetic test includes beneficial faults and pure higher-order score
changes. A bounded pinned test enumerates sign/LSB candidate states on already
exposed complete files 1 and 141, retains every reconstruction window, and
compares response integration with direct protected XOR evaluation. It also
checks exact state restoration/events. This validates the task-level identity,
not an inference-quality result on a meaningful anomaly cohort.

Production blockers are explicit:

1. The existing ranking cohort is development-train normal files; it cannot
   define ROC-AUC. Empirical and final cohorts are disjoint parity-stratified
   halves of the 248 official files. The canonical finalized estimator study
   already used 64 files in each half; union 128 recorded file identities.
   These cannot be relabeled untouched. A complete cross-worktree file exposure
   audit and prespecified both-class model-selection/final cohorts are needed.
2. Existing schema-6 sample bookkeeping assumes one input row/label per sample.
   ToyCar arrays contain 196 windows/file with reconstruction targets and local
   file IDs. A file-level identity hook must hash ordered complete windows,
   labels and mapping, never zip windows with file indices.
3. Existing cross-pool optional diagnostics use additive per-input CE and
   verify its mean against scalar risk. No per-file AUC-loss vector can replace
   that invariant. Add a separately versioned scalar-cohort risk contract and
   field/pool-only inference first. A later stratified whole-file bootstrap
   would recompute AUC from paired saved scores; it must not resample window
   losses or present pairwise comparisons as independent files.
4. Risk names, units and reports must explicitly be `1−file_ROC_AUC`, with
   reconstruction MSE separate. Never feed MSE into a CE column, or claim the
   S5 reconstruction objective directly optimizes AUC. A frozen S5 source can
   be reused as a surrogate-based baseline only with this alignment caveat.

No new ToyCar production adapter registration or partition choice is made in
this milestone. Required follow-up is this bounded file-cohort/schema gate,
not new ranking theory, retraining, or a new optimizer.

## Researcher commands and export

Run from the isolated worktree in a foreground tmux session:

```bash
cd /projects/netsurf/manuelbv/workspace/implementation-worktrees/m98
/projects/kona-assets/manuelbv/.venv-mlperf/bin/python -m netsurf campaign validate \
  workspace/runs/m98_resnet8_cross_pool_k6_v1/campaign.json
```

Only the researcher starts these production phases after reviewing exposure,
cost and the modest pilot design:

```bash
/projects/kona-assets/manuelbv/.venv-mlperf/bin/python -m netsurf campaign run \
  workspace/runs/m98_resnet8_cross_pool_k6_v1/campaign.json --phase selection
/projects/kona-assets/manuelbv/.venv-mlperf/bin/python -m netsurf campaign seal \
  workspace/runs/m98_resnet8_cross_pool_k6_v1/campaign.json
/projects/kona-assets/manuelbv/.venv-mlperf/bin/python -m netsurf campaign run \
  workspace/runs/m98_resnet8_cross_pool_k6_v1/campaign.json --phase final
/mnt/raid0/ml/conda_envs/manuelbv/envs/netsurf/bin/python -m netsurf campaign aggregate \
  workspace/runs/m98_resnet8_cross_pool_k6_v1/campaign.json \
  --output workspace/runs/m98_resnet8_production_analysis_v1
```

Operational `--pools pool_000 pool_001` batches are supported; sealing requires
every pool. `--no-progress` is available. Completed receipts resume after
verification; interrupted attempts fail closed and require inspection.
All exact static baseline/source inputs in the template must remain accessible.
No dependency installation or modern-TF substitution is needed.

To change the design **before outcomes**, edit a copy of the request/template
and freeze to a new workspace destination using `campaign freeze REQUEST
--output NEW_DIRECTORY`. Do not edit/regenerate this frozen plan. Its final
TOMLs are generated by `campaign seal`, not hand-bound to a guessed selection.
Direct Python APIs remain `load_conditional`, `execute_conditional_damage`,
`create_conditional_bundle`, `freeze_campaign`, `seal_selections`,
`run_campaign`, and offline `aggregate_campaign`.

```bash
# Read-only scientific verification; no model execution:
/projects/kona-assets/manuelbv/.venv-mlperf/bin/python -m netsurf campaign validate \
  workspace/runs/m98_resnet8_engineering_v1/campaign.json

# New derived report destination; no TensorFlow evaluation:
/mnt/raid0/ml/conda_envs/manuelbv/envs/netsurf/bin/python -m netsurf campaign aggregate \
  workspace/runs/m98_resnet8_engineering_v1/campaign.json \
  --output workspace/runs/m98_resnet8_engineering_analysis_user
```

Export only new implementation/plans/evidence, no models/datasets or historical
results. A delivery archive is created at `workspace/exports/m98_handoff.zip`.
It includes a source archive, this report/configs, the frozen unexecuted plan,
engineering evidence/plots and KWS compatibility replay. External source S5
and asset dependencies are referenced, not redistributed.

```bash
sha256sum workspace/exports/m98_handoff.zip
# On the Mac, replace KONA_SSH_HOST with the actual configured alias:
scp manuelbv@KONA_SSH_HOST:/projects/netsurf/manuelbv/workspace/implementation-worktrees/m98/workspace/exports/m98_handoff.zip ~/Downloads/
shasum -a 256 ~/Downloads/m98_handoff.zip
unzip -l ~/Downloads/m98_handoff.zip
```

No new package is required. Optional generic `jsonschema` is absent; strict
executable Python validators and rejection tests are active. The pinned runtime
has no Matplotlib; plots use the already installed canonical environment.

## Validation and changed files

All tests run from the M98 worktree. No production campaign phase was run.
The complete canonical regression suite passed **333 tests, 15 skipped**, in
85.97 seconds. Skips are explicit pinned-backend/preprocessing requirements,
three optional generic-JSON-schema checks, and optional local/upstream evidence.
Five warnings concern the modern interpreter's existing Lite interpreter
deprecation, not use of that interpreter for new authoritative experiments.

The pinned MLPerf/backend/conditional suite passed **125 tests, 5 skipped**,
in 230.50 seconds. The skips are three absent optional `jsonschema` checks and
two optional worktree-local historical evidence checks. Real KWS, ResNet and
ToyCar adapter/conformance tests all ran. Five existing dependency-deprecation
warnings remain visible. No environment was changed to eliminate warnings.

```bash
/mnt/raid0/ml/conda_envs/manuelbv/envs/netsurf/bin/python -m pytest -q -rs --tb=short
/projects/kona-assets/manuelbv/.venv-mlperf/bin/python -m pytest -q -rs --tb=short \
  tests/test_mlperf_tiny_adapter.py tests/test_m93_multibench.py \
  tests/test_m95_conditional_damage.py tests/test_m95_restricted_subcube.py \
  tests/test_m953_same_distribution.py tests/test_m955_paired_swap.py \
  tests/test_m956_preflight.py tests/test_m957_replication.py \
  tests/test_m96_cross_pool.py tests/test_m97_multiber.py \
  tests/test_m97_capabilities.py tests/test_m98_conditional_adapters.py
```

New regression coverage includes disjoint role windows and official-subset
identity aliases, exposure-payload corruption, allowed same-campaign reuse,
selection-only sample access, immutable selection/final identity, context
cardinality, joint conditional pairing/event counts, repeated scientific hashes,
large exact-event CSV fields, CLI JSON output, exact real ResNet sign/LSB XOR,
full suppression/restoration, captured input losses and authoritative output
conformance. ToyCar tests retain beneficial and higher-order AUC responses;
the real test uses two complete, already exposed files and twelve evaluations.

The existing M97 capability test intentionally changes only ResNet's expected
conditional readiness from BLOCKED to VERIFIED; no reference outputs, storage
populations, asset pins or conformance thresholds were relaxed. All five new
finalized bundles and both frozen campaign inventories pass checksum validation.
The engineering runs retain their original pre-commit source/dirty-state
provenance; they were not rewritten to refer to a later commit.

| Area | Changed paths |
|---|---|
| Capability integration | `netsurf/conditional_benchmarks.py`, `netsurf/benchmarks/capabilities.py` |
| Existing orchestration | `netsurf/conditional_config.py`, `netsurf/conditional_damage.py`, `netsurf/conditional_bundle.py`, `netsurf/selection_preflight.py`, `netsurf/cross_pool.py` |
| Offline/CLI | `netsurf/cross_pool_analysis.py`, `netsurf/__main__.py` |
| Schemas | `netsurf/schemas/conditional/v1_7.schema.json`, `netsurf/schemas/campaign/v1_1.schema.json`, `netsurf/schemas/README.md` |
| Editable plans | `config/examples/m98_resnet8_selection_template.toml`, `m98_resnet8_cross_pool_k6.toml`, `m98_resnet8_engineering.toml` |
| Tests | `tests/test_m98_conditional_adapters.py`, `tests/test_m97_capabilities.py` |
| Documentation | this report, `docs/SCIENTIFIC_CONTRACTS.md`, `docs/ARCHITECTURE.md`, `docs/REPRODUCIBILITY.md`, `docs/M97_BENCHMARK_ADAPTER_STATUS.md` |

No source changes to the XOR engine, restricted response-table mathematics,
paired-swap estimator, FKeras scores, M97 multi-BER engine, M96 portfolio,
historical candidate pools, source S5 allocation or asset locks are included.
