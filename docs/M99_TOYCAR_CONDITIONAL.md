# M9.9 — Complete-file ToyCar conditional protection

## Delivery and scientific status

Implementation branch/worktree: `m99-toycar-conditional`, at
`<repo>/workspace/implementation-worktrees/m99-toycar-conditional`.
Base: `m98-conditional-adapters`, `6b184e0867f8c51799bb0a588c9d3f2171f6b5ed`.
Implementation commit: `8209475` (documentation follows separately).
No merge, push, installation, retraining, new optimizer, asset replacement or
production campaign execution was performed. The original M98 worktree and its
ResNet campaign were not modified. Historical evidence remains in place.

**Ready for prespecified complete-file engineering/pilot studies**, not an
efficacy claim. Schema 6.8 and campaign 1.2 reuse the existing exact selector,
Bernoulli backgrounds, fixed-budget exchanges and paired-swap evaluator.
The authoritative risk is exclusively `one_minus_file_roc_auc`.

| Capability | Evidence/status |
|---|---|
| Pinned ToyCar reference / exact INT8 kernel XOR | VERIFIED; unchanged TF 2.14.1 BUILTIN_REF and two-thread preprocessing gate |
| Complete-file nonadditive risk | VERIFIED; all windows, one score/file, tied-score half credit |
| Restricted selection, frozen final comparison | VERIFIED; two K=6 engineering pools, sealed before final execution |
| Independent candidate pools / whole-field statistics | VERIFIED infrastructure; tiny smoke is not powered inference |
| KWS / ResNet compatibility | Exact historical engineering payload hashes reproduced |
| Production plan | FROZEN metadata only; no selection/final outcomes accessed |
| Generalization to unseen files / new devices / checkpoints | NOT ESTABLISHED |
| Input-population confidence intervals | NOT IMPLEMENTED; explicitly excluded, not approximated by additive losses |

## Model, data and protection identities

The pinned model is
`/data/models/models--mlcommons--tiny/snapshots/4addd0fa08d216e20637637874e084895f289da4/benchmark/training/anomaly_detection/trained_models/ad01_int8.tflite`.

| Identity | Value |
|---|---|
| Model SHA-256 | `87cf24194ef93d1d9b11a591d805526b98008e351655d29883c825c9c106ba24` |
| Asset lock | `<assets>/locks/mlperf-tiny/ad.lock.json` |
| Lock file SHA-256 | `7fafb690025f40903d92fecbd88adc1c5ebab2bd41626073ed82174d1ce12a84` |
| Canonical lock digest | `sha256:2e362d617a104ca3749d47e1fd6a24ba0558b2be996bd16fb2fdd3cefa499b8c` |
| Dataset root | `/data/datasets/dcase2020-toycar` |
| Official index | `benchmark/evaluation/datasets/ad01/y_labels.csv` |
| Index SHA-256 | `4ecd91868e197ee0a6739d4bd7abde73eac2fa31e9a88d6bc6aedefa136ff2a4` |
| Stored eligible population | 10 INT8 kernel tensors; 264,192 words; **2,113,536 bits** |
| Fixed S5 12.5% budget | **264,192 protected bits** |

INT32 biases, affine scales/zero points, activations and other state remain
outside the injectable population. Allocated zero codes remain eligible.
XOR acts on the actual stored INT8 code; sign is two's complement, bit 0 LSB.
Multiple events in one word compose through XOR. The inherited adapter patches
private TFLite bytes, keeps that state for the entire cohort, and restores in
`finally`. No gradient/surrogate computation is involved in measured task risk.

The verified source mask is **replication 0**, selected by prespecified identity,
not by comparing replication performance:

```text
<repo>/workspace/runs/m93_toycar_static_ber_pilot64_parallel_repair_v1/20261029T234704.724222Z-3913968c
artifact: artifacts/protection_masks.json
key: replication_0/per_example_absolute_global@0.125000
artifact SHA-256: 16855e9b4464b4f9b6845d59cd39e0cb1b1949a1e15db8363fa350efb2e23684
scientific payload: 5d9c1eaa69e24b5f2139b7bf69ae0d775cc8727d458d0b66c5cfe897ab44b554
```

The 516,836,798-byte artifact, finalized marker, source payload, checkpoint,
cardinality, unique eligible identities and current adapter population were
checked. A bounded streaming reader extracts only the selected mask rather
than materializing every policy/replication. It does not rewrite the source.
This S5 ranking used normal **development-train** files 0–15, not the proposed
final cohort. Its differentiable objective is reconstruction MSE, not AUC;
that objective mismatch is a scientific feature of this comparison, not a
reason to redefine S5.

## Authoritative complete-file risk and evidence

The existing pinned preprocessing is reused: 128-bin log-mel features,
five-frame concatenation, 196 windows of 640 features per file, fixed INT8
input encoding, and float64 reconstruction targets. The scoped BLAS thread
limit remains **two**, as required by the repaired conformance gate.

For each fault state, the existing adapter evaluates **every window of every
selected file**, decodes exact INT8 outputs, computes the mean squared error
per window, then the pinned mean across all windows of each file. These file
scores are ingredients of AUC, not additive task losses. For normal scores
`n` and anomalous scores `a`:

`AUC = mean(1[a > n] + 0.5 * 1[a == n]); risk = 1 - AUC`.

The normal/anomalous pair matrix is descriptive. Its entries share files and
are not independent observations. No window-level AUC, average of per-file
AUCs, mean-MSE proxy or categorical-cross-entropy column is substituted.

`artifacts/cohort_scores.json` (independent schema 1.0.0) records:

- Original official indices, source split, WAV filenames/content hashes and labels.
- All 196 ordered window identities per file and input/target hashes.
- Exact cohort identity, source-index hash and window/file counts.
- For every actually evaluated state: schedule/protection hashes, raw file scores,
  float64 hexadecimal values and little-endian score digest, exact integer-output
  digest, AUC/risk and event accounting.
- Clean-relative concordant/discordant/tied pairs and strict reversals. Reversals
  are not treated as independent sample counts or a new optimization metric.

Scores and diagnostics come from the same authoritative calls, not a second
model execution. One additional clean cohort evaluation per phase/pool supplies
the diagnostic reference. Selection-only loads no final cohort and generates
no final fields. Frozen selection contracts include the explicit cohorts.

`cohort_analysis.verify_score_evidence` reproduces risk and verifies score
representations offline. Bundle verification checks the artifact/payload hashes.
The existing response CSV and paired-step CSV retain pairing keys and exact
outside/master faults. Naturally unflipped coordinates are not assumed to be
protected under arbitrary non-Bernoulli laws.

## Exact identities and uncertainty

The response table evaluates all `2^K` candidate states under each independent
outside field. Exact Bernoulli integration retains every candidate interaction
order. Each eligible candidate mask has two protected bits; the fixed outside
context has 264,190, preserving **264,192 globally** for all comparators.

Both `V_b = E[R(G+b)-R(G)]` and the paired-swap telescoping identity
`J(target)-J(baseline) = p E[R(G+removed)-R(G+added)]` apply to a scalar cohort
set function; they require no additive loss decomposition. Synthetic complete-
cohort score tests cover beneficial faults, ties, strict reversals and pure
higher-order responses, comparing every equal-budget mask and paired path.

Outside fields are independent Bernoulli on allowed coordinates. Enumeration
forces candidate states; these rows are not claimed to be naturally sampled
fault realizations. Context-suppressed coordinates are excluded by design;
reported proposed/applied/suppressed counts describe the actual conditional
evaluations. Separate direct XOR tests verify nonzero suppression and full
protection. No fault distribution or suppression guarantee was changed.

Final comparisons are prespecified, paired within each whole master field:

1. Restricted-selected minus original S5 — primary.
2. Restricted-selected minus exact isolated — secondary.
3. Exact isolated minus original S5 — secondary.

Pool reports retain marginal Student-t field intervals and the existing
three-contrast Bonferroni intervals. Campaign 1.2 averages whole fields within
each pool, then gives equal weight to every prespecified pool. Primary pool-t
inference conditions on the shared complete-file cohort, checkpoint and selection
protocol. The two secondary pool tests use Holm; secondary simultaneous mean
intervals use 97.5% confidence. No pooled field×file or comparison-pair count
inflates the sample size. Degenerate nonzero-mean/zero-SE p-values remain null.

Input-population uncertainty is deliberately **not estimated**. A future such
analysis must resample complete normal/anomalous files within class, reuse the
same draws across paired states, recompute each AUC and then recompute the
whole paired-swap estimator. Averaging per-window or per-file contributions
would be invalid. Campaign bootstrap settings are retained reserved fields;
this scalar version does not invoke the classification input bootstrap.

## Exposure inventory and frozen cohort plan

The initial audit found nine pre-M99 finalized ToyCar bundles across canonical
and implementation worktrees. They used **128 distinct official files**:
normal indices **0–63**, anomalous indices **140–203**. The 320-candidate empirical
study used the even within-class positions; repaired static studies used the
odd positions. The repaired two-replication source used the same 64 final files
in both replications, not 128 independent files. Development-train ranking
indices are separately recorded and are not official-test exposures.

All **248 official files** already underwent clean-reference evaluation,
including the approximately 0.862037 full-cohort AUC. Consequently, nothing
remaining is inference-naive. The available distinction is **no recorded
fault-outcome exposure**, not an independently collected dataset. Remaining:
**76 normal (64–139), 44 anomalous (204–247)**. Current engineering replays use
only previously exposed indices and do not reduce that remainder.

The finalized metadata-only inventory is
`workspace/runs/m99_exposure_inventory_v1/`, payload
`2204bd1f858ae10b7dfd4f0f6f90e6e4d1d397df290a33f3afcb6314404070f9`.
It lists source bundles, checksums, role-specific indices and source hashes.
The known-root scan cannot prove absence of manual, incomplete, concurrent or
out-of-root use. Disclose such exposure and rerun the audit before executing.

| Role | Original normal indices | Original anomalous indices | Files | Status |
|---|---|---|---:|---|
| Engineering selection | 0 | 140 | 2 | Previously exposed |
| Engineering final | 1 | 141 | 2 | Previously exposed; also conformance gate |
| Reserved calibration metadata | 2 | 142 | 2 | Previously exposed; unused by separated modes |
| Proposed selection pilot | 64–75 | 204–215 | 24 | No recorded prior fault outcome |
| Proposed independent final | 76–99 | 216–239 | 48 | No recorded prior fault outcome |

The official index is class-ordered (140 normals then 108 anomalies); templates
interleave explicitly listed normal/anomaly identities. Contiguous within-class
selection does not establish exchangeability, identical distributions or device
generalization. No new scores were inspected to choose these cohorts. The final
48-file cohort has 24×24 dependent comparison pairs, not 576 independent inputs.
Four pools and 32 final fields per pool are a pilot budget, **not a power claim**.
Additional independent labeled ToyCar recordings/device conditions would be needed
for stronger external confirmation. Do not create synthetic replacement labels.

## Bounded engineering evidence

Frozen campaign: `workspace/runs/m99_engineering_campaign_v1/campaign.json`, SHA
`74a0b8edf171a368258a75749c2e50f9855a4e64aba913d6cbe5fae30ec960eb`.
Two independent K=6 pools, two selection fields per pool (two seeds), two final
master fields per pool, BER 0.001. All 15 two-bit candidate masks are integrated.
The source S5 context and all selected masks were sealed before final use.

| Pool / phase | Scientific payload SHA-256 | Actual cohort evaluations | Windows, excluding load gate |
|---|---|---:|---:|
| 000 selection | `9f4fbfb8f38279216a75f74cefce45303d6c6e5890e5b76658acce48af76efaa` | 136 | 53,312 |
| 001 selection | `fe871cd9583cb15291af4caaf0fecf75d8a57f683959a642a5472bc5b3d02099` | 136 | 53,312 |
| 000 final | `2c5204e1e8a5616883120687a5d6dc0052e512edfe2a2c362f562f2ae2103f75` | 5 | 1,960 |
| 001 final | `f9b097b2ac04cd1f754e2b263a34d506efd7db93977f3846dabbddca7f319473` | 1 | 392 |

Actual: **278 cohort evaluations, 556 evaluated files, 108,976 window inferences**,
plus **1,568 conformance windows** across four adapter loads. Selection includes
256 enumerated candidate states, 14 isolated/clean requests and two additional
clean cohorts. Final reuses identical states/masks: pool 001 needs no faulted
calls because all comparators coincide. Zero-length paths are not invented
independent model evaluations; field records still identify the planned fields.

Pool 000 original indices `[1,2]`, isolated/selected `[0,1]`: its selection
responses tie in AUC, so this mask change is deterministic tie-breaking, not
evidence of improved protection. Pool 001 original/isolated/selected `[0,3]`:
33 candidate states reverse the single normal/anomaly ordering; the chosen masks
coincide. All observed paired final differences are zero. Tiny 1+1 cohorts and
two fields cannot establish equivalence or superiority. Synthetic tests, not
this smoke, establish beneficial-fault and higher-order algebra coverage.

The pinned two-file gate reproduces file scores
`10.168487922577622`, `11.777840215450889` and exact score digest
`cd135be4abd970a9856891741a8fb09b7f67af6bd0552a9c23be28c848ef9fb1`.
The same state persists through all 392 windows. Full protection recovers the
clean scores; repeated faults replay exactly; XOR and state restoration pass.

Derived aggregate: `workspace/runs/m99_engineering_analysis_v1/report.md`, with
`pool_effects.png`; payload
`fe52983f3878bd6fc37e1c67034213fb5845f64015a9ab94ed268e5028140bd5`.
Regeneration uses a separate destination, no TensorFlow and no result mutation.
The committed implementation replayed pool 000 selection into the new bundle
`committed-replay-8209475` with exactly the same scientific hash. Independent
offline regeneration in `workspace/runs/m99_engineering_analysis_replay` likewise
reproduced the aggregate digest exactly; original receipts were not replaced.

## Computational cost and production plan

Measured on this workstation, with existing workloads left alone:

- Two-file preprocessing: 0.038 s; warm 392-window evaluation: 0.0375–0.0379 s.
- Initial pinned adapter load in the isolated timing check: 11.1 s.
- Selection phase: 271.44 s wall time, 1,538,196 KiB peak RSS (~1.47 GiB).
- Final phase: 264.55 s wall time, 1,540,524 KiB peak RSS (~1.47 GiB).
- Individual engine runtimes: selection 121.77 / 120.54 s; final 115.56 / 115.26 s.
- Historical mask parsing, integrity checks and 2.1-million-bit population
  bookkeeping dominate this tiny smoke. Inference-only extrapolation would be
  misleading. The initial character-stream baseline took 209.7 s to read the
  full mask archive; the implemented buffered tokenizer reduces that overhead
  while retaining full-file checksum verification and bounded memory.
- Engineering campaign evidence occupies approximately 241.5 MB uncompressed,
  primarily repeated explicit original/outside mask identities. No scientific
  fault identities or responses were compressed away in the bundle format.

Researcher-editable request: `config/examples/m99_toycar_cross_pool_k6.toml`,
source/cohort template: `config/examples/m99_toycar_publication_template.toml`.
The already frozen, **unexecuted** plan is
`workspace/runs/m99_toycar_production_plan_v1/campaign.json`, SHA
`1a5965251494a108e0903f99a629df2b236480dae4a1aad242d96c83a3e4f802`.

Design: four pools sampled uniformly within S5-protected/unprotected strata,
two protected + four unprotected candidates/pool, K=6, global budget 12.5%,
BER **0.001**, eight selection outside fields/pool (two seeds × four fields),
32 independent final master fields/pool. Selection/master/pool seeds are all
explicit, with derived-seed collision rejection. Every pool is retained.
The sampling design is conditional on S5 status, not uniform over all six-bit
subsets or an unbiased estimate of whole-network optimality.

| Planned quantity | Count |
|---|---:|
| Candidate states per outside field | 64 |
| Eligible two-bit masks per pool | 15 |
| Selection outside fields | 32 total |
| Enumerated response-table states | 2,048 |
| Selection state requests (isolated controls included) | 519/pool |
| Additional clean cohorts | 1/phase/pool |
| Selection evaluated files | 49,920 |
| Selection window inferences | 9,784,320 |
| Final independent master fields | 128 total |
| Final state requests | At most 384/pool before exact reuse |
| Final evaluated files | At most 73,920 |
| Final window inferences | At most 14,488,320 |
| Load-gate window inferences | 3,136 |
| Total window upper bound | **24,275,776** |
| Analytical candidate-mask integrations | 480 |

Warm inference alone extrapolates to roughly 39 minutes at the upper bound.
Allow approximately **50–75 minutes** including eight verified loads, population
bookkeeping, XOR/restoration, hashing and serialization; actual frozen paths can
reduce final cost substantially. This is an extrapolation, not a timing guarantee.
Budget roughly **2–3 GiB RAM** per sequential process and **0.5–1.5 GB** output
headroom; more fields/pools increase exact response and event records. Window
inference scales as files × 196 × states; selection scales exponentially with K.

BER **0.01** is a separately planned exploratory stress regime, not part of
this frozen primary plan. It implies about 21,135 whole-population proposed
flips (versus 2,114 at 0.001) before suppression; allowed outside faults depend
on the fixed context. It increases event/patching costs roughly with event count,
not window count. If reusing the primary final files, label the second study
exploratory (`purpose="engineering"` is the current non-confirmatory execution
label), use new explicit field/pool seeds, and freeze a distinct destination.
Do not advertise the same final cohort as untouched confirmation again. No
0.01 campaign was frozen or executed here.

## Configuration/API contract and foreground commands

All commands below start in the M99 worktree. Use explicit interpreters and
`python -m netsurf` so the editable package in another worktree cannot silently
run the wrong source. No `proj` changes or installation are required.

New schema 6.8 keys:

- `schema_version="6.8.0"`, `scientific_protocol="scalar-cohort-conditional-protection/v9"`.
- `[benchmark] adapter="mlperf_tiny_ad_int8_v1"`,
  `id="mlperf_tiny_ad_toycar_int8_v1"`, `split="official_248_files"`;
  existing absolute checkpoint/dataset/asset_lock fields and frozen backend/INT8
  quantization fields, `injectable_parameters=["kernels"]`.
- `[risk] metric="one_minus_file_roc_auc"`, `direction="minimize"`.
- `[cohorts] conditional_calibration`, `conditional_model_selection`,
  `conditional_final_evaluation`: explicit unique original index arrays.
  `[samples]` role/count fields must match; all offsets zero; disjointness true.
- Existing `[backgrounds]`, `[restricted_oracle]`, `[protection_context]`,
  `[paired_swap_evaluation]`, `[execution]` and `[output]` fields retain meanings.
- `[workflow] mode="selection_only"` or `"final_only"`, existing purpose,
  candidate provenance, source bundle/hashes and campaign/sealed-registry bindings;
  `record_per_input=false` is mandatory for bound campaigns and true is rejected.
- Campaign `schema_version="1.2.0"` otherwise retains the 1.1 request tables:
  `design.candidate_count`, `pool_seeds`, `selection_seed_bases`,
  `selection_replicates`, `selection_fields_per_seed`, `final_seeds`,
  `final_fields`, `logical_ber`; zero-offset sample counts must match template cohorts.

```bash
cd <repo>/workspace/implementation-worktrees/m99-toycar-conditional

# Validate frozen plans; no model inference:
<assets>/.venv-mlperf/bin/python -m netsurf campaign validate \
  workspace/runs/m99_toycar_production_plan_v1/campaign.json

# Metadata-only exposure snapshot into a NEW destination:
<netsurf-env>/bin/python -m netsurf.toycar_exposure \
  --output workspace/runs/m99_exposure_user

# Researcher launches these deliberately, in tmux/foreground:
<assets>/.venv-mlperf/bin/python -m netsurf campaign run \
  workspace/runs/m99_toycar_production_plan_v1/campaign.json --phase selection
<assets>/.venv-mlperf/bin/python -m netsurf campaign seal \
  workspace/runs/m99_toycar_production_plan_v1/campaign.json
<assets>/.venv-mlperf/bin/python -m netsurf campaign run \
  workspace/runs/m99_toycar_production_plan_v1/campaign.json --phase final
<netsurf-env>/bin/python -m netsurf campaign aggregate \
  workspace/runs/m99_toycar_production_plan_v1/campaign.json \
  --output workspace/runs/m99_toycar_production_analysis
```

To change the design **before outcomes**, edit copies of the two templates and
freeze to a new directory with `campaign freeze REQUEST --output NEW_DIRECTORY`.
Do not edit the frozen plan or hand-bind guessed selection paths. `campaign seal`
requires every selected pool artifact before generating final TOMLs. Interrupted
attempts fail closed; do not delete attempt records to bypass exposure auditing.

Python APIs: existing `load_conditional`, `execute_conditional_damage`,
`create_conditional_bundle`, `freeze_campaign`, `run_campaign`, `seal_selections`
and `aggregate_campaign`. New offline helpers are `cohort_risk`,
`ordering_diagnostics`, `verify_score_evidence`, `scalar_pool_statistics`, and
`toycar_exposure.inventory`. No TensorFlow is required for reports/plots.

## Tests, compatibility and environment limitations

Full canonical suite: **346 passed, 16 skipped** (88.07 s initial complete run;
84.56 s committed-code repeat).
Pinned backend/conditional suite: **139 passed, 5 skipped** (314.78 s).
Focused post-change pinned checks passed **18 tests** (26.14 s). Exact commands:

```bash
<netsurf-env>/bin/python -m pytest -q -rs --tb=short
<assets>/.venv-mlperf/bin/python -m pytest -q -rs --tb=short \
  tests/test_mlperf_tiny_adapter.py tests/test_m93_multibench.py \
  tests/test_m95_conditional_damage.py tests/test_m95_restricted_subcube.py \
  tests/test_m953_same_distribution.py tests/test_m955_paired_swap.py \
  tests/test_m956_preflight.py tests/test_m957_replication.py \
  tests/test_m96_cross_pool.py tests/test_m97_multiber.py \
  tests/test_m97_capabilities.py tests/test_m98_conditional_adapters.py tests/test_m99_cohort.py
```

The canonical suite uses a read-only symlink to the existing MNIST checkpoint
under this worktree's ignored `workspace/benchmarks/`; it does not retrain or
copy that checkpoint. Optional `jsonschema` and some local upstream/evidence
fixtures are absent. Strict executable Python validation/rejection tests remain
active. If generic JSON-Schema checks are desired, the researcher may install
`jsonschema>=4,<5` in the chosen environment; no installation was performed.
The pinned runtime lacks Matplotlib; offline figures use the canonical interpreter.
Existing TensorFlow/librosa dependency deprecations remain visible and unchanged.

Compatibility replays generated **new M99 bundles**, not changes to the originals:

- KWS schema-6.4 final payload:
  `5b4efe74ccf9b81f9e685b1203d88168d9a4ba7acb2054faedf12d857b0d3556`.
- ResNet schema-6.7 selection payload:
  `9efe2b2a39cdae684ea59d720be7932b6a1582fb41efc3a62b96973528667806`.

Both exactly match their historical M98 receipts, not merely aggregate metrics.
The prior tests expecting ToyCar conditional support to remain blocked were
deliberately updated; all other benchmark-family/precision capability gates remain.

## Files and export

Core additions: `adapters/toycar_conditional.py`, `cohort_risk.py`,
`cohort_analysis.py`, `toycar_exposure.py`, `mask_stream.py`. Narrow extensions:
conditional config/capability/engine/bundle, selection binding, existing campaign
freeze/verify/aggregate and large-mask reading. New schema definitions, four M99
templates and `tests/test_m99_cohort.py`; capability expectations updated only
for ToyCar's newly verified path. No estimator or optimizer definition changes.

Delivery ZIP: `workspace/exports/m99_handoff.zip`. It contains implementation
source/docs/tests/configs, the frozen unexecuted plan, new engineering bundles,
offline reports/plots, exposure inventory and compatibility replay receipts.
No dataset, audio, model, external source mask or historical bundle is copied.
External pinned inputs and S5 context remain referenced dependencies on Kona.

```bash
# Kona: verify and inspect the prepared delivery
sha256sum workspace/exports/m99_handoff.zip
unzip -l workspace/exports/m99_handoff.zip

# Mac: replace KONA_SSH_HOST with your actual configured SSH host/alias
scp <user>@<host>:<repo>/workspace/implementation-worktrees/m99-toycar-conditional/workspace/exports/m99_handoff.zip ~/Downloads/
shasum -a 256 ~/Downloads/m99_handoff.zip
```

No further optimization phase is implied. The next step is researcher-owned
review of the prespecified pilot budget, exposure disclosures and inferential
scope, followed by deliberate execution of the frozen selection/final plan.
