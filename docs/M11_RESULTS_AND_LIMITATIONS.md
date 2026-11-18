# M11 — Conditional response scaling: evidence and handoff

Date: 2026-11-02. Worktree:
`/projects/netsurf/manuelbv/workspace/implementation-worktrees/m11-conditional-scaling`.
Branch `m11-conditional-scaling`, based on
`10cb2f24d273cdba9c7a99e32ae207b278238f51`. Initial implementation commit:
`65bb127f5d72cc079b8d4d23fdbfa8efdb5819c0`. Follow-up reporting/provenance changes
are identified by the delivered Git log and export index. No merge or push.

## What has, and has not, been established

The candidate-forced-clean, outside-conditioned representation is implemented
with **K+1 exact cohort evaluations per outside field**, through K=64. It uses
the existing pinned TF 2.14.1 BUILTIN_REF adapters, stored INT8 XOR, ideal event
suppression, and complete-file ToyCar risk. It is not full conditional influence,
a new fault law, or an adaptive optimizer. See [the formal theory](M11_THEORY.md).

The proposed bounded-risk inequality is true and sharp. Restricted regret and
finite-field sampling guarantees are proved separately. Pure pairwise and
higher-order counterexamples disprove universal ranking optimality. A real KWS
K=8 audit also has nonzero selection regret. Thus success is not defined as
passing a smoke or finding only favorable pools.

The 56-pool historical reaudit, new exact K=6/8/12 audits, K=32/64 execution,
and frozen-mask replay establish reproducible measurement and bounded numerical
behavior. **They do not identify the empirical breakdown regime across
architectures or establish a protection improvement on a population.** Those
questions require the prespecified researcher-run studies below. No production
or new-confirmation campaign was executed for M11.

## Implementation and contracts

| Component | Responsibility |
|---|---|
| `netsurf/conditional_scaling.py` | Pure numerical representation, exact oracle, error/regret bounds, curvature audit, signed selection, paired uncertainty |
| `netsurf/scaling_study.py` | Strict request validation, source/S5 verification, frozen probability sample, independent streams, selection/seal/final lifecycle |
| `netsurf/scaling_reporting.py` | Verified saved-data reports and figures, no inference |
| `netsurf/scaling_historical.py` | Separate derived analysis of all M9.10 production sources |
| `netsurf/schemas/scaling/v1.schema.json` | Independent M11 request schema 1.0.0 |
| `tests/test_conditional_scaling.py`, `tests/test_scaling_lifecycle.py` | Mathematical and immutable-lifecycle regression tests |

Protocol identifier: `outside-conditioned-forced-clean/v1`. Existing experiment
schema versions, estimators, optimizers, fault algorithms and adapter numerical
implementations are unchanged. This orchestration reuses them; it does not
replace the static or conditional experiment engines. API: `load_request`,
`freeze`, `execute`, `seal`, `verify_directory`, `analyze`, `oracle_summary`.

Each outside field records exact fault identities, probability-law identity,
candidate responses, sample/context hashes and pairing keys. Every actual state
checks both clean model bytes and stored parameter values after restoration;
proposed = applied + suppressed is enforced. Whole-network cardinality is fixed.
Final full-network Bernoulli fields are shared across masks and independent of
selection fields. All pool/arm selections must finalize and seal before final
execution. Final evaluation cannot change candidates or masks.

Frozen plans/results are write-once and checksummed. Runtime and refreshed
exposure-audit metadata do not enter the scientific digest. ToyCar's complete
file scores are a separate, checksummed and scientifically bound artifact:
`artifacts/cohort_scores.json`. No score/window information is discarded to make
the scalar-cohort objective appear additive. The final loss bootstrap uses
complete files within class and recomputes AUC with half-credit ties.

## Populations and exposure normalization

All populations are stored INT8 kernels only. Biases, quantizer metadata,
activation storage and physically absent connections are not newly added.

| Model | Eligible bits N | Frozen S5 budget B | N−B | p for λ=10 | p for λ=100 | p for λ=1000 |
|---|---:|---:|---:|---:|---:|---:|
| KWS | 176128 | 22016 | 154112 | 6.4887874e-5 | 6.4887874e-4 | .0064887874 |
| ResNet-8 | 618880 | 77360 | 541520 | 1.8466539e-5 | 1.8466539e-4 | .0018466539 |
| ToyCar | 2113536 | 264192 | 1849344 | 5.4073228e-6 | 5.4073228e-5 | .00054073228 |

At common p=.003 the expected applied counts are respectively 462.336,
1624.56 and 5548.032. Equal BER is not equal total fault exposure. Equal λ is
only an exposure-normalization control, not equivalent perturbation severity,
physical radiation reliability or architecture. Never pool raw CE and AUC risk.

The original finalized source campaign, checkpoint, lock, whole-network S5 mask,
mask checksum, population and source payload are reverified at freeze and run.
Their paths are explicit in the TOMLs; no historical S5 location is invented.

## Sampling and statistical design

Primary pools use simple random sampling without replacement separately within
original S5-protected/unprotected strata. Candidate budget r=(K+1)//3; every
eligible bit has positive inclusion probability in its stratum. Exact π,
inverse-π weight, source population hash, RNG states, tensor/word/bit, original
status, encoded value, scale and zero point are retained. No damage-based
accept/reject step exists. Small pools need not contain every tensor or bit
tier; coverage is observable, not guaranteed. Joint mask performance is a
pool-design estimand, not an automatically unbiased bit-population estimate.

Secondary configurations restrict candidate support to structural bit tiers6/7.
They are separately labeled, not combined with the primary design or represented
as all-bit population evidence. The choice is prespecified, not outcome-selected.

Field streams are frozen from `SeedSequence(master, pool_seed, K, arm, phase)`.
Exact derived seeds and collision checks are in each operation record. Candidate,
outside-field, final-field and bootstrap seeds are separate. Shared cohorts
across pools do not constitute new independent input replicates.

Within each pool/arm, all frozen-mask pair differences use shared whole fields,
paired Student-t intervals and Bonferroni over the reported pair family. These
are approximate intervals, not finite-sample equivalence certificates. The
separate bounded-risk Hoeffding result does not assume independent bits' responses.
Equal-pool aggregation uses pool means, not pools multiplied by field counts.
Input/field crossed bootstrap draws are retained separately, conditional on
selected masks. Cumulative selection masks describe stability, not independent
mask-selection repetitions. Across-K/BER/benchmark comparisons remain exploratory;
there is no automatic familywise claim over the entire sweep.

All null/adverse pools are retained. Zero observed differences and degenerate
engineering intervals do not demonstrate equality or lack of interactions.

## Exposure audit and engineering cohorts

No new confirmation inputs were consumed. The exact role-specific indices,
ordered identities and sample hashes are in each finalized payload.

| Model | Selection engineering identities | Final engineering identities | Unit |
|---|---|---|---|
| KWS | original test2024,2025 | test3060,3061 | complete MFCC input/clip |
| ResNet-8 | test1000,1001 | test3000,3001 | image |
| ToyCar | files0,140 | files1,141 | normal + anomalous file, 196 windows each |

The adapter's inherited bounded conformance inputs are additionally recorded
(16 KWS clips,200 ResNet images,392 ToyCar windows); they are not new
independent fault trials. ToyCar preserves the established two-thread
preprocessing gate, exact score digest
`cd135be4abd970a9856891741a8fb09b7f67af6bd0552a9c23be28c848ef9fb1`.
No lock, tolerance or delegate was changed.

Known exposure roots are scanned at freeze and execution. Engineering requires
already-exposed identities; confirmation rejects overlaps. Results belonging to
the same frozen campaign may intentionally share its cohorts. This cannot prove
absence of unrecorded manual or concurrently created exposures elsewhere; the
researcher must freeze a complete external exposure ledger for confirmation.

The research templates deliberately reuse previously exposed cohorts and say
`purpose="exploratory"`: KWS selection2088–2151/final3450–3961; ResNet selection
1024–1055/final3225–3352. ToyCar copies its existing complete source cohorts:
selection normals64–75/anomalies204–215 and final normals76–99/anomalies216–239.
The prior ToyCar ledger has200/248 files fault-exposed, leaving40 normal and
8 anomalous files; all248 have clean exposure. M11 does not consume the remainder
or assert that it is an adequate confirmation cohort.

## Exact audits: numerical results, including failures

All following quantities come from finalized checksummed evidence, not invented
examples or timing extrapolations. `workspace/runs/m11_historical_reaudit` rereads
and verifies all56 historical pools without new inference or changing source
hashes. Its scientific digest is
`55d02a91f9d1382ae41b8a4484531795e9fcaecbebd694a59cad7bd9c5670aa7`.

| Source | p | Pools | Mean max outside-additive error | Mean outside selection regret | Mean clean-isolated selection regret |
|---|---:|---:|---:|---:|---:|
| M96 KWS | .01 |12|3.1862604e-6|0|9.771906e-5|
| M97 KWS | .001 |12|1.8459882e-8|0|4.3183715e-6|
| M97 KWS | .003 |12|4.1494309e-7|0|2.7949262e-5|
| M97 KWS | .01 |12|7.1622758e-6|0|1.750362e-4|
| M98 ResNet-8 | .01 |4|3.9408739e-5|0|.0019568009|
| M99 ToyCar | .001 |4|7.5961354e-9|2.1679709e-10|9.1095924e-6|

These are finite-selection-field restricted optima, not held-out protection
effects. KWS pool identities repeat across BER arms. The historical M9.10 final
contrasts remain the appropriate separate evidence; these rows cannot promote
an exploratory selection advantage into confirmation.

New bounded exact audits at p=.003:

| Model / K | Outside fields | Max outside error | Outside selection regret | Strict ordering reversals | Clean selection regret |
|---|---:|---:|---:|---:|---:|
| KWS6 |2|2.6785298e-8|0|0|5.9493061e-6|
| KWS8 |1|1.6173224e-7|3.5766981e-8|3|5.9941768e-6|
| KWS12 |1|2.7596937e-6|0|769|6.3128219e-5|
| ResNet6 |2|8.6098862e-7|0|0|4.0479843e-5|
| ToyCar6 |2|0|0|0|0|

KWS8 demonstrates a real, small wrong selection. KWS12 demonstrates that an
unchanged winner does not imply identical orderings. Ties and tiny contrasts
are retained, and centering-closure errors are recorded rather than using an
arbitrary tolerance to erase disagreements. ToyCar's two-file cohort is too
coarse for scientific conclusions. Absolute clean-additive risk error also
includes the changed background intercept; it is not solely a ranking error.
Different K pools/fields are not a controlled causal experiment on K alone.

The full bounded selection/final lifecycle completed K6/32/64 and two independent
fields per phase on each model:18 finalized real-model bundles, with every
restoration/event-accounting check passing. Every final within-field mask
contrast in these nine tiny cases was zero. ToyCar K6 had cohort risks1 and0
across the two fields; K32/64 had risk0 in both fields, identically across masks.
This is retained null evidence, not equality or absence of interactions. A direct
Bernoulli field often misses the few exchanged candidate coordinates. These
checks are not a replacement for a detection-budget study. For K>12 the report
deliberately has no exact-optimum/error/regret values.

Synthetic falsification includes pure third-order damage at K6,r2,p=.5 with
additive-selected regret .125, and an analytic K64 pair-interaction counterexample.
The sharp bound is independently checked over all Boolean K3 truth tables and
all budgets, plus exhaustive additive K6/8/12 tables and the existing oracle.

Saved oracle report/figures: `workspace/runs/m11_oracle_analysis_v2` and its
`-figures` directory; scientific digest
`cbb6e36fc7c3ab8b3b733468587aab2317af23b0649efd4cc1f62cb9201ffe4b`.
An earlier plot attempt failed after finalizing the numerical report; it remains
untouched at `m11_oracle_analysis`. The `_v2` delivery fixes plotting only.

## Cost: measured versus projected

Two-input engineering selection measurements (two outside fields):

| Model | K6 oracle calls / seconds | K32 linear calls / seconds | K64 linear calls / seconds | Initialization per run |
|---|---:|---:|---:|---:|
| KWS |136 /1.818|100 /1.156|196 /2.225|138–156s|
| ResNet-8 |136 /5.372|100 /3.562|196 /6.315|152–156s|
| ToyCar |136 /20.903|100 /12.273|196 /22.911|281–292s|

Calls include one clean-reference call and K+1 clean isolated calls. Each ToyCar
call is392 window inferences, not two windows. Times include fault application,
inference and restoration; additional restoration verification is reported
separately. KWS K8 oracle:266 calls,3.667s; K12:4110 calls,58.295s. Per outside
field, exact/linear call counts are64/7,256/9,4096/13 for K6/8/12. These ratios
are not end-to-end speedups; common clean work and substantial initialization
remain. Actual K32/64 measurements demonstrate linear-state execution.

Detailed manifests separate output evaluation, model apply/restore, background
generation, risk extraction, selection/approximation and verification. Example
KWS K12: output23.050s, apply.846s, restore.823s, outside generation.00684s.
Residual aggregation time includes schedule/encoding overhead and is not claimed
to be pure statistical aggregation. Repeated startup validates historical source
mask payloads and exposure records and dominates these tiny jobs. No safeguard
was bypassed for a speed claim.

A separate metadata-only freeze measured .327s for three KWS candidate pools
and identity annotation,74.561s for source/exposure verification. It is saved in
`m11_kws_sampling_cost_plan/execution.json`. Earlier plans did not retain that
timer; their manifests are not rewritten. New freezes record timing outside the
scientific plan. Peak process RSS was approximately0.6–0.75GiB for classification
and1.37GiB for ToyCar (process high-water marks, not incremental per-case usage).

Each full three-arm example contains48 pool/arm cases (4 sizes ×4 seeds ×3 arms):
410232 selection cohort calls and5424 final calls, **before inherited conformance
checks and source verification**. Per K/arm/pool selection counts are520,32782,
298,586. Final counts are129,129,97,97; identical masks are still explicitly
evaluated, not miscounted as independent observations.

| Model / one axis | Total network input/window evaluations | Rough compute planning range |
|---|---:|---|
| KWS |29031936|roughly one to two days|
| ResNet-8 |13821696|roughly one to two days|
| ToyCar |1980760320|roughly three days or more|

These are extrapolations from small concurrent-load measurements, not measured
campaign runtimes or promises. Output time scales with cohort size; startup and
encoding/aggregation add overhead. The K12 oracle dominates. Exact score/event
records and repeated source-context provenance can require gigabytes per ToyCar
axis; each bounded ToyCar bundle already carries roughly20MB of provenance.
Whole-cohort score capture grows with states × files, and JSON serialization
and bootstrap memory need a representative larger-cohort pilot before full runs.
No campaign was launched to obtain these estimates.

## Exact supported request fields

All keys below are required; extra/unsupported keys fail. Schema file and actual
validator are exercised by tests. TOML values select existing capabilities, not
unimplemented model/backend fallbacks.

| Section | Fields and accepted semantics |
|---|---|
| root | `schema_version="1.0.0"`, `name`, `purpose` engineering/exploratory/confirmation, absolute `source_campaign`, `source_campaign_sha256` |
| design | `candidate_counts` subset6,8,12,32,64; `pool_seeds`; `sampling` primary/structural_bit_tiers; `bit_positions` unique0..7 (primary must include all); `axis` ber/expected_applied_flips; positive `values`; distinct `selection_seeds`, `final_seeds`; `exhaustive_max_k`0..12 |
| samples | `selection_offset`, `selection_count`, `final_offset`, `final_count`; exact inherited role partitioning; ToyCar requires whole source cohorts/counts and zero offsets |
| statistics | `confidence`, `bootstrap_seed`, `bootstrap_replicates`>=2 |

All seeds, counts, masks, source identities and final comparisons must be frozen
before final execution. Changing a request requires a NEW frozen plan/output.
Resuming by silently selecting the successful cases or best seeds is prohibited.

Maintained examples under `config/examples/`: for each of `kws`, `resnet8`,
`toycar`, `m11_<model>_ber.toml`, `_expected_applied_flips.toml`, `_enriched.toml`,
and `_engineering.toml`; plus `m11_kws_oracle_engineering.toml`. Nine research
plans are frozen under `workspace/runs/m11_frozen_<model>_<axis>/plan.json`.
All nine freezes are metadata-only; their execution budgets are NOT authorization
to run them automatically.

## Foreground Kona commands

Run from this implementation worktree, not the active canonical research tree.
No installation, environment modification, background wrapper or automatic merge.

```bash
cd /projects/netsurf/manuelbv/workspace/implementation-worktrees/m11-conditional-scaling

# Fast validation and tests; canonical environment, not authoritative TFLite inference.
/mnt/raid0/ml/conda_envs/manuelbv/envs/netsurf/bin/python -m netsurf scaling validate config/examples/m11_kws_ber.toml
/mnt/raid0/ml/conda_envs/manuelbv/envs/netsurf/bin/python -m pytest -q

# Freeze an edited request to a NEW directory; metadata audit, not inference.
/projects/kona-assets/manuelbv/.venv-mlperf/bin/python -m netsurf scaling freeze config/examples/m11_kws_ber.toml --output workspace/runs/my_m11_plan

# Explicit single selection case (researcher launches after approving cost).
/projects/kona-assets/manuelbv/.venv-mlperf/bin/python -m netsurf scaling run workspace/runs/my_m11_plan/plan.json --pool k6_seed111101 --arm 0 --phase selection --output workspace/runs/my_m11_selection/k6_seed111101_arm0

# After ALL prescribed cases finish; refuses incomplete/duplicate selections.
/projects/kona-assets/manuelbv/.venv-mlperf/bin/python -m netsurf scaling seal workspace/runs/my_m11_plan/plan.json workspace/runs/my_m11_selection --output workspace/runs/my_m11_registry

# Frozen final masks only; does not rerun selection.
/projects/kona-assets/manuelbv/.venv-mlperf/bin/python -m netsurf scaling run workspace/runs/my_m11_plan/plan.json --pool k6_seed111101 --arm 0 --phase final --selection-registry workspace/runs/my_m11_registry/plan.json --output workspace/runs/my_m11_final/k6_seed111101_arm0

# After all finals, regenerate a NEW derived analysis without inference.
MPLCONFIGDIR=workspace/cache/matplotlib /mnt/raid0/ml/conda_envs/manuelbv/envs/netsurf/bin/python -m netsurf scaling analyze workspace/runs/my_m11_plan/plan.json workspace/runs/my_m11_registry/plan.json workspace/runs/my_m11_final --output workspace/runs/my_m11_analysis
```

The frozen `operation_counts` list is the authoritative queue. To run every
case sequentially in foreground, use the shared API, not a new research engine:

```bash
/projects/kona-assets/manuelbv/.venv-mlperf/bin/python - <<'PY'
from pathlib import Path
from netsurf.scaling_study import verify_directory, execute
root = Path('workspace/runs/my_m11_plan')
plan = verify_directory(root, frozen=True)
for op in plan['operation_counts']:
    execute(root/'plan.json', op['pool'], op['arm'], 'selection',
            Path('workspace/runs/my_m11_selection')/f"{op['pool']}_arm{op['arm']}")
PY
```

Use a fresh selection root if the single-case example was already run; output
overwrite is rejected. For final, use the same loop with phase`final`, a new
final root, and `selection_registry=Path('workspace/runs/my_m11_registry/plan.json')`.
Existing source campaigns, S5 artifacts, exact locked assets and pinned backend
must remain accessible. Moving only the ZIP cannot recreate those dependencies.

## Priorities, unresolved issues and publication scope

1. Review the sharp bound, counterexamples and real ordering reversals before
   making a representation claim. No theorem-priority claim is established.
2. Run the primary KWS same-BER and matched-λ plans first, retaining all pools.
   Review compute/storage and detection budgets before the costly K12 cases.
3. Run analogous ResNet/ToyCar plans to separate exposure from architecture,
   without treating different task units as directly comparable. Report coarse
   AUC/null/saturated regimes rather than dropping them.
4. Analyze independently paired frozen-mask outcomes, not selection regret as
   if it were a held-out improvement. For rare exchanged-bit hits,32 final fields
   may be inadequate; prespecify a larger precision budget in a NEW plan before
   final evaluation. The current engine does not implement a new guided sampler.
5. Only after exploratory results, define a separately exposure-audited
   confirmation protocol and adequate fresh inputs. Limited untouched ToyCar
   anomalies are an explicit blocker to broad confirmation claims.

Literature audit is in the theory and verified BibTeX. The supplied Zenodo record
is incorporated as a public companion dataset, not a verified peer-reviewed
manuscript. Its layer interactions are relevant precedents, not identical
encoded-bit Möbius effects. EISFINN full text remains inaccessible; its complete
weighting methodology is not claimed audited. No changes to FKeras definitions,
no MCMC/RL/importance sampling, no new estimator family or hardware mitigation.

## Verification and export

Canonical final suite:400 passed,16 skipped,5 upstream TFLite deprecation warnings
in88.42s. Pinned selected adapter/theory suite:92 passed,5 upstream deprecation
warnings in104.56s. Targeted M11 tests passed17/17 in3.76s. Existing
KWS/ResNet/ToyCar tests were not weakened. The16 canonical skips retain the
existing optional-integration/environment gates, not newly relaxed checks.

Pinned regression command:

```bash
/projects/kona-assets/manuelbv/.venv-mlperf/bin/python -m pytest -q tests/test_m93_multibench.py tests/test_mlperf_tiny_adapter.py tests/test_m98_conditional_adapters.py tests/test_m99_cohort.py tests/test_m910_attribution.py tests/test_conditional_scaling.py
```

KWS K32 final scientific digest repeated exactly:
`a010b2b458c55b1efe94cd7ef219fa0b8201035bcf14683f2b861a9b612ec28a`.
Primary evidence roots are `m11_<model>_validated_plan`, `_selection`, `_registry`,
`_final`, and `m11_<model>_analysis`; per-bundle manifests contain all hashes and
dirty-source fingerprints. Early `*_engineering_*` prototype outputs remain
untouched and are not release evidence. Some early validated KWS runs predate
the initial implementation commit; their exact source fingerprint is recorded,
and the final replay verifies numerical identity with the delivered implementation.

The handoff ZIP includes source, docs, tests, configs, frozen plans and new bounded
evidence/derived plots, plus a hash inventory. It excludes datasets, model binaries,
historical bundles, proprietary assets and the third-party source snapshot.
It is not a standalone redistribution of the research datasets or S5 archives.

```bash
# Kona, after delivery:
sha256sum workspace/exports/m11_handoff.zip
unzip -t workspace/exports/m11_handoff.zip

# To make another archive from the committed source and verified evidence:
PYTHONPATH=. /mnt/raid0/ml/conda_envs/manuelbv/envs/netsurf/bin/python workflows/export_m11.py --output workspace/exports/m11_handoff_copy.zip

# Mac; replace KONA_SSH_HOST with the researcher's configured SSH hostname.
scp manuelbv@KONA_SSH_HOST:/projects/netsurf/manuelbv/workspace/implementation-worktrees/m11-conditional-scaling/workspace/exports/m11_handoff.zip ~/Downloads/
shasum -a 256 ~/Downloads/m11_handoff.zip
```
