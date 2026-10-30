# M9.6 — probability-sampled cross-pool protection comparison

Implementation: `m96-cross-pool`, isolated at
`/projects/netsurf/manuelbv/workspace/implementation-worktrees/m96`, based on
`e0ddf3a3cb0f68e6de84c1e4fc9f95ffb0cf88e1`. No merge or push. The twelve-pool
production study has **not run**. Only two small engineering pools used already
exposed inputs. This is orchestration and analysis of the existing exact selector
and paired-swap estimator, not a new optimizer or fault model.

## Scientific question and authority

Estimate the distribution of held-out CE differences for independently sampled
candidate pools around one frozen S5 allocation. Negative target-minus-baseline
differences favor the target. This concerns a defined *pool-sampling design*, not
uniformly sampled individual bits, a global optimum, other checkpoints, or SEUs.
A conditionally selected allocation is still a static deployment mask.

The finalized M957 replication was located and verified after its implementation
document was written. Its source is
`../m957/workspace/runs/m957_kws_frozen_replication_k8_v1/20261030T184931.331632Z-27c6efd5`,
payload `fc33c74c598eebff8ae2fe24493cfb2af0b2686f54bb1c2bb9ef30bb4b3cdbfa`.
Selected−S5 CE was −0.0003688943305, field CI
[−0.0004281698969, −0.0003096187640]; selected−isolated was −0.0003810317092,
CI [−0.0004403464920, −0.0003217169264]. These remain **one historically
selected exploratory pool**, not one of the twelve new pools.

### Verified fixed inputs

| Item | Identity |
|---|---|
| Checkpoint | `aeea436800704fce17b17292e4412630ad856e9d777c044c64ef748a880bd0ae` |
| KWS asset-lock file | `09fc5867d721363a0156e2f9f6fdedefe2aa0ae7384cc72396bc594e283bfac7` |
| Authoritative runtime | `/projects/kona-assets/manuelbv/.venv-mlperf/bin/python`; TF 2.14.1 `BUILTIN_REF` |
| Source S5 run | `/projects/netsurf/manuelbv/workspace/runs/m93_kws_static_ber_sweep1000_v1/20261029T183410.488907Z-32c31070` |
| Source scientific payload | `b7b725fd1c196f2d5896b67aba34dfe8aa864b8308875df098d48be6135975e0` |
| Source protection artifact file | `2c5753a46bf09cb4f9f90bb73b80245ce70b2f57dfee52bf97c0e191e2e710b9` |
| Source mask key | `replication_0/per_example_absolute_global@0.125000` |
| Population | 176,128 eligible INT8 kernel bits; 22,016 protected; 154,112 unprotected |

Zero-valued allocated kernel words remain eligible; INT32 biases and affine
metadata remain excluded. Private model-buffer XOR, exact two's-complement
semantics, restoration, independent Bernoulli faults and ideal event suppression
are unchanged. Model bytes, asset-lock bytes, source membership and universe
hashes are checked. No shared asset is edited.

## Pool selection and frozen plan

`netsurf.cross_pool.sample_pools` uses `Generator(PCG64(seed))`. In each pool it
draws two protected identities without replacement, then four unprotected
identities without replacement; both source strata and the final candidate
sequence use canonical `BitIdentity` ordering. Different pools use independent
seeds and are **not** forced to be disjoint. Every pairwise overlap is recorded.
No outcome, gradient, bit tier or prior candidate score enters this sampling.

Within-pool marginal inclusion probabilities are 2/22,016 and 4/154,112.
For the optional exploratory K=8 design the second quota is six. They are not
equal-population bit inclusion probabilities. No Horvitz–Thompson claim is made:
the primary estimand averages independently generated pools of this design.
RNG/version, tensor/word/bit identity, signed code, unsigned word, scale,
zero-point, source protection status and candidate-order hash are retained.

Every pool has local budget two; `P_out` contains 22,014 bits. All 15 K=6
allocations therefore protect exactly 22,016 bits. Original S5 is among them.
The exact isolated comparator uses signed single-bit CE damage on selection
inputs under the same outside protection context. The selected comparator
minimizes model-selection response-table risk. Exact `2^K` enumeration retains
every candidate interaction order, including beneficial faults. No final
outcome participates in selection.

The maintained campaign request is `config/examples/m96_kws_cross_pool_k6.toml`.
Its already generated, validated manifest is:

`workspace/runs/m96_kws_cross_pool_k6_v1/campaign.json`

SHA-256: `46674ad4f90e8bc49fcee3ef826bbb57130005f239d192700cd09d8f106d3ff3`.
It freezes all twelve candidate lists, all seeds, source context, member plans,
sample windows and statistical family. `configs/pool_000_selection.toml` through
`pool_011_selection.toml` are present. There are no production results or sealed
selection artifacts yet. Do not regenerate this list after examining outcomes.

## Exact configuration interface

Campaign request `schema_version="1.0.0"` is distinct from member experiment
schema `6.6.0`, protocol `cross-pool-frozen-protection/v7`. Existing 6.0–6.5
readers retain their behavior. Unknown request keys are errors.

| Key | Supplied value / meaning |
|---|---|
| `name`, `purpose` | Filesystem-safe name; `confirmation` or explicitly `engineering` |
| `benchmark_template` | Relative read-only existing schema-6.4 selection TOML binding the exact assets/context |
| `design.candidate_count` | 6; also supports 8, always two protected candidates |
| `design.pool_seeds` | 960101–960112; one independent pool RNG each |
| `design.selection_seed_bases` | 960201, 960205, …, 960245 |
| `design.selection_replicates` | 4 consecutive base seeds per pool |
| `design.selection_fields_per_seed` | 8; 32 independent selection fields per pool |
| `design.final_seeds` | 960501–960512; separate from all selection streams |
| `design.final_fields` | 128 per pool |
| `design.logical_ber` | 0.01; v1 rejects another BER |
| `samples.selection_offset`, `selection_count` | 64, 64; original KWS test indices 2088–2151 |
| `samples.final_offset`, `final_count` | 402, 512; original indices 3450–3961 |
| `statistics.confidence` | 0.95; v1 fixed |
| `statistics.bootstrap_seed`, `bootstrap_replicates` | 960801, 1000; at least 100 diagnostics |

Field seed j equals its recorded base plus j×1,000,003. Derived seed collisions
across the new selection/final plan are rejected. Inputs are shared across all
pools, not freshly sampled per field. The model's original training split is
unchanged; upstream checkpoint training exposure is not re-audited here.

Generated member `[workflow]` adds `campaign_manifest`, `campaign_sha256`,
`pool_id`, `record_per_input=true`, and the precisely named probability-sampling
provenance. Final members add `final_registry`, `final_registry_sha256`, and
the existing selection bundle/payload/artifact hashes. These are generated
bindings, not researcher-editable tuning knobs. Altered candidates, sample
counts, seeds, budgets, risk, backend or context are rejected against the frozen
plan. Separate final configurations are generated only after every selection
bundle is finalized and verified.

## Exposure audit and dependence

The frozen manifest includes the complete machine-readable inventory, hashes
and exact original indices: 35 known finalized records/copies, union 1,470
exposed identities, maximum 3449. This extends the M957 ledger (1,214 identities)
with the actual 3194–3449 replication. All earlier M953–M955 final smokes,
M956 original/engineering finals, static/conditional evidence and embedded
selection identities remain conservatively exposed. One deliberately corrupted
M957 test copy is explicitly marked invalid evidence; its intact payload indices
are conservatively included. It is not silently accepted as a valid bundle.

No inventoried prior fault-evaluation identity intersects 3450–3961. The locked
test population has 4,890 examples, so this window exists. M96 engineering
uses only 2024–2025 selection and 3060–3061 final clips. Aggregate clean-reference
and label audits already covered the full source split: “unexposed” means no
recorded prior **fault-outcome** use, not inference-naive data.

At execution, known canonical and implementation run/scratch roots are rescanned.
Other-campaign exposure rejects confirmation; already completed members of this
same frozen campaign may share the prespecified input set. Unknown manual,
failed, out-of-root or concurrent experiments remain a disclosure obligation.
There is no global multi-campaign reservation service: do not concurrently use
these confirmation clips elsewhere. Contiguous TFDS interleaved-block windows
are not guaranteed class-identical or iid; execution records their identities,
hashes and class composition. No new-window inference was performed to tune it.

## Analysis and uncertainty

Primary: selected−S5. Secondary: selected−isolated and isolated−S5. Per pool,
the existing paired-swap estimator averages independent whole master fields;
input/step/contrast records sharing a field are not independent replications.
All source response tables, isolated scores, runner-up margins, cumulative
winners, replicate winners and winner switches remain in finalized selections.

Across pools, the primary estimate equally weights twelve pool means, with a
two-sided Student-t interval and test over **pools**, conditional on the common
inputs and fixed selection set. Two secondary tests use Holm adjustment and
separate Bonferroni 97.5% intervals. Negative/zero/adverse estimated effects,
quantiles, medians and individual field intervals are retained. Wilson intervals
for the fraction of negative *estimated* pool effects do not prove the
probability of true improvement: field noise can change a pool's observed sign.

Prespecified deterministic diagnostics separately resample pools, fields within
fixed pools, common input indices synchronously across all pools, and a crossed
pool/field/shared-input bootstrap. Contrasts always stay paired. Shared-input
resampling assumes exchangeable clips; contiguous source ordering may violate
that assumption. Crossed/nested resampling may double-count finite-field noise
and is labeled an approximate sensitivity analysis, not another confirmatory
test. Twelve pools give limited distributional precision; field-count choices
are planning choices, not a power calculation or a universal superiority test.

The historical selected−S5 whole-field standard deviation was 0.00077916 in
M956 (64 fields) and 0.00048159 in M957 (256 fields). If those scales applied,
128 fields would give a marginal 95% half-width roughly 0.000136 or 0.000084.
This is only a sensitivity calculation: the new inputs, sampled pools and masks
can have very different variance, and one old pool cannot estimate between-pool
variance. The 32-selection/128-final-field plan is a bounded construction-versus-
evaluation compromise, not a claim of twelve-pool statistical power.

The aggregator requires every prespecified final receipt; it cannot drop adverse
or unfinished pools. Scalar paired-field measurements are authoritative;
per-input vectors support diagnostics and are checked against their reductions.
Scientific hashes omit runtime metadata. Regeneration writes a **new** analysis
directory and never modifies source bundles. No TensorFlow is needed for it.

## Work and resource plan

| Work | Per pool | All 12 pools |
|---|---:|---:|
| Model-selection response-table states | 32×64 = 2,048 | 24,576 |
| Clean + isolated selection states | 7 | 84 |
| Total selection state requests | 2,055 | 24,660 |
| Selection clip inferences (64 clips) | 131,520 | 1,578,240 |
| Analytical mask integrations | 32×15 = 480 | 5,760 |
| Final master fields | 128 | 1,536 |
| Final paired-state requests, upper bound | 128×3×2 swaps×2 states = 1,536 | 18,432 |
| Final clip inferences, upper bound (512 clips) | 786,432 | 9,437,184 |

Final paths are fixed after selection, at most two swaps per contrast. Exact
schedule reuse and identical masks reduce actual authoritative calls; reported
actual counts distinguish requests from unique evaluations. Sixteen bounded
reference clips per adapter initialization are additional. No inference is
needed to reuse the response tables for the fifteen masks.

Measured historical planning anchors: M956 selection took 1,818.27 s for 8,201
states ×64 inputs; proportional K=6 selection is about 456 s/pool, about 91 min
for twelve, before changes in fixed overhead. M957 final took 678.70 s for
768 unique states ×256 inputs. Applying its average cost to the conservative
new final clip upper bound gives about 9.05 CPU-hours. Cache/path reuse can
make this substantially smaller. Reserve roughly 10–12 CPU-hours for sequential
planning, **not a runtime guarantee**, and operationally batch three pools at a
time if useful. Do not reduce scientific counts based on observed effect signs.

New tiny runs each took 42–54 s including expensive fixed verification/loading,
so they are not good large-run per-clip throughput estimates. Existing KWS
adapter evidence is roughly 0.5 GiB peak; allow 1–2 GiB for engineering and more
for Python identity/event records. The full analysis data cube is about 19 MiB
before bootstrap work. Four tiny bundles total about 21 MiB; production records
scale with fields, outside fault multiplicity and per-input diagnostics. Reserve
several GiB (2–5 GiB planning allowance, not a measured upper bound). Exact
events/response tables are not discarded. This is much more construction work
than reusing a static ranking; report it alongside any small CE improvement.

## Lifecycle, commands and failure recovery

Run in the isolated worktree, in the foreground (tmux is suitable). No install,
environment modification, wrapper or GPU is needed. Modern TF 2.21 is **not**
the pinned reference backend.

```bash
cd /projects/netsurf/manuelbv/workspace/implementation-worktrees/m96
/projects/kona-assets/manuelbv/.venv-mlperf/bin/python -m netsurf campaign validate \
  workspace/runs/m96_kws_cross_pool_k6_v1/campaign.json

# Researcher starts these production phases after reviewing the cost/protocol:
/projects/kona-assets/manuelbv/.venv-mlperf/bin/python -m netsurf campaign run \
  workspace/runs/m96_kws_cross_pool_k6_v1/campaign.json --phase selection
/projects/kona-assets/manuelbv/.venv-mlperf/bin/python -m netsurf campaign seal \
  workspace/runs/m96_kws_cross_pool_k6_v1/campaign.json
/projects/kona-assets/manuelbv/.venv-mlperf/bin/python -m netsurf campaign run \
  workspace/runs/m96_kws_cross_pool_k6_v1/campaign.json --phase final
/mnt/raid0/ml/conda_envs/manuelbv/envs/netsurf/bin/python -m netsurf campaign aggregate \
  workspace/runs/m96_kws_cross_pool_k6_v1/campaign.json \
  --output workspace/runs/m96_kws_cross_pool_analysis_v1
```

Optional operational batches append `--pools pool_000 pool_001 pool_002` to
`campaign run`; repeat for all twelve. `seal` refuses an incomplete list.
Progress uses one existing Rich queue when available, or plain stderr; stdout
retains JSON. Counters represent actual states plus setup/finalization. Pending
pool costs use observed operation classes; initial ETA is uncertain and final
work starts from a conservative path upper bound. `--no-progress` is supported.

Before outcomes, a researcher can edit a **copy** of the request and freeze a
separate design using `campaign freeze CONFIG --output NEW_WORKSPACE_DIRECTORY`.
Changing an already frozen campaign is forbidden. The supplied K=6 production
manifest is already frozen: do not execute `freeze` again at its destination.
K=8 uses 28 masks and 256 states/field, about four times response-table work.

Verified completed receipts resume without inference or duplicate results.
An exclusive campaign lock prevents concurrent writers. An interrupted attempt
without a receipt fails closed: it may have consumed final inputs. Preserve its
outputs and request an explicit recovery audit; there is no automatic rerun,
seed replacement, or outcome-dependent skipping. A partially failed seal also
requires inspection rather than overwriting `final-plan`. Use the campaign
interface, not direct member `conditional run`, for lifecycle bookkeeping.
Generated plans contain absolute pinned asset/source paths; transplanting them
requires those paths and hashes or a separately identified campaign, not editing
finalized provenance in place.

## Bounded validation evidence

Engineering request: `config/examples/m96_kws_engineering.toml`; frozen campaign
`workspace/runs/m96_engineering_campaign_v1/campaign.json`, SHA
`54c6f6c5fc507c8531f99928b62e09fe021f2a828bc65c87ba344d55394c4c8b`.
Two independently sampled K=6 pools, two selection fields, two final fields,
two already exposed clips per role. Selection accessed no final samples;
loaded masks were identical to the finalized selection; restoration and global
cardinality checks passed. Each selection used 135 states; each final used
four unique states for eight path requests. These outcomes validate plumbing,
not estimator superiority.

| Pool / phase | Scientific payload SHA-256 | Runtime s |
|---|---|---:|
| 000 selection | `9a358b0d7b1f95268287b04056662ffc352621e9d3f199f12632d49129098a41` | 42.88 |
| 001 selection | `f5018d4fad590f4b9afebfb79e892c42cc468fd723a02b2dc311e120c5ba272b` | 41.91 |
| 000 final | `d2499fa21380eaa9412e786f28628f36d762276f2d0577344d0592688b2d6b1d` | 54.10 |
| 001 final | `fca2dd29f5c47a159a8cda6849d983978ff06c50e5465f93b845e7fb9246578e` | 51.65 |

`workspace/runs/m96_engineering_analysis_v5/report.md` and `pool_effects.png`
are regenerated offline. A second derivation in `m96_engineering_analysis_v3`
has the identical scientific hash
`c6eaaa697f020a3e9166253e48a21f9b132aeb2857e4cb30690c0ced72f78841`.
No historical finalized bundle was edited.

Test commands/results and external-state failures are recorded in the delivery
notes below and final handoff. Generic `jsonschema` is absent; the executable
strict Python validators are active and tested, with optional generic-schema
checks explicitly skipped. It is not a requirement to install it for this run.

### Regression results

- Focused M96 suite: **17 passed, 1 skipped** in canonical and pinned Python.
- Pinned M93/M95/M953/M955/M956/M957/M96 suite: **89 passed, 3 skipped**,
  196.25 s. Existing KWS/ToyCar/reference/restoration tests pass. Optional
  generic-schema checks are skipped, not silently replaced by a different ML
  backend. Existing audio-library deprecation warnings remain visible.
- Full canonical suite: **299 passed, 8 skipped, 6 failed**, 90.20 s. All six
  failures are the pre-existing external-state failures also documented in M957:
  four MNIST configuration tests cannot resolve the worktree-local prepared
  checkpoint; one ResNet test expects an older lock digest (`f9533…` versus
  current `d0daf…`); another uses an old `/ml` checkpoint path inconsistent with
  the current lock's resolved path. No lock, expected digest, test threshold,
  checkpoint or researcher TOML was changed to conceal these failures.

Reproduce with fresh pytest temporary directories (pytest clears a reused
`--basetemp`):

```bash
/mnt/raid0/ml/conda_envs/manuelbv/envs/netsurf/bin/python -m pytest -q
/projects/kona-assets/manuelbv/.venv-mlperf/bin/python -m pytest -q \
  tests/test_m96_cross_pool.py tests/test_m957_replication.py \
  tests/test_m956_preflight.py tests/test_m953_same_distribution.py \
  tests/test_m955_paired_swap.py tests/test_m95_conditional_damage.py \
  tests/test_m95_restricted_subcube.py tests/test_m93_multibench.py
```

The failures are respectively
`test_controlled_progress_and_multiber_plan_are_complete`,
`test_all_maintained_m4_configurations_validate`,
`test_all_maintained_m43_configurations_validate`,
`test_wide_ber_configuration_is_maintained_and_valid`,
`test_neutral_asset_lock_verifies_without_kona_assets_import`, and
`test_official_subset_identity_reference_backend_and_real_image_fault`.
Resolving them requires a separate asset/path-compatibility review, not a change
to the M96 numerical contract. The implementation does not claim a green full
repository suite.

## Export to a Mac

The delivery archive `workspace/exports/m96_handoff.zip` contains source changes,
documentation, requests, the frozen twelve-pool manifest/configs, and tiny
engineering bundles/reports. It does not distribute datasets or checkpoints.
It is evidence/documentation, not a self-contained replacement for pinned Kona
assets or the source S5 bundle. From the Mac, using the researcher's configured
SSH host alias:

```bash
scp kona:/projects/netsurf/manuelbv/workspace/implementation-worktrees/m96/workspace/exports/m96_handoff.zip .
shasum -a 256 m96_handoff.zip
unzip -l m96_handoff.zip
```

Replace `kona` with the configured SSH alias if different. The final handoff
reports the archive checksum. No production experiment is started by export.
