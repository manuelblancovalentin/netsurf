# M9.10 — Interaction attribution and decision detectability

## Scope and scientific status

Branch/worktree: `m910-interaction-attribution`, based exactly on M99
`c6bf8cab2a87526a6385026ee5592242033f4687`, at
`/projects/netsurf/manuelbv/workspace/implementation-worktrees/m910-interaction-attribution`.
No merge, push, asset-lock change, environment change or new scientific inference.
Historical bundles remain in their original worktrees. No production campaign
or new optimizer was run. Existing regression tests may perform their usual
bounded engineering checks; those are not new research observations.

This is **exploratory offline analysis**, not a newly prespecified confirmatory
hypothesis family. The earlier handoff documents correctly described production
plans as unexecuted at their delivery, but the researcher has since completed
all six production campaign/BER arms below. Finalized records take precedence
over those old status statements.

Main result: on these model-selection subcubes, **outside-background conditioning
accounts for substantially more decision reordering than simultaneous
within-candidate interactions**. This algebraic attribution is not a causal
claim about architectural redundancy, nor proof that a conditional selector
improves over an exact isolated comparator on unseen data. The independently
evaluated equal-pool selected-minus-isolated intervals span zero in all six
campaigns. More estimators or an optimizer are not justified by this analysis.

## Evidence inventory and verification

The hash-pinned offline input inventory is `config/analysis/m910_sources.json`.
It includes every planned production pool, without outcome-based exclusion:

| Source | BER | Pools | Selection fields/pool | Final fields/pool | Selection / final inputs | Risk |
|---|---:|---:|---:|---:|---|---|
| KWS M96 | .01 | 12 | 32 | 128 | 64 / 512 clips | Mean exact integer-softmax CE |
| KWS M97 | .001 | 12 | 32 | 128 | 64 / 512 clips | Same CE contract |
| KWS M97 | .003 | 12 | 32 | 128 | 64 / 512 clips | Same CE contract |
| KWS M97 | .01 | 12 | 32 | 128 | 64 / 512 clips | Same CE contract |
| ResNet-8 M98 | .01 | 4 | 8 | 32 | 32 / 128 images | Mean exact integer-softmax CE |
| ToyCar M99 | .001 | 4 | 8 | 32 | 24 / 48 complete files | **1 − file ROC-AUC** |

There are **56 pool/BER conditions and 112 finalized selection/final bundles**.
The twelve KWS pools recur across M96/M97 BERs: these are not 48 independently
sampled pools. M96 and M97 use different input windows. Raw CE and AUC differences
are never pooled across architectures or treated as commensurate units.

Readers verify frozen campaign checksums, complete sealed selection registries,
joint-M97 bindings, every source-bundle inventory, canonical scientific payload,
finalization marker, exact selected-mask source binding and checkpoint/lock pins.
Response CSVs are reconstructed using their original native types and checked
against scientific-record hashes. Each field additionally checks:

- Exact candidate order and tensor/word/bit identities.
- Independent-Bernoulli probability law, BER and field/seed/replicate identity.
- Outside fault identities/hash, absence of candidates and protected outside bits.
- Sample partition, protection-context identity and fixed global cardinality.
- All `2^K` states, no duplicate states, finite responses and event accounting.
- Response-table hash and independently reintegrated every-mask risks.
- Same-cohort empty/single-bit controls and saved isolated damages.

Source payload and checksum-inventory identities are checked again at completion.
Missing fields, stale hashes or incompatible records fail closed. There is no
"best available pool" fallback. Models, datasets and historical mask archives
are not loaded for this analysis.

Engineering campaigns are analyzed separately (seven pools across M96, M97,
M98 and M99), never mixed into production inference. Payload-identical replays
are not extra replications. The superseded, unexecuted M97 v1 plan is excluded.
M95–M957 historical explicit pools remain exploratory antecedents, not members
of these probability-sampled campaigns.

## Exact decomposition

Let `C` be the ordered candidate pool, `Q` its protected subset, `U=C\Q`,
`G` the fixed sampled outside fault field, and `h_G(A)=R(G∪A)`. Let `R0` be the
clean risk on **the same model-selection cohort**, and

```text
d0_b = R({b}) − R0
dG_b = h_G({b}) − h_G(∅)
J_G(Q) = Σ[A⊆U] p^|A| (1−p)^(|U|−|A|) h_G(A)
```

The reported absolute-risk components are

```text
J_G(Q) = [R0 + p Σ[b∈U] d0_b]                  clean additive prediction
       + [h_G(∅) − R0]                         background intercept shift
       + [p Σ[b∈U] (dG_b − d0_b)]              outside-background correction
       + [J_G(Q) − h_G(∅) − p Σ[b∈U] dG_b]     within-candidate correction.
```

For any target `Q'` minus baseline `Q`, the intercept cancels. The remaining
three **signed** contributions sum to the exact contrast for every field.
All equal-budget pairs are exported, not only the favorable or frozen pairs.
Beneficial faults and adverse protection effects are retained. Centered
integration removes the common intercept before small differences are taken;
its equivalence to the original absolute-risk integration is checked separately.
It does not alter historical selection or risk values.

The optional anchored Boolean Möbius coefficients satisfy
`h_G(A)=Σ[S⊆A] m_G(S)` and
`J_G(Q)=Σ[S⊆U] p^|S| m_G(S)`. Orders ≥2 reproduce the within-candidate residual.
All orders through K are saved. This is a local exact candidate-subcube expansion,
not a whole-network spectral decomposition. It applies to scalar cohort AUC
without assuming additivity over files. Its Bernoulli weighting cannot be reused
as-is for correlated/fixed-count/attack-guided fault laws.

### What the decomposition does not identify

It explains contrasts on the recorded **selection** cohort and outside fields.
Final paired-swap records do not contain complete final subcubes or matched
clean-isolated controls on those same final cohorts. Therefore a decomposition
of the final expected-risk difference is not identifiable from the archive.
Differences between selection and final outcomes confound finite-field selection
noise, independent final-field noise and changed input populations. We do not
assign their residual to a preferred causal mechanism.

## Numerical findings

The first complete offline derivation verified field/mask closure with maximum
absolute error **3.55×10⁻¹⁵**. Strict ranking reversals below exclude pairs tied
within a diagnostic absolute threshold of `1e-12`; raw exact ties and raw-rank
Spearman/Kendall statistics are separately saved. This threshold never changes
a frozen mask or measured response.

| Campaign | Clean-additive → exact strict reversals | Background-conditioned additive → exact reversals |
|---|---:|---:|
| KWS M96 .01 | 396 | 8 |
| KWS M97 .001 | 328 | 0 |
| KWS M97 .003 | 305 | 0 |
| KWS M97 .01 | 376 | 4 |
| ResNet M98 .01 | 201 | 1 |
| ToyCar M99 .001 | 196 | 0 |

Denominators and all per-pool results are in `mechanism_summary.csv` and
`report.md`. Pairs share masks/fields and are **not independent binomial trials**.
The small residual reordering does not establish zero candidate interactions.
At low p, simultaneous candidate terms also carry powers `p²,p³,…`; small
decision contributions can coexist with large forced-state interactions.

Illustrative equal-pool signed components for selected-minus-exact-isolated on
the model-selection data (different metric units must not be compared):

| Campaign | Clean isolated | Outside correction | Candidate correction | Exact selection contrast |
|---|---:|---:|---:|---:|
| KWS M97 .01, CE | +5.204e-5 | −2.278e-4 | +7.334e-7 | −1.750e-4 |
| ResNet M98 .01, CE | +5.173e-3 | −7.133e-3 | +3.840e-6 | −1.957e-3 |
| ToyCar M99 .001, AUC risk | +3.299e-5 | −4.210e-5 | +4.991e-9 | −9.110e-6 |

The archive also reports noncancelling absolute field and absolute field-mean
component magnitudes across **all** mask pairs; signed averages alone can hide
heterogeneity. Selection regret is empirical within the bounded pool, selected
on these same fields, not an unbiased population regret or global optimum.

One concrete wrong-sign case is KWS M97 BER .01, pool 000: the clean isolated
prediction for protecting candidate indices `[0,4]` (selected) rather than
`[1,4]` (isolated) is **+3.23349e-5 CE**, whereas the exact selection contrast
is **−3.25639e-4 CE**. The outside correction is −3.66675e-4 and the within-pool
correction +8.70035e-6. The full candidate identities and all other pools are
saved; this illustrative case is not a separately selected confirmation test.

### Independent held-out selected-minus-isolated contrasts

| Campaign | Mean | Descriptive pool-t 95% interval |
|---|---:|---|
| KWS M96 .01 | −2.02436e-4 | [−4.07816e-4, +2.94429e-6] |
| KWS M97 .001 | −3.39827e-7 | [−7.67165e-6, +6.99200e-6] |
| KWS M97 .003 | −1.45116e-5 | [−3.99828e-5, +1.09596e-5] |
| KWS M97 .01 | −1.22785e-4 | [−2.88804e-4, +4.32347e-5] |
| ResNet M98 .01 | +1.10811e-4 | [−1.68151e-4, +3.89772e-4] |
| ToyCar M99 .001 | +7.10720e-6 | [−9.95590e-6, +2.41703e-5] |

These intervals preserve candidate pools as the principal units. They are
descriptive marginal intervals, not replacements for original M97 joint-family
Holm tests or the other protocols' primary/secondary conventions. There is no
new significance claim. KWS selected-minus-S5 means are negative in each campaign,
but that is a weaker question than selected versus the exact isolated comparator.
ResNet and ToyCar currently provide no resolved equal-pool advantage over isolated.

### Supported and unsupported claims by benchmark

| Benchmark | Supported by these finalized records | Not established |
|---|---|---|
| KWS | Background conditioning changes many selection-mask orderings; the selected-minus-S5 mean is negative in each recorded campaign. All pools and independently evaluated final masks are retained. | A resolved cross-pool selected-minus-exact-isolated advantage, a universal ranking improvement, or causal architectural redundancy. The recurring pools are not independent across BERs. |
| ResNet-8 | The exact decomposition closes; outside-conditioned additive ordering nearly matches the integrated ordering on these selection fields. Eight-field selection margins can be small relative to field variability. | Held-out conditional-selection superiority, an accuracy/chance floor inferred from CE, or unique attribution of selection/final disagreement to input shift rather than field noise. |
| ToyCar | Complete-cohort AUC is nonadditive and many forced candidate states change it. Background AUC degrades substantially, while expected mask contrasts are much smaller; cohort differences and unstable winners are observable. | Absence of interactions, zero true protection benefit, an additive reconstruction-loss explanation of AUC, or untouched confirmation on the now-exposed production files. |

## ToyCar resolution, degradation and exposure

Complete-file scoring is unchanged. The 24-file selection cohort has 12 normal
and 12 anomalous files; clean ROC-AUC is **0.8263888889**. The 48-file final
cohort has 24 of each class; clean ROC-AUC is **0.9201388889**. This measurable
cohort difference is not by itself a causal explanation of mask effects.

- Selection pair-reversal lattice: `1/144 ≈ .00694444`; half-credit ties give
  `1/288`. Final lattice: `1/576 ≈ .00173611`; ties give `1/1152`.
- This is the resolution of an individual cohort AUC, **not** a lower bound
  on expected-risk contrasts. Multiplication by p, response integration and
  field averaging can produce far smaller exact contrasts.
- Nonempty forced candidate states changing AUC relative to the same field's
  empty state: pools 000–003 respectively **390/504, 296/504, 332/504, 288/504**
  (77.38%, 58.73%, 65.87%, 57.14%). These forced states are not equally likely
  Bernoulli deployment outcomes.
- Mean selection outside-field risks are approximately .4523, .3889, .4366,
  .4444 (AUC .5477, .6111, .5634, .5556), compared with clean AUC .8264.
  Substantial degradation is observed; a universal chance floor is not asserted.
- Two pools have two near-best masks. The diagnostic selected-mask near-best
  bootstrap frequencies are approximately .124, .454, .794, .444. They are
  finite-field stability diagnostics, not posterior optimality probabilities.
- The selected and original masks coincide in pool 003: its selected-minus-S5
  final contrast is structurally zero. In other pools many final field contrasts
  are nonzero. Lack of a resolved aggregate improvement is **not** explained by
  all responses or interactions vanishing.

The best-versus-runner-up selection gaps can be tiny relative to paired-field
variation; the eight-field selection budget is weak for some pools. Candidate
pool design, quantized task resolution, independent final noise and cohort
variation remain entangled. The current archive cannot uniquely apportion
their causal importance.

All production selection/final files are now fault-exposed. The earlier M99
inventory's "no prior fault outcome" designation was correct **before** execution,
not permission to reuse them as untouched data. A refreshed metadata-only exposure
inventory is delivered separately under this M910 worktree. Additional genuinely
unexposed labeled recordings/device conditions are needed for strong external
confirmation; a small remainder of official files is not a new independent
dataset. No data or labels were invented.

The refreshed ledger verifies 22 finalized source records (including duplicate
replays for exposure accounting only): **200/248 files have recorded fault-outcome
exposure**. The remainder is **40 normal and 8 anomalous files**. All 248 already
underwent clean-reference inference. Failed/manual/out-of-root work may add
exposure; this metadata inventory is not an assertion that no unrecorded use
occurred or a reservation of the remainder for a future campaign.

## Decision detectability and computational cost

Four uncertainty levels remain separate:

1. **Fixed-mask fault fields:** whole-path paired t intervals, all fields retained.
2. **Choosing among masks:** shared-field bootstrap over the entire risk vector,
   cumulative/individual/seed-batch winners, margins, leave-one-field-out stability.
   Post-selection gaps/intervals are descriptive and selection-biased.
3. **Candidate pools:** equal-pool summaries under the original status-stratified
   sampling rule. Shared KWS pools across BERs are not independent new pools.
4. **Input population:** classification shared-clip resampling and ToyCar stratified
   **whole-file** resampling, conditional on saved fields/masks. ToyCar recomputes
   each paired state's complete-cohort AUC before summing paths. No per-file AUC
   loss, per-window average or independent pair count is substituted.

The input resampling assumes within-stratum exchangeability; source/device order
may invalidate it. Its percentile ranges are **sensitivity diagnostics**, not
additional confirmation intervals. There is no resampling of model training,
checkpoint selection or architectural families.

`final_field_sensitivity.json` and `selection_stability.field_count_sensitivity`
give prospective half-widths `t*.observed_SD/sqrt(n)` for larger field counts.
Zero empirical variance does not certify equivalence or exclude unseen tails.
These scenarios fix masks, risk, cohort and variance: **not formal power**.
Changing files or BER changes the estimand and may change the variance.

For ResNet, clean selection CE is 1.53854 nats/image. Mean outside-field CE
across the four pools ranges from 2.67592 to 2.94077, with one field reaching
6.19248. This supports investigating a lower-BER regime, but CE alone does not
establish chance-level classification or a hard saturation floor. Stronger
selection sampling and lower BER answer different questions and need separate
frozen designs.

Every K6 selection field costs 64 full-cohort model states; 15 legal masks are
integrated offline. Four times the selection fields costs approximately four
times state inference, not four times statistical information about inputs.
All exact tables permit alternative additive diagnostics **without new calls**.

| Historical completed campaign | Selection engine seconds | Final engine seconds |
|---|---:|---:|
| KWS M96 .01 | 5,840 | 7,462 |
| KWS M97 .001 | 5,162 | 7,990 |
| KWS M97 .003 | 5,315 | 8,287 |
| KWS M97 .01 | 5,965 | 8,762 |
| ResNet M98 .01 | 771 | 476 |
| ToyCar M99 .001 | 1,552 | 1,067 |

These are saved engine times including setup, not machine-independent throughput.
For a four-pool ToyCar selection increase from 8 to 32 fields, exact response
work grows from 2,048 to 8,192 cohort states: at 24 complete files and 196
windows/file, **38,535,168 response-table window inferences**, plus isolated,
clean and conformance calls. A linear total-engine-time planning scale is
about 6,208 seconds, likely conservative because setup is not repeated four times.
For ResNet the analogous 8,192 states ×32 images is **262,144 image inferences**,
plus controls; the conservative time scale is about 3,084 seconds. Neither plan
has been frozen or executed; choosing a lower BER requires a new variance pilot.

The first complete offline derivation took **91.94 seconds**, peak RSS
**346,264 KiB (~338 MiB)**, and **zero new inference calls**. It processes one
pool at a time. Complete exact fault lists are verified but are not duplicated
into the derived CSVs; source hashes/pairing keys provide replay links.
The supported source studies here use K=6 and 15 masks. Algebra supports K≤12,
but all-pair export scales as fields × binomial(K,budget)²: this is not a claim
of constant memory for arbitrary large subcubes. Stability diagnostics require
at least two fields and two distinct masks; they fail explicitly otherwise.

## Priority plan for the November 12 TECS draft

1. **Freeze the narrative now:** exact stored-bit methodology, meaningful task
   metrics, strong isolated/static comparators, uncertainty and demonstrable
   context dependence. Do not frame a universal new estimator as established.
2. **KWS first:** use all existing pools/BERs for the conditional-versus-isolated
   question. Explicitly distinguish the earlier hand-picked pool from the later
   probability-sampled pool results. If funding another study, prespecify stronger
   selection sampling based only on selection diagnostics and use independent
   final schedules; exhausted KWS windows cannot become untouched again.
3. **ResNet second:** the eight-field selection winner gap/SE ratios are roughly
   .93, .48, 1.43, .05. Stronger selection sampling is defensible as a pilot;
   a lower-BER comparison on already exposed engineering inputs can diagnose
   regime sensitivity. Saved CE alone does not establish an accuracy floor.
   Freeze any subsequent confirmation design before observing its outcomes.
4. **ToyCar third:** exploratory reuse of exposed official cohorts is appropriate
   for resolution/metric-alignment analysis. A mechanism-enriched candidate rule
   must use selection-only information, freeze its changed sampling estimand and
   preserve independent final fields. Genuine independent confirmation needs
   additional unexposed labeled recordings, not post-hoc relabeling of these files.
5. **Equal-compute unguided random-swap comparator:** before outcomes, draw legal
   swaps uniformly from the same source-status universe, include a no-change
   control, charge every proposal and exact selection evaluation against the same
   inference budget, freeze the resulting mask, and evaluate it with independent
   paired final fields. Exact-table counterfactuals can describe selection cost,
   but no held-out performance is available for masks not actually evaluated.

Do not spend the pre-draft interval adding MCMC/RL, attack-guided proposals,
FKeras changes, new architectures or the Paper 2 optimizer. A null or unresolved
conditional-versus-isolated comparison is a legitimate result, not a software
failure to conceal.

## Suggested manuscript sections and figures

- **Methods — Exact finite-bit task-risk experiments:** fault population/lifetime,
  ideal suppression, full-budget context and authoritative reference execution.
- **Methods — Decision attribution and uncertainty:** the three-term contrast,
  anchored candidate interactions, independent field/pool units and selection bias.
- **Results — Context changes isolated ordering:** Figure 1 and all-pair magnitude
  table. Do not label outside-background correction a causal redundancy effect.
- **Results — Independent protection decision quality:** Figure 2, every pool and
  the selected-minus-isolated table with original multiplicity context.
- **Results — Resolution and selection noise:** Figure 3 cumulative margins and
  Figure 4 ToyCar AUC distributions/state changes.
- **Limitations:** single checkpoints, small status-stratified pools, task/BER
  confounding, fixed cohorts/exposure and no whole-network optimality claim.

Every figure has PNG and vector PDF output with a caption in the generated report.
All numerical points derive solely from saved exact responses or paired fields.

## Implementation, outputs and reproducibility

New code only:

- `netsurf/interaction_attribution.py`: exact algebra, Möbius coefficients,
  ordering/stability/detectability diagnostics, CLI and pinned-source reader.
- `netsurf/interaction_evidence.py`: fail-closed source readers, whole-file
  bootstrap, one-pool-at-a-time analysis and write-once output.
- `netsurf/interaction_reporting.py`: tables, captioned figures and report.
- `tests/test_m910_attribution.py`: deterministic algebra/provenance/statistics.
- `config/analysis/m910_sources.json`: six exact production campaign hashes.

No fault engine, estimator definition, protection optimizer, experiment schema,
source configuration or historical artifact is changed. Analysis schema/protocol:
`1.0.0`, `offline-exact-interaction-attribution/v1`. The input inventory is an
offline recipe, **not a new scientific experiment TOML**.

Per-pool outputs retain all signed field×mask risks, all field×mask-pair
decompositions, every Möbius coefficient, source identities, complete ordering
and stability diagnostics, paired final comparisons and detectability scenarios.
The top-level scientific payload excludes its output path and wall times; costs
and execution provenance are separate. A new output directory is mandatory.
Failure leaves unfinalized diagnostic evidence; no overwrite/resume is implied.

```bash
cd /projects/netsurf/manuelbv/workspace/implementation-worktrees/m910-interaction-attribution

# Offline only; no TensorFlow needed. Use a NEW destination for every derivation.
/mnt/raid0/ml/conda_envs/manuelbv/envs/netsurf/bin/python -m netsurf.interaction_attribution \
  --sources config/analysis/m910_sources.json \
  --output workspace/runs/m910_user_analysis \
  --seed 910001 --bootstrap-replicates 500

# Repeated --campaign /absolute/campaign.json arguments allow another complete
# campaign family; every specified campaign still requires all its planned pools.
# --no-plots omits figures, not scientific measurements or provenance checks.

/mnt/raid0/ml/conda_envs/manuelbv/envs/netsurf/bin/python -m pytest -q tests/test_m910_attribution.py
/mnt/raid0/ml/conda_envs/manuelbv/envs/netsurf/bin/python -m pytest -q -rs --tb=short
```

Use the canonical environment's existing NumPy, SciPy and Matplotlib. The pinned
TF2.14.1 interpreter can run algebra/reader tests but has no Matplotlib; use
`--no-plots` there. No dependencies were installed. Optional JSON-schema and
existing pinned-backend/local-evidence skips remain explicit in the full suite.
The full suite's preexisting MNIST tests use a new read-only symlink in this
worktree's ignored `workspace/benchmarks/`, not a changed checkpoint.

## Verified delivery

Implementation commit: `f2d8254ae7fc0fae4a3c59d239b6008182db5114`, under
`manuelblancovalentin <manuel.blanco.valentin@gmail.com>`. The final documentation
commit follows it. No other worktree or branch was merged or changed.

The clean committed-source replay is finalized at:

```text
/projects/netsurf/manuelbv/workspace/implementation-worktrees/m910-interaction-attribution/workspace/runs/m910_production_attribution_committed
```

Its scientific payload is **byte-identical** to the preceding complete derivation:

```text
cb9099abed7a74c06dcae1c96a8f1122c3a8efdb6527e1a28475560601399ace
```

All 296 derived artifact checksums verify. All 112 historical scientific payloads
and checksum inventories were verified before and after analysis. Maximum
decomposition closure error is `3.552713678800501e-15`. The committed replay
took **81.68 seconds**, peak RSS **352,884 KiB (~345 MiB)**, zero new model calls.
Environment: Python 3.11.17, NumPy 2.4.6, SciPy 1.17.1, Matplotlib 3.11.2.

Separate supporting outputs:

- `workspace/runs/m910_engineering_all_v1`: all seven engineering pool conditions,
  payload `11cec468570e41aa7d9aab5dfeb63e23da789651594c6b0099ff6d7029c9c948`.
- `workspace/runs/m910_toycar_exposure_v1`: refreshed metadata-only ledger,
  payload `dc9cd42a58ccfd4d99ed0a502344f47c5ad1e39faf2a124470a37a7b363b8154`.

Regression results on final implementation:

- Canonical full suite: **383 passed, 16 skipped**, 83.70 seconds; five existing
  TensorFlow Lite deprecation warnings remain visible.
- New offline tests: **37 passed**. Cover all signed decomposition terms,
  additive/pairwise/pure higher-order/beneficial risks, BER boundaries, ties,
  whole-cohort AUC, exact complete-file resampling, strict response/provenance
  rejection, deterministic replay, every-pool reporting and overwrite rejection.
- Pinned MLPerf interpreter: **54 passed** across the new tests and existing
  restricted-subcube/paired-swap algebra tests, 1.51 seconds. No inference is
  required by these offline tests.
- The 16 canonical skips are existing version-gated TF2.14.1 checks, optional
  local evidence/upstream checkout checks, and three absent-`jsonschema` checks.
  No required offline dependency is missing. No package changes were made.

The delivery archive `workspace/exports/m910_handoff.zip` contains the report,
incremental implementation/tests, source inventory, a Git patch against the M99
base, complete derived production/engineering evidence and the exposure ledger.
It excludes original datasets/models and historical bundles. The report and
plots are directly readable; rerunning analysis requires the repository and
the pinned historical source paths. The archive is a handoff, not a stand-alone
model or dataset distribution.

```bash
# Kona: inspect the already-created delivery archive; no experiment is launched.
sha256sum workspace/exports/m910_handoff.zip
unzip -t workspace/exports/m910_handoff.zip

# Mac: replace KONA_SSH_HOST with your actual configured host/alias.
scp manuelbv@KONA_SSH_HOST:/projects/netsurf/manuelbv/workspace/implementation-worktrees/m910-interaction-attribution/workspace/exports/m910_handoff.zip ~/Downloads/
shasum -a 256 ~/Downloads/m910_handoff.zip
```
