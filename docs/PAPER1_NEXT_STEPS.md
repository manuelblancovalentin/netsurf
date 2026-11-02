# Paper 1: delivery gate, missing experiments and editorial decisions

This is the prioritized plan accompanying the completed scientific reconstruction,
not authorization to run campaigns. Existing evidence is sufficient for a complete,
explicitly exploratory internal draft. No new multi-hour run is its prerequisite.

## 1. What the next manuscript should actually do

Keep the authoritative uploaded ACM `acmart` project as the formatting/source
starting point, not the rejected internal draft's argument. Preserve the original
project archive; write the replacement in a new isolated publication directory.
Use the ten-section engineering structure requested by the researcher:

1. Constrained reliability of quantized edge inference and the allocation question.
2. Related work, including the **published FCCM predecessor**, EISFINN, FKeras,
   PrioriFI, BFA, selective hardening and the qualified Zenodo dataset.
3. NetSuRF's executable design flow, from frozen model to decision evidence.
4. Exact encoded bits, fault field/lifetime and logical mitigation contract.
5. Estimators, ordering, tied tiers, protection masks and resource definition.
6. Limits of isolated ordering: one clear counterexample and conditional value.
7. Models, partitions, baselines, task metrics, paired fields and uncertainty.
8. Full engineering results: susceptibility validity, budget/BER curves, costs,
   then compact conditional diagnostics including unresolved/negative findings.
9. Interpretation, model/task differences, physical-mapping and validity limits.
10. Conclusions restricted to those measurements; proofs in an appendix.

Target five main figures from the audit, not one figure per milestone. The main
text must use scientific names, not S2/S5, run IDs or pool numbers. Use “exact
optimization over a bounded candidate set,” not an unexplained “oracle.” Keep
hashes, exposure ledgers and implementation checks in the supplement.

Explicitly cite the FCCM invited paper and state the added methodology/evidence.
Do not reuse its hardware diagram/numbers as new results. Confirm authorship/order
with the collaborators. No tool or single-author placeholder can settle that.

## 2. Optional-week priorities, ordered by scientific necessity

### Priority 0 — complete the draft from existing evidence; no new inference

Produce all three models' budget/BER plots, all-method paired-effect tables,
sample-limited empirical validation, cost table and the compact all-arm conditional
diagnostic. Use the fresh verification receipts. Treat all existing results as
exploratory where design choices followed earlier observations. Report effect
sizes even when multiplicity-adjusted evidence is inconclusive.

This is higher priority than adding architectures or another estimator. The
existing ResNet positive result, KWS small/null effects and ToyCar unstable effects
already provide a coherent engineering investigation with limits.

### Priority 1 — meaningful partial-tier FKeras comparison, KWS first

- **Hypothesis/estimand:** at equal eligible-bit cardinality, do paper/repository
  FKeras tier-first masks change expected CE and accuracy relative to corrected
  MSB and gradient methods, particularly inside rather than at complete tiers?
- **Model/data:** same pinned KWS DS-CNN, existing train64 calibration and official
  1,000-clip test cohort (already exposed, therefore exploratory), same preprocessing.
- **Methods:** random, corrected MSB, numerical jump, global first-order,
  per-example absolute/positive, both FKeras variants. Shared eigenpairs for both
  formulas. Check convergence/residuals; do not reduce accuracy gates to finish.
- **Grid:** BER 1e−4, .001, .003, .01; coverage 6.25%, 12.5%, 18.75%, 25%, plus
  unprotected/full controls. Record full-tier identical masks rather than claiming
  separate independent evidence.
- **Budget:** initial 32 paired fields per BER, four independent random-mask seeds
  × eight fields; no assertion that this is powered for tiny differences. New,
  frozen fault seeds. Test prespecified pairs with within-BER/metric Holm correction;
  distinguish mask-level variability from conditional field intervals.
- **Operations:** at most `4 × 32 × (2 + 8 × 4) = 4,352` cohort evaluations,
  4,352,000 clip inferences, plus clean/calibration/conformance. Identical-mask
  execution reuse must be verified before subtracting operations from the estimate.
- **Cost:** observed KWS six-method grid: 3,584 cohort rows in 45.0 min at four
  effective workers. Linear row scaling gives about **55 min plus the larger
  Hessian construction**. This is a planning estimate, not a runtime guarantee.
  The two-example eigensolve does not justify a precise 64-example cost; use its
  HVP cap/residual gate and log actual cost. Allow roughly 1–3 GB ordinary outputs
  plus stored eigenvectors/score records; inspect existing serialization before
  claiming a tighter disk bound.
- **Required before authoring final TOML:** agree eigenpair/probe limits and the
  primary pair family. Existing schema 3.5 already expresses both FKeras variants
  through `config/examples/m97_kws_fkeras_protection_smoke.toml`; no new estimator
  implementation is necessary.

### Priority 2 — fill budget gaps for the six common policies, not new models

- **Hypothesis:** apparent superiority at one complete tier is stable—or not—at
  partial budgets. Primary estimand is paired task-metric difference at each fixed
  BER and exact protected cardinality, not maximum observed gain over the grid.
- **ResNet:** retain the frozen checkpoint and exposed 1,000-image cohort, use
  BER 1e−4/.001/.003/.01 and add **6.25%/18.75%** only. Six methods, controls,
  four random-mask seeds × eight independent fields per BER. Operation count
  `4 × 32 × 14 = 1,792` cohort rows = 1,792,000 image inferences. Approximately
  **100 min at the historical two-worker throughput**, with around 1–3 GB output.
  New seeds must not be combined as matched pairs with old schedules. If the
  purpose is direct cross-budget paired inference, reproduce the old schedule
  identities deliberately in a new bundle or include all budgets in one new run;
  never silently join by trial number.
- **KWS:** priority 1 already covers these cells; do not run a duplicate campaign.
- **ToyCar:** integrate these budgets into priority 3, after agreeing the cohort.
- **Uncertainty:** field-paired intervals; method/coverage families declared before
  execution; no new significance claims from post-hoc curve selection. Expected
  unprotected count is Np and applied count is `(N−protected_bits)p`; save actual
  counts as well. Larger BERs remain stress conditions.

### Priority 3 — stronger ToyCar evidence with an honest exposure label

- **Question:** does a ranking that predicts reconstruction MSE preserve file ROC-AUC,
  and are observed budget/BER reversals reproducible over fault fields?
- **Metric:** complete-file ROC-AUC (risk `1−AUC`) only for task efficacy; MSE
  separate. Persistent faults span every window/file in each trial. No window-level
  AUC or independent-normal/anomalous-pair assumptions.
- **Proposed grid:** BER 1e−4/.001/.003/.01; six policies; all four interior
  budgets plus controls; four random-mask seeds × eight fields initially.
  For the existing exposed 64-file cohort, this is `4 × 32 × 26 = 3,328` cohort
  evaluations, **212,992 file evaluations / 41,746,432 window inferences**.
- **Cost:** historical 448 rows took 21.5 min at two workers; linear scaling gives
  about **160 min**, excluding changes in setup/scoring reuse. Existing outputs
  were 4.04 GB despite only 448 rows, because full-population score/mask storage is
  large. Budget **8–15 GB**, not a precise prediction; memory was already ~15 GB
  aggregate workers plus coordinator. Do not increase workers blindly.
- **Exposure:** the verified ledger records prior clean use of all 248 official
  files and fault-outcome use of 200, leaving only 40 normal and 8 anomalous files
  without recorded fault exposure. This is not a fresh balanced confirmation set.
  More fields on the old 64-file cohort improves conditional Monte Carlo precision,
  not independent input-population confirmation. Additional authoritative data or
  an explicitly exploratory reuse decision is required; do not invent a partition.
- **Input uncertainty:** if saved paired file scores are available, stratified
  bootstrap complete files and recompute AUC. Keep this interval separate from
  field/mask uncertainty; do not substitute a per-file additive loss.

### Priority 4 — bridge estimator validity to task risk, without another theory

Reanalyse existing sampled candidates by tensor/bit tier and objective; report
tie-aware top-k, signed errors and conditional sample-design intervals. For ToyCar,
first inspect whether saved isolated-fault file scores permit direct AUC damage.
If not, a prespecified bounded task-metric audit on already exposed files is a
separate authorized run, not permission to infer AUC ranking from MSE correlation.
No new qPolarGrad variant or large-model/Hessian family is needed for the draft.

### Defer

Transformer/LLM expansion; new universal scores; guided search/importance sampling;
large conditional optimizers; physical ECC/TMR/ABFT/Ranger comparison without an
implemented fault/cost contract; redoing historical synthesis to recover old
numbers. These do not fix the current paper's most consequential gaps in one week.

## 3. Configuration and execution gate

No new publication TOMLs were created during the **first reconstruction gate**.
They should be prepared only after the researcher accepts the scope, FKeras
numerical budget, ToyCar exposure classification, comparison families and resources
above. This avoids freezing scientifically consequential choices by assumption.
Existing validated schemas/configurations are sufficient for the planned static
cells; no experiment engine rewrite is proposed.

The subsequent configuration delivery must include complete literal paths and
`/projects/kona-assets/manuelbv/.venv-mlperf/bin/python -m netsurf validate ...`
followed by the corresponding foreground `run ...` command, operating from the
isolated source revision with the unchanged asset locks. Do not use the modern
canonical TensorFlow as a substitute for the pinned 2.14.1 reference backend.
Record which source revision is actually imported; avoid editable-install path
ambiguity. No `nohup`, background wrappers, mask regeneration of historical runs
or overwrite of finalized destinations.

At that second delivery, validate the TOMLs without inference; estimate operation
counts from the resolved configurations, and export those exact files with the
protocol, source revision and checksum receipts. Cost estimates above are explicit
linear extrapolations from finalized runs, not measured costs of new campaigns.

## 4. Decisions needed before manuscript/submission claims

1. Accept the engineering comparison/decision-validity contribution rather than
   qPolarGrad or conditional-optimizer superiority.
2. Confirm physical resource claims are out of scope; logical bit budget is the
   controlled proxy, with physical realization left unresolved.
3. Confirm authorship and how the new article extends the FCCM publication.
4. Decide whether to spend the extra week on partial-tier FKeras and stronger
   ToyCar fields; accept that reused ToyCar data remain exploratory.
5. Choose a prespecified practical effect threshold and testing family before
   new trials. A tiny nonzero conditional effect need not justify its computation.

None of these decisions prevents an honest complete internal draft from the
existing evidence. They determine how strong and how broadly scoped a submission
can responsibly be.
