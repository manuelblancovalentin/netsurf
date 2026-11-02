# Optional additional week: decisions and missing evidence

Software only. These are priorities for researcher approval, not an executed
campaign or a claim that missing results are positive. Do not consume new
confirmation inputs until their exposure ledger and prespecification are frozen.

## 1. Meaningful partial-tier baseline comparison — highest necessity

Question: does score-dependent FKeras tie-breaking change task quality relative
to corrected MSB and per-example gradient masks when the masks actually differ?

- Begin with the already validated KWS FKeras pathway; do not assume the same
  Hessian companion exists and passes validation for the other two models.
- Compare paper-form and repository-form FKeras, corrected MSB, numerical jump,
  global first-order, per-example absolute/positive and seeded random.
- Missing coverages: 6.25% and 18.75%. Retain 12.5%/25% mask-identity controls
  without treating identical sets as independent methods.
- Suggested matched logical BERs: .001, .003, .01. Start with 32 paired fields
  per BER, four independent random-mask seeds, a fixed total evaluation budget.
- Freeze eigenpair convention, calibration identity, convergence/residual gate,
  task metric and comparison family before viewing outcomes. A two-example
  engineering eigensolve is not a publication-calibrated score.
- Report paired accuracy and cross-entropy effects, construction time and
  failure-to-converge explicitly. Use the existing within-BER correction; do
  not select only the best Hessian variant afterward.

Planning count for eight distinct methods at two partial budgets, two shared
controls, three BERs and 32 fields is at most **1,728 full-cohort evaluations**
(before verified mask deduplication). At 1,000 KWS clips this is 1,728,000 exact
reference invocations. The prior six-method/eight-BER static study took 2,701 s
with four workers for 3,584 evaluations; a purely linear evaluation-only
projection is about 1,302 s. It excludes the materially different Hessian
construction, initialization and reporting, so it is not an ETA promise.
The old 0.718 GB bundle gives only a rough lower-order storage reference;
eigenpair/probe records and estimator serialization require a fresh preflight.

## 2. ToyCar field precision and task alignment — high necessity

Question: do the observed nonmonotonic AUC curves persist with better Monte
Carlo precision on the same complete-file cohort? Use the same six static
policies, .0001/.001/.01, 12.5%/25%, and 32 paired fields rather than eight.
Preserve complete-file ROC-AUC, half-credit ties and shared persistent fields.
Treat MSE as a diagnostic, never the authoritative task objective.

This is 1,344 cohort evaluations, each 64 files ×196 windows, or **16,859,136
reference window invocations**. The earlier 448-evaluation study took 1,288 s
with two workers and 4.04 GB of output. Linear projection gives roughly 3,865 s
and 12 GB, before differences in output configuration. Additional fields reduce
fault Monte Carlo error; they do not restore input-level independence. All 248
official files have prior clean exposure and much of the set has fault exposure.

## 3. Complete logical budget curves — useful after baseline readiness

For ResNet-8 and KWS, add the two partial-tier coverages on the existing matched
input populations. For ToyCar, prioritize field precision before enlarging the
budget/BER grid. Freeze plans jointly; do not pick models, BERs or masks based
on the most favorable cells in this draft. Include exact protected cardinality,
expected and realized applied flips, clean/full controls and ranking costs.

## 4. Improve inference, not nominal trial counts

Prespecify separate uncertainty targets: fault fields for fixed masks/cohorts;
random-mask seeds for ranking variability; stratified candidate resampling for
susceptibility; complete-file resampling for ToyCar input variability. With
only four/two random masks, a new population-level superiority claim is not
supported by merely pooling more policies or window observations.

## 5. Do not spend the week on these

New optimizer families, a spectral scaling program, unverified transformer
targets, retraining, or physical mitigation comparisons do not repair the
principal baseline/precision gaps. More favorable small-pool conditional
examples would not resolve the current independent-pool negative result.

## Execution boundary

No new experiment TOMLs are created here: the latest request is manuscript
delivery, and the exact FKeras calibration/solver budget and exposure decisions
still require researcher approval. Existing validated commands and schemas are
documented in `docs/M97_FKERAS_REPRODUCTION.md` and the first-gate evidence
inventory. After the researcher supplies a validated new TOML, execute in the
foreground using the pinned interpreter, for example:

```bash
/projects/kona-assets/manuelbv/.venv-mlperf/bin/python -m netsurf validate <approved-config.toml>
/projects/kona-assets/manuelbv/.venv-mlperf/bin/python -m netsurf run <approved-config.toml>
```

Run from the explicitly chosen worktree with the intended package on its import
path. Do not copy a config into an actively running worktree or assume arbitrary
schema combinations are supported. A runnable publication command can be frozen
only after those choices; no placeholder above is represented as a validated run.
