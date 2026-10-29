# M9.3 cross-benchmark vulnerability-mechanism audit

## Scope

This audit derives exploratory diagnostics from finalized ResNet-8, DS-CNN KWS,
and ToyCar bundles.  It does not rerun inference, alter bundles, fit a new
estimator, or establish a causal mechanism.  Source artifacts used by the first
run were checked against their bundle checksums.  Multi-gigabyte estimator
tables are streamed and only the recorded empirical candidates are retained.
Exact mask overlap is computed by streaming mask JSON into temporary sorted
policy files and merge-joining them; no complete prediction or mask artifact is
loaded into memory.

The reproducible derived output is under the implementation worktree at:

```text
workspace/mechanism-audit/m93_cross_benchmark_initial_v3/
```

It contains `audit.json`, a Markdown report, candidate diagnostics,
heterogeneity summaries, exact mask overlaps, saved paired effects, plots, and
derived-output checksums.  It is not a replacement finalized scientific bundle.

## Initial evidence

### Directional coherence

For every recorded candidate with nonzero S5, the audit calculates

```text
C = S4_signed / S5
```

and independently checks the saved-score identity

```text
S6 = (S5 + S4_signed) / 2.
```

S4 uses one gradient of a mean loss, whereas S5/S6 reduce separately evaluated
per-example gradients.  These operations are mathematically equivalent for the
mean signed contribution but are not bitwise identical in float32.  The audit
therefore records both absolute/relative residuals and the coherence implied by
S5/S6.  ToyCar has 28 zero-S5 candidates and a maximum observed coherence
disagreement of about `0.091`; this is a numerical/reduction diagnostic, not a
new estimator result.

On the stratified 320-candidate samples:

| Benchmark | Spearman(C, exact signed loss change) | Mean C harmful − beneficial (95% stratified-bootstrap CI) |
|---|---:|---:|
| ResNet-8 | 0.019 | -0.002 [-0.071, 0.076] |
| KWS | -0.111 | -0.134 [-0.230, -0.029] |
| ToyCar | 0.437 | 0.678 [0.600, 0.764] |

These are sample-limited associations.  They do not show that coherence causes
protection benefit, and the architectures use different tasks and losses.

### Encoded-bit and tensor heterogeneity

Sign/MSB tiers are more damaging on average in every sampled architecture, but
within-tier and tensor heterogeneity remains substantial.  Population-weighted
and unweighted estimates differ materially for some strata, confirming that the
stratified 320-bit sets cannot be read as uniform whole-network samples.

The exact S2/S5 mask Jaccard ranges across saved budgets/replications were:

| Benchmark | Minimum | Maximum | Mean |
|---|---:|---:|---:|
| ResNet-8 | 0.421 | 0.621 | 0.521 |
| KWS | 0.424 | 0.608 | 0.516 |
| ToyCar | 0.248 | 0.403 | 0.325 |

ToyCar therefore reallocates more of the corrected MSB mask under S5, but this
fact alone does not establish that the reallocation improves ROC-AUC.

### Loss/task alignment

The sample-limited Spearman association between exact isolated loss damage and
the saved primary-metric damage was approximately 0.305 for ResNet-8 accuracy,
0.227 for KWS accuracy, and 0.317 for ToyCar ROC-AUC.  ToyCar's per-window
reconstruction loss is therefore not interchangeable with file-level ROC-AUC.
Existing finalized runs do not retain per-file clean/faulted scores, so normal/
anomalous score-order reversals cannot be reconstructed retrospectively.

`write_toycar_file_score_artifact` is an opt-in API for a bounded future run. It
records deterministic sample IDs, labels, raw float64 file scores, exact fault
accounting, and checksums.  Ordinary production bundles remain unchanged.  Held-
out final labels must not be used to fit a label-dependent ranking surrogate.

### Protection evidence

The saved paired results show a strong architecture/BER/budget interaction in
the *observed outcome*, without identifying its cause:

- ResNet-8: at 12.5% coverage S5 minus corrected S2 accuracy was `+0.0414`
  at BER `1e-3` (95% CI `[0.0237, 0.0592]`) and `+0.3239` at BER `1e-2`
  (`[0.2717, 0.3761]`), both with saved within-BER Holm-adjusted
  `p=0.0042`.
- KWS: most low-BER S5/S2 differences include zero.  At BER `1e-2`, 12.5%
  coverage, S5 minus S2 was `+0.0352` (`[0.0110, 0.0595]`), but the saved
  Holm-adjusted value was `0.0584`.  At 25% the difference was near zero.
- ToyCar: only one ranking replication and eight paired fault fields are
  available.  S5 minus S2 at BER `1e-2` was positive at 12.5% and 25%, but the
  saved Holm-adjusted values were `1.0`; these are screening observations, not
  confirmatory evidence.

The complete condition table is in the derived `report.md` and
`paired_effects.csv`.  These differences cannot distinguish nonlinear fault
interaction from saturation, task-metric alignment, mask composition, or
network redundancy.

## What the current data cannot establish

- The saved candidate studies measure isolated faults; protection studies
  measure fixed masks under multi-fault fields.  They do not observe each bit's
  marginal protection benefit under the same background fields.
- Activation traces, ReLU gate changes, and matched layer ablations were not
  saved.  Cross-architecture comparisons cannot causally identify gating or
  redundancy.
- Repeated fault schedules under one mask are not independent ranking-mask
  replications.  ToyCar currently has only one finalized ranking mask per
  nonrandom method.
- Zero-valued stored weights are allocated fault candidates.  They are not
  physically removed connections and do not prove redundancy.
- No universal estimator is supported by these observations.

## Prioritized controlled experiments

1. **ToyCar repaired parallel pilot.** Copy the researcher-owned v3 pilot to a
   new configuration named `m93_toycar_static_ber_pilot64_parallel_repair_v1`.
   Preserve all scientific fields and seeds; use two execution workers only as
   an execution setting. Confirm identical sequential/parallel payloads before
   increasing trials.
2. **Independent mask replications.** Proposed TOMLs:
   `m95_resnet8_static_mask_replications_v1.toml`,
   `m95_kws_static_mask_replications_v1.toml`, and
   `m95_toycar_static_mask_replications_v1.toml`.  Advance estimator/random-mask
   seeds separately from fault seeds and retain 12.5%/25% budgets with matched
   held-out fields.  Power should be selected from pilot paired variances.
3. **Isolated/background factorial diagnostic.** For a prespecified uniform and
   stratified candidate set, evaluate exact isolated damage and the exact
   marginal benefit of suppressing that same bit under common Bernoulli
   backgrounds at BER `1e-4`, `1e-3`, and `1e-2`.  Keep final fields independent
   of selection.  This is the minimum intervention needed to test whether
   isolated susceptibility ceases to predict conditional protection benefit.
4. **ToyCar file-score ordering diagnostic.** Use a bounded calibration split
   to write per-file clean/faulted scores and count normal/anomalous order
   reversals.  Do not use final-evaluation labels to construct a ranker.
5. **Activation/gating intervention.** Only after the prior gates, record layer
   outputs and ReLU gate changes for the same exact bit identities on calibration
   data.  Test within each frozen architecture.  A pruning/ablation intervention
   requires a new checkpoint identity and cannot be inferred from weight zeros.

These could support a mechanism-focused paper only if the isolated/background
intervention reproduces across independently seeded candidate sets and masks.
The current observational audit is useful motivation, not that paper's causal
result.

## Hessian status

No M9.4/Hessian branch or worktree existed at the start of this milestone.
NetSuRF retains earlier dense/NumPy exact-diagonal and Hutchinson infrastructure,
but the M9.3 TFLite companion adapters explicitly raise `NotImplementedError`
for Hessian-vector products.  No Hessian code was added here.  A separate bounded
extension would first need an architecture-specific HVP correspondence test,
probe convergence diagnostics, memory estimates, and exact-bit empirical
validation on a restricted candidate set.

## CLI

The reusable interface is:

```bash
netsurf mechanism-audit \
  --benchmark NAME /path/to/estimator-bundle /path/to/protection-bundle \
  --output workspace/mechanism-audit/NAME
```

Repeat `--benchmark` for multiple architectures.  Checksums of every used source
artifact are verified by default. `--skip-checksums` exists only for a repeated
presentation pass after the same immutable artifacts have already been checked.
