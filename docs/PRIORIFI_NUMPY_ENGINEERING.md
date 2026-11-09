# PrioriFI scheduling: bounded NumPy engineering gate

This is an engineering validation, **not** an additional publication protection
campaign or evidence that PrioriFI is a protection optimizer.

## Method and source boundary

The [public PrioriFI example](https://github.com/KastnerRG/priorifi/blob/6e7637b92315e5df75bb6ce258972bdf1131bc20/examples/smart-pixel/fault_injection_campaign.py)
at commit `6e7637b92315e5df75bb6ce258972bdf1131bc20` supplies a
feedback-guided order for **isolated fault-injection measurements**. It first
Hessian-ranks parameters, places their bits into significance-tier queues,
probes one bit from every nonempty tier, then repeatedly chooses a tier from
recent measured feedback and probes that tier's next bit. Its complete example
ultimately probes every parameter bit. It does not specify fixed-budget
protection-mask construction.

`netsurf.priorifi.significance_tiers` and `probe_schedule` reproduce this
scheduling structure with canonical NetSuRF bit identities. The source's branch
is preserved: use the median of the last `k` scores when the **remaining tier
length** is at least `k`; otherwise use the mean of the retained scores. The
source does not prescribe a portable tie-break, so this implementation chooses
lower significance-tier queue index on equal feedback. `max_probes` permits a
bounded prefix without changing its order. It is not a statistical correction
for adaptively selected bits.

The NumPy adapter uses the exact 17-dimensional continuous-loss Hessian on its
two calibration examples. The eight eigenpairs with largest absolute
eigenvalue feed the pinned **repository-form** FKeras seed score, matching the
public PrioriFI example's FKeras-package initialization in mathematical form.
This is not the published-form FKeras protection baseline and is not a claim
that the NumPy or exact integer fault program has the source QKeras model's
Hessian. The score is used only to set the *within-tier probe order*.

The feedback metric is the number of predictions changed by one exact encoded
XOR bit among initially correct calibration inputs, as in the public example.
Every injection restores the clean model. Calibration input indices `[0, 4]`,
empirical bit-validation indices `[1, 5, 6]`, and final matched-fault input
indices `[2, 3, 7]` are disjoint. This packaged reference is an engineering
fixture, not untouched publication data.

`measured_bit_allocation` is a **separate NetSuRF adaptation**: rank probed
bits by observed nonnegative prediction-change count, tie-breaking by probe
sequence. It cannot allocate more bits than it has measured. After exhaustive
probing it is simply an exact empirical isolated ranking for that feedback
metric; PrioriFI has changed measurement order, not the set of measurements.
The allocation does not infer unprobed bits' damage and is not a published
PrioriFI protection rule.

## Reproduce the 102-bit engineering application

From the isolated implementation worktree, using the existing environment:

```bash
/mnt/raid0/ml/conda_envs/manuelbv/envs/netsurf/bin/python -m pytest -q tests/test_priorifi.py
/mnt/raid0/ml/conda_envs/manuelbv/envs/netsurf/bin/python -m netsurf.priorifi_numpy_reference \
  --output workspace/runs/priorifi_numpy_reference_engineering_v2
```

The output directory must not already exist. The receipt stores asset and
partition hashes, the complete probe sequence and feedback, separate
calibration/validation loss changes, exact mask identities, paired field
hashes, proposed/applied/suppressed counts, clean results, runtime and a SHA-256
checksum. `v1` is the write-once local engineering result from this work.

The `v2` result measured all 102 bits in about 0.054 seconds and evaluated 12
independent proposed Bernoulli fields at `p=0.05` under each of four masks (48
policy evaluations). Sixteen bits changed one calibration prediction, one bit
changed two, and 85 changed none. The derived 17-bit mask overlaps the 17-bit
MSB mask in eight bits. On three final fixture inputs, mean accuracies were:
unprotected 0.750, MSB 0.833, derived 0.833 and full protection 1.000. Mean
cross-entropies were 0.7514, 0.5695, 0.6411 and 0.3260, respectively. These
small, tie-heavy engineering figures do **not** establish method superiority.
The exact signed validation loss changes show both damaging and beneficial
faults; selecting by prediction-change count is not equivalent to selecting by
cross-entropy damage.

The eighth Hessian mode on this two-example calibration set is numerically
null (eigenvalue around `-2.5e-13`). Eigenpair residuals are therefore checked
as matrix-scale backward errors, not divided by the near-zero eigenvalue. The
receipt records the eigenvalues, residuals and orthogonality error. The `v1`
engineering receipt was not modified when this diagnostic was added.

## Publication-benchmark gate

The smallest validated INT8 publication benchmark is KWS, with 176,128 eligible
kernel bits. A full source-style PrioriFI traversal would require 176,128
isolated-fault measurements **plus** Hessian initialization per calibration
cohort; the exact application cannot be equated with an inexpensive ranker.
Before any KWS effectiveness campaign, freeze a calibration-only probe budget,
feedback metric, deterministic rule for unmeasured eligible bits, mask budgets,
ranking-cost receipt and independent final fields. A cost-capped prefix alone
is not a complete ranking over 176,128 bits. No KWS PrioriFI-derived protection
effectiveness is measured or claimed here.
