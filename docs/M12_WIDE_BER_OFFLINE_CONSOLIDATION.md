# M12 W8A8 wide-BER offline consolidation

This is a derived database and analysis of **saved** observations. It neither
reruns inference nor changes ranking masks, fault schedules, scientific source
bundles, or the preceding comparative database. The Python entry point is
`scripts/consolidate_wide_ber_evidence.py`; its arguments identify the read-only
v3 comparative SQLite database and the two saved scientific study roots.

## Source status and accounting

| Study | Source status | Fields per BER | Paired fields | Policy observations | Conditions | Paired contrasts |
|---|---|---:|---:|---:|---:|---:|
| DistilBERT/SST-2 W8A8 | `FINALIZED_MEASURED` | 24 | 192 | 4,992 | 208 | 160 |
| Pythia-160M/WikiText-2 W8A8 | clean-conformant derived subset; original 24-field campaign **not finalized** | 18 | 144 | 3,744 | 208 | 160 |

Both studies use eight logical BERs from 1e-5 through 1e-1, the same four
interior fractions (6.25%, 12.5%, 18.75%, 25%), six ranking methods, and
unprotected/full-protection controls. One constructed ranking mask per method
and budget is fixed across fields. Confidence bands in the plots are marginal
95% field-bootstrap intervals for condition means. Saved paired comparisons
match identical BER and proposed-field identities and apply Holm correction
over 160 within-study contrasts against MSB-first. These intervals condition
on the frozen model, cohort and masks; they do not measure independent mask,
checkpoint, or input-population variability.

DistilBERT's manifest marker, payload digest, and all 5,390 recorded scientific
file hashes verified. Its plan payload is
`20756b6dc02a898a1db5cf892c17a9962115e7303e5e1f33c40ff511b35322f4`.
Pythia's previously audited, write-once recovery manifest is
`bb0c12af797fbbc0981ca199a88656c3f0ae02b95a35b31716cd3f1cb71006bb`.
The Pythia source's shard 0 failed the exact clean-conformance gate. Its 1,248
saved policy observations are quarantined, not discarded or reported as a
finalized study. The derived subset selects shards 1–3 by conformance status,
not by ranker outcome; nevertheless it is exploratory, not the complete frozen
24-field campaign.

The consolidation independently recomputes every condition mean, sample
standard deviation, fault count, control equality, and paired difference from
saved trial rows. It checks all source manifest hashes, CSV cardinalities,
paired fault identities and finalization status. The old 864-row
`current_comparative_conditions` SQLite view is retained in the new copied
database. The new `current_wide_ber_conditions` view contains 416 rows, while
`finalized_wide_ber_conditions` (DistilBERT) and
`exploratory_wide_ber_conditions` (Pythia) prevent accidental status pooling.
No old and new low-/wide-BER field observations are pooled.

## Initial interpretation

The new exposures locate substantial degradation that the earlier 4–256
expected-fault grid did not. At BER 0.01, DistilBERT's unprotected mean
SST-2 cross-entropy is 0.87629 nats versus 0.37473 clean. At 12.5% protection,
per-example absolute-gradient allocation is 0.3560, MSB-first is 0.4571,
and per-example positive-gradient allocation is 2.1040. The saved
absolute-minus-MSB paired difference is -0.10111 nats, 95% field-bootstrap
interval [-0.12125, -0.08068], 24/24 favorable fields and Holm-adjusted
randomization p approximately 0.016. This is strong *within the tested fixed
model, cohort, mask and BER*; it does not identify the causal mechanism.

At BER 0.01, Pythia's unprotected mean WikiText-2 token NLL is 22.45675 nats
versus 3.52956 clean. At 12.5% protection, absolute-gradient allocation is
8.3685 versus 13.1273 under MSB-first. Its paired difference is -4.75884
nats, interval [-5.26824, -4.23934]. At 25% protection the direction reverses:
6.6254 versus 5.2629, difference +1.36252 nats, interval [1.06784, 1.69227].
Both contrasts have 18/18 fields in the corresponding direction and saved
Holm-adjusted p approximately 0.016. These are striking budget-dependent
observations, but the full Pythia campaign did not finalize, so a conformant
independent repetition is needed before a confirmatory publication claim.

At the highest BER, model behavior approaches a task-specific saturation
regime. DistilBERT unprotected accuracy approaches 50% on this binary cohort;
Pythia token losses approach the frozen 30-nat **per-token cap**. Zero
`failed_units` means there were no reported nonfinite inference failures;
it does not mean that the cap was never reached by finite losses. These BERs
are software stress conditions, not radiation-calibrated SEU rates. The two
losses are both logarithmic but arise from different tasks and must not be
averaged or plotted on a shared quantitative scale.

Observed ranker differences and budget-dependent reversals establish that
allocation matters under these stress conditions. They do **not** by
themselves isolate higher-order bit interactions, show that static masks
cannot be effective, or establish superiority of an MC-guided search. The
wide grid was selected after a separate engineering pilot and remains
exploratory until replicated with independent masks/cohorts and prespecified
contrasts.
