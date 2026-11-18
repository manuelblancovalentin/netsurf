# M9.5.3 same-distribution validation and restricted protection regret

## Why the partition changed

M9.5.2 compared a training-split calibration arm with an official-test
evaluation arm. The observed calibration/test isolated-ranking disagreement
therefore confounded input-population shift with conditional fault effects.
Schema 6.1 does not use that design for mask selection.

The pinned Speech Commands lock contains four hashed test TFRecord shards in a
documented round-robin-block-16 order. The official MLPerf subset and earlier
NetSuRF evidence used original test indices 0--999. M9.5.3 excludes those
indices and fixes the remainder before any mask outcome is inspected:

| Role | Original TFDS test indices | Capacity | Use |
|---|---:|---:|---|
| Conditional calibration | 1000--2023 | 1024 | descriptive conditional-damage analysis |
| Model selection | 2024--3047 | 1024 | isolated and restricted mask construction |
| Final evaluation | 3048--4889 | 1842 | mask comparison on independent backgrounds |

Offsets in a TOML are relative to the relevant fixed window; manifests retain
the original test indices. Every used example records its source split,
`source_split:index` ID, label, encoded-input SHA-256, and a label-bound
partition hash. The engine rejects overlapping source-split/index identities.

The final window is untouched by NetSuRF candidate and mask selection. It is
not literally inference-naive: the pinned asset lock records an earlier
aggregate clean-reference inference over all 4,890 test clips. That aggregate
quality check did not expose per-candidate fault outcomes or select a mask.
This limitation must accompany any confirmation claim.

## Exact restricted oracle

For each outside-network Bernoulli field `G`, the engine excludes all `K`
candidate bits from `G` and evaluates all `2^K` exact candidate states using
the frozen INT8 TFLite `BUILTIN_REF` model. For a fixed protected candidate
mask `M`, it integrates the saved state responses with the exact Bernoulli
probability over unprotected candidate bits. All fixed-cardinality masks are
enumerated.

The restricted empirical optimum is the lowest mean integrated risk over
model-selection outside fields. The isolated comparator protects the same
number of candidates with the largest exact clean-background isolated loss
damage, using the model-selection samples. Both fixed masks are then evaluated
on the same independently seeded final-evaluation outside fields. The primary
paired contrast is

`risk(restricted selected mask) - risk(isolated-bit mask)`.

Negative values favor the restricted mask. A post-hoc best held-out mask and
its regret are retained only as a diagnostic. They are explicitly marked as
selection-biased and cannot be presented as a valid selected policy. No result
is a global optimum over the 176,128 eligible KWS kernel bits. The 618,880-bit
population belongs to the pinned ResNet-8 benchmark, not KWS.

## Work counts and scaling

For each BER, the restricted stage requests

`K + 1 + (B_select + B_final) * 2^K`

state evaluations. The conditional-damage stage adds

`[K(1+2B_cal)+1] + [K(1+2B_eval)+1]`.

The maintained K=6 smoke requests 365 evaluations: 38 conditional and 327
restricted-oracle evaluations. The maintained K=8 template requests 1,755:
210 conditional and 1,545 restricted. Runtime also scales with sample count,
so evaluation count alone is not a wall-time estimate.

## Commands

```bash
<assets>/.venv-mlperf/bin/python -m netsurf conditional validate \
  config/examples/m953_kws_restricted_k6_smoke.toml --resolved

NETSURF_WORKSPACE="$PWD/workspace" \
  <assets>/.venv-mlperf/bin/python -m netsurf conditional run \
  config/examples/m953_kws_restricted_k6_smoke.toml
```

The K=8 file is a researcher-editable bounded template, not an instruction to
launch a publication run automatically:

```bash
<assets>/.venv-mlperf/bin/python -m netsurf conditional validate \
  config/examples/m953_kws_restricted_k8_template.toml --resolved
```

## Bounded K=6 engineering validation

The pinned TensorFlow 2.14.1 environment completed the maintained K=6 smoke in
67.17 seconds for 365 requested evaluations. The finalized bundle is under the
implementation worktree's gitignored `workspace/runs/` and has scientific
payload SHA-256
`3c6656a53df39148afe2f505d70480a312385c45695b349ed1708127ba5d090a`.

The three fixed partition identities were:

- calibration indices 1000--1003, sample SHA-256
  `b2ae886717f321b78f95e5d902624ffe89b361e363aca11567e0102c5c5e589d`;
- model-selection indices 2024--2027, sample SHA-256
  `c03fa90418d5133044ea28793637ea194724e5fb65b5f201bbfd4f1a49925153`;
- final-evaluation indices 3048--3053, sample SHA-256
  `dc611f68262dcdd74bf1487a117cbdedfddd1832db596370b6aee349bf326059`.

At BER 0.01 and budget 2, model selection chose candidate indices `(4, 5)`;
the isolated-bit comparator chose `(0, 1)`. Reintegrating the 327 saved exact
TFLite response records independently reproduced both masks and the three
held-out risks `(1.0783419866, 10.7463980930, 6.2551678058)`, whose mean
`6.0266359618` exactly matches the bundle summary. The paired selected-minus-
isolated mean was `+0.0002285351`; three outside fields are far too few for a
scientific effectiveness conclusion.

The K=8 template requests 1,755 evaluations, but uses substantially larger
sample partitions (32/64/128 rather than 4/4/6). A conservative CPU estimate
from the K=6 measurement is roughly 25--45 minutes and scales linearly in
sample count and outside fields but exponentially in K. Validate the resolved
operation count before increasing either dimension. Response-table storage is
small at this scale (one row per requested restricted state evaluation), while
inference time dominates.

## Interpretation boundaries

- The calibration arm is descriptive; it cannot inspect final outcomes to
  change candidates, budgets, masks, seeds, or hypotheses.
- Final background fields are independent of model-selection fields.
- Confidence intervals use outside-background fields as the sampling units.
- Candidate-state enumeration is exact within each sampled outside field;
  outside-network Bernoulli risk remains Monte Carlo estimated.
- The isolated mask and restricted mask share the same candidate pool and
  fixed budget.
- Logical BER is not a physical SEU rate.
- Ideal candidate suppression is not ECC, TMR, ABFT, or fabricated hardware.
- A post-hoc held-out optimum is diagnostic, not confirmation evidence.
