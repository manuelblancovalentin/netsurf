# M9.1: ResNet-8 static-estimator publication evidence protocol

## Status and scientific boundary

This document audits the M9 straight-through companion and prespecifies the
first publication-scale static comparison. It does **not** claim estimator
superiority. The two M9.1 configurations must produce finalized, hashed bundles
before their numerical results can be used in a paper.

The authoritative model is the immutable INT8 TFLite ResNet-8 checkpoint with
SHA-256 `3c002613d1b2475eb51dd78dfb85a546c8ae658dee71cf6ade43b022fe205415`.
Every clean, isolated-bit and multi-fault measurement uses TensorFlow 2.14.1
`tf.lite.Interpreter`, `BUILTIN_REF`, no delegates and one thread. The
differentiable graph is only a ranking surrogate.

## A. Forward and gradient correspondence audit

The M9 aggregate gate reported 197/200 matching class predictions, output-code
agreement 85.05%, probability MAE 0.00423047 and maximum probability error
0.308594. The three class disagreements are:

| Official position | CIFAR-10 index | Stable filename | Label | TFLite class | Companion class | Maximum code error |
|---:|---:|---|---:|---:|---:|---:|
| 30 | 2216 | `cock_s_000054.png` | 2 | 7 | 5 | 79 |
| 82 | 7820 | `maltese_dog_s_000318.png` | 5 | 5 | 3 | 22 |
| 161 | 7041 | `rangifer_caribou_s_000175.png` | 4 | 0 | 2 | 35 |

Position 30 contains the global maximum discrepancy: the authoritative class-7
softmax code is 11 and the companion code is -68, a difference of 79/256. Both
graphs misclassify that image. At position 82 the exact graph is correct and
the companion is wrong; this accounts for the 0.875 versus 0.870 clean-accuracy
difference. Both graphs misclassify position 161.

Operation-level diagnostics show small errors accumulating through the
quantized residual graph. On the three disagreement images, the first
convolution is exact for 99.90--99.92% of activation codes with maximum error
one. The first residual-add output is exact for 97.28--98.08% with maximum
error three. At the last residual-add output, exact-code fractions are
82.15--86.33% and maximum errors are four. Average pooling differs by at most
one code. Final logits differ by at most six, four and three codes for the
three images, respectively. The softmax magnifies these modest logit shifts
when competing classes are close. Preserved-intermediate tensors 24, 28 and 32
are excluded because the reference interpreter reuses their backing buffers
after residual additions; reading them after invocation does not recover their
original operation outputs.

The variable and bit mapping itself is explicit and testable:

- the ten companion variables are created one-for-one from TFLite kernel
  tensors 8--16 and 7, with identical parameter names, C-order shapes and
  dequantized values;
- convolution kernels are transformed only from TFLite OHWI to TensorFlow
  HWIO at the operation boundary, and the dense kernel remains TFLite OI;
- every score uses the matching variable element at the exact TFLite flat
  index;
- `delta_b` is the per-channel affine decode of `code XOR (1 << b)` minus the
  original decode, tested for every INT8 code and all eight bit positions;
- the stored checkpoint and authoritative interpreter are restored after
  every empirical or scheduled fault evaluation.

The remaining approximation is therefore the derivative, not an unidentified
kernel mapping. Gradients are derivatives of mean sparse cross-entropy on the
companion's requantized pre-softmax logits, with identity straight-through
derivatives through activation rounding and clipping. They are not derivatives
of the exact integer TFLite program or of NetSuRF's probability-based empirical
loss. Ranking is defensible as a surrogate hypothesis when local sensitivity
is informative and must be validated against exact isolated-bit measurements.
It may fail near rounding or saturation boundaries, for large sign-bit jumps,
after accumulated residual-path rounding, and when small logit errors are
amplified by softmax. M9's 16 candidates established execution only; they do
not establish predictive usefulness.

## B. Prespecified empirical validation

`m91_resnet8_estimator_validation_stratified320_v1.toml` selects 320 common
candidate identities: four without-replacement samples from every combination
of ten kernel tensors and eight stored bit positions. Within a tensor/bit
stratum of `n` scalars, the recorded inclusion probability is `4/n`. This is a
balanced diagnostic design, not a self-weighting whole-network sample.

Ranking uses CIFAR-10 training indices 0--63. Exact empirical validation uses
training indices 25,000--25,063. Clean final evaluation uses the pinned MLPerf
Tiny official 200-image test subset. These roles are disjoint and their exact
indices and hashes are serialized.

S2 through S6 and the random reference operate on the identical candidates.
The saved tables retain signed loss change, positive loss damage, absolute loss
change and accuracy damage. Rank metrics include Spearman correlation against
positive and absolute damage, Kendall tau-b against positive damage,
tie-aware sample top-k recall, the `k/N` random expectation, layer and bit
position breakdowns, signed-error metrics for S4, and construction time.
Schema 3.3 adds deterministic 1,000-replicate, within-recorded-stratum
bootstrap intervals for overall Spearman correlations. Those intervals measure
sensitivity to the sampled candidates under this design; they are not
population-wide confidence intervals. Sample top-k recall is never labelled
whole-network top-k recall.

## C. Prespecified protection pilot

`m91_resnet8_static_protection_pilot_v1.toml` compares random, corrected
word-relative MSB-first, absolute affine XOR jump, global first-order,
per-example absolute first-order and per-example positive first-order rankings.
All use the same 618,880 kernel bits, 64 ranking samples, authoritative held-out
200-image inference and matched persistent Bernoulli schedules at logical BER
`1e-4`.

The engine supplies single no-protection and full-protection controls. Interior
budgets are 12.5% and 25%. Four replications advance random-estimator seeds by
1,009 and fault-schedule seeds by 100,003; Hutchinson, candidate and sample
seeds remain fixed. Each replication uses eight schedules. Thus fixed-mask
methods have 32 paired held-out fault realizations, while the random family has
four independently constructed masks with eight fault schedules per mask.
Schema 3.3 records these two sources of variation separately. Event proposals,
applications, suppressions, unique applied bits, schedule digests, mask
digests and restoration checks remain part of the immutable payload.

This pilot is exploratory. Pairwise accuracy and cross-entropy effects must use
matching replication, BER, trial and schedule identities. Holm-adjusted paired
tests and confidence intervals may support a follow-up confirmatory design;
they do not by themselves license a final superiority statement.

## Commands and expected cost

Use the validated interpreter explicitly:

```bash
<assets>/.venv-mlperf/bin/python -m netsurf validate \
  config/experiments/m91_resnet8_estimator_validation_stratified320_v1.toml --resolved

<assets>/.venv-mlperf/bin/python -m netsurf run \
  config/experiments/m91_resnet8_estimator_validation_stratified320_v1.toml

<assets>/.venv-mlperf/bin/python -m netsurf validate \
  config/experiments/m91_resnet8_static_protection_pilot_v1.toml --resolved

<assets>/.venv-mlperf/bin/python -m netsurf run \
  config/experiments/m91_resnet8_static_protection_pilot_v1.toml --workers 4 --threads-per-worker 1
```

The M9 16-candidate validation took 2:56.9, peaked at 2.23 GB RSS and produced
464 MB, with exact empirical evaluation averaging 0.1997 seconds per candidate
for 16 images. Linear isolated-bit scaling predicts about 256 seconds for 320
candidates at the configured 64 images before per-example gradients and
serialization. The configured validation is expected to require approximately
6--10 minutes, 2.5--3 GB peak RSS and roughly 0.5 GB output. These are planning
estimates until measured.

The protection pilot evaluates 14 masks (two controls plus six methods at two
budgets) for 8 schedules in each of 4 replications: 448 mask/schedule
conditions and 89,600 reference inferences. Replicated full-score serialization
can approach 2 GB. A four-worker run may require roughly 9--12 GB aggregate
RSS. Run it in `tmux` without `nohup`; reduce workers if memory pressure is a
concern. No publication result is recorded here until the finalized bundle and
scientific payload hash exist.

## D. Finalized susceptibility evidence

The prescribed 320-candidate validation finalized at
`workspace/runs/m91_resnet8_estimator_validation_stratified320_v1/20261029T014136.049835Z-c23f1d77`
with scientific payload SHA-256
`c38df68e491f5bdc96b99ac11343fa2688d3c3bc17c12ec1dd3a31d04c41d4c5`.
Checksums verify. The authoritative clean result is 0.875 accuracy and 0.533269
cross-entropy on the official 200-image subset. All 80 tensor-by-bit strata
contain exactly four candidates. Of 320 exact flips, 186 increased aggregate
loss, 98 decreased it and 36 produced no measured loss change.

| Estimator | Spearman positive damage | Sample-design 95% interval | Spearman absolute change | Top-10 recall | Tie-aware top-10 | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|
| S2 MSB-first | 0.2559 | [0.1729, 0.3403] | 0.3693 | 0.30 | 0.10 | 0.000 |
| S3 absolute bit jump | 0.0695 | [-0.0186, 0.1624] | 0.0416 | 0.10 | 0.10 | 0.887 |
| S4 global first-order | 0.3642 | [0.2851, 0.4510] | 0.5144 | 0.90 | 0.90 | 1.213 |
| S5 per-example absolute | 0.3763 | [0.2913, 0.4594] | 0.5883 | 0.70 | 0.70 | 17.170 shared |
| S6 per-example positive | 0.3643 | [0.2840, 0.4498] | 0.5725 | 0.70 | 0.70 | 17.170 shared |
| Random reference | -0.0540 | [-0.1492, 0.0416] | -0.0379 | 0.00 | 0.00 | 0.000 |

Within this balanced candidate sample, all three gradient methods have stronger
positive- and absolute-damage rank association than the particular random
reference, and S4 recovers nine of the ten most damaging sampled candidates.
S2's deterministic top-10 value is inflated by stable-identity ordering within
its complete sign-bit tie; its tie-aware expected recall is 0.10. S3 is close
to zero correlation, showing that jump magnitude alone is not a strong task
sensitivity proxy in this sample.

These results support taking the straight-through estimators into the held-out
protection pilot. They do **not** establish whole-network top-k performance or
actual protection benefit. The correlation intervals are conditional on the
recorded stratified candidate design and are not formal confidence intervals
for pairwise differences between estimators. The S4/S5/S6 positive-damage
intervals overlap, so this study does not distinguish those three methods.
S5 and S6 share the same 17.17-second per-example Jacobian and their runtimes
must not be added.

The run used 421.86 seconds of coordinator time, peaked at 1.90 GB coordinator
RSS and produced a 445 MB finalized bundle. Derived figures and the expanded
tensor/bit-position report are generated under
`workspace/derived-analysis/m91_resnet8_estimator_validation_stratified320_v1/`.

The paired protection comparison has not been executed automatically. Until a
finalized protection-pilot bundle exists, there is no evidence-backed answer
about which method improves paired accuracy or cross-entropy relative to random
or MSB-first. This is the remaining required experiment, not a reason to weaken
the protocol or substitute susceptibility correlation for protection benefit.

## Publication decision rule

For each interior budget, report paired effects relative to random and
MSB-first in both accuracy and cross-entropy, the 95% paired interval, adjusted
and raw permutation p-values, lower-tail outcomes, mask-construction time and
the number of independent ranking masks. A method is not declared better from
one completed pilot, a point estimate, or non-overlapping marginal intervals.
The answer to the paper's central question remains open until both prescribed
runs are finalized and independently reviewed.
