# M9.5 exact conditional-damage evidence

## Scope and definitions

M9.5 adds a bounded evidence protocol; it does not define M10 or a new
universal susceptibility estimator. For task risk `R`, candidate bit `b`, and
a Bernoulli background field `G` over all eligible bits except `b`, the
authoritative quantity is

`D_b(G) = R(G ∪ {b}) - R(G)`.

Both members of a pair use the same samples and background. The candidate is
removed from the sampled master field before either evaluation, including when
the master draw happened to select it. Faults are exact XOR operations on the
frozen INT8 TFLite kernel buffer. Calibration and held-out evaluation use
different source splits and different background seeds. Reported intervals
therefore describe background-field variation conditional on the frozen model,
candidate list, samples, and logical BER. They do not describe candidate
sampling, model training, or physical-radiation uncertainty.

Schema `6.0.0` is deliberately separate from protection schemas. The CLI is:

```bash
/projects/kona-assets/manuelbv/.venv-mlperf/bin/python -m netsurf conditional validate \
  config/examples/m95_kws_conditional_smoke.toml --resolved

NETSURF_WORKSPACE=/path/to/workspace \
  /projects/kona-assets/manuelbv/.venv-mlperf/bin/python -m netsurf conditional run \
  config/examples/m95_kws_conditional_smoke.toml
```

The maintained example prespecifies ten candidate identities spanning all ten
KWS kernels and several bit positions. It is a smoke test, not a population
sample. `artifacts/background_fields.json` records the exact common fields and
checksums; `measurements/conditional_damage.csv` retains every pair and signed
effect; the finalized payload excludes timestamps and absolute output paths.

## Restricted exact subcube

`netsurf.restricted_subcube.select_restricted_subcube_mask` exactly integrates
all `2^K` candidate-subcube states for each fixed outside background, enumerates
all masks of a specified restricted budget, selects only on calibration
backgrounds, and reports risks on separate held-out backgrounds. It is bounded
to `K <= 12`. Its result is a restricted candidate-pool optimum, never a
whole-network optimum. The KWS smoke leaves this optional stage disabled until
the researcher specifies a candidate-selection rule and sufficient independent
background fields.

## FKeras reconstruction

The implemented baseline follows the paper-form principal weight-Hessian
score

`H' = sum_i lambda_i (v_i^T theta) v_i`,

using deterministic orthogonalized power iteration and continuous companion
loss Hessian-vector products. Bits are ordered first by word-relative MSB tier
and then by the absolute parameter score. This is not a Hessian diagonal, a
trace estimate, or a Hessian of the exact integer TFLite program.

The public FKeras repository was inspected at commit
`68e775c77670c9a1aab4e1b5295b523347d97013`. Its `do_sum_hessian_rank`
implementation appears to multiply the eigenvalue once when storing the
eigenvector and again during accumulation, yielding an effective squared
eigenvalue factor. NetSuRF implements the single-eigenvalue expression printed
in the paper and records this divergence rather than silently treating the two
as identical. A bounded KWS check mapped all 10 kernel tensors / 22,016
parameters, produced finite deterministic HVPs, and obtained HVP SHA-256
`c30fe5d3ecdfe2fdd78d8e52a1ad9fd26430a16390a9223230211c89202404bf`.
No full KWS eigensolve or protection experiment was run.

## Bounded KWS pilot evidence

The smoke completed 244 planned paired evaluations in 6.47 seconds under the
pinned TensorFlow 2.14.1 `BUILTIN_REF` environment. Its scientific payload is
`f17771e842941010ff928fa81eb5fe8c79ac159399b405c4bc5e8da786c3c3c2`.
At BER zero, isolated and conditional orders agreed as required. At BER
`1e-3`, the tiny pilot observed 21/22 strict calibration pair reversals and 8/9
held-out reversals; 7/20 calibration and 5/30 held-out signed conditional
effects were negative. These results validate that the machinery can expose
context dependence and beneficial effects. With only two and three background
fields and very small sample partitions, they are not stable effect estimates
and do not establish a protection-policy advantage.

## Scaling and next researcher-run gate

Exact conditional evaluation scales approximately as
`splits * BERs * K * (1 + 2 * background_fields)` authoritative inferences,
with reuse when exact schedules repeat. Exact restricted subcube evaluation
scales as `outside_backgrounds * 2^K` unique states per split before mask
integration. The next defensible TOML should retain the ten fixed candidates,
increase held-out samples and independently seeded background fields, and be
frozen before inspecting results. Only after stable rank changes appear should
the restricted subcube be enabled on a prespecified small pool.

## Scientific boundaries

- Logical BER is not an SEU rate.
- Exact encoded-XOR TFLite inference remains authoritative.
- Companion Hessians are derivatives of a continuous STE surrogate.
- Negative `D_b(G)` values are retained, never clipped.
- Candidate-list conclusions are not population-wide conclusions.
- Restricted-subcube optima are not whole-network optima.
- No MCMC, guided search, activation faults, mitigation hardware, training, or
  publication-scale sweep is part of M9.5.
