# Scientific contracts

## Quantization and bit identity

For historical `q<6,2,1>`, the recovered convention is six encoded bits, two integer magnitude bits, one sign bit, three fractional bits, step `0.125`, and range `[-4, 3.875]`. Values are rounded to nearest with ties to even and saturated before encoding. Encoded bit index `0` is the LSB and index `5` is the two's-complement sign/MSB.

For value `w` and bit `k`:

```text
delta(w, k) = decode(encode(w) XOR (1 << k)) - decode(encode(w))
```

Fault injection adds this exact delta to the checkpoint value. It does not silently replace the checkpoint with a fully rounded model. The exhaustive 64-code/384-flip test is mandatory.

## Estimator versus policy

An estimator maps bit identities to scalar scores. It does not choose protected bits. A policy combines scores, ordering constraints, costs, and a budget to produce an immutable mask.

- `first_order_gradient_delta`: `abs(dL/dw * delta(w,k))`.
- `legacy_qpolargrad`: `abs(P(w,k) * dL/dw)`, where dense-kernel `P` is mean input activation times bit delta. In the recovered runtime, downstream layer inputs come from a global all-bit attack. The v2 implementation reproduces the first-call attacked activation source but eliminates accidental persistent `*_N` accumulation.
- `msb_first`: encoded bit index descending. At exactly one bit per scalar, a position-first policy must produce the same mask as correct MSB-first.

No current estimator is claimed globally optimal.

## Fault model

The first milestone uses independent Bernoulli logical weight-bit events, sampled once per Monte Carlo trial and held persistently for the evaluated test subset. Protected events are perfectly suppressed. Base model state is restored exactly after every policy/trial evaluation. Policies in a trial consume the same schedule hash.

This probability is a logical BER parameter, not fluence, cross-section, FIT, SEU rate, or mission-time reliability.

## Statistics

Trial results are independent across explicitly recorded seeds. Policy comparisons are paired by schedule. Reports include trial count, sample mean, sample standard deviation, standard error, and a two-sided Student-t 95% interval. Accuracy intervals are clipped to its physical `[0,1]` range. These intervals characterize the bounded simulation only.

## Result immutability

A finalized run contains a manifest, scientific payload, trial records, exact ranking scores, exact protection-mask identities, exact fault events, a human-readable report, and `FINALIZED` digest. Files are created exclusively and made read-only. Hashes of the exact artifacts are embedded in the scientific payload. A repeated run must produce the same canonical scientific-payload SHA-256; timestamps and run paths are excluded from that digest.
