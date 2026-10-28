# MLPerf Tiny image-classification M7.1 conformance evidence

Date: 2026-10-28. These are bounded integration measurements, not a research
campaign or publication result.

## Immutable inputs

- Kona Assets lock digest:
  `sha256:f953370a8b98f7a016e6907e4b001f3b099bbe216c3202d0306fbbca00702687`
- Model SHA-256:
  `3c002613d1b2475eb51dd78dfb85a546c8ae658dee71cf6ade43b022fe205415`
- CIFAR-10 archive SHA-256:
  `6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce`
- Test-batch SHA-256:
  `f53d8d457504f7cff4ea9e021afcf0e0ad8e24a91f3fc42091b8adef61157831`
- Official 200-index SHA-256:
  `3bd4a88eeb4c50fad652d0f24c8af13bc9219ba2878aea47c6536bfbeb43024d`
- Official label-index SHA-256:
  `79e1f19ad731f046983a1a4480bda6369df26c5acee47505a860f164fb58ccd4`

The neutral lock's 27 named files were independently size- and SHA-256-
verified by NetSuRF. The first official indices are 7657, 8959, 3596, 327 and
2783; their CIFAR filenames and labels match every corresponding row of
`y_labels.csv` after the documented `.png` to `.bin` name conversion.

## Numerical backend comparison

All executions used one thread. `BUILTIN_REF` and
`BUILTIN_WITHOUT_DEFAULT_DELEGATES` requested no delegates explicitly. The
AUTO path created the normal XNNPACK delegate. Output digests are SHA-256 over
the C-contiguous INT8 array. Cross-entropy is
`mean(-ln(max((output[label] + 128) / 256, 1e-12)))`; integer softmax outputs
are not renormalized.

### Official 200 images

| Runtime / resolver | Output SHA-256 | Accuracy | Arrays changed vs TF 2.14 reference | Elements changed | Max INT8 difference | Predictions changed | Cross-entropy |
|---|---|---:|---:|---:|---:|---:|---:|
| TensorFlow 2.14.1 `BUILTIN_REF` | `f045f8f1...38b9` | 0.875 | 0 | 0 | 0 | 0 | 0.5332691329 |
| TensorFlow 2.21.0 `BUILTIN_REF` | `64003b1f...7c4a` | 0.875 | 5 | 13 | 8 | 0 | 0.5332815448 |
| TensorFlow 2.21.0 optimized built-in | `ffc942ef...24b` | 0.870 | 87 | 250 | 54 | 3 | not measured |
| TensorFlow 2.21.0 AUTO/XNNPACK | `081ac82f...201` | 0.870 | 98 | 296 | 79 | 3 | not measured |

### Complete 10,000-image test set

| Runtime / resolver | Output SHA-256 | Accuracy | Arrays changed vs TF 2.14 reference | Elements changed | Max INT8 difference | Predictions changed | Cross-entropy |
|---|---|---:|---:|---:|---:|---:|---:|
| TensorFlow 2.14.1 `BUILTIN_REF` | `2cd2747d...fd1` | 0.8707 | 0 | 0 | 0 | 0 | 0.5521686722 |
| TensorFlow 2.21.0 `BUILTIN_REF` | `fbeb1a2e...617` | 0.8707 | 238 | 534 | 11 | 1 | 0.5521185166 |

Conclusion: resolver name alone is insufficient numerical provenance.
TensorFlow 2.21 `BUILTIN_REF` retains aggregate accuracy but is not an exact
replacement for the registered TensorFlow 2.14.1 backend. Reference scientific
runs must use the recorded runtime or establish a separately reviewed backend
identity. Lock-backed experiment execution enforces the official 200-output
digest at adapter load and fails before ranking or fault evaluation when it
does not match.

## Storage and fault conformance

The complete eligible storage contains 77,360 INT8 kernel words (618,880 bits)
and 346 INT32 bias words (11,072 bits), for 629,952 bits. NetSuRF exposes two
distinct populations: kernels only, and kernels plus biases. Faults XOR actual
little-endian storage bytes in a private flatbuffer copy. Bounded tests cover
INT8/INT32 transitions, two flips in one scalar, zero/full protection, proposed
and suppressed event counts, restoration and repeated-output determinism on
real CIFAR-10 samples. Equal BER across storage classes remains an idealized
logical assumption.

## Bounded empirical check

Six prespecified bits (kernel and bias, low and sign positions) were evaluated
on the first eight official images using TensorFlow 2.14.1 `BUILTIN_REF`.
Sample identity digest:
`d07bdf8d33f96180f3371d8856501fddb8ad5365b03e9918c1431a2dbd7011d2`.
Clean accuracy was 0.75 and clean cross-entropy 3.8119322184. Four selected
low-order flips produced no output-level metric change on this tiny sample.
One kernel sign flip increased loss by 0.0769248035 without changing a
prediction. One INT32 bias sign flip changed all eight predictions, reduced
accuracy from 0.75 to 0, and increased loss by 23.8190888975. This is a
conformance illustration over a prespecified six-bit subset, not an exhaustive
or statistically representative network ranking.

## Gradient boundary

Gradient, Hessian and qPolarGrad estimators remain blocked. A later gate must
map float checkpoint tensor names/layouts to fused TFLite constants, reproduce
per-channel folding, establish clean-output equivalence, and compare
representative exact-bit fault effects. Existence of the HDF5 companion is not
sufficient.
