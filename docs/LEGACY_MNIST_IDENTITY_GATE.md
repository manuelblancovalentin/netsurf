# Legacy MNIST-lite identity gate

## Status and scope

The versioned machine-readable identity is
`netsurf/benchmarks/identities/legacy_mnist_lite.v1.json`. It describes the
checkpoint with SHA-256
`ba48aee8adbe7d04092d74f41958ec99112ae3e5f88ed56552ea9d1c4dba3006`.
The checkpoint and dataset hashes were reverified during the bounded gate.

This file is a recovered benchmark candidate. It is **not** established as the
source of the manuscript's Table II result. The manuscript's value 7,845 is
exactly the checkpoint's nonzero trainable count, whereas the architecture has
7,980 allocated trainable scalars. That correspondence is suggestive but is
not provenance.

## Topology and populations

The recovered path is `28x28x1 -> Flatten -> Dense(10) -> QQApplyAlpha(10) ->
quantized ReLU -> Dense(10) -> softmax`. The checkpoint stores float32 shadow
variables.

| Tensor | Role | Shape | Allocated | Nonzero | Exactly q&lt;6,2,1&gt; | Exact-bit gate treatment |
|---|---|---:|---:|---:|---:|---|
| `fc0/kernel:0` | kernel | 784x10 | 7,840 | 7,707 | 175 | explicitly convert; eligible |
| `fc0/bias:0` | bias | 10 | 10 | 10 | 0 | explicitly convert; eligible |
| `alpha0/alpha:0` | learned scale | 10 | 10 | 10 | 0 | unsupported pending format/folding |
| `alpha0/beta:0` | learned offset | 10 | 10 | 10 | 0 | unsupported pending format/folding |
| `fc_out/kernel:0` | kernel | 10x10 | 100 | 98 | 4 | explicitly convert; eligible |
| `fc_out/bias:0` | bias | 10 | 10 | 10 | 0 | explicitly convert; eligible |
| **Total** |  |  | **7,980** | **7,845** | **179** |  |

The distinct populations are:

- 7,980 allocated trainable scalars / 47,880 historical logical candidates
  when every trainable variable is included.
- 7,845 nonzero trainables / 47,070 bits. This is a value-based count, not a
  physical-storage count.
- 7,960 allocated kernel/bias scalars / 47,760 exact-conversion candidates.
  This population includes all 135 allocated zero-valued kernel words.
- 7,825 nonzero kernel/bias scalars / 46,950 bits.
- 20 learned alpha/beta scalars. Their deployment representation is unresolved
  and they are excluded from the exact kernel/bias population.
- The number of physically implemented storage elements is unknown. Neither a
  Keras allocation nor a zero/nonzero test proves hardware presence or absence.

Prune masks, attack deltas/counters/rankings, the regularization-loss tracker,
and Adam optimizer slots are present in the HDF5 container but are not
trainable inference parameters or fault candidates in this gate.

## Bounded clean-inference comparison

The comparison used `mnist_test[0:256]`, sample identity SHA-256
`b6f23a48d5a82a46dc1525db9a41fad51f9f2e824c13ca2386d9626d997b4eef`.
The preserved TensorFlow 2.11 environment produced bitwise-identical repeated
float-shadow outputs.

| Representation | Accuracy | Cross-entropy | Prediction disagreements | Max output deviation | Max pre-softmax deviation |
|---|---:|---:|---:|---:|---:|
| Original float shadow | 0.886719 | 0.414021 | 0 | 0 | 0 |
| Historical delta-on-float, no fault | 0.886719 | 0.414021 | 0 | 0 | 0 |
| q&lt;6,2,1&gt; kernels/biases + float alpha/beta | 0.894531 | 0.411999 | 5 / 256 | 0.118469 | 0.740838 |
| Float alpha/beta folded into `fc0` (equivalence control) | 0.886719 | 0.414021 | 0 | 0 | 0 |
| Folded then q&lt;6,2,1&gt; kernels/biases | 0.101562 | 3.137367 | 232 / 256 | 0.989407 | 17.439682 |
| Common q&lt;6,2,1&gt; for every trainable (rejected diagnostic) | 0.101562 | 3.137367 | 232 / 256 | 0.989407 | 17.439682 |

No checkpoint trainable is outside `[-4,3.875]`; therefore conversion clips
zero values due to range violation. Forty-two `fc0` kernel values and two
`fc_out` kernel values are already at an encoded range endpoint. Maximum
rounding displacement is 0.062492907. Learned alpha values lie approximately
between -0.0001263 and 0.0065223, so a shared 0.125-step codec rounds every
alpha to zero. This explains the collapse and makes common-codec auxiliary
conversion scientifically unsupported.

The float folding control is algebraically exact for the recovered
`alpha * (xW+b) + beta` operation. Its subsequent six-bit conversion is not
adequate because folding changes the required dynamic range and precision. A
different, explicitly justified folded precision may be viable but was not
selected by this gate.

## Fault-representation conformance

The codec exhausts all 64 words and all 384 single-bit XORs. It also checks all
960 combinations of a code word and two distinct flipped positions. Bit 0 is
the LSB and bit 5 is the two's-complement sign bit.

Across the 47,760 supported kernel/bias scalar-bit cases, historical
delta-on-float and exact decoded XOR agree in 1,074 cases and differ in 46,686.
The agreement count is exactly six flips for each of the 179 already-on-lattice
scalars. For an off-lattice shadow `w`, the difference after a fault is the
unchanged residual `w - decode(encode(w))`; the largest observed absolute
state difference is 0.062492907.

A deterministic schedule applied LSB and sign-bit events to the allocated zero
word `fc0/kernel:0[2]`. With no protection, both proposed events were applied
and the exact state became -3.875. With full protection, both were suppressed
and the state was unchanged. Scenario execution restored every model tensor to
its original value.

## Scientific recommendation

The checkpoint is usable for a **scoped exact-bit kernel/bias comparison** only
after treating the converted hybrid model as a new, explicitly identified
baseline and remeasuring its clean performance. It is not currently usable as
an exact-bit reproduction of the complete historical model because alpha/beta
storage is unresolved. Historical delta-on-float experiments remain valid as
recovered historical semantics but must not be described as exact stored-state
XOR.

Static estimators now have explicit held-out/calibration partitions,
prediction/output interfaces, an exact converted kernel/bias state, and a
47,760-bit identity enumeration. Remaining work before a normal experiment
configuration can use this path is:

1. approve the converted hybrid benchmark identity and auxiliary storage
   assumption;
2. add an explicit schema field selecting the eligible tensor population and
   representation semantics (never infer this from adapter name);
3. record the converted-state digest in result manifests; and
4. decide whether alpha/beta remain trusted float auxiliaries or receive a
   separately validated folded/storage format.

Large Hessian work, training, and fault campaigns were not performed.

## Reproduction

Identity inspection in the canonical Python 3.11 environment:

```bash
proj activate netsurf
netsurf benchmarks identity legacy_mnist_lite --verify-files
```

The historical checkpoint needs the preserved TensorFlow 2.11 environment.
The dedicated module avoids the canonical CLI's Python 3.11-only dependencies:

```bash
proj activate netsurf-legacy
cd <repo>
PYTHONPATH="$PWD" python -m netsurf.legacy_identity conformance \
  --samples 256 \
  --output workspace/runs/legacy_mnist_identity_gate/<new-run>/conformance.json
```

The Python 3.11 `netsurf` environment is intentionally not used to load this
checkpoint: its modern stack does not include the historical
`tensorflow_model_optimization` dependency and the preserved QKeras/Keras-v2
loader is validated only with TensorFlow 2.11. No package was installed or
changed for this gate.

The output is created exclusively and is never overwritten. The validated
bounded evidence from this milestone is under
`workspace/runs/legacy_mnist_identity_gate/bounded-256-v3/` and is gitignored.
