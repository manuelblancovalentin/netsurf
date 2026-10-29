# ToyCar parallel conformance repair

## Failure and bounded reproduction

The schema-3.4 ToyCar adapter passed its two-file bounded conformance gate in a
sequential process but failed during spawned replication initialization when
`threads_per_worker=1`.  A two-replication engineering configuration reproduced
the behavior with identical scientific seeds: one execution worker finalized,
while two workers failed during `MlperfTinyToyCarAdapter.load()` before any
fault trial was evaluated.

The diagnostic replay records process/runtime identity separately from the
scientific manifest and records the following deterministic chain:

| Quantity | Exact value |
|---|---|
| Selected official file indices | `1, 141` |
| Sample identity SHA-256 | `f06ed6c6bbb2ac595f316eb8e4ab8a6be63fcc22a539be4db2593fe4e50eb99d` |
| Preprocessed INT8 input SHA-256 | `0a65d3fee9fa0519850298c1e17dff1e785f63aa8a2be37702bfec30e1fc7eed` |
| Exact TFLite output SHA-256 | `b2e66aff043b9c8c9acfc2ec4bd7a6008e6ecdb6a8fd92e685b6602510af2202` |
| Expected float64 file-score SHA-256 | `cd135be4abd970a9856891741a8fb09b7f67af6bd0552a9c23be28c848ef9fb1` |

The pinned reference file scores are exactly
`10.168487922577622` (`0x1.456440c8a7bd6p+3`) and
`11.777840215450889` (`0x1.78e41129dc015p+3`).  Under the spawned
one-thread OpenBLAS environment the corresponding values became
`10.168487927031869` and `11.777840213940504`.  The INT8 inputs and authoritative
TFLite outputs remained byte-identical.  Therefore the divergence did not
originate in sample selection, seeds, encoded preprocessing, TFLite inference,
or file aggregation.  It originated in the float64 librosa mel computation
used both as a pre-quantization input and as the reconstruction target.

## Root cause and correction

The pinned NumPy wheel uses an OpenBLAS build with a compile-time default of two
threads.  The original full-conformance and bounded score digests were generated
under that mode.  NetSuRF's spawn initializer deliberately sets numerical
libraries to `threads_per_worker`; changing OpenBLAS to one thread changes the
float64 mel target by a few ulps.  Those changes survive reconstruction-MSE
aggregation even though they do not cross an INT8 input quantization boundary.

ToyCar preprocessing now enters a narrowly scoped `threadpoolctl` context with
two BLAS threads, reproducing the already pinned numerical contract.  TFLite
inference remains TensorFlow 2.14.1 `BUILTIN_REF`, no delegates, one thread.
The asset locks, expected digests, model, dataset, file selection, score
definition, and exact-equality requirement are unchanged.  Missing
`threadpoolctl` is an explicit dependency failure rather than a relaxed gate.

`bounded_reference_diagnostics()` exposes process identity, runtime versions,
thread environment, selected IDs, input/output digests, raw score values and
hexadecimal representations, expected digest, and the first bitwise-different
score when present.  Process-specific fields are diagnostic only and are not
placed in the scientific payload.

## Interpretation

This was a reproducibility defect in the preprocessing runtime contract, not a
scientific effect of multiprocessing or a failure of exact INT8 TFLite
inference.  The repair does not authorize numerical tolerance and does not make
logical BER a physical upset-rate model.  Publication runs must continue to use
the pinned TensorFlow/librosa/NumPy environment and exact asset locks.
