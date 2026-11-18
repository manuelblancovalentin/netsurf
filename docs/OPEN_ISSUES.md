# Open issues

## Scientific blockers

1. The recovered checkpoint stores learned alpha values near zero as floating point; wholesale fixed-point replacement collapses accuracy. The first milestone therefore adds exact encoded flip deltas to checkpoint values, matching legacy injection. Hardware export must establish how alpha/beta are folded and encoded.
2. Legacy qPolarGrad uses globally corrupted downstream activations and a trained-point gradient. Its theoretical interpretation and empirical rank correlation with single-bit effects remain unvalidated.
3. Position-first qPolarGrad and correct MSB-first are necessarily identical at full one-MSB coverage; historical contrary comparisons require regeneration.
4. A physical fault-exposure model, logical-to-physical register map, and trustworthy zero/full synthesis endpoints are absent.
5. KWS and ToyCar STE companions are not exact integer-program derivatives.
   Their ranking value must be established through exact TFLite isolated-bit
   and protection studies on independent samples.
6. VWW evaluation-image provenance is unresolved; local derivative images are
   not accepted as the official MLPerf benchmark dataset.

## Engineering blockers

- The canonical legacy loader calls undefined `strip_pruning`; v2 currently uses its custom-object map directly.
- `pytest` is optional; `python -m tests.run` is the dependency-free test command.
- The exact milestone stack is TensorFlow 2.11.1 on Python 3.10. The canonical
  environment is Python 3.11.17, for which TensorFlow 2.11 wheels do not exist.
  Do not declare a newer TensorFlow equivalent until the payload is compared.
- `proj` shell integration and Lmod function are not initialized in the execution shell, but the private metadata profile is valid.
- Wolf exists at `<home>/wolf` commit `a3f967b` on `multibackend` with two unrelated modified TSMC28 templates. It is not on PATH. No NetSuRF Wolf manifest has been created yet.
- Wolf provides numbered immutable ASIC runs and provenance, but not a general ML dependency solver, sweep scheduler, or content-addressed stage cache.
- ToyCar preprocessing requires librosa, which is not installed in every
  NetSuRF Python environment. The validated pinned reference environment has it;
  environment changes remain researcher-managed.
- ToyCar's 2,113,536 Python bit identities and score maps have significant
  memory cost. Bounded validation peaked near 880 MiB before publication-scale
  result serialization.
