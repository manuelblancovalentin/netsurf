# Milestone 001 report: deterministic MNIST-lite fault evaluation

## Outcome

Milestone 001 passed its engineering reproducibility criteria. Two independent executions from v2 source commit `85b00d6024144070e08572632a80e8c5ac448ce1` produced byte-identical scientific payloads and canonical scientific hash:

```text
07f4e88d7a36663fb120fafb97c28b80bab5dd3be13b91700755a4784b30bc1b
```

The definitive artifacts are under `/ml/experiments/netsurf/runs/mnist_lite_milestone/commit-85b00d6/`. They are read-only after finalization. The two runs include complete bit scores, exact protected-bit identities, exact fault events, trial metrics, source/environment provenance, and matching artifact hashes.

## Executed configuration

- Checkpoint: `mnist_fnn_lite`, saved 2025-04-14, SHA-256 `ba48aee8adbe7d04092d74f41958ec99112ae3e5f88ed56552ea9d1c4dba3006`.
- Dataset: local Keras MNIST cache, SHA-256 `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`.
- Split: first 256 held-out MNIST test examples for evaluation and estimator inputs.
- Normalization: source `[0,255]` mapped to `[-4,3.875]`, matching historical q<6,2,1> preprocessing.
- Addressable state: 7,980 trainable scalars / 47,880 logical bits, including kernels, biases, alpha, and beta.
- Fault process: independent Bernoulli logical bit events at BER `0.001`, drawn once per trial and persistent across its 256 inferences.
- Protection: exactly 4,788 bits (10%) per policy.
- Sampling: 12 fault schedules, paired across every policy.
- Estimators/policies: historical qPolarGrad with position-first policy, correct MSB-first, seeded random global ranking, and elementary first-order gradient×delta global ranking.

This fault process is a deterministic logical experiment, not a calibrated radiation or SEU-rate model.

## Numerical results

The unfaulted checkpoint obtained accuracy `0.886719` and categorical cross entropy `0.414021` on the fixed 256-example subset.

| Policy | Mean accuracy | 95% trial CI | Mean cross entropy |
|---|---:|---:|---:|
| Legacy qPolarGrad, position-first | 0.876953 | [0.865145, 0.888761] | 0.468398 |
| Correct MSB-first | 0.874349 | [0.862290, 0.886408] | 0.471944 |
| Seeded random | 0.875000 | [0.867010, 0.882990] | 0.447500 |
| First-order gradient×delta | 0.874674 | [0.862312, 0.887037] | 0.469703 |

Paired against correct MSB-first, legacy qPolarGrad improved mean accuracy by `0.002604` (0.26 percentage points; paired `p=0.412`) and reduced mean cross entropy by `0.003546` (`p=0.0031`). Against random, qPolarGrad improved mean accuracy by `0.001953` (`p=0.593`) but had `0.020898` worse mean cross entropy (`p=0.474`). Against first order, its accuracy difference was `0.002279` (`p=0.409`) and cross-entropy difference was `-0.001305` (`p=0.450`). P-values are descriptive for this small bounded study and are not corrected for multiple comparisons.

The full one-MSB-tier qPolarGrad position-first mask and correct MSB-first mask are exactly equal, as required. At 10% coverage their masks differ because qPolarGrad chooses within the MSB tier.

## qPolarGrad assessment

Legacy qPolarGrad is not yet mathematically or experimentally defensible as the accepted v2 estimator.

The milestone confirms that its recovered first-call formula can be computed deterministically as `abs(P_bit * dL/dw)` and that its position-first mask can be reproduced. It does not establish that the activation-drift factor is theoretically valid, that globally attacked activations are appropriate, or that the estimator predicts individual-bit loss effects better than simpler alternatives. Accuracy differences in this experiment are small and statistically unresolved; random has the best mean cross entropy. The paired MSB cross-entropy result is a useful signal for follow-up, not validation.

The next estimator milestone must measure empirical single-bit effects on a smaller exhaustive case, then report rank correlation, top-k overlap, calibration, and matched protection outcomes for clean versus attacked qPolarGrad, first order, gradient magnitude, Hessian, MSB, and random.

## Implemented foundation

- Side-effect-free canonical `netsurf` package (originally reconstructed as `netsurf_v2`).
- Exact fixed-point codec and exhaustive q<6,2,1> tests.
- Immutable bit identities and exact masks.
- Separate estimators and protection policies.
- Replayable event schedules with exact proposed/applied/suppressed counts.
- Checkpoint state context that verifies restoration after every trial.
- Correct sample statistics and confidence intervals.
- Write-once result directories with exact artifact bindings.
- Isolated legacy checkpoint adapter.
- Thirteen dependency-free regression tests.
- Project continuity documentation and private `proj` registration.

## Remaining defects and theoretical questions

1. Learned alpha/beta storage and hardware encoding are unresolved. Replacing all checkpoint parameters with decoded q<6,2,1> values collapses the model; the current injector therefore adds exact bit delta to the original checkpoint value, matching legacy behavior.
2. Historical qPolarGrad's global attacked-activation source and trained-point gradient lack a settled interpretation.
3. Pruned-bit exclusion and the physical status of biases/alpha/beta need a canonical benchmark manifest tied to hardware export.
4. This milestone implements persistent-trial faults; per-inference weight faults and bounded activation faults have schedule support but need execution/evaluation adapters and tests.
5. Hessian and empirical single-bit estimators remain to be implemented in v2.
6. No current result is physically calibrated to radiation exposure.

## proj and Wolf status

The canonical `proj` project is `netsurf`. The original artifact was produced in
the preserved Python 3.10 `netsurf-legacy` environment; environment selection
does not itself lock dependencies or run workflows.

Wolf was inspected at external commit `a3f967b` on branch `multibackend`; it was not modified. Wolf provides declarative ASIC environments, numbered runs, immutable resolved manifests, stage execution, and initial metrics. Its Cadence backend is explicitly a compatibility path. It does not provide the general ML DAG, cache, sweep, or result-store features needed by this software milestone. Wolf integration is therefore deferred to the hardware boundary.

## Exact reproduction command

See `docs/REPRODUCIBILITY.md`. The environment used by the definitive execution
was subsequently renamed `netsurf-legacy`; its current equivalent command is:

```bash
env HOME=/tmp/netsurf_milestone_home CUDA_VISIBLE_DEVICES=-1 \
  TF_DETERMINISTIC_OPS=1 TF_CPP_MIN_LOG_LEVEL=2 \
  /mnt/raid0/ml/conda_envs/manuelbv/envs/netsurf-legacy/bin/python \
  -m netsurf milestone \
  --repo /mnt/raid0/asic/projects/NU/netsurf/manuelbv \
  --output-root /ml/experiments/netsurf/runs/mnist_lite_milestone/commit-85b00d6 \
  --dataset /ml/datasets/mnist/mnist.npz \
  --samples 256 --ranking-samples 256 --trials 12 \
  --ber 0.001 --coverage 0.10 --repeats 2
```

## Next bounded hardware milestone

Use the same frozen checkpoint and 32–64 fixed test vectors. First establish numerical equivalence across the loaded NetSuRF model, correctly folded QKeras export, existing HLS executable/C simulation, and RTL simulation within declared fixed-point tolerances. Then generate a machine-readable mapping from each logical protected bit identity to an RTL register instance. Only after that mapping passes should Wolf receive a project-owned copy of the historical Cadence scripts and a local licensed-PDK manifest for read-only zero/full/selected-point synthesis reproduction.

No EDA run should begin until the checkpoint/export equivalence test, target register list, tool/constraint manifest, and bounded stage command are reviewed.

## Mitigation implementation order

1. TMR: validate the existing register triplication and voter boundary against deterministic replica faults.
2. Parity: define packed storage words and a real detected-error action, then exhaustively test one- and two-bit patterns.
3. SECDED: implement shortened packed-word codewords with SEC, DED, and detected-uncorrectable accounting.
4. Ranger: calibrate activation ranges on a separate calibration split and synthesize comparison/clamp logic.
5. ABFT: select a specific supported GEMM/streaming architecture and define detection, correction, and retry/reload/recomputation semantics before implementation.

This order follows dependency readiness, not a claim that one mitigation is scientifically superior.
