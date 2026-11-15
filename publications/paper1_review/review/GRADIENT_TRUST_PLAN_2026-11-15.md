# Gradient-trust test: can a fault-free check tell which ranking to trust? (plan, 15 November 2026)

**Status:** draft plan, written at the researcher's request ("write the plan"). Nothing in it has been run. The
predictions in Section 4 become binding when Phase 1 is committed, before any fault run of Phase 2.

## 1. The hypothesis, in plain words

Two rankings choose which bits to protect:
- **pos1024** ranks bits by the signed gradient. It favours bits whose flips raise the loss on average.
- **abs1024** ranks bits by the gradient's size and ignores its sign.

What we have seen:
- **ResNet-8:** pos1024 wins at low fault rates and abs1024 at high rates. The formula predicted where they switch
  to within about 12% (Stage 1, confirmed).
- **DS-CNN:** there is no switch; abs1024 is the safe choice everywhere (Stage 4, the follow-up searches).

**Hypothesis:** the signed ranking can only help when the average gradient over inputs is real rather than sampling
noise. Near convergence that average is small, so with a finite calibration set it is mostly noise, and its sign
cannot be trusted. Then abs1024, which does not use the sign, is the ranking to trust.

**Operational check, needing no fault runs:**
- Draw several disjoint calibration sets (1,024 inputs each) and compute the per-weight average gradient on each.
- **D1 (agreement):** the median correlation of those average gradients between pairs of draws.
- **Evidence so far:** found after the fact, in `RUN_R_STAGE4_DIAGNOSTICS_2026-11-13.md`.

  | Model | D1 | Share of gradient energy that is real (D2) |
  |---|---|---|
  | DS-CNN | 0.21 | 0.23 |
  | ResNet-8, nine checkpoints | 0.47–0.92 | 0.37–0.85 |

  So D1 separated the model without a switch from the nine with one.
- **What is new here:** that separation has never been tested on a model it was not found on.

## 2. Definitions (fixed; the same code as the Stage 4 diagnostics)

Code: `grad_stats` and `grad_block` in `workflows/runR_stage4_diagnostics.py` (branch `paper1-runR-v2-validated`, f2ad1e8).
Inputs per checkpoint: R = 6 disjoint calibration draws of 1,024 inputs.
- **D1:** median over the pairs of draws of the Pearson correlation of `gbar`, the per-weight mean gradient of the
  cross-entropy on that draw. Primary.
- **D2:** the median over draws of (Σ gbar² − Σ se²) / Σ gbar², where se is the per-weight standard error of
  `gbar`. Secondary.
- **D3:** the share of the first-order advantage of pos1024 over abs1024 that survives when scored with the other
  draws' gradient (`out_over_in`). Descriptive.
- **Distribution figure:** for every model, the histogram of the per-weight ratio t = gbar / se, with the standard
  normal curve drawn for reference (the curve is what pure noise gives). This is the figure that shows, model by
  model, whether the average gradient carries signal.

## 3. Models

The thresholds come from the DS-CNN and the nine ResNet-8 checkpoints. Those models are used for calibration only and
are not part of the test.

| Id | Model | Weights (approx.) | Source | Role |
|---|---|---|---|---|
| jet-e{1,3,10,30,100} | hls4ml jet-tagging MLP, 16-64-32-32-5 | 4,400 | trained here: data on kona; enabol has only a 2,000-sample training cache, so the full data must be extracted | training-progress series |
| mnist-tcad | MNIST MLP 784-32-10 (the TCAD draft model) | 25,000 | `workspace/benchmarks/mnist_quantized/checkpoint.npz`, float32, needs INT8 conversion | small model |
| mnist-e{1,3,10,30} | same architecture, retrained | 25,000 | trained here | training-progress series |
| vww | MLPerf-tiny Visual Wake Words, MobileNetV1 | 210,000 | locked: `locks/mlperf-tiny/vww.lock.json` | the largest model tested |
| ad | MLPerf-tiny ToyCar anomaly autoencoder | 270,000 | locked: `locks/mlperf-tiny/ad.lock.json` | exploratory only: the metric is reconstruction loss and AUC, not accuracy |

Not used: ECON-T (restricted access), SmartPixel (no read access), and the language models (out of scope for this
paper).

The training-progress series tests the cause rather than only the check. If closeness to convergence is what makes
the gradient noisy, then D1 falls as training proceeds, and the switch disappears or moves.

## 4. Predictions (binding once Phase 1 is committed)

For each test checkpoint, D1 is computed in Phase 1 and the prediction written down before Phase 2:

| D1 | Prediction |
|---|---|
| < 0.30 | **One regime:** pos1024 is never usefully ahead of abs1024; trust abs1024. |
| > 0.40 | **Two regimes:** pos1024 is ahead at low rates and abs1024 at high rates. |
| 0.30–0.40 | No prediction. Reported only. |

The two cut-offs bracket the gap between the DS-CNN (0.21) and the lowest ResNet-8 (0.47) with a margin. They were
chosen after those data were seen, so only the new models test them.

**Secondary predictions:**
- In each training-progress series, the Spearman correlation between epoch and D1 is negative.
- For a checkpoint measured as two regimes, the switch rate is within a factor of 1.5 of the formula's prediction
  (the Stage 1 rule R1c, per checkpoint). This applies only if the formula code can be extended to these models
  before Phase 2; otherwise it is dropped.

## 5. Fault runs and the measured regime (Phase 2)

**Protocol:** the Stage 1 and Stage 4 protocol, applied per checkpoint.
- **Faults:** INT8 stored-bit XOR flips at logical bit-error rate p, run on TFLite with the reference kernels.
- **Rates:** 13, from 1e-4 to 3e-2: 1e-4, 3e-4, 1e-3, 1.5e-3, 2e-3, 2.5e-3, 3e-3, 4e-3, 5e-3, 7e-3, 1e-2, 1.5e-2,
  3e-2.
- **Budgets:** 12.5% and 25% of the bits.
- **Arms:**
  - pos1024 and abs1024, built from each draw's calibration set;
  - msb as a reference.
- **Replications:** 6, one per calibration draw. Each has its own test window of 1,000 inputs and 8 fault fields.
- **Usable cell:** the better of pos1024 and abs1024 is within 5 points of clean accuracy.

**Measured regime:**
- **Two regimes:** at least one usable cell where pos1024 is ahead of abs1024 by at least 0.2 point with Holm
  p < 0.05. The test is the paired sign-flip test of Stage 2, over the checkpoint's usable cells.
- **One regime:** otherwise.

## 6. Decision rule (Phase 3, run once)

Count, among test checkpoints with a prediction, how many match their measured regime.

| Outcome | Condition |
|---|---|
| **Supported** | At least 80% match, **and** each prediction class contains at least one matching checkpoint. |
| **Not tested both ways** | All predictions fall in one class. The result is reported as one-sided. |
| **Failed** | Fewer than 60% match. |
| **Inconclusive** | Anything in between. |

**What may be claimed if supported:** a check that needs only gradients predicts, on these models, whether the
signed ranking can help or abs1024 is the one to trust. We say how many models and checkpoints, and that the
thresholds came from two earlier models.

**Not claimed:**
- that it holds on larger or different models;
- anything about physical fault rates;
- that convergence is the cause, unless the training series also behaves as predicted.

**Bigger models:**
- The check costs one gradient pass over 6 × 1,024 inputs, so it can be reported for any model.
- Validation is claimed only where fault runs confirmed it. VWW is the largest validation point.

## 7. Phases and resources

**Phase 0 – assets** (kona-assets; see Section 8): data partitions, trained checkpoints, INT8 TFLite files, locks.
Training the MLPs takes minutes on CPU. Bounded commands and output paths go in the handoff.

**Phase 1 – diagnostics:**
- **What it computes:**
  - D1–D3 and the t-histograms for every checkpoint;
  - descriptive: D1 at calibration sizes 256, 512, 1,024 and 2,048, to check that agreement rises with sample size as
    the noise explanation implies.
- **Cost:** gradients only, minutes per model on CPU; VWW under an hour.
- **Output:** `workspace/runs/paper1_gradtrust_diag_v1/`.
- **Finish:** the predictions file is written, hashed and committed here before Phase 2.

**Phase 2 – fault runs:**

| Models | Cost estimate |
|---|---|
| MLPs (10 checkpoints) | 13 rates × 2 budgets × 3 arms × 6 replications × 8 fields × 1,000 inputs ≈ 3.7 M inferences each; under an hour of CPU per checkpoint |
| VWW (96×96 MobileNet) | about 6–10 CPU-hours with the reference kernels; to be timed on a one-cell smoke run first; Quest `short` partition if kona is busy |

Output: `workspace/runs/paper1_gradtrust_faults_v1/`.

**Phase 3 – analysis:** one script, tested on synthetic data with planted regimes, committed before Phase 2 ends and
run once. Output: `workspace/runs/paper1_gradtrust_analysis_v1/`.

**Code (NetSuRF, branch `paper1-runR-v2-validated`):**
- a generic INT8 model adapter for dense, convolution and depthwise layers: weights, bit jumps, and a differentiable
  companion for per-example gradients. KwsModel already covers depthwise convolutions.
- deterministic tests: on a toy network, the companion matches TFLite and the bit jumps are exact.

## 8. Handoff to kona-assets (assets only; no NetSuRF code)

Each asset is registered with a lock (files by sha256), as done for the ResNet-8 variants and ECON-T.

1. **Jet tagging data:**
   - the 16 high-level features and 5 labels from `/data/datasets/JetTagging/hls4ml_LHCjet_30p_{train,val}.tar.gz`,
     or the public OpenML `hls4ml_HLF` file if the tarballs do not contain them;
   - standardisation fitted on training data only;
   - disjoint index files, chosen by sorting on sha256 of the sample index (no random generator):
     - training;
     - 6 calibration draws of 1,024 each;
     - a held-out set of 512;
     - 6 evaluation windows of 1,000 each.
2. **Jet MLP:**
   - architecture 16-64-32-32-5, ReLU, softmax;
   - trained with Adam (learning rate 1e-3), batch 1,024, seed fixed, TF 2.14.1;
   - float checkpoints saved after epochs 1, 3, 10, 30 and 100; training and validation loss per epoch logged.
3. **MNIST:**
   - the same kind of index files for the existing MNIST data in `/data/datasets/mnist`;
   - the TCAD checkpoint converted as it is;
   - plus the retrained 784-32-10 MLP saved after epochs 1, 3, 10 and 30, with the same logging.
4. **Conversion:**
   - every checkpoint to full-integer INT8 TFLite (TF 2.14.1 converter);
   - the representative set is 500 training inputs, fixed by index;
   - the clean accuracy of each TFLite file is recorded on every evaluation window with the reference kernels.
5. **VWW and AD:**
   - confirm that the existing locks are usable;
   - add the same index files: 6 calibration draws of 1,024 from training data, 6 evaluation windows of 1,000.

## 9. Decisions (researcher, 15 November 2026: "Perfect … Send the plan. And let's run stuff.")

- **Thresholds:** the two cut-offs 0.30 and 0.40 stand, with the no-prediction band between them.
- **ToyCar AD:** stays in as exploratory only. It is not counted in the decision rule.
- **Section 8:** sent to the kona-assets session as written.
- **Where to run:** Quest, on CPU.
  - The TFLite reference kernels run only on CPU.
  - The masks must also be built on CPU: on the GPU, summation order moved 85–100 near-tied bits of an abs1024 mask
    (14 November).
  - GPUs would only speed up the companion passes, which this plan does not need in its fault runs.
  - Phase 1 and Phase 2 therefore run as Quest `short`-partition CPU arrays, each preceded by a one-cell timing smoke
    run.
