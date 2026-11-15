# What sets the gradient agreement? (exploratory, 15 November 2026)

**Status:** exploratory. Run after the gradient-trust predictions were frozen (7f4fedb); it changes no prediction or
rule. The researcher asked for it: "investigate what this complex relationship is … generate whatever plots are
necessary".

**Code:** `workflows/runG_gradtrust_explore.py` and `workflows/runG_gradtrust_explore_inputs.py`, branch
`paper1-runR-v2-validated`.

**Output:** `workspace/runs/paper1_gradtrust_explore_v1/`: five figures, `summary.json` and `inputs_checks.json`,
locked, with SHA-256 sums.

**Inputs:**
- the gradient-trust Phase 1 statistics;
- the DS-CNN Stage 4 calibration statistics;
- the ResNet-8 Stage 2 calibration statistics;
- the kona-assets training logs.

ResNet-8's draws are 1,024 images that start every 1,000 images, so consecutive draws share 24 images. This inflates
its agreement slightly.

## The simple model

Each calibration set's average gradient is the true average gradient plus noise. The noise shrinks with the number of
inputs n. So the agreement between two sets should be about

    agreement ≈ n·s / (n·s + 1),   with   s = (true average gradient)² / (spread of single-input gradients)

The agreement is therefore set by three things: the size of the true average gradient (the signal), the spread of
single-input gradients, and n.

## Findings

1. **Calibration size always helps (Figure 1).** Agreement rises with n on every model.
   - The simple model predicts the level well for VWW, the DS-CNN, jet-e1, jet-e3 and the early MNIST models.
   - It underpredicts the later-epoch jet and MNIST models: measured 0.40–0.51 against predicted 0.15–0.26. Their
     single-input gradients are far from bell-shaped (finding 3). The noise is dominated by rare inputs with large
     gradients, so a typical pair of sets agrees better than the average noise level suggests.
2. **Parameter count is not what matters (Figure 2, left).** The models fall along the lines of equal agreement with
   no relation to size:

   | Model | Weights | Agreement |
   |---|---|---|
   | VWW | 208k | 0.94 |
   | ResNet-8 | 77k | 0.47–0.92 |
   | MNIST MLP | 25k | 0.32–0.61 |
   | DS-CNN | 22k | 0.21 |
   | jet MLP | 4k | 0.25–0.51 |

   - **VWW** stands out: a large true average gradient for its spread, with n·s = 8.9 against 0.2–1.0 for the MLPs.
     The weight-by-weight picture (Figure 3) shows a broad spread of real signal, far wider than noise alone gives.
   - **The DS-CNN** has a small signal for its spread.
3. **Single-input gradients are heavy-tailed or sparse everywhere (Figure 4, right).** The tail ratio is 0.01–0.18,
   against 0.64 for a bell curve. It is lowest for MNIST: most pixels are blank, so most inputs give exactly zero
   gradient to most first-layer weights. In MNIST the 5 largest-gradient inputs of a set, all badly misclassified,
   carry 18–29% of the set's squared gradient.
4. **The average gradient is a near-cancellation of much larger per-class gradients (Figure 5).**
   - Each class's average gradient is 4–12 times larger than the overall average.
   - The cancellation weakens steadily with training: jet from 12× at epoch 1 to 6× at epoch 100; MNIST from 9× to 4×.
   - **jet-e1 is the extreme case.** There, small differences in class counts between sets swing the average
     direction:
     - Draw 0 has 222 jets of one class against 191 of another.
     - Draws 0 and 5 disagree with every other draw (Figure 4, left).
     - Re-weighting each set to an equal class mix raises its agreement from 0.25 to 0.43.
     - Removing extreme inputs does not change it (0.25 → 0.23).

     This explains jet-e1's dip at 1,024: the agreement depends on which class counts the sets happen to have.
   - **For the later jet epochs, equalising the class mix lowers agreement** (0.51 → 0.31 at epoch 30). There, part of
     the real signal is the data's class mix itself.
5. **Training state matters, but not as "converged means noisy" (Figure 2, middle and right).** In both series n·s
   peaks at epoch 3 and then falls to 0.2–0.35. Why it does not fall monotonically:
   - The signal does not simply shrink. The calibration sets are held out from training, so the gap between held-out
     and training loss keeps a real average gradient alive.
   - The spread rises as the model grows confident: misclassified inputs get large gradients.
   - At epoch 1 the cancellation between classes is strongest.

## What this suggests (hypotheses, not results)

The agreement is not one property of the model. It reflects:
- **where the calibration data sits relative to the loss minimum.** This is about held-out against training data and
  the gap between them, more than training epochs;
- **how heavy-tailed the single-input gradients are;**
- **how much the per-class gradients cancel, and the class mix of the calibration sets;**
- **the number of calibration inputs.**

Parameter count did not matter here. Two practical points follow if the fault runs bear them out:
- **Stratify calibration sets by class**, or report agreement with balanced class mixes, so that class-count luck
  does not decide the check.
- **Report agreement at more than one calibration size**, as in Figure 1. A model whose curve is still rising steeply
  at 1,024 needs more calibration data before the signed ranking can be trusted.

Whether the agreement actually predicts which ranking wins is the job of the pre-registered fault runs, which are
running now. The DS-CNN calibration-size test, also pre-registered, follows them.
