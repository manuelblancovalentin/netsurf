# What Paper 1 sells: story proposal (15 November 2026)

Written after re-reading the whole manuscript (commit acc58e5) and drawing six candidate figures from locked data
(`workspace/runs/paper1_story_candidates_v1/`, script `scripts/build_story_candidates.py`). Nothing here is in the
manuscript yet.

## 1. The three options the researcher raised

| Option | What it would sell | Does the evidence carry it? |
|---|---|---|
| NetSuRF as a framework for finding the vulnerable subset, neutral on rankers | The paired protocol: rankings, masks and fault fields kept separate; masks compared on the same fault fields; a usefulness criterion | **Not as the headline.** It is careful but not new enough to sell alone: FKeras, PrioriFI and Thales also evaluate rankings. Keep it as the method that makes the findings trustworthy. |
| The analysis: static rankings are not always best, though one of them is best for us | When a fixed ranking fails, and a default that works | **Yes. This is where the evidence is broad:** 24 checkpoints, nine architectures, frozen rules. But "not always best, but one is best" reads as two half-claims. It needs one idea that explains both. |
| The Gaussian (two-moment) math plus tuning from a static ranking, to avoid a full search | A cheap optimizer | **No, not in this paper.** The math works as an explanation and a predictor on ResNet-8 (switch located within a factor of 1.5). As an optimizer it is weak: the analytic mask overfits (it fails its rule on ResNet-8 and loses up to 10 points on DS-CNN); the signed two-moment mask on DistilBERT (Run F) lost 27–43 points; the switch prediction failed on DS-CNN; and searches did not beat the absolute gradient. Only the selected mask (v2) held, on one model family, and it was defined after the first version's overfitting was seen. This is the follow-up paper, with real protection mechanisms. |

## 2. Proposed story: size, not sign

**One claim.** When the fault rate is not known in advance, protect stored bits by the *size* of their effect on
the loss, not by their bit position and not by the *direction* of their effect. Bit position wastes the budget.
Direction can pay a little when faults are sparse, but costs a lot when they are denser than assumed.

**One idea explains the whole paper.** What a mask leaves unprotected has two parts (Eq. 3 of the draft):
- a **spread**: random scatter of the weights, which hurts at any fault rate. Ranking by size reduces it.
- a **drift**: a coherent push in one direction. It is zero on average for size-based masks and large for
  direction-based ones.

At sparse faults a drift that points downhill can help, almost like a training step. At dense faults it pushes the
weights far off the loss surface. Every result in the paper is a case of this:

| Result | How the idea covers it | Figure |
|---|---|---|
| Significance-first falls far behind | Bit position ignores the spread that actually hurts | **A** |
| The absolute gradient is a strong default on every model | It targets the spread and leaves almost no drift (DistilBERT drift table) | Table 3, A |
| The positive gradient wins at sparse faults on ResNet-8 and collapses at dense faults | Drift helps, then hurts; the formula predicts the switch | **B, C** |
| Choosing the mask by the expected rate is never far behind on ResNet-8 | It picks the right side of the switch | **D** |
| Building for too low a rate costs up to 70 points; too high, at most 0.5–2.6 | Our reading: low-rate masks lean on the signed term, high-rate masks do not | **E** (new) |
| Protection worse than none (DistilBERT, Pythia) | Extreme drift | existing Fig. 4 |
| Ranked triplication beats SEC-DED at dense faults | Hardware form of "spend the budget where the spread is" | existing Fig. 6 |

**The asymmetry in Figure E is the new strong point** (descriptive analysis, fixed in the Stage 2 plan). A mask tuned
for a higher rate than the one it meets loses at most 0.5 point on the pinned ResNet-8 (2.6 on the under-trained
one). A mask tuned for a lower rate loses more and more once the true rate passes about 2e-3: at 1e-2, 8 to 70 points
depending on how low it was tuned. That is a concrete design rule: when in doubt, design for dense faults. The
absolute gradient behaves like the "dense-fault" choice. That last sentence is our reading, not a tested result.

**What stays honestly open, and leads to the follow-up.** Why ResNet-8 is the only model where direction pays.
Figure F shows it is not just that ResNet-8 has more damage left to recover: the jet models lose as much at 3e-3 and
gain nothing. Gradient agreement did not predict it either (pre-registered, failed). The follow-up paper then becomes
rate-aware protection with real mechanisms: per-bit TMR, 5MR and SEC-DED residual-error curves, the two-moment
objective as the allocator, hardware cost and physical rates.

## 3. Draft abstract (about 200 words; numbers to be re-checked when writing)

> Selective protection of neural-network weights must decide which stored bits to protect, usually without
> knowing the fault rate the device will meet. We compare bit rankings as equal-budget protection masks under exact
> INT8 stored-bit faults on 24 checkpoints of nine architectures, from 4,000-weight MLPs to a 0.6-billion-parameter
> language model, evaluating all masks on common fault fields in the task metric. Ranking bits by the per-example size
> of their effect on the loss is a strong default: at 12.5% protection it tolerates about three times the
> bit-error probability of significance-first protection, which trails it by up to 24 accuracy points on Visual Wake
> Words. Using the predicted direction of a flip as well gains at most 2.7 points, only on ResNet-8 and only at sparse
> faults, and loses up to 57 points when faults are denser; a mask tuned for too low a fault rate loses up to 70. A
> second-order model of the expected loss explains the asymmetry: a direction-aware mask leaves a one-sided set of
> faults that helps while faults are sparse and drives the weights off the loss surface when they are not. On
> ResNet-8 the model predicts the fault rate at which the preferred ranking switches, within a factor of 1.5. At dense
> fault rates, triplicating ranked bits outperforms SEC-DED and checksum correction of equal storage.

Title options:
1. *Size, Not Sign: Which Stored Bits to Protect When the Fault Rate Is Unknown*
2. *NetSuRF: Protecting the Right Stored Bits in INT8 Networks Under Unknown Fault Rates*

## 4. Main text (about 14 pages) and appendix

**Main text:**
1. Introduction, with the single claim.
2. Related work.
3. Fault model and the paired protocol (NetSuRF).
4. Rankings, plus spread and drift (Section 5's two-term paragraph becomes the centre).
5. Setup, in one table for all 24 checkpoints.
6. Results:
   - **6.1** Bit position wastes the budget (Fig. A replaces the current Fig. 2 or joins it).
   - **6.2** Size is a strong default (Table 3, plus a compact form of Table 5).
   - **6.3** Direction pays only at sparse faults, and only on ResNet-8 (Figs. B and C).
   - **6.4** Betting on a rate: choosing by rate on ResNet-8, and the asymmetry (Figs. D and E).
   - **6.5** When drift dominates, protection is worse than none (current Fig. 4).
   - **6.6** Ranked triplication against blanket codes (current Fig. 6).
7. Discussion and limits.

**Appendix:**
- repeated evaluations and partial tiers;
- calibration size (Run Q, and DS-CNN at 8,192 clips);
- search, relaxed and literature allocators (Run N) and the DS-CNN search follow-ups (winner's curse);
- gradient agreement: the failed prediction and the five exploration figures;
- the open question (Fig. F);
- rate-aware mask definitions and conduct;
- conditional pools; fault motivation; implementation.

**Out of the paper, or kept for the follow-up:** the DistilBERT rate-aware masks (Run F), the mean-variance masks
(Run P), and the drift-diffusion prototype.

## 5. Decisions for the researcher

1. Is the "size, not sign" story the one to tell, or is another framing preferred?
2. Title: one of the two above, or another?
3. Is the v2 amendment (selected rate-aware mask) accepted? Figure D relies on it.
4. Is about 14 pages of main text still the target?
