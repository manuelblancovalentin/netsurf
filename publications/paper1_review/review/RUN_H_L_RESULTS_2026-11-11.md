# Runs H and L: activation faults, checksums and correcting codes (exploratory, 11 November 2026)

Not used in the manuscript. DistilBERT W8A8, the study's frozen 512 sentences, clean accuracy 91.4%, two fault draws per point, no intervals. Designs are in the headers of `workflows/activation_abft_check.py` (Run H) and `workflows/correction_mechanisms.py` (Run L); both were committed before launch. Outputs: `workspace/runs/paper1_runH_activations_abft_v1/`, `workspace/runs/paper1_runL_correction_mechanisms_v1/`. Plots: `workspace/figures_for_discussion/`.

## Run H: weight faults against activation faults at the same per-bit probability

| Fault rate | 1e-4 | 3e-4 | 1e-3 | 3e-3 | 1e-2 |
|---|---|---|---|---|---|
| Activation faults only | 89.8 | 87.4 | 70.5 | 57.5 | 52.1 |
| Weight faults only | 90.7 | 90.9 | 90.6 | 86.7 | 61.5 |
| Weight faults, 12.5% absolute-gradient mask | 90.6 | 90.6 | 91.2 | 91.1 | 90.7 |
| Both, weights masked | 90.8 | 86.2 | 68.6 | 55.6 | 51.2 |

- Activation faults are redrawn at every forward call; about 4.2e7 activation bits are exposed per sentence against 3.4e8 stored weight bits.
- At equal per-bit probability, activation faults degrade the model about ten times sooner.
- A per-layer output checksum (stored column sums, fault-free) fails on 100% of sentences under weight faults at every rate from 1e-5, and on 0% under activation faults, because it reads the same corrupted activation as the multiplier.

## Run L: selective triplication against blanket correction (every stored bit, copies and check bits included, flips with p)

| Scheme | Extra storage | Highest p within 5 points | Accuracy at p = 0.01 | at 0.03 |
|---|---|---|---|---|
| No protection | 0% | 1e-3 | 59.1 | 55.7 |
| ABFT checksum correction (two checksum words per row of stored weights; repairs one faulty weight per row) | ~1% | 1e-3 | 59.1 | 55.7 |
| SEC-DED over 64 / 32 / 16 / 8 data bits | 12.5 / 21.9 / 37.5 / 62.5% | 3e-3 / 3e-3 / 1e-2 / 1e-2 | 76.2 / 83.2 / 87.4 / 91.0 | 52.9 / 54.1 / 60.6 / 54.2 |
| TMR on all sign bits (FORTUNE-style) | 25% | 3e-3 | 83.3 | 54.8 |
| TMR on the absolute-gradient mask, 6.25 / 12.5 / 18.75 / 25% of bits | 12.5 / 25 / 37.5 / 50% | 1e-2 / 1e-2 / 1e-2 / 3e-2 | 90.6 / 91.1 / 90.1 / 91.2 | 70.3 / 81.8 / 85.3 / 86.5 |

- At equal extra storage, triplicating the ranked mask tolerates about three times the fault rate of a SEC-DED code at 12.5% and 25% storage, and keeps 90-91% at p = 0.01 where the codes give 76-83%.
- At 25% storage, ranked TMR keeps 91.1% at p = 0.01 against 83.3% for TMR on all sign bits.
- ABFT checksum correction repairs 88% of faults at 1e-5 and almost none from 1e-3 on, because almost every row then holds two or more faulty weights.
- Below 1e-3 every scheme is within a point of clean; the codes leave far fewer faulty bits there (SEC-DED-8 leaves 1.2% of raw faults at 1e-3, ranked TMR 87.5%), so for very rare faults the codes are the efficient choice.

## Not modelled, all of which matter for a hardware claim

Logic area and power of voters, encoders and decoders; miscorrection of three or more errors (favours the codes); interleaving; scrubbing, which turns accumulated faults into faults per scrub interval and changes which p applies; a code combined with a mask. One model, one checkpoint, two draws.

## Run I: ResNet-8 with INT32 biases in the fault population (engine run, finalized)

Bundle `paper1_runI_resnet8_kernels_biases_v1` in the Run A worktree (commit `d368969`), all engine checks true. 629,952 eligible bits (77,360 INT8 kernel words, 346 INT32 bias words = 11,072 bits, 1.8%). Six 1000-image windows (0-5999, all used before), four fields per BER per window. Only data-free rankings. Accuracy, mean over 24 fields; clean 87.0%.

| Ranking | 1e-4 | 3e-4 | 1e-3 | 3e-3 |
|---|---|---|---|---|
| No protection | 62.3 | 45.0 | 14.3 | 9.9 |
| MSB-first (by position within the word), 12.5% | 62.9 | 47.5 | 15.2 | 9.9 |
| Numerical jump, 12.5% | 85.1 | 81.9 | 71.3 | 48.3 |
| MSB-first, 25% | 67.8 | 48.4 | 15.6 | 10.6 |
| Numerical jump, 25% | 85.5 | 83.4 | 76.8 | 64.8 |
| Kernels only, no protection (Run A, windows 1-9) | 84.8 | 82.7 | 70.9 | 42.5 |
| Kernels only, MSB-first 12.5% (Run A) | 86.5 | 84.7 | 82.3 | 70.9 |

- With biases exposed, the unprotected model loses 25 points at 1e-4, where the kernel-only model loses 2.
- MSB-first as implemented ranks bits by position relative to the word, so it protects the sign bit of each bias but not bits 30 down to 8 of the INT32 word, whose flips move the bias by up to 2^30 quantization steps. It recovers almost nothing.
- Numerical jump ranks by the actual change of the stored value, protects the upper bias bits first, and comes close to the kernel-only results.
- The result depends on how "MSB-first" is defined for words of different widths; a rule ranking absolute bit index (bit 31 of INT32 before bit 7 of INT8) would protect the bias bits. FORTUNE is defined for weights only.
- Windows and fields differ from the kernel-only comparison rows; the differences are far larger than that variation, but no paired contrast was computed.

## Run L2: Run L repeated on ResNet-8 (8 fault draws, TFLite reference executable)

Design: `RUN_J_K_L2_FROZEN_PLANS_2026-11-11.md`. Analysis: `workflows/analyze_correction_resnet8.py`, output `workspace/runs/paper1_runL2_correction_resnet8_v1_analysis.json`. Clean accuracy 87.4%.

| Scheme | Extra storage | Highest p within 5 points | Accuracy at 3e-3 | at 1e-2 |
|---|---|---|---|---|
| No protection | 0% | 1e-4 | 39.7 | 14.5 |
| ABFT checksum correction | ~1% | 3e-4 | 45.7 | 16.4 |
| SEC-DED over 64 bits | 12.5% | 1e-3 | 81.0 | 31.3 |
| TMR, ranked mask of 6.25% of bits | 12.5% | 3e-3 | 83.0 | 66.0 |
| SEC-DED over 32 bits | 21.9% | 3e-3 | 83.2 | 45.6 |
| TMR, all sign bits (FORTUNE-style) | 25% | 1e-3 | 69.1 | 42.3 |
| TMR, ranked mask of 12.5% of bits | 25% | 3e-3 | 84.5 | 75.7 |
| SEC-DED over 16 bits | 37.5% | 3e-3 | 86.1 | 53.2 |
| TMR, ranked mask of 18.75% of bits | 37.5% | 3e-3 | 86.1 | 80.1 |
| TMR, ranked mask of 25% of bits | 50% | 1e-2 | 86.5 | 83.3 |
| SEC-DED over 8 bits | 62.5% | 3e-3 | 85.9 | 73.8 |

Paired differences, ranked TMR minus the alternative of equal (or larger) storage, accuracy points with 95% intervals over 8 draws:

| Pair | p = 1e-3 | 3e-3 | 1e-2 | 3e-2 |
|---|---|---|---|---|
| 6.25% ranked TMR vs SEC-DED-64 (12.5% each) | -1.4 [-2.1, -0.8] | +2.0 [-1.9, +6.0] | +34.7 [+25.7, +43.8] | +10.7 [+5.6, +15.8] |
| 12.5% ranked TMR vs SEC-DED-32 (25% vs 21.9%) | -0.8 [-1.5, 0.0] | +1.4 [-1.1, +3.8] | +30.1 [+21.0, +39.2] | +18.6 [+10.7, +26.5] |
| 12.5% ranked TMR vs sign-bit TMR (25% each) | +2.8 [+0.2, +5.3] | +15.4 [+7.1, +23.6] | +33.4 [+24.1, +42.7] | +12.0 [+3.7, +20.3] |
| 18.75% ranked TMR vs SEC-DED-16 (37.5% each) | -0.7 [-1.4, 0.0] | -0.1 [-1.7, +1.6] | +26.9 [+15.9, +38.0] | +26.9 [+17.9, +36.0] |
| 25% ranked TMR vs SEC-DED-8 (50% vs 62.5%) | -0.7 [-1.2, -0.2] | +0.6 [-2.0, +3.2] | +9.5 [+3.8, +15.2] | +26.4 [+15.3, +37.4] |

- Same pattern as DistilBERT, now with intervals: the codes are better by about one point up to 1e-3, the two are equivalent near 3e-3, and ranked triplication is better by 10 to 35 points from 1e-2 on.
- Against triplicating all sign bits at the same storage, ranked triplication is better at every fault rate from 1e-3 on.
- By the five-point criterion the ranked mask extends the tolerated rate by a factor of three over the code only at 12.5% and 50% storage; at 25% and 37.5% both stop at 3e-3.
