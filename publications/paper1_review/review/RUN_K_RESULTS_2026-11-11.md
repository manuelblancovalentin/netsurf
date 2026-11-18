# Run K: the seven rankings on pruned and under-trained ResNet-8 checkpoints (11 November 2026)

Design and pre-set reading: `RUN_J_K_L2_FROZEN_PLANS_2026-11-11.md`. Engine branch `paper1-runK-retrained-engine`, commit `9b57733`; all engine checks true; each bundle records its checkpoint hash. Eight replications (CIFAR-10 test windows 0-7999, all used before, each with its own calibration draw and rebuilt masks), two fault fields per BER per replication; all eight replications analysed (`--min-rep 0`; no window is untouched here). Analysis: `scripts/tecs_runA_analysis.py`, summaries `workflows/summarize_runK.py`; tracked tables in `data/tecs_runK/`. Retrained seeds are still running.

Pre-set reading: the absolute-gradient mask is "broken" on a checkpoint if another ranking is better by at least one accuracy point in a decision-useful cell with a Holm-adjusted field-level p below 0.05.

| | Pinned (Run A) | Pruned 50% | Pruned 70% | Under-trained (100 epochs) |
|---|---|---|---|---|
| Clean accuracy | 87.1% | 85.7% | 83.7% | 81.5% |
| Decision-useful cells | 20 | 20 | 18 | 25 |
| Absolute-gradient largest regret | 0.19 | 0.40 | 0.46 | 1.46 |
| Broken by the pre-set reading | no | no | no | **yes, 3 cells** |
| MSB-first largest regret | 13.1 | 19.4 | 28.9 | 49.1 |
| FKeras largest regret | 13.1 | 19.4 | 8.8 | 24.9 |
| Positive-gradient largest regret | 36.9 | 25.1 | 5.7 | 58.6 |
| Highest useful BER at 12.5%: MSB-first / absolute gradient | 1e-3 / 3e-3 | 1e-3 / 3e-3 | 3e-4 / 1e-3 | 1e-3 / 1e-2 |
| Accuracy lost at 3e-3, 12.5%: MSB-first / absolute gradient | 16.1 / 3.0 | 21.9 / 3.6 | 34.1 / 5.3 | 8.8 / 0.6 |

Regrets in accuracy points against the best of the seven rankings in each decision-useful cell. Pinned values use Run A windows 1-9.

## Under-trained checkpoint: a resolved reversal inside the useful region

Error-rate difference, absolute gradient minus positive gradient (positive = positive gradient better), with field-level and replication-level 95% intervals:

| Budget | p = 1e-3 | p = 1e-2 |
|---|---|---|
| 12.5% | +1.30 [+0.95, +1.65], replications [+0.83, +1.77], Holm p 0.009 | -22.6 [-28.5, -16.8], Holm p 0.006 |
| 25% | +1.09 [+0.84, +1.35], replications [+0.81, +1.38], Holm p 0.010 | -12.3 [-15.7, -8.8], Holm p 0.005 |

Both cells of each row are decision-useful. At 1e-3 and 3e-3 the positive-gradient mask leaves the model slightly more accurate than without faults (accuracy lost -1.2 and -0.5 points at 12.5%): the flips it leaves unprotected are biased towards lowering the loss, which helps a model that is not at a loss minimum, and overshoots when faults are dense. This is the mechanism of the DistilBERT harm with the opposite sign at low fault rates.

## Reading

- The default holds on both pruned checkpoints by the pre-set reading; its regret stays below half a point.
- On the under-trained checkpoint no single static ranking is best within the useful region: the direction-aware ranking wins at 1e-3 and loses by 12 to 23 points at 1e-2.
- Significance-first (MSB-first, FORTUNE-style) protection falls further behind as the checkpoint becomes more marginal: its largest regret grows from 13 to 19, 29 and 49 points.
- Limits: one training seed per variant; windows reused; the replication-level one-point bound of Run A is not met on any variant (largest bounds 1.0 to 2.5 points with eight replications); the under-trained model's clean accuracy is just above the pre-set 80% threshold.
