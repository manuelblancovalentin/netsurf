# Run N0 and Run N: method campaigns on ResNet-8, as frozen (11 November 2026)

Approved by the researcher on 11 November 2026 ("Agreed with all", for campaigns C1 to C5; "Implement it. And call it by its brand name" for the single-fault ranking). Code: `workflows/single_flip_resnet8.py` (Run N0) and `workflows/method_campaigns_resnet8.py` (Run N); the design of each campaign is in the header of the latter and is not repeated here. Nothing here is a result.

## Shared protocol

- Pinned ResNet-8, INT8 kernel bits only (618,880), TFLite BUILTIN_REF executable for every accuracy and loss used to accept, select or evaluate a mask; the differentiable companion only for gradients.
- Images: calibration (gradients, single-flip table) training 2000-2063; validation (search objectives) training 27000-27255; final evaluation test 2000-2999. The three sets are disjoint.
- Final evaluation: the 16 fault fields per rate of Run M (same seeds), rates 3e-3, 1e-2, 3e-2, budgets 12.5% and 25%, every mask evaluated on the same fields, so all comparisons are paired. References evaluated in the same run: the engine's absolute-gradient mask (Run A replication 2), the engine's MSB-first mask, the Run M rebuilt masks.
- Primary contrast per method and cell: accuracy minus the absolute-gradient mask at the target rate (masks that do not depend on p: at every rate), paired over 16 fields. A method "beats" the default in a cell if the gain is at least one point and its 95% interval excludes zero. Holm over all primary contrasts is reported as well.

## Campaigns

| | Name in plots | What it is | Depends on p |
|---|---|---|---|
| C1 | MC search | Monte Carlo-guided search: proposals from gradients of faulted models, kept only if they lower the validation loss over common fault fields | yes |
| C2 | Relaxed | stochastic-gradient optimization of per-bit protection probabilities over sampled fault fields | yes |
| C3 | Group CEM | cross-entropy-method search over 80 (tensor, bit position) groups | yes |
| C4 | Thales | bits ranked by exhaustive single-flip accuracy loss (Thales' per-fault accuracy with uniform fault probability; Tyagi et al., HPCA 2023); also Thales-loss (single-flip cross-entropy increase) | no |
| C5 | HarDNN | whole output filters ranked by mean single-flip loss change (HarDNN's delta-loss) or by mean gradient magnitude (HarDNN's best cheap heuristic); Mahmoud et al. 2020 | no |

Thales and HarDNN define vulnerability measures for transient faults; the protection rankings built from them here are our adaptations to stored-weight faults and are named after the measure they use.

## Run N0: single-flip table

618,880 flips, each evaluated on the 64 calibration images (accuracy and mean cross-entropy); 32 shards; about 70 CPU-hours. Output `workspace/runs/paper1_runN0_single_flip_resnet8_v1/`.

## Resources

C1, C2 and C3: 18 processes (two TensorFlow threads each), 30 to 80 minutes. Evaluation: about 30 masks x 48 evaluations, 12 processes. Output `workspace/runs/paper1_runN_method_campaigns_resnet8_v1/`.

## Amendment, 11 November 2026, before any Run N outcome existed

At the researcher's request, C2 gains a range-optimized variant ("relaxed, range"): the fault rate of every optimization step is drawn log-uniformly from [3e-3, 3e-2], the range of the final evaluation; 4000 steps; one mask per budget; evaluated at every rate in the same paired evaluation. It answers how much is lost by committing to a mask optimized for one fault rate, against one optimized for the range.
