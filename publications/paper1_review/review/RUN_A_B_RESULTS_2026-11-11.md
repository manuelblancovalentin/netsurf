# Run A and Run B: outcomes (11 November 2026)

Designs, seeds and the decision rule are in `RUN_A_B_FROZEN_PLANS_2026-11-11.md`, written before launch. This file records what came out. Tables: `data/tecs_runA/`, `data/tecs_runB/`. Hashes: `data/tecs_revision_provenance.json`.

## Run A: TFLite confirmation

Both engine bundles finalized with all engine checks true (matched schedules, model state restored, disjoint partitions, exact quantized state). Evaluation windows and calibration draws are as planned. Analysis: `scripts/tecs_runA_analysis.py`, output `workspace/paper1_tecs_revision/runA_analysis_v2/` (v1 is kept; v2 adds one secondary reading and leaves the decision rule unchanged).

| | ResNet-8 | DS-CNN |
|---|---|---|
| New replications (fields per BER) | 9 (36) | 3 (24) |
| Decision-useful cells on the new data | 20 of 32 | 22 of 32 |
| Largest regret of the absolute-gradient mask | 0.19 points | 0.13 points |
| Largest upper 95% bound, replication as unit | 0.64 points | 3.78 points |
| Cells with that bound below 1 point | 20 of 20 | 14 of 22 |
| Largest upper 95% bound, fields pooled | 0.49 points | 0.54 points |
| Holm-resolved contrasts for / against absolute gradient | 63 / 0 | 24 / 0 |
| **Decision rule** | **met** | **not met** |

- DS-CNN fails the rule because three replications give a Student-t quantile of 4.3. Its estimates agree with the main grid. Against the best other ranking only (a reading added after seeing the first result) the bound exceeds 1 point in 4 of 22 cells, up to 2.36.
- On DS-CNN, MSB-first and FKeras are not resolved worse than the absolute-gradient mask in any decision-useful cell of the new data (largest regrets 2.5 and 2.1 points).
- Highest useful BER at 12.5% on the new ResNet-8 windows: 3e-4 unprotected, 1e-3 MSB-first, 3e-3 absolute gradient. The main grid's single window gave 1e-4, 3e-4, 3e-3. The manuscript's "ten times MSB-first and thirty times no protection" was lowered to three and ten.
- Within-tier control at 6.25% and 18.75%: a uniform random fill of the partial tier is resolved worse than FKeras in 9 of 16 ResNet-8 cells and worse than the storage-order fill in 12 (up to 16.0 and 18.4 points); on DS-CNN in 4 and 5 cells. FKeras against storage order: nothing resolved in a decision-useful cell; storage order better by 4.0 and 2.5 points on ResNet-8 at p = 1e-2.
- Replication 0 (the main grid's window, four or eight new fields) is tabulated separately in `data/tecs_runA/replication0_cells.csv` and is not used in any statement.

Deviations from the frozen plan: none in the run. In the analysis, the decision rule was applied in its strictest form (largest upper bound over all other rankings); the reading against the best other ranking is reported as secondary.

## Run B: cap-statistics replay

All 9,600 observations (4,800 per model) reproduced the finalized risk records, per-token statistics hashes and event counts exactly. No failed unit. Tabulation: `scripts/tecs_runB_tabulate.py`.

| | Pythia-160M | Qwen3-0.6B |
|---|---|---|
| Useful conditions | 82 | 108 |
| Useful conditions with any capped token | 1 (one token) | 0 |
| Decision-useful cells where the best ranking changes without the cap | 0 of 17 | 0 of 21 |
| Largest change of absolute-gradient regret without the cap | below 1e-5 nats | 0 |
| Unprotected model, capped token fraction at p = 1e-2 / 3e-2 / 1e-1 | 18% / 83% / 99% | 1% / 9% / 55% |
| Whole-grid cells where the best ranking changes without the cap | 2 (p >= 3e-2) | 0 |

- The positive-gradient mask remains worse than no protection on Pythia in all eight cells without the cap. At p = 3e-3 at most 6% of token losses are capped (uncapped difference 1.1 to 5.5 nats); at p = 1e-2 the cap hides most of an uncapped difference of 10 to 13 nats.

## What these runs do not show

- Nothing about other trained checkpoints.
- No confirmation for ToyCar or the transformers.
- Calibration draw and evaluation window vary jointly in Run A and cannot be separated.
