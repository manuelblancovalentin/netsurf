# Run A and Run B as frozen and launched (11 November 2026)

The researcher approved both runs on 11 November 2026 ("Go for it. Run it overnight in tmux/bg."). This file records exactly what was frozen and started. It supersedes the sketches in `TECS_DECISION_MEMO_2026-11-11.md` where they differ; the differences are listed explicitly. Nothing here is a result.

Monitor (read-only):

```bash
/mnt/raid0/ml/conda_envs/manuelbv/envs/netsurf/bin/python /projects/netsurf/manuelbv/workspace/implementation-worktrees/paper1-tecs-revision-v1/workflows/monitor_runs.py
```

## Run B: cap-statistics replay of the language-model grids

- **Estimand.** For every saved (BER, field, policy) observation of the finalized Pythia-160M and Qwen3-0.6B wide-BER grids: finite-token mean, median, 95th percentile and maximum NLL, the fraction of finite token losses at or above the 30-nat cap, and the failed fraction. No new effect is estimated.
- **Inputs, all read-only.** The finalized runs' frozen plans, their saved proposed fault fields, the sealed masks and the 4,096-token test cohort. No new field, mask, seed or cohort is created.
- **Code.** `workflows/w8a8_cap_replay.py` (paper worktree, commit `7e419cb`) imports the `netsurf` package of the worktree that produced each source run (`pythia-wide-confirm`, `w8a8-ber-stress`); the source run's own validator confirms those files still match the hashes in its frozen plan.
- **Acceptance gate.** Each replayed risk record, including the SHA-256 of the per-token statistics, and the event counts must equal the finalized observation exactly. Mismatches are recorded per observation; more than three abort a shard. Clean inference must reproduce the finalized clean record at shard start and end.
- **Smoke.** One Pythia field, three policies: 3 of 3 exact, 12.5 s per evaluation.
- **Frozen replay plans.** Pythia `ee207a09cbb4679a663f591e445324ce5ca3fe2a821d6db113a562f0d530532c` (6 shards); Qwen `39fe58e080848e311729ef434fe845d5de97b9000bb32f0bd97e72cdb52bb2ab` (8 shards). Four threads per shard, as in the source runs.
- **Size.** 4,800 evaluations per model.
- **Outputs.** `workspace/runs/w8a8_cap_replay_v1/<target>/` in the paper worktree; one write-once JSON per observation. tmux sessions `runB_pythia`, `runB_qwen`.
- **Not yet done.** An offline finalization step that tabulates the observations and hashes the result. It needs no inference.

## Run A: TFLite confirmation with rebuilt masks, new inputs and a within-tier control

- **Code.** Worktree `paper1-runA-tflite-confirm`, branched from `c27c346` (the engine revision that ran the ResNet-8 FKeras study), configurations committed at `1d11df2`. The launch scripts refuse a different or dirty revision.
- **Configurations.** `config/experiments/paper1_runA_resnet8_confirm_v1.toml` and `paper1_runA_kws_confirm_v1.toml` in that worktree; their SHA-256 values are printed at the top of each run log.
- **Estimands.**
  1. Paired error-rate (and loss) regret of the absolute-gradient mask against every other measured mask in each cell, with the decision-useful cells determined on the new data by the same criterion as in the manuscript (accuracy within 5 points of clean).
  2. At 6.25% and 18.75% protection, the paired difference between the FKeras mask and the mean of four masks that fill the partial tier at random, and between the storage-order MSB-first mask and those same four.
- **Replications.** The engine's replication index shifts every sample role together, so calibration draw, evaluation window and fault fields vary jointly:
  - ResNet-8: 10 replications. Replication *i* evaluates CIFAR-10 test images [1000*i*, 1000*i*+1000) and calibrates on training images [1000*i*, 1000*i*+64). Four fields per BER per replication.
  - DS-CNN: 4 replications. Replication *i* evaluates test clips [1000*i*, 1000*i*+1000) of the pinned 4,890-clip order and calibrates on training clips [1000*i*, 1000*i*+64). Eight fields per BER per replication.
- **Exposure.** Replication 0 is the window used by the main grid and is reported separately. ResNet-8 windows 1 to 9 were not used by the main grid; the official 200-image performance subset used in early engineering runs was not audited against them. DS-CNN windows 1 to 3 (clips 1000 to 3999) were used earlier by the bounded six-bit conditional-selection studies, never for whole-mask ranking comparisons.
- **Fault regime.** The complete existing grid: eight BERs, four budgets. No subset was chosen from outcomes.
- **Masks per replication.** Random, storage-order MSB-first, numerical jump, global gradient, absolute gradient, positive gradient and FKeras, all rebuilt on that replication's calibration draw; plus four MSB-first masks with a seeded random within-tier tie-break (fixed seeds, same in every replication).
- **Seeds.** Fault seeds 5121201 (ResNet-8) and 5934201 (DS-CNN) with stride 100003 per replication; random-ranking seeds 5121102 and 5934102 with stride 1009; within-tier seeds 5121901 to 5121904 and 5934901 to 5934904. None coincides with a main-grid seed.
- **Size.** 46 evaluations per field (2 controls and 11 masks at 4 budgets). ResNet-8: 10 x 4 x 8 x 46 = 14,720. DS-CNN: 4 x 8 x 8 x 46 = 11,776.
- **Uncertainty plan.** Primary: the replication is the unit (nine fresh ResNet-8 replications, three DS-CNN), with a Student-t interval over per-replication paired means. Secondary: field-level paired intervals pooled within model. Sign-flip tests with at least 10^5 draws and Holm correction over the stated family. With three fresh DS-CNN replications the replication-level interval will be wide; that is a known limit of the available test set.
- **Decision rule.** If the upper 95% bound of absolute-gradient error-rate regret stays below 1 percentage point in every decision-useful cell, the default claim is confirmed for these two checkpoints over calibration draws and input windows. Otherwise the cells where it fails are reported.
- **Outputs.** `workspace/runs/paper1_runA_<model>_confirm_v1/<run id>/` in the Run A worktree, finalized by the engine. tmux sessions `runA_resnet8`, `runA_kws`.

### Differences from the decision memo's sketch

| Memo sketch | As launched | Reason |
|---|---|---|
| One new 1,000-input window, five calibration draws | One window and one calibration draw per replication (10 and 4) | The validated engine shifts all sample roles together; changing that would have meant new, untested engine code overnight |
| 32 fields per BER on the new window | 40 (ResNet-8) and 32 (DS-CNN) fields per BER, nested in replications | Same reason |
| Eight within-tier random seeds | Four | Cost; each seed adds four evaluations per field |
| Tier-first policy with a random score | MSB-first score with a seeded random tie-break | The engine rejects raw bit-significance ordering for non-FKeras policies on TFLite storage; the supported form selects the same family of masks |
| Calibration and cohort variance separable | Confounded within replication | Consequence of the first row; the replication-level interval covers both jointly |

Still fixed in every replication: the trained checkpoint. Run A says nothing about other checkpoints.
