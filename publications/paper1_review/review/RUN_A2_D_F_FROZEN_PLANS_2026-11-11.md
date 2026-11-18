# Runs A2, D and F as frozen and launched (11 November 2026)

The researcher asked on 11 November 2026 to start the larger DS-CNN confirmation and the retraining "right away", and to start any test that could show fixed rankings are not enough. This file records what was frozen before each launch. Nothing here is a result.

## Run A2: larger DS-CNN confirmation

- **Why.** Run A had three new DS-CNN replications and could not meet its decision rule.
- **Code and configuration.** Run A worktree, commit `c21b354`, `config/experiments/paper1_runA2_kws_confirm12_v1.toml`. Same engine revision as Run A plus this one configuration file.
- **Design.** Twelve replications. Replication *i* evaluates test clips [405*i*, 405*i*+400) of the pinned order and builds all seven masks from training clips [405*i*, 405*i*+64). Eight new fault fields per BER per replication (fault seed 5934801, stride 100003). Complete grid: eight BERs, four budgets. The four within-tier controls of Run A are not repeated. 23,040 evaluations.
- **Exposure.** The test clips were all used before (main grid: clips 0 to 999; Run A: clips 0 to 3999). Calibration draws 1 to 11, the masks built from them and all fault fields are new. Replication 0 reuses the main grid's calibration draw and is reported separately. This run adds replications of mask construction and faults; it is not a test on untouched inputs.
- **Decision rule (same as Run A, written into the configuration before launch).** The default is confirmed for DS-CNN if, over replications 1 to 11, the upper 95% bound of the error-rate difference between the absolute-gradient mask and every other ranking is below one percentage point in every decision-useful cell of these data. The reading against the best other ranking only is reported as secondary.
- **Resources.** 12 workers, one thread each; about one hour; under 1 GB of output. tmux session `runA2_kws`.

## Run D: retrained ResNet-8 checkpoints

- **Why.** Every result so far is conditional on one trained checkpoint per model.
- **Code.** `workflows/retrain_resnet8.py`, commit `4cf52c6` of the paper worktree. Model definition imported unmodified from mlcommons/tiny at commit `4addd0fa` (public clone under `workspace/external/`, not committed).
- **Recipe.** The reference recipe: 500 epochs, batch 32, Adam, learning rate 0.001 x 0.99^epoch, rotation 15, shifts 0.1, horizontal flip, all 50,000 training images, then full-INT8 post-training quantization with the reference representative set. Differences: recorded seeds (7101 to 7105); 16 base filters, which is the architecture of the pinned checkpoint (the reference script at that commit defaults to 26), with weight shapes asserted equal to the pinned model; no plots.
- **Outputs.** `workspace/runs/paper1_runD_resnet8_retrain_v1/seed_<seed>/` with the HDF5 model, the INT8 TFLite model, a training log and a manifest with hashes and FP32 and INT8 test accuracy. Write-once.
- **Resources.** Five processes, four threads each; about 6 hours (44 s per epoch measured); about 1.2 MB per seed. tmux session `runD_resnet8`.
- **Not yet done, and needed before these checkpoints can be compared.** (1) An engine adapter variant that accepts a retrained model with its own recorded hash; the current adapter accepts only the pinned model. This is new code on a separate branch and leaves every existing asset lock unchanged. (2) A check that the engine's gradient companion matches each retrained INT8 model. (3) The frozen plan of the repeated ranking comparison itself. No ranking comparison on retrained checkpoints has been launched.
- **DS-CNN retraining is not started.** The reference recipe needs `tensorflow_datasets` and the raw background-noise recordings, neither of which is in the pinned environment.

## Run F: fault-rate-aware masks on DistilBERT (exploratory)

- **Question.** Does a mask rebuilt for a specific fault rate beat the per-example absolute-gradient mask, which is built once on the fault-free model?
- **Code.** `workflows/fault_aware_masks.py`; all definitions are in its header and were fixed before any candidate was evaluated.
- **Candidates.** For a target BER *p* and budget *B*, start from the sealed absolute-gradient mask and repeat twice: draw four new fault fields at *p*, apply the faults the current mask leaves unprotected, recompute the study's decoded-weight gradients on the same 64 calibration sentences on each faulted model, and re-rank all bits. C1 ranks by the mean absolute gradient under faults times the bit's jump. C2 ranks by the two-moment benefit, *p d g* + 0.5 *p*(1-*p*) *d*² *f*, with *g* the mean gradient and *f* the mean squared gradient under faults. Targets: *p* in {0.01, 0.03, 0.1}, *B* in {12.5%, 25%}; twelve masks.
- **Gates.** Fault-free gradients must equal the saved selection moments and must reproduce the sealed absolute-gradient mask (smoke: zero difference in both). Before evaluation, one finalized observation is replayed per shard and must match bit for bit.
- **Evaluation.** Every mask on all 24 saved fields at *p* in {0.003, 0.01, 0.03, 0.1}, exact integer executable, frozen 512-sentence cohort; paired with the finalized outcomes of the fixed rankings on the same fields. 1,152 evaluations.
- **Primary contrast.** Candidate built for (*p*, *B*), evaluated at (*p*, *B*), minus the fixed absolute-gradient mask at *B*. Twelve contrasts, Holm. A gain matters if it is at least one accuracy point with a 95% interval excluding zero. Off-target evaluations are secondary.
- **Known weakness, stated in advance.** In the smoke build C2 changed two thirds of the protected bits and did not settle between its two iterations. It is evaluated anyway, as specified.
- **Scope.** One model, one checkpoint, an evaluation cohort used before. A positive result would need confirmation on fresh inputs and other models before any manuscript claim.
- **Resources.** Build: six processes, four threads, about 8 minutes each. Evaluation: eight shards, four threads, about one hour. Output under `workspace/runs/paper1_runF_fault_aware_masks_v1/`, about 0.6 GB of masks.

## Prototype tail check (exploratory, no plan needed beyond this note)

`workflows/drift_diffusion_prototype.py --variants hybrid exact`: the two-moment surrogate with the two highest bits drawn as real flips, and a fully real-flip variant that checks the prototype's evaluation path against the measured means. Five masks at 12.5%, six BERs, three draws. Output `workspace/prototypes/drift_diffusion_v2_tail/`.
