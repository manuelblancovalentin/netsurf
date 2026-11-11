# Runs J, K and L2 as frozen and launched (11 November 2026)

Approved by the researcher on 11 November 2026 ("Start the under-trained and pruned ResNet-8 ablation too"; "Yes, repeat it"). Nothing here is a result.

## Run J: marginal ResNet-8 checkpoints

- Code: `workflows/train_resnet8_variants.py` (paper worktree, commit `2842c3a`). Architecture, conversion and representative set of the pinned checkpoint.
- Variants: under-trained (reference recipe stopped after 10, 30, 100 epochs; seed 7201); pruned (pinned FP32 model, per-layer magnitude pruning of 50% and 70% of kernel weights, 40 epochs of fine-tuning with the pruned positions held at zero; seed 7301).
- Rule fixed before any ranking comparison: a checkpoint is functional if its INT8 test accuracy is at least 80% (pinned: 87%).
- Outcome of training: pruned 50% 85.7%, pruned 70% 83.9%, under-trained 100 epochs 81.8% (functional); under-trained 10 epochs 73.8% and 30 epochs 77.4% (excluded by the rule).

## Run K: the seven rankings on other checkpoints

- Engine: branch `paper1-runK-retrained-engine` (worktree of the same name), commit `9b57733`. Two changes against the Run A engine, both in commit `dbbc14a`: the adapter loads a retrained model only when `NETSURF_RETRAINED_IC_SHA256` names its hash, after the pinned model has passed the runtime equivalence gate (asset locks unchanged); the gradient companion reads the numbering of kernel and bias tensors from the operators and refuses any other activation wiring (identity for the pinned model, checked). Engine tests for the ResNet-8 path pass (19). An engine smoke run on a briefly trained checkpoint completed and recorded that checkpoint's hash.
- Checkpoints: pruned 50%, pruned 70%, under-trained 100 epochs, and the five Run D retrained seeds when they finish (started automatically by `workspace/launch/orchestrate_runK.sh` in the engine worktree; the launcher refuses checkpoints below 80%).
- Design per checkpoint (`config/experiments/paper1_runK_resnet8_<name>_v2.toml`): random, MSB-first, numerical jump, global gradient, per-example absolute gradient, per-example positive gradient, FKeras; four budgets; eight BERs; eight replications (CIFAR-10 test windows 0-7999, all used before; each with its own 64-image calibration draw and rebuilt masks); two fault fields per BER per replication; new seeds per checkpoint. 3,840 evaluations per checkpoint, eight workers, about 3 hours.
- Pre-set reading (written into each configuration): the absolute-gradient mask is counted as broken on a checkpoint if another ranking is better by at least one accuracy point in a decision-useful cell with a Holm-adjusted field-level p below 0.05. MSB-first and FKeras regrets are compared with the pinned checkpoint (Run A). Replication-level intervals are reported as well.
- A first launch with four replications of four fields was stopped after one minute, before any bundle was written, because the engine runs one worker per replication; the v2 layout has the same evaluation count.

## Run L2: correction mechanisms on ResNet-8

- Code: `workflows/correction_mechanisms_resnet8.py` (paper worktree, commit `ea3cc98`); conditions as in Run L.
- Pinned ResNet-8 through the TFLite BUILTIN_REF executable, INT8 kernels only; CIFAR-10 test images 1000-1999; absolute-gradient masks of Run A replication 1 read from the finalized bundle; clean accuracy asserted equal to Run A replication 1 (87.4%).
- Eight BERs, eight fault draws each; 704 evaluations. Output `workspace/runs/paper1_runL2_correction_resnet8_v1/`.
