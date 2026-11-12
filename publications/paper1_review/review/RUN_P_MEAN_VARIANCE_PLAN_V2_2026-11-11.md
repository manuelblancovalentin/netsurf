# Run P, version 2: mean-variance protection masks with held-out evaluation

Status: written 11 November 2026 at the researcher's request ("Yes, write version 2"), after the Phase 0 result
of version 1 (`RUN_P_PHASE0_RESULTS_2026-11-11.md`). **This version was designed after seeing that result.** It
changes only what the Phase 0 diagnosis identified, and its gate is tested on data that played no part in the
diagnosis. The gate stage (Section 3) is frozen as of this commit and runs offline (no fault injection). Phases 1
and 2 need the researcher's separate go-ahead.

Everything in version 1 (`RUN_P_MEAN_VARIANCE_DRAFT_PLAN_2026-11-11.md`) stays in force unless changed here:
the objective (Section 2 there), the greedy construction, the two rate versions (single-rate, range W =
[3e-4, 3e-2] log-uniform with 8 rates; range N = [3e-3, 3e-2] for the comparison with Run N), the replay design,
the decision rules of Phase 2, the tests and the naming rule (not qPolarGrad; code name `mean_variance_mask`).

## 1. What changes and why

| Version 1 | Version 2 | Reason (Phase 0 of version 1) |
|---|---|---|
| Objective of a saved mask evaluated on the 64 images the mask was built from | Evaluated on 512 held-out images that no mask was built from | In-sample bias: on the pinned checkpoint the positive-gradient mask's mean term was -0.10 to -0.15 nats on its own calibration images, while the same mean weights cost 0.5-6 accuracy points on test images |
| New masks built from 64 calibration images | Built from 512 calibration images (P0d rule) | With 64 images the method made the held-out objective worse on both tested checkpoints |
| Gate on the pinned and under-trained checkpoints | Gate on the five retrained seeds | Both earlier checkpoints were used in the diagnosis; the retrained seeds were not |
| Absolute-gradient reference from 64 images only | Added: absolute gradient from the same 512 images | Separates "more data" from "new objective" (P0d rule for n > 64) |

Not changed, although Phase 0 showed a second, smaller discrepancy (the variance term ranks MSB-first far below the
absolute gradient on the pinned checkpoint while their measured losses are close): changing the objective after
seeing that would be fitting it to one checkpoint.

## 2. Held-out images

512 CIFAR-10 training images 28000-28511 (MLPerf Tiny role `empirical_validation`, offset 3000). They are
disjoint from every calibration draw (training images 0-24999), from the 256 images used in P0d (27000-27255) and
from all test windows. The same 512 images serve every checkpoint and replication.

## 3. Gate stage (offline; frozen now)

For each of the five retrained seeds (Run K bundles `paper1_runK_resnet8_retrained_seed710{1..5}_v2`), each of its
8 replications, each of the seven saved rankings at the four budgets, and BER 1e-3, 3e-3, 1e-2, 3e-2, 1e-1:
compute the objective on the held-out images,

    J_ho = [L_ho(theta + mu) - L_ho(theta)] + 1/2 p (1-p) sum_{unprotected b} G2_ho,i Delta_b^2 ,

with `G2_ho` the Fisher diagonal from per-example gradients on the held-out images, and average over replications.

**Gate, fixed now.** For each seed, the median over decision-useful cells with BER >= 1e-3 of the within-cell
Spearman correlation between `J_ho` and the measured mean accuracy loss across the seven rankings. **Pass if the
median is at least 0.7 on at least four of the five seeds.** Otherwise the method is not pursued further, and the
report says the objective does not rank masks on independently trained copies of ResNet-8.

Reported, not part of the gate: the same quantity on the pinned, under-trained and pruned checkpoints (the pinned
and under-trained ones were used in the diagnosis); and the mean-term-only and variance-term-only correlations.

Cost: one process per checkpoint, about 10-15 minutes each, CPU only. Outputs:
`workspace/runs/paper1_runP_v2_gate/` in the paper worktree.

## 4. Phases 1 and 2 (unchanged from version 1 except as listed; separate go-ahead)

- Phase 1: masks from 512 calibration images per replication (training images [1000 r, 1000 r + 512)), and an
  absolute-gradient mask from the same 512 images. Every built mask's objective is also reported on the held-out
  images. Added check, fixed now: Phase 2 runs on a checkpoint only if the median held-out reduction of the range-W
  objective relative to the 512-image absolute-gradient mask, over replications, is positive.
- Phase 2: as in version 1, with the 512-image absolute-gradient mask as an extra arm. The primary contrast stays
  range W against the paper's absolute-gradient mask; the contrast against the 512-image absolute gradient is
  reported alongside, and a claim that the new objective helps requires the primary criterion to hold against
  both.
- Checkpoints: primary under-trained, pruned 70%, pinned ResNet-8; secondary pruned 50%, DS-CNN, the five retrained
  seeds.

## 5. What would not be claimed

As in version 1. In addition: a passed gate shows that the objective ranks existing masks on held-out images; it
does not show that masks built from it are better. Only Phase 2 can show that.
