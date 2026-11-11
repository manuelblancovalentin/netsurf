# TECS revision notes (fifth delivery, 11 November 2026)

Source commit of the previous draft: `fdba0d0f46593d1f3d129d4ac14b5d4266f69893`. This revision lives on branch `paper1-tecs-revision-v1` in its own worktree. Nothing was merged or pushed, no finalized bundle or asset lock was touched, and `delivery_v4` is unchanged.

## What changed

- **Title and framing.** From *Is There a Universal Bit Ranking?* to *Which Stored Bits Are Worth Protecting? A Paired Evaluation of Static Bit Rankings for INT8 Neural Networks*. The paper no longer leads with preference reversals.
- **Task metrics.** Classifier conclusions use accuracy; ToyCar uses file AUC; language models use NLL with the failed fraction stated. The TFLite cross-entropy definition and its floor are spelled out.
- **Usefulness region.** A fixed, task-specific criterion defines where comparisons matter. Figures shade it.
- **Whole-grid analysis.** Six hand-picked contrasts are replaced by per-cell paired regret, resolved win/loss counts and a full-grid map.
- **New findings from the saved data.** Absolute gradient as a strong default among measured rankings; positive-gradient protection worse than none, with a one-sided-residual mechanism on DistilBERT; partial-tier MSB-first as a storage-order artefact.
- **Conditional study.** Demoted to a short negative note with its effect ceiling and selection noise.
- **Theory.** Presented as background and attributed to reliability-importance theory; the appendix bound and the in-sample attribution table are removed.
- **Identity appendix.** Checkpoints, backends, quantization rule, eligible and excluded tensors, full-set references.
- **Figures.** Three new figures (`tecs_task_quality_ber_125`, `tecs_grid_regret_map`, `tecs_protection_vs_unprotected_125`); the workflow figure keeps its structure with plainer labels; the KWS pool figure moves to the appendix.
- **Bibliography.** Added Birnbaum (1969), Rausand and Hoyland (2004) and Stutz et al. (2021).

## What was not done

- No new inference. No new production run was launched; two frozen plans are in `review/TECS_DECISION_MEMO_2026-11-11.md`.
- Cap frequency for the main language-model grids remains unavailable.
- FKeras candidate correlations for DS-CNN and ToyCar remain uncomputed.
- No comparison with search-based or learned allocators.

## Remaining submission risks

1. The default claim rests on one mask per ranking, one checkpoint and previously exposed cohorts. A reviewer can fairly call it exploratory. Run A in the decision memo addresses masks and cohorts; checkpoints need retraining and separate approval.
2. Software-only scope with bit-count cost is thin for an embedded-systems venue. The irregularity of gradient masks is disclosed but unpriced.
3. Novelty is in the measurement, not the theory. FKeras and PrioriFI already show that significance is an imperfect proxy; the companion *Learned or Ranked?* release compares allocators at matched cost on other models.
4. The positive-gradient mechanism is verified on one model only.
5. The BER regime where masks matter is dense memory error, not rare upsets; the paper says so, and a reviewer may question the use case.
6. The TECS page limit was not verified.
