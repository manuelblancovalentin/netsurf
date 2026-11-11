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

## Update after the overnight runs (11 November 2026)

- Run A (TFLite confirmation) and Run B (language-model cap replay) were approved, executed and analyzed. Frozen designs: `review/RUN_A_B_FROZEN_PLANS_2026-11-11.md`. Outcomes: `review/RUN_A_B_RESULTS_2026-11-11.md`. Tables: `data/tecs_runA/`, `data/tecs_runB/`.
- Manuscript changes: new Section 7.3 and Table 5 (confirmation run); the tolerated-BER factors in the abstract, introduction, Section 7.1 and the discussion were lowered to what the new windows support; the FKeras partial-tier paragraph now reports the random within-tier control; a cap-replay paragraph was added to Section 7.4 and the positive-gradient harm on Pythia is now qualified by cap frequency; limits and conclusion were updated.
- The exploratory mean-field drift prototype (`workflows/meanfield_drift_prototype.py`) is not used in the manuscript.

## What was not done

- No independently trained checkpoint. No confirmation run for ToyCar or the transformers.
- DS-CNN confirmation has three new replications and does not meet the decision rule.
- FKeras candidate correlations for DS-CNN and ToyCar remain uncomputed.
- No comparison with search-based or learned allocators.
- The positive-gradient mechanism check (Table 6) still covers DistilBERT only.

## Remaining submission risks

1. One trained checkpoint per model. The default is confirmed over calibration draws and input windows on ResNet-8 only; a reviewer can fairly ask for DS-CNN with more replications and for the transformers.
2. Software-only scope with bit-count cost is thin for an embedded-systems venue. The irregularity of gradient masks is disclosed but unpriced.
3. Novelty is in the measurement, not the theory. FKeras and PrioriFI already show that significance is an imperfect proxy; the companion *Learned or Ranked?* release compares allocators at matched cost on other models.
4. The margin of the default over MSB-first is small on DS-CNN (not resolved on the new data), so the practical case rests mainly on ResNet-8 and the transformers.
5. The BER regime where masks matter is dense memory error, not rare upsets; the paper says so, and a reviewer may question the use case.
6. The TECS page limit was not verified (the draft is 20 pages in the review format).
