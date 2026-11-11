# Third delivery: compact manuscript revision

This is an editorial and presentation revision of the authoritative second delivery, not a new scientific study. Source baseline: `fb360377b80afa1a095f1f4dee1056d0fe02a7f2`. The preceding 24-page PDF and archives remain unchanged.

| Requested correction | Implemented change |
|---|---|
| Excessive length | 13 pages total in the same ACM manuscript/review class. Main conclusion on page 10; references and appendices occupy the remainder. No font, margin or line-spacing reduction. |
| Suggestive title without a new method claim | **Is There a Universal Bit Ranking? Evaluating Fault Protection in Quantized Neural Networks**. NetSuRF remains the framework name in the text. |
| Do not single out FKeras | Removed its standalone methodology/results subsection. It is one row of the common method table and appears in matched contrasts alongside other methods. Published projection and complete-tier degeneracy are retained. |
| Remove irrelevant history | No qPolarGrad or legacy-estimator discussion in the manuscript. Earlier audit files remain supplementary provenance, not reader-facing results. |
| Delta matrix unnecessary | Removed the matrix figure. One concrete INT8 sign-flip example explains the executable perturbation; the exact affine formula is in Appendix A. |
| Clear notation, fewer equations | Three numbered main equations. Explain field, mask, BER, union and set difference when introduced. Set difference means proposed flips except protected bits. Proofs, swap identities, response-table weights and bounds moved to appendices. |
| Include MC results honestly | Main section reports all 32 KWS pool effects, 27 changed masks, an unresolved primary conditional-versus-isolated comparison, and secondary improvements over the initial gradient mask. No assertion that MC cannot work or that a future method will always win. |
| RL and attack coverage | Cite actual Q-learning attack and RL protection methods separately. Explain different objectives and absent matched implementations; do not claim that RL cannot represent interactions. |
| Remove experiment diary | Removed CSV field-limit, recovery, sharding/debug and milestone narrative. Methods retain only information needed to interpret or reproduce the estimand. |
| Compact experimental setup | Two subsections and one six-row benchmark table, retaining cohorts, fault law, masks, units, capping, pairing and uncertainty. |
| Compact results | Four short subsections: isolated association/task quality, paired reversals, conditional allocation, cost. Negative results and limitations retained. |
| Improve BER/budget graphics | Smaller panel labels, common x labels, task-specific y units, thin translucent curves and lighter bands; no large chart heading. All saved numerical values and missing cells preserved. |
| Replace ugly combined figure | One compact dot distribution displays every pool effect plus the mean and 95% pool interval. Paired reversals remain a concise table; a separate forest plot is supplementary. |

## Scientific meaning preserved

The paper does not say no static ranking can ever be optimal. It demonstrates preference reversals among measured methods and an analytical lack of a general guarantee for clean isolated ranking. It does not interpret uncertainty overlap as proof of equivalence or attribute every reversal to interactions. ToyCar AUC, classification CE and token NLL remain separate. Qwen clipping and Pythia's missing finite-cap diagnostic remain disclosed. No new data, masks, fields, significance tests or effectiveness result were generated.

The primary KWS result is a mean conditional-minus-isolated difference of −0.000082388 nats/clip, 95% pool-level interval [−0.001693487, 0.001528711], two-sided p-value 0.917606. It is an unresolved improvement, not superiority and not proof of no possible benefit. The paper cannot promise that a future optimization method will always outperform static alternatives.

## Verification

- ACM PDF built with `latexmk`; no undefined references or overfull boxes.
- Rendered pages inspected, including both six-panel reliability plots and the all-pool distribution.
- Mathematical checks cover the printed counterexample, bound and exact INT8 XOR transitions.
- All 1,254 condition rows and supporting numerical files remain byte-identical to the preceding source revision.
- Bibliography has 41 unique entries and 24 cited keys; original methods and publication status checked in the companion literature audit.
- New plots bind input and output checksums in `compact_figure_provenance.json`.
- Venue rules distinguish IEEE two-column pages from this ACM review format; no unverified TECS page cap is asserted.

The main remaining submission risk is novelty and external validity, not typography: one checkpoint/cohort per model, severe-corruption regimes, missing matched learned-allocation baselines, and no resolved conditional-over-isolated gain. Compression does not conceal these limits.
