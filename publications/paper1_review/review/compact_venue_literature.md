# Compact related work, venue rules, and novelty boundaries

Checked 10 November 2026. This note supplements, rather than replaces, the scientific evidence audit. It changes no numerical result, baseline implementation, or finalized artifact.

## Editorial recommendation

Aim for roughly 12–14 pages of readable ACM `acmsmall` main text, with detailed source identities, cohort receipts, derivations, expanded grids, and reproduction logs in the supplement. This is an editorial target, **not a verified TECS page limit**. Compressing the paper should remove repetition and development history, not negative results or essential uncertainty. Keep related work to about one page; the revised section is approximately 427 words, with three thematic subsections and no FKeras-only subsection.

The credible core remains a checkpoint- and operating-condition-specific evaluation methodology with exact encoded faults and controlled protection comparisons. The contribution is not the invention of sensitivity ranking, adaptive injection, learned protection, or the fact that joint faults can interact. A concise journal paper is possible, but acceptance risk remains substantial if the practical takeaway is only that several known rankings trade places. The strongest version explains which concrete decisions the measurement protocol changes, with held-out effect sizes, costs, and honest limits. A software-only manuscript should not imply radiation qualification or hardware mitigation validation.

## Verified format rules and access limitations

**IEEE TCAD:** The current [official CEDA instructions](https://ieee-ceda.org/publications/tcad/tcad-paper-submissions) specify IEEE two-column layout, minimum 10-point type, and a **14-page hard maximum** for regular papers. The revision must still fit after adding author biographies/photos; references are not exempted. Previously reviewed versions from the past two years, including rejected submissions, must be disclosed. The earlier published FCCM work must be identified with substantive differences. The instructions also require an acknowledgments statement describing whether and how generative AI was used. These are TCAD-specific requirements, not automatically ACM rules. A 22-page ACM PDF cannot be judged against this limit without actual IEEE reformatting.

**ACM TECS:** The current [official journal author-guidelines endpoint](https://dl.acm.org/journal/tecs/author-guidelines) returned HTTP 403 during this audit. The maintained [ACM class source](https://github.com/borisveytsman/acmart/blob/primary/acmart.dtx) lists TECS under `acmsmall`, the small single-column journal format. This establishes the class layout, not a current submission page cap or exemption. Do not present the often-repeated 25-page figure as verified current policy. Check the live submission instructions with the authors' browser or editorial office before submission. A [2020 CEDA editors' panel](https://ieee-ceda.org/files/ieeeceda/Questions%2520and%2520Answers_DAWN%25204_09282020_V2.pdf) discussed conflicting TECS page guidance even then; it is historical evidence of uncertainty, not a rule for 2026.

The current compact revision should therefore retain the supplied ACM project rather than speculate about another venue's typography. TECS remains a plausible scope fit for reliable edge inference and constrained protection decisions, but the absence of measured physical mechanisms and limited learned-allocation comparisons should be explicit. TCAD is not a safer fallback merely because it is familiar: the current paper's central results are evaluation methodology, not a newly validated circuit-design algorithm. The longer venue assessment remains in `venue_and_novelty.md`.

## Objective distinctions that must survive compression

| Work | Verified original method | Relationship to this study |
|---|---|---|
| MSB / FORTUNE | FORTUNE combines quantization with two additional copies of each weight's MSB and majority voting. | A significance preference is relevant precedent. Ideal suppression of an arbitrary fraction of fixed INT8 bits does not reproduce its encoding search, redundancy, voting, or costs. |
| FKeras | Leading weight-loss Hessian eigenpair projection and significance-tier ordering. | A meaningful curvature-based comparator. It must not be described as diagonal-only or unable to express any cross-parameter structure. |
| PrioriFI | Hessian-initialized queues, then observed-damage feedback guides which faults to inject. | Prioritization does not uniquely specify a fixed-cardinality mask. A derived allocation must declare treatment of unprobed bits, feedback cost, and independent final evaluation. |
| EISFINN | Attribution-guided importance sampling of neuron/bit faults. | Directed sampling is prior art. A changed sampling law cannot be equated with Bernoulli expected risk without appropriate probability correction. |
| Progressive Bit Search / DeepHammer | Sequential gradient-guided damaging flips; DeepHammer additionally imposes Rowhammer feasibility. | Adaptive attack discovery, not RL and not a completed equal-budget expected-risk allocation comparison. |
| FlipLLM | Q-learning seeks a small, damaging bit set after sensitivity screening. | A genuine RL attack precedent. Its attack endpoint does not establish equal-budget protection under an independent random field. |
| Su et al. | DQN learns layer-wise bit-position protection masks with a redundancy/performance tradeoff. | A genuine protection-allocation method. Missing matched results are a limitation; this cannot be dismissed as merely attack search. |
| Learned or Ranked? | Public selective-ECC release includes QMIX and other learned/rule-based allocations, held-out comparisons, and layer-interaction diagnostics. | Close allocation/context precedent; publication status is separately qualified. No superiority over its algorithms is established by this manuscript. |

### Attack literature: original methods and publication records

**Progressive Bit Search.** The [original author manuscript](https://arxiv.org/pdf/1903.12269) was inspected beyond its abstract. It ranks bits within layers using gradients, tests candidate changes, selects the layer yielding the largest loss, and repeats after committing the attack. Thus the gradient is reevaluated in an already perturbed network; describing all previous work as clean isolated sensitivity would be false. [Publisher DOI](https://doi.org/10.1109/ICCV.2019.00130), ICCV 2019, pages 1211–1220. Added key `rakin2019bitflip`.

**DeepHammer.** The [official USENIX proceedings page](https://www.usenix.org/conference/usenixsecurity20/presentation/yao) and [full paper](https://www.usenix.org/system/files/sec20-yao.pdf) establish a targeted gradient-guided search combined with achievable Rowhammer patterns and memory placement. It is not evidence of reinforcement learning. USENIX Security 2020, pages 1463–1480. Added key `yao2020deephammer`.

**FlipLLM.** The [author manuscript](https://arxiv.org/html/2512.09872v1) was inspected at the problem formulation and Algorithms 1–3: sensitivity screening, layer selection, and Q-learning with add/remove/shift actions over selected bit positions, followed by MSB flips. Its publication is verified through the [official HOST 2026 list](https://host.conferences.computer.org/2026/accepted-papers/), the [author publication record](https://khurramkhalil.github.io/publications.html), and publisher-deposited metadata for [DOI 10.1109/HOST68814.2026.11604973](https://doi.org/10.1109/HOST68814.2026.11604973), pages 356–366. It should not be called an unreviewed preprint. Added key `khalil2026flipllm`.

Reproduction caution, not a manuscript allegation: the accessible arXiv version prints a negative damage-per-flip reward together with max-Q updates and inconsistent fraction/percentage notation for initial candidate count. These require clarification before faithfully implementing that version. The final publisher PDF was not accessible here, so do not attribute those printed issues to an inspected camera-ready version or claim they invalidate reported results. The manuscript cites only the established high-level method, not an independently reproduced efficacy claim.

### Learned protection is not excluded by the attack distinction

The full [Su et al. Sensors article](https://www.mdpi.com/1424-8220/25/13/4196), especially its model and RL sections, explicitly learns protection. Each layer's mask selects bit positions shared across that layer's weights; its DQN reward trades discrepancy in model behavior against redundancy. That is different granularity and cost modeling from an arbitrary individual-bit mask, but still a relevant competitor. Existing key `su2025rl` is now cited. A fair comparison needs a declared adaptation and held-out evaluation, not a claim that RL cannot model interactions.

The [Learned or Ranked? release](https://zenodo.org/records/23054820) is a public dataset/code companion, with QMIX, VDN, IQL, DQN variants and search methods for selective ECC. Its code includes pairwise layer interaction and context-dependent marginal diagnostics. It therefore materially limits novelty claims about learned allocation and contextual protection. Its accompanying manuscript's peer-reviewed publication was not independently established; the dataset is cited as such. Different granularity, ECC model and cost units prevent direct numerical substitution into our suppression curves. They do not make it irrelevant or establish our superiority.

### Existing sensitivity, mitigation, and prior-flow sources retained

- [FKeras original paper](https://kastner.ucsd.edu/wp-content/uploads/2024/05/admin/jats24-fkeras.pdf), DOI [10.1145/3665334](https://doi.org/10.1145/3665334).
- [PrioriFI original paper](https://kastner.ucsd.edu/wp-content/uploads/2026/05/admin/asplos2026-priorifi.pdf), DOI [10.1145/3779212.3790204](https://doi.org/10.1145/3779212.3790204).
- EISFINN full user-supplied paper, DOI [10.1109/IOLTS60994.2024.10616075](https://doi.org/10.1109/IOLTS60994.2024.10616075).
- FORTUNE full accepted author paper, DOI [10.1109/ATS64447.2024.10915463](https://doi.org/10.1109/ATS64447.2024.10915463).
- [Ranger full author paper](https://zitaoc.github.io/files/Ranger-DSN21.pdf), DOI [10.1109/DSN48987.2021.00018](https://doi.org/10.1109/DSN48987.2021.00018). Range restriction is not stored-weight ECC.
- [FT-CNN full author paper](https://arxiv.org/pdf/2003.12203), DOI [10.1109/TPDS.2020.3043449](https://doi.org/10.1109/TPDS.2020.3043449). Checksums have explicit detection/correction and recovery behavior; the method is not restricted to activation faults alone.
- Selective channel TMR, DOI [10.1109/MDAT.2022.3174181](https://doi.org/10.1109/MDAT.2022.3174181).
- Published NetSuRF-related FCCM flow, DOI [10.1109/FCCM62733.2025.00014](https://doi.org/10.1109/FCCM62733.2025.00014). Its ranking/budget/HLS/TMR motivation is inherited, not a new contribution of this software paper.

## Citation validation and limits

The revised source has 41 bibliography records; all 24 currently cited keys exist and no duplicate keys were found at the audit snapshot. Three verified references were added; existing unrelated records were not deleted. Zotero readiness was checked read-only, but no local API or profile was available, so publisher metadata and original papers were used directly. No environment settings were changed. The main build remains the responsibility of the consolidated manuscript build step.

No matched experiment was fabricated for any cited method. Absence of an executable matched baseline should appear as a limitation in the results/discussion, not as an assertion that its authors studied the wrong problem. In particular, attack success, discovery efficiency, expected random-fault risk, and constrained protection quality are four distinct endpoints.
