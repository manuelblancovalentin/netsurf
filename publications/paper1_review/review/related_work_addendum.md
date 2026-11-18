# Focused related-work and benchmark-citation addendum

Verification date: 2026-11-10. This supplements, rather than replaces, `venue_and_novelty.md` and `bibliography_audit.md`. No experiment or manuscript source was changed.

## Manuscript-ready comparison text

**Quantization and selective redundancy.** FORTUNE combines post-training affine quantization with replication of each weight's most-significant bit. Its illustrated three-bit representation stores two additional MSB copies and uses majority voting during inference; its design-space search balances quantization accuracy, fault response and resource requirements. Thus FORTUNE is related to significance-based protection, but it is not equivalent to our MSB-first mask: it changes the representation and includes redundant storage and voting, whereas we freeze the encoded checkpoint and measure ideal suppression at a declared bit budget. We do not claim to reproduce its implementation or hardware trade-offs. [Accepted author manuscript, Sec. II and Fig. 4](https://d197for5662m48.cloudfront.net/documents/publicationstatus/231624/preprint_pdf/19c1f8a46e5ee0c13975ff55f41587bc.pdf); citation key `nazari2024fortune`.

**Activation and computation protection.** Ranger restricts intermediate value ranges to reduce the effect of transient datapath corruption; its main fault model assumes protected memory. FT-CNN instead develops convolution-checksum schemes with distinct detection and correction capabilities, including weight reloading and recomputation in its recovery workflow. These methods address complementary fault domains and cannot be represented by changing the ordering of our stored-weight protection masks. [Ranger, Secs. II-C and III](https://zitaoc.github.io/files/Ranger-DSN21.pdf), [FT-CNN, Secs. 3 and 4](https://arxiv.org/pdf/2003.12203); keys `chen2021ranger`, `zhao2021ftcnn`.

**Selective TMR.** García Bertoa et al. use fault-sensitivity reports to select channels for triplication in FINN-generated quantized FPGA accelerators, targeting a specified minimum accuracy under a single channel-stuck fault. That granularity, correction mechanism and resource objective differ from encoded-weight-bit allocation under simultaneous Bernoulli fields. Our measured protected-bit fractions are therefore not estimates of their hardware overhead. [Author manuscript, Secs. III–VI](https://pureadmin.qub.ac.uk/ws/portalfiles/portal/319647049/SelectiveTMR_SourceSubmission_.pdf); key `garciabertoa2023selective`.

These three short paragraphs can be consolidated into one related-work subsection. They should not become additional curves in the reliability plots: the current artifact inventory contains no matched implementations of these mechanisms.

## Concise comparison rows

| Method | What it selects or changes | Relationship to this paper |
|---|---|---|
| Published FKeras | Hessian-projection parameter sensitivity, ordered within bit-significance tiers | Actual ranking comparator where a verified execution exists; complete tiers can yield the same mask as MSB-first |
| PrioriFI | Fault-injection order updated from observed damage | A probing policy; conversion to fixed-budget protection is separately labeled NetSuRF adaptation |
| FORTUNE | Quantization precision plus redundant MSB copies | Related protection mechanism, not an interchangeable mask baseline |
| Ranger | Intermediate numeric ranges | Different transient/datapath intervention, not stored-bit allocation |
| FT-CNN | Convolution checksum and recovery workflow | Explicit detection/correction mechanism, not ideal suppression |
| Selective TMR | Sensitive channels and mapped hardware redundancy | Different fault unit, overhead and guarantee |
| EISFINN | Importance-weighted fault sampling | Campaign-efficiency precedent, not a fixed-budget allocator |
| Learned or Ranked? companion | Cost-matched layer/word protection policies, including learned and rule-based methods | Important overlap in allocation and context diagnostics; public dataset, not verified peer-reviewed manuscript |

The FKeras, PrioriFI, EISFINN and companion-release source inspection and precise caveats are in `venue_and_novelty.md`. Do not repeat a blanket claim that earlier methods ignore interactions or cannot adapt to context.

## Metadata corrections and interpretation boundaries

- Existing `nazari2024fortune` is the correct published ATS reference; add `pages = {1--6}`. DOI: [10.1109/ATS64447.2024.10915463](https://doi.org/10.1109/ATS64447.2024.10915463). Do not replace it with the stale Authorea citation in the public code README. Its reported low timing overhead is model-specific, not a universal claim.
- Existing `chen2021ranger`, `zhao2021ftcnn`, and `garciabertoa2023selective` already identify the proper published works. Selective TMR appeared online in 2022 and in IEEE Design & Test 40(2), 67–74, in 2023; the existing 2023 issue citation is appropriate.
- FT-CNN includes memory corruption as well as computational faults. Do not inaccurately describe it as activation-only. Its analytical correction model and reload/recompute behavior still differ from this study's persistent faulted-weight fields.
- Ranger mainly studies single transient faults but also reports a multi-bit experiment. Do not claim its paper has no multi-bit analysis.
- FORTUNE's majority vote is not mathematically perfect suppression under arbitrary multiple corruption of the three copies. The present paper does not model or compare that residual failure mechanism.

## New benchmark citations and placement

The separate `model_dataset_additions.bib` provides seven missing entries verified against original sources. Cite them once in the benchmark table or its introducing paragraph, not repeatedly in result captions.

| Use | Citation key | Primary metadata verification |
|---|---|---|
| DistilBERT architecture | `sanh2019distilbert` | [Author arXiv record](https://arxiv.org/abs/1910.01108), including NeurIPS 2019 workshop acceptance note; cite the arXiv manuscript, not a journal article |
| Pythia architecture/checkpoint family | `biderman2023pythia` | [ICML/PMLR 202, pp. 2397–2430](https://proceedings.mlr.press/v202/biderman23a.html) |
| Qwen3 architecture/checkpoint family | `yang2025qwen3` | [Official repository citation](https://github.com/QwenLM/Qwen3#citation), [technical report](https://arxiv.org/abs/2505.09388); do not present as a verified peer-reviewed journal paper |
| SST-2 source dataset | `socher2013sentiment` | [ACL Anthology, EMNLP 2013, pp. 1631–1642](https://aclanthology.org/D13-1170/) |
| WikiText source corpus | `merity2017pointer` | [Original paper](https://arxiv.org/abs/1609.07843), [author's ICLR 2017 publication list](https://smerity.com/abme.html); the OpenReview page itself required browser verification during this check |
| Speech Commands source dataset | `warden2018speechcommands` | [Original dataset paper](https://arxiv.org/abs/1804.03209); exact title does not end in “Tasks” |
| ToyCar source dataset | `koizumi2019toyadmos` | [Author paper](https://arxiv.org/abs/1908.03299), [published DOI](https://doi.org/10.1109/WASPAA.2019.8937164), WASPAA 2019, pp. 313–317; author order follows the paper and DOI registry |

Use the existing MLPerf Tiny paper and pinned repository citation for the particular ResNet-8, DS-CNN and autoencoder benchmark definitions. Generic architecture papers are not substitutes for these exact benchmark identities. Likewise the original DistilBERT, Pythia and Qwen papers do not establish the correctness of this study's derived W8A8 checkpoints: retain the separate hash-bound conversion/backend receipts and corpus preprocessing records.

## Reference-count discipline

A focused paper can stay near twenty to twenty-five substantive references: the eight comparison/precedent families above, the published FCCM predecessor, seven architecture/dataset sources, MLPerf Tiny and its pinned executable specification, and only the mathematical/statistical references actually used. Remove unrelated pruning or hardware citations if the revised text no longer discusses them. The goal is coverage of the closest methods, not a lengthy catalogue.

## Tooling note

The Zotero skill's read-only readiness check found no active local API or connector; no settings were changed. Public author/publisher metadata was used instead. The Hugging Face paper CLI was used read-only to cross-check Qwen3 metadata; no model files or packages were downloaded or changed.
